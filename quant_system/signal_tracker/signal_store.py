"""
signal_store — 信号存储 (V5)

本地 JSON Lines 存储历史信号，供后续表现评估/衰减分析/相关性分析使用。

Schema:
    {
        "signal_id": str,           # UUID
        "timestamp": str,           # ISO8601
        "signal_type": str,         # "breadth_thrust" / "volume_divergence" / ...
        "direction": str,           # "bullish" / "bearish" / "neutral"
        "strength": float,          # 0~1
        "target": str,              # "沪深300" / "全A" / ""
        "value": float,             # 触发时的值
        "threshold": float,         # 触发阈值
        "source_module": str,       # "market_depth.breadth_thrust"
        "metadata": dict,           # 额外信息
        "expiry": str,              # ISO8601, 信号过期时间
    }

存储路径: quant_system/data/signals/<signal_type>.jsonl (按信号类型分文件)
         以及汇总索引 quant_system/data/signals/_all.jsonl (全量流水，便于 load(全部)时快速读取)

设计取舍：
  为了兼顾"按类型查询"和"全量查询"两种常见场景，save() 同时写入：
    1. 按类型的分文件 <signal_type>.jsonl
    2. 汇总文件 _all.jsonl
  P2-Q11-fix(M044)：双写在文件锁（fcntl）内完成，防并发交叉；读路径
  load()/count()/get_stats() 合并 _all.jsonl 与各类型分文件并按 signal_id
  去重，进程中断导致"只落一侧"的记录也能被恢复，避免漏读。
"""

from __future__ import annotations
import logging

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - 非POSIX平台（如Windows）降级为无锁，不影响功能
    _HAS_FCNTL = False

CST = timezone(timedelta(hours=8))

# 数据目录: quant_system/data/signals
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "signals"

REQUIRED_FIELDS = (
    "signal_id", "timestamp", "signal_type", "direction", "strength",
    "target", "value", "threshold", "source_module", "metadata", "expiry",
)


class SignalStore:
    """信号本地存储引擎（JSON Lines 格式）。

    Attributes:
        data_dir: 信号存储目录
        all_file: 汇总流水文件路径
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir: Path = data_dir if data_dir is not None else DATA_DIR
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.getLogger(__name__).error(f"[signal_store] 操作失败: {e}", exc_info=True)
        self.all_file: Path = self.data_dir / "_all.jsonl"

    # ────────────────────────────────────────────────────────────
    #  内部辅助
    # ────────────────────────────────────────────────────────────

    def _type_file(self, signal_type: str) -> Path:
        """按信号类型的分文件路径（简单清洗文件名中的非法字符）。"""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(signal_type))
        return self.data_dir / f"{safe_name}.jsonl"

    @staticmethod
    def _normalize(signal: dict[str, Any]) -> dict[str, Any]:
        """补全缺失字段并规范化信号记录。"""
        now_iso = datetime.now(CST).isoformat()
        record: dict[str, Any] = {
            "signal_id": str(signal.get("signal_id") or uuid.uuid4()),
            "timestamp": str(signal.get("timestamp") or now_iso),
            "signal_type": str(signal.get("signal_type") or "unknown"),
            "direction": str(signal.get("direction") or "neutral"),
            "strength": float(signal.get("strength") or 0.0),
            "target": str(signal.get("target") or ""),
            "value": float(signal.get("value") or 0.0),
            "threshold": float(signal.get("threshold") or 0.0),
            "source_module": str(signal.get("source_module") or ""),
            "metadata": dict(signal.get("metadata") or {}),
            "expiry": str(signal.get("expiry") or ""),
        }
        return record

    @staticmethod
    def _parse_ts(ts: str) -> Optional[datetime]:
        """尽力解析 ISO8601 时间戳，失败返回 None。"""
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CST)
            return dt
        except Exception:
            try:
                return datetime.strptime(ts[:10], "%Y-%m-%d").replace(tzinfo=CST)
            except Exception:
                return None

    @contextmanager
    def _locked(self):
        """跨文件写锁：save() 双写与 delete_expired() 重写期间独占，防并发交叉。

        P2-Q11-fix(M044): 用一个独立锁文件 .signals.lock（不参与 *.jsonl 数据
        读取）配合 fcntl.flock 实现互斥；非 POSIX 平台无 fcntl 时退化为无锁
        （仍是单进程安全）。
        """
        lock_path = self.data_dir / ".signals.lock"
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            if _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if _HAS_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def _append_line(self, path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def _read_lines(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[signal_store] 操作失败: {e}", exc_info=True)
                        continue
        except Exception as e:
            logging.getLogger(__name__).error(f"[signal_store] 操作失败: {e}", exc_info=True)
        return records

    def _write_lines(self, path: Path, records: list[dict[str, Any]]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            logging.getLogger(__name__).error(f"[signal_store] 操作失败: {e}", exc_info=True)

    @staticmethod
    def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 signal_id 去重（保留先出现的记录）；无 signal_id 的记录原样保留。"""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for r in records:
            sid = r.get("signal_id") or ""
            if sid and sid in seen:
                continue
            if sid:
                seen.add(sid)
            out.append(r)
        return out

    def _read_all_merged(self) -> list[dict[str, Any]]:
        """全量读取并合并 _all.jsonl 与各类型分文件（按 signal_id 去重）。

        P2-Q11-fix(M044): save() 双写 _all.jsonl + 类型分文件，进程在两次追加
        之间被中断时可能只落到其中一个文件。读路径原只读 _all.jsonl，会漏掉
        只写入分文件的记录；此处合并两侧并按 signal_id 去重，恢复此类记录。
        """
        records = self._read_lines(self.all_file)
        for tf in self.data_dir.glob("*.jsonl"):
            if tf.name == "_all.jsonl":
                continue
            records += self._read_lines(tf)
        return self._dedupe(records)

    # ────────────────────────────────────────────────────────────
    #  公开方法
    # ────────────────────────────────────────────────────────────

    def save(self, signal: dict[str, Any]) -> dict[str, Any]:
        """保存一条信号记录。

        Args:
            signal: 信号字典，缺失字段会被自动补全 (signal_id 用 UUID, timestamp 用当前时间)。

        Returns:
            规范化后落盘的信号记录。
        """
        try:
            record = self._normalize(signal)
            # P2-Q11-fix(M044): 双写 _all.jsonl + 类型分文件在文件锁内完成，
            # 避免并发进程交叉追加导致两文件不一致；配合 load() 的合并去重，
            # 进程中断落在单文件上的记录也能被读路径恢复。
            with self._locked():
                self._append_line(self.all_file, record)
                self._append_line(self._type_file(record["signal_type"]), record)
            return record
        except Exception as exc:
            return {"error": str(exc)}

    def load(
        self,
        signal_type: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """加载信号记录，支持按类型 / 时间区间过滤。

        Args:
            signal_type: 指定信号类型，None 表示全部类型。
            start: 起始日期/时间 (ISO8601 或 YYYY-MM-DD)，包含。
            end: 结束日期/时间 (ISO8601 或 YYYY-MM-DD)，包含。

        Returns:
            过滤后的信号列表，按时间升序排列。
        """
        try:
            # P2-Q11-fix(M044): 读路径合并 _all.jsonl 与类型分文件并按
            # signal_id 去重，防止双写中断导致的"只落一侧"记录被漏读。
            if signal_type:
                records = self._read_lines(self.all_file)
                records += self._read_lines(self._type_file(signal_type))
                records = self._dedupe(records)
                records = [r for r in records if r.get("signal_type") == signal_type]
            else:
                records = self._read_all_merged()

            start_dt = self._parse_ts(start) if start else None
            end_dt = self._parse_ts(end) if end else None

            def in_range(r: dict[str, Any]) -> bool:
                ts = self._parse_ts(r.get("timestamp", ""))
                if ts is None:
                    return True
                if start_dt and ts < start_dt:
                    return False
                if end_dt and ts > end_dt:
                    return False
                return True

            filtered = [r for r in records if in_range(r)]
            filtered.sort(key=lambda r: r.get("timestamp", ""))
            return filtered
        except Exception:
            return []

    def delete_expired(self) -> int:
        """删除已过期的信号（expiry < 当前时间）。

        Returns:
            删除的记录数。
        """
        now = datetime.now(CST)
        removed = 0
        try:
            # P2-Q11-fix(M044): 清理期间持锁，防止并发 save() 追加进
            # 正在重写的文件而被覆盖丢失。
            with self._locked():
                all_records = self._read_lines(self.all_file)
                kept: list[dict[str, Any]] = []
                expired_by_type: dict[str, int] = {}
                for r in all_records:
                    expiry_raw = r.get("expiry", "")
                    expiry_dt = self._parse_ts(expiry_raw) if expiry_raw else None
                    if expiry_dt is not None and expiry_dt < now:
                        removed += 1
                        expired_by_type[r.get("signal_type", "unknown")] = (
                            expired_by_type.get(r.get("signal_type", "unknown"), 0) + 1
                        )
                        continue
                    kept.append(r)
                self._write_lines(self.all_file, kept)

                # 同步清理各类型分文件
                for stype in {r.get("signal_type", "unknown") for r in all_records}:
                    type_path = self._type_file(stype)
                    type_records = self._read_lines(type_path)
                    type_kept = []
                    for r in type_records:
                        expiry_raw = r.get("expiry", "")
                        expiry_dt = self._parse_ts(expiry_raw) if expiry_raw else None
                        if expiry_dt is not None and expiry_dt < now:
                            continue
                        type_kept.append(r)
                    self._write_lines(type_path, type_kept)
        except Exception as e:
            logging.getLogger(__name__).error(f"[signal_store] 操作失败: {e}", exc_info=True)
        return removed

    def count(self, signal_type: Optional[str] = None) -> int:
        """统计信号数量。

        Args:
            signal_type: 指定信号类型，None 表示全部。

        Returns:
            信号条数。
        """
        try:
            if signal_type:
                return len(self.load(signal_type=signal_type))
            return len(self._read_all_merged())
        except Exception:
            return 0

    def get_stats(self) -> dict[str, Any]:
        """按信号类型统计数量、方向分布及最近信号时间。

        Returns:
            {
                "total": int,
                "by_type": {signal_type: count, ...},
                "by_direction": {direction: count, ...},
                "latest_timestamp": str,
                "earliest_timestamp": str,
            }
        """
        try:
            records = self._read_all_merged()
            by_type: dict[str, int] = {}
            by_direction: dict[str, int] = {}
            timestamps: list[str] = []
            for r in records:
                stype = r.get("signal_type", "unknown")
                direction = r.get("direction", "neutral")
                by_type[stype] = by_type.get(stype, 0) + 1
                by_direction[direction] = by_direction.get(direction, 0) + 1
                ts = r.get("timestamp", "")
                if ts:
                    timestamps.append(ts)
            timestamps.sort()
            return {
                "total": len(records),
                "by_type": by_type,
                "by_direction": by_direction,
                "latest_timestamp": timestamps[-1] if timestamps else "",
                "earliest_timestamp": timestamps[0] if timestamps else "",
            }
        except Exception as exc:
            return {"error": str(exc), "total": 0, "by_type": {}, "by_direction": {}}


def main() -> None:
    """示例：写入几条模拟信号并读取统计结果。"""
    store = SignalStore()

    now = datetime.now(CST)
    demo_signals = [
        {
            "signal_type": "breadth_thrust",
            "direction": "bullish",
            "strength": 0.8,
            "target": "沪深300",
            "value": 0.72,
            "threshold": 0.6,
            "source_module": "market_depth.breadth_thrust",
            "metadata": {"note": "示例信号1"},
            "timestamp": (now - timedelta(days=30)).isoformat(),
            "expiry": (now + timedelta(days=5)).isoformat(),
        },
        {
            "signal_type": "volume_divergence",
            "direction": "bearish",
            "strength": 0.6,
            "target": "全A",
            "value": -0.3,
            "threshold": -0.2,
            "source_module": "market_depth.volume_divergence",
            "metadata": {"note": "示例信号2"},
            "timestamp": (now - timedelta(days=10)).isoformat(),
            "expiry": (now - timedelta(days=1)).isoformat(),  # 已过期
        },
    ]

    for sig in demo_signals:
        saved = store.save(sig)
        print(f"  保存信号: {saved.get('signal_type')} / {saved.get('signal_id', '')[:8]}")

    print("═" * 55)
    print("  信号统计")
    print("═" * 55)
    stats = store.get_stats()
    print(f"  总数: {stats.get('total')}")
    print(f"  按类型: {stats.get('by_type')}")
    print(f"  按方向: {stats.get('by_direction')}")

    removed = store.delete_expired()
    print(f"\n  清理过期信号: {removed} 条")
    print(f"  剩余总数: {store.count()}")


if __name__ == "__main__":
    main()
