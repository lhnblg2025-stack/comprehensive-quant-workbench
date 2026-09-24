#!/usr/bin/env python3
"""storage_ilm.py — data_warehouse kline 数据分层与容量治理。

职责（不做原文件删除/移动，保留可重拉与回测全历史）：
1. scan: 遍历全部 parquet 统计容量，并单独统计 kline 冷/温/热行数。
2. dry_run: 按 POLICY 对 kline 逐文件报告冷/温/热分类与可节省空间，不写盘。
3. archive_cold: 将 kline 中超过 warm_days 的旧行聚合为月线，写到
   data_warehouse/kline_monthly/（派生快照，原文件保持不变）。
4. apply_prune_old_columns: 将 kline 中超过 warm_days 的旧行非 OHLCV 列置空
   重写，以节省 parquet 存储；仅 CLI --apply 且先通过 --dry-run 门禁时执行。
5. 容量超阈值时输出告警，并可复用 scripts/feishu_sender.py 发飞书。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024

DEFAULT_POLICY: dict[str, Any] = {
    "hot_days": 180,
    "warm_days": 1095,
    "archive_threshold_gb": 2.0,
    "cold_agg": "monthly",
}

# OHLCV 是回测必需的最小列集；date 列名允许灵活识别。
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
WARM_SAVE_RATIO = 0.40
COLD_SAVE_RATIO = 0.90
PRUNE_SAVE_RATIO = 0.40


def load_policy(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取 workspace config/policy.yaml 的 ilm 节，缺省回退内置 POLICY。"""
    policy = dict(DEFAULT_POLICY)
    path = Path(config_path) if config_path else ROOT / "config" / "policy.yaml"
    if not path.exists():
        return policy
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ilm = data.get("ilm")
        if isinstance(ilm, dict):
            for key, value in ilm.items():
                if key in policy and value is not None:
                    policy[key] = value
    except Exception:
        # 配置损坏时不能阻断统计，回退内置策略。
        pass
    return policy


POLICY = load_policy()


def _policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_POLICY)
    if policy is None:
        merged.update(POLICY)
    else:
        merged.update(policy)
    return merged


def _as_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value if value is not None else pd.Timestamp.now())
    return ts.normalize()


def _kline_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "kline"


def _iter_kline_files(data_root: str | Path) -> list[Path]:
    path = _kline_dir(data_root)
    if not path.is_dir():
        return []
    return sorted(p for p in path.glob("*.parquet") if p.is_file())


def _pick_date_column(columns: Iterable[str]) -> str | None:
    cols = list(columns)
    for candidate in ("date", "trade_date", "datetime", "time", "timestamp", "day"):
        if candidate in cols:
            return candidate
    for col in cols:
        lowered = str(col).lower()
        if "date" in lowered or "time" in lowered:
            return col
    return None


def _read_date_series(path: Path) -> tuple[pd.Series, str | None]:
    """只读取日期列，避免为 5200+ 个 kline 文件反复加载全列。"""
    try:
        schema = pq.ParquetFile(path).schema_arrow
        columns = list(schema.names)
    except Exception:
        columns = []
    date_col = _pick_date_column(columns)
    if date_col is None:
        return pd.Series(dtype="datetime64[ns]"), None
    try:
        frame = pd.read_parquet(path, columns=[date_col])
    except Exception:
        return pd.Series(dtype="datetime64[ns]"), date_col
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return dates, date_col


def _tier_counts(dates: pd.Series, today: pd.Timestamp, hot_days: int, warm_days: int) -> tuple[int, int, int]:
    """按日期返回 hot / warm / cold 三档行数。

    hot:   >= today - hot_days
    warm:  >= today - warm_days 且 < today - hot_days
    cold:  < today - warm_days
    """
    clean = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    if clean.empty:
        return 0, 0, 0
    hot_cutoff = today - pd.Timedelta(days=hot_days)
    warm_cutoff = today - pd.Timedelta(days=warm_days)
    hot = int((clean >= hot_cutoff).sum())
    warm = int(((clean < hot_cutoff) & (clean >= warm_cutoff)).sum())
    cold = int((clean < warm_cutoff).sum())
    return hot, warm, cold


def _file_category(last_date: Any, today: pd.Timestamp, hot_days: int, warm_days: int) -> str:
    if pd.isna(last_date):
        return "cold"
    hot_cutoff = today - pd.Timedelta(days=hot_days)
    warm_cutoff = today - pd.Timedelta(days=warm_days)
    if last_date >= hot_cutoff:
        return "hot"
    if last_date >= warm_cutoff:
        return "warm"
    return "cold"


def _all_parquet_paths(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def _size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / MB
    except OSError:
        return 0.0


def _build_alert(total_gb: float, threshold_gb: float) -> str:
    return f"⚠️ 数据仓库 {total_gb:.2f}GB > {threshold_gb:.2f}GB"


def _send_feishu(message: str) -> bool:
    """复用 scripts/feishu_sender.py，若不可用则失败但不抛出。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from feishu_sender import send_simple_text

        return bool(send_simple_text(message, title="⚠️ 数据仓库 ILM 告警"))
    except Exception:
        return False


def _maybe_notify(total_gb: float, threshold_gb: float, send_alert: bool) -> tuple[str | None, bool]:
    if total_gb <= threshold_gb:
        return None, False
    message = _build_alert(total_gb, threshold_gb)
    print(message)
    sent = _send_feishu(message) if send_alert else False
    return message, sent


def scan(
    data_root: str | Path = "data_warehouse",
    threshold_gb: float | None = None,
    today: Any = None,
    policy: dict[str, Any] | None = None,
    send_alert: bool = False,
) -> dict[str, Any]:
    """统计 data_root 全部 parquet 与 kline 冷/温/热分层。"""
    cfg = _policy(policy)
    root = Path(data_root)
    ts = _as_timestamp(today)
    threshold = float(threshold_gb if threshold_gb is not None else cfg["archive_threshold_gb"])

    all_files = _all_parquet_paths(root)
    total_mb = sum(_size_mb(p) for p in all_files)
    total_gb = total_mb / 1024

    kline_files = _iter_kline_files(root)
    kline_mb = sum(_size_mb(p) for p in kline_files)
    hot = warm = cold = 0
    for path in kline_files:
        dates, _ = _read_date_series(path)
        h, w, c = _tier_counts(dates, ts, int(cfg["hot_days"]), int(cfg["warm_days"]))
        hot += h
        warm += w
        cold += c

    over_threshold = total_gb > threshold
    alert_message, feishu_sent = _maybe_notify(total_gb, threshold, send_alert)

    return {
        "data_root": str(root),
        "total_gb": round(total_gb, 6),
        "file_count": len(all_files),
        "threshold_gb": threshold,
        "over_threshold": over_threshold,
        "alert_message": alert_message,
        "feishu_sent": feishu_sent,
        "kline": {
            "count": len(kline_files),
            "total_mb": round(kline_mb, 6),
            "hot_count": hot,
            "warm_count": warm,
            "cold_count": cold,
        },
    }


def dry_run(
    data_root: str | Path = "data_warehouse",
    threshold_gb: float | None = None,
    today: Any = None,
    policy: dict[str, Any] | None = None,
    send_alert: bool = False,
) -> dict[str, Any]:
    """生成 ILM 分层报告，估算可节省空间，不写任何文件。"""
    cfg = _policy(policy)
    root = Path(data_root)
    ts = _as_timestamp(today)
    threshold = float(threshold_gb if threshold_gb is not None else cfg["archive_threshold_gb"])

    all_files = _all_parquet_paths(root)
    total_mb = sum(_size_mb(p) for p in all_files)
    total_gb = total_mb / 1024

    kline_files = _iter_kline_files(root)
    files: list[dict[str, Any]] = []
    summary = {
        "file_count": len(kline_files),
        "total_mb": round(sum(_size_mb(p) for p in kline_files), 6),
        "hot_files": 0,
        "warm_files": 0,
        "cold_files": 0,
        "hot_rows": 0,
        "warm_rows": 0,
        "cold_rows": 0,
        "estimated_save_mb": 0.0,
        "potential_save_mb": 0.0,
    }

    for path in kline_files:
        size_mb = _size_mb(path)
        dates, _ = _read_date_series(path)
        hot, warm, cold = _tier_counts(dates, ts, int(cfg["hot_days"]), int(cfg["warm_days"]))
        rows = hot + warm + cold
        first_date = dates.dropna().min() if not dates.dropna().empty else pd.NaT
        last_date = dates.dropna().max() if not dates.dropna().empty else pd.NaT
        category = _file_category(last_date, ts, int(cfg["hot_days"]), int(cfg["warm_days"]))

        if rows:
            estimated_save = size_mb * (cold / rows) * PRUNE_SAVE_RATIO
            potential_save = (
                size_mb * (warm / rows) * WARM_SAVE_RATIO
                + size_mb * (cold / rows) * COLD_SAVE_RATIO
            )
        else:
            estimated_save = potential_save = 0.0

        summary[f"{category}_files"] += 1
        summary["hot_rows"] += hot
        summary["warm_rows"] += warm
        summary["cold_rows"] += cold
        summary["estimated_save_mb"] += estimated_save
        summary["potential_save_mb"] += potential_save

        files.append(
            {
                "path": str(path),
                "file": path.name,
                "rows": rows,
                "first_date": first_date.isoformat() if pd.notna(first_date) else None,
                "last_date": last_date.isoformat() if pd.notna(last_date) else None,
                "category": category,
                "hot_rows": hot,
                "warm_rows": warm,
                "cold_rows": cold,
                "size_mb": round(size_mb, 6),
                "estimated_save_mb": round(estimated_save, 6),
                "potential_save_mb": round(potential_save, 6),
            }
        )

    summary["estimated_save_mb"] = round(summary["estimated_save_mb"], 6)
    summary["potential_save_mb"] = round(summary["potential_save_mb"], 6)

    over_threshold = total_gb > threshold
    alert_message, feishu_sent = _maybe_notify(total_gb, threshold, send_alert)

    return {
        "data_root": str(root),
        "total_gb": round(total_gb, 6),
        "file_count": len(all_files),
        "threshold_gb": threshold,
        "over_threshold": over_threshold,
        "alert_message": alert_message,
        "feishu_sent": feishu_sent,
        "summary": summary,
        "files": files,
    }


def _monthly_aggregate(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """把日线 frame 聚合为月线，输出 date + OHLCV。"""
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df["__ym"] = df[date_col].dt.to_period("M")
    agg: dict[str, tuple[str, str]] = {}
    for col, method in (("open", "first"), ("high", "max"), ("low", "min"), ("close", "last"), ("volume", "sum")):
        if col in df.columns:
            agg[col] = (col, method)

    out = df.groupby("__ym", as_index=False).agg(**agg)
    out["date"] = out["__ym"].dt.to_timestamp()
    ordered = ["date"] + [col for col in ("open", "high", "low", "close", "volume") if col in out.columns]
    return out[ordered]


def archive_cold(
    data_root: str | Path = "data_warehouse",
    dest_root: str | Path | None = None,
    today: Any = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 kline 中超过 warm_days 的旧行聚合为月线，写入 kline_monthly。

    这是派生快照，原 kline 文件保持不变；不做移动/删除，保证回测全历史。
    """
    cfg = _policy(policy)
    root = Path(data_root)
    ts = _as_timestamp(today)
    dest = Path(dest_root) if dest_root else root / "kline_monthly"
    dest.mkdir(parents=True, exist_ok=True)

    files_written = 0
    input_files = 0
    skipped = 0
    rows_in = 0
    rows_out = 0
    warm_cutoff = ts - pd.Timedelta(days=int(cfg["warm_days"]))

    for path in _iter_kline_files(root):
        input_files += 1
        try:
            frame = pd.read_parquet(path)
        except Exception:
            skipped += 1
            continue
        date_col = _pick_date_column(list(frame.columns))
        if date_col is None:
            skipped += 1
            continue
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        old = frame.loc[dates < warm_cutoff].copy()
        if old.empty:
            skipped += 1
            continue

        monthly = _monthly_aggregate(old, date_col)
        if monthly.empty:
            skipped += 1
            continue
        monthly.to_parquet(dest / f"{path.stem}.parquet", index=False)
        files_written += 1
        rows_in += int(len(old))
        rows_out += int(len(monthly))

    return {
        "data_root": str(root),
        "dest_root": str(dest),
        "cold_agg": cfg["cold_agg"],
        "files_written": files_written,
        "input_files": input_files,
        "skipped": skipped,
        "rows_in": rows_in,
        "rows_out": rows_out,
    }


def apply_prune_old_columns(
    data_root: str | Path = "data_warehouse",
    today: Any = None,
    policy: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """将 kline 中超过 warm_days 的旧行非 OHLCV 列置空并重写。

    保留全部行与日期/OHLCV，仅把旧行的扩展列置空，让 parquet 压缩更充分。
    """
    cfg = _policy(policy)
    root = Path(data_root)
    ts = _as_timestamp(today)
    warm_cutoff = ts - pd.Timedelta(days=int(cfg["warm_days"]))

    files_processed = 0
    files_written = 0
    rows_pruned = 0
    skipped = 0
    for path in _iter_kline_files(root):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            skipped += 1
            continue
        date_col = _pick_date_column(list(frame.columns))
        if date_col is None:
            skipped += 1
            continue
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        mask = dates < warm_cutoff
        if not bool(mask.any()):
            skipped += 1
            continue

        keep_cols = [date_col] + [col for col in OHLCV_COLUMNS if col != "date" and col in frame.columns]
        drop_cols = [col for col in frame.columns if col not in keep_cols]
        rows_pruned += int(mask.sum())
        files_processed += 1

        if write and drop_cols:
            for col in drop_cols:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    frame.loc[mask, col] = float("nan")
                else:
                    frame.loc[mask, col] = pd.NA
            tmp = path.with_suffix(path.suffix + ".tmp")
            frame.to_parquet(tmp, index=False)
            tmp.replace(path)
            files_written += 1
        elif write:
            files_written += 1

    return {
        "data_root": str(root),
        "files_processed": files_processed,
        "files_written": files_written,
        "rows_pruned": rows_pruned,
        "skipped": skipped,
        "warm_cutoff": warm_cutoff.isoformat(),
    }


def _print_scan(result: dict[str, Any]) -> None:
    kline = result["kline"]
    print(
        f"数据仓库 {result['total_gb']:.3f}GB / {result['file_count']} 个 parquet "
        f"(阈值 {result['threshold_gb']:.3f}GB)"
    )
    print(
        f"kline: {kline['count']} 个文件 / {kline['total_mb']:.2f}MB / "
        f"hot={kline['hot_count']} warm={kline['warm_count']} cold={kline['cold_count']}"
    )
    if result["over_threshold"]:
        print(f"over_threshold={result['over_threshold']}")


def _print_dry_run(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"数据仓库 {report['total_gb']:.3f}GB / {report['file_count']} 个 parquet")
    print(
        f"kline 分层: hot={summary['hot_files']} warm={summary['warm_files']} cold={summary['cold_files']} 文件; "
        f"hot={summary['hot_rows']} warm={summary['warm_rows']} cold={summary['cold_rows']} 行"
    )
    print(
        f"可节省: prune 后约 {summary['estimated_save_mb']:.2f}MB; "
        f"分层潜力约 {summary['potential_save_mb']:.2f}MB"
    )
    for item in report["files"]:
        print(
            f"- {item['file']}: {item['category']:>4} rows={item['rows']} "
            f"size={item['size_mb']:.2f}MB save={item['estimated_save_mb']:.2f}MB"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="data_warehouse ILM 生命周期管理")
    parser.add_argument("--data-root", default="data_warehouse", help="数据仓库根目录")
    parser.add_argument("--threshold-gb", type=float, default=None, help="告警阈值，覆盖 POLICY")
    parser.add_argument("--scan", action="store_true", help="扫描容量与 kline 分层")
    parser.add_argument("--dry-run", action="store_true", help="生成分层报告，不写盘")
    parser.add_argument("--apply", "--prune-old-columns", action="store_true", help="执行旧列裁剪（需先 --dry-run 确认）")
    args = parser.parse_args(argv)

    threshold = args.threshold_gb
    if args.apply and not args.dry_run:
        print("安全门：--apply 前必须先执行 --dry-run 确认。")
        print("请先运行: python scripts/storage_ilm.py --dry-run --data-root data_warehouse")
        return 2

    if args.scan:
        result = scan(args.data_root, threshold_gb=threshold, send_alert=True)
        _print_scan(result)
        return 0

    if args.dry_run:
        report = dry_run(args.data_root, threshold_gb=threshold, send_alert=True)
        _print_dry_run(report)
        if args.apply:
            print("已通过 --dry-run 确认，开始 --apply 旧列裁剪（本机谨慎执行）。")
            applied = apply_prune_old_columns(args.data_root, write=True)
            print(
                f"完成: files_written={applied['files_written']} "
                f"rows_pruned={applied['rows_pruned']}"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
