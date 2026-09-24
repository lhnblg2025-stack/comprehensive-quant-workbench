# -*- coding: utf-8 -*-
"""
replay_center.py — 复盘工作台（研究分析层 · MVP）
=================================================
按日期索引已有 日报 / 快照 / 信号 文件的读取器。
不做完整回放 UI，提供按日聚合查询接口：
  - index():   扫描各输出目录，建立 {日期: {类型: [文件路径]}} 索引
  - get_day(): 取某一天的全部产物
  - summary(): 某一天的文字摘要

纯只读分析：只扫描/读取已有文件，不写入、不改参数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging
log = logging.getLogger("quant_platform.replay_center")


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    WORKSPACE / "generated" / "journal",     # 决策日志
    WORKSPACE / "generated" / "report",      # 日报
    WORKSPACE / "generated" / "snapshot",    # 快照
    WORKSPACE / "generated" / "signal",      # 信号
    WORKSPACE / "generated" / "ic_report",   # IC 报告
    WORKSPACE / "generated" / "funnel",      # 漏斗数据
]
_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


@dataclass
class ReplayIndex:
    """日期 → {类型: [路径]} 索引。"""
    by_date: dict[str, dict[str, list[Path]]] = field(default_factory=dict)

    def dates(self) -> list[str]:
        return sorted(self.by_date.keys(), reverse=True)

    def kinds(self, date: str) -> list[str]:
        return sorted(self.by_date.get(date, {}).keys())

    def files(self, date: str, kind: Optional[str] = None) -> list[Path]:
        d = self.by_date.get(date, {})
        if kind:
            return d.get(kind, [])
        return [p for ps in d.values() for p in ps]


class ReplayCenter:
    """按日期聚合查询各输出目录的复盘工作台。"""

    def __init__(self, roots: Optional[list[Path | str]] = None) -> None:
        self.roots = [Path(r) for r in (roots or DEFAULT_ROOTS)]
        self.index = ReplayIndex()

    def _kind_of(self, path: Path) -> str:
        """由父目录推断产物类型。"""
        parent = path.parent.name
        return {"journal": "决策日志", "report": "日报", "snapshot": "快照",
                "signal": "信号", "ic_report": "IC报告", "funnel": "漏斗"}.get(
                    parent, parent)

    def _date_of(self, path: Path) -> Optional[str]:
        """从文件名或父目录提取 YYYY-MM-DD。"""
        for part in (path.name, path.parent.name, str(path.parent.parent.name)):
            m = _DATE_RE.search(part)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return None

    def index_all(self) -> ReplayIndex:
        """扫描全部根目录，重建索引。"""
        by_date: dict[str, dict[str, list[Path]]] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if not f.is_file() or f.suffix not in (".md", ".json", ".csv", ".txt", ".parquet"):
                    continue
                d = self._date_of(f)
                if not d:
                    continue
                kind = self._kind_of(f)
                by_date.setdefault(d, {}).setdefault(kind, []).append(f)
        self.index = ReplayIndex(by_date=by_date)
        log.info("replay index: %d 个日期", len(by_date))
        return self.index

    def list_dates(self) -> list[str]:
        """可复盘的日期（倒序）。"""
        return self.index.dates() or self.index_all().dates()

    def get_day(self, date_str: str) -> dict[str, list[Path]]:
        """某一天的全部产物（按类型）。"""
        return self.index.by_date.get(date_str, {})

    def summary(self, date_str: str) -> str:
        """某一天的文字摘要。"""
        files = self.get_day(date_str)
        if not files:
            return f"{date_str}: 无复盘产物"
        parts = [f"{date_str}: {len(files)} 类产物"]
        for kind, paths in sorted(files.items()):
            parts.append(f"  - {kind} × {len(paths)}: "
                         + ", ".join(str(p.name) for p in paths[:3]))
        return "\n".join(parts)

    def search(self, keyword: str, date_str: Optional[str] = None) -> list[Path]:
        """按关键词过滤文件（文件名或内容首行）。"""
        hits: list[Path] = []
        # 遍历索引内全部文件（可按日期过滤）
        for d in self.index.by_date:
            if date_str and d != date_str:
                continue
            for kind, paths in self.index.by_date[d].items():
                for p in paths:
                    if keyword in p.name:
                        hits.append(p)
                        continue
                    try:
                        head = p.read_text(encoding="utf-8", errors="ignore")[:500]
                        if keyword in head:
                            hits.append(p)
                    except Exception:
                        pass
        return hits


# ── 演示 / 冒烟 ─────────────────────────────────────────
def demo(tmp_root: Optional[Path] = None) -> str:
    """用临时目录构造假产物 → 索引 → 按日聚合查询。"""
    import tempfile

    root = tmp_root or Path(tempfile.mkdtemp(prefix="qv6_replay_"))
    (root / "journal").mkdir(parents=True, exist_ok=True)
    (root / "report").mkdir(parents=True, exist_ok=True)
    (root / "signal").mkdir(parents=True, exist_ok=True)
    (root / "journal" / "2026-08-05.md").write_text("# 决策日志 2026-08-05\n仓位 55%", encoding="utf-8")
    (root / "report" / "2026-08-05.md").write_text("# 日报 2026-08-05\n收益 +0.3%", encoding="utf-8")
    (root / "journal" / "2026-08-06.md").write_text("# 决策日志 2026-08-06\n仓位 65%", encoding="utf-8")
    (root / "signal" / "2026-08-06.json").write_text("{\"n\": 12}", encoding="utf-8")

    rc = ReplayCenter(roots=[root])
    rc.index_all()
    out: list[str] = [f"[ok] 索引日期: {rc.list_dates()}"]
    out.append(rc.summary("2026-08-06"))
    out.append(f"[ok] get_day(2026-08-06) 类型: {rc.get_day('2026-08-06').keys()}")
    out.append(f"[ok] search('仓位') 命中: {[p.name for p in rc.search('仓位')]}")
    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    demo()
