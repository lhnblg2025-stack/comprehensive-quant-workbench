# -*- coding: utf-8 -*-
"""
funnel_visualizer.py — 策略过滤漏斗（研究分析层 · 第一梯队）
============================================================
读取 orchestrator_v7 选股流程各阶段股票数
（全市场 5000 → ST过滤 4820 → 流动性 3110 → 预测 860 → 风险 210 → 行业 58 → 最终 12），
输出漏斗图数据（每层名称 / 股票数 / 过滤量 / 过滤率），供 UI 渲染（JSON）。

纯只读分析：不交易、不写仓库、不改参数。可读取 orchestrator 运行产物
（generated/funnel/*.json / *.csv），缺省使用配置中的默认阶段。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging
log = logging.getLogger("quant_platform.funnel_visualizer")
from quant_platform.research_fmt import bar


WORKSPACE = Path(__file__).resolve().parents[2]
FUNNEL_DIR = WORKSPACE / "generated" / "funnel"

# orchestrator_v7 选股流程默认阶段（来自 V8 融合方案 W2 说明）
DEFAULT_STAGES: list[tuple[str, int]] = [
    ("全市场", 5000),
    ("ST过滤", 4820),
    ("流动性过滤", 3110),
    ("预测筛选", 860),
    ("风险过滤", 210),
    ("行业约束", 58),
    ("最终组合", 12),
]


@dataclass
class FunnelStage:
    """单层漏斗。"""
    name: str
    count: int

    @property
    def dropped(self) -> int:
        return self._prev - self.count if self._prev is not None else 0

    @property
    def drop_pct(self) -> float:
        if not self._prev:
            return 0.0
        return round((self._prev - self.count) / self._prev * 100, 1)

    _prev: Optional[int] = field(default=None, repr=False)


class FunnelVisualizer:
    """选股漏斗：阶段数据 → 供 UI 渲染的 JSON / ASCII / markdown。"""

    def __init__(self, stages: Optional[list[tuple[str, int]]] = None) -> None:
        raw = stages or DEFAULT_STAGES
        prev: Optional[int] = None
        self.stages: list[FunnelStage] = []
        for name, cnt in raw:
            s = FunnelStage(name=name, count=int(cnt))
            s._prev = prev
            self.stages.append(s)
            prev = int(cnt)

    # ── 数据来源 ─────────────────────────────────────────
    @classmethod
    def from_defaults(cls) -> "FunnelVisualizer":
        return cls()

    @classmethod
    def from_file(cls, path: Path | str) -> "FunnelVisualizer":
        """从 orchestrator 落盘的 JSON 读取（[{name, count}] 或 {stages:[...]}）。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"漏斗文件不存在: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("stages", data) if isinstance(data, dict) else data
        stages = [(s["name"], int(s["count"])) for s in raw]
        return cls(stages)

    @classmethod
    def from_counts(cls, counts: dict[str, int], order: Optional[list[str]] = None) -> "FunnelVisualizer":
        """从 orchestrator 返回的 {阶段名: 股票数} 构建（保持传入顺序或给定顺序）。"""
        order = order or list(counts.keys())
        return cls([(k, counts[k]) for k in order])

    def auto_load(self) -> bool:
        """尝试读取 generated/funnel/ 下最新漏斗文件；成功返回 True。"""
        if not FUNNEL_DIR.exists():
            return False
        files = sorted(FUNNEL_DIR.glob("*.json"))
        if not files:
            return False
        try:
            self.stages = self.from_file(files[-1]).stages
            return True
        except Exception:
            return False

    # ── 输出 ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        """供 UI 渲染的漏斗图数据。"""
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_dropped": self.stages[0].count - self.stages[-1].count,
            "keep_rate": round(self.stages[-1].count / max(self.stages[0].count, 1) * 100, 3),
            "stages": [
                {"name": s.name, "count": s.count,
                 "dropped": s.dropped, "drop_pct": s.drop_pct}
                for s in self.stages
            ],
        }

    def to_json(self, path: Optional[Path | str] = None, indent: int = 2) -> str:
        """输出 JSON 字符串；给 path 则同时落盘。"""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            log.info("funnel json -> %s", p)
        return text

    def render_ascii(self, width: int = 30) -> str:
        """终端 ASCII 漏斗（宽层在上、窄层在下）。"""
        lines: list[str] = []
        max_w = max(s.count for s in self.stages) or 1
        for i, s in enumerate(self.stages):
            w = max(1, int(s.count / max_w * width))
            pad = " " * ((width - w) // 2)
            drop = "" if i == 0 else f"  (-{s.dropped}, {s.drop_pct}%)"
            lines.append(f"{pad}{'█' * w} {s.count:>5d}  {s.name}{drop}")
        return "\n".join(lines)

    def render_markdown(self) -> str:
        """markdown 表格 + ASCII 图。"""
        lines: list[str] = []
        lines.append("# 策略过滤漏斗")
        lines.append("")
        lines.append("| 阶段 | 股票数 | 过滤量 | 过滤率 |")
        lines.append("| --- | --- | --- | --- |")
        for i, s in enumerate(self.stages):
            drop = "-" if i == 0 else f"{s.dropped}"
            pct = "-" if i == 0 else f"{s.drop_pct}%"
            lines.append(f"| {s.name} | {s.count} | {drop} | {pct} |")
        lines.append("")
        lines.append("```")
        lines.append(self.render_ascii())
        lines.append("```")
        lines.append("")
        d = self.to_dict()
        lines.append(f"- 总过滤: **{d['total_dropped']}** 只 | 保留率 **{d['keep_rate']}%**")
        lines.append("")
        lines.append("---")
        lines.append(f"_quant_v6.research.funnel_visualizer · {d['generated_at']}_")
        return "\n".join(lines)


# ── 演示 / 冒烟 ─────────────────────────────────────────
def demo() -> str:
    """默认阶段 → JSON + ASCII + markdown；并把 JSON 样例落到 generated/funnel/。"""
    fv = FunnelVisualizer()
    print(fv.render_markdown())
    print("\n--- 供 UI 渲染的 JSON ---")
    js = fv.to_json(FUNNEL_DIR / "funnel_demo.json")
    print(js)
    print(f"\n[ok] JSON 已落盘: {FUNNEL_DIR / 'funnel_demo.json'}")
    return js


if __name__ == "__main__":
    demo()
