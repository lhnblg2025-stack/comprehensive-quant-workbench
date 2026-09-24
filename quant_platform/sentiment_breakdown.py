# -*- coding: utf-8 -*-
"""
sentiment_breakdown.py — 情绪温度拆解（研究分析层 · MVP）
=========================================================
把综合情绪分（0-100）拆解为各分量：涨停贡献 / 炸板 / 成交活跃 / 连板高度 / 赚钱效应。
每个分量给出：原始指标 → z-score → 0-100 分 → 贡献占比。
口径参考 strategy/sentiment_thermometer 的权重体系，独立实现拆解逻辑。

纯只读分析：不交易、不改参数。涨红跌绿。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

import logging
log = logging.getLogger("quant_platform.sentiment_breakdown")
from quant_platform.research_fmt import zscore_to_100


# 分量定义：指标 → (权重, 方向, 历史均值, 历史标准差, 中文名)
# 方向: 1=越高越热, -1=越高越冷
DEFAULT_COMPONENTS: dict[str, tuple[float, int, float, float, str]] = {
    "limit_up_ratio": (0.20, 1, 0.02, 0.015, "涨停贡献"),      # 涨停比例
    "zhaban_rate":    (0.15, -1, 0.30, 0.10, "炸板"),          # 炸板率（反向）
    "amount_ratio":   (0.15, 1, 1.00, 0.25, "成交活跃"),       # 成交额/5日均
    "max_lianban":    (0.20, 1, 5.0, 2.0, "连板高度"),          # 最高连板
    "profit_effect":  (0.30, 1, 0.5, 1.2, "赚钱效应"),          # 热股等权涨跌%
}
WEIGHTS = {k: v[0] for k, v in DEFAULT_COMPONENTS.items()}
DIRECTIONS = {k: v[1] for k, v in DEFAULT_COMPONENTS.items()}
HIST = {k: (v[2], v[3]) for k, v in DEFAULT_COMPONENTS.items()}


@dataclass
class ComponentPart:
    name: str
    score: float          # 0-100
    weight: float
    contribution: float   # score × weight（对总分的贡献）
    share: float          # 贡献占比
    raw: float


@dataclass
class BreakdownResult:
    total: float
    parts: list[ComponentPart] = field(default_factory=list)
    explained: float = 0.0


class SentimentBreakdown:
    """情绪温度拆解：综合分 → 各分量贡献。"""

    def __init__(self, components: Optional[dict] = None) -> None:
        self.components = components or DEFAULT_COMPONENTS

    def _part_score(self, key: str, value: float, history: Optional[dict]) -> float:
        if history and key in history:
            mu, sd = history[key]
        else:
            mu, sd = HIST.get(key, (0.0, 1.0))
        if sd is None or sd < 1e-9:
            return 50.0
        z = (value - mu) / sd * DIRECTIONS.get(key, 1)
        return zscore_to_100(z)

    def decompose(self, total_score: Optional[float] = None,
                  metrics: Optional[dict[str, float]] = None,
                  history: Optional[dict[str, tuple]] = None) -> BreakdownResult:
        """
        拆解综合情绪分。
        total_score: 综合情绪分（缺省用加权分量自动合成）。
        metrics: {指标key: 当日值}，缺省用演示值。
        """
        m = metrics or {"limit_up_ratio": 0.045, "zhaban_rate": 0.22,
                        "amount_ratio": 1.35, "max_lianban": 7.0,
                        "profit_effect": 1.8}
        parts: list[ComponentPart] = []
        for key, (w, _dir, _mu, _sd, name) in self.components.items():
            raw = float(m.get(key, HIST[key][0]))
            score = self._part_score(key, raw, history)
            parts.append(ComponentPart(name=name, score=round(score, 1), weight=w,
                                       contribution=round(score * w, 2),
                                       share=0.0, raw=raw))
        total = sum(p.contribution for p in parts) / sum(p.weight for p in parts)
        if total_score is not None:
            total = float(total_score)
        for p in parts:
            p.share = round(p.contribution / (total + 1e-9), 3)
        return BreakdownResult(total=round(total, 1), parts=parts,
                               explained=round(sum(p.contribution for p in parts), 1))

    def render_markdown(self, res: BreakdownResult) -> str:
        lines = ["# 情绪温度拆解", ""]
        lines.append(f"> 综合情绪分: **{res.total:.1f}/100**"
                     f"（分量加权解释 {res.explained:.1f} 分）")
        lines.append("")
        lines.append("| 分量 | 原始值 | 分量分 | 权重 | 贡献 | 贡献占比 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for p in res.parts:
            lines.append(f"| {p.name} | {p.raw:.3f} | {p.score:.0f} | {p.weight:.2f} "
                         f"| {p.contribution:.1f} | {p.share*100:.0f}% |")
        lines.append("")
        lines.append("---")
        lines.append(f"_quant_v6.research.sentiment_breakdown · {datetime.now():%Y-%m-%d %H:%M:%S}_")
        return "\n".join(lines)


# ── 演示 / 冒烟 ─────────────────────────────────────────
def demo() -> str:
    sb = SentimentBreakdown()
    res = sb.decompose()
    md = sb.render_markdown(res)
    print(md)
    print(f"\n[ok] 综合分={res.total:.1f} | 分量数={len(res.parts)} | "
          f"主导分量={max(res.parts, key=lambda p: p.share).name}")
    return md


if __name__ == "__main__":
    demo()
