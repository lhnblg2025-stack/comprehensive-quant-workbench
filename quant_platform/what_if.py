# -*- coding: utf-8 -*-
"""
what_if.py — 假设分析（研究分析层 · MVP）
=========================================
场景：关闭某模块 / 改仓位百分比 / 板块超配 → 收益影响估算。
方法：历史信号回放近似——用该模块历史信号的平均贡献 × 权重占比 线性外推，
**不做全量重算**（与完整回测区分，结果仅作量级参考）。

纯只读分析：不交易、不改参数。涨红跌绿。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

import logging
log = logging.getLogger("quant_platform.what_if")
from quant_platform.research_fmt import fmt_pct



@dataclass
class Scenario:
    """假设场景。kind: disable_module | position_shift | sector_overweight"""
    name: str
    kind: str
    target: str = ""        # 模块名 / 目标仓位% / 板块名
    amount: float = 0.0     # 变更幅度（仓位百分点 / 超配比例）


@dataclass
class WhatIfResult:
    scenario: str
    base_return: float = 0.0
    new_return: float = 0.0
    impact: float = 0.0
    method: str = "历史信号回放近似（非全量重算）"
    detail: str = ""


class WhatIfEngine:
    """假设分析引擎。

    signal_history: DataFrame[symbol?, date, module, weight, signal_ret]
        signal_ret = 该信号命中后（如次日）收益%；用于估算模块贡献。
    缺省时用内置演示数据。
    """

    def __init__(self, signal_history: Optional[pd.DataFrame] = None,
                 base_position_pct: float = 60.0) -> None:
        self.base_position_pct = float(base_position_pct)
        self.signal_history = signal_history if signal_history is not None else self._demo_signals()

    @staticmethod
    def _demo_signals() -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(3)
        rows = []
        for mod, n in (("择时", 80), ("选股", 200), ("行业", 60), ("风险", 40)):
            for _ in range(n):
                rows.append({"module": mod, "weight": rng.uniform(0.02, 0.08),
                             "signal_ret": rng.normal(0.35 if mod != "风险" else -0.2, 1.8)})
        return pd.DataFrame(rows)

    # ── 模块贡献估算 ─────────────────────────────────────
    def module_contribution(self, module: str) -> float:
        """模块历史平均贡献（百分点）：Σ 权重×信号收益 的均值 × 100。"""
        sub = self.signal_history[self.signal_history["module"] == module]
        if not len(sub):
            return 0.0
        return float((sub["weight"] * sub["signal_ret"]).mean() * 100)

    def run(self, scenarios: list[Scenario]) -> list[WhatIfResult]:
        """依次估算各场景的收益影响。"""
        out: list[WhatIfResult] = []
        for sc in scenarios:
            out.append(self._estimate(sc))
        return out

    def _estimate(self, sc: Scenario) -> WhatIfResult:
        base = 0.0  # 基准 = 各模块贡献之和（近似组合日收益）
        modules = sorted(self.signal_history["module"].unique())
        contrib = {m: self.module_contribution(m) for m in modules}
        base = sum(contrib.values())

        if sc.kind == "disable_module":
            delta = -contrib.get(sc.target, 0.0)
            detail = f"关闭模块[{sc.target}]（历史贡献 {contrib.get(sc.target, 0.0):+.2f}pct）"
        elif sc.kind == "position_shift":
            new_pct = float(sc.amount)
            ratio = new_pct / self.base_position_pct if self.base_position_pct else 1.0
            delta = base * (ratio - 1.0)
            detail = (f"仓位 {self.base_position_pct:.0f}% → {new_pct:.0f}%："
                      f"收益按仓位比例线性缩放（{ratio:.2f}x）")
        elif sc.kind == "sector_overweight":
            # 板块超配：用该板块信号的平均收益 × 超配比例近似
            sub = self.signal_history[self.signal_history["module"] == "行业"]
            avg_ret = float(sub["signal_ret"].mean()) if len(sub) else 0.0
            delta = avg_ret * sc.amount / 100.0 * 100  # 超配 amount% 的增量贡献
            detail = f"板块[{sc.target}]超配 {sc.amount:.0f}%：按历史平均信号收益 {avg_ret:+.2f}% 外推"
        else:
            delta, detail = 0.0, f"未知场景类型 {sc.kind}"

        return WhatIfResult(scenario=sc.name, base_return=base,
                            new_return=base + delta, impact=delta, detail=detail)

    def render_markdown(self, results: list[WhatIfResult]) -> str:
        lines = ["# 假设分析（What-If）", ""]
        lines.append(f"> 方法: 历史信号回放近似，非全量重算；基准组合日收益 ≈ "
                     f"{results[0].base_return:+.2f}pct" if results else "> 无场景")
        lines.append("")
        lines.append("| 场景 | 基准 | 变更后 | 影响 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in results:
            lines.append(f"| {r.scenario} | {fmt_pct(r.base_return)} | {fmt_pct(r.new_return)} "
                         f"| {fmt_pct(r.impact)} | {r.detail} |")
        lines.append("")
        lines.append("---")
        lines.append(f"_quant_v6.research.what_if · {datetime.now():%Y-%m-%d %H:%M:%S}_")
        return "\n".join(lines)


# ── 演示 / 冒烟 ─────────────────────────────────────────
def demo() -> str:
    eng = WhatIfEngine()
    scenarios = [
        Scenario("关闭择时模块", "disable_module", "择时"),
        Scenario("仓位降到 40%", "position_shift", amount=40.0),
        Scenario("半导体超配 10%", "sector_overweight", "半导体", amount=10.0),
    ]
    results = eng.run(scenarios)
    md = eng.render_markdown(results)
    print(md)
    print(f"\n[ok] 场景数={len(results)} | "
          f"关闭择时影响={results[0].impact:+.2f}pct | "
          f"降仓影响={results[1].impact:+.2f}pct")
    return md


if __name__ == "__main__":
    demo()
