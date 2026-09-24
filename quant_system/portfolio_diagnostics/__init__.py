"""
portfolio_diagnostics — 持仓诊断引擎 (V5)

机构级持仓分析，回答"我的持仓为什么涨/跌"：
  1. 因子暴露 — 持仓在哪些因子上有敞口
  2. PnL拆解 — 收益来自 Beta/Alpha/行业/选股
  3. 集中度 — 个股/HHI/行业/风格集中度
  4. 回撤归因 — 哪只股票/哪个因子导致了最大回撤
  5. 综合报告 — 一键生成诊断结论

对标：Barra / Axioma / 券商场外分析系统
"""

from .exposure import FactorExposure
from .attribution import PnLAttribution
from .concentration import ConcentrationAnalysis
from .drawdown_analyzer import DrawdownAnalyzer
from .report import DiagnosticReport

__all__ = [
    "FactorExposure", "PnLAttribution", "ConcentrationAnalysis",
    "DrawdownAnalyzer", "DiagnosticReport",
]
