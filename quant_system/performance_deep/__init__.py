"""
performance_deep — 业绩归因深潜模块 (V5)

在标准 Brinson 归因基础上进一步深潜：
  1. Brinson 归因      — 配置/选择/交互效应拆解
  2. 因子择时能力评估   — 暴露调整是否领先因子收益
  3. 滚动窗口归因       — 归因结果随时间的演变
  4. 归因稳定性         — 效应是持续能力还是运气
  5. 基准比较分析       — 追踪误差/信息比率/alpha/beta

对标：机构级业绩归因报告（GIPS / Brinson-Fachler / Grinold-Kahn 框架）
"""

from .brinson_attribution import BrinsonAttribution
from .factor_timing import FactorTiming
from .rolling_attribution import RollingAttribution
from .attribution_stability import AttributionStability
from .benchmark_analysis import BenchmarkAnalysis

__all__ = [
    "BrinsonAttribution",
    "FactorTiming",
    "RollingAttribution",
    "AttributionStability",
    "BenchmarkAnalysis",
]
