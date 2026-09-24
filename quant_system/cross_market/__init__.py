"""
cross_market — 跨市场验证引擎 (V5)

A股信号需经跨市场数据验证：
  1. A-H溢价 — A股相对港股的估值偏差
  2. 汇率影响 — 人民币波动对各行业的影响方向
  3. 股债性价比 — ERP 的历史分位
  4. 北向资金 — 外资的行业/个股偏好
  5. 两融深度 — 分板块/分行业的融资融券分析

每个维度输出验证结论：支撑 / 反对 / 中性
"""

from .hk_stock_link import HkStockLink
from .fx_impact import FxImpact
from .bond_equity import BondEquity
from .north_flow_deep import NorthFlowDeep
from .margin_deep import MarginDeep
from .composite import CrossMarketComposite

__all__ = [
    "HkStockLink", "FxImpact", "BondEquity",
    "NorthFlowDeep", "MarginDeep", "CrossMarketComposite",
]
