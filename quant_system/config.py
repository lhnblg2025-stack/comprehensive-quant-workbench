# V4.1
"""
quant_system.config — 策略/组合默认配置。

A股费率假设（design 文档确认的有意券商假设，P2-Q28-fix 文档化）：
  - commission_pct: 佣金 万0.85（0.0085%），最低 5 元/笔（无"免五"优惠）
  - stamp_tax_pct:  印花税 万5（0.05%），仅卖出收取
  - transfer_fee_pct: 过户费 万0.1（0.001%），双边收取（与 trading_system/
    execution_broker 的 DEFAULT_TRANSFER_FEE_RATE 口径一致）
  - slippage_pct:   滑点 0.1%
cost 模型（backtest.py/risk.py 等）通过 getattr(portfolio, 'transfer_fee_pct', …)
读取，已统一计入过户费。
"""
from __future__ import annotations

from dataclasses import dataclass

# D3/D组收敛: 费率唯一真源为 execution_broker 常量（同数值同方向），此处改为引用。
try:
    from quant_system import execution_broker as _execution_broker
except ImportError:  # 允许以独立脚本方式运行（repo root 直接 import config）
    import execution_broker as _execution_broker


@dataclass(frozen=True)
class StrategyConfig:
    fast_ma: int = 20
    slow_ma: int = 60
    trend_ma: int = 144
    long_trend_ma: int = 300
    volume_ma: int = 20
    max_risk_score_for_new_buy: int = 5
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.24
    trail_stop_pct: float = 0.12


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 1_000_000.0
    max_position_pct: float = 0.20
    risk_per_trade_pct: float = 0.01
    commission_pct: float = _execution_broker.COMMISSION_RATE      # 佣金: 万0.85 (0.0085%)
    stamp_tax_pct: float = _execution_broker.STAMP_TAX_RATE        # A股印花税: 万5 (0.05%), 仅卖出时收
    transfer_fee_pct: float = _execution_broker.TRANSFER_FEE_RATE  # A股过户费: 0.001%, 双边收取  # P2-Q28-fix(M364)
    min_commission: float = _execution_broker.MIN_COMMISSION       # 最低佣金: 5元/笔
    min_stamp_tax: float = 0.0          # 最低印花税: 0元/笔
    slippage_pct: float = _execution_broker.DEFAULT_SLIPPAGE_RATE  # 滑点: 0.1% (P1-6: 引用 execution_broker 唯一真源 DEFAULT_SLIPPAGE_RATE)
    # fixed | market_cap_based  (P1-1: 废弃死配置 volatility_based)
    # 注意: 默认保持 fixed(引用 DEFAULT_SLIPPAGE_RATE=10bp)。market_cap_based 需
    # 数据行含 market_cap(流通市值,元) 字段才有意义; 数据缺该字段时回退保守20bp,
    # 故不设为默认, 避免"看似按市值实则普遍加价一倍"的名不副实。
    slippage_mode: str = "fixed"


DEFAULT_STRATEGY = StrategyConfig()
DEFAULT_PORTFOLIO = PortfolioConfig()
