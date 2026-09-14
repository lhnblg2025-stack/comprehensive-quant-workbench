"""Illustrative parameters only; no production account configuration."""
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategyConfig:
    fast_ma: int = 10
    slow_ma: int = 30
    trend_ma: int = 30
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.20
    trail_stop_pct: float = 0.10

@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 100000.0
    max_position_pct: float = 0.20
    commission_pct: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_pct: float = 0.0005
    min_stamp_tax: float = 0.0
    transfer_fee_pct: float = 0.00001
    slippage_pct: float = 0.001
    slippage_mode: str = 'fixed'

DEFAULT_STRATEGY = StrategyConfig()
DEFAULT_PORTFOLIO = PortfolioConfig()
