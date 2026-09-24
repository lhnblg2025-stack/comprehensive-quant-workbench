"""
integrations — 专业工具链适配层 (V5)
"""

from .alphalens_adapter import AlphalensFactorAnalysis
from .riskfolio_adapter import RiskfolioOptimizer
from .pyfolio_adapter import PyfolioAnalyzer
from .backtrader_adapter import BacktraderValidator

# V5.1: TDX 实时数据 (optional)
try:
    from .tdx_adapter import (  # type: ignore
        TdxConnection, check_tdx, realtime_quote,
        historical_bars, snapshot_market, symbol_list,
    )
except ImportError:
    pass

__all__ = [
    "AlphalensFactorAnalysis",
    "RiskfolioOptimizer",
    "PyfolioAnalyzer",
    "BacktraderValidator",
]
