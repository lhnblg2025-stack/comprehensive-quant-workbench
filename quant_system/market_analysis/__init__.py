"""market_analysis — 市场分析套件（V4.1 feature）"""

from .breadth import (
    AdvanceDeclineLine, McClellanOscillator, PercentAboveMA,
    NewHighNewLow, BreadthThrust, BreadthReport,
)
from .sentiment import (
    MarginAnalysis, HotStockAnalysis, FundFlowAnalysis,
    LimitUpLimitDown, SentimentIndex, SentimentReport,
)
from .volume import (
    VolumeAnalysis, VWAPAnalysis, OBVAnalysis,
    AccumulationDistribution, VPINAnalysis, VolumeReport,
)
from .rotation import (
    SectorRotation, StyleRotation, IndustryChainTransmission,
    RelativeStrength, RotationReport,
)
from .regime import (
    HMMRegime, TrendRegime, VolatilityRegime,
    LiquidityRegime, CompositeRegime, RegimeReport,
)
from .hot_stocks import (
    HotStockRanking, MidLargeCapMovers,
    AbnormalMoveDetection, HotStockMechanismAnalysis, HotStockReport,
)
from .mechanism import (
    MacroTransmission, PolicyAnalysis, MarketNarrative,
    CapitalFlowMechanism, IndustryCycleAnalysis, MechanismReport,
)

__all__ = [
    # breadth
    "AdvanceDeclineLine", "McClellanOscillator", "PercentAboveMA",
    "NewHighNewLow", "BreadthThrust", "BreadthReport",
    # sentiment
    "MarginAnalysis", "HotStockAnalysis", "FundFlowAnalysis",
    "LimitUpLimitDown", "SentimentIndex", "SentimentReport",
    # volume
    "VolumeAnalysis", "VWAPAnalysis", "OBVAnalysis",
    "AccumulationDistribution", "VPINAnalysis", "VolumeReport",
    # rotation
    "SectorRotation", "StyleRotation", "IndustryChainTransmission",
    "RelativeStrength", "RotationReport",
    # regime
    "HMMRegime", "TrendRegime", "VolatilityRegime",
    "LiquidityRegime", "CompositeRegime", "RegimeReport",
    # hot_stocks
    "HotStockRanking", "MidLargeCapMovers",
    "AbnormalMoveDetection", "HotStockMechanismAnalysis", "HotStockReport",
    # mechanism
    "MacroTransmission", "PolicyAnalysis", "MarketNarrative",
    "CapitalFlowMechanism", "IndustryCycleAnalysis", "MechanismReport",
]
