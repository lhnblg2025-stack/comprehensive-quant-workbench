"""
market_depth — 市场深度诊断 (V5)

不只看涨跌，看涨跌结构：
  - breadth_thrust:     广度推力（涨跌比+新高新低）
  - volume_divergence:  量价背离检测（缩量新高/放量新低）
  - limit_up_depth:     涨停板深度（连板高度/封板率/炸板率）
  - leader_laggard:     领涨/领跌板块轮动追踪

P2-Q22-fix(L253): 原文档提及 historical_pctile 子模块，实际不存在，
已移除。各子模块输出中的 percentile 字段为 score 线性映射值
（score_mapped_pctile，见各模块 percentile_basis/percentile_note），
并非真实历史分位。
"""

from .breadth_thrust import BreadthThrust
from .volume_divergence import VolumeDivergence
from .limit_up_depth import LimitUpDepth
from .leader_laggard import LeaderLaggard

__all__ = ["BreadthThrust", "VolumeDivergence", "LimitUpDepth", "LeaderLaggard"]
