"""
signal_tracker — 信号回溯追踪 (V5)

历史信号存档与效果回溯分析：
  - signal_store:        本地信号存储 (JSON Lines)
  - signal_performance:  信号历史表现评估（前瞻收益/胜率/Sharpe）
  - signal_decay:        信号衰减分析（半衰期/最佳持有期）
  - best_conditions:     信号最佳市场条件（市场状态×波动率分组胜率）
  - signal_correlation:  信号相关性分析（识别冗余信号对）

设计目标：让各分析模块产生的信号（广度推力/量价背离/资金情绪...）
可以被持续记录、事后验证，回答"这类信号到底有没有用"。
"""

from .signal_store import SignalStore
from .signal_performance import SignalPerformance
from .signal_decay import SignalDecay
from .best_conditions import BestConditions
from .signal_correlation import SignalCorrelation

__all__ = [
    "SignalStore",
    "SignalPerformance",
    "SignalDecay",
    "BestConditions",
    "SignalCorrelation",
]
