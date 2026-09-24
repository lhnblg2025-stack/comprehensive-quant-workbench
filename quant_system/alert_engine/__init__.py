"""
alert_engine — 实时预警系统 (V5)

机构级实时风控预警，回答"我的持仓正在出现什么风险，该通知谁"：
  1. 规则引擎 — 声明式DSL定义风控规则（单票/行业集中度、止损、连续下跌、放量）
  2. 统计异常检测 — Z-score/MAD/IQR 三种方法扫描组合各维度的统计异常
  3. 风格漂移检测 — 短期vs长期因子暴露对比，识别无意识的风格漂移
  4. Beta突变检测 — 60日滚动Beta vs 前60日Beta，识别组合系统性风险结构性变化
  5. 成交量异常扫描 — 全市场/自选股池扫描放量个股，捕捉资金异动早期信号
  6. 预警分发器 — 飞书/日志/控制台多渠道分发，批量分发不阻断

对标：券商风控系统 / PagerDuty + Barra 风险监控的组合
"""

from .rules_engine import RulesEngine
from .anomaly_detector import AnomalyDetector
from .style_drift import StyleDrift
from .beta_break import BetaBreak
from .volume_spike import VolumeSpike
from .alert_dispatcher import AlertDispatcher

__all__ = [
    "RulesEngine",
    "AnomalyDetector",
    "StyleDrift",
    "BetaBreak",
    "VolumeSpike",
    "AlertDispatcher",
]
