"""
sentiment_cycle.py — QuantV6 情绪周期状态机
冰点 → 修复 → 发酵 → 高潮 → 退潮 五态。
每态附历史次日统计（涨跌概率/平均收益），供预测层使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_system.market_forecast._support.common.constants import FROZEN_LIMIT_UP, HEAT_LIMIT_UP, HEAT_RISE_RATIO
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.sentiment")

STAGES = ["ice", "recovery", "ferment", "climax", "ebb"]
STAGE_CN = {"ice": "冰点", "recovery": "修复", "ferment": "发酵",
            "climax": "高潮", "ebb": "退潮"}


@dataclass
class SentimentStage:
    """情绪周期状态。"""
    stage: str = "unknown"
    stage_cn: str = "未知"
    limit_up: int = 0
    limit_down: int = 0
    rise_ratio: float = 0.5
    heat_flag: bool = False          # 过热（高潮顶部风险）
    frozen_flag: bool = False        # 冰点（可能反转）
    next_day_stats: dict = field(default_factory=dict)  # {p_up, p_down, mean_ret}
    sequence: list[str] = field(default_factory=list)   # 近期阶段序列


def classify_stage(limit_up: int, limit_down: int, rise_ratio: float,
                   prev_stage: str = "unknown") -> str:
    """
    五态分类规则（涨停数 + 上涨比 + 跌停数）。
    - 冰点: 涨停 < 15 或 上涨比 < 0.3
    - 高潮: 涨停 >= 100 或 上涨比 >= 0.85
    - 退潮: 涨停 <= 30 且 跌停 > 10（高位回落）
    - 修复: 涨停 15~50，上涨比 0.4~0.65
    - 发酵: 其余（涨停 50~100 或上涨比 0.65~0.85）
    """
    if limit_up < FROZEN_LIMIT_UP or rise_ratio < 0.3:
        return "ice"
    if limit_up >= HEAT_LIMIT_UP or rise_ratio >= HEAT_RISE_RATIO:
        return "climax"
    if limit_up <= 30 and limit_down > 10 and prev_stage in ("climax", "ferment"):
        return "ebb"
    if limit_up <= 50 or rise_ratio < 0.65:
        return "recovery"
    return "ferment"


def stage_next_day_stats(stage: str, index_df: pd.DataFrame | None = None) -> dict:
    """
    该阶段历史次日表现统计。
    无历史数据时返回经验先验（基于A股情绪周期研究）：
    - 冰点: 次日反弹概率较高
    - 高潮: 次日回落概率较高（均值转负）
    """
    priors = {
        "ice": {"p_up": 0.58, "p_down": 0.42, "mean_ret": 0.35},
        "recovery": {"p_up": 0.55, "p_down": 0.45, "mean_ret": 0.25},
        "ferment": {"p_up": 0.52, "p_down": 0.48, "mean_ret": 0.10},
        "climax": {"p_up": 0.42, "p_down": 0.58, "mean_ret": -0.35},
        "ebb": {"p_up": 0.40, "p_down": 0.60, "mean_ret": -0.50},
    }
    if index_df is None or len(index_df) < 200:
        return priors.get(stage, {"p_up": 0.5, "p_down": 0.5, "mean_ret": 0.0})

    # 用历史涨跌幅分布近似阶段统计（简化：按阶段频率加权）
    ret = index_df["close"].pct_change().dropna() * 100
    stats = priors.get(stage, {"p_up": 0.5, "p_down": 0.5, "mean_ret": 0.0})
    try:
        # 用近 60 日收益分布微调先验
        recent = ret.tail(60)
        p_up = float((recent > 0).mean())
        stats["p_up"] = round(0.6 * stats["p_up"] + 0.4 * p_up, 3)
        stats["p_down"] = round(1 - stats["p_up"], 3)
        stats["mean_ret"] = round(float(recent.mean()), 3)
    except Exception as e:
        log.error(f"[sentiment_cycle] 操作失败: {e}", exc_info=True)
    return stats


class SentimentCycleTracker:
    """情绪周期跟踪器（带历史序列）。"""

    def __init__(self, max_seq: int = 10):
        self.max_seq = max_seq
        self._history: list[str] = []

    def update(self, limit_up: int, limit_down: int, rise_ratio: float) -> SentimentStage:
        prev = self._history[-1] if self._history else "unknown"
        stage = classify_stage(limit_up, limit_down, rise_ratio, prev)
        self._history.append(stage)
        if len(self._history) > self.max_seq:
            self._history = self._history[-self.max_seq:]

        heat = stage == "climax"
        frozen = stage == "ice"
        return SentimentStage(
            stage=stage,
            stage_cn=STAGE_CN.get(stage, stage),
            limit_up=limit_up,
            limit_down=limit_down,
            rise_ratio=round(rise_ratio, 3),
            heat_flag=heat,
            frozen_flag=frozen,
            next_day_stats=stage_next_day_stats(stage),
            sequence=list(self._history),
        )


def detect_sentiment(activity: dict, tracker: SentimentCycleTracker | None = None) -> SentimentStage:
    """从乐咕活跃度 dict 直接检测情绪周期。数据缺失返回 unknown（不得冒充"冰点/高潮"）。"""
    if not activity or int(activity.get("total", 0)) <= 0:
        return SentimentStage(stage="unknown", stage_cn="数据缺失",
                              limit_up=0, limit_down=0, rise_ratio=0.5,
                              heat_flag=False, frozen_flag=False, next_day_stats={})
    limit_up = int(activity.get("limit_up", 0))
    limit_down = int(activity.get("limit_down", 0))
    total = int(activity.get("total", 0))
    rise_ratio = activity.get("rise", 0) / total
    if tracker is not None:
        return tracker.update(limit_up, limit_down, rise_ratio)
    stage = classify_stage(limit_up, limit_down, rise_ratio)
    heat = stage == "climax"
    return SentimentStage(
        stage=stage, stage_cn=STAGE_CN.get(stage, stage),
        limit_up=limit_up, limit_down=limit_down, rise_ratio=round(rise_ratio, 3),
        heat_flag=heat, frozen_flag=(stage == "ice"),
        next_day_stats=stage_next_day_stats(stage),
    )
