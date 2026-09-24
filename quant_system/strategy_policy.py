"""可验证的多频率、多因子择时策略策略。

频率只决定信号何时重新评估；是否交易由信号强度、预期收益和盈亏比决定。
这里登记的是公开可复现的研究代理，不声称恢复任何私有策略源码。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import math


# 15 个彼此可区分的因子族：技术、量价、情绪、估值和风险都有覆盖。
FACTOR_15 = (
    "momentum",
    "trend_ma",
    "rsi",
    "kdj",
    "macd",
    "cci",
    "volatility",
    "volume",
    "liquidity",
    "sentiment",
    "market_regime",
    "value",
    "quality",
    "growth",
    "risk_reward",
)

FREQUENCIES = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class TradeDecision:
    action: str
    frequency: str
    score: float
    expected_return: float
    downside: float
    risk_reward: float
    reasons: tuple[str, ...] = ()


def decide_trade(
    factors: Mapping[str, float],
    *,
    frequency: str = "daily",
    expected_return: float | None = None,
    downside: float | None = None,
    min_score: float = 0.55,
    min_expected_return: float = 0.02,
    min_risk_reward: float = 1.5,
) -> TradeDecision:
    """按机会质量决定交易，避免固定日/周/月换手。

    ``factors`` 的值应先在横截面标准化到 [-1, 1]。缺失因子被忽略，
    但至少需要 3 个因子；expected_return/downside 应来自训练后的
    OOS 预测或 ATR/历史波动估计，不能使用未来真实收益。
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"unsupported frequency: {frequency}")
    vals = [float(v) for v in factors.values() if v is not None and math.isfinite(float(v))]
    score = sum(vals) / len(vals) if vals else 0.0
    er = float(expected_return if expected_return is not None else max(score, 0.0) * 0.04)
    dd = abs(float(downside if downside is not None else min(score, 0.0) * 0.04))
    rr = er / dd if dd > 1e-12 else (float("inf") if er > 0 else 0.0)
    reasons: list[str] = []
    if len(vals) < 3:
        reasons.append("usable_factors<3")
    if score < min_score:
        reasons.append("score_below_threshold")
    if er < min_expected_return:
        reasons.append("expected_return_below_threshold")
    if rr < min_risk_reward:
        reasons.append("risk_reward_below_threshold")
    action = "buy" if not reasons else "hold"
    return TradeDecision(action, frequency, score, er, dd, rr, tuple(reasons))


__all__ = ["FACTOR_15", "FREQUENCIES", "TradeDecision", "decide_trade"]
