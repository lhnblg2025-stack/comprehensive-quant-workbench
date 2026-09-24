"""
trend.py — QuantV6 趋势延续/反转预测
均线排列、动量加速度、量价配合；识别"动量衰竭"（价格新高但加速度转负）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_system.market_forecast._support.common.indicators import ma
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.trend")


@dataclass
class TrendPrediction:
    """趋势状态预测。"""
    continuation_prob: float = 0.5   # 趋势延续概率
    reversal_prob: float = 0.5       # 反转概率
    momentum_exhausted: bool = False # 动量衰竭（新高但加速度转负）
    trend_state: str = "unknown"     # strong_up/weak_up/strong_down/weak_down/choppy
    ma_alignment: str = "unknown"    # 多头排列/空头排列/纠缠
    detail: dict = field(default_factory=dict)


def _ma_alignment(close: pd.Series) -> str:
    """均线排列：MA5>MA10>MA20>MA60 多头；反之空头；否则纠缠。"""
    ma5, ma10, ma20, ma60 = ma(close, 5), ma(close, 10), ma(close, 20), ma(close, 60)
    if pd.isna(ma60.iloc[-1]):
        return "unknown"
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
        return "bull_aligned"
    if ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
        return "bear_aligned"
    return "mixed"


def predict_trend(index_df: pd.DataFrame) -> TrendPrediction:
    """趋势延续/反转预测。"""
    if index_df is None or len(index_df) < 80:
        return TrendPrediction()

    close = index_df["close"].astype(float)
    pct = close.pct_change() * 100

    # 动量与加速度
    mom20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0.0
    mom10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) > 11 else 0.0
    accel = mom10 - (close.iloc[-11] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0.0

    # 新高检测 + 动量衰竭
    high60 = close.rolling(60).max().iloc[-2] if len(close) > 60 else close.iloc[-2]
    is_new_high = close.iloc[-1] >= high60
    momentum_exhausted = is_new_high and accel < 0

    alignment = _ma_alignment(close)

    # 量价配合：近5日涨跌幅与量能方向
    vol_ok = True
    if "volume" in index_df.columns:
        vol5 = index_df["volume"].tail(5).mean()
        vol_prev = index_df["volume"].tail(15).head(10).mean()
        vol_ok = vol5 >= vol_prev if vol_prev > 0 else True

    # 趋势状态打分
    if alignment == "bull_aligned" and mom20 > 5:
        state = "strong_up"
        continuation = 0.7 if (not momentum_exhausted and vol_ok) else 0.5
    elif alignment == "bull_aligned":
        state = "weak_up"
        continuation = 0.55
    elif alignment == "bear_aligned" and mom20 < -5:
        state = "strong_down"
        continuation = 0.25  # 空头延续概率低（均值回归+政策底）
    elif alignment == "bear_aligned":
        state = "weak_down"
        continuation = 0.4
    else:
        state = "choppy"
        continuation = 0.5

    if momentum_exhausted:
        continuation = min(continuation, 0.4)  # 衰竭 → 反转风险

    return TrendPrediction(
        continuation_prob=round(continuation, 3),
        reversal_prob=round(1 - continuation, 3),
        momentum_exhausted=momentum_exhausted,
        trend_state=state,
        ma_alignment=alignment,
        detail={"mom20": round(mom20, 2), "mom10": round(mom10, 2),
                "accel": round(accel, 2), "is_new_high": bool(is_new_high)},
    )
