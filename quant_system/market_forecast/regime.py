"""
regime.py — QuantV6 市场状态识别
牛熊判断（300MA/144MA）、波动率聚类、马尔可夫状态转移。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.indicators import ma, rolling_vol
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.regime")


@dataclass
class Regime:
    """市场状态。"""
    trend: str = "unknown"            # bull / bear / range
    volatility: str = "medium"        # low / medium / high
    regime_label: str = "unknown"     # 组合标签，如 bull_high
    current_state: str = "unknown"    # 固定阈值粗分状态（老版独有修复反向移植）
    price_vs_300ma: float = 0.0       # 价格相对300MA偏离%
    price_vs_144ma: float = 0.0
    vol_annual: float = 0.0           # 年化波动率%
    transition_probs: dict = field(default_factory=dict)  # 状态转移矩阵
    lookback_days: int = 300


def detect_regime(index_df: pd.DataFrame, lookback: int = 300) -> Regime:
    """
    识别市场状态。
    index_df: 指数日线（索引=日期，含 close 列）。
    """
    if index_df is None or len(index_df) < 150:
        return Regime()

    close = index_df["close"].astype(float)
    close = close.tail(lookback + 100)

    ma300 = ma(close, 300)
    ma144 = ma(close, 144)
    price = close.iloc[-1]

    # 牛熊：价格与长均线关系 + 均线斜率
    p300 = ma300.iloc[-1] if not pd.isna(ma300.iloc[-1]) else price
    p144 = ma144.iloc[-1] if not pd.isna(ma144.iloc[-1]) else price
    vs300 = (price / p300 - 1) * 100 if p300 else 0.0
    vs144 = (price / p144 - 1) * 100 if p144 else 0.0

    ma300_slope = (ma300.iloc[-1] - ma300.iloc[-20]) / ma300.iloc[-20] * 100 if len(ma300) > 20 and ma300.iloc[-20] else 0.0

    if vs300 > 3 and ma300_slope > 0:
        trend = "bull"
    elif vs300 < -3 and ma300_slope < 0:
        trend = "bear"
    else:
        trend = "range"

    # 固定阈值粗分（老版独有修复反向移植）：300MA 偏离 ±3% 直接定牛熊，
    # 与斜率判断解耦，避免斜率平缓时把强趋势误判为震荡。
    if vs300 > 3:
        current_state = "bull"
    elif vs300 < -3:
        current_state = "bear"
    else:
        current_state = "range"

    # 波动率聚类
    vol = rolling_vol(close, 20).iloc[-1]
    if vol < 18:
        vol_level = "low"
    elif vol < 35:
        vol_level = "medium"
    else:
        vol_level = "high"

    reg = Regime(
        trend=trend,
        volatility=vol_level,
        regime_label=f"{trend}_{vol_level}",
        current_state=current_state,
        price_vs_300ma=round(vs300, 2),
        price_vs_144ma=round(vs144, 2),
        vol_annual=round(float(vol), 2),
        lookback_days=lookback,
    )
    reg.transition_probs = estimate_transition(close)
    return reg


def estimate_transition(close: pd.Series, n_states: int = 3) -> dict:
    """
    马尔可夫状态转移：把日收益离散为 跌/平/涨 三态，估计转移概率矩阵。
    返回 {(prev_state, state): prob}。
    """
    ret = close.pct_change().dropna()
    if len(ret) < 100:
        return {}
    # 分位数离散化
    q33, q66 = ret.quantile([1 / 3, 2 / 3])
    if q33 == q66:
        # 常数收益序列（恒定涨幅/横盘）: 分位边界重合会导致 pd.cut 重复边界报错，
        # 退化为平/涨两档离散化。
        states = pd.cut(ret, bins=[-np.inf, q66, np.inf], labels=["flat", "up"])
    else:
        states = pd.cut(ret, bins=[-np.inf, q33, q66, np.inf], labels=["down", "flat", "up"])
    names = ["down", "flat", "up"]
    trans: dict[tuple[str, str], float] = {}
    counts = {a: {b: 0 for b in names} for a in names}
    prev = None
    for s in states:
        if prev is not None:
            counts[prev][s] += 1
        prev = s
    for a in names:
        total = sum(counts[a].values())
        for b in names:
            trans[(a, b)] = round(counts[a][b] / total, 3) if total else 0.0
    return trans


def next_state_odds(regime: Regime) -> dict:
    """基于转移矩阵的次日状态概率。"""
    if not regime.transition_probs:
        return {"down": 0.333, "flat": 0.333, "up": 0.333}
    probs = {k: v for k, v in regime.transition_probs.items() if k[0] == "up"}
    if not probs:
        return {"down": 0.333, "flat": 0.333, "up": 0.333}
    total = sum(probs.values())
    if total <= 0:
        return {"down": 0.333, "flat": 0.333, "up": 0.333}
    return {k[1]: round(v / total, 3) for k, v in probs.items()}


def is_bull(regime: Regime) -> bool:
    return regime.trend == "bull"


def is_bear(regime: Regime) -> bool:
    return regime.trend == "bear"
