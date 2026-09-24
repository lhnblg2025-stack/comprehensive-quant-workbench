"""
tail_risk.py — QuantV6 尾部风险预测
大跌(<-2%)/大涨(>+2%)条件概率：历史极值分布 + 过热信号 + 乖离极端值。
直接回答"明天会不会大跌"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_system.market_forecast._support.common.constants import HEAT_LIMIT_UP, HEAT_RISE_RATIO
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.tailrisk")

DROP_TH = -2.0
RISE_TH = 2.0


@dataclass
class TailRisk:
    """尾部风险预测。"""
    p_drop_2pct: float = 0.10      # 次日大跌(<-2%)概率
    p_rise_2pct: float = 0.10      # 次日大涨(>+2%)概率
    heat_flag: bool = False
    overheat_after_big_rise: bool = False  # 过热+大涨后（高危形态）
    overbought_flag: bool = False  # 超买标志（BIAS 乖离极端，老版独有修复反向移植）
    sentiment_ok: bool = True      # 情绪活跃度数据是否可用（False=数据缺失，不得据此判"无尾部风险"）
    historical_base_rate: float = 0.08
    conditional_stats: dict = field(default_factory=dict)


def historical_base_rates(index_df: pd.DataFrame) -> tuple[float, float]:
    """历史次日大跌/大涨基础概率。"""
    if index_df is None or len(index_df) < 300:
        return 0.08, 0.10
    ret = index_df["close"].pct_change() * 100
    next_ret = ret.shift(-1)
    drop_rate = float((next_ret < DROP_TH).mean())
    rise_rate = float((next_ret > RISE_TH).mean())
    return drop_rate, rise_rate


def overheat_condition_stats(index_df: pd.DataFrame) -> dict:
    """
    过热日（单日大涨>=2.5%）后次日统计。
    这是用户最关心的"大涨次日"风险：历史上大涨次日收益显著衰减。
    """
    if index_df is None or len(index_df) < 300:
        return {}
    close = index_df["close"].astype(float)
    pct = close.pct_change() * 100
    next_pct = pct.shift(-1)
    hot = (pct >= 2.5)
    df = pd.DataFrame({"hot": hot, "next": next_pct})
    hot_next = df[df["hot"]]["next"].dropna()
    if len(hot_next) < 10:
        return {}
    return {
        "n": int(len(hot_next)),
        "next_mean": round(float(hot_next.mean()), 2),
        "next_median": round(float(hot_next.median()), 2),
        "p_down": round(float((hot_next < 0).mean()), 3),
        "p_drop_2pct": round(float((hot_next < DROP_TH).mean()), 3),
        "p_rise_2pct": round(float((hot_next > RISE_TH).mean()), 3),
    }


def predict_tail_risk(index_df: pd.DataFrame, activity: dict | None = None) -> TailRisk:
    """综合尾部风险预测。"""
    drop_rate, rise_rate = historical_base_rates(index_df)
    heat_flag = False
    overheat_after = False
    cond = overheat_condition_stats(index_df)
    # W2.5 P0 修复：数据缺失(total<=0)时不得当作"无过热/无尾部风险"，显式标记 sentiment_ok=False
    sentiment_ok = bool(activity) and int(activity.get("total", 0)) > 0

    # 过热信号（今日）
    if sentiment_ok:
        limit_up = int(activity.get("limit_up", 0))
        total = int(activity.get("total", 0))
        rise_ratio = activity.get("rise", 0) / total
        heat_flag = limit_up >= HEAT_LIMIT_UP or rise_ratio >= HEAT_RISE_RATIO

        # 今日大涨 + 过热 → 高危（历史：大涨次日收益衰减 + 下跌概率上升）
        if index_df is not None and len(index_df) > 1:
            close = index_df["close"].astype(float)
            today_pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            if today_pct >= 2.5 and heat_flag:
                overheat_after = True
                # 用条件统计放大风险
                if cond:
                    drop_rate = 0.5 * drop_rate + 0.5 * cond["p_drop_2pct"]
                    rise_rate = 0.5 * rise_rate + 0.5 * cond["p_rise_2pct"]

    # 乖离极端值（超买）：BIAS 过高时大跌概率上升
    overbought_flag = False
    if index_df is not None and len(index_df) >= 30:
        close = index_df["close"].astype(float)
        ma20 = close.rolling(20).mean()
        bias20 = (close.iloc[-1] / ma20.iloc[-1] - 1) * 100
        if bias20 > 10:
            drop_rate = min(0.5, drop_rate * 1.4)
            overheat_after = overheat_after or True
            overbought_flag = True

    return TailRisk(
        p_drop_2pct=round(min(drop_rate, 0.8), 3),
        p_rise_2pct=round(min(rise_rate, 0.8), 3),
        heat_flag=heat_flag,
        overheat_after_big_rise=overheat_after,
        overbought_flag=overbought_flag,
        sentiment_ok=sentiment_ok,
        historical_base_rate=round(drop_rate, 3),
        conditional_stats=cond,
    )
