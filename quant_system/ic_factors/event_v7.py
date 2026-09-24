"""
event_v7.py — V7.0 事件驱动因子（巨潮公告/增减持/回购/业绩超预期）
====================================================================
数据源: 巨潮 cninfo 结构化事件 + akshare 业绩预告。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor, FREQ_EVENT

log = get_logger("qv6.factor.event_v7")


@register_factor(name="insider_buy_amount", category="event", freq=FREQ_EVENT,
                 data_deps=["insider_trade"],
                 description="高管/股东增持金额(万)", direction=1)
def insider_buy_amount(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("insider")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "amount" in df.columns:
        return pd.to_numeric(df.set_index("code")["amount"], errors="coerce").dropna()
    return pd.Series(dtype=float)


@register_factor(name="buyback_progress", category="event", freq=FREQ_EVENT,
                 data_deps=["buyback"],
                 description="回购实施进度(已回购金额/预案)", direction=1)
def buyback_progress(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("buyback")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "progress" in df.columns:
        return pd.to_numeric(df.set_index("code")["progress"], errors="coerce").dropna()
    return pd.Series(dtype=float)


@register_factor(name="earnings_surprise", category="event", freq=FREQ_EVENT,
                 data_deps=["earnings_forecast"],
                 description="业绩预告超预期度(预增幅度)", direction=1)
def earnings_surprise(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("earnings")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "surprise" in df.columns:
        return pd.to_numeric(df.set_index("code")["surprise"], errors="coerce").dropna()
    if "change_ratio" in df.columns:
        return pd.to_numeric(df.set_index("code")["change_ratio"], errors="coerce").dropna()
    return pd.Series(dtype=float)


@register_factor(name="inq_letter_risk", category="event", freq=FREQ_EVENT,
                 data_deps=["cninfo_letters"],
                 description="监管问询函风险(有=1, 反向)", direction=-1)
def inq_letter_risk(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("letters")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    return pd.Series(1.0, index=df["code"].unique()).dropna()


@register_factor(name="forecast_upgrade", category="event", freq=FREQ_EVENT,
                 data_deps=["earnings_forecast"],
                 description="业绩预告类型(预增/略增=1, 预减/首亏=-1)", direction=1)
def forecast_upgrade(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("earnings")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "type" not in df.columns:
        return pd.Series(dtype=float)
    m = {"预增": 1.0, "略增": 0.5, "扭亏": 0.8, "续盈": 0.3,
         "预减": -1.0, "略减": -0.5, "首亏": -0.8, "续亏": -0.8, "不确定": 0.0}
    return df.set_index("code")["type"].map(m).dropna()
