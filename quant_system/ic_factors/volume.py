"""
volume.py — QuantV6 量能因子库
量能趋势/量比/OBV/量价配合。输入 {code: DataFrame}，输出截面 Series。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.indicators import obv


def _vol(df: pd.DataFrame) -> pd.Series | None:
    if df is None or "volume" not in df.columns:
        return None
    return df["volume"].astype(float)


def volume_trend(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """量能趋势：近5日均量 / 前15日均量 - 1（放量为正）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < window + 5:
            continue
        recent = v.tail(5).mean()
        prev = v.tail(window).head(window - 5).mean()
        out[sym] = float(recent / prev - 1) if prev else 0.0
    return pd.Series(out, dtype=float)


def volume_surge(data: dict[str, pd.DataFrame]) -> pd.Series:
    """当日量能放大倍数（vs 20日均量）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < 21:
            continue
        avg = v.tail(21).head(20).mean()
        out[sym] = float(v.iloc[-1] / avg) if avg else 0.0
    return pd.Series(out, dtype=float)


def vol_20d(data: dict[str, pd.DataFrame]) -> pd.Series:
    """20日成交量均值（对数，流动性偏好）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < 20:
            continue
        out[sym] = float(np.log1p(v.tail(20).mean()))
    return pd.Series(out, dtype=float)


def volume_ratio(data: dict[str, pd.DataFrame], window: int = 5) -> pd.Series:
    """量比：当日量 / 前5日均量。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < window + 1:
            continue
        avg = v.tail(window + 1).head(window).mean()
        out[sym] = float(v.iloc[-1] / avg) if avg else 0.0
    return pd.Series(out, dtype=float)


def obv_slope(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """OBV 斜率（能量潮趋势）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        o = obv(close, v).tail(window)
        if len(o) < 5:
            continue
        x = np.arange(len(o))
        slope = np.polyfit(x, o.values, 1)[0]
        out[sym] = float(slope / (abs(o.mean()) + 1e-9))
    return pd.Series(out, dtype=float)


def volume_price_fit(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """量价配合度：量变化与价格变化方向一致性（同向为正）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        v = _vol(df)
        if v is None or len(v) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        p_ret = close.pct_change().tail(window)
        v_ret = v.pct_change().tail(window)
        corr = p_ret.corr(v_ret)
        out[sym] = float(corr) if not np.isnan(corr) else 0.0
    return pd.Series(out, dtype=float)


def turnover_rate(data: dict[str, pd.DataFrame]) -> pd.Series:
    """换手率近似：当日量 / 20日均量（已有 volume_ratio 语义，此处为绝对值档位）。
    D4收敛登记: 独特因子保留
    """
    return volume_ratio(data, 20)
