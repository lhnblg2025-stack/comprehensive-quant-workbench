"""
technical.py — QuantV6 技术指标因子库
MACD背离/KDJ/威廉/DMI/唐奇安/布林带宽。方向统一：多头偏好。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.indicators import boll, donchian, dmi, kdj, macd, ma, williams_r


def macd_divergence(data: dict[str, pd.DataFrame]) -> pd.Series:
    """MACD 顶/底背离：价格新高但 MACD 柱更低 → 看空（方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 60 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        m = macd(close)
        hist = m["hist"].tail(30)
        price = close.tail(30)
        # 顶背离：价格创30日新高但 hist 峰值下降
        if price.iloc[-1] >= price.max() * 0.98 and hist.iloc[-1] < hist.max() * 0.9:
            out[sym] = -1.0
        elif price.iloc[-1] <= price.min() * 1.02 and hist.iloc[-1] > hist.min() * 1.1:
            out[sym] = 1.0  # 底背离看多
        else:
            out[sym] = 0.0
    return pd.Series(out, dtype=float)


def kdj_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """KDJ J 值（超买反向，方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 30 or not {"high", "low", "close"}.issubset(df.columns):
            continue
        k = kdj(df)
        out[sym] = float(k["j"].iloc[-1])
    return pd.Series(out, dtype=float)


def williams_factor(data: dict[str, pd.DataFrame], window: int = 14) -> pd.Series:
    """威廉 WR（0~-100，越接近-100 超卖越看多；方向=1 用 -WR）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 5 or not {"high", "low", "close"}.issubset(df.columns):
            continue
        wr = williams_r(df, window).iloc[-1]
        out[sym] = float(-wr)  # 超卖 → 正值 → 看多
    return pd.Series(out, dtype=float)


def dmi_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """DMI 趋向：PDI-MDI（多头强度）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 40 or not {"high", "low", "close"}.issubset(df.columns):
            continue
        d = dmi(df)
        out[sym] = float(d["pdi"].iloc[-1] - d["mdi"].iloc[-1])
    return pd.Series(out, dtype=float)


def donchian_breakout(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """唐奇安通道突破：收盘价相对通道位置（突破上轨=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 5 or "close" not in df.columns or "high" not in df.columns:
            continue
        dc = donchian(df, window)
        close = df["close"].astype(float)
        upper, lower = dc["upper"].iloc[-1], dc["lower"].iloc[-1]
        c = close.iloc[-1]
        if upper and lower and upper > lower:
            out[sym] = float((c - lower) / (upper - lower))
        else:
            out[sym] = 0.5
    return pd.Series(out, dtype=float)


def boll_width(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """布林带宽（波动通道宽度；方向=-1 收窄偏好/突破前夜）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        b = boll(df["close"].astype(float), window)
        out[sym] = float(b["width"].iloc[-1])
    return pd.Series(out, dtype=float)


def ma_trend_strength(data: dict[str, pd.DataFrame]) -> pd.Series:
    """均线趋势强度：MA20/MA60 - 1（多头排列强度）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 70 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        m20 = ma(close, 20).iloc[-1]
        m60 = ma(close, 60).iloc[-1]
        out[sym] = float(m20 / m60 - 1) if m60 else 0.0
    return pd.Series(out, dtype=float)
