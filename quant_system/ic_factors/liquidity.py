"""
liquidity.py — QuantV6 流动性因子库
成交额/Amihud 非流动性/换手稳定性。方向：流动性越好越偏好。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amount_liquidity(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """20日平均成交额（对数，流动性偏好）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window:
            continue
        amt = df["成交额"].astype(float) if "成交额" in df.columns else \
            (df["amount"].astype(float) if "amount" in df.columns else None)
        if amt is None:
            continue
        out[sym] = float(np.log1p(amt.tail(window).mean()))
    return pd.Series(out, dtype=float)


def amihud_illiq(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """Amihud 非流动性：mean(|ret| / 成交额)。方向=-1。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        amt = df["成交额"].astype(float) if "成交额" in df.columns else \
            (df["amount"].astype(float) if "amount" in df.columns else None)
        if amt is None:
            continue
        ret = close.pct_change().abs()
        ratio = ret / amt.replace(0, np.nan)
        out[sym] = float(ratio.tail(window).mean() * 1e6) if ratio.tail(window).notna().any() else 0.0
    return pd.Series(out, dtype=float)


def turnover_stability(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """换手稳定性：量能变异系数取负（越稳定越偏好）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window:
            continue
        v = df["volume"].astype(float) if "volume" in df.columns else None
        if v is None:
            continue
        vol = v.tail(window)
        mean_v = vol.mean()
        out[sym] = float(-vol.std(ddof=0) / mean_v) if mean_v > 0 else 0.0
    return pd.Series(out, dtype=float)


def spread_approx(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """买卖价差近似：日内振幅均值（越小流动性越好；方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or not {"high", "low", "close"}.issubset(df.columns):
            continue
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        spread = ((high - low) / close.replace(0, np.nan)).tail(window)
        out[sym] = float(spread.mean())
    return pd.Series(out, dtype=float)
