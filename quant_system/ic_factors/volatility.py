"""
volatility.py — QuantV6 波动率因子库
已实现波动率/下行波动/Beta/最大回撤/ATR。方向：低波动偏好（负向）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.indicators import atr, max_drawdown


def _close(df: pd.DataFrame) -> pd.Series | None:
    if df is None or "close" not in df.columns:
        return None
    return df["close"].astype(float)


def realized_volatility(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """已实现波动率（20日收益标准差，年化%）。方向=-1。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < window:
            continue
        r = c.pct_change().tail(window)
        out[sym] = float(r.std(ddof=0) * np.sqrt(252) * 100)
    return pd.Series(out, dtype=float)


def downside_volatility(data: dict[str, pd.DataFrame], window: int = 60) -> pd.Series:
    """下行波动率（仅负收益的标准差）。方向=-1。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < window:
            continue
        r = c.pct_change().tail(window)
        neg = r[r < 0]
        out[sym] = float(neg.std(ddof=0) * np.sqrt(252) * 100) if len(neg) > 2 else 0.0
    return pd.Series(out, dtype=float)


def max_dd_12m(data: dict[str, pd.DataFrame]) -> pd.Series:
    """近250日最大回撤（百分比，负数；方向=1 回撤小偏好）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < 60:
            continue
        out[sym] = float(max_drawdown(c.tail(250), 250).iloc[-1])
    return pd.Series(out, dtype=float)


def beta_60d(data: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None) -> pd.Series:
    """60日 Beta（对沪深300/基准指数，无基准时对全体均值）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    bm_ret = None
    if benchmark is not None and len(benchmark) > 60 and "close" in benchmark.columns:
        bm_ret = benchmark["close"].astype(float).pct_change()
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < 61:
            continue
        r = c.pct_change().tail(60)
        if bm_ret is not None:
            b = bm_ret.tail(60)
            df2 = pd.DataFrame({"s": r, "b": b}).dropna()
            if len(df2) < 30:
                continue
            var_b = df2["b"].var()
            out[sym] = float(df2["s"].cov(df2["b"]) / var_b) if var_b > 0 else 1.0
        else:
            out[sym] = float(r.mean() * 100)
    return pd.Series(out, dtype=float)


def downside_beta(data: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None) -> pd.Series:
    """下行 Beta：仅市场下跌日的个股 Beta（方向=-1，抗跌偏好）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    bm_ret = None
    if benchmark is not None and len(benchmark) > 60 and "close" in benchmark.columns:
        bm_ret = benchmark["close"].astype(float).pct_change()
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < 61:
            continue
        r = c.pct_change().tail(60)
        if bm_ret is None:
            out[sym] = 1.0
            continue
        b = bm_ret.tail(60)
        df2 = pd.DataFrame({"s": r, "b": b}).dropna()
        down = df2[df2["b"] < 0]
        if len(down) < 10:
            continue
        var_b = down["b"].var()
        out[sym] = float(down["s"].cov(down["b"]) / var_b) if var_b > 0 else 1.0
    return pd.Series(out, dtype=float)


def vol_change(data: dict[str, pd.DataFrame]) -> pd.Series:
    """波动率变化率：20日 vs 60日（波动放大为正；方向=1 波动收敛偏好）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < 80:
            continue
        r = c.pct_change()
        v20 = r.tail(20).std(ddof=0)
        v60 = r.tail(60).std(ddof=0)
        out[sym] = float(v20 / v60 - 1) if v60 > 0 else 0.0
    return pd.Series(out, dtype=float)


def atr_ratio(data: dict[str, pd.DataFrame]) -> pd.Series:
    """ATR/价格 比率（相对波动；方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        c = _close(df)
        if c is None or len(c) < 20 or not {"high", "low"}.issubset(df.columns):
            continue
        a = float(atr(df, 14).iloc[-1])
        price = float(c.iloc[-1])
        out[sym] = float(a / price) if price else 0.0
    return pd.Series(out, dtype=float)
