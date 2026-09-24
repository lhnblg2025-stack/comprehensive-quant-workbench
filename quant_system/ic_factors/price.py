"""
price.py — QuantV6 价格因子库
动量/反转/均线/通道/超买超卖类因子。输入 {code: DataFrame}，输出截面 Series。
因子方向统一：数值越大越看多（多头），反转/超买类取负。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.indicators import bias, boll, cci, macd, ma, rsi
from quant_system.ic_factors.base import latest_return


def _series(func):
    """装饰器：把 dict 结果包装成 Series。"""
    import functools

    @functools.wraps(func)
    def wrapper(data: dict[str, pd.DataFrame], *args, **kwargs) -> pd.Series:
        out = func(data, *args, **kwargs)
        return out if isinstance(out, pd.Series) else pd.Series(out, dtype=float)
    return wrapper


def momentum(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """N 日动量（窗口可配）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    return latest_return(data, window)


def momentum_1m(data: dict[str, pd.DataFrame]) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    return latest_return(data, 21)


def momentum_3m(data: dict[str, pd.DataFrame]) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    return latest_return(data, 63)


def momentum_6m(data: dict[str, pd.DataFrame]) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    return latest_return(data, 126)


def momentum_12m(data: dict[str, pd.DataFrame]) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    return latest_return(data, 252)


def reversal(data: dict[str, pd.DataFrame], window: int = 5) -> pd.Series:
    """短期反转（方向=1：跌多了看多）。
    D4收敛登记: 独特因子保留
    """
    return -latest_return(data, window)


def ma_cross(data: dict[str, pd.DataFrame], fast: int = 5, slow: int = 20) -> pd.Series:
    """均线金叉强度：MA_fast/MA_slow - 1。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < slow + 1 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        mf = ma(close, fast).iloc[-1]
        ms = ma(close, slow).iloc[-1]
        out[sym] = float(mf / ms - 1) if ms else 0.0
    return pd.Series(out, dtype=float)


def macd_hist_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """MACD 柱状图（快线12/慢线26/信号9）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 40 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        m = macd(close)
        out[sym] = float(m["hist"].iloc[-1])
    return pd.Series(out, dtype=float)


def macd_cross(data: dict[str, pd.DataFrame]) -> pd.Series:
    """MACD 金叉信号（DIF-DEA 变化率，按价格归一化）。

    P2-12：原始输出为 gap 绝对差，高价股 DIF 天然大 → 归一化为
    相对价格的百分比变化，截面可比。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 40 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        m = macd(close)
        dif, dea = m["dif"], m["dea"]
        if len(dif) < 2:
            continue
        gap = dif.iloc[-1] - dea.iloc[-1]
        gap_prev = dif.iloc[-2] - dea.iloc[-2]
        px = float(close.iloc[-1])
        out[sym] = float((gap - gap_prev) / px * 100) if px > 0 else 0.0
    return pd.Series(out, dtype=float)


def boll_pos_factor(data: dict[str, pd.DataFrame], window: int = 20) -> pd.Series:
    """布林带位置（0~1，越接近1越超买；方向=-1 超买反向）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        b = boll(close, window)
        out[sym] = float(b["position"].iloc[-1])
    return pd.Series(out, dtype=float)


def cci_20(data: dict[str, pd.DataFrame]) -> pd.Series:
    """CCI 顺势指标（方向=-1，超买反向）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 30 or not {"high", "low", "close"}.issubset(df.columns):
            continue
        out[sym] = float(cci(df, 20).iloc[-1])
    return pd.Series(out, dtype=float)


def rsi_factor(data: dict[str, pd.DataFrame], window: int = 14) -> pd.Series:
    """RSI（方向=-1，超买反向）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window + 5 or "close" not in df.columns:
            continue
        out[sym] = float(rsi(df["close"].astype(float), window).iloc[-1])
    return pd.Series(out, dtype=float)


def high_low_position(data: dict[str, pd.DataFrame], window: int = 252) -> pd.Series:
    """52周高低位置：(close - low252) / (high252 - low252)，0~1。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < 60 or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        h = close.rolling(window).max().iloc[-1]
        l = close.rolling(window).min().iloc[-1]
        out[sym] = float((close.iloc[-1] - l) / (h - l)) if h > l else 0.5
    return pd.Series(out, dtype=float)


def bias_factor(data: dict[str, pd.DataFrame], window: int = 12) -> pd.Series:
    """BIAS 乖离率（方向=-1，乖离过大反向）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        out[sym] = float(bias(df["close"].astype(float), window).iloc[-1])
    return pd.Series(out, dtype=float)


def new_high_distance(data: dict[str, pd.DataFrame], window: int = 60) -> pd.Series:
    """距 60 日新高距离%（负数=未创新高，正数=创历史新高后回落幅度）。

    P2-12：优先用 high 列（真实高点），无 high 列时退化为 close。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        if "high" in df.columns:
            hi = df["high"].astype(float).rolling(window).max().iloc[-1]
        else:
            hi = close.rolling(window).max().iloc[-1]
        out[sym] = float(close.iloc[-1] / hi - 1) * 100 if hi else 0.0
    return pd.Series(out, dtype=float)


def new_low_distance(data: dict[str, pd.DataFrame], window: int = 60) -> pd.Series:
    """距 60 日新低距离%（负数=创新低）。

    P2-12：优先用 low 列（真实低点），无 low 列时退化为 close。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) < window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        if "low" in df.columns:
            lo = df["low"].astype(float).rolling(window).min().iloc[-1]
        else:
            lo = close.rolling(window).min().iloc[-1]
        out[sym] = float(close.iloc[-1] / lo - 1) * 100 if lo else 0.0
    return pd.Series(out, dtype=float)
