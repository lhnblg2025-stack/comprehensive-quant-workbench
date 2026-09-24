"""
indicators.py — QuantV6 技术指标库
纯 pandas 实现：MA/EMA/MACD/RSI/KDJ/BOLL/ATR/BIAS/量比/OBV/CCI。
输入 DataFrame（索引=日期，列含 open/high/low/close/volume），输出 Series。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 均线类 ────────────────────────────────────────────


def ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: 返回 DIF/DEA/HIST 三列。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI 相对强弱指标。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ 随机指标: 返回 K/D/J。"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def boll(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """布林带: 返回 mid/upper/lower/width/position。"""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    position = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower,
                         "width": width, "position": position})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR 平均真实波幅。"""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def bias(close: pd.Series, window: int = 12) -> pd.Series:
    """BIAS 乖离率: (close - MA) / MA * 100。"""
    m = ma(close, window)
    return (close - m) / m.replace(0, np.nan) * 100


def volume_ratio(volume: pd.Series, window: int = 5) -> pd.Series:
    """量比: 当日量 / 前 window 日均量。"""
    return volume / volume.shift(1).rolling(window).mean().replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV 能量潮。"""
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """CCI 顺势指标。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(window).mean()
    md = (tp - ma_tp).abs().rolling(window).mean()
    return (tp - ma_tp) / (0.015 * md).replace(0, np.nan)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """威廉指标 WR (0~-100，越接近-100越超卖)。"""
    high_n = df["high"].rolling(window).max()
    low_n = df["low"].rolling(window).min()
    return (high_n - df["close"]) / (high_n - low_n).replace(0, np.nan) * -100


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """唐奇安通道: 返回 upper/lower。"""
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    return pd.DataFrame({"upper": upper, "lower": lower})


def dmi(df: pd.DataFrame, n: int = 14, m: int = 6) -> pd.DataFrame:
    """DMI 趋向指标: 返回 PDI/MDI/ADX。"""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.rolling(n).sum()
    pdi = 100 * plus_dm.rolling(n).sum() / atr_n.replace(0, np.nan)
    mdi = 100 * minus_dm.rolling(n).sum() / atr_n.replace(0, np.nan)
    dx = (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) * 100
    adx = dx.rolling(m).mean()
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": adx})


def max_drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    """滚动最大回撤（百分比，负数）。"""
    roll_max = close.rolling(window, min_periods=2).max()
    return (close / roll_max - 1) * 100


def rolling_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """滚动年化波动率。"""
    ret = close.pct_change()
    return ret.rolling(window).std(ddof=0) * np.sqrt(252) * 100
