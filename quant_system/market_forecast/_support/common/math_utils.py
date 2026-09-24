"""
math_utils.py — QuantV6 数学工具
去极值/标准化/加权/指数衰减/线性回归斜率等纯函数实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """MAD/分位数去极值：将超出 [lower, upper] 分位数的值截断。"""
    s = s.astype(float)
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def mad_winsorize(s: pd.Series, n: float = 3.0) -> pd.Series:
    """MAD 去极值：|x - median| > n * 1.4826 * MAD 的值截断。"""
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return s
    scale = 1.4826 * mad
    lo, hi = med - n * scale, med + n * scale
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """z-score 标准化，clip ±3。全 NaN 或零标准差时返回 0。"""
    s = s.astype(float)
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    out = (s - s.mean()) / std
    return out.clip(-3.0, 3.0)


def pct_rank(s: pd.Series) -> pd.Series:
    """横截面百分位排名 0~1（越大越好）。"""
    return s.rank(pct=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """加权均值，忽略 NaN。"""
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def exp_decay_weights(n: int, halflife: int) -> np.ndarray:
    """指数衰减权重（最近权重最大），半衰期 halflife。"""
    decay = 0.5 ** (1.0 / halflife)
    w = np.array([decay ** (n - 1 - i) for i in range(n)])
    return w / w.sum()


def linreg_slope(y: np.ndarray) -> float:
    """序列线性回归斜率（按索引归一化）。NaN 填充为线性插值。"""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    if np.isnan(y).any():
        idx = np.arange(len(y))
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            return 0.0
        slope = np.polyfit(idx[valid], y[valid], 1)[0]
        return float(slope)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，除零返回 default。"""
    try:
        if b == 0 or pd.isna(b):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def clip_pct(x: float, lo: float = -100.0, hi: float = 100.0) -> float:
    """百分比截断。"""
    return max(lo, min(hi, x))


def exponential_smooth(s: pd.Series, alpha: float = 0.1) -> pd.Series:
    """指数平滑。"""
    return s.ewm(alpha=alpha, adjust=False).mean()


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 20) -> pd.Series:
    """滚动相关系数。"""
    return a.rolling(window).corr(b)


def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """协方差矩阵（NaN 安全）。"""
    return df.astype(float).cov()


def normalize_weights(d: dict) -> dict:
    """权重字典归一化（和为1），全零则等权。"""
    total = sum(v for v in d.values() if v > 0)
    if total <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d} if n else {}
    return {k: (v / total if v > 0 else 0.0) for k, v in d.items()}
