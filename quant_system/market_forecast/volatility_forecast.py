"""
volatility_forecast.py — QuantV6 波动率预测
EWMA + 已实现波动率 + GARCH(1,1) 近似，预测次日/次5日波动区间。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.vol")


@dataclass
class VolForecast:
    """波动率预测。"""
    vol_1d: float = 0.0        # 次日预期波动率（年化%）
    vol_5d: float = 0.0        # 次5日预期波动率（年化%）
    vol_percentile: float = 0.5  # 当前波动率在历史分位
    realized_20d: float = 0.0
    garch_omega: float = 0.0
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    forecast_range: dict = field(default_factory=dict)  # {lo_pct, hi_pct}


def realized_vol(close: pd.Series, window: int = 20) -> float:
    ret = close.pct_change().dropna()
    if len(ret) < window:
        return 0.0
    return float(ret.tail(window).std(ddof=0) * np.sqrt(252) * 100)


def ewma_vol(close: pd.Series, span: int = 20) -> float:
    ret = close.pct_change().dropna()
    if len(ret) < 10:
        return 0.0
    var = ret.ewm(span=span, adjust=False).var().iloc[-1]
    return float(np.sqrt(max(var, 0)) * np.sqrt(252) * 100)


def garch11_approx(close: pd.Series) -> tuple[float, float, float, float]:
    """
    GARCH(1,1) 近似：用矩估计 omega/alpha/beta。
    返回 (omega, alpha, beta, forecast_var)。
    """
    ret = close.pct_change().dropna() * 100
    if len(ret) < 100:
        return 0.0, 0.1, 0.85, 1.0
    sigma2 = ret.var(ddof=0)
    # 简化矩估计（用一阶自相关近似 beta，无条件方差反推 omega）
    rho = ret.autocorr(1) if len(ret) > 2 else 0.05
    beta = max(0.1, min(0.95, abs(rho) * 0.9 + 0.7))
    alpha = 0.1
    omega = sigma2 * (1 - alpha - beta)
    # 一步预测
    last_sigma2 = ret.var(ddof=0)
    f_var = omega + alpha * (ret.iloc[-1] ** 2) + beta * last_sigma2
    return float(omega), alpha, beta, float(max(f_var, 0.01))


def predict_volatility(index_df: pd.DataFrame) -> VolForecast:
    """综合波动率预测。"""
    if index_df is None or len(index_df) < 60:
        return VolForecast()

    close = index_df["close"].astype(float)
    rv20 = realized_vol(close, 20)
    ewma = ewma_vol(close, 20)
    omega, alpha, beta, f_var = garch11_approx(close)
    garch_vol = float(np.sqrt(f_var))  # 日波动率%

    # 次日年化波动：EWMA 与 GARCH 加权
    vol_1d = 0.5 * ewma + 0.5 * garch_vol
    # 次5日：向长期均值回归
    long_vol = realized_vol(close, 120) or vol_1d
    vol_5d = 0.6 * vol_1d + 0.4 * long_vol

    # 历史分位
    hist = close.pct_change().dropna().rolling(20).std() * np.sqrt(252) * 100
    hist = hist.dropna()
    percentile = float((hist < rv20).mean()) if len(hist) else 0.5

    return VolForecast(
        vol_1d=round(vol_1d, 2),
        vol_5d=round(vol_5d, 2),
        vol_percentile=round(percentile, 3),
        realized_20d=round(rv20, 2),
        garch_omega=round(omega, 4),
        garch_alpha=round(alpha, 4),
        garch_beta=round(beta, 4),
        forecast_range={
            "lo_pct": round(-2 * vol_1d / 100 * 0.45, 2),   # 粗略 1 日 1σ 区间
            "hi_pct": round(2 * vol_1d / 100 * 0.45, 2),
        },
    )
