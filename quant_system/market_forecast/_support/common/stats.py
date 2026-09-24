"""
stats.py — QuantV6 绩效统计工具
夏普/索提诺/卡玛/最大回撤/年化波动/IC/IR 等。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.constants import TRADING_DAYS_PER_YEAR


def annualized_return(equity: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化收益率。"""
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / periods
    if years <= 0:
        return 0.0
    return (1 + total) ** (1 / years) - 1 if total > -1 else -1.0


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化波动率。"""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods))


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """夏普比率。"""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    excess = r.mean() * periods - risk_free
    return float(excess / (r.std(ddof=0) * np.sqrt(periods)))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """索提诺比率（下行波动）。"""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0].std(ddof=0)
    if pd.isna(downside) or downside == 0:
        return 0.0
    excess = r.mean() * periods - risk_free
    return float(excess / (downside * np.sqrt(periods)))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（百分比，负数）。"""
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1)
    return float(dd.min())


def calmar(equity: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|。"""
    ann = annualized_return(equity, periods)
    mdd = abs(max_drawdown(equity))
    return float(ann / mdd) if mdd > 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    """胜率。"""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """盈亏比。"""
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0


def information_coefficient(factor: pd.Series, fwd_return: pd.Series, method: str = "spearman") -> float:
    """IC：因子值与未来收益的截面相关性。"""
    df = pd.DataFrame({"f": factor, "r": fwd_return}).dropna()
    if len(df) < 5:
        return 0.0
    if method == "spearman":
        return float(df["f"].corr(df["r"], method="spearman"))
    return float(df["f"].corr(df["r"]))


def ic_series(factor_df: pd.DataFrame, fwd_return_df: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """逐期 IC 序列（行=日期，列=股票）。"""
    dates = factor_df.index.intersection(fwd_return_df.index)
    out = {}
    for d in dates:
        f = factor_df.loc[d]
        r = fwd_return_df.loc[d]
        df = pd.DataFrame({"f": f, "r": r}).dropna()
        if len(df) >= 5:
            out[d] = df["f"].corr(df["r"], method=method)
    return pd.Series(out, dtype=float)


def ic_ir(ic: pd.Series) -> tuple[float, float]:
    """IC 均值与 IR（IC均值/IC标准差）。"""
    ic = ic.dropna()
    if len(ic) == 0:
        return 0.0, 0.0
    m, s = ic.mean(), ic.std(ddof=0)
    return float(m), float(m / s) if s > 0 else 0.0


def hit_rate(ic: pd.Series) -> float:
    """IC 为正的比例。"""
    ic = ic.dropna()
    return float((ic > 0).mean()) if len(ic) else 0.0


def var_historic(returns: pd.Series, confidence: float = 0.95) -> float:
    """历史 VaR（负数表示损失）。"""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float(np.percentile(r, (1 - confidence) * 100))


def cvar_historic(returns: pd.Series, confidence: float = 0.95) -> float:
    """历史 CVaR。"""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    var = var_historic(r, confidence)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else var


def skew_kurt(returns: pd.Series) -> tuple[float, float]:
    """偏度与峰度。"""
    r = returns.dropna()
    if len(r) < 3:
        return 0.0, 0.0
    return float(r.skew()), float(r.kurt())


def metrics_report(equity: pd.Series, returns: pd.Series, risk_free: float = 0.0) -> dict:
    """综合绩效指标字典。"""
    return {
        "年化收益": round(annualized_return(equity) * 100, 2),
        "年化波动": round(annualized_vol(returns) * 100, 2),
        "夏普": round(sharpe(returns, risk_free), 2),
        "索提诺": round(sortino(returns, risk_free), 2),
        "卡玛": round(calmar(equity), 2),
        "最大回撤": round(max_drawdown(equity) * 100, 2),
        "胜率": round(win_rate(returns) * 100, 2),
        "盈亏比": round(profit_factor(returns), 2),
        "VaR95": round(var_historic(returns) * 100, 2),
        "CVaR95": round(cvar_historic(returns) * 100, 2),
    }
