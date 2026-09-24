"""
evaluator.py — QuantV6 因子评价
IC 序列/IC均值IR/分层回测/多空收益/稳定性。评估因子预测能力。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 因子评估(IC/分层/换手)独特保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.stats import ic_ir, ic_series


def information_coefficient(factor: pd.Series, forward_return: pd.Series,
                            method: str = "spearman", direction: int = 1) -> float:
    """截面 IC。

    P0-2：direction 参数（默认 1，向后兼容）用于原始因子值评价时
    同步乘方向，使 IC 口径与合成层一致（正 IC = 越大越看多）。
    """
    f = factor * (1 if direction == 1 else -1.0)
    df = pd.DataFrame({"f": f, "r": forward_return}).dropna()
    if len(df) < 5:
        return 0.0
    return float(df["f"].corr(df["r"], method=method))


def ic_series_over_time(factor_matrix: pd.DataFrame, return_matrix: pd.DataFrame,
                        method: str = "spearman") -> pd.Series:
    """逐期 IC 序列（行=日期）。"""
    return ic_series(factor_matrix, return_matrix, method)


def ic_summary(ic: pd.Series) -> dict:
    """IC 摘要：均值/IR/命中率/正IC占比。"""
    m, ir = ic_ir(ic)
    return {
        "ic_mean": round(m, 4),
        "ic_ir": round(ir, 4),
        "hit_rate": round(float((ic > 0).mean()), 4) if len(ic) else 0.0,
        "n": int(ic.notna().sum()),
    }


def quantile_returns(factor: pd.Series, forward_return: pd.Series,
                     q: int = 5) -> pd.Series:
    """分层回测：按因子分 q 组，各组的未来收益均值。"""
    df = pd.DataFrame({"f": factor, "r": forward_return}).dropna()
    if len(df) < q * 2:
        return pd.Series(dtype=float)
    try:
        bucket = pd.qcut(df["f"].rank(method="first"), q, labels=False) + 1
    except Exception:
        return pd.Series(dtype=float)
    return df["r"].groupby(bucket).mean()


def long_short_return(factor: pd.Series, forward_return: pd.Series, q: int = 5) -> float:
    """多空收益（最高组 - 最低组，A股不可做空仅作评价指标）。"""
    qr = quantile_returns(factor, forward_return, q=q)
    if qr.empty:
        return 0.0
    return float(qr.iloc[-1] - qr.iloc[0])


def top_quantile_return(factor: pd.Series, forward_return: pd.Series,
                        q: int = 5) -> float:
    """最优组收益（选股实际关心的）。"""
    qr = quantile_returns(factor, forward_return, q=q)
    return float(qr.iloc[-1]) if len(qr) else 0.0


def factor_stability(ic: pd.Series, window: int = 20) -> float:
    """因子稳定性：IC 滚动均值符号一致性（正占比）。"""
    ic = ic.dropna()
    if len(ic) < window:
        return float((ic > 0).mean()) if len(ic) else 0.0
    rolling = ic.rolling(window).mean().dropna()
    return float((rolling > 0).mean())


def turnover_rate(factor: pd.Series, prev_factor: pd.Series, top_n: int = 50) -> float:
    """因子换手率：本期 topN 与上期 topN 的 Jaccard 差异。"""
    cur = set(factor.sort_values(ascending=False).head(top_n).index)
    prev = set(prev_factor.sort_values(ascending=False).head(top_n).index)
    if not cur and not prev:
        return 0.0
    return 1 - len(cur & prev) / len(cur | prev)


def evaluate_factor(factor: pd.Series, forward_return: pd.Series,
                    method: str = "spearman", direction: int = 1) -> dict:
    """单期因子评价汇总（P0-2：支持 direction，与合成层同口径）。"""
    ic = information_coefficient(factor, forward_return, method, direction=direction)
    f = factor * (1 if direction == 1 else -1.0)
    ls = long_short_return(f, forward_return)
    top = top_quantile_return(f, forward_return)
    return {
        "ic": round(ic, 4),
        "long_short": round(ls, 4),
        "top_quantile": round(top, 4),
        "n": int(pd.DataFrame({"f": factor, "r": forward_return}).dropna().shape[0]),
    }


def evaluate_over_time(factor_matrix: pd.DataFrame, return_matrix: pd.DataFrame,
                       method: str = "spearman",
                       directions: dict[str, int] | None = None) -> dict:
    """多期因子评价。

    P0-2：directions 为 {因子名: direction} 映射（如 zoo.factor_meta()），
    逐列乘方向后再算 IC，保证反向因子 IC 与合成层同向（正 IC = 越大越看多）。
    """
    if directions:
        factor_matrix = factor_matrix.copy()
        for col in factor_matrix.columns:
            d = directions.get(col, 1)
            if d != 1:
                factor_matrix[col] = factor_matrix[col] * d
    ic = ic_series_over_time(factor_matrix, return_matrix, method)
    summ = ic_summary(ic)
    summ["stability"] = round(factor_stability(ic), 4)
    # 平均多空收益
    ls_list = []
    for d in ic.index:
        f = factor_matrix.loc[d]
        r = return_matrix.loc[d]
        ls_list.append(long_short_return(f, r))
    summ["avg_long_short"] = round(float(np.mean(ls_list)), 4) if ls_list else 0.0
    return summ
