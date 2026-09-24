"""
neutralize.py — V7.0 因子中性化与正交化（多因子七步法第 4 步）
=============================================================
- 行业中性化: 行业哑变量回归取残差（消除行业暴露）
- 市值中性化: 对 log(流通市值) 回归取残差
- Gram-Schmidt 正交化: 按 ICIR 从高到低逐步正交，消除因子间冗余
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 中性化/正交化独特保留；common_dates 已收敛至 base.py。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.base import common_dates

log = get_logger("qv6.factor.neutralize")


def neutralize(factor_wide: pd.DataFrame,
               industry_map: pd.Series | None = None,
               market_cap: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐日横截面中性化。

    factor_wide: index=date, columns=股票代码（因子原始值）
    industry_map: Series(index=股票代码 → 行业名)（单期快照；跨期用最新）
    market_cap: DataFrame(index=date, columns=股票代码) 流通市值（元）
    返回: 残差 DataFrame（同形状）
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    out = pd.DataFrame(index=factor_wide.index, columns=factor_wide.columns, dtype=float)
    for d in factor_wide.index:
        f = factor_wide.loc[d].astype(float)
        valid = f.dropna()
        if len(valid) < 50:
            out.loc[d] = f
            continue
        X = pd.DataFrame(index=valid.index)
        y = valid.values.astype(float)

        if industry_map is not None:
            ind = industry_map.reindex(valid.index).fillna("未知")
            dummies = pd.get_dummies(ind, prefix="ind", drop_first=False)
            # 鲁棒性（P1-1 接入生产后）：剔除只有 1 只股票的行业哑变量
            # （该股暴露无法从行业暴露中分离，落入截距），避免设计阵欠秩/近奇异
            # 触发 lstsq 的 DLASCL 数值告警而影响残差稳定性。
            keep_cols = [c for c in dummies.columns if dummies[c].sum() >= 2]
            if keep_cols:
                dummies = dummies[keep_cols]
            else:
                dummies = dummies.iloc[:, :0]
            X = pd.concat([X, dummies], axis=1)
        if market_cap is not None and d in market_cap.index:
            mc = market_cap.loc[d].reindex(valid.index)
            mc_log = np.log(mc.clip(lower=1e6))
            X = X.copy()
            X["log_mc"] = mc_log.values
        X = X.astype(float)

        # 最小二乘残差（带截距；行列数足够时用正规方程）
        X1 = np.column_stack([np.ones(len(y)), X.values])
        try:
            beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            resid = y - X1 @ beta
        except np.linalg.LinAlgError:
            resid = y - y.mean()
        out.loc[d, valid.index] = resid
    return out


def orthogonalize(panels: dict[str, pd.DataFrame],
                  order: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Gram-Schmidt 正交化（按 order 顺序逐步，逐日横截面）。

    panels: {因子名: DataFrame(date × stocks)}
    返回: {因子名: DataFrame} 正交化后的面板（同形状）。
    D4收敛登记: 独特因子保留
    """
    cols = [c for c in (order or list(panels.keys())) if c in panels]
    if not cols:
        return {}
    out = {c: pd.DataFrame(index=panels[c].index, columns=panels[c].columns, dtype=float)
           for c in cols}
    dates = common_dates(panels)
    for d in dates:
        cross = {c: panels[c].loc[d].astype(float) for c in cols}
        for i, c in enumerate(cols):
            y = cross[c].dropna()
            if len(y) < 30:
                out[c].loc[d] = cross[c]
                continue
            if i == 0:
                out[c].loc[d, y.index] = y
                continue
            X = pd.DataFrame(index=y.index)
            ok = True
            for j in range(i):
                pv = out[cols[j]].loc[d].reindex(y.index).astype(float)
                if pv.notna().sum() < 30:
                    ok = False
                    break
                X[cols[j]] = pv
            if not ok or X.empty or len(X.columns) == 0:
                out[c].loc[d, c] = y
                continue
            X1 = np.column_stack([np.ones(len(y)), X.values])
            yv = y.values.astype(float)
            try:
                beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
                resid = yv - X1 @ beta
                out[c].loc[d, y.index] = resid
            except np.linalg.LinAlgError:
                out[c].loc[d, c] = y
    return out


def mad_winsorize_series(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 截尾（横截面）。
    D4收敛登记: 独特因子保留
    """
    from quant_system.market_forecast._support.common.math_utils import mad_winsorize
    return mad_winsorize(s.dropna().astype(float)).reindex(s.index)
