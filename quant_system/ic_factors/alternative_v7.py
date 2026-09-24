"""
alternative_v7.py — V7.0 另类域因子（票房/影视/市场情绪代理）
=============================================================
数据源: akshare（猫眼实时票房）
说明: 快递/手机出货/新能源车等接口在 akshare 1.18.64 无对应函数，
      用票房 + 现有情绪数据做另类域试点，后续接口可用再补。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor
from quant_system.ic_factors.derivatives_v7 import _mk, _market

log = get_logger("qv6.factor.alternative_v7")


@register_factor(name="boxoffice_total", category="sentiment",
                 data_deps=["market.boxoffice"],
                 description="实时票房总额(节假日/档期热度)", direction=1)
def boxoffice_total(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("boxoffice")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = [c for c in df.columns if "综合票房" in c or "票房" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.sum())


@register_factor(name="boxoffice_top_share", category="sentiment",
                 data_deps=["market.boxoffice"],
                 description="票房冠军占比(头部集中度=观影热度)", direction=1)
def boxoffice_top_share(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("boxoffice")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = [c for c in df.columns if "综合票房" in c or "票房" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df[col[0]], errors="coerce").dropna()
    if len(v) < 2 or v.sum() <= 0:
        return pd.Series(dtype=float)
    return _mk(v.iloc[0] / v.sum())
