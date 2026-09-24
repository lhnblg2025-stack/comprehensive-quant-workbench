"""
valuation_v7.py — V7.0 估值域因子（baostock 历史估值 → 分位/水平）
====================================================================
数据源: baostock query_history_k_data_plus（peTTM/pbMRQ/psTTM/pcfNcfTTM）
因子语义: 个股横截面。分位因子用近 3 年历史窗口，低分位=相对便宜。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.valuation_v7")


# 标准估值列（东财新格式）；baostock 旧格式读取后统一映射到这些列
STANDARD_VALUATION_COLUMNS = [
    "date", "close", "total_mv", "float_mv",
    "pe_ttm", "pe_lyr", "pb", "peg", "pcf", "ps",
]
_OLD_TO_STD = {
    "peTTM": "pe_ttm", "pbMRQ": "pb",
    "psTTM": "ps", "pcfNcfTTM": "pcf",
}


def normalize_valuation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一估值列名层：旧格式(baostock) → 标准格式(东财)，缺失标准列补 NaN。

    旧→新映射: peTTM→pe_ttm, pbMRQ→pb, psTTM→ps, pcfNcfTTM→pcf。
    已标准化的输入为幂等操作；其他列（code/date/isST 等）原样保留。
    新旧列同时存在（如 peTTM 与 pe_ttm 并存）时合并：
    标准列优先保留非 NaN 值，随后删除旧列，避免 rename 产生重复列。
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for old, std in _OLD_TO_STD.items():
        if old not in df.columns:
            continue
        if std in df.columns:
            # 新旧列共存：合并（标准列优先）后删除旧列
            df[std] = df[std].fillna(df[old])
            df = df.drop(columns=[old])
        else:
            df = df.rename(columns={old: std})
    for c in STANDARD_VALUATION_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _frame(data: dict) -> pd.DataFrame:
    df = data.get("valuation")
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_valuation_columns(df)


def _percentile_cross(df: pd.DataFrame, col: str) -> pd.Series:
    """每只股票最新值在自身历史(近3年)的分位。"""
    if col not in df.columns:
        return pd.Series(dtype=float)
    out = {}
    for code, g in df.groupby("code"):
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(v) < 60:
            continue
        cur = v.iloc[-1]
        out[code] = float((v <= cur).mean())
    return pd.Series(out, dtype=float)


def _level_cross(df: pd.DataFrame, col: str) -> pd.Series:
    """最新估值水平（原始值）。"""
    if col not in df.columns:
        return pd.Series(dtype=float)
    out = {}
    for code, g in df.groupby("code"):
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        if not v.empty:
            out[code] = v.iloc[-1]
    return pd.Series(out, dtype=float)


@register_factor(name="val_pe_ttm_pct", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PE-TTM近3年分位(低=便宜)", direction=-1)
def val_pe_ttm_pct(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _percentile_cross(_frame(data), "pe_ttm")


@register_factor(name="val_pb_pct", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PB近3年分位(低=便宜)", direction=-1)
def val_pb_pct(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _percentile_cross(_frame(data), "pb")


@register_factor(name="val_ps_pct", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PS-TTM近3年分位(低=便宜)", direction=-1)
def val_ps_pct(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _percentile_cross(_frame(data), "ps")


@register_factor(name="val_pcf_pct", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PCF近3年分位(低=便宜)", direction=-1)
def val_pcf_pct(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _percentile_cross(_frame(data), "pcf")


@register_factor(name="val_pe_ttm", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PE-TTM水平(低=便宜)", direction=-1)
def val_pe_ttm(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _level_cross(_frame(data), "pe_ttm")


@register_factor(name="val_pb", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PB水平(低=便宜)", direction=-1)
def val_pb(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _level_cross(_frame(data), "pb")


@register_factor(name="val_ps", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PS-TTM水平(低=便宜)", direction=-1)
def val_ps(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _level_cross(_frame(data), "ps")


@register_factor(name="val_pe_pb_ratio", category="value", freq="daily",
                 data_deps=["valuation"],
                 description="PE/PB比值(高=轻资产成长溢价)", direction=1)
def val_pe_pb_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _frame(data)
    if "pe_ttm" not in df.columns or "pb" not in df.columns:
        return pd.Series(dtype=float)
    out = {}
    for code, g in df.groupby("code"):
        pe = pd.to_numeric(g["pe_ttm"], errors="coerce").dropna()
        pb = pd.to_numeric(g["pb"], errors="coerce").dropna()
        if pe.empty or pb.empty or pb.iloc[-1] == 0:
            continue
        out[code] = float(pe.iloc[-1] / pb.iloc[-1])
    return pd.Series(out, dtype=float)
