"""
fundamental.py — QuantV6 基本面因子库
估值/盈利/成长类因子。输入 {code: {pe, pb, roe, ...}} 或 DataFrame，输出截面 Series。
方向统一：数值越大越看多（低PE → 取负处理）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get(df: pd.DataFrame, field: str) -> pd.Series | None:
    if df is None:
        return None
    if field in df.columns:
        return df[field].astype(float)
    return None


def pe_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """PE 估值（低估值偏好，方向=-1：PE 低 → 高分）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        pe = _get(df, "pe") if df is not None else None
        if pe is None or len(pe) == 0:
            continue
        v = float(pe.iloc[-1])
        out[sym] = float(v) if v > 0 else 0.0
    return pd.Series(out, dtype=float)


def pb_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """PB 估值（方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        pb = _get(df, "pb") if df is not None else None
        if pb is None or len(pb) == 0:
            continue
        v = float(pb.iloc[-1])
        out[sym] = float(v) if v > 0 else 0.0
    return pd.Series(out, dtype=float)


def roe_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """ROE 盈利能力（方向=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        roe = _get(df, "roe") if df is not None else None
        if roe is None or len(roe) == 0:
            continue
        out[sym] = float(roe.iloc[-1])
    return pd.Series(out, dtype=float)


def gross_margin_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """毛利率（方向=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        gm = _get(df, "gross_margin") if df is not None else None
        if gm is None or len(gm) == 0:
            continue
        out[sym] = float(gm.iloc[-1])
    return pd.Series(out, dtype=float)


def revenue_growth_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """营收增速（方向=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        g = _get(df, "revenue_growth") if df is not None else None
        if g is None or len(g) == 0:
            continue
        out[sym] = float(g.iloc[-1])
    return pd.Series(out, dtype=float)


def profit_growth_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """利润增速（方向=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        g = _get(df, "profit_growth") if df is not None else None
        if g is None or len(g) == 0:
            continue
        out[sym] = float(g.iloc[-1])
    return pd.Series(out, dtype=float)


def dividend_yield_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """股息率（方向=1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        dy = _get(df, "dividend_yield") if df is not None else None
        if dy is None or len(dy) == 0:
            continue
        out[sym] = float(dy.iloc[-1])
    return pd.Series(out, dtype=float)


def peg_factor(data: dict[str, pd.DataFrame]) -> pd.Series:
    """PEG = PE / 利润增速（低 PEG 偏好，方向=-1）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        if df is None:
            continue
        pe = _get(df, "pe")
        g = _get(df, "profit_growth")
        if pe is None or g is None or len(pe) == 0 or len(g) == 0:
            continue
        pe_v = float(pe.iloc[-1])
        g_v = float(g.iloc[-1])
        if pe_v > 0 and g_v > 0:
            out[sym] = float(pe_v / g_v)
        else:
            out[sym] = 0.0
    return pd.Series(out, dtype=float)


def log_market_cap(data: dict[str, pd.DataFrame]) -> pd.Series:
    """市值对数（大盘偏好；方向=1，用于市值中性化与风格判断）。
    D4收敛登记: 独特因子保留
    """
    out = {}
    for sym, df in data.items():
        mc = _get(df, "market_cap") if df is not None else None
        if mc is None or len(mc) == 0:
            continue
        v = float(mc.iloc[-1])
        out[sym] = float(np.log1p(v)) if v > 0 else 0.0
    return pd.Series(out, dtype=float)
