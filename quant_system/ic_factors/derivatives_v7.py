"""
derivatives_v7.py — V7.0 衍生品域因子（股指基差/国债期货/期权波指/转债/资金面）
=============================================================================
数据源: akshare（新浪期货主力/乐咕QVIX/转债快照/回购利率/Shibor）
因子语义: 市场级（返回 {"MARKET": value}），用于 regime/择时/健康监控；
          横截面合成请用 exposure_v7 的行业映射版。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.derivatives_v7")


def _mk(value) -> pd.Series:
    """市场级单值 → 横截面 Series。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": float(value)})


def _market(data: dict) -> dict:
    m = data.get("market") or {}
    return m if isinstance(m, dict) else {}


# ── 1. 股指期货基差 ──────────────────────────────────────
@register_factor(name="if_basis_rate", category="sentiment",
                 data_deps=["market.futures_basis"],
                 description="沪深300股指期货主力基差率(升水>0乐观/贴水=恐慌)", direction=1)
def if_basis_rate(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("futures_basis")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    sub = df[df["contract"] == "IF"]
    if sub.empty:
        return pd.Series(dtype=float)
    return _mk(sub["basis_rate"].iloc[-1])


@register_factor(name="ic_im_basis_diff", category="sentiment",
                 data_deps=["market.futures_basis"],
                 description="IC与IM基差之差(小盘风格资金预期)", direction=1)
def ic_im_basis_diff(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("futures_basis")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    ic = df[df["contract"] == "IC"]
    im = df[df["contract"] == "IM"]
    if ic.empty or im.empty:
        return pd.Series(dtype=float)
    return _mk(ic["basis_rate"].iloc[-1] - im["basis_rate"].iloc[-1])


@register_factor(name="if_basis_5d", category="sentiment",
                 data_deps=["market.futures_basis"],
                 description="IF基差率5日变化(贴水加深=情绪恶化)", direction=-1)
def if_basis_5d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("futures_basis")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    sub = df[df["contract"] == "IF"]
    if len(sub) < 6:
        return pd.Series(dtype=float)
    return _mk(sub["basis_rate"].iloc[-1] - sub["basis_rate"].iloc[-6])


# ── 2. 国债期货（利率预期）───────────────────────────────
@register_factor(name="t_futures_ret20", category="bond",
                 data_deps=["market.bond_futures"],
                 description="国债期货主力20日动量(利率下行利好债券)", direction=1)
def t_futures_ret20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("bond_futures")
    if df is None or df.empty or len(df) < 21:
        return pd.Series(dtype=float)
    px = df["收盘价"].astype(float)
    return _mk(px.iloc[-1] / px.iloc[-21] - 1)


@register_factor(name="t_futures_ret5", category="bond",
                 data_deps=["market.bond_futures"],
                 description="国债期货主力5日动量", direction=1)
def t_futures_ret5(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("bond_futures")
    if df is None or df.empty or len(df) < 6:
        return pd.Series(dtype=float)
    px = df["收盘价"].astype(float)
    return _mk(px.iloc[-1] / px.iloc[-6] - 1)


# ── 3. 期权波动率 QVIX ───────────────────────────────────
@register_factor(name="qvix50_level", category="volatility",
                 data_deps=["market.qvix"],
                 description="50ETF期权中国波指QVIX水平(高=恐慌)", direction=-1)
def qvix50_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("qvix")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "50etf"]
    if sub.empty:
        return pd.Series(dtype=float)
    return _mk(sub["close"].astype(float).iloc[-1])


@register_factor(name="qvix50_change5", category="volatility",
                 data_deps=["market.qvix"],
                 description="QVIX 5日变化(上升=恐慌加剧)", direction=-1)
def qvix50_change5(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("qvix")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "50etf"]
    if len(sub) < 6:
        return pd.Series(dtype=float)
    v = sub["close"].astype(float)
    return _mk(v.iloc[-1] - v.iloc[-6])


@register_factor(name="qvix300_minus_50", category="volatility",
                 data_deps=["market.qvix"],
                 description="300ETF与50ETF QVIX之差(大盘股相对紧张度)", direction=-1)
def qvix300_minus_50(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("qvix")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    d50 = df[df["symbol"] == "50etf"]
    d300 = df[df["symbol"] == "300etf"]
    if d50.empty or d300.empty:
        return pd.Series(dtype=float)
    return _mk(float(d300["close"].iloc[-1]) - float(d50["close"].iloc[-1]))


# ── 4. 可转债（风险偏好）─────────────────────────────────
@register_factor(name="cb_double_low_median", category="value",
                 data_deps=["market.cb_spot"],
                 description="全市场转债双低中位数(低=市场便宜)", direction=-1)
def cb_double_low_median(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("cb_spot")
    if df is None or df.empty or "double_low" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["double_low"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.median())


@register_factor(name="cb_premium_avg", category="sentiment",
                 data_deps=["market.cb_spot"],
                 description="转债平均转股溢价率(高=投机情绪热)", direction=1)
def cb_premium_avg(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("cb_spot")
    if df is None or df.empty or "premium_rt" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["premium_rt"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.mean())


@register_factor(name="cb_avg_price", category="value",
                 data_deps=["market.cb_spot"],
                 description="转债平均价格(100以下占比高=熊市底部区)", direction=-1)
def cb_avg_price(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("cb_spot")
    if df is None or df.empty or "trade" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["trade"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.mean())


# ── 5. 资金面（回购/Shibor）─────────────────────────────
@register_factor(name="dr007_level", category="bond",
                 data_deps=["market.repo_rate"],
                 description="银行间7天回购利率水平(高=流动性紧)", direction=-1)
def dr007_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("repo_rate")
    if df is None or df.empty or "FR007" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["FR007"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="dr007_change5", category="bond",
                 data_deps=["market.repo_rate"],
                 description="FR007 5日变化(上升=资金面收紧)", direction=-1)
def dr007_change5(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("repo_rate")
    if df is None or df.empty or "FR007" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["FR007"], errors="coerce").dropna()
    if len(v) < 6:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] - v.iloc[-6])


@register_factor(name="shibor3m_level", category="bond",
                 data_deps=["market.shibor"],
                 description="Shibor 3M水平(中长期资金价格)", direction=-1)
def shibor3m_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("shibor")
    if df is None or df.empty or "rate" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["rate"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="shibor3m_change10", category="bond",
                 data_deps=["market.shibor"],
                 description="Shibor 3M 10日变化", direction=-1)
def shibor3m_change10(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("shibor")
    if df is None or df.empty or "rate" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["rate"], errors="coerce").dropna()
    if len(v) < 11:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] - v.iloc[-11])
