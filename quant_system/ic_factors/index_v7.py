"""
index_v7.py — V7.0 指数/风格域因子（估值分位/风格差/行业动量/市场热度）
=======================================================================
数据源: akshare（乐咕指数PE/市场PE/拥挤度/巴菲特指标/申万行业）
因子语义: 市场级（返回 {"MARKET": value}）+ 行业横截面（申万31行业映射）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor
from quant_system.ic_factors.derivatives_v7 import _mk, _market

log = get_logger("qv6.factor.index_v7")

# 乐咕指数 PE 的 symbol 名 → 内部名
_PE_SYMBOLS = ["上证50", "沪深300", "中证500", "中证1000", "创业板指"]


def _pe_frame(data: dict) -> pd.DataFrame:
    df = _market(data).get("index_pe")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    return df


@register_factor(name="hs300_pe_percentile", category="value",
                 data_deps=["market.index_pe"],
                 description="沪深300 PE历史分位(近5年, 低=便宜)", direction=-1)
def hs300_pe_percentile(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _pe_frame(data)
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "沪深300"]
    col = [c for c in sub.columns if "滚动市盈率" in c and "等权" not in c]
    if sub.empty or not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(sub[col[0]], errors="coerce").dropna()
    if len(v) < 250:
        return pd.Series(dtype=float)
    tail = v.tail(1250)  # 近5年
    return _mk((tail <= v.iloc[-1]).mean())


@register_factor(name="zz500_pe_percentile", category="value",
                 data_deps=["market.index_pe"],
                 description="中证500 PE历史分位(近5年)", direction=-1)
def zz500_pe_percentile(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _pe_frame(data)
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "中证500"]
    col = [c for c in sub.columns if "滚动市盈率" in c and "等权" not in c]
    if sub.empty or not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(sub[col[0]], errors="coerce").dropna()
    if len(v) < 250:
        return pd.Series(dtype=float)
    tail = v.tail(1250)
    return _mk((tail <= v.iloc[-1]).mean())


@register_factor(name="zz1000_pe_percentile", category="value",
                 data_deps=["market.index_pe"],
                 description="中证1000 PE历史分位(小盘估值)", direction=-1)
def zz1000_pe_percentile(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _pe_frame(data)
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "中证1000"]
    col = [c for c in sub.columns if "滚动市盈率" in c and "等权" not in c]
    if sub.empty or not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(sub[col[0]], errors="coerce").dropna()
    if len(v) < 250:
        return pd.Series(dtype=float)
    tail = v.tail(1250)
    return _mk((tail <= v.iloc[-1]).mean())


@register_factor(name="cyb_pe_percentile", category="value",
                 data_deps=["market.index_pe"],
                 description="创业板指 PE历史分位(成长风格估值)", direction=-1)
def cyb_pe_percentile(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _pe_frame(data)
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "创业板指"]
    col = [c for c in sub.columns if "滚动市盈率" in c and "等权" not in c]
    if sub.empty or not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(sub[col[0]], errors="coerce").dropna()
    if len(v) < 250:
        return pd.Series(dtype=float)
    tail = v.tail(1250)
    return _mk((tail <= v.iloc[-1]).mean())


@register_factor(name="small_minus_large_pe", category="value",
                 data_deps=["market.index_pe"],
                 description="小盘PE分位-大盘PE分位(风格估值差, 高=小盘贵)", direction=-1)
def small_minus_large_pe(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    small = zz1000_pe_percentile(data)
    large = hs300_pe_percentile(data)
    if small.empty or large.empty:
        return pd.Series(dtype=float)
    return _mk(float(small["MARKET"]) - float(large["MARKET"]))


@register_factor(name="market_pe_level", category="value",
                 data_deps=["market.market_heat"],
                 description="全市场平均PE(乐咕, 高=整体贵)", direction=-1)
def market_pe_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("market_heat") or {}
    d = df.get("market_pe") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "平均市盈率" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="market_congestion", category="sentiment",
                 data_deps=["market.market_heat"],
                 description="A股拥挤度(乐咕, 高=交易过热风险)", direction=-1)
def market_congestion(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("market_heat") or {}
    d = df.get("congestion") if isinstance(df, dict) else None
    if d is None or d.empty or "congestion" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["congestion"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="buffett_index", category="value",
                 data_deps=["market.market_heat"],
                 description="巴菲特指标(总市值/GDP, 高=泡沫风险)", direction=-1)
def buffett_index(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("market_heat") or {}
    d = df.get("buffett") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "指标" in c or "值" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[-1]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 申万行业估值（行业横截面）────────────────────────────
@register_factor(name="sw_industry_pe_rank", category="industry",
                 universe="sw_industry", freq="daily",
                 data_deps=["market.sw_industry"],
                 description="申万一级行业PE分位排名(低PE行业=相对便宜)", direction=-1)
def sw_industry_pe_rank(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("sw_industry")
    if df is None or df.empty or "TTM(滚动)市盈率" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["TTM(滚动)市盈率"], errors="coerce")
    names = df["行业名称"].astype(str)
    out = pd.Series(v.values, index=names).dropna()
    if out.empty:
        return pd.Series(dtype=float)
    return out.rank(pct=True)


@register_factor(name="sw_industry_pb_rank", category="industry",
                 universe="sw_industry", freq="daily",
                 data_deps=["market.sw_industry"],
                 description="申万一级行业PB分位排名(低PB行业=相对便宜)", direction=-1)
def sw_industry_pb_rank(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("sw_industry")
    if df is None or df.empty or "市净率" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["市净率"], errors="coerce")
    names = df["行业名称"].astype(str)
    out = pd.Series(v.values, index=names).dropna()
    if out.empty:
        return pd.Series(dtype=float)
    return out.rank(pct=True)


@register_factor(name="sw_industry_dividend_rank", category="industry",
                 universe="sw_industry", freq="daily",
                 data_deps=["market.sw_industry"],
                 description="申万一级行业股息率排名(高股息行业=防御价值)", direction=1)
def sw_industry_dividend_rank(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("sw_industry")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = [c for c in df.columns if "股息" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df[col[0]], errors="coerce")
    names = df["行业名称"].astype(str)
    out = pd.Series(v.values, index=names).dropna()
    if out.empty:
        return pd.Series(dtype=float)
    return out.rank(pct=True)
