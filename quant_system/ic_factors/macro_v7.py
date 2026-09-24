"""
macro_v7.py — V7.0 宏观域因子（PMI/CPI/PPI/M2/社融/利率/汇率/商品/BDI）
=======================================================================
数据源: akshare（统计局/央行/海关/乐咕/新浪期货/BDI）
因子语义: 市场级（返回 {"MARKET": value}），供 regime/择时/健康监控；
          行业映射版见 exposure_v7.py。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor
from quant_system.ic_factors.derivatives_v7 import _mk, _market

log = get_logger("qv6.factor.macro_v7")


# ── 1. 景气度：PMI ───────────────────────────────────────
@register_factor(name="pmi_level", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="制造业PMI水平(>50扩张/<50收缩)", direction=1)
def pmi_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("pmi") if isinstance(df, dict) else None
    if d is None or d.empty or "制造业-指数" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["制造业-指数"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="pmi_change", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="制造业PMI环比变化(边际改善)", direction=1)
def pmi_change(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("pmi") if isinstance(df, dict) else None
    if d is None or d.empty or "制造业-指数" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["制造业-指数"], errors="coerce").dropna()
    if len(v) < 2:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] - v.iloc[-2])


@register_factor(name="non_man_pmi", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="非制造业PMI(服务业景气)", direction=1)
def non_man_pmi(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("pmi") if isinstance(df, dict) else None
    if d is None or d.empty or "非制造业-指数" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["非制造业-指数"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 2. 通胀：CPI/PPI ─────────────────────────────────────
@register_factor(name="cpi_yoy", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="CPI同比(通胀水平)", direction=-1)
def cpi_yoy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("cpi") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "同比" in c and "全国" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="ppi_yoy", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="PPI同比(工业品价格/上游利润)", direction=1)
def ppi_yoy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("ppi") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "同比" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="ppi_cpi_gap", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="PPI-CPI剪刀差(上游利润挤压方向)", direction=1)
def ppi_cpi_gap(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    ppi = ppi_yoy(data)
    cpi = cpi_yoy(data)
    if ppi.empty or cpi.empty:
        return pd.Series(dtype=float)
    return _mk(float(ppi["MARKET"]) - float(cpi["MARKET"]))


# ── 3. 流动性：M2/社融 ───────────────────────────────────
@register_factor(name="m2_yoy", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="M2同比(货币供给增速)", direction=1)
def m2_yoy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("m2") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    sub = d[d["商品"].astype(str).str.contains("货币和准货币", na=False)]
    if sub.empty:
        sub = d
    col = [c for c in sub.columns if "今值" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(sub[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="social_financing", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="社融增量(最新月份, 亿元)", direction=1)
def social_financing(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("shrzgm") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "社会融资规模增量" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 4. 利率：LPR ─────────────────────────────────────────
@register_factor(name="lpr_1y", category="bond",
                 data_deps=["market.rates"],
                 description="1年期LPR(低利率环境利好权益)", direction=-1)
def lpr_1y(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("rates") or {}
    d = df.get("lpr") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "LPR" in c and ("1Y" in c or "1年" in c or "一年" in c)]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 5. 海外：美债/汇率 ───────────────────────────────────
@register_factor(name="us10y_yield", category="bond",
                 data_deps=["market.rates"],
                 description="美国10年期国债收益率(全球无风险利率)", direction=-1)
def us10y_yield(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("rates") or {}
    d = df.get("us_rate") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "美国国债收益率10年" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="usdcny_level", category="macro",
                 data_deps=["market.rates"],
                 description="美元兑人民币(升值=人民币贬值, 利空外资)", direction=-1)
def usdcny_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("rates") or {}
    d = df.get("fx") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "中行汇买价" in c or "汇买价" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 6. 商品：原油/生猪/BDI ───────────────────────────────
@register_factor(name="crude_ret20", category="commodity",
                 data_deps=["market.commodity"],
                 description="原油20日动量(成本端/通胀预期)", direction=1)
def crude_ret20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("commodity") or {}
    d = df.get("crude") if isinstance(df, dict) else None
    if d is None or d.empty or "收盘价" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["收盘价"], errors="coerce").dropna()
    if len(v) < 21:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] / v.iloc[-21] - 1)


@register_factor(name="pig_fut_ret20", category="commodity",
                 data_deps=["market.commodity"],
                 description="生猪期货20日动量(猪周期/CPI食品项)", direction=1)
def pig_fut_ret20(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("commodity") or {}
    d = df.get("lh") if isinstance(df, dict) else None
    if d is None or d.empty or "收盘价" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["收盘价"], errors="coerce").dropna()
    if len(v) < 21:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] / v.iloc[-21] - 1)


@register_factor(name="bdi_level", category="commodity",
                 data_deps=["market.commodity"],
                 description="波罗的海干散货指数(全球贸易景气)", direction=1)
def bdi_level(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("commodity") or {}
    d = df.get("bdi") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "值" in c or "指数" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[-1]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


# ── 7. 进出口 ────────────────────────────────────────────
@register_factor(name="exports_yoy", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="出口金额同比(外需)", direction=1)
def exports_yoy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("exports") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "今值" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


@register_factor(name="imports_yoy", category="macro",
                 data_deps=["market.macro_monthly"],
                 description="进口金额同比(内需)", direction=1)
def imports_yoy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("imports") if isinstance(df, dict) else None
    if d is None or d.empty:
        return pd.Series(dtype=float)
    col = [c for c in d.columns if "今值" in c]
    if not col:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d[col[0]], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])
