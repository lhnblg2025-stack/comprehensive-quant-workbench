"""
exposure_v7.py — V7.0 市场因子 → 行业暴露映射（宏观/商品/利率 × 行业敏感度）
=============================================================================
- 把市场级因子（PMI/PPI/CPI/M2/利率/汇率/原油/猪价/BDI/DR007/票房/拥挤度）
  按经济逻辑的行业敏感系数映射为申万一级行业横截面因子
- 输出: 行业得分 Series(index=申万行业名)，universe="sw_industry"
- 策略层行业轮动/行业分层可直接消费；敏感系数后续可用历史回归校准
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor
from quant_system.ic_factors.derivatives_v7 import _market

log = get_logger("qv6.factor.exposure_v7")

# 申万一级行业（31个，与 sw_index_first_info 对齐）
SW_IND = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "银行", "非银金融", "综合", "建筑材料",
    "建筑装饰", "电力设备", "国防军工", "计算机", "传媒", "通信", "煤炭",
    "石油石化", "环保", "美容护理", "机械设备",
]

# 行业敏感系数表: 指标 → {行业: 系数(-3..+3)}，经济逻辑人工设定
# 未列出的行业默认 0（中性）
INDUSTRY_SENSITIVITY: dict[str, dict[str, float]] = {
    # 景气上行：周期资源品受益
    "pmi": {
        "钢铁": 2.0, "有色金属": 1.8, "基础化工": 1.5, "机械设备": 1.5,
        "建筑材料": 1.5, "建筑装饰": 1.2, "煤炭": 1.2, "石油石化": 1.0,
        "交通运输": 1.0, "电力设备": 0.8,
        "食品饮料": -0.5, "医药生物": -0.3,
    },
    # PPI上行：上游受益，下游成本承压
    "ppi": {
        "煤炭": 2.0, "石油石化": 1.8, "钢铁": 1.8, "有色金属": 1.5,
        "基础化工": 1.2, "建筑材料": 1.0,
        "家用电器": -1.0, "汽车": -0.8, "食品饮料": -0.5,
        "电力设备": -0.8, "机械设备": -0.5,
    },
    # CPI上行：必需消费受益，可选消费/银行承压
    "cpi": {
        "食品饮料": 1.5, "农林牧渔": 1.5, "商贸零售": 1.0,
        "医药生物": 0.8, "美容护理": 0.8, "社会服务": 0.5,
        "汽车": -0.5, "家用电器": -0.5, "银行": -0.3,
    },
    # M2扩张：流动性宽松，券商/地产/成长受益
    "m2": {
        "非银金融": 1.8, "房地产": 1.5, "计算机": 1.2, "电子": 1.0,
        "传媒": 1.0, "通信": 0.8, "国防军工": 0.8,
        "银行": 0.5, "公用事业": -0.3, "煤炭": -0.5,
    },
    # LPR下调：地产/基建融资成本降；银行息差收窄
    "lpr": {
        "房地产": 2.0, "建筑材料": 1.5, "建筑装饰": 1.5, "家用电器": 1.0,
        "汽车": 0.8, "银行": -1.5, "非银金融": -0.5,
    },
    # 美债利率上行：高估值成长承压；银行息差受益
    "us10y": {
        "电子": -1.5, "计算机": -1.5, "国防军工": -1.2, "医药生物": -1.0,
        "传媒": -1.0, "通信": -1.0, "电力设备": -0.8,
        "银行": 1.2, "煤炭": 0.8, "石油石化": 0.5,
    },
    # 人民币贬值：出口链受益；美元负债行业受损
    "usdcny": {
        "纺织服饰": 2.0, "家用电器": 1.5, "机械设备": 1.2, "电子": 1.0,
        "汽车": 0.8, "基础化工": 0.8, "轻工制造": 1.0,
        "交通运输": -1.5, "石油石化": -0.8, "公用事业": -0.5,
    },
    # 原油上行：上游开采受益；航空/交运成本受损
    "crude": {
        "石油石化": 2.0, "煤炭": 1.0, "有色金属": 0.5,
        "交通运输": -1.8, "基础化工": -0.5, "公用事业": -0.3,
        "汽车": -0.3,
    },
    # 猪价上行：养殖受益
    "pig": {
        "农林牧渔": 2.5, "食品饮料": -0.5,
    },
    # BDI上行：航运/港口景气
    "bdi": {
        "交通运输": 2.0, "机械设备": 0.8, "钢铁": 0.5, "煤炭": 0.5,
    },
    # 资金面收紧（DR007上行）：高杠杆/成长承压
    "dr007": {
        "房地产": -1.5, "非银金融": -1.5, "计算机": -1.0, "电子": -0.8,
        "银行": 0.5, "煤炭": 0.3,
    },
    # 票房上行：影视传媒受益
    "boxoffice": {
        "传媒": 2.5, "社会服务": 0.5, "商贸零售": 0.3,
    },
    # 市场拥挤度过高：全行业承压（风险偏好回落）
    "congestion": {
        "计算机": -2.0, "电子": -1.8, "传媒": -1.5, "通信": -1.2,
        "国防军工": -1.0, "电力设备": -1.0, "非银金融": -0.8,
        "银行": 0.8, "公用事业": 0.8, "煤炭": 0.8, "食品饮料": 0.5,
    },
}


def exposure_scores(market_value: float, metric: str) -> pd.Series:
    """市场值 × 行业敏感系数 → 行业得分（未列行业=0）。
    D4收敛登记: 独特因子保留
    """
    sens = INDUSTRY_SENSITIVITY.get(metric)
    if sens is None or market_value is None or not np.isfinite(market_value):
        return pd.Series(dtype=float)
    out = pd.Series(0.0, index=SW_IND, dtype=float)
    for ind, s in sens.items():
        if ind in out.index:
            out.loc[ind] = s * market_value
    return out


# ── 注册 12 个行业暴露因子（市场值来自 market data dict）──

@register_factor(name="exp_pmi", category="industry", universe="sw_industry",
                 data_deps=["market.macro_monthly"],
                 description="PMI行业暴露(景气上行利好周期资源)", direction=1)
def exp_pmi(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("macro_monthly") or {}
    d = df.get("pmi") if isinstance(df, dict) else None
    if d is None or d.empty or "制造业-指数" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["制造业-指数"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return exposure_scores(v.iloc[-1], "pmi")


@register_factor(name="exp_ppi", category="industry", universe="sw_industry",
                 data_deps=["market.macro_monthly"],
                 description="PPI行业暴露(上游受益/下游承压)", direction=1)
def exp_ppi(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "ppi")


@register_factor(name="exp_cpi", category="industry", universe="sw_industry",
                 data_deps=["market.macro_monthly"],
                 description="CPI行业暴露(通胀受益消费)", direction=1)
def exp_cpi(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "cpi")


@register_factor(name="exp_m2", category="industry", universe="sw_industry",
                 data_deps=["market.macro_monthly"],
                 description="M2行业暴露(流动性利好券商地产成长)", direction=1)
def exp_m2(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "m2")


@register_factor(name="exp_lpr", category="industry", universe="sw_industry",
                 data_deps=["market.rates"],
                 description="LPR行业暴露(降息利好地产/利空银行)", direction=1)
def exp_lpr(data: dict, **kw) -> pd.Series:
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
    # LPR 越低越利好，用负值让方向一致（exposure 因子 direction=1）
    return exposure_scores(-v.iloc[-1], "lpr")


@register_factor(name="exp_us10y", category="industry", universe="sw_industry",
                 data_deps=["market.rates"],
                 description="美债收益率行业暴露(高利率压制成长)", direction=-1)
def exp_us10y(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "us10y")


@register_factor(name="exp_usdcny", category="industry", universe="sw_industry",
                 data_deps=["market.rates"],
                 description="人民币汇率行业暴露(贬值利好出口链)", direction=1)
def exp_usdcny(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "usdcny")


@register_factor(name="exp_crude", category="industry", universe="sw_industry",
                 data_deps=["market.commodity"],
                 description="原油行业暴露(油价上行利好上游/利空航空)", direction=1)
def exp_crude(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("commodity") or {}
    d = df.get("crude") if isinstance(df, dict) else None
    if d is None or d.empty or "收盘价" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["收盘价"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return exposure_scores(v.iloc[-1], "crude")


@register_factor(name="exp_pig", category="industry", universe="sw_industry",
                 data_deps=["market.commodity"],
                 description="猪价行业暴露(猪价上行利好养殖)", direction=1)
def exp_pig(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("commodity") or {}
    d = df.get("lh") if isinstance(df, dict) else None
    if d is None or d.empty or "收盘价" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["收盘价"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return exposure_scores(v.iloc[-1], "pig")


@register_factor(name="exp_bdi", category="industry", universe="sw_industry",
                 data_deps=["market.commodity"],
                 description="BDI行业暴露(航运景气利好交运)", direction=1)
def exp_bdi(data: dict, **kw) -> pd.Series:
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
    return exposure_scores(v.iloc[-1], "bdi")


@register_factor(name="exp_dr007", category="industry", universe="sw_industry",
                 data_deps=["market.repo_rate"],
                 description="资金面行业暴露(DR007上行利空高杠杆成长)", direction=-1)
def exp_dr007(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("repo_rate")
    if df is None or df.empty or "FR007" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["FR007"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return exposure_scores(v.iloc[-1], "dr007")


@register_factor(name="exp_congestion", category="industry", universe="sw_industry",
                 data_deps=["market.market_heat"],
                 description="拥挤度行业暴露(过热回落利空热门成长)", direction=-1)
def exp_congestion(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = _market(data).get("market_heat") or {}
    d = df.get("congestion") if isinstance(df, dict) else None
    if d is None or d.empty or "congestion" not in d.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(d["congestion"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return exposure_scores(v.iloc[-1], "congestion")
