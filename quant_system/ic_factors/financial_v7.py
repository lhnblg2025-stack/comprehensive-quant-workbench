"""
financial_v7.py — V7.0 财务分析指标因子（新浪86列 → 盈利能力/营运/偿债）
=========================================================================
数据源: akshare stock_financial_analysis_indicator（新浪，季度）
区别于 fundamental_v7（baostock 三表）: 本模块用新浪财务分析指标，
覆盖总资产利润率/营业利润率/成本费用利润率/每股经营现金流等 86 列。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.financial_v7")


# 真实公告日字段（按优先级匹配；新浪财务分析指标无公告日，回退到估算锚点）
_PUB_DATE_FIELDS = ("公告日期", "发布日期", "公告日", "pub_date",
                    "pubDate", "ann_date", "announce_date", "announcement_date")


def _est_pub_date(report_period) -> pd.Timestamp:
    """保守公告日锚点：季报 = 报告期末 + 90 天，年报 = 报告期末 + 150 天。

    实盘在锚点日之前该报告期数据不可用，回测据此避免前视。
    解析失败返回 pd.NaT。
    """
    try:
        d = pd.to_datetime(report_period, errors="coerce")
    except Exception:  # noqa: BLE001
        return pd.NaT
    if pd.isna(d):
        return pd.NaT
    days = 150 if d.month == 12 else 90
    return d + pd.Timedelta(days=days)


def _pub_date_of(df: pd.DataFrame) -> pd.Series:
    """每行的公告日：优先真实公告日期字段，缺失时用 _est_pub_date(报告期)。"""
    for c in _PUB_DATE_FIELDS:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s
    if "日期" in df.columns:
        return df["日期"].map(_est_pub_date)
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _frame(data: dict) -> pd.DataFrame:
    df = data.get("financial")
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _latest(df: pd.DataFrame, col: str, asof=None) -> pd.Series:
    """取每只股票最新一期指标（横截面）。df 需含 code/日期/指标列。

    asof 为 None 时保持旧行为（按报告期排序取最新一期）；
    传入当前交易日（str/Timestamp）时，仅取"截至该日已公告"的最新一期：
    公告日 = 数据中的真实公告日期字段（若有），否则按报告期估算
    （季报 +90 天 / 年报 +150 天）。锚点前的报告期不可见 → 该股返回 NaN。
    """
    if col not in df.columns:
        return pd.Series(dtype=float)
    if asof is not None:
        asof = pd.to_datetime(asof, errors="coerce")
        if pd.isna(asof):
            log.warning(f"_latest({col}) asof 解析失败，退回不过滤")
            asof = None
        elif "pub_date" not in df.columns:
            df = df.copy()
            df["pub_date"] = _pub_date_of(df)
    out = {}
    for code, g in df.groupby("code"):
        g = g.copy()
        if "日期" in g.columns:
            g = g.sort_values("日期")  # 按报告期排序取最新
        if asof is not None and "pub_date" in g.columns:
            g = g[g["pub_date"] <= asof]  # 截至当前交易日已公告
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        if not v.empty:
            out[code] = v.iloc[-1]
    return pd.Series(out, dtype=float)


@register_factor(name="fa_roa", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="总资产利润率%(新浪, 资产回报)", direction=1)
def fa_roa(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "总资产利润率(%)", asof=kw.get("asof"))


@register_factor(name="fa_operating_margin", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="营业利润率%(经营效率)", direction=1)
def fa_operating_margin(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "营业利润率(%)", asof=kw.get("asof"))


@register_factor(name="fa_net_margin", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="销售净利率%(盈利质量)", direction=1)
def fa_net_margin(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "销售净利率(%)", asof=kw.get("asof"))


@register_factor(name="fa_cost_profit_ratio", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="成本费用利润率%(费用控制)", direction=1)
def fa_cost_profit_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "成本费用利润率(%)", asof=kw.get("asof"))


@register_factor(name="fa_roe_adj", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="净资产报酬率%(调整后ROE)", direction=1)
def fa_roe_adj(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "净资产报酬率(%)", asof=kw.get("asof"))


@register_factor(name="fa_eps_cfo", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="每股经营性现金流(盈利现金含量)", direction=1)
def fa_eps_cfo(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "每股经营性现金流(元)", asof=kw.get("asof"))


@register_factor(name="fa_capital_reserve", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="每股资本公积金(送转/扩张潜力)", direction=1)
def fa_capital_reserve(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "每股资本公积金(元)", asof=kw.get("asof"))


@register_factor(name="fa_retained_eps", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="每股未分配利润(累计盈利储备)", direction=1)
def fa_retained_eps(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "每股未分配利润(元)", asof=kw.get("asof"))


@register_factor(name="fa_main_cost_ratio", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="主营业务成本率%(成本占比, 反向)", direction=-1)
def fa_main_cost_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "主营业务成本率(%)", asof=kw.get("asof"))


@register_factor(name="fa_asset_turnover", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="总资产周转率(营运效率)", direction=1)
def fa_asset_turnover(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "总资产周转率(次)", asof=kw.get("asof"))


@register_factor(name="fa_interest_cover", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="利息保障倍数(偿债安全)", direction=1)
def fa_interest_cover(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "利息保障倍数", asof=kw.get("asof"))


@register_factor(name="fa_equity_growth", category="growth", freq="quarterly",
                 data_deps=["financial"],
                 description="股东权益增长率%(扩张性)", direction=1)
def fa_equity_growth(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "股东权益增长率(%)", asof=kw.get("asof"))


@register_factor(name="fa_sustain_growth", category="growth", freq="quarterly",
                 data_deps=["financial"],
                 description="可持续增长率%(内生增长能力)", direction=1)
def fa_sustain_growth(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "可持续增长率(%)", asof=kw.get("asof"))


@register_factor(name="fa_eps_adjusted", category="growth", freq="quarterly",
                 data_deps=["financial"],
                 description="扣非每股收益(核心盈利)", direction=1)
def fa_eps_adjusted(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "扣除非经常性损益后的每股收益(元)", asof=kw.get("asof"))


# ══════════════════════════════════════════════════════════
# 第二批：新浪86列剩余维度（每股/偿债/营运/现金流质量）
# ══════════════════════════════════════════════════════════

@register_factor(name="fa_eps_weighted", category="growth", freq="quarterly",
                 data_deps=["financial"],
                 description="加权每股收益(核心EPS)", direction=1)
def fa_eps_weighted(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "加权每股收益(元)", asof=kw.get("asof"))


@register_factor(name="fa_bps", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="每股净资产调整前(资产厚实度)", direction=1)
def fa_bps(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "每股净资产_调整前(元)", asof=kw.get("asof"))


@register_factor(name="fa_total_asset_profit", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="总资产净利润率%(ROA口径)", direction=1)
def fa_total_asset_profit(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "总资产净利润率(%)", asof=kw.get("asof"))


@register_factor(name="fa_main_biz_profit", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="主营业务利润率%(主业盈利)", direction=1)
def fa_main_biz_profit(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "主营业务利润率(%)", asof=kw.get("asof"))


@register_factor(name="fa_roe_weighted", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="净资产收益率加权(ROE口径)", direction=1)
def fa_roe_weighted(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "净资产收益率加权(%)", asof=kw.get("asof"))


@register_factor(name="fa_cfo_to_np", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="经营现金流/净利润(盈利含金量)", direction=1)
def fa_cfo_to_np(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    cfo = _latest(_frame(data), "每股经营性现金流(元)", asof=kw.get("asof"))
    np_ = _latest(_frame(data), "摊薄每股收益(元)", asof=kw.get("asof"))
    idx = cfo.index.intersection(np_.index)
    out = pd.Series(dtype=float)
    for c in idx:
        npv = np_[c]
        if npv is not None and abs(npv) > 1e-9:
            out[c] = cfo[c] / npv
    return out


@register_factor(name="fa_quick_ratio", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="速动比率(短期偿债)", direction=1)
def fa_quick_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "速动比率", asof=kw.get("asof"))


@register_factor(name="fa_current_ratio", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="流动比率(偿债安全)", direction=1)
def fa_current_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "流动比率", asof=kw.get("asof"))


@register_factor(name="fa_inventory_turnover", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="存货周转率(库存管理)", direction=1)
def fa_inventory_turnover(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "存货周转率(次)", asof=kw.get("asof"))


@register_factor(name="fa_receivable_turnover", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="应收账款周转率(回款效率)", direction=1)
def fa_receivable_turnover(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "应收账款周转率(次)", asof=kw.get("asof"))


@register_factor(name="fa_operating_cycle", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="营业周期(天, 越短周转越快, 反向)", direction=-1)
def fa_operating_cycle(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "营业周期(天)", asof=kw.get("asof"))


@register_factor(name="fa_cash_ratio", category="quality", freq="quarterly",
                 data_deps=["financial"],
                 description="现金比率%(现金偿债能力)", direction=1)
def fa_cash_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    return _latest(_frame(data), "现金比率(%)", asof=kw.get("asof"))
