"""
fundamental_v7.py — V7.0 基本面因子扩充（baostock 三表 → 因子）
==============================================================
基于 baostock 季频财务数据构建 15 大类质量/成长/价值因子。
数据源: baostock（云端已装 00.9.30）；缺失时回退 akshare。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor, FREQ_QUARTERLY

log = get_logger("qv6.factor.fundamental_v7")

# ── 数据抓取（带缓存）──────────────────────────────────────

_bs = None


def _baostock():
    global _bs
    if _bs is None:
        import baostock as bs
        _bs = bs
        lg = _bs.login()
        if lg.error_code != "0":
            log.warning(f"baostock 登录失败: {lg.error_msg}")
    return _bs


def _bs_code(code) -> str:
    """6位代码 → baostock 9位格式。"""
    s = str(code).zfill(6)
    return f"sh.{s}" if s.startswith("6") else f"sz.{s}"


def fetch_quarterly_profit(codes: list[str], years: list[int]) -> pd.DataFrame:
    """季度利润表（baostock query_profit_data）。"""
    bs = _baostock()
    if bs is None:
        return pd.DataFrame()
    rows = []
    for code in codes:
        bc = _bs_code(code)
        for year in years:
            for quarter in (1, 2, 3, 4):
                try:
                    rs = bs.query_profit_data(code=bc, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        rows.append(rs.get_row_data())
                except Exception as e:  # noqa: BLE001
                    log.error(f"[fundamental_v7] 操作失败: {e}", exc_info=True)
                    continue
    cols = ["code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin",
            "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])] if rows else cols)
    df["code"] = df["code"].str.replace(r"^(sh|sz)\.", "", regex=True)
    return df


def fetch_quarterly_growth(codes: list[str], years: list[int]) -> pd.DataFrame:
    """成长能力（baostock query_growth_data）。"""
    bs = _baostock()
    if bs is None:
        return pd.DataFrame()
    rows = []
    for code in codes:
        bc = _bs_code(code)
        for year in years:
            for quarter in (1, 2, 3, 4):
                try:
                    rs = bs.query_growth_data(code=bc, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        rows.append(rs.get_row_data())
                except Exception as e:  # noqa: BLE001
                    log.error(f"[fundamental_v7] 操作失败: {e}", exc_info=True)
                    continue
    if not rows:
        return pd.DataFrame()
    cols = ["code", "pubDate", "statDate", "YOYEquity", "YOYAsset", "YOYNI",
            "YOYEPSBasic", "YOYPNI"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    df["code"] = df["code"].str.replace(r"^(sh|sz)\.", "", regex=True)
    return df


def fetch_quarterly_balance(codes: list[str], years: list[int]) -> pd.DataFrame:
    """偿债能力（baostock query_balance_data）。"""
    bs = _baostock()
    if bs is None:
        return pd.DataFrame()
    rows = []
    for code in codes:
        bc = _bs_code(code)
        for year in years:
            for quarter in (1, 2, 3, 4):
                try:
                    rs = bs.query_balance_data(code=bc, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        rows.append(rs.get_row_data())
                except Exception as e:  # noqa: BLE001
                    log.error(f"[fundamental_v7] 操作失败: {e}", exc_info=True)
                    continue
    if not rows:
        return pd.DataFrame()
    cols = ["code", "pubDate", "statDate", "currentRatio", "quickRatio",
            "cashRatio", "YOYLiability", "liabilityToAsset", "assetToEquity"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    df["code"] = df["code"].str.replace(r"^(sh|sz)\.", "", regex=True)
    return df


def fetch_quarterly_cashflow(codes: list[str], years: list[int]) -> pd.DataFrame:
    """现金流（baostock query_cash_flow_data）。"""
    bs = _baostock()
    if bs is None:
        return pd.DataFrame()
    rows = []
    for code in codes:
        bc = _bs_code(code)
        for year in years:
            for quarter in (1, 2, 3, 4):
                try:
                    rs = bs.query_cash_flow_data(code=bc, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        rows.append(rs.get_row_data())
                except Exception as e:  # noqa: BLE001
                    log.error(f"[fundamental_v7] 操作失败: {e}", exc_info=True)
                    continue
    if not rows:
        return pd.DataFrame()
    cols = ["code", "pubDate", "statDate", "CAToAsset", "NCAToAsset",
            "tangibleAssetToAsset", "ebitToInterest", "CFOToOR", "CFOToNP",
            "CFOToGr"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    df["code"] = df["code"].str.replace(r"^(sh|sz)\.", "", regex=True)
    return df


# ── 因子计算（handler 签名: fn(data: dict, **params) -> pd.Series）────

def _norm_code(code) -> str:
    s = str(code).zfill(6)
    if s.startswith(("6", "0", "3")):
        return s
    return s


def _to_series(df: pd.DataFrame, col: str, stat_date: str = "") -> pd.Series:
    """把 baostock 结果转成 Series(index=股票代码)。
    - 同一股票多季度 → 取 statDate 最新一期
    - 空字符串 → NaN
    """
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df.copy()
    d["code"] = d["code"].astype(str).str.replace(r"^(sh|sz)\.", "", regex=True)
    # 数值化
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])
    if d.empty:
        return pd.Series(dtype=float)
    # 去重: 每股票取 statDate 最新
    if "statDate" in d.columns:
        d["statDate"] = pd.to_datetime(d["statDate"], errors="coerce")
        d = d.sort_values("statDate").drop_duplicates(subset=["code"], keep="last")
    else:
        d = d.drop_duplicates(subset=["code"], keep="last")
    out = d.set_index("code")[col]
    out.index = [_norm_code(c) for c in out.index]
    return out.dropna()


@register_factor(name="roe_ttm", category="quality", freq=FREQ_QUARTERLY,
                 universe="all_ashare", data_deps=["fundamental_quarterly"],
                 description="ROE(平均, 单季)", direction=1)
def roe_ttm(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("profit")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return _to_series(df, "roeAvg")


@register_factor(name="gp_margin", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_quarterly"],
                 description="销售毛利率", direction=1)
def gp_margin(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("profit")
    return _to_series(df, "gpMargin") if df is not None else pd.Series(dtype=float)


@register_factor(name="np_margin", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_quarterly"],
                 description="销售净利率", direction=1)
def np_margin(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("profit")
    return _to_series(df, "npMargin") if df is not None else pd.Series(dtype=float)


@register_factor(name="eps_ttm", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_quarterly"],
                 description="每股收益TTM", direction=1)
def eps_ttm(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("profit")
    return _to_series(df, "epsTTM") if df is not None else pd.Series(dtype=float)


@register_factor(name="yoy_ni", category="growth", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_growth"],
                 description="净利润同比增长率", direction=1)
def yoy_ni(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("growth")
    return _to_series(df, "YOYNI") if df is not None else pd.Series(dtype=float)


@register_factor(name="yoy_eps", category="growth", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_growth"],
                 description="每股收益同比增长率", direction=1)
def yoy_eps(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("growth")
    return _to_series(df, "YOYEPSBasic") if df is not None else pd.Series(dtype=float)


@register_factor(name="yoy_equity", category="growth", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_growth"],
                 description="净资产同比增长率", direction=1)
def yoy_equity(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("growth")
    return _to_series(df, "YOYEquity") if df is not None else pd.Series(dtype=float)


@register_factor(name="current_ratio", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_balance"],
                 description="流动比率", direction=1)
def current_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    df = data.get("balance")
    return _to_series(df, "currentRatio") if df is not None else pd.Series(dtype=float)


@register_factor(name="quick_ratio", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_balance"],
                 description="速动比率", direction=1)
def quick_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("balance")
    return _to_series(df, "quickRatio") if df is not None else pd.Series(dtype=float)


@register_factor(name="liability_ratio", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_balance"],
                 description="资产负债率(反向)", direction=-1)
def liability_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("balance")
    return _to_series(df, "liabilityToAsset") if df is not None else pd.Series(dtype=float)


@register_factor(name="asset_to_equity", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_balance"],
                 description="权益乘数(反向, 杠杆风险)", direction=-1)
def asset_to_equity(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("balance")
    return _to_series(df, "assetToEquity") if df is not None else pd.Series(dtype=float)


@register_factor(name="cfo_to_np", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_cashflow"],
                 description="现金流/净利润(盈利含金量)", direction=1)
def cfo_to_np(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("cashflow")
    return _to_series(df, "CFOToNP") if df is not None else pd.Series(dtype=float)


@register_factor(name="cfo_to_or", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_cashflow"],
                 description="经营现金流/营收", direction=1)
def cfo_to_or(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("cashflow")
    return _to_series(df, "CFOToOR") if df is not None else pd.Series(dtype=float)


@register_factor(name="ebit_to_interest", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_cashflow"],
                 description="EBIT/利息(偿债安全)", direction=1)
def ebit_to_interest(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("cashflow")
    return _to_series(df, "ebitToInterest") if df is not None else pd.Series(dtype=float)


@register_factor(name="tangible_asset_ratio", category="quality", freq=FREQ_QUARTERLY,
                 data_deps=["fundamental_cashflow"],
                 description="有形资产/总资产(商誉风险反向)", direction=1)
def tangible_asset_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("cashflow")
    return _to_series(df, "tangibleAssetToAsset") if df is not None else pd.Series(dtype=float)
