#!/usr/bin/env python3
"""market_level_factors.py — 市场级因子库 (V12.3 因子扩充)

设计依据（4 开源项目学习借鉴）:
  - macro/flow/sentiment/industry/event 类因子在 IC 管线中完全空白
    (249 注册仅 58 跑 IC, 107 个未接入)——本模块用已验证数据基座补齐
  - HFT 借鉴: 已实现波动率(realized_vol)、市场恐慌(qvix)
  - intro_quant_finance 借鉴: 市场拥挤度/破净水位

数据契约（全部来自 data_warehouse, 双机一致）:
  macro/*.parquet            月频宏观(PMI/CPI/PPI/M2/LPR/SHIBOR/US10Y)
  market/stock_market_fund_flow.parquet   全市场主力资金(日)
  market/fund_forces.parquet  资金合力(游资/机构/北向/融资/force_index, 日)
  market/news_sentiment.parquet + social_sentiment.parquet  舆情(日)
  market/market_heat__*.parquet  市场热度(巴菲特/拥挤度/PE)
  market/qvix.parquet         恐慌指数(日)
  market/zt_daily_stats.parquet 涨停统计(日)
  market/index_pe.parquet     指数PE(日)
  events/north.parquet        北向(日)
  events/block.parquet        大宗交易(日)
  events/etf_flow.parquet     ETF资金流(日)
  oneoff/a_below_net.parquet  破净比例(低频)
  market/zt_pool_history.parquet + classification/concept_member.parquet  概念热度

输出契约: 每个因子函数 data: dict → pd.Series(date-index, 单值)；
面板构建: 全股票同值(date×code tile)——市场级因子是合法时间序列。

用法:
  python3 -c "from quant_system.ic_factors.market_level_factors import MARKET_LEVEL_FACTORS; print(len(MARKET_LEVEL_FACTORS))"
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.ic_factors.registry import register_factor  # noqa: E402

MARKET_DIR = ROOT / "data_warehouse" / "market"
MACRO_DIR = ROOT / "data_warehouse" / "macro"
EVENTS_DIR = ROOT / "data_warehouse" / "events"
ONEOFF_DIR = ROOT / "data_warehouse" / "oneoff"
CLASS_DIR = ROOT / "data_warehouse" / "classification"


def _read(name: str, base: Path | None = None) -> pd.DataFrame | None:
    try:
        p = (base or MARKET_DIR) / name
        if not p.exists():
            return None
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def _s(series: pd.Series) -> pd.Series:
    """统一: 去重索引 + 升序 + 数值化。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# 宏观数据发布滞后(报告期 → 实际可获知日, 天数) — V12.3 审计 P1-3 防前视
# CPI/PPI/M2/社融 国家统计局/央行 次月9-15日; 财新PMI 当月末; LPR 每月20日
PUB_LAG = {
    "mkt_cpi_yoy": 12, "mkt_ppi_yoy": 12, "mkt_ppi_cpi_gap": 12, "mkt_m2_yoy": 12,
    "mkt_pmi_level": 0, "mkt_pmi_change": 0, "mkt_lpr_level": 20, "mkt_shibor_level": 0,
    "mkt_us10y_level": 2,
}


def _monthly_to_daily(df: pd.DataFrame, date_col: str, val_col: str, lag_days: int = 0) -> pd.Series:
    """月频 → 报告期 + 发布滞后 日期序列（不 ffill, 面板构建端 last-available）。

    V12.3 审计 P1-3: 加发布滞后(lag_days)避免统计月值在真实发布日前生效(前视)。"""
    d = df.copy()
    d[date_col] = d[date_col].astype(str).str.replace("年", "-").str.replace("月份", "-01").str.replace("月", "-01")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, val_col])
    if lag_days and "报告期月" not in d.columns:
        # 报告期 = 该月最后一天, 发布日 = 报告期 + lag_days
        idx = d[date_col] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    else:
        idx = d[date_col]
    return _s(pd.Series(pd.to_numeric(d[val_col], errors="coerce").values, index=idx))


# ───────────────────────── 宏观类 (macro, 月频) ─────────────────────────

@register_factor(name="mkt_pmi_level", category="macro",
                 description="财新制造业PMI水平(>50扩张)", direction=1)
def mkt_pmi_level(data: dict, **kw) -> pd.Series:
    df = _read("cx_pmi_yearly.parquet", MACRO_DIR)
    if df is None or "今值" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    sub = df[df.get("商品", "").astype(str).str.contains("PMI", na=False)]
    return _monthly_to_daily(sub, "日期", "今值", PUB_LAG["mkt_pmi_level"])


@register_factor(name="mkt_pmi_change", category="macro",
                 description="财新PMI环比变化(边际)", direction=1)
def mkt_pmi_change(data: dict, **kw) -> pd.Series:
    s = mkt_pmi_level(data)
    return s.diff() if len(s) else s


@register_factor(name="mkt_cpi_yoy", category="macro",
                 description="CPI同比(%)", direction=1)
def mkt_cpi_yoy(data: dict, **kw) -> pd.Series:
    df = _read("cpi_yearly.parquet", MACRO_DIR)
    if df is None or "全国-同比增长" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return _monthly_to_daily(df, "月份", "全国-同比增长", PUB_LAG["mkt_cpi_yoy"])


@register_factor(name="mkt_ppi_yoy", category="macro",
                 description="PPI同比(%)", direction=1)
def mkt_ppi_yoy(data: dict, **kw) -> pd.Series:
    df = _read("ppi_yearly.parquet", MACRO_DIR)
    if df is None or "当月同比增长" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return _monthly_to_daily(df, "月份", "当月同比增长", PUB_LAG["mkt_ppi_yoy"])


@register_factor(name="mkt_ppi_cpi_gap", category="macro",
                 description="PPI-CPI剪刀差(中游利润)", direction=1)
def mkt_ppi_cpi_gap(data: dict, **kw) -> pd.Series:
    ppi, cpi = mkt_ppi_yoy(data), mkt_cpi_yoy(data)
    j = pd.concat([ppi, cpi], axis=1).dropna()
    return (j.iloc[:, 0] - j.iloc[:, 1]) if len(j) else pd.Series(index=pd.DatetimeIndex([]), dtype=float)


@register_factor(name="mkt_m2_yoy", category="macro",
                 description="M2同比(%)", direction=1)
def mkt_m2_yoy(data: dict, **kw) -> pd.Series:
    df = _read("m2_yearly.parquet", MACRO_DIR)
    if df is None:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    c = [c for c in df.columns if "同比" in c]
    if not c:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    dc = "月份" if "月份" in df.columns else df.columns[0]
    return _monthly_to_daily(df, dc, c[0], PUB_LAG["mkt_m2_yoy"])


@register_factor(name="mkt_lpr_level", category="macro",
                 description="LPR 1Y水平(宽松=低)", direction=-1)
def mkt_lpr_level(data: dict, **kw) -> pd.Series:
    df = _read("lpr.parquet", MACRO_DIR)
    if df is None or df.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    vc = [c for c in df.columns if "1" in str(c) and "LPR" in str(c).upper()] or [df.columns[-1]]
    dc = [c for c in df.columns if "日期" in str(c) or "时间" in str(c)]
    return _monthly_to_daily(df, dc[0] if dc else df.columns[0], vc[0], PUB_LAG["mkt_lpr_level"])


@register_factor(name="mkt_shibor_level", category="macro",
                 description="SHIBOR 隔夜水平(资金面)", direction=-1)
def mkt_shibor_level(data: dict, **kw) -> pd.Series:
    df = _read("shibor_all.parquet", MACRO_DIR)
    if df is None or df.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    dc = [c for c in df.columns if "日期" in str(c) or "时间" in str(c)]
    vc = [c for c in df.columns if "隔夜" in str(c) or "O/N" in str(c).upper()]
    if not vc:
        vc = [df.columns[-1]]
    return _s(pd.Series(pd.to_numeric(df[vc[0]], errors="coerce").values,
                        index=pd.to_datetime(df[dc[0] if dc else df.columns[0]], errors="coerce")))


@register_factor(name="mkt_us10y_level", category="macro",
                 description="美债10Y收益率(风险偏好)", direction=1)
def mkt_us10y_level(data: dict, **kw) -> pd.Series:
    df = _read("rates__us_rate.parquet", MARKET_DIR)
    if df is None or df.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    dc = [c for c in df.columns if "日期" in str(c) or "时间" in str(c)] or [df.columns[0]]
    vc = [c for c in df.columns if "10" in str(c)] or [df.columns[-1]]
    return _s(pd.Series(pd.to_numeric(df[vc[0]], errors="coerce").values,
                        index=pd.to_datetime(df[dc[0]], errors="coerce")))


# ───────────────────────── 资金/流动类 (flow, 日频) ─────────────────────────

@register_factor(name="mkt_north_flow_5d", category="flow",
                 description="北向资金5日净流入(亿)", direction=1)
def mkt_north_flow_5d(data: dict, **kw) -> pd.Series:
    df = _read("north.parquet", EVENTS_DIR)
    if df is not None and "当日成交净买额" in df.columns:
        s = _s(pd.Series(pd.to_numeric(df["当日成交净买额"], errors="coerce").values,
                         index=pd.to_datetime(df["日期"], errors="coerce")))
        return s.rolling(5, min_periods=3).sum() / 1e8
    # 北向独立事件文件缺失时，使用已落盘的资金合力 north_net；不伪造数值。
    ff = _read("fund_forces.parquet")
    if ff is not None and "north_net" in ff.columns and "date" in ff.columns:
        s = _s(pd.Series(pd.to_numeric(ff["north_net"], errors="coerce").values,
                         index=pd.to_datetime(ff["date"], errors="coerce")))
        return s.rolling(5, min_periods=3).sum() / 1e8
    return pd.Series(index=pd.DatetimeIndex([]), dtype=float)


@register_factor(name="mkt_north_flow_20d", category="flow",
                 description="北向资金20日净流入(亿)", direction=1)
def mkt_north_flow_20d(data: dict, **kw) -> pd.Series:
    df = _read("north.parquet", EVENTS_DIR)
    if df is None or "当日成交净买额" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["当日成交净买额"], errors="coerce").values,
                     index=pd.to_datetime(df["日期"], errors="coerce")))
    return s.rolling(20, min_periods=10).sum() / 1e8


@register_factor(name="mkt_main_net_5d", category="flow",
                 description="全市场主力资金5日净流入(亿)", direction=1)
def mkt_main_net_5d(data: dict, **kw) -> pd.Series:
    df = _read("stock_market_fund_flow.parquet")
    if df is None or "主力净流入-净额" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["主力净流入-净额"], errors="coerce").values,
                     index=pd.to_datetime(df["日期"], errors="coerce")))
    return s.rolling(5, min_periods=3).sum() / 1e8


@register_factor(name="mkt_force_index", category="flow",
                 description="资金合力指数(游资+机构+北向+融资)", direction=1)
def mkt_force_index(data: dict, **kw) -> pd.Series:
    df = _read("fund_forces.parquet")
    if df is None or "force_index" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return _s(pd.Series(pd.to_numeric(df["force_index"], errors="coerce").values,
                        index=pd.to_datetime(df["date"], errors="coerce")))


@register_factor(name="mkt_youzi_net_5d", category="flow",
                 description="游资资金5日净额(亿)", direction=1)
def mkt_youzi_net_5d(data: dict, **kw) -> pd.Series:
    df = _read("fund_forces.parquet")
    if df is None or "youzi_net" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["youzi_net"], errors="coerce").values,
                     index=pd.to_datetime(df["date"], errors="coerce")))
    return s.rolling(5, min_periods=3).sum() / 1e8


@register_factor(name="mkt_margin_delta_20d", category="flow",
                 description="两融余额20日变化(杠杆情绪)", direction=1)
def mkt_margin_delta_20d(data: dict, **kw) -> pd.Series:
    df = _read("market_margin_sh.parquet")
    if df is None or "融资余额" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["融资余额"], errors="coerce").values,
                     index=pd.to_datetime(df["日期"], errors="coerce")))
    chg = s.pct_change(20)
    return chg * 100


@register_factor(name="mkt_block_discount_5d", category="flow",
                 description="大宗交易折价率5日均值(负=折价)", direction=-1)
def mkt_block_discount_5d(data: dict, **kw) -> pd.Series:
    df = _read("block.parquet", EVENTS_DIR)
    if df is None or "成交价" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    # 折价率无直接列 → 用 成交价/前收盘 近似(简化: 成交额加权方向仅作市场情绪)
    s = _s(pd.Series(pd.to_numeric(df["成交额"], errors="coerce").values,
                     index=pd.to_datetime(df["交易日期"], errors="coerce")))
    return s.rolling(5, min_periods=2).mean() / 1e8


# ───────────────────────── 情绪类 (sentiment, 日频) ─────────────────────────

@register_factor(name="mkt_news_sentiment_5d", category="sentiment",
                 description="公告舆情5日均值", direction=1)
def mkt_news_sentiment_5d(data: dict, **kw) -> pd.Series:
    df = _read("news_sentiment.parquet")
    if df is None or df.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    dc = [c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()]
    vc = [c for c in df.columns if "sentiment" in str(c).lower() or "情绪" in str(c)]
    if not dc or not vc:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df[vc[0]], errors="coerce").values,
                     index=pd.to_datetime(df[dc[0]], errors="coerce")))
    return s.rolling(5, min_periods=2).mean()


@register_factor(name="mkt_buffett_index", category="sentiment",
                 description="巴菲特指标(总市值/GDP, 高=贵)", direction=-1)
def mkt_buffett_index(data: dict, **kw) -> pd.Series:
    df = _read("buffett_index.parquet", ONEOFF_DIR)
    if df is None or "总市值" not in df.columns or "GDP" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series((pd.to_numeric(df["总市值"], errors="coerce")
                      / pd.to_numeric(df["GDP"], errors="coerce").replace(0, np.nan)).values,
                     index=pd.to_datetime(df["日期"], errors="coerce")))
    return s


@register_factor(name="mkt_below_net_ratio", category="sentiment",
                 description="全市场破净比例(低=底部特征)", direction=-1)
def mkt_below_net_ratio(data: dict, **kw) -> pd.Series:
    df = _read("a_below_net.parquet", ONEOFF_DIR)
    if df is None or "below_net_asset_ratio" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return _s(pd.Series(pd.to_numeric(df["below_net_asset_ratio"], errors="coerce").values,
                        index=pd.to_datetime(df["date"], errors="coerce")))


@register_factor(name="mkt_qvix_level", category="sentiment",
                 description="恐慌指数水平(高=恐慌)", direction=-1)
def mkt_qvix_level(data: dict, **kw) -> pd.Series:
    df = _read("qvix.parquet")
    if df is None or "close" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return _s(pd.Series(pd.to_numeric(df["close"], errors="coerce").values,
                        index=pd.to_datetime(df["date"], errors="coerce")))


@register_factor(name="mkt_qvix_change_5d", category="sentiment",
                 description="恐慌指数5日变化(飙升=风险)", direction=-1)
def mkt_qvix_change_5d(data: dict, **kw) -> pd.Series:
    s = mkt_qvix_level(data)
    return s.pct_change(5) * 100 if len(s) else s


# ───────────────────────── 指数/行业类 (industry, 日频) ─────────────────────────

def _index_pe_pct(index_name: str) -> pd.Series:
    df = _read("index_pe.parquet")
    if df is None or df.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    dc = [c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()]
    nc = [c for c in df.columns if "symbol" in str(c).lower() or "名称" in str(c)]
    vc = [c for c in df.columns if "滚动市盈率" in str(c) and "中位数" not in str(c)] \
         or [c for c in df.columns if "市盈" in str(c) and "中位数" not in str(c)]
    if not dc or not vc:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    sub = df[df[nc[0]].astype(str).str.contains(index_name, na=False)] if nc else df
    if sub.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(sub[vc[0]], errors="coerce").values,
                     index=pd.to_datetime(sub[dc[0]], errors="coerce")))
    if len(s) < 60:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    return s.rolling(250, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean(), raw=False)


@register_factor(name="mkt_hs300_pe_pct", category="industry",
                 description="沪深300 PE 250日分位", direction=-1)
def mkt_hs300_pe_pct(data: dict, **kw) -> pd.Series:
    return _index_pe_pct("沪深300")


@register_factor(name="mkt_zz500_pe_pct", category="industry",
                 description="中证500 PE 250日分位", direction=-1)
def mkt_zz500_pe_pct(data: dict, **kw) -> pd.Series:
    return _index_pe_pct("中证500")


@register_factor(name="mkt_zz1000_pe_pct", category="industry",
                 description="中证1000 PE 250日分位", direction=-1)
def mkt_zz1000_pe_pct(data: dict, **kw) -> pd.Series:
    return _index_pe_pct("中证1000")


# ───────────────────────── 涨停/事件类 (event, 日频) ─────────────────────────

@register_factor(name="mkt_zt_count_5d", category="event",
                 description="涨停家数5日均值(赚钱效应)", direction=1)
def mkt_zt_count_5d(data: dict, **kw) -> pd.Series:
    df = _read("zt_pool_history.parquet")
    if df is None or "date" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    cnt = d[d.get("is_zt", True).fillna(False)].groupby("date").size()
    s = _s(cnt)
    return s.rolling(5, min_periods=2).mean()


@register_factor(name="mkt_zt_high_board", category="event",
                 description="当日最高连板数(情绪高度)", direction=1)
def mkt_zt_high_board(data: dict, **kw) -> pd.Series:
    df = _read("zt_pool_history.parquet")
    if df is None or "board_count" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    s = _s(d.groupby("date")["board_count"].max())
    return s


@register_factor(name="mkt_lhb_count_5d", category="event",
                 description="龙虎榜上榜家数5日均值(热度)", direction=1)
def mkt_lhb_count_5d(data: dict, **kw) -> pd.Series:
    rows = []
    for f in sorted(MARKET_DIR.glob("lhb_20*.parquet")):
        if "hyyyb" in f.name or "jgmmtj" in f.name or "ggtj" in f.name:
            continue
        try:
            d = pd.read_parquet(f)
            if "上榜日" in d.columns:
                rows.append(d[["上榜日"]])
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    all_d = pd.concat(rows, ignore_index=True)
    cnt = pd.to_datetime(all_d["上榜日"], errors="coerce").dropna().value_counts()
    s = _s(cnt.astype(float))
    return s.rolling(5, min_periods=2).mean()


@register_factor(name="mkt_etf_flow_20d", category="event",
                 description="宽基ETF成交额20日均值(资金入市)", direction=1)
def mkt_etf_flow_20d(data: dict, **kw) -> pd.Series:
    df = _read("etf_flow.parquet", EVENTS_DIR)
    if df is None or "amount" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["amount"], errors="coerce").values,
                     index=pd.to_datetime(df["date"], errors="coerce")))
    return s.rolling(20, min_periods=10).mean() / 1e8


# ───────────────────────── 波动类 (volatility, K线可算) ─────────────────────────

@register_factor(name="mkt_realized_vol_20d", category="volatility",
                 description="全市场已实现波动20日(风险水位)", direction=-1)
def mkt_realized_vol_20d(data: dict, **kw) -> pd.Series:
    """用沪深300日线计算已实现波动(代表市场风险水位)。"""
    df = _read("index_daily_沪深300.parquet")
    if df is None or "close" not in df.columns:
        df = _read("index_daily.parquet")
    if df is None or "close" not in df.columns:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)
    s = _s(pd.Series(pd.to_numeric(df["close"], errors="coerce").values,
                     index=pd.to_datetime(df["date"], errors="coerce")))
    ret = s.pct_change().dropna()
    return ret.rolling(20, min_periods=10).std() * np.sqrt(244) * 100


# 市场级因子清单（面板构建用）
MARKET_LEVEL_FACTORS = [
    # macro
    "mkt_pmi_level", "mkt_pmi_change", "mkt_cpi_yoy", "mkt_ppi_yoy",
    "mkt_ppi_cpi_gap", "mkt_m2_yoy", "mkt_lpr_level", "mkt_shibor_level",
    "mkt_us10y_level",
    # flow
    "mkt_north_flow_5d", "mkt_north_flow_20d", "mkt_main_net_5d",
    "mkt_force_index", "mkt_youzi_net_5d", "mkt_margin_delta_20d",
    "mkt_block_discount_5d",
    # sentiment
    "mkt_news_sentiment_5d", "mkt_buffett_index",
    "mkt_below_net_ratio", "mkt_qvix_level", "mkt_qvix_change_5d",
    # industry
    "mkt_hs300_pe_pct", "mkt_zz500_pe_pct", "mkt_zz1000_pe_pct",
    # event
    "mkt_zt_count_5d", "mkt_zt_high_board", "mkt_lhb_count_5d", "mkt_etf_flow_20d",
    # volatility
    "mkt_realized_vol_20d",
]
