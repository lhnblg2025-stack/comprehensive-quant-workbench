"""
flow_v7.py — V7.0 资金行为与筹码因子（北向/两融/大宗/ETF/主力资金）
====================================================================
数据源: akshare（东财/交易所官网），全部经缓存层（TTL 24h，盘中 15min）。
handler 输入 data dict，由数据层组装（见 data_loader_v7.py）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.flow_v7")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@register_factor(name="north_flow_5d", category="flow",
                 data_deps=["north_daily"],
                 description="北向资金5日净流入(亿)", direction=1)
def north_flow_5d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("north")
    if df is None or df.empty or "net_inflow" not in df.columns:
        return pd.Series(dtype=float)
    # df: index=date, columns=[code...] 或长表
    if "code" in df.columns:
        return df.groupby("code")["net_inflow"].transform(lambda s: s.rolling(5).sum()).drop_duplicates()
    return _num(df.iloc[-1]).dropna()


@register_factor(name="margin_balance_ratio", category="flow",
                 data_deps=["margin_daily"],
                 description="融资余额/流通市值(杠杆筹码占比)", direction=-1)
def margin_balance_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("margin")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "code" in df.columns and "balance" in df.columns and "float_mv" in df.columns:
        out = df.set_index("code")
        return _num(out["balance"]) / _num(out["float_mv"]).clip(lower=1)
    return pd.Series(dtype=float)


@register_factor(name="margin_flow_20d", category="flow",
                 data_deps=["margin_daily"],
                 description="融资余额20日变化率", direction=1)
def margin_flow_20d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("margin")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    g = df.groupby("code")["balance"]
    return (g.last() / g.first() - 1).dropna()


@register_factor(name="block_trade_discount", category="position",
                 data_deps=["block_trade"],
                 description="大宗交易折价率(负=折价, 反向)", direction=-1)
def block_trade_discount(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("block")
    if df is None or df.empty or "premium" not in df.columns:
        return pd.Series(dtype=float)
    return _num(df.set_index("code")["premium"]).dropna()


@register_factor(name="block_trade_net_5d", category="position",
                 data_deps=["block_trade"],
                 description="大宗交易净额5日(亿)", direction=1)
def block_trade_net_5d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("block")
    if df is None or df.empty or "amount" not in df.columns:
        return pd.Series(dtype=float)
    return _num(df.set_index("code")["amount"]).dropna()


@register_factor(name="lhb_net_buy", category="flow",
                 data_deps=["lhb_daily"],
                 description="龙虎榜净买入额(万)", direction=1)
def lhb_net_buy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("lhb")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "code" in df.columns and "net" in df.columns:
        return _num(df.set_index("code")["net"]).dropna()
    return pd.Series(dtype=float)


@register_factor(name="lhb_institution_buy", category="flow",
                 data_deps=["lhb_daily"],
                 description="龙虎榜机构净买入(万)", direction=1)
def lhb_institution_buy(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("lhb")
    if df is None or df.empty or "inst_net" not in df.columns:
        return pd.Series(dtype=float)
    return _num(df.set_index("code")["inst_net"]).dropna()


@register_factor(name="holder_count_change", category="position",
                 data_deps=["holders_count"],
                 description="股东户数环比变化(反向: 户数降=筹码集中)", direction=-1)
def holder_count_change(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("holders")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    g = df.groupby("code")["holder_count"]
    return (g.last() / g.first() - 1).dropna()


@register_factor(name="pledge_ratio", category="position",
                 data_deps=["pledge"],
                 description="股权质押比例(反向: 高风险)", direction=-1)
def pledge_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("pledge")
    if df is None or df.empty or "ratio" not in df.columns:
        return pd.Series(dtype=float)
    return _num(df.set_index("code")["ratio"]).dropna()


@register_factor(name="unlock_pressure_30d", category="event",
                 data_deps=["restricted_shares"],
                 description="未来30天解禁市值/流通市值(反向)", direction=-1)
def unlock_pressure_30d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("unlock")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "ratio" in df.columns:
        return _num(df.set_index("code")["ratio"]).dropna()
    if "amount" in df.columns and "float_mv" in df.columns:
        out = df.set_index("code")
        return (_num(out["amount"]) / _num(out["float_mv"]).clip(lower=1)).dropna()
    return pd.Series(dtype=float)


@register_factor(name="etf_flow_20d", category="flow",
                 data_deps=["etf_flow"],
                 description="宽基ETF份额20日变化(市场增量资金)", direction=1)
def etf_flow_20d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("etf")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    # 市场级指标 → 单值广播
    val = df["shares"].iloc[-1] / df["shares"].iloc[-21] - 1 if len(df) > 21 else np.nan
    return pd.Series({"MARKET": val}).dropna()


@register_factor(name="main_flow_5d", category="flow",
                 data_deps=["main_flow"],
                 description="主力资金5日净流入(亿)", direction=1)
def main_flow_5d(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("mainflow")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    g = df.groupby("code")["net"].transform(lambda s: s.rolling(5).sum())
    return g.drop_duplicates().dropna()


@register_factor(name="industry_rotation_speed", category="industry",
                 data_deps=["industry_daily"],
                 description="行业排名变动速度(反向: 高=轮动快, 不稳定)", direction=-1)
def industry_rotation_speed(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("industry")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    # df: index=date, columns=行业涨幅
    ranks = df.rank(axis=1)
    speed = ranks.diff().abs().mean().mean() if len(ranks) > 1 else np.nan
    return pd.Series({"MARKET": speed}).dropna()
