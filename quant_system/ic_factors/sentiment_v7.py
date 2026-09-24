"""
sentiment_v7.py — V7.0 情绪舆情因子（新闻/热榜/涨停梯队/市场温度）
====================================================================
数据源: 财联社电报/东财热榜/涨停池（akshare）+ 本地 FinBERT（可选）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.registry import register_factor

log = get_logger("qv6.factor.sentiment_v7")


@register_factor(name="news_sentiment_mean", category="sentiment",
                 data_deps=["news_daily"],
                 description="个股新闻情感均分(FinBERT/词典)", direction=1)
def news_sentiment_mean(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("news")
    if df is None or df.empty or "score" not in df.columns:
        return pd.Series(dtype=float)
    if "code" in df.columns:
        return df.groupby("code")["score"].mean().dropna()
    return pd.Series({"MARKET": df["score"].mean()}).dropna()


@register_factor(name="news_sentiment_std", category="sentiment",
                 data_deps=["news_daily"],
                 description="新闻情感分歧度(反向)", direction=-1)
def news_sentiment_std(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("news")
    if df is None or df.empty or "score" not in df.columns:
        return pd.Series(dtype=float)
    if "code" in df.columns:
        return df.groupby("code")["score"].std().dropna()
    return pd.Series({"MARKET": df["score"].std()}).dropna()


@register_factor(name="hot_rank_momentum", category="sentiment",
                 data_deps=["hot_rank"],
                 description="热股榜排名动量(排名上升=关注度升)", direction=1)
def hot_rank_momentum(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("hot")
    if df is None or df.empty or "code" not in df.columns:
        return pd.Series(dtype=float)
    if "rank" in df.columns and "prev_rank" in df.columns:
        out = df.set_index("code")
        return (pd.to_numeric(out["prev_rank"], errors="coerce")
                - pd.to_numeric(out["rank"], errors="coerce")).dropna()
    return pd.Series(dtype=float)


@register_factor(name="hot_pool_equal_ret", category="sentiment",
                 data_deps=["hot_pool_ret"],
                 description="热股池等权涨跌幅(赚钱效应)", direction=1)
def hot_pool_equal_ret(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("hotret")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": df["ret"].mean()}).dropna()


@register_factor(name="limit_up_count", category="sentiment",
                 data_deps=["limit_pool"],
                 description="涨停家数(情绪热度)", direction=1)
def limit_up_count(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("limit")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": float(len(df))}).dropna()


@register_factor(name="zhaban_rate", category="sentiment",
                 data_deps=["limit_pool"],
                 description="炸板率(涨停打开率, 反向)", direction=-1)
def zhaban_rate(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("limit")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    total = len(df)
    zhaban = int((df.get("炸板次数", pd.Series(dtype=float)) > 0).sum()) if "炸板次数" in df.columns else 0
    if total == 0:
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": zhaban / total}).dropna()


@register_factor(name="max_lianban", category="sentiment",
                 data_deps=["limit_pool"],
                 description="市场最高连板高度", direction=1)
def max_lianban(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("limit")
    if df is None or df.empty or "连板数" not in df.columns:
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": float(pd.to_numeric(df["连板数"], errors="coerce").max())}).dropna()


@register_factor(name="up_down_ratio", category="sentiment",
                 data_deps=["market_breadth"],
                 description="涨跌家数比(市场宽度)", direction=1)
def up_down_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("breadth")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    up = float(df.get("up", 0))
    down = float(df.get("down", 1))
    return pd.Series({"MARKET": up / max(down, 1)}).dropna()


@register_factor(name="pb_below_1_ratio", category="value",
                 data_deps=["valuation_snapshot"],
                 description="破净股占比(市场温度, 高=底部区)", direction=1)
def pb_below_1_ratio(data: dict, **kw) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    df = data.get("val")
    if df is None or df.empty or "pb" not in df.columns:
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": float((df["pb"] < 1).mean())}).dropna()
