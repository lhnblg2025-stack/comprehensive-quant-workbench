#!/usr/bin/env python3
"""platform.data — 统一数据访问层（骨架收口 P0-2）

所有数据集读取的唯一入口，内部路由到 quant_system.data_store.DataStore。
禁止调用方直接 pd.read_parquet 散落读取。
"""
from __future__ import annotations

import pandas as pd

from quant_system.data_store import DataStore, get_store
from quant_platform import WAREHOUSE

_store: DataStore | None = None


def _ds() -> DataStore:
    global _store
    if _store is None:
        _store = get_store()
    return _store


# ── 元数据 ──────────────────────────────────────────

def list_datasets() -> list[str]:
    """全部已登记数据集名。"""
    return _ds().list_datasets()


def schema_of(dataset: str) -> dict:
    """数据集 schema（列/频率/来源）。"""
    return _ds().schema_of(dataset)


def warehouse_root():
    """数据仓库根目录。"""
    return _ds().warehouse_root()


# ── 核心读取（统一契约）────────────────────────────

def kline(symbol: str, days: int = 250, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """日K线。列: date/open/high/low/close/volume/amount/turnover/..."""
    df = _ds().get(symbol, days=days)
    if df is None or df.empty:
        return pd.DataFrame()
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df


def kline_many(symbols: list[str], days: int = 250) -> dict[str, pd.DataFrame]:
    """多标的K线，返回 {code: df}。"""
    return _ds().get_many(symbols, days=days)


def valuation(symbol: str, days: int = 800) -> pd.DataFrame:
    """估值。列: date/peTTM/pbMRQ/psTTM/pcfNcfTTM"""
    # V3 修复: 原实现引用不存在的 quant_system.factor_extreme_alert → 直接读仓库文件
    try:
        code = str(symbol).zfill(6)
        p = _ds().warehouse_root() / "valuation" / f"{code}.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
        if days and "date" in df.columns:
            df = df.sort_values("date").tail(days)
        return df
    except Exception:
        return pd.DataFrame()


def financial(symbol: str) -> pd.DataFrame:
    """财务摘要。"""
    # V3 审计修复: 原引用不存在的 fundamental_analysis.get_fundamentals（死引用）
    #   → 改用 financial_data.get_financial_summary
    try:
        from quant_system.financial_data import get_financial_summary
        d = get_financial_summary(symbol)
        if not d:
            return pd.DataFrame()
        return pd.DataFrame([d])
    except Exception:
        return pd.DataFrame()


def realtime_latest() -> pd.DataFrame:
    """最近一个盘中快照（realtime_snapshot/）。"""
    snap_dir = WAREHOUSE / "realtime_snapshot"
    if not snap_dir.exists():
        return pd.DataFrame()
    days = sorted(d for d in snap_dir.iterdir() if d.is_dir())
    if not days:
        return pd.DataFrame()
    files = sorted(days[-1].glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


# ── 新鲜度/状态 ────────────────────────────────────

def status() -> dict:
    """数据仓库健康报告。"""
    return _ds().status()
