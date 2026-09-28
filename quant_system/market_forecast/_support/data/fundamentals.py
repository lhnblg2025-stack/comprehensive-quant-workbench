# -*- coding: utf-8 -*-
"""
fundamentals.py — 基本面财务指标数据模块

通过 akshare 拉取个股财务分析指标（PE/PB/ROE/毛利率/营收增速等）。
所有网络函数均捕获异常并返回空 DataFrame，绝不向上抛出，保证数据层健壮性。

对外主要接口:
    get_financial_indicator(symbol, start_year)  个股财务分析指标
    get_valuation_indicator(symbol)              个股估值指标(PE/PB/PS 等)
    get_fundamentals_batch(symbols, ...)         批量拉取并合并
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover - 环境缺失时兜底
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)

#: 财务分析指标中常用字段名（akshare 返回的中文列名）
KEY_COLUMNS = {
    "roe": "净资产收益率(%)",
    "gross_margin": "毛利率(%)",
    "revenue_growth": "主营业务收入增长率(%)",
    "profit_growth": "净利润增长率(%)",
    "eps": "摊薄每股收益(元)",
}


def get_financial_indicator(
    symbol: str = "600004",
    start_year: str = "2020",
) -> pd.DataFrame:
    """拉取单只股票的财务分析指标（ROE/毛利率/营收增速等）。

    Args:
        symbol: 6 位股票代码，如 "600004"。
        start_year: 起始年份，如 "2020"。

    Returns:
        财务指标 DataFrame；失败时返回空 DataFrame（保留列名）。
    """
    if ak is None:
        return pd.DataFrame(columns=list(KEY_COLUMNS.values()))
    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
        if df is None or df.empty:
            return pd.DataFrame(columns=list(KEY_COLUMNS.values()))
        df = df.copy()
        df.insert(0, "symbol", symbol)
        return df
    except Exception as exc:
        logger.warning("拉取财务指标失败 symbol=%s: %s", symbol, exc)
        return pd.DataFrame(columns=list(KEY_COLUMNS.values()))


def get_valuation_indicator(symbol: str = "600004") -> pd.DataFrame:
    """拉取个股估值指标（PE/PE_TTM/PB 等，乐咕乐股接口）。

    Args:
        symbol: 6 位股票代码。

    Returns:
        估值指标 DataFrame；失败时返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_a_indicator_lg(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "symbol", symbol)
        return df
    except Exception as exc:
        logger.warning("拉取估值指标失败 symbol=%s: %s", symbol, exc)
        return pd.DataFrame()


def get_fundamentals_batch(
    symbols: list,
    start_year: str = "2020",
    interval: float = 0.3,
) -> pd.DataFrame:
    """批量拉取多只股票财务指标，慢速遍历避免触发反爬。

    Args:
        symbols: 股票代码列表。
        start_year: 起始年份。
        interval: 每次请求间隔秒数。

    Returns:
        合并后的财务指标 DataFrame；全部失败时返回空 DataFrame。
    """
    import time

    frames: list = []
    for sym in symbols:
        df = get_financial_indicator(symbol=sym, start_year=start_year)
        if not df.empty:
            frames.append(df)
        time.sleep(interval)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_pe_pb_summary(symbol: str = "600004") -> Optional[dict]:
    """获取最新一期的 PE/PB 摘要（估值指标最后一行的快照）。

    Args:
        symbol: 6 位股票代码。

    Returns:
        dict 形式的估值快照；失败返回 None。
    """
    df = get_valuation_indicator(symbol)
    if df.empty:
        return None
    try:
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "date": str(latest.get("trade_date", "")),
            "pe": latest.get("pe", None),
            "pe_ttm": latest.get("pe_ttm", None),
            "pb": latest.get("pb", None),
            "ps": latest.get("ps", None),
            "total_mv": latest.get("total_mv", None),
        }
    except Exception as exc:
        logger.warning("生成 PE/PB 摘要失败 symbol=%s: %s", symbol, exc)
        return None


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    demo = get_financial_indicator("600004", "2022")
    print("财务指标行数:", len(demo))
    print("估值摘要:", get_pe_pb_summary("600004"))
