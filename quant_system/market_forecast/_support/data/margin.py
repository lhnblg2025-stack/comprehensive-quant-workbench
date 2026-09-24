# -*- coding: utf-8 -*-
"""
margin.py — 两融(融资融券)数据模块

拉取沪深交易所融资融券汇总数据:
    - 上交所: ak.stock_margin_sse   （融资余额/融券余额/融资融券余额）
    - 深交所: ak.stock_margin_szse   （兼容接口，同样容错）
    - 个股:   ak.stock_margin_detail_sse

所有网络函数捕获异常并返回空容器，绝不崩溃。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)

#: 常用列名常量（上交所 stock_margin_sse 返回的中文列）
SSE_COLUMNS = [
    "信用交易日期", "融资买入额", "融资余额", "融券卖出量",
    "融券余量", "融券余额", "融资融券余额",
]


def _default_date_range() -> tuple[str, str]:
    """默认日期范围：当年年初至当前日期（P1-6，不再硬编码 2024）。
    使用 time_utils 当前 CST 时间；交易日精度由交易所接口自行裁剪。
    """
    from quant_system.market_forecast._support.common.time_utils import now_cst
    now = now_cst()
    return f"{now.year}0101", now.strftime("%Y%m%d")


def _empty_margin_df() -> pd.DataFrame:
    """返回带标准列名的空 DataFrame，便于上层统一处理。"""
    return pd.DataFrame(columns=SSE_COLUMNS)


def get_margin_sse(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """拉取上交所融资融券汇总数据。

    Args:
        start_date: 起始日期，格式 YYYYMMDD。缺省为当年年初。
        end_date: 结束日期，格式 YYYYMMDD。缺省为当前日期。

    Returns:
        两融汇总 DataFrame；失败返回空 DataFrame（保留标准列名）。
    """
    if ak is None:
        return _empty_margin_df()
    if not start_date or not end_date:
        start_date, end_date = _default_date_range()
    try:
        df = ak.stock_margin_sse(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return _empty_margin_df()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取上交所两融数据失败 %s~%s: %s", start_date, end_date, exc)
        return _empty_margin_df()


def get_margin_szse(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """拉取深交所融资融券汇总数据（接口变动时自动容错）。

    Args:
        start_date: 起始日期，格式 YYYYMMDD。缺省为当年年初。
        end_date: 结束日期，格式 YYYYMMDD。缺省为当前日期。

    Returns:
        两融汇总 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return _empty_margin_df()
    if not start_date or not end_date:
        start_date, end_date = _default_date_range()
    try:
        df = ak.stock_margin_szse(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return _empty_margin_df()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取深交所两融数据失败 %s~%s: %s", start_date, end_date, exc)
        return _empty_margin_df()


def get_margin_detail_sse(market: str = "60", symbol: str = "600004") -> pd.DataFrame:
    """拉取上交所个股融资融券明细。

    Args:
        market: 市场代码段，如 "60"(沪主板)。
        symbol: 6 位股票代码。

    Returns:
        个股两融明细 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_margin_detail_sse(market=market, symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "symbol", symbol)
        return df
    except Exception as exc:
        logger.warning("拉取个股两融明细失败 %s: %s", symbol, exc)
        return pd.DataFrame()


def get_margin_balance_latest() -> Optional[dict]:
    """获取最新一日上交所融资余额与两融余额快照。

    Returns:
        dict: {"date":..., "fin_balance":..., "margin_balance":...}；失败返回 None。
    """
    df = get_margin_sse()
    if df.empty:
        return None
    try:
        latest = df.iloc[-1]
        return {
            "date": str(latest.get("信用交易日期", "")),
            "fin_balance": latest.get("融资余额", None),   # 融资余额
            "short_balance": latest.get("融券余额", None),  # 融券余额
            "margin_balance": latest.get("融资融券余额", None),
        }
    except Exception as exc:
        logger.warning("生成两融快照失败: %s", exc)
        return None


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    print("上交所两融行数:", len(get_margin_sse("20250101", "20250131")))
    print("两融快照:", get_margin_balance_latest())
