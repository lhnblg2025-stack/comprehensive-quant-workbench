# -*- coding: utf-8 -*-
"""
limit_pool.py — 涨跌停池数据模块

拉取每日涨停/跌停/炸板股票池（东方财富接口）:
    - ak.stock_zt_pool_em        涨停股池（含连板数/封板资金/所属行业）
    - ak.stock_zt_pool_dtgc_em   跌停股池（跌停股池）
    - ak.stock_zt_pool_zbgc_em   炸板股池（炸板股池）

所有网络函数捕获异常并返回空容器（DataFrame 或 dict），绝不崩溃。
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)


def _normalize_date(date) -> str:
    """将输入日期统一为 YYYYMMDD 字符串。"""
    if isinstance(date, datetime):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "")


def get_limit_up_pool(date) -> pd.DataFrame:
    """拉取指定日期涨停股池（含连板数、封板资金等）。

    Args:
        date: 日期，支持 "20250106" 或 "2025-01-06" 或 datetime。

    Returns:
        涨停池 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    date_str = _normalize_date(date)
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "date", date_str)
        return df
    except Exception as exc:
        logger.warning("拉取涨停池失败 %s: %s", date_str, exc)
        return pd.DataFrame()


def get_limit_down_pool(date) -> pd.DataFrame:
    """拉取指定日期跌停股池。

    Args:
        date: 日期。

    Returns:
        跌停池 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    date_str = _normalize_date(date)
    try:
        df = ak.stock_zt_pool_dtgc_em(date=date_str)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "date", date_str)
        return df
    except Exception as exc:
        logger.warning("拉取跌停池失败 %s: %s", date_str, exc)
        return pd.DataFrame()


def get_zhaban_pool(date) -> pd.DataFrame:
    """拉取指定日期炸板股池（盘中曾涨停但收盘未封住）。

    Args:
        date: 日期。

    Returns:
        炸板池 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    date_str = _normalize_date(date)
    try:
        df = ak.stock_zt_pool_zbgc_em(date=date_str)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "date", date_str)
        return df
    except Exception as exc:
        logger.warning("拉取炸板池失败 %s: %s", date_str, exc)
        return pd.DataFrame()


def get_limit_pool_summary(date) -> dict:
    """获取某日涨跌停家数汇总（涨停/跌停/炸板家数及涨停列表代码）。

    Args:
        date: 日期。

    Returns:
        dict: {"date":..., "limit_up_count":..., "limit_down_count":...,
               "zhaban_count":..., "limit_up_codes":[...], ...}
        全部失败时返回含 0 家的空 dict。
    """
    date_str = _normalize_date(date)
    result = {
        "date": date_str,
        "limit_up_count": 0,
        "limit_down_count": 0,
        "zhaban_count": 0,
        "limit_up_codes": [],
        "limit_down_codes": [],
        "zhaban_codes": [],
    }
    up = get_limit_up_pool(date_str)
    down = get_limit_down_pool(date_str)
    zhaban = get_zhaban_pool(date_str)

    if not up.empty:
        result["limit_up_count"] = int(len(up))
        result["limit_up_codes"] = list(up["代码"].astype(str)) if "代码" in up.columns else []
    if not down.empty:
        result["limit_down_count"] = int(len(down))
        result["limit_down_codes"] = list(down["代码"].astype(str)) if "代码" in down.columns else []
    if not zhaban.empty:
        result["zhaban_count"] = int(len(zhaban))
        result["zhaban_codes"] = list(zhaban["代码"].astype(str)) if "代码" in zhaban.columns else []
    # W2.5 修复：三池全空 = 数据源失败(真"0家"几乎不存在)，显式标记 ok=False
    result["ok"] = bool(len(up) or len(down) or len(zhaban))
    if not result["ok"]:
        result["error"] = "limit_up/down/zhaban 三池均为空(疑似数据源失败)"
    return result


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    print("涨跌停汇总:", get_limit_pool_summary("20250106"))
