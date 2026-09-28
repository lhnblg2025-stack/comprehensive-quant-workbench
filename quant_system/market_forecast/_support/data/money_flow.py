# -*- coding: utf-8 -*-
"""
money_flow.py — 资金流数据模块

提供两类资金数据:
    1. 个股资金流（主力/超大单/大单净流入）: ak.stock_individual_fund_flow
    2. 沪深港通(北向)资金汇总: ak.stock_hsgt_fund_flow_summary_em

所有网络函数捕获异常并返回空容器，绝不崩溃。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)


def get_individual_fund_flow(
    stock: str = "600004",
    market: str = "sh",
) -> pd.DataFrame:
    """拉取个股历史资金流（含主力净流入、超大单/大单/中单/小单净额与占比）。

    Args:
        stock: 6 位股票代码。
        market: 市场标识，sh=沪市, sz=深市, bj=北交所。

    Returns:
        资金流 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_individual_fund_flow(stock=stock, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.insert(0, "stock", stock)
        return df
    except Exception as exc:
        logger.warning("拉取个股资金流失败 %s.%s: %s", market, stock, exc)
        return pd.DataFrame()


def get_main_net_inflow(stock: str = "600004", market: str = "sh") -> pd.DataFrame:
    """提取个股"主力净流入-净额"时间序列，便于因子计算。

    Args:
        stock: 6 位股票代码。
        market: 市场标识。

    Returns:
        只含日期与主力净流入的 DataFrame；失败返回空 DataFrame。
    """
    df = get_individual_fund_flow(stock, market)
    if df.empty:
        return pd.DataFrame()
    try:
        cols = ["日期", "主力净流入-净额"]
        if all(c in df.columns for c in cols):
            return df[cols].copy()
        # 兼容列名变化：取第一列为日期、含"主力"且"净额"的列
        main_col = [c for c in df.columns if "主力" in c and "净额" in c]
        if main_col:
            return df[[df.columns[0], main_col[0]]].copy()
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("提取主力净流入失败 %s: %s", stock, exc)
        return pd.DataFrame()


def get_northbound_flow_summary() -> pd.DataFrame:
    """拉取沪深港通(北向)资金流向汇总（沪股通/深股通 成交净买额等）。

    Returns:
        北向资金汇总 DataFrame；失败返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取北向资金汇总失败: %s", exc)
        return pd.DataFrame()


def get_northbound_net_buy_today() -> Optional[dict]:
    """获取今日北向资金净买入快照（沪股通+深股通）。

    Returns:
        dict: {"date":..., "hgt":..., "sgt":..., "total":...}；失败返回 None。
    """
    df = get_northbound_flow_summary()
    if df.empty:
        return None
    try:
        hgt = sgt = None
        for _, row in df.iterrows():
            direction = str(row.get("资金方向", ""))
            val = row.get("成交净买额", None)
            if "沪股通" in direction:
                hgt = val
            elif "深股通" in direction:
                sgt = val
        total = None
        if hgt is not None and sgt is not None:
            try:
                total = float(hgt) + float(sgt)
            except (TypeError, ValueError):
                total = None
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "hgt_net_buy": hgt,
            "sgt_net_buy": sgt,
            "total_net_buy": total,
        }
    except Exception as exc:
        logger.warning("生成北向快照失败: %s", exc)
        return None


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    print("个股主力净流入行数:", len(get_main_net_inflow("600004")))
    print("北向今日快照:", get_northbound_net_buy_today())
