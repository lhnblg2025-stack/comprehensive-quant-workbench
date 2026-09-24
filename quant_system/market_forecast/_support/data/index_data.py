"""
index_data.py — QuantV6 指数行情
新浪实时指数（8大指数）+ 历史日线。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.constants import INDEX_MAP
from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.fetcher import safe_call
from quant_system.market_forecast._support.data.kline import get_index_kline

log = get_logger("qv6.index")

_SINA_PREFIX = {
    "000001": "sh000001", "399001": "sz399001", "399006": "sz399006",
    "000688": "sh000688", "000300": "sh000300", "000905": "sh000905",
    "000852": "sh000852", "932000": "sz932000",
}


def get_index_spot() -> dict[str, dict]:
    """
    新浪指数实时行情。
    返回 {名称: {price, change_pct}}，失败返回空 dict。
    """
    spot = safe_call("stock_zh_index_spot_sina")
    out: dict[str, dict] = {}
    if spot is None or len(spot) == 0:
        return out
    spot["代码"] = spot["代码"].astype(str)
    for name, code in INDEX_MAP.items():
        r = spot[spot["代码"].str.endswith(code)]
        if not r.empty:
            try:
                out[name] = {
                    "price": float(r.iloc[0].get("最新价", 0)),
                    "change_pct": float(r.iloc[0].get("涨跌幅", 0)),
                }
            except Exception as e:
                log.error(f"[index_data] 操作失败: {e}", exc_info=True)
                continue
    return out


def get_index_history(symbol: str, days: int = 1500) -> pd.DataFrame:
    """指数历史日线（新浪）。symbol 带前缀如 sh000001。"""
    return get_index_kline(symbol, days)


def get_index_history_by_code(code: str, days: int = 1500) -> pd.DataFrame:
    """按 6 位代码取历史（自动补前缀）。"""
    sym = _SINA_PREFIX.get(code)
    if not sym:
        return pd.DataFrame()
    return get_index_history(sym, days)


def get_csi300_history(days: int = 1500) -> pd.DataFrame:
    return get_index_history_by_code("000300", days)


def get_cyb_history(days: int = 1500) -> pd.DataFrame:
    """创业板指历史（预测层主要标的）。"""
    return get_index_history_by_code("399006", days)


def get_sh_history(days: int = 1500) -> pd.DataFrame:
    """上证指数历史。"""
    return get_index_history_by_code("000001", days)


def all_index_history(days: int = 1500) -> dict[str, pd.DataFrame]:
    """全部 8 大指数历史。"""
    out: dict[str, pd.DataFrame] = {}
    for name, code in INDEX_MAP.items():
        df = get_index_history_by_code(code, days)
        if len(df):
            out[name] = df
    return out
