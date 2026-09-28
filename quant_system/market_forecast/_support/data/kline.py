"""
kline.py — QuantV6 K线数据
前复权日线获取，统一列名，窗口对齐，停牌检测。
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from quant_system.market_forecast._support.common.constants import HIST_DAYS, TTL_DAILY, TTL_INTRADAY
from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.common.time_utils import fmt_date, in_trading_session, now_cst
from quant_system.market_forecast._support.common.validators import normalize_symbol
from quant_system.market_forecast._support.data.cache import get, put
from quant_system.market_forecast._support.data.fetcher import call, safe_call

log = get_logger("qv6.kline")

_STD_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]


def get_kline(symbol: str, days: int = HIST_DAYS, adjust: str = "qfq",
              use_cache: bool = True) -> pd.DataFrame:
    """
    获取个股前复权日线。
    返回标准列 DataFrame（索引=日期），失败返回空 DataFrame。
    """
    code = normalize_symbol(symbol)
    start = (now_cst() - timedelta(days=days * 1.6)).strftime("%Y%m%d")
    end = fmt_date(now_cst()).replace("-", "")
    cache_key = f"kline:{code}:{start}:{end}:{adjust}"

    if use_cache:
        # P2-6：盘中缓存 60s（盘口数据实时性要求高），收盘后 1 天
        ttl = TTL_INTRADAY if in_trading_session(now_cst()) else TTL_DAILY
        cached = get("kline", ttl, code, start, end)
        if cached is not None and len(cached):
            return cached

    try:
        df = call("stock_zh_a_hist", symbol=code, period="daily",
                  start_date=start, end_date=end, adjust=adjust)
    except Exception as e:
        log.warning(f"K线获取失败 {code}: {str(e)[:80]}")
        return pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 统一列名（兼容中英文）
    rename = {"date": "日期", "open": "开盘", "close": "收盘",
              "high": "最高", "low": "最低", "volume": "成交量",
              "amount": "成交额", "pct_chg": "涨跌幅"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = [c for c in _STD_COLS if c not in df.columns]
    if missing:
        log.warning(f"{code} K线缺列 {missing}")
        return pd.DataFrame()

    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index()
    # P2-6：astype(float) 对脏数据（空串/None/异常符号）会抛异常崩管道；
    # 改 pd.to_numeric(errors="coerce") 逐列转换，坏值置 NaN 后整行剔除
    df[_STD_COLS] = df[_STD_COLS].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["收盘", "开盘"])
    if len(df) < before:
        log.info(f"{code} K线剔除 {before - len(df)} 行脏数据")
    # 停牌检测：末根K线距今超过3个自然日 → 视为停牌，返回空（P2-8：
    # 原实现只打日志不剔除，上层仍会按旧价撮合；改为返回空由调用方剔除)
    if len(df):
        last_date = df.index[-1]
        if (now_cst().date() - last_date.date()).days > 3:
            log.warning(f"{code} 疑似停牌（末根K线 {last_date.date()}），剔除出池")
            return pd.DataFrame()
    if use_cache:
        put("kline", df, ttl, code, start, end)
    return df


def get_kline_batch(symbols: list[str], days: int = HIST_DAYS) -> dict[str, pd.DataFrame]:
    """批量获取（逐个，失败跳过）。"""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = get_kline(s, days)
        if len(df):
            out[s] = df
    return out


def is_suspended(df: pd.DataFrame, max_gap_days: int = 3) -> bool:
    """停牌判断。"""
    if len(df) == 0:
        return True
    gap = (now_cst().date() - df.index[-1].date()).days
    return gap > max_gap_days


def last_price(df: pd.DataFrame) -> float:
    return float(df["收盘"].iloc[-1]) if len(df) else 0.0


def daily_return(df: pd.DataFrame) -> pd.Series:
    return df["收盘"].pct_change()


def get_index_kline(symbol: str, days: int = 1500) -> pd.DataFrame:
    """
    指数历史日线（新浪源 stock_zh_index_daily）。
    symbol: 'sh000001' / 'sz399006' 带前缀。
    返回列 date/open/high/low/close/volume（索引=date）。
    """
    df = safe_call("stock_zh_index_daily", symbol=symbol)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.rename(columns={"date": "日期"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index()
    if len(df) > days:
        df = df.tail(days)
    return df
