"""
trade_calendar.py — QuantV6 交易日历
ak.tool_trade_date_hist_sina 获取真实交易日，30 天缓存，失败退化工作日。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.fetcher import safe_call

log = get_logger("qv6.calendar")

_cache: set[str] | None = None
_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 30 * 86400  # 30 天


def _load() -> set[str]:
    df = safe_call("tool_trade_date_hist_sina")
    if df is None or len(df) == 0:
        return set()
    col = df.columns[0]
    return {str(d)[:10] for d in df[col].tolist()}


def get_calendar(force: bool = False) -> set[str]:
    """真实交易日集合（'YYYY-MM-DD'）。"""
    global _cache, _cache_ts
    now = time.time()
    with _lock:
        if (force or _cache is None or now - _cache_ts > _TTL):
            days = _load()
            if days:
                _cache = days
                _cache_ts = now
            elif _cache is None:
                _cache = set()  # 加载失败：空日历，退化工作日判断
                _cache_ts = now
        return _cache or set()


def is_trading_day(d: datetime) -> bool:
    """真实日历判断（节假日返回 False）。日历不可用时退化为非周末。"""
    cal = get_calendar()
    s = d.strftime("%Y-%m-%d")
    if cal:
        return s in cal
    return d.weekday() < 5


def next_trading_day(d: datetime) -> datetime:
    for _ in range(15):
        d += timedelta(days=1)
        if is_trading_day(d):
            return d
    return d


def prev_trading_day(d: datetime) -> datetime:
    for _ in range(15):
        d -= timedelta(days=1)
        if is_trading_day(d):
            return d
    return d


def trading_days_between(start: datetime, end: datetime) -> list[datetime]:
    out: list[datetime] = []
    d = start
    while d <= end and len(out) < 800:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out
