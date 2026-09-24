"""
time_utils.py — QuantV6 时间工具
统一 CST (UTC+8) 时区；交易日判断；交易时段判断；窗口计算。
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta

from quant_system.market_forecast._support.common.constants import (
    AFTERNOON_START, CLOSE, COLLECTION_START, CONTINUOUS_START,
    CST, MORNING_END,
)
from quant_system.utils import now_cst


def to_cst(dt: datetime) -> datetime:
    """任意 datetime 转 CST。naive 视为 CST。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=CST)
    return dt.astimezone(CST)


def _is_weekend(d: datetime) -> bool:
    return d.weekday() >= 5


def is_trading_day(d: datetime | None = None, calendar_days: set[str] | None = None) -> bool:
    """
    判断是否为交易日。
    calendar_days: 真实交易日历（'YYYY-MM-DD' 集合），来自 data.trade_calendar；
    为 None 时退化为"非周末"（保守估计）。
    """
    d = to_cst(d or now_cst())
    if _is_weekend(d):
        return False
    if calendar_days is not None:
        return d.strftime("%Y-%m-%d") in calendar_days
    return True


def session_name(t: datetime | None = None) -> str:
    """当前交易时段名称: premarket / morning / lunch / afternoon / closed。"""
    t = to_cst(t or now_cst()).time()
    if COLLECTION_START <= t < CONTINUOUS_START:
        return "premarket"       # 集合竞价（不交易）
    if CONTINUOUS_START <= t <= MORNING_END:
        return "morning"         # 早盘
    if MORNING_END < t < AFTERNOON_START:
        return "lunch"           # 午休
    if AFTERNOON_START <= t <= CLOSE:
        return "afternoon"       # 下午盘
    return "closed"              # 收盘后


def in_trading_session(t: datetime | None = None) -> bool:
    """是否处于可交易时段（连续竞价）。"""
    return session_name(t) in ("morning", "afternoon")


def in_collection(t: datetime | None = None) -> bool:
    """是否处于集合竞价时段。"""
    return session_name(t) == "premarket"


def next_quarter(t: datetime | None = None) -> datetime:
    """下一个 15 分钟窗口起点。"""
    t = to_cst(t or now_cst())
    minute = (t.minute // 15 + 1) * 15
    nxt = t.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
    return nxt


def next_trading_day(d: datetime | None = None, calendar_days: set[str] | None = None) -> datetime:
    """下一个交易日（跳过周末，日历存在时跳过节假日）。"""
    d = to_cst(d or now_cst())
    for _ in range(15):
        d += timedelta(days=1)
        if is_trading_day(d, calendar_days):
            return d
    raise RuntimeError("无法找到下一个交易日")


def prev_trading_day(d: datetime | None = None, calendar_days: set[str] | None = None) -> datetime:
    """上一个交易日。"""
    d = to_cst(d or now_cst())
    for _ in range(15):
        d -= timedelta(days=1)
        if is_trading_day(d, calendar_days):
            return d
    raise RuntimeError("无法找到上一个交易日")


def trading_days_between(start: datetime, end: datetime, calendar_days: set[str] | None = None) -> list[datetime]:
    """两个日期之间的所有交易日（含端点）。"""
    out: list[datetime] = []
    d = to_cst(start)
    end = to_cst(end)
    while d <= end:
        if is_trading_day(d, calendar_days):
            out.append(d)
        d += timedelta(days=1)
        if len(out) > 800:
            break
    return out


def fmt(dt: datetime | None = None) -> str:
    """格式化时间戳 YYYY-MM-DD HH:MM:SS。"""
    return to_cst(dt or now_cst()).strftime("%Y-%m-%d %H:%M:%S")


def fmt_date(dt: datetime | None = None) -> str:
    """格式化日期 YYYY-MM-DD。"""
    return to_cst(dt or now_cst()).strftime("%Y-%m-%d")


def is_month_end(d: datetime | None = None) -> bool:
    d = to_cst(d or now_cst())
    return d.day == calendar.monthrange(d.year, d.month)[1]
