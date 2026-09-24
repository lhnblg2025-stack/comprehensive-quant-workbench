"""
统一市场时钟 — 交易日历 + 交易时段合并模块

功能:
  1. 交易日历: 从 akshare 获取真实A股交易日历（含节假日），带缓存与本地降级
  2. 交易时段: 复用 market_rules 的竞价/连续竞价阶段定义
  3. 统一接口: is_trading_day / is_trading_session / next_trading_day / prev_trading_day

设计目标:
  - data_quality._get_trade_calendar 与此处日历合并，全系统唯一交易日历源
  - 回测/实盘/盘中监控统一使用本模块判断"今天是否交易日/当前是否可交易"

用法:
  from market_clock import is_trading_day, is_trading_session, next_trading_day
  python3 -m quant_system.market_clock --self-test
"""

from __future__ import annotations
import logging

import threading
import time as _time
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    from quant_system.market_rules import (
        current_phase,
        is_in_call_auction,
        is_in_continuous_auction,
        is_trading_session as _rules_trading_session,
    )
except Exception:
    from market_rules import (  # type: ignore[no-redef]
        current_phase,
        is_in_call_auction,
        is_in_continuous_auction,
        is_trading_session as _rules_trading_session,
    )

CST = timezone(timedelta(hours=8))

# 交易日历缓存（模块级单例，带锁）
_TRADE_CALENDAR_CACHE: Optional[set] = None
_TRADE_CALENDAR_LOCK = threading.Lock()
_CALENDAR_SOURCE = "akshare.tool_trade_date_hist_sina"
# P2-Q4-fix(M419): 获取失败后缓存空集但记录失败时间戳；超过该 TTL 后允许重试
_CALENDAR_RETRY_TTL = 3600.0  # 秒
_CALENDAR_LAST_FAIL_TS = 0.0
# 本地日历文件缓存（2026-08-07: 本机虚拟机网络受限，akshare 东财/新浪接口不通，
# 从腾讯云服务器导出的真实交易日历存到 quant_system/data/trade_calendar.csv，
# 优先读文件，网络请求失败不再阻塞）
import os as _os
_CALENDAR_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "trade_calendar.csv")
_CAL_MAX_DATE: Optional[date] = None  # V11: 本地日历最大日期（越界回退工作日粗筛）


def _load_calendar_from_file() -> Optional[set]:
    """从本地 CSV 加载真实交易日历（不联网）。失败返回 None。"""
    try:
        if not _os.path.exists(_CALENDAR_FILE):
            return None
        df = pd.read_csv(_CALENDAR_FILE, dtype=str)
        col = df.columns[0]
        raw = [str(d).strip()[:10] for d in df[col].tolist() if str(d).strip()]
        dates = set(raw)
        if raw:
            try:
                global _CAL_MAX_DATE
                _CAL_MAX_DATE = max(pd.to_datetime(raw)).date()
            except Exception as e:
                logging.getLogger(__name__).error(f"[market_clock] 操作失败: {e}", exc_info=True)
        return dates if dates else None
    except Exception:
        return None


def get_trade_calendar() -> set:
    """获取A股交易日历（YYYY-MM-DD 字符串集合），全局唯一缓存。

    - 数据源: 优先本地文件 quant_system/data/trade_calendar.csv（2026-08-07 起，
      真实交易日历快照，避免本机网络受限导致阻塞）；其次 akshare 拉取
    - 失败降级: 仅按周末粗筛（周一到周五），并打印警告
    - P2-Q4-fix(M419): 失败时缓存空集但记录时间戳，超过 _CALENDAR_RETRY_TTL 后
      重新尝试拉取，避免进程生命周期内永久降级为周末粗筛模式。
    """
    global _TRADE_CALENDAR_CACHE, _CALENDAR_LAST_FAIL_TS
    now = _time.time()
    if _TRADE_CALENDAR_CACHE is not None:
        # 缓存有效：非空直接返回；空缓存（上次失败）需超过 TTL 才重试
        if _TRADE_CALENDAR_CACHE or now - _CALENDAR_LAST_FAIL_TS < _CALENDAR_RETRY_TTL:
            return _TRADE_CALENDAR_CACHE
    with _TRADE_CALENDAR_LOCK:
        if _TRADE_CALENDAR_CACHE is not None:
            if _TRADE_CALENDAR_CACHE or now - _CALENDAR_LAST_FAIL_TS < _CALENDAR_RETRY_TTL:
                return _TRADE_CALENDAR_CACHE
        # 1) 本地文件优先（不联网、不阻塞）
        file_cal = _load_calendar_from_file()
        if file_cal:
            _TRADE_CALENDAR_CACHE = file_cal
            _CALENDAR_LAST_FAIL_TS = 0.0
            return _TRADE_CALENDAR_CACHE
        try:
            import akshare as ak
            cal_df = ak.tool_trade_date_hist_sina()
            dates = pd.to_datetime(cal_df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
            _TRADE_CALENDAR_CACHE = set(dates)
            _CALENDAR_LAST_FAIL_TS = 0.0
        except Exception as exc:  # pragma: no cover - 网络异常降级
            print(f"⚠️ [market_clock] 交易日历获取失败({exc})，降级为周末粗筛，"
                  f"{int(_CALENDAR_RETRY_TTL)}s 后重试")
            if _TRADE_CALENDAR_CACHE is None:
                _TRADE_CALENDAR_CACHE = set()
            _CALENDAR_LAST_FAIL_TS = _time.time()
        return _TRADE_CALENDAR_CACHE


def is_trading_day(dt: Optional[datetime | date | str] = None) -> bool:
    """判断指定日期是否为A股交易日。

    Args:
        dt: datetime/date/YYYY-MM-DD 字符串；None=当前时间(Asia/Shanghai)

    Returns:
        True=交易日
    """
    if dt is None:
        d = datetime.now(CST).date()
    elif isinstance(dt, str):
        d = pd.to_datetime(dt).date()
    else:
        d = dt.date() if isinstance(dt, datetime) else dt

    cal = get_trade_calendar()
    if cal:
        ds = d.strftime("%Y-%m-%d")
        if ds in cal:
            return True
        # V11 审计修复（Medium）: 本地日历只到 2026-12-31，2027 年后所有日期
        # 都不在日历里 → is_trading_day 恒 False（系统 5 个月后整体失效）。
        # 修正: 日期超出日历范围时回退工作日粗筛，并尝试刷新日历。
        _max_d = _CAL_MAX_DATE
        if _max_d is not None and d > _max_d:
            try:
                get_trade_calendar()  # 审计 2026-08-16：真正重新加载/刷新日历，而非不存在的 .refresh 属性
            except Exception as e:
                logging.getLogger(__name__).error(f"[market_clock] 操作失败: {e}", exc_info=True)
            cal2 = _TRADE_CALENDAR_CACHE
            if cal2 and ds in cal2:
                return True
            return d.weekday() < 5  # 日历过期降级：仅周末粗筛
        return False
    return d.weekday() < 5  # 降级：仅周末


def is_trading_session(dt: Optional[datetime] = None, market: str = "沪深") -> bool:
    """当前/指定时刻是否处于可交易时段（集合竞价+连续竞价+收盘竞价）。

    先查真实交易日历，再交给 market_rules 判断日内时段。
    """
    dt = dt or datetime.now(CST)
    if not is_trading_day(dt):
        return False
    return _rules_trading_session(dt, market)


def is_call_auction(dt: Optional[datetime] = None) -> bool:
    """是否处于集合竞价（开盘 9:15-9:25 / 收盘 14:57-15:00）。"""
    dt = dt or datetime.now(CST)
    return is_in_call_auction(dt)


def is_continuous(dt: Optional[datetime] = None) -> bool:
    """是否处于连续竞价（9:30-11:30 / 13:00-14:57）。"""
    dt = dt or datetime.now(CST)
    return is_in_continuous_auction(dt)


def phase_name(dt: Optional[datetime] = None, market: str = "沪深") -> Optional[str]:
    """当前阶段名（同 market_rules.current_phase）。"""
    dt = dt or datetime.now(CST)
    return current_phase(dt, market)


def _shift_trading_day(d: date, delta: int) -> Optional[date]:
    """从 d 向前/向后找第 |delta| 个交易日。"""
    cal = get_trade_calendar()
    if not cal:
        # 降级：按自然日推进（周末跳过）
        cur = d
        step = 1 if delta > 0 else -1
        n = abs(delta)
        while n > 0:
            cur += timedelta(days=step)
            if cur.weekday() < 5:
                n -= 1
        return cur
    dates = sorted(cal)
    dstr = d.strftime("%Y-%m-%d")
    if delta == 0:
        return d if dstr in cal else None

    if delta > 0:
        candidates = [x for x in dates if x > dstr]
        target = delta - 1
    else:
        candidates = [x for x in reversed(dates) if x < dstr]
        target = abs(delta) - 1

    if target < 0 or target >= len(candidates):
        return None
    return pd.to_datetime(candidates[target]).date()


def next_trading_day(dt: Optional[datetime | date | str] = None,
                     n: int = 1) -> Optional[date]:
    """下一个交易日（n=1 默认次日；n=2 次次日，依此类推）。

    Returns:
        date 对象；超出日历范围返回 None
    """
    if dt is None:
        d = datetime.now(CST).date()
    elif isinstance(dt, str):
        d = pd.to_datetime(dt).date()
    else:
        d = dt.date() if isinstance(dt, datetime) else dt
    return _shift_trading_day(d, n)


def prev_trading_day(dt: Optional[datetime | date | str] = None,
                     n: int = 1) -> Optional[date]:
    """上一个交易日。"""
    if dt is None:
        d = datetime.now(CST).date()
    elif isinstance(dt, str):
        d = pd.to_datetime(dt).date()
    else:
        d = dt.date() if isinstance(dt, datetime) else dt
    return _shift_trading_day(d, -n)


def latest_trading_day(dt: Optional[datetime | date | str] = None) -> date:
    """最近的一个交易日（若 dt 本身是交易日则返回 dt，否则向前找）。"""
    if dt is None:
        d = datetime.now(CST).date()
    elif isinstance(dt, str):
        d = pd.to_datetime(dt).date()
    else:
        d = dt.date() if isinstance(dt, datetime) else dt
    if is_trading_day(d):
        return d
    p = prev_trading_day(d)
    return p if p is not None else d


def latest_completed_trading_day(dt: Optional[datetime | date | str] = None) -> date:
    """最近已收盘交易日；盘前/盘中不把当天未收盘数据当作完整日报。"""
    if isinstance(dt, str):
        parsed = pd.Timestamp(dt).to_pydatetime()
        now = parsed.astimezone(CST) if parsed.tzinfo is not None else parsed.replace(tzinfo=CST)
    else:
        now = dt if isinstance(dt, datetime) else datetime.now(CST) if dt is None else None
        if isinstance(now, datetime):
            now = now.astimezone(CST) if now.tzinfo is not None else now.replace(tzinfo=CST)
    if now is not None and is_trading_day(now.date()) and now.strftime("%H:%M") < "15:05":
        return prev_trading_day(now.date()) or now.date()
    return latest_trading_day(dt)


def is_market_open_now() -> bool:
    """快捷判断：此刻是否在交易时段内（供盘中监控/定时任务使用）。"""
    return is_trading_session()


def self_test() -> list[str]:
    ok: list[str] = []
    cal = get_trade_calendar()
    assert isinstance(cal, set), "日历应为set"
    assert len(cal) > 200, f"日历天数异常: {len(cal)}"
    # 2024年国庆假期（10-01 ~ 10-07 休市）不应是交易日
    if "2024-10-01" in cal:
        assert not is_trading_day("2024-10-01"), "2024-10-01 国庆不应是交易日"
    # 普通工作日应为交易日（如 2024-06-03 周一）
    assert is_trading_day("2024-06-03"), "2024-06-03 周一应为交易日"
    # 周末不是交易日
    assert not is_trading_day("2024-06-01"), "2024-06-01 周六不应是交易日"
    # 下个交易日推算：2024-06-07(周五) 下个交易日应为 2024-06-11(周二，端午假期后)
    nxt = next_trading_day("2024-06-07")
    assert nxt is not None and nxt.strftime("%Y-%m-%d") == "2024-06-11", f"端午后首日推算错误: {nxt}"
    prv = prev_trading_day("2024-06-11")
    assert prv is not None and prv.strftime("%Y-%m-%d") == "2024-06-07", f"端午前一日推算错误: {prv}"
    # 交易时段函数可调用
    _ = is_trading_session(), is_call_auction(), is_continuous(), phase_name()
    ok.append(f"market_clock self-test: ALL PASS ✅ (日历 {len(cal)} 天)")
    return ok


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        for line in self_test():
            print(line)
    else:
        cal = get_trade_calendar()
        print(f"交易日历: {len(cal)} 天 | 源: {_CALENDAR_SOURCE}")
        print(f"今天 {datetime.now(CST).date()}: {'交易日' if is_trading_day() else '非交易日'}")
        print(f"当前阶段: {phase_name() or '非交易时段'}")
        print(f"下一交易日: {next_trading_day()}")
        print(f"上一交易日: {prev_trading_day()}")
