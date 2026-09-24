"""Sina-finance based real-time A-share quote fetcher — works from abroad."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import requests

SINA_URL = "https://hq.sinajs.cn/list={codes}"
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}


@dataclass
class RealtimeQuote:
    symbol: str          # 002714
    name: str            # 牧原股份
    open: float
    prev_close: float
    price: float          # 现价
    high: float
    low: float
    volume: int           # 成交量(股)
    amount: float         # 成交额(元)
    bid: float
    ask: float
    date: str             # 2026-07-27
    time: str             # 15:35:45

    @property
    def change(self) -> float:
        return self.price - self.prev_close

    @property
    def change_pct(self) -> float:
        if self.prev_close == 0:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close * 100

    @property
    def amplitude(self) -> float:
        """日内振幅百分比"""
        if self.prev_close == 0:
            return 0.0
        return (self.high - self.low) / self.prev_close * 100


def _market_code(symbol: str) -> str:
    """Convert 002714 → sz002714, 601899 → sh601899, sh000001 → sh000001.

    Q10-fix: already-prefixed codes (sh000001/sz399001/bj...) must pass through
    unchanged; previously they fell to the final else and became szsh000001.
    """
    symbol = symbol.strip().split('.')[0]
    low = symbol.lower()  # P2-Q26-fix: 归一化小写，避免大写前缀(SH600519)落入末位 else
    if low.startswith(('sh', 'sz', 'bj')) and len(low) >= 8:
        return low
    if low.startswith(('6', '9')):
        return f"sh{low}"
    elif low.startswith(('0', '3')):
        return f"sz{low}"
    elif low.startswith(('4', '8')):
        return f"bj{low}"
    return f"sz{low}"


def _parse_sina_line(var_name: str, raw: str) -> RealtimeQuote | None:
    """Parse a single Sina JS var line.
    var_name is like 'hq_str_sz002714' or 'hq_str_sh601899'.
    """
    try:
        parts = raw.strip().strip('"').split(',')
        if len(parts) < 30:
            return None
        # Extract raw code like 'sz002714' from 'var hq_str_sz002714'
        raw_code = var_name.replace("var hq_str_", "").replace("hq_str_", "").strip()
        # Normalize to short form
        if raw_code.startswith("sh"):
            norm = raw_code[2:]
        elif raw_code.startswith("sz"):
            norm = raw_code[2:]
        elif raw_code.startswith("bj"):
            norm = raw_code[2:]
        else:
            norm = raw_code
        return RealtimeQuote(
            symbol=norm,
            name=parts[0],
            open=float(parts[1]),
            prev_close=float(parts[2]),
            price=float(parts[3]),
            high=float(parts[4]),
            low=float(parts[5]),
            bid=float(parts[6]) if parts[6] else 0,
            ask=float(parts[7]) if parts[7] else 0,
            volume=int(parts[8]) if parts[8] else 0,
            amount=float(parts[9]) if parts[9] else 0,
            date=parts[30],
            time=parts[31],
        )
    except (ValueError, IndexError):
        return None


def fetch_realtime(symbols: list[str], timeout: int = 10) -> dict[str, RealtimeQuote]:
    """Batch fetch real-time quotes for a list of A-share symbols.

    Uses Sina finance API which works from abroad.
    Returns dict of {symbol: RealtimeQuote}.
    """
    if not symbols:
        return {}
    codes = [_market_code(s) for s in symbols]
    url = SINA_URL.format(codes=",".join(codes))
    try:
        r = requests.get(url, headers=SINA_HEADERS, timeout=timeout)
        r.encoding = "gbk"
    except requests.RequestException:
        return {}

    result: dict[str, RealtimeQuote] = {}
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        var_name, raw_value = line.split("=", 1)
        quote = _parse_sina_line(var_name, raw_value)
        if quote:
            result[quote.symbol] = quote
    return result


def watchlist_from_file(path: str) -> list[str]:
    """Load symbol list from quant workstation's watchlist.json."""
    import json as _json
    try:
        with open(path) as f:
            data = _json.load(f)
        raw = data if isinstance(data, list) else data.get("symbols", [])
        return [s.strip().split(".")[0] for s in raw if s.strip()]
    # P2-Q26-fix: 模块级未 import json（仅函数内 import json as _json），
    # 原引用 json.JSONDecodeError 在配置文件为无效 JSON 时抛 NameError 而非返回 []。
    except (FileNotFoundError, _json.JSONDecodeError):
        return []


def is_trading_time(now: _time.struct_time | None = None) -> bool:
    """Check if current time falls within A-share trading hours (Asia/Shanghai).

    Note: 未接交易日历，春节/国庆等节假日按普通交易日判断（轮询层需自行处理）。
    """
    if now is None:
        now = _time.localtime()
    h, m = now.tm_hour, now.tm_min
    weekday = now.tm_wday
    if weekday >= 5:  # Saturday/Sunday
        return False
    # 9:30-11:30, 13:00-15:00
    if (h == 9 and m >= 30) or (h == 10) or (h == 11 and m <= 30):
        return True
    # P2-Q26-fix: 15:00:00 收盘集合竞价已结束（原 h==15 and m==0 误判为交易中）
    if (h == 13) or (h == 14):
        return True
    return False


def seconds_until_market_open(now: _time.struct_time | None = None) -> int:
    """Seconds until next market open (for sleep calculation)."""
    if now is None:
        now = _time.localtime()
    h, m = now.tm_hour, now.tm_min
    weekday = now.tm_wday

    # P2-Q26-fix: 先判断周末再判断时段（原实现 h>=15 and weekday<5 分支在周五收盘后
    # 计 days_ahead=0 指向周六 9:30；周末早盘也会误判为当日开盘）。
    if weekday >= 5:  # 周六/周日 → 下周一 9:30
        target_h, target_m = 9, 30
        days_ahead = 7 - weekday
    elif h >= 15:  # 收盘后 → 下一交易日 9:30（周五收盘计3天，指向周一）
        target_h, target_m = 9, 30
        days_ahead = 3 if weekday == 4 else 1
    elif h < 9 or (h == 9 and m < 30):  # 盘前 → 当日 9:30
        target_h, target_m = 9, 30
        days_ahead = 0
    elif 11 < h < 13 or (h == 11 and m > 30):  # 午休 → 13:00
        target_h, target_m = 13, 0
        days_ahead = 0
    else:
        # Currently in trading - return 0
        return 0

    # Localtime in Shanghai timezone
    # Simple approach: calculate seconds to target
    now_sec = h * 3600 + m * 60 + now.tm_sec
    target_sec = target_h * 3600 + target_m * 60
    if days_ahead == 0 and target_sec > now_sec:
        return target_sec - now_sec
    # Add days
    return (days_ahead * 86400) + (target_sec - now_sec if days_ahead > 0 else target_sec - now_sec + 86400)


if __name__ == "__main__":
    # Quick test
    syms = ["002714", "601899", "000858"]
    quotes = fetch_realtime(syms)
    for sym, q in sorted(quotes.items()):
        print(f"{q.name:6s} ({sym}): {q.price:>7.2f}  {q.change_pct:>+6.2f}%  量{q.volume//10000}万")
    print(f"Trading now: {is_trading_time()}")
