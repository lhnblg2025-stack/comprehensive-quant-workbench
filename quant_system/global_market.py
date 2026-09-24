"""港美股实时行情 — 腾讯 qt.gtimg.cn 为主（海外可用），新浪/东方财富为备。"""

from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import requests as _req


# ── dataclass ──────────────────────────────────────────────────────────────

@dataclass
class GlobalQuote:
    symbol: str       # e.g. "hk00700", "usAAPL"
    name: str
    price: float
    prev_close: float
    open_p: float     # today open
    high: float
    low: float
    change: float
    change_pct: float
    volume: int
    amount: float
    time: str


# ── Tencent API ────────────────────────────────────────────────────────────

def _parse_tencent(symbol: str, raw: str) -> GlobalQuote | None:
    """解析腾讯行情原始响应（~ 分隔）为 GlobalQuote。

    P1-Q23-fix(H02): 实测 hk00700 响应字段位序为
      parts[29]=成交量, parts[30]=时间, parts[31]=涨跌额, parts[32]=涨跌幅,
      parts[33]=最高, parts[34]=最低, parts[37]=成交额。
    原代码 change=parts[29]（港股取到成交量、美股空串→0）、
    time_str=parts[31]（实为涨跌额），字段错位。这里改为
      change ← parts[31]、time ← parts[30]。
    独立成函数便于离线回归测试（见文件末尾 __main__）。
    """
    if not raw or "=" not in raw:
        return None
    parts = raw.split("~")
    if len(parts) < 10:
        return None
    name = parts[1] if parts[1] else symbol
    price = _f(parts[3])
    prev_close = _f(parts[4])
    open_p = _f(parts[5])
    volume = _i(parts[6])  # P2-Q23-fix(L267): 股（港股/美股成交量单位均为股，非"手"）
    high = _f(parts[33]) if len(parts) > 33 else price
    low = _f(parts[34]) if len(parts) > 34 else price
    time_str = parts[30] if len(parts) > 30 else ""
    amount = _f(parts[37]) if len(parts) > 37 else 0.0
    change = _f(parts[31]) if len(parts) > 31 else price - prev_close
    change_pct = _f(parts[32]) if len(parts) > 32 else 0.0
    return GlobalQuote(
        symbol=symbol,
        name=name,
        price=price,
        prev_close=prev_close,
        open_p=open_p,
        high=high,
        low=low,
        change=change,
        change_pct=change_pct,
        volume=volume,
        amount=amount,
        time=time_str,
    )


def _fetch_tencent(symbol: str, timeout: int = 10) -> GlobalQuote | None:
    """Fetch HK/US quote from Tencent qt.gtimg.cn."""
    url = f"http://qt.gtimg.cn/q={symbol}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com"}
    try:
        r = _req.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        return _parse_tencent(symbol, r.text.strip())
    except Exception:
        return None


# ── Sina API ───────────────────────────────────────────────────────────────

def _fetch_sina_hk(symbol: str, timeout: int = 10) -> GlobalQuote | None:
    """Fetch HK quote from Sina (备用).

    P2-Q23-fix(L268): hq.sinajs.cn 在本机不可达（实测 ConnectionError），该
    兜底在此环境通常失效——调用方表现为该符号缺失（部分结果），属可见降级。
    """
    url = f"http://hq.sinajs.cn/list=hk{symbol.replace('hk', '')}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    try:
        r = _req.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        raw = r.text.strip()
        if not raw or "=" not in raw:
            return None
        parts = raw.split("\"")[1].split(",") if "\"" in raw else raw.split(",")
        if len(parts) < 10:
            return None
        # P2-Q23-fix(L267): 修正新浪港股字段位序——原实现整体错位（price 取到
        # 成交量、prev_close 取到今开、name 取到现价、time 取到名称），导致
        # change/change_pct 恒 0 且各字段错误。sina 港股格式（逗号分隔）：
        #   [0]名称 [1]现价 [2]昨收 [3]今开 [4]最高 [5]最低
        #   [6]成交量(股) [7]成交额 [8]日期 [9]时间
        name = parts[0] if parts[0] else symbol
        price = _f(parts[1]) if len(parts) > 1 else 0
        prev_close = _f(parts[2]) if len(parts) > 2 else 0
        change = price - prev_close
        return GlobalQuote(
            symbol=symbol,
            name=name,
            price=price,
            prev_close=prev_close,
            open_p=_f(parts[3]) if len(parts) > 3 else 0,
            high=_f(parts[4]) if len(parts) > 4 else 0,
            low=_f(parts[5]) if len(parts) > 5 else 0,
            change=change,
            change_pct=(change / prev_close * 100) if prev_close else 0.0,
            volume=_i(parts[6]) if len(parts) > 6 else 0,
            amount=_f(parts[7]) if len(parts) > 7 else 0,
            time=f"{parts[8]} {parts[9]}" if len(parts) > 9 else (parts[8] if len(parts) > 8 else ""),
        )
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────

# Common HK stock codes for quick lookup
HK_KNOWN = {
    "00700": "腾讯控股", "03690": "美团-W", "09988": "阿里巴巴-W",
    "09999": "网易-S", "01810": "小米集团-W", "09618": "京东集团-SW",
    "02015": "理想汽车-W", "09888": "百度集团-SW", "01024": "快手-W",
    "00388": "香港交易所", "00005": "汇丰控股", "01299": "友邦保险",
    "00941": "中国移动", "00883": "中国海洋石油", "01398": "工商银行",
    "03988": "中国银行", "02318": "中国平安", "00728": "中国电信",
    "02331": "李宁", "09901": "新东方在线", "06618": "京东健康",
    "02269": "药明生物", "06160": "百济神州",
}

US_KNOWN = {
    "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌", "AMZN": "亚马逊",
    "META": "Meta", "NVDA": "英伟达", "TSLA": "特斯拉", "AMD": "超威半导体",
    "JPM": "摩根大通", "V": "Visa", "JNJ": "强生", "WMT": "沃尔玛",
    "PG": "宝洁", "UNH": "联合健康", "HD": "家得宝", "DIS": "迪士尼",
    "MA": "万事达", "BAC": "美国银行", "NFLX": "奈飞", "ADBE": "Adobe",
    "CRM": "Salesforce", "INTC": "英特尔", "CSCO": "思科", "PEP": "百事",
    "KO": "可口可乐", "BABA": "阿里巴巴", "JD": "京东", "BIDU": "百度",
    "NIO": "蔚来", "XPEV": "小鹏汽车", "LI": "理想汽车",
}

# ── Convert user input ──

def normalize_global_symbol(symbol: str) -> str:
    """Normalize HK/US symbol for Tencent API.

    HK: '00700' or 'hk00700' or '0700.HK' → 'hk00700'
    US: 'AAPL' or 'usAAPL' or 'AAPL.US' → 'usAAPL'
    """
    s = str(symbol).strip().upper().replace(" ", "")
    if s.startswith("HK"):
        return f"hk{s[2:].zfill(5)}"
    if s.startswith("US"):
        return f"us{s[2:]}"
    if ".HK" in s:
        return f"hk{s.replace('.HK', '').zfill(5)}"
    if ".US" in s or ".OQ" in s or ".N" in s:
        return f"us{s.split('.')[0]}"
    if s.isdigit():
        return f"hk{s.zfill(5)}"
    # Assume US stock
    return f"us{s}"


def resolve_global_symbol(symbol: str) -> str:
    """Resolve a full name or code to a normalized symbol.

    '腾讯' → 'hk00700', '苹果' → 'usAAPL', '00700' → 'hk00700'
    """
    s = str(symbol).strip()
    # Check name maps
    for code, name in HK_KNOWN.items():
        if s in name or name in s:
            return f"hk{code}"
    for code, name in US_KNOWN.items():
        if s in name or name in s:
            return f"us{code}"
    return normalize_global_symbol(s)


def fetch_global_quotes(
    symbols: Sequence[str], timeout: int = 10
) -> dict[str, GlobalQuote]:
    """Batch fetch HK/US realtime quotes.

    Parameters
    ----------
    symbols : list of str — e.g. ['hk00700', 'usAAPL', '腾讯', '苹果']
    timeout : int

    Returns
    -------
    dict[symbol → GlobalQuote]

    P2-Q23-fix(L267): 原实现逐个符号串行请求腾讯接口，一次调用 N 只股票需
    N 次往返。腾讯支持逗号批量查询(q=sym1,sym2,...)，改为单次批量请求，
    失败时再逐个回退；港股仍保留新浪兜底。
    """
    normalized = [resolve_global_symbol(s) for s in symbols]
    results: dict[str, GlobalQuote] = {}

    batch_url = f"http://qt.gtimg.cn/q={','.join(normalized)}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com"}
    try:
        r = _req.get(batch_url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            text = r.text
            for sym in normalized:
                quote = None
                marker = f'v_{sym}="'
                start = text.find(marker)
                if start >= 0:
                    start += len(marker)
                    end = text.find('"', start)
                    if end > start:
                        quote = _parse_tencent(sym, f'{marker}{text[start:end]}"')
                if quote is not None:
                    results[sym] = quote
                elif sym.startswith("hk"):
                    # 批量中缺失 → 新浪兜底（港股）
                    quote = _fetch_sina_hk(sym, timeout=timeout)
                    if quote is not None:
                        results[sym] = quote
            if results:
                return results
    except Exception as e:
        logging.getLogger(__name__).error(f"[global_market] 操作失败: {e}", exc_info=True)

    # 批量请求失败/无结果 → 逐个回退（保持原有行为）
    for sym in normalized:
        quote = _fetch_tencent(sym, timeout=timeout)
        if quote is not None:
            results[sym] = quote
        elif sym.startswith("hk"):
            quote = _fetch_sina_hk(sym, timeout=timeout)
            if quote is not None:
                results[sym] = quote
    return results


def fetch_hk_market_summary(timeout: int = 15) -> dict:
    """Fetch HK market summary via akshare (当备用).

    P2-Q23-fix(L268): 快照表 df.head(1) 是接口返回的首行，并非"时间序列最新"，
    原字段名 latest 语义误导，改名 first_row 并附说明。
    """
    try:
        import akshare as ak
        df = ak.stock_hk_spot_em()
        if df is None or df.empty:
            return {"ok": False, "error": "empty"}
        first_row = df.head(1).to_dict(orient="records")[0]
        return {
            "ok": True,
            "total": len(df),
            "first_row": first_row,
            "note": "first_row 为接口快照首行(未按时间排序),非行情最新",
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)[:80]}


def fetch_us_market_summary(timeout: int = 15) -> dict:
    """Fetch US market summary via akshare."""
    try:
        import akshare as ak
        df = ak.stock_us_spot_em()
        if df is None or df.empty:
            return {"ok": False, "error": "empty"}
        return {"ok": True, "total": len(df)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)[:80]}


# ── K-Line ────────────────────────────────────────────────────────────────

def _yahoo_symbol(symbol: str) -> str:
    """Convert our symbol to Yahoo Finance symbol.

    hk00700 → 0700.HK
    usAAPL  → AAPL
    """
    s = symbol.lower()
    if s.startswith("hk"):
        code = s[2:].lstrip("0")
        # Pad to 4 digits for Yahoo (e.g. 0700.HK, 0005.HK)
        code = code.zfill(4) if len(code) <= 4 else code
        return code + ".HK"
    if s.startswith("us"):
        return s[2:]
    return symbol


def fetch_global_kline(symbol: str, period: str = "daily", count: int = 120) -> pd.DataFrame:
    """Fetch HK/US stock K-line via Yahoo Finance (works from abroad).

    Parameters
    ----------
    symbol : str   — normalized symbol e.g. 'hk00700', 'usAAPL', '腾讯'
    period : str   — 'daily' | 'weekly' | 'monthly'
    count : int    — number of data points

    Returns
    -------
    pd.DataFrame with columns [date, open, high, low, close, volume]
    """
    from datetime import datetime, timezone

    sym = resolve_global_symbol(symbol)
    yahoo_sym = _yahoo_symbol(sym)

    # Map period to Yahoo interval
    interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    interval = interval_map.get(period, "1d")

    # P2-Q23-fix(M266): 删除从未使用的 start/end 计算（请求实际走 range 参数，
    # 此前的 end/start 为死代码）。

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + yahoo_sym
    # V11 审计修复（Medium）: 原 range=count*1.5 天，月线/周线 bar 数严重不足
    # （月线 120 根需 ~3600 天，原只取 180 天 ≈ 6 根，MA60/布林带全 NaN）。
    # 修正: 按 interval 折算天数（日=1.5x / 周=10x / 月=45x 安全倍数）。
    _scale = {"1d": 1.5, "1wk": 10, "1mo": 45}.get(interval, 1.5)
    params = {
        "range": f"{int(count * _scale)}d",
        "interval": interval,
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = _req.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()
        timestamps = result[0].get("timestamp", [])
        quotes = result[0].get("indicators", {}).get("quote", [{}])
        if not quotes:
            return pd.DataFrame()
        quote = quotes[0]
        rows = []
        for i, ts in enumerate(timestamps):
            o = (quote.get("open") or [])[i] if quote.get("open") else None
            h = (quote.get("high") or [])[i]
            l = (quote.get("low") or [])[i]
            c = (quote.get("close") or [])[i]
            v = (quote.get("volume") or [])[i]
            if o is not None and c is not None:
                # P2-Q23-fix(M266): 原 datetime.fromtimestamp(ts) 用本地时区(上海)
                # 转日期——美股收盘(16:00 EST=次日05:00 上海)会被标成下一个交易日，
                # 跨市场日期偏移一天。改 UTC 对齐交易所交易日历（美/港收盘时间戳
                # 转 UTC 后日期即当日）。
                rows.append({
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "open": float(o),
                    "high": float(h) if h else float(o),
                    "low": float(l) if l else float(o),
                    "close": float(c),
                    "volume": float(v) if v else 0,
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_global_kline_with_indicators(
    symbol: str, period: str = "daily", count: int = 120
) -> pd.DataFrame:
    """Fetch K-line and add MA indicators."""
    df = fetch_global_kline(symbol, period=period, count=count)
    if df.empty:
        return df
    # Add MAs
    for ma in [5, 10, 20, 60]:
        if len(df) >= ma:
            df[f"ma{ma}"] = df["close"].rolling(ma).mean()
    # Add Bollinger Bands (20,2)
    if len(df) >= 20:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["boll_upper"] = mid + 2 * std
        df["boll_mid"] = mid
        df["boll_lower"] = mid - 2 * std
    return df


# ── helpers ──

def _f(val: str) -> float:
    try:
        return float(val.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _i(val: str) -> int:
    try:
        return int(val.replace(",", ""))
    except (ValueError, AttributeError):
        return 0


def _self_test() -> None:
    """P1-Q23-fix(H02) 离线回归测试：验证腾讯行情字段位序映射。

    按实测 hk00700 响应位序构造夹具：
      parts[29]=成交量, parts[30]=时间, parts[31]=涨跌额, parts[32]=涨跌幅,
      parts[33]=最高, parts[34]=最低, parts[37]=成交额。
    若字段映射回退（如 change 又取到 parts[29] 成交量），断言将失败。
    """
    fields = [""] * 40
    fields[0] = 'v_hk00700="100'
    fields[1] = "腾讯控股"
    fields[2] = "00700"
    fields[3] = "280.000"                       # 现价
    fields[4] = "279.000"                       # 昨收
    fields[5] = "279.800"                       # 今开
    fields[6] = "25968050"                      # 成交量(股)
    fields[29] = "25968050"                     # 成交量
    fields[30] = "2021-09-01 16:08:00"          # 时间
    fields[31] = "+1.000"                       # 涨跌额
    fields[32] = "0.36"                         # 涨跌幅(%)
    fields[33] = "282.800"                      # 最高
    fields[34] = "275.000"                      # 最低
    fields[37] = "7350268000"                   # 成交额(元)
    raw = "~".join(fields) + '"'

    q = _parse_tencent("hk00700", raw)
    assert q is not None, "解析失败"
    assert q.price == 280.0, f"price={q.price}"
    assert q.prev_close == 279.0, f"prev_close={q.prev_close}"
    assert q.change == 1.0, f"change={q.change} (应为涨跌额 1.0，而非成交量)"
    assert q.change_pct == 0.36, f"change_pct={q.change_pct}"
    assert q.high == 282.8, f"high={q.high}"
    assert q.low == 275.0, f"low={q.low}"
    assert q.amount == 7350268000.0, f"amount={q.amount}"
    assert q.time == "2021-09-01 16:08:00", f"time={q.time} (应为时间，而非涨跌额)"
    print("global_market P1-Q23-fix(H02) 字段映射回归测试通过 ✓")


if __name__ == "__main__":
    _self_test()
