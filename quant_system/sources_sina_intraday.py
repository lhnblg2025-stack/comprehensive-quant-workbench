"""
Sina minute K-line data — works from abroad (UK server OK).

Endpoint (found 2026-07-28):
  https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
    ?symbol=sh600519&scale=60&ma=5&datalen=200

Parameters:
  symbol : sh600519 / sz002714
  scale  : 5/15/30/60 (minutes); 1 分钟新浪不支持
  ma     : include MA lines (ma_price5, ma_volume5, etc.)
  datalen: number of bars to return (上限约 2000, 超限会被服务端截断)
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import requests

# P2-Q2-fix: L199 改用 https, 避免明文被劫持/拦截
SINA_KLINE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn",
}


def fetch_sina_intraday(
    symbol: str,
    scale: int = 60,
    datalen: int = 200,
) -> Optional[pd.DataFrame]:
    """
    从新浪获取分钟K线数据。

    Args:
        symbol: 股票代码如 600519
        scale:  分钟数 (5/15/30/60; 1 分钟新浪不支持, 返回 None)
        datalen: 最大返回行数 (上限约 2000, 超限会被服务端截断, 这里钳制到 2000)

    Returns:
        DataFrame with columns [date, open, high, low, close, volume,
                                 ma_price5, ma_volume5] 或 None
    """
    # P2-Q2-fix: L200 入参校验, 避免 scale=1 静默失败、datalen 超限被截断无提示
    try:
        scale = int(scale)
        datalen = int(datalen)
    except (TypeError, ValueError):
        return None
    if scale not in (5, 15, 30, 60):
        return None
    if datalen < 1:
        datalen = 1
    if datalen > 2000:
        datalen = 2000

    # 解析交易所前缀 (P2-Q2-fix: M191 北交所 4/8/920 前缀)
    from .sources_all import market_prefix

    sym = symbol.strip()
    sina_sym = f"{market_prefix(sym)}{sym}"

    params = {
        "symbol": sina_sym,
        "scale": scale,
        "ma": "5",
        "datalen": datalen,
    }

    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        r = requests.get(SINA_KLINE_URL, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            _log.warning("[sina_intraday] %s 状态码 %s", sina_sym, r.status_code)
            return None
        raw = r.text.strip()
        if not raw or raw == "null":
            _log.warning("[sina_intraday] %s 返回空/无数据", sina_sym)
            return None

        data = json.loads(raw)
        if not data or not isinstance(data, list):
            _log.warning("[sina_intraday] %s 返回格式异常(非列表)", sina_sym)
            return None

        df = pd.DataFrame(data)
        # 数值化
        for col in ["open", "high", "low", "close", "volume", "ma_price5", "ma_volume5"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["day"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as exc:
        _log.warning("[sina_intraday] %s 拉取失败: %s", sina_sym, exc)
        return None


def fetch_intraday_with_indicators(
    symbol: str,
    scale: int = 60,
    datalen: int = 200,
) -> pd.DataFrame:
    """
    获取分钟K线并计算技术指标 (含 CCI, EMA 5/10/21, MA排列等)。

    Returns:
        DataFrame with indicators, 或空的 DataFrame
    """
    df = fetch_sina_intraday(symbol, scale, datalen)
    if df is None or df.empty:
        return pd.DataFrame()

    from .indicators import add_intraday_indicators
    return add_intraday_indicators(df)


def _rnd(value: Any, ndigits: int, default: float | None = 0.0):
    """P2-Q2-fix: L200 NaN-aware 取整, 空值/NaN 返回 default, 避免 round(NaN)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(v):
        return default
    return round(v, ndigits)


def analyze_intraday_hourly(
    symbol: str,
    datalen: int = 200,
) -> dict[str, Any]:
    """
    小时级（60分钟）K线分析。

    Returns:
        dict with CCI, MA signals, trend assessment
    """
    df = fetch_intraday_with_indicators(symbol, scale=60, datalen=datalen)
    if df.empty:
        return {"ok": False, "error": f"无法获取{symbol}60分钟K线"}

    latest = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict() if len(df) >= 2 else {}

    result = {"ok": True}
    result["symbol"] = symbol
    result["scale"] = "60min"
    result["n_bars"] = len(df)
    result["latest_time"] = str(latest.get("date", ""))

    # CCI
    cci = latest.get("cci_10")
    result["cci"] = {
        "value": round(cci, 1) if cci and not pd.isna(cci) else None,
        "prev": _rnd(prev.get("cci_10", 0), 1, None) if prev else None,
        "zone": ("超买(>200)" if cci and not pd.isna(cci) and cci > 200 else
                 "超买(>100)" if cci and not pd.isna(cci) and cci > 100 else
                 "超卖(<-200)" if cci and not pd.isna(cci) and cci < -200 else
                 "超卖(<-100)" if cci and not pd.isna(cci) and cci < -100 else
                 "正常"),
    }

    # MA交叉信号
    cross = latest.get("ma_cross", 0)
    result["ma_signal"] = {
        "cross": "金叉(做多)" if cross == 1 else "死叉(做空)" if cross == -1 else "无",
        "bull_arrange": bool(latest.get("ma_bull", 0)),
        "bear_arrange": bool(latest.get("ma_bear", 0)),
        "ema_5": _rnd(latest.get("ema_5", 0), 2),
        "ema_10": _rnd(latest.get("ema_10", 0), 2),
        "ema_21": _rnd(latest.get("ema_21", 0), 2),
        "ma_5": _rnd(latest.get("ma_5", 0), 2),
        "ma_20": _rnd(latest.get("ma_20", 0), 2),
        "ma_60": _rnd(latest.get("ma_60", 0), 2),
    }

    # RSI
    rsi = latest.get("rsi_14", 50)
    result["rsi_14"] = round(rsi, 1) if not pd.isna(rsi) else None

    # MACD
    result["macd"] = {
        "hist": _rnd(latest.get("macd_hist", 0), 3),
        "dif": _rnd(latest.get("macd_dif", 0), 2),
        "dea": _rnd(latest.get("macd_dea", 0), 2),
    }

    # Volume
    vol = latest.get("volume", 0)
    vol_ratio = latest.get("volume_ratio", 1.0)
    result["volume"] = {
        "current": vol,
        "ratio": _rnd(vol_ratio, 2, 1.0),
    }

    # ATR
    result["atr_pct"] = _rnd(latest.get("atr_pct", 0), 2)

    # 综合小时信号
    signals = []
    if cci and not pd.isna(cci) and cci > 100:
        signals.append(f"CCI超买({cci:.0f})")
    if cci and not pd.isna(cci) and cci < -100:
        signals.append(f"CCI超卖({cci:.0f})")
    if cross == 1:
        signals.append("EMA金叉")
    if cross == -1:
        signals.append("EMA死叉")
    if latest.get("ma_bull", 0):
        signals.append("多头排列")
    if latest.get("ma_bear", 0):
        signals.append("空头排列")

    result["hourly_signals"] = " | ".join(signals) if signals else "无明显信号"

    # 最近5根K线趋势
    last5_close = df.tail(5)["close"].values
    if len(last5_close) >= 2:
        result["last_5_trend"] = "上涨" if last5_close[-1] > last5_close[0] else "下跌" if last5_close[-1] < last5_close[0] else "震荡"
    else:
        result["last_5_trend"] = "N/A"

    return result
