"""
分析工具箱 — 给助手自己调用的高层分析函数。

用法：在对话中调用这些函数做深度分析，不要自己拼数据管道。
"""

from __future__ import annotations
import logging

import time
from datetime import datetime
from typing import Any

import pandas as pd

from .config import DEFAULT_STRATEGY
from .data import fetch_daily, resolve_symbol, search_stock_name
from .global_market import (
    fetch_global_quotes,
    fetch_global_kline_with_indicators,
    resolve_global_symbol,
    HK_KNOWN,
    US_KNOWN,
)
from .indicators import add_technical_indicators, latest_indicator_snapshot
from .margin import fetch_margin_summary, fetch_margin_individual
from .signals import latest_signal
from .realtime import fetch_realtime
# P2-Q27-fix(L348): 移除未使用的 from .risk import enrich_trade_plan 导入

# ── 工具函数 ──────────────────────────────────────────────────────


def is_trading_hours() -> bool:
    """A股是否在交易时段 (9:30-11:30 和 13:00-15:00, 周一到周五)。"""
    now = datetime.now().astimezone()
    if now.weekday() >= 5:  # 周末
        return False
    t = now.hour * 60 + now.minute
    # V4.1 fix: 排除午休 11:30-13:00
    morning = 9 * 60 + 30 <= t <= 11 * 60 + 30
    afternoon = 13 * 60 <= t <= 15 * 60
    return morning or afternoon


def fmt_price(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if v < 1000 else f"{v:.1f}"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}%"


def fmt_amount(v: float | None) -> str:
    if v is None or v == 0:
        return "-"
    if v >= 1e8:
        return f"{v / 1e8:.1f}亿"
    if v >= 1e4:
        return f"{v / 1e4:.0f}万"
    return f"{v:.0f}"


# ── 单股深度分析 ──────────────────────────────────────────────────


def analyze_stock_daily(
    symbol: str,
    period: str = "daily",
    lookback: int = 120,
) -> dict[str, Any]:
    """单只股票日线深度分析。

    返回完整快照 + 技术指标 + 信号 + 两融。
    """
    sym = resolve_symbol(symbol)
    name = ""
    try:
        results = search_stock_name(sym)
        name = results[0]["name"] if results else sym
    except Exception as e:
        logging.getLogger(__name__).error(f"[analysis_toolkit] 操作失败: {e}", exc_info=True)

    # 日线数据 + 指标
    from datetime import datetime as _dt, timedelta as _td
    end_str = _dt.now().strftime("%Y%m%d")
    # 动态起点（近 2 年），避免固定日期导致回看窗口随时间缩窄
    start_str = (_dt.now() - _td(days=365 * 2)).strftime("%Y%m%d")
    df = fetch_daily(sym, start=start_str, end=end_str, use_cache=False)
    if df.empty:
        start_retry = (_dt.now() - _td(days=365 * 3)).strftime("%Y%m%d")
        df = fetch_daily(sym, start=start_retry, end=end_str, use_cache=False)
    if df.empty:
        return {"ok": False, "error": f"无法获取 {sym} 的数据"}

    df_indicators = add_technical_indicators(df, DEFAULT_STRATEGY)
    snapshot = latest_indicator_snapshot(sym, df_indicators, DEFAULT_STRATEGY)
    signal = latest_signal(sym, df_indicators, DEFAULT_STRATEGY)

    # 两融（如果有）
    margin = fetch_margin_individual(sym)

    # 最近 N 条
    rows = df_indicators.tail(lookback)
    latest = rows.iloc[-1].to_dict() if len(rows) > 0 else {}

    return {
        "ok": True,
        "symbol": sym,
        "name": name,
        "snapshot": snapshot,
        "signal": signal,
        "margin": margin,
        "latest_bar": {
            "date": str(latest.get("date", "")),
            "open": latest.get("open"),
            "close": latest.get("close"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "volume": latest.get("volume"),
            "amount": latest.get("amount"),
            "pct_chg": latest.get("pct_chg"),
        },
        "trend": {
            "ma20": latest.get("ma_fast"),
            "ma60": latest.get("ma_slow"),
            "ma144": latest.get("ma_trend"),
            "ma300": latest.get("ma_long_trend"),
        },
        "momentum": {
            "macd_dif": latest.get("macd_dif"),
            "macd_dea": latest.get("macd_dea"),
            "macd_hist": latest.get("macd_hist"),
            "rsi_14": latest.get("rsi_14"),
            "kdj_j": latest.get("kdj_j"),
        },
        "bollinger": {
            "upper": latest.get("boll_upper"),
            "mid": latest.get("boll_mid"),
            "lower": latest.get("boll_lower"),
        },
        "risk": {
            "atr_pct": latest.get("atr_pct"),
            "risk_score": latest.get("risk_score"),
            "drawdown_60": latest.get("drawdown_60_pct"),
        },
        "composite_score": latest.get("composite_score"),
        "n_bars": len(rows),
        "data_period": f"{rows.iloc[0]['date']} ~ {rows.iloc[-1]['date']}" if len(rows) >= 2 else "N/A",
    }


def analyze_global_stock_daily(
    symbol: str,
    count: int = 120,
) -> dict[str, Any]:
    """港美股单只日线分析，返回快照 + 指标。"""
    sym = resolve_global_symbol(symbol)
    df = fetch_global_kline_with_indicators(sym, count=count)
    if df.empty:
        return {"ok": False, "error": f"无法获取 {sym} 的全球K线数据"}

    rows = df.tail(min(count, len(df)))
    latest = rows.iloc[-1].to_dict() if len(rows) > 0 else {}

    return {
        "ok": True,
        "symbol": sym,
        "latest_bar": {
            "date": str(latest.get("date", "")),
            "close": latest.get("close"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
        },
        "bollinger": {
            "upper": latest.get("boll_upper"),
            "mid": latest.get("boll_mid"),
            "lower": latest.get("boll_lower"),
        },
        "n_bars": len(rows),
        "data_period": f"{rows.iloc[0]['date']} ~ {rows.iloc[-1]['date']}" if len(rows) >= 2 else "N/A",
    }


# ── 市场全景 ─────────────────────────────────────────────────────


def analyze_market_overview() -> dict[str, Any]:
    """A 股市场全景：两融 + 风格指数（快速版）。"""
    margin = fetch_margin_summary()

    return {
        "margin": margin,
        "trading_hours": is_trading_hours(),
        "analysis_time": datetime.now().astimezone().isoformat(),
        "note": "指数行情需单独调 analyze_index_daily(code) 查询",
    }


# 常见 A 股指数代码（用于路由到指数行情通道）
_INDEX_CODES = {
    "000001", "000002", "000010", "000016", "000017", "000028", "000300",
    "000688", "000852", "000905", "000906", "000908", "000919", "000922",
    "399001", "399005", "399006", "399101", "399106", "399300", "399905",
    "399950",
}


def _is_index_code(code: str) -> bool:
    """判断是否是指数代码。

    P1-Q27-fix: 指数代码不能走 fetch_daily 个股通道
    （000300→sz000300 无数据、000905→厦门港务 个股），
    需路由到 stock_zh_index_daily 指数通道。
    """
    code = str(code).zfill(6)
    return code in _INDEX_CODES or code.startswith("399")


def _fetch_index_daily(code: str):
    """通过新浪指数通道获取指数日线。"""
    import akshare as ak
    prefix = "sz" if str(code).zfill(6).startswith("399") else "sh"
    df = ak.stock_zh_index_daily(symbol=f"{prefix}{code}")
    if df is None or df.empty:
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out["pct_chg"] = out["close"].pct_change() * 100
    return out


def analyze_index_daily(code: str = "000300") -> dict[str, Any]:
    """单只指数日线分析。

    Parameters
    ----------
    code : str — 指数代码，如 000300(沪深300), 000905(中证500)
    """
    from datetime import datetime as _dt, timedelta as _td
    end_str = _dt.now().strftime("%Y%m%d")
    # P2-Q27-fix(L347): 原硬编码 start="20260101"，年份漂移后数据量失控；
    # 改为相对窗口（约 2 个自然年 ≈ 2×252 交易日 + 冗余）。
    start_str = (_dt.now() - _td(days=730)).strftime("%Y%m%d")
    try:
        if _is_index_code(code):
            # P1-Q27-fix: 指数走新浪指数通道，不再经 fetch_daily 个股通道
            df = _fetch_index_daily(code)
            if df is None or df.empty:
                return {"ok": False, "error": f"指数 {code} 无数据"}
        else:
            df = fetch_daily(code, start=start_str, end=end_str, use_cache=False)
            if df.empty:
                return {"ok": False, "error": "无数据"}
        c = float(df.iloc[-1]["close"])
        pct = (c / float(df.iloc[-2]["close"]) - 1) * 100 if len(df) >= 2 else 0
        df_ind = add_technical_indicators(df, DEFAULT_STRATEGY)
        snap = latest_indicator_snapshot(code, df_ind, DEFAULT_STRATEGY)
        return {
            "ok": True,
            "code": code,
            "channel": "index" if _is_index_code(code) else "stock",
            "close": c,
            "pct_chg": pct,
            "n_bars": len(df),
            "range": f"{df.iloc[0]['date']} ~ {df.iloc[-1]['date']}",
            "ma20": snap.get("ma_fast") if snap else None,
            "ma60": snap.get("ma_slow") if snap else None,
            "boll_upper": snap.get("boll_upper") if snap else None,
            "boll_lower": snap.get("boll_lower") if snap else None,
            "rsi14": snap.get("rsi_14") if snap else None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:80]}


def analyze_hk_us_overview() -> dict[str, Any]:
    """港美股主要行情快照。"""
    hk_symbols = list(HK_KNOWN.keys())[:10]
    us_symbols = list(US_KNOWN.keys())[:10]
    all_symbols = [f"hk{s}" for s in hk_symbols] + [f"us{s}" for s in us_symbols]

    quotes = fetch_global_quotes(all_symbols)
    result = []
    for sym, q in sorted(quotes.items()):
        result.append({
            "symbol": sym,
            "name": q.name,
            "price": q.price,
            "change_pct": q.change_pct,
            "volume": q.volume,
        })
    return {"quotes_count": len(result), "quotes": result}


# ── 盘中快照 ─────────────────────────────────────────────────────


def analyze_realtime_watchlist() -> dict[str, Any]:
    """盘中自选股实时行情（仅交易时段可用）。"""
    watchlist = ["600519", "000858", "002714", "000333", "300750",
                 "601318", "600036", "000001", "002594", "600900"]
    results = []
    for sym in watchlist:
        try:
            data = fetch_realtime(sym, timeout=8)
            if data:
                results.append(data)
        except Exception as e:
            logging.getLogger(__name__).error(f"[analysis_toolkit] 操作失败: {e}", exc_info=True)

    return {
        "trading_hours": is_trading_hours(),
        "count": len(results),
        "data": results,
        "time": datetime.now().astimezone().isoformat(),
    }


# ── 用户可调用的函数清单（给助手自己看）───────────────────

ANALYSIS_FUNCTIONS = {
    # P2-Q27-fix(M346): 移除本模块未定义的 decide_stock/scan_market/format_decision/
    # assess_market_triple_screen/calc_position_size 条目（实测 hasattr=False，
    # 误导助手调用不存在函数），并补入实际存在的 analyze_index_daily。
    "analyze_stock_daily(symbol, lookback=120)":
        "A股单股日线深度分析（技术指标+信号+两融+BOLL）",
    "analyze_index_daily(code='000300')":
        "单只指数日线分析（沪深300/中证500等指数代码）",
    "analyze_global_stock_daily(symbol, count=120)":
        "港美股日线分析（含BOLL）",
    "analyze_market_overview()":
        "A股市场全景（两融+指数风格）",
    "analyze_hk_us_overview()":
        "港美股主要行情快照",
    "analyze_realtime_watchlist()":
        "盘中自选股实时行情",
    "is_trading_hours()":
        "判断当前是否A股交易时段",
}

__all__ = [
    "analyze_stock_daily",
    "analyze_index_daily",
    "analyze_global_stock_daily",
    "analyze_market_overview",
    "analyze_hk_us_overview",
    "analyze_realtime_watchlist",
    "is_trading_hours",
    "ANALYSIS_FUNCTIONS",
]
