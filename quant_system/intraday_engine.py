# 已融合: 原 quant_v6/engine/intraday_engine.py 移植为 quant_system 自包含实现（2026-08-08）
# -*- coding: utf-8 -*-
"""
intraday_engine.py — 盘中分时扫描引擎 (V6.1, T1: 分钟线盘中决策闭环)

职责:
    1. 盘中(9:30-15:00)用**分钟线**并发扫描候选池, 计算分时信号
    2. 输出与日K扫描同构的机会列表, 供 opportunity_monitor 盘中推送
    3. 收盘后由调用方切回日K扫描 (本模块只做盘中)

数据源: 新浪分钟线 (Vultr 实测可用; 东财接口被 Vultr IP 风控)。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from quant_system.intraday_signal import compute_intraday_signals

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))

_MAX_WORKERS = 12  # 并发拉取 (Vultr 1 核; 实测 12 并发 351/351 全成功, 更高触发新浪限流)


def fetch_minute_kline_sina(symbol: str, scale: int = 5, datalen: int = 60) -> Optional[pd.DataFrame]:
    """拉取新浪分钟 K 线。

    Args:
        symbol: 6 位股票代码。
        scale: 分钟周期 5/15/30/60。
        datalen: K 线根数 (新浪最多 ~1023)。

    Returns:
        DataFrame (open/high/low/close/volume, 时间升序)；失败返回 None。
    """
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]  # workspace
        qs_path = str(root / "quant_system")
        if qs_path not in sys.path:
            sys.path.insert(0, qs_path)
        # quant_system 是包, 必须用包导入 (直接 import 会因内部相对导入报错)
        from quant_system.sources_sina_intraday import fetch_sina_intraday
        df = fetch_sina_intraday(symbol, scale=scale, datalen=datalen)
        if df is None or len(df) < 30:
            return None
        need = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in need):
            logger.warning("新浪分钟K缺列 %s: %s", symbol, list(df.columns))
            return None
        return df[need].copy()
    except Exception as exc:
        logger.warning("拉取新浪分钟K失败 %s(%dmin): %s", symbol, scale, str(exc)[:100])
        return None


def scan_intraday(symbols: list[str], min_score: float = 2.5, scale: int = 5,
                  quotes: Optional[dict[str, dict]] = None) -> dict[str, Any]:
    """盘中分时扫描。

    Args:
        symbols: 候选股票代码列表。
        min_score: 最低信号分。
        scale: 分钟周期 (默认 5 分钟, 盘中信号灵敏)。
        quotes: {symbol: 行情dict} 可选, 复用调用方行情 (含实时价/名称等)。

    Returns:
        {
          "results": [机会 dict, 与日K扫描同构],
          "summary": {"total": n, "kline_ok": 成功数, "results_count": 命中数, "elapsed_s": s},
        }
    """
    t0 = time.time()
    quotes = quotes or {}
    results: list[dict] = []
    kline_ok = 0

    def _work(sym: str) -> Optional[dict]:
        df = fetch_minute_kline_sina(sym, scale=scale)
        if df is None:
            return None
        q = quotes.get(sym, {})
        price = q.get("price")
        ind = compute_intraday_signals(df, symbol=sym, price=price)
        if not ind:
            return None
        if ind.get("signal_count", 0) < min_score:
            return None
        return {
            "symbol": sym,
            "name": q.get("name", sym),
            "price": ind.get("price"),
            "change_pct": q.get("change_pct"),
            "market_cap_yi": q.get("market_cap_yi"),
            "pe_ttm": q.get("pe_ttm"),
            "pb": q.get("pb"),
            "tier": q.get("tier"),
            "sector": q.get("sector"),
            "signal_count": ind.get("signal_count"),
            "signal_summary": " | ".join(list(ind.get("signal_reasons", {}).values())[:6]),
            "signal_reasons": ind.get("signal_reasons"),
            "signals": ind.get("signals"),
            "rsi": ind.get("rsi"),
            "cci": ind.get("cci"),
            "macd_hist": ind.get("macd_hist"),
            "vol_ratio": ind.get("vol_ratio"),
            "bars": ind.get("bars"),
            "mode": "intraday",
            "period_min": scale,
        }

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futs = {ex.submit(_work, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
                    kline_ok += 1
            except Exception as exc:
                logger.warning("盘中扫描单只失败 %s: %s", futs[fut], str(exc)[:80])

    results.sort(key=lambda r: (-r["signal_count"], r.get("rsi", 50)))
    elapsed = time.time() - t0
    summary = {
        "total": len(symbols),
        "kline_ok": kline_ok,
        "results_count": len(results),
        "elapsed_s": round(elapsed, 1),
        "scale_min": scale,
    }
    return {"results": results, "summary": summary}


def is_trading_time(now: Optional[datetime] = None) -> bool:
    """是否盘中 (9:30-11:30 / 13:00-15:00, 工作日)。"""
    now = now or datetime.now(CST)
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return (930 <= hm <= 1130) or (1300 <= hm <= 1500)


def is_after_close(now: Optional[datetime] = None) -> bool:
    """是否已收盘 (>= 15:00, 工作日)。"""
    now = now or datetime.now(CST)
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return hm >= 1500


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    now = datetime.now(CST)
    print("当前:", now.strftime("%Y-%m-%d %H:%M"), "| 盘中:", is_trading_time(now),
          "| 已收盘:", is_after_close(now))
    sample = ["600519", "000858", "601899", "300750", "600036"]
    res = scan_intraday(sample, min_score=2.0)
    print("扫描:", res["summary"])
    for r in res["results"][:5]:
        print(f"  {r['symbol']} {r['name']}: score={r['signal_count']} | {r['signal_summary'][:60]}")
