#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全池回测引擎（2026-08-22 —— 用户"回测量化策略要多写"）

扫 data_warehouse/kline 全池 → 动量/三策略/合流选股:
  1. momentum_scan():   20/60日动量 Top/Flop
  2. strategy_panel():  核心票三策略信号(均线金叉/RSI超卖/动量突破)
  3. pooled_picks():    策略合流选股(均线多头+动量正)
纯本地kline, 带缓存1h。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KL = ROOT / "data_warehouse" / "kline"
CACHE = ROOT / "generated" / "backtest_pool_cache.json"


def _load(sym):
    import pandas as pd
    p = KL / f"{sym}.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def momentum_scan(limit: int = 400) -> dict:
    rows = []
    for f in sorted(KL.glob("*.parquet"))[:limit]:
        sym = f.stem
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
            c = df["close"].astype(float)
            if len(c) < 61:
                continue
            mom20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100
            mom60 = (c.iloc[-1] / c.iloc[-61] - 1) * 100
            rows.append({"symbol": sym, "mom20": round(mom20, 2), "mom60": round(mom60, 2)})
        except Exception:
            continue
    rows.sort(key=lambda x: -x["mom60"])
    return {"top": rows[:12], "flop": rows[-12:][::-1], "n": len(rows)}


def strategy_panel(core_symbols=None) -> dict:
    syms = core_symbols or ["600519", "000001", "000858", "600036", "601318"]
    out = {}
    for sym in syms:
        df = _load(sym)
        if df is None or len(df) < 40:
            continue
        c = df["close"].astype(float)
        signals = []
        ma5 = c.tail(5).mean()
        ma10 = c.tail(10).mean()
        if ma5 > ma10:
            signals.append("均线多头")
        delta = c.diff().tail(15)
        up = delta.clip(lower=0).mean()
        dn = (-delta.clip(upper=0)).mean()
        rsi = 100 - 100 / (1 + up / dn) if dn else 50
        if rsi < 35:
            signals.append("RSI超卖(%d)" % rsi)
        elif rsi > 70:
            signals.append("RSI超买(%d)" % rsi)
        if c.iloc[-1] >= c.tail(21).max():
            signals.append("20日新高")
        out[sym] = {"signals": signals, "close": round(float(c.iloc[-1]), 2),
                    "mom20": round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2) if len(c) > 21 else 0}
    return out


def pooled_picks(limit: int = 400) -> dict:
    picks = []
    for f in sorted(KL.glob("*.parquet"))[:limit]:
        sym = f.stem
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
            c = df["close"].astype(float)
            if len(c) < 30:
                continue
            ma5 = c.tail(5).mean()
            ma10 = c.tail(10).mean()
            mom20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else 0
            if ma5 > ma10 and mom20 > 0:
                picks.append({"symbol": sym, "mom20": round(mom20, 1),
                              "ma_spread": round((ma5 / ma10 - 1) * 100, 2)})
        except Exception:
            continue
    picks.sort(key=lambda x: -x["ma_spread"])
    return {"picks": picks[:15], "count": len(picks)}


def _build_or_cache():
    """缓存校验: mom非空才用缓存(2026-08-22 修空缓存)."""
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 3600:
        try:
            r = json.loads(CACHE.read_text(encoding="utf-8"))
            if r.get("mom", {}).get("n"):
                return r
        except Exception:
            pass
    r = {"mom": momentum_scan(), "panel": strategy_panel(), "picks": pooled_picks()}
    CACHE.write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")
    return r


def backtest_pool_md() -> str:
    r = _build_or_cache()
    L = ["## 📈 全池回测策略（动量/三策略/合流选股）"]
    if not r:
        return "## 📈 回测\n- 数据不足"
    mo = r.get("mom", {})
    if mo.get("top"):
        _tp = "、".join("%s(%+.0f%%)" % (x["symbol"], x["mom60"]) for x in mo["top"][:8])
        L.append("- 动量Top(60日): " + _tp)
    if mo.get("flop"):
        _tf = "、".join("%s(%+.0f%%)" % (x["symbol"], x["mom60"]) for x in mo["flop"][:5])
        L.append("- 动量Flop: " + _tf)
    pc = r.get("picks", {})
    if pc.get("picks"):
        _pk = "、".join("%s(%+.0f%%)" % (x["symbol"], x["mom20"]) for x in pc["picks"][:6])
        L.append("- 策略合流选股(%s只): %s" % (pc.get("count"), _pk))
    pa = r.get("panel", {})
    if pa:
        L.append("- 核心票策略信号:")
        for sym, sv in list(pa.items())[:4]:
            _sg = "、".join(sv["signals"]) or "无信号"
            L.append("  - %s: %s %s (mom20 %+.1f%%)" % (sym, sv["close"], _sg, sv["mom20"]))
    return "\n".join(L)


def backtest_data() -> dict:
    return _build_or_cache()


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(backtest_pool_md())