from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .indicators import add_technical_indicators


def add_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    return add_technical_indicators(df, config)


def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = add_indicators(df, config)
    above_slow = out["close"] > out["ma_slow"]
    above_trend = out["close"] > out["ma_trend"]
    fast_above_slow = out["ma_fast"] > out["ma_slow"]
    volume_ok = out["volume_ratio"].fillna(1.0) >= 1.0
    # P2-Q9-fix (Q9-M534): indicators.py 中 highest_since_60 已 shift(1)
    # （close.rolling(60).max().shift(1)），此处再 shift(1) 构成双重 shift，
    # 突破条件实际比较 2 日前的 60 日高点，信号系统性滞后 1 日。统一约定：
    # 指标层只 shift 一次，调用方不再 shift。
    breakout = out["close"] >= out["highest_since_60"]

    out["buy_setup"] = above_slow & above_trend & fast_above_slow & (volume_ok | breakout)
    out["sell_setup"] = (out["close"] < out["ma_slow"]) | (out["ma_fast"] < out["ma_slow"])
    out["signal"] = 0
    out.loc[out["buy_setup"], "signal"] = 1
    out.loc[out["sell_setup"], "signal"] = -1
    return out


def latest_signal(symbol: str, df: pd.DataFrame, config: StrategyConfig) -> dict:
    sig = generate_signals(df, config).dropna(subset=["ma_slow", "ma_trend"])
    if sig.empty:
        return {"symbol": symbol, "status": "insufficient_data"}
    row = sig.iloc[-1]
    action = "BUY_WATCH" if row["signal"] == 1 else "SELL_OR_AVOID" if row["signal"] == -1 else "HOLD"
    return {
        "symbol": symbol,
        "date": str(row["date"].date()),
        "close": round(float(row["close"]), 3),
        "action": action,
        "ma_fast": round(float(row["ma_fast"]), 3),
        "ma_slow": round(float(row["ma_slow"]), 3),
        "ma_trend": round(float(row["ma_trend"]), 3),
        "volume_ratio": round(float(row.get("volume_ratio", 1.0)), 2),
        "rsi_14": round(float(row.get("rsi_14", 0.0)), 2),
        "macd_hist": round(float(row.get("macd_hist", 0.0)), 3),
        "atr_pct": round(float(row.get("atr_pct", 0.0)), 2),
        "trend_score": round(float(row.get("trend_score", 0.0)), 1),
        "momentum_score": round(float(row.get("momentum_score", 0.0)), 1),
        "risk_score": round(float(row.get("risk_score", 0.0)), 1),
        "composite_score": round(float(row.get("composite_score", 0.0)), 1),
    }
