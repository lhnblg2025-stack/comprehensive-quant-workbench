#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""波浪理论·大盘结构判断（2026-08-22 —— 用户"波浪理论对应缠论")

用月线/日线高低点识别波浪结构(5浪推进/3浪调整)，输出三剧本(替代缠论三剧本):
  - 浪结构: 数月线枢轴高低点 → 当前处于推进浪(1/3/5)或调整浪(2/4)
  - 三剧本: 强势(突破前高继续5浪) / 中性(区间震荡) / 弱势(破前低走调整C)
  - 大盘适配度: 均线排列(MA5/10/20/60) + 缺口 + 位置

纯本地K线, 秒级。待用户发缠论图后可再接入精确缠论数据。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MK = ROOT / "data_warehouse" / "market"


def _load_index(name: str = "沪深300") -> Optional[object]:
    import pandas as pd
    p = MK / f"index_daily_{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _monthly_pivots(df, lookback: int = 40) -> list:
    """月线枢轴高低点(简化: 月度高低点序列做波浪标记)。"""
    import pandas as pd
    m = df.set_index("date").resample("ME").agg(
        close=("close", "last"), high=("high", "max"), low=("low", "min")).dropna()
    m = m.tail(lookback)
    closes = m["close"].tolist()
    highs = m["high"].tolist()
    lows = m["low"].tolist()
    # 近似波浪: 识别显著高低点(相对前后±3%)
    pivots = []
    for i in range(2, len(closes) - 2):
        c = closes[i]
        if c >= max(closes[i-2:i]) * 1.02 or c <= min(closes[i-2:i]) * 0.98:
            pivots.append((m.index[i].strftime("%Y-%m"), round(c, 1),
                           "高" if c >= max(closes[i-2:i]) * 1.02 else "低"))
    return pivots


def wave_structure() -> dict:
    """波浪结构识别 + 三剧本 + 适配度。"""
    df = _load_index("沪深300")
    if df is None or len(df) < 120:
        return {"error": "指数数据缺失"}
    import pandas as pd
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    ma5, ma10, ma20, ma60 = (float(c.tail(n).mean()) for n in (5, 10, 20, 60))
    # 关键位
    recent = df.tail(60)
    high60 = float(recent["high"].max())
    low60 = float(recent["low"].min())
    # 波浪位置近似: 近期高低点幅度
    pivot = _monthly_pivots(df)
    # 最近枢轴方向
    recent_dir = "推进(5浪?)" if last >= high60 * 0.99 else ("调整(3浪C?)" if last <= low60 * 1.01 else "区间")
    # 距关键位
    dist_high = (high60 - last) / high60 * 100
    dist_low = (last - low60) / low60 * 100
    # 三剧本
    bull_trigger = high60 * 0.995
    weak_trigger = low60 * 1.005
    scripts = [
        {"剧本": "强势(5浪延续)", "触发": f"突破{high60:.0f}", "含义": "趋势推进, 持股"},
        {"剧本": "中性(区间调整B)", "触发": f"{low60:.0f}-{high60:.0f}震荡", "含义": "高抛低吸"},
        {"剧本": "弱势(调整浪C)", "触发": f"跌破{low60:.0f}", "含义": "降仓防御"},
    ]
    # 适配度
    bullish_ma = ma5 > ma10 > ma20
    above_ma60 = last > ma60
    outlook = "偏强" if (bullish_ma and above_ma60) else ("偏弱" if (ma5 < ma10 < ma20) else "震荡")
    return {
        "index": "沪深300", "last": round(last, 1),
        "ma_struct": f"MA5 {ma5:.0f}/MA10 {ma10:.0f}/MA20 {ma20:.0f}/MA60 {ma60:.0f}",
        "pivot": pivot[-6:], "position": recent_dir,
        "high60": round(high60, 1), "low60": round(low60, 1),
        "dist_high_pct": round(dist_high, 1), "dist_low_pct": round(dist_low, 1),
        "scripts": scripts, "outlook": outlook,
        "note": "波浪理论近似(月线枢轴), 待用户缠论图精确替换",
    }


def wave_all_indices() -> list[dict]:
    """多指数波浪结构总览(上证50/沪深300/中证500/创业板/科创50/中证1000)。"""
    out = []
    for nm in ("上证50", "沪深300", "中证500", "创业板指", "科创50", "中证1000"):
        df = _load_index(nm)
        if df is None or len(df) < 120:
            continue
        import pandas as pd
        c = df["close"].astype(float)
        last = float(c.iloc[-1])
        ma5, ma20, ma60 = (float(c.tail(n).mean()) for n in (5, 20, 60))
        r60 = df.tail(60)
        h60, l60 = float(r60["high"].max()), float(r60["low"].min())
        pos = (last - l60) / (h60 - l60) * 100 if h60 > l60 else 50
        # 结构: 均线多头+趋势
        struct = "多头(5>20>60)" if (ma5 > ma20 > ma60) else ("空头" if (ma5 < ma20 < ma60) else "震荡")
        out.append({"指数": nm, "收盘": round(last, 1), "结构": struct,
                    "60日分位": round(pos, 0), "MA5/20": f"{ma5:.0f}/{ma20:.0f}",
                    "高60": round(h60, 1), "低60": round(l60, 1)})
    return out


def wave_all_md() -> str:
    """多指数波浪总览 → Markdown。"""
    L = ["## 🌊 波浪理论多指数总览"]
    for x in wave_all_indices():
        L.append(f"- {x['指数']}: {x['收盘']} {x['结构']} 60日分位{x['60日分位']:.0f}% (高{x['高60']}/低{x['低60']})")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    import json
    print(json.dumps(wave_structure(), ensure_ascii=False, indent=1)[:800])
    print()
    print(wave_all_md())