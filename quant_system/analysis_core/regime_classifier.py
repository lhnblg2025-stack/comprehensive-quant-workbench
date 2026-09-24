"""
regime_classifier — 市场机制识别（V11 自适应规则层）

识别 趋势/震荡/高波 三种机制（用户硬规则: 趋势衡量用 300MA/144MA，
MA20/MA60 仅作短线辅助），供 battle_map 自适应规则:

  趋势市:  顺势策略可用，打板/接力容忍度正常
  震荡市:  止盈止损线收紧一半（-5% → -2.5%），打板禁用只做低吸
  高波市:  打板全部禁用，只允许低吸；仓位系数 ×0.7
  冰点/退潮 + 高波 = 全面防守

指标:
  年化波动率(20日)  >45% → 高波, <20% → 低波
  close vs MA300 乖离 + MA300 斜率 → 趋势/震荡

数据: 沪深300日线（akshare stock_zh_index_daily，缓存 index_daily.parquet 增量更新）

输出: generated/regime_{date}.json + regime_history.parquet

用法:
  python3 -m quant_system.analysis_core.regime_classifier [--refresh] [--date 2026-08-07]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402

CST = timezone(timedelta(hours=8))
INDEX_DAILY = MARKET_DIR / "index_daily.parquet"
REGIME_HIST = MARKET_DIR / "regime_history.parquet"

HIGH_VOL, LOW_VOL = 0.45, 0.20
TREND_BAND = 0.06      # 乖离带: |close/MA300-1| < 6% 视为围绕均线
TREND_SLOPE = 0.0003   # MA300 日均斜率阈值（0.03%/日）


def ensure_index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """沪深300日线，本地缓存 + 增量更新。"""
    df = None
    if INDEX_DAILY.exists():
        df = pd.read_parquet(INDEX_DAILY)
        df["date"] = pd.to_datetime(df["date"])
    last = df["date"].max() if df is not None else None
    need = last is None or last.date() < datetime.now(CST).date() - timedelta(days=3)
    if need:
        try:
            import akshare as ak
            raw = ak.stock_zh_index_daily(symbol=symbol)
            raw = raw.rename(columns={"date": "date", "close": "close"})
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            if df is not None:
                raw = pd.concat([df[df["date"] < raw["date"].min()], raw], ignore_index=True)
            raw = raw.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            raw.to_parquet(INDEX_DAILY, index=False)
            df = raw
            print(f"[regime] 指数日线更新: {len(df)} 行 (至 {df['date'].max().date()})")
        except Exception as e:
            print(f"[regime] 指数日线更新失败({str(e)[:80]}), 用本地缓存")
    if df is None or df.empty:
        raise FileNotFoundError("指数日线不可用: 先手动 python3 -c 调 akshare 或检查网络")
    return df


def classify(date: str | None = None, refresh: bool = False) -> dict:
    if refresh or not INDEX_DAILY.exists():
        df = ensure_index_daily()
    else:
        df = pd.read_parquet(INDEX_DAILY)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    requested = pd.Timestamp(date) if date else df["date"].max()
    data_last = df["date"].max()
    # 防伪造: 目标日期晚于数据实际日期时，拒绝生成"当日"结果（防前视/伪当日）
    stale = data_last.date() < requested.date()
    target = data_last if stale else requested
    df = df[df["date"] <= target]
    if len(df) < 320:
        raise ValueError(f"指数日线不足320行: {len(df)}")

    close = df["close"]
    ret = close.pct_change().dropna()
    vol20 = float(ret.tail(20).std() * np.sqrt(252))

    ma300 = close.rolling(300).mean()
    ma144 = close.rolling(144).mean()
    slope300 = (ma300.iloc[-1] / ma300.iloc[-21] - 1) if len(ma300) > 21 and ma300.iloc[-21] > 0 else 0.0
    bias300 = close.iloc[-1] / ma300.iloc[-1] - 1

    # 机制判定
    tags = []
    if bias300 > TREND_BAND and slope300 > TREND_SLOPE:
        tags.append("趋势市")
    elif abs(bias300) <= TREND_BAND:
        tags.append("震荡市")
    elif bias300 < -TREND_BAND and slope300 < -TREND_SLOPE:
        tags.append("下跌趋势")
    else:
        tags.append("过渡市")

    if vol20 > HIGH_VOL:
        tags.append("高波")
    elif vol20 < LOW_VOL:
        tags.append("低波")
    else:
        tags.append("中波")

    regime = "+".join(tags)
    # 规则注入
    rules = {"打板": "允许" if "震荡" not in regime and "高波" not in regime else "禁用",
             "低吸": "允许",
             "止损线": "-2.5%" if "震荡" in regime else ("-3.5%" if "高波" in regime else "-5%"),
             "仓位系数": 0.7 if "高波" in regime else 1.0,
             "趋势策略": "顺势" if "趋势" in regime else "观望"}

    res = {
        "date": str(target.date()),
        "requested_date": str(requested.date()) if date else str(target.date()),
        "stale": stale,
        "regime": regime,
        "vol20_annual": round(vol20 * 100, 1),
        "bias_ma300": round(bias300 * 100, 2),
        "slope_ma300": round(slope300 * 100, 2),
        "close": float(close.iloc[-1]),
        "ma300": float(ma300.iloc[-1]),
        "ma144": float(ma144.iloc[-1]),
        "rules": rules,
    }
    out = ROOT / "generated" / f"regime_{res['date']}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 历史累积
    row = pd.DataFrame([{**res, "rules": json.dumps(rules, ensure_ascii=False)}])
    if REGIME_HIST.exists():
        old = pd.read_parquet(REGIME_HIST)
        row = pd.concat([old, row], ignore_index=True)
    row = row.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    row.to_parquet(REGIME_HIST, index=False)
    return res


def get_regime(date: str | None = None) -> dict:
    """供 battle_map 调用的轻量接口（无网络，只读缓存）。"""
    if REGIME_HIST.exists():
        df = pd.read_parquet(REGIME_HIST)
        if date:
            # 审计 2026-08-16：历史查询应取 <=date 的最近一条，禁止回退到最新未来机制
            hit = df[df["date"] == date]
            if not hit.empty:
                r = hit.iloc[-1].to_dict()
                r["rules"] = json.loads(r["rules"]) if isinstance(r.get("rules"), str) else r.get("rules", {})
                r["stale"] = False
                return r
            hist = df[pd.to_datetime(df["date"]) <= pd.to_datetime(date)]
            if not hist.empty:
                r = hist.iloc[-1].to_dict()
                r["rules"] = json.loads(r["rules"]) if isinstance(r.get("rules"), str) else r.get("rules", {})
                r["stale"] = True
                r["requested_date"] = date
                return r
            return {"date": date, "regime": "未知", "rules": {}, "stale": True,
                    "note": "无 <= 指定日期的机制缓存"}
        if len(df):
            r = df.iloc[-1].to_dict()
            r["rules"] = json.loads(r["rules"]) if isinstance(r.get("rules"), str) else r.get("rules", {})
            return r
    return {"date": date or "", "regime": "未知", "rules": {}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="市场机制识别")
    ap.add_argument("--refresh", action="store_true", help="强制更新指数日线")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    r = classify(args.date, refresh=args.refresh)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
