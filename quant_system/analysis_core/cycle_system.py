#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cycle_system — 体系11 市场周期系统（Appel 周期分割 / T形底部 / VIX 买入区 / 周期共振）

方法论（skills/appel-market-cycle-segmentation + appel-t-formation-market-cycle
        + appel-vix-buy-zone + internal-order-six-stage-cycle）:
  市场周期分割: 上涨/顶部/下跌/底部/筑底 五态
               MA20/MA60 排列 + MA20 斜率 + ROC20/ROC5 动量
  T形底部:     底部横盘（价格波动<5%达N日）+ 放量突破（量比>1.5）→ 'T形底突破'
  VIX 买入区:  VIX>30（或温度<30 替代）= 恐慌买入区（逆向）
               温度>85 + VIX<15 = 风险区（过热）
  周期共振:    多指数同阶段 → 市场周期一致（强信号）；分歧 → 结构分化
  综合:        阶段 × 共振 × VIX → 市场周期状态 + 策略建议
               （底部/筑底=布局 / 上涨=持有 / 顶部=减仓 / 下跌=防守）

输入:
  data_warehouse/market/index_daily_{沪深300,上证50,创业板指,科创50}.parquet
  data_warehouse/market/fusion.parquet            温度（VIX 缺失时的替代/风险区判据）
  generated/macro_overseas_{date}.json            VIX（可选，缺失自动跳过）
  data_warehouse/market/a_high_low.parquet        新高/新低宽度

统一接口:
  from quant_system.analysis_core.cycle_system import CycleSystem
  cs = CycleSystem()
  res  = cs.detect(date=None)      # 全量检测（市场级 + 指数级）
  path = cs.report(date=None)      # 写 generated/cycle_report_{date}.md
  view = cs.view(date=None)        # multi_agent 兼容
                                   # {agent:'市场周期', signal, view, confidence,
                                   #  evidence, weight, status, detail}

防御: 指数缺失跳过；单指数失败不崩；全部缺失降级为 degraded。

用法:
  python3 -m quant_system.analysis_core.cycle_system [--date 2026-08-10] [--report]
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent          # workspace（data_warehouse 所在）
REPO_ROOT = Path(__file__).resolve().parent.parent             # 仓库根（generated/ 输出目录）
sys.path.insert(0, str(ROOT))

# 离线优先：sentence-transformers 在本环境不可下载模型 → knowledge_rag 走关键词兜底
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.common import (  # noqa: E402
    fmt,
    norm_date,
    num,
    signal_view,
    
)
from quant_system.analysis_core.fusion import read_fusion_latest  # noqa: E402

CST = timezone(timedelta(hours=8))
DEFAULT_OUT_DIR = REPO_ROOT / "generated"

# ── 指数数据源（优先显式四指数，全部缺失时回退通用 index_daily.parquet）─────
INDEX_SOURCES: list[tuple[str, str]] = [
    ("沪深300", "index_daily_沪深300.parquet"),
    ("上证50", "index_daily_上证50.parquet"),
    ("创业板指", "index_daily_创业板指.parquet"),
    ("科创50", "index_daily_科创50.parquet"),
]
FALLBACK_INDEX = ("大盘指数", "index_daily.parquet")

LOOKBACK_DAYS = 420          # 日历回看窗口（≈300 交易日，覆盖 MA60 + 动量余量）
MIN_BARS = 80                # 周期阶段最少日线样本（MA60=60 + 斜率余量）
MIN_BARS_T = 65              # T形底检测最少样本

# ── 周期阶段参数 ─────────────────────────────────────────
SLOPE_UP = 0.05              # MA20 5日斜率阈值（%/日）
SLOPE_DOWN = -0.05
SLOPE_FLAT = 0.10            # |MA20斜率| 低于此视为走平
ROC_UP = 1.0                 # ROC20 动量阈值
ROC_DOWN = -1.0
ROC_FLAT = 5.0               # |ROC20| 低于此视为动量归零
NEAR_HIGH_PCT = 4.0          # 距 60日高点 <4% 视为高位
NEAR_LOW_PCT = 8.0           # 距 60日低点 <8% 视为低位
DEEP_LOW_PCT = 6.0           # 距 60日低点 <6%（筑底更严）
RANGE_FLAT_PCT = 5.0         # 20日波动幅度 <5% 视为横盘

# ── T形底部参数 ──────────────────────────────────────────
T_BASE_DAYS = 15             # 底部横盘窗口（交易日）
T_RANGE_PCT = 5.0            # 横盘期价格波动 <5%
T_VOL_RATIO = 1.5            # 突破日量比 >1.5（当日量/前5日均量）
T_NEAR_LOW_PCT = 8.0         # 横盘区间须在 60日低点 +8% 以内
T_SCAN_DAYS = 5              # 突破日回溯窗口（含当日，容忍短滞后）

# ── VIX 买入区参数 ───────────────────────────────────────
VIX_PANIC = 30.0             # VIX>30 → 恐慌买入区
TEMP_PANIC_SUB = 30.0        # VIX 缺失时 温度<30 → 恐慌买入区（温度替代）
VIX_RISK_LOW = 15.0          # VIX<15 且 温度>85 → 风险区
TEMP_RISK_HIGH = 85.0        # 高温阈值
TEMP_RISK_SUB = 85.0         # VIX 缺失时 温度>85 → 高温风险区（温度替代）

# ── RAG / 输出 ───────────────────────────────────────────
RAG_QUERY = "市场周期 筑底 T形底部 VIX"
RAG_K = 3
VIEW_WEIGHT = 1.0            # view() 权重（multi_agent 仲裁权重在 multi_agent.BASE_WEIGHTS 维护）

# ── 五态阶段（价格结构周期，区别于 emotion_system 的六阶段情绪循环）────────
STAGES = ["上涨", "顶部", "下跌", "底部", "筑底"]
STAGE_STRATEGY = {           # 阶段 × 策略建议（综合后可能被 VIX 区修正）
    "上涨": ("持有", "多"),
    "顶部": ("减仓", "空"),
    "下跌": ("防守", "空"),
    "底部": ("布局", "多"),
    "筑底": ("布局", "多"),
}
# 同分时的阶段优先级（更保守/趋势定义者优先）
STAGE_PRIORITY = {"下跌": 5, "上涨": 4, "顶部": 3, "底部": 2, "筑底": 1}

T_FORMATION_SIGNAL = "T形底突破"


# ════════════════════════════════════════════════════════════
# 基础工具
# ════════════════════════════════════════════════════════════
def _fmt_pct(v, nd: int = 2) -> str:
    try:
        f = float(v)
        return f"{f:.{nd}f}%" if np.isfinite(f) else "NA"
    except (TypeError, ValueError):
        return "NA"


# ════════════════════════════════════════════════════════════
# 数据加载（防御：缺失/异常一律返回 None 并标注原因）
# ════════════════════════════════════════════════════════════
def _load_index_df(path: Path, target: pd.Timestamp) -> pd.DataFrame | None:
    """指数日K（date/open/high/low/close/volume）≤target 截断规整。"""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if not {"date", "close"}.issubset(df.columns):
        return None
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= target]
    if df.empty:
        return None
    return df


def _load_fusion_temperature(date: str) -> tuple[float | None, str | None, int | None]:
    """fusion 温度 ≤date 最新一日。返回 (温度, as_of, 落后天数)。"""
    row = read_fusion_latest(date, ["temperature"])
    if row is None:
        return None, None, None
    temp = num(row.get("temperature"))
    if temp is None:
        return None, None, None
    as_of = row["date"]
    lag = max(0, (pd.Timestamp(date) - pd.Timestamp(as_of)).days)
    return temp, as_of, int(lag)


def _load_vix(date: str) -> tuple[float | None, str | None, str]:
    """VIX ≤date 最新 macro_overseas 报告。返回 (值, as_of, 状态/原因)。"""
    gen = REPO_ROOT / "generated"
    best: tuple[pd.Timestamp, Path] | None = None
    for f in sorted(gen.glob("macro_overseas_*.json")):
        try:
            d = pd.Timestamp(f.name[len("macro_overseas_"):-len(".json")])
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[cycle_system] 操作失败: {e}", exc_info=True)
            continue
        if d <= pd.Timestamp(date) and (best is None or d > best[0]):
            best = (d, f)
    if best is None:
        return None, None, "无 macro_overseas 报告"
    d, f = best
    as_of = d.strftime("%Y-%m-%d")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        vix = (data.get("assets") or {}).get("vix") or {}
        latest = vix.get("latest")
        if latest is None:
            reason = vix.get("reason") or vix.get("status") or "latest=null"
            return None, as_of, f"VIX不可用({reason[:60]})"
        return float(latest), as_of, "ok"
    except Exception as e:  # noqa: BLE001
        return None, as_of, f"VIX解析失败({str(e)[:60]})"


def _load_breadth(date: str) -> dict | None:
    """新高/新低宽度 ≤date 最新一日（a_high_low.parquet）。"""
    path = MARKET_DIR / "a_high_low.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        hist = df[df["date"] <= pd.Timestamp(date)].sort_values("date")
        if hist.empty:
            return None
        r = hist.iloc[-1]
        def _i(v, d=0):
            x = num(v, d)
            return int(x)
        high20, low20 = _i(r.get("high20")), _i(r.get("low20"))
        high60, low60 = _i(r.get("high60")), _i(r.get("low60"))
        return {
            "as_of": r["date"].strftime("%Y-%m-%d"),
            "high20": high20, "low20": low20,
            "high60": high60, "low60": low60,
            "ratio20": round((high20 + 1) / (low20 + 1), 3),
        }
    except Exception:  # noqa: BLE001
        return None


# ════════════════════════════════════════════════════════════
# 周期阶段识别（每指数: MA20/MA60 排列 + MA斜率 + ROC 动量 → 五态）
# ════════════════════════════════════════════════════════════
def _ma_slope(ma: pd.Series, n: int = 5) -> float:
    """MA 最近 n 日线性斜率（%/日，相对 MA 现值）。"""
    seg = ma.iloc[-n:].to_numpy(dtype=float)
    base = float(ma.iloc[-1])
    if not np.all(np.isfinite(seg)) or not np.isfinite(base) or base <= 0:
        return np.nan
    return float(np.polyfit(np.arange(n), seg, 1)[0]) / base * 100.0


def _stage_features(df: pd.DataFrame) -> dict:
    """周期阶段特征向量（不足样本返回 None 键）。"""
    c = pd.to_numeric(df["close"], errors="coerce")
    ma20 = c.rolling(20, min_periods=20).mean()
    ma60 = c.rolling(60, min_periods=60).mean()
    f: dict = {"ok": False}
    if len(c) < MIN_BARS:
        return f
    close_t, ma20_t, ma60_t = float(c.iloc[-1]), float(ma20.iloc[-1]), float(ma60.iloc[-1])
    if not all(np.isfinite(x) for x in (close_t, ma20_t, ma60_t)):
        return f
    slope = _ma_slope(ma20)
    roc20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan
    roc5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else np.nan
    hi20, lo20 = float(df["high"].iloc[-20:].max()), float(df["low"].iloc[-20:].min())
    range20 = (hi20 - lo20) / float(c.iloc[-20:].mean()) * 100
    lo60 = float(df["low"].iloc[-60:].min())
    hi60 = float(df["high"].iloc[-60:].max())
    dist_low60 = (close_t - lo60) / lo60 * 100 if lo60 > 0 else np.nan
    dist_high60 = (hi60 - close_t) / close_t * 100 if close_t > 0 else np.nan
    bull = close_t > ma20_t > ma60_t
    bear = close_t < ma20_t < ma60_t
    above20 = close_t >= ma20_t
    below_ma20 = close_t < ma20_t
    above60 = close_t >= ma60_t
    ma20_ge_ma60 = ma20_t >= ma60_t
    ma20_lt_ma60 = ma20_t < ma60_t
    f.update({
        "ok": True, "close": close_t, "ma20": ma20_t, "ma60": ma60_t,
        "slope": slope, "roc20": roc20, "roc5": roc5,
        "range20": range20, "dist_low60": dist_low60, "dist_high60": dist_high60,
        "bull": bull, "bear": bear, "above20": above20, "above60": above60,
        "below_ma20": below_ma20, "below_ma60": not above60,
        "ma20_ge_ma60": ma20_ge_ma60, "ma20_lt_ma60": ma20_lt_ma60,
        "slope_up": bool(np.isfinite(slope) and slope > SLOPE_UP),
        "slope_down": bool(np.isfinite(slope) and slope < SLOPE_DOWN),
        "slope_flat": bool(np.isfinite(slope) and abs(slope) < SLOPE_FLAT),
        "roc_up": bool(np.isfinite(roc20) and roc20 > ROC_UP),
        "roc_down": bool(np.isfinite(roc20) and roc20 < ROC_DOWN),
        "roc_flat": bool(np.isfinite(roc20) and abs(roc20) < ROC_FLAT),
        "roc_not_down": bool(np.isfinite(roc20) and roc20 >= ROC_DOWN),
        "roc_turn": bool(np.isfinite(roc5) and np.isfinite(roc20) and roc5 > roc20),
        "weak": (not above20) or bool(np.isfinite(slope) and slope < SLOPE_DOWN),
        "near_high": bool(np.isfinite(dist_high60) and dist_high60 < NEAR_HIGH_PCT),
        "near_low": bool(np.isfinite(dist_low60) and dist_low60 < NEAR_LOW_PCT),
        "deep_low": bool(np.isfinite(dist_low60) and dist_low60 < DEEP_LOW_PCT),
        "flat20": bool(np.isfinite(range20) and range20 < RANGE_FLAT_PCT),
    })
    return f


# 每阶段: (门控条件, [(特征, 权重)])
_STAGE_RULES: dict[str, tuple[list[str], list[tuple[str, int]]]] = {
    "上涨": (["bull"], [("bull", 3), ("slope_up", 2), ("roc_up", 2), ("above20", 1), ("ma20_ge_ma60", 1)]),
    "顶部": (["ma20_ge_ma60", "no_bull"], [("ma20_ge_ma60", 2), ("weak", 2), ("roc_down", 2), ("near_high", 1), ("above60", 1)]),
    "下跌": (["ma20_lt_ma60", "no_bull"], [("bear", 3), ("slope_down", 2), ("roc_down", 2), ("below_ma20", 1)]),
    "底部": (["ma20_lt_ma60"], [("above20", 2), ("slope_up", 2), ("roc_turn", 2), ("roc_not_down", 1), ("below_ma60", 1), ("near_low", 1)]),
    "筑底": (["flat20"], [("flat20", 3), ("slope_flat", 1), ("roc_flat", 2), ("deep_low", 2), ("no_bear", 1)]),
}
_STAGE_MAX = {k: sum(w for _, w in rules) for k, (gates, rules) in _STAGE_RULES.items()}


def _detect_stage(df: pd.DataFrame) -> dict:
    """单指数周期阶段 → {stage, confidence, features, scores, note}。"""
    base = {"stage": "未知", "confidence": 0.0, "scores": {}, "note": "样本不足", "features": {}}
    f = _stage_features(df)
    if not f.get("ok"):
        base["note"] = f"样本不足(<{MIN_BARS}日)或MA缺失"
        return base
    scores: dict[str, int] = {}
    candidates: list[str] = []
    for stage, (gates, rules) in _STAGE_RULES.items():
        if any(g == "no_bull" and f["bull"] for g in gates):
            continue
        if any(g == "no_bear" and f["bear"] for g in gates):
            continue
        if any(g not in ("no_bull", "no_bear") and not f[g] for g in gates):
            continue
        candidates.append(stage)
        scores[stage] = sum(w for feat, w in rules if f.get(feat))
    if not candidates:
        # 门控全不满足 → 结构性兜底
        if f["bull"]:
            stage = "上涨"
        elif f["bear"]:
            stage = "下跌"
        elif f["above20"] and f["ma20_lt_ma60"]:
            stage = "底部"
        elif f["ma20_ge_ma60"]:
            stage = "顶部"
        else:
            stage = "筑底"
        scores = {s: 0 for s in STAGES}
        scores[stage] = 1
        conf = 0.4
        note = f"门控不满足→结构性兜底({stage})"
    else:
        stage = max(candidates, key=lambda s: (scores[s], STAGE_PRIORITY[s]))
        conf = round(min(scores[stage] / _STAGE_MAX[stage], 0.95), 2)
        note = ""
    feats = {k: v for k, v in f.items() if k != "ok"}
    return {"stage": stage, "confidence": conf, "scores": scores, "features": feats,
            "note": note, "close": f["close"], "ma20": f["ma20"], "ma60": f["ma60"],
            "slope": f["slope"], "roc20": f["roc20"], "range20": f["range20"],
            "dist_low60": f["dist_low60"], "dist_high60": f["dist_high60"]}


# ════════════════════════════════════════════════════════════
# T形底部检测（底部横盘 <5% + 放量突破 量比>1.5）
# ════════════════════════════════════════════════════════════
def _detect_t_formation(df: pd.DataFrame) -> dict:
    """回溯最近 T_SCAN_DAYS 个交易日找 'T形底突破'。

    条件（Appel T-formation）:
      1. 突破日前 T_BASE_DAYS 日横盘: (区间高-区间低)/区间均收 < 5%
      2. 横盘区间贴近 60日低点（均值距低点 < 8%）→ 确认是底部而非中继
      3. 突破日: 收盘 > 横盘区间最高价（放量突破）
      4. 量比 = 突破日成交量 / 前5日均量 > 1.5
    """
    base = {"signal": None, "index": "", "note": "未检出", "range_pct": np.nan,
            "volume_ratio": np.nan, "breakout_pct": np.nan}
    if df is None or len(df) < MIN_BARS_T:
        base["note"] = f"样本不足(<{MIN_BARS_T}日)"
        return base
    d = df.reset_index(drop=True)
    close = pd.to_numeric(d["close"], errors="coerce")
    if close.isna().all():
        base["note"] = "收盘数据缺失"
        return base
    n = len(d)
    for b in range(max(T_BASE_DAYS, n - T_SCAN_DAYS), n):      # 突破日候选（含当日）
        s = b - T_BASE_DAYS                                     # 横盘窗口起点
        if s < 0:
            continue
        w = d.iloc[s:b]
        hi, lo = float(w["high"].max()), float(w["low"].min())
        base_mean = float(close.iloc[s:b].mean())
        if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(base_mean) or base_mean <= 0:
            continue
        rng_pct = (hi - lo) / base_mean * 100.0
        lo60 = float(d["low"].iloc[max(0, b - 60):b].min())
        if not np.isfinite(lo60) or lo60 <= 0:
            continue
        dist_low = (base_mean - lo60) / lo60 * 100.0
        if rng_pct >= T_RANGE_PCT or dist_low >= T_NEAR_LOW_PCT:
            continue
        vol5 = float(d["volume"].iloc[max(0, b - 5):b].mean())
        vol_t = float(d["volume"].iloc[b])
        if not np.isfinite(vol_t) or not np.isfinite(vol5) or vol5 <= 0:
            continue
        vr = vol_t / vol5
        close_t = float(close.iloc[b])
        brk = (close_t - hi) / hi * 100.0 if hi > 0 else np.nan
        if close_t > hi and vr > T_VOL_RATIO:
            return {
                "signal": T_FORMATION_SIGNAL,
                "date": str(d["date"].iloc[b].date()),
                "base_days": T_BASE_DAYS,
                "range_pct": round(rng_pct, 2),
                "volume_ratio": round(vr, 2),
                "breakout_pct": round(brk, 2),
                "base_low": round(lo, 2), "base_high": round(hi, 2),
                "dist_low60_pct": round(dist_low, 2),
                "confidence": round(min(0.9, 0.55 + 0.1 * (vr - T_VOL_RATIO) + 0.05 * min(brk, 3)), 2),
                "note": "底部横盘+放量突破",
            }
    # 未突破但存在贴近低点的横盘 → 提示观察（不构成信号）
    last = d.iloc[-T_BASE_DAYS:]
    hi, lo = float(last["high"].max()), float(last["low"].min())
    base_mean = float(close.iloc[-T_BASE_DAYS:].mean())
    if np.isfinite(hi) and np.isfinite(lo) and np.isfinite(base_mean) and base_mean > 0:
        rng_pct = (hi - lo) / base_mean * 100.0
        if rng_pct < T_RANGE_PCT:
            base["note"] = "底部横盘形态存在但尚未放量突破（观察）"
    return base


# ════════════════════════════════════════════════════════════
# VIX 买入区
# ════════════════════════════════════════════════════════════
def _detect_vix_zone(vix: float | None, temp: float | None) -> dict:
    """VIX>30（或温度<30替代）→ 恐慌买入区；温度>85+VIX<15 → 风险区。"""
    if vix is not None:
        if vix > VIX_PANIC:
            return {"zone": "恐慌买入区", "level": "buy", "reason": f"VIX={vix:.1f}>30（恐慌，逆向）"}
        if temp is not None and temp > TEMP_RISK_HIGH and vix < VIX_RISK_LOW:
            return {"zone": "风险区", "level": "risk",
                    "reason": f"温度{temp:.0f}>85 + VIX={vix:.1f}<15（过热风险）"}
        if vix >= VIX_RISK_LOW:
            return {"zone": "中性区", "level": "neutral", "reason": f"VIX={vix:.1f}∈[15,30]"}
        # VIX<15 但温度未过热
        if temp is not None and temp > TEMP_RISK_HIGH:
            return {"zone": "偏热观察", "level": "watch",
                    "reason": f"温度{temp:.0f}>85（VIX={vix:.1f}偏低，注意过热）"}
        return {"zone": "中性区", "level": "neutral", "reason": f"VIX={vix:.1f}<15 且温度正常"}
    # VIX 缺失 → 温度替代
    if temp is not None:
        if temp < TEMP_PANIC_SUB:
            return {"zone": "恐慌买入区", "level": "buy",
                    "reason": f"VIX缺失，温度{temp:.0f}<30 → 恐慌买入区（温度替代）"}
        if temp > TEMP_RISK_SUB:
            return {"zone": "高温风险区", "level": "risk",
                    "reason": f"VIX缺失，温度{temp:.0f}>85 → 高温风险区（温度替代）"}
    return {"zone": "中性区", "level": "neutral",
            "reason": "VIX缺失且温度无极端值 → 中性"}


# ════════════════════════════════════════════════════════════
# 周期共振 / 综合
# ════════════════════════════════════════════════════════════
def _resonance(stages: list[str]) -> dict:
    """多指数阶段共振：同阶段占比≥60% → 一致(占比≥75% 且指数≥3 → 强)；否则结构分化。"""
    total = len(stages)
    if total == 0:
        return {"state": "无数据", "dominant": None, "share": 0.0, "counts": {}, "note": "无可用指数"}
    counts: dict[str, int] = {}
    for s in stages:
        counts[s] = counts.get(s, 0) + 1
    dominant = max(counts, key=lambda k: (counts[k], STAGE_PRIORITY[k]))
    share = counts[dominant] / total
    if total >= 2 and share >= 0.6:
        state = "周期一致" if share < 0.75 or total < 3 else "周期一致(强)"
        note = f"{dominant}阶段 {counts[dominant]}/{total} 指数同向"
    else:
        state = "结构分化"
        note = f"指数阶段分歧（主导{dominant}仅{counts[dominant]}/{total}）"
    return {"state": state, "dominant": dominant, "share": round(share, 2),
            "counts": counts, "note": note}


def _composite(res: dict) -> dict:
    """阶段 × 共振 × VIX → 市场周期状态 + 策略建议。

    组合逻辑:
      基准: 主导阶段映射策略（底部/筑底=布局, 上涨=持有, 顶部=减仓, 下跌=防守）
      共振: 一致→上调置信; 结构分化→降级为中性/结构机会
      VIX: 恐慌买入区→强化逆向布局; 风险区→降级防守/减仓
      T形底突破: 强化底部布局信号
    """
    reso = res["resonance"]
    vix_zone = res["vix_zone"]
    dominant = reso.get("dominant")
    t_hits = res.get("t_hits") or []
    breadth = res.get("breadth")

    if dominant is None:
        return {"state": "无数据", "strategy": "观望", "signal": "震荡",
                "confidence": 0.0, "suggestions": [], "reasons": []}

    strat, base_sig = STAGE_STRATEGY[dominant]
    reasons = [f"主导阶段={dominant}（{counts_desc(reso)}）"]
    if reso["state"].startswith("周期一致"):
        reasons.append(f"周期{reso['state']} → 多指数同向，信号可信")
    else:
        reasons.append("结构分化 → 指数阶段分歧，按主导阶段降档处理")

    # VIX 修正
    if vix_zone["level"] == "buy":
        if dominant in ("底部", "筑底"):
            strat = f"逆向布局（{vix_zone['zone']}）"
            base_sig = "多"
            reasons.append(f"{vix_zone['zone']} → 恐慌=逆向机会，需价格/宽度确认（Appel 分批）")
        elif dominant == "下跌":
            strat = f"恐慌观察（{vix_zone['zone']}）"
            base_sig = "震荡"
            reasons.append("恐慌买入区但趋势仍下跌 → 观察等待企稳，不盲目抄底")
        else:
            reasons.append(f"{vix_zone['zone']} 提示")
    elif vix_zone["level"] == "risk":
        strat = f"减仓防守（{vix_zone['zone']}）"
        base_sig = "空"
        reasons.append(f"{vix_zone['zone']} → 低波动高温=过热风险，降低仓位")
    elif vix_zone["level"] == "watch":
        reasons.append("温度偏高注意过热风险")

    # T形底突破强化
    if t_hits:
        h0 = t_hits[0]
        names = "、".join(h["index"] for h in t_hits[:3])
        reasons.append(f"{names}出现{T_FORMATION_SIGNAL}"
                       f"（{h0['date']}, 横盘{h0.get('range_pct')}%, 量比{h0.get('volume_ratio')}）"
                       f"→ 底部结构确认")
        if dominant in ("底部", "筑底") and base_sig == "多":
            strat += " + T形底突破确认"

    # 宽度辅助证据
    if breadth is not None:
        if breadth.get("low20") == 0 and breadth.get("high20", 0) > 0:
            reasons.append(f"宽度修复: 新低0家/新高{breadth['high20']}家（宽度支持底部）")
        elif breadth.get("low20", 0) > 100:
            reasons.append(f"宽度恶化: 新低{breadth['low20']}家（下跌未止）")

    # 结构分化 → 中性降档
    if reso["state"] == "结构分化" and vix_zone["level"] not in ("buy", "risk"):
        if base_sig != "震荡":
            base_sig = "震荡"
            reasons.append("结构分化 → 综合信号降为震荡（结构机会为主）")

    # 置信度
    conf = 0.55 + 0.22 * reso["share"]
    if reso["state"].startswith("周期一致"):
        conf += 0.05
    if vix_zone["level"] == "buy":
        conf += 0.08
    elif vix_zone["level"] == "risk":
        conf += 0.05
    if t_hits:
        conf += 0.05
    if reso["state"] == "结构分化":
        conf -= 0.15
    conf = round(min(max(conf, 0.15), 0.92), 2)

    state = f"{dominant}期" if reso["state"].startswith("周期一致") else "结构分化期"
    suggestions = _suggestions(dominant, reso, vix_zone, t_hits)
    return {"state": state, "strategy": strat, "signal": base_sig,
            "confidence": conf, "suggestions": suggestions, "reasons": reasons}


def counts_desc(reso: dict) -> str:
    c = reso.get("counts") or {}
    if not c:
        return "无"
    return " / ".join(f"{k}{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def _suggestions(dominant: str, reso: dict, vix_zone: dict, t_hits: list) -> list[str]:
    s: list[str] = []
    if dominant in ("底部", "筑底"):
        s.append("底部/筑底阶段：分批布局，逢低吸纳主线，控制单笔仓位")
    elif dominant == "上涨":
        s.append("上涨阶段：持股为主，回踩 MA20 不破可持有")
    elif dominant == "顶部":
        s.append("顶部阶段：分批减仓，兑现高位浮盈，降低杠杆")
    elif dominant == "下跌":
        s.append("下跌阶段：防守为主，控制仓位，等待止跌信号")
    if vix_zone["level"] == "buy":
        s.append("恐慌买入区：逆向分批（先小仓试探，价格/宽度确认后加仓），设定明确止损")
    elif vix_zone["level"] == "risk":
        s.append("风险区：控制仓位，防过热回撤，避免追高")
    if reso["state"] == "结构分化":
        s.append("结构分化：以结构性机会为主，选强指数/强板块，避免满仓单一方向")
    if t_hits:
        s.append(f"{T_FORMATION_SIGNAL}确认指数可重点跟踪（突破回踩不破为佳）")
    if not s:
        s.append("维持中性仓位，等待周期信号明确")
    return s


# ════════════════════════════════════════════════════════════
# RAG 解释
# ════════════════════════════════════════════════════════════
class CycleSystem:
    """市场周期系统 — detect/report/view 统一接口。"""

    def __init__(self, out_dir: Path | str | None = None):
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self._rag_cache: dict | None = None

    # ── RAG ────────────────────────────────────────────────
    def _rag(self) -> dict:
        if self._rag_cache is not None:
            return self._rag_cache
        try:
            from quant_system.analysis_core import knowledge_rag
            hits = knowledge_rag.search(RAG_QUERY, k=RAG_K)
            if hits:
                self._rag_cache = {
                    "query": RAG_QUERY, "available": True, "basis": "检索可用",
                    "hits": [{"file": h.get("file"), "cat": h.get("cat", ""),
                              "score": h.get("score"),
                              "summary": (h.get("text") or "")[:200]} for h in hits],
                }
            else:
                self._rag_cache = {"query": RAG_QUERY, "available": False,
                                   "basis": "检索无命中", "hits": []}
        except Exception as e:  # noqa: BLE001
            self._rag_cache = {"query": RAG_QUERY, "available": False, "hits": [],
                               "basis": "检索不可用", "error": str(e)[:120]}
        return self._rag_cache

    # ── detect ─────────────────────────────────────────────
    def detect(self, date: str | None = None) -> dict:
        """市场级 + 指数级周期检测。返回 {date, indexes, resonance, vix_zone,
        breadth, t_hits, composite, signal, confidence, evidence, rag, status}。"""
        t0 = time.time()
        date = norm_date(date)
        target = pd.Timestamp(date)
        data_status: dict[str, str] = {}

        # 1. 市场温度 + VIX + 宽度
        temp, temp_as_of, temp_lag = _load_fusion_temperature(date)
        if temp is None:
            data_status["fusion温度"] = f"缺失（≤{date} 无 fusion 记录）→ VIX区仅用VIX/跳过"
        elif (temp_lag or 0) > 3:
            data_status["fusion温度"] = f"落后 {temp_lag} 日(as_of={temp_as_of})"
        vix, vix_as_of, vix_note = _load_vix(date)
        if vix is None:
            data_status["VIX"] = vix_note or "缺失"
        breadth = _load_breadth(date)
        if breadth is None:
            data_status["新高新低宽度"] = "缺失"

        # 2. 指数级检测（单指数失败不崩）
        sources = list(INDEX_SOURCES)
        if not any((MARKET_DIR / fn).exists() for _, fn in sources):
            sources = [FALLBACK_INDEX]
        indexes: list[dict] = []
        skipped = 0
        for name, fn in sources:
            path = MARKET_DIR / fn
            if not path.exists():
                skipped += 1
                data_status[f"指数:{name}"] = f"缺失 {fn}"
                continue
            try:
                df = _load_index_df(path, target)
                if df is None or df.empty:
                    skipped += 1
                    data_status[f"指数:{name}"] = "无数据"
                    continue
                if len(df) < MIN_BARS:
                    skipped += 1
                    data_status[f"指数:{name}"] = f"样本<{MIN_BARS}日(实际{len(df)})"
                    continue
                win = df.tail(LOOKBACK_DAYS * 2).reset_index(drop=True)
                stage = _detect_stage(win)
                t_form = _detect_t_formation(win)
                t_form["index"] = name
                row = {
                    "name": name, "source": fn, "ok": stage["stage"] != "未知",
                    "date": str(win["date"].iloc[-1].date()),
                    "close": num(stage.get("close")),
                    "stage": stage["stage"],
                    "stage_confidence": stage["confidence"],
                    "stage_note": stage["note"],
                    "scores": stage["scores"],
                    "metrics": {
                        "ma20": num(stage.get("ma20")), "ma60": num(stage.get("ma60")),
                        "ma20_slope_pct": num(stage.get("slope")),
                        "roc20_pct": num(stage.get("roc20")),
                        "range20_pct": num(stage.get("range20")),
                        "dist_low60_pct": num(stage.get("dist_low60")),
                        "dist_high60_pct": num(stage.get("dist_high60")),
                    },
                    "t_formation": {k: v for k, v in t_form.items()},
                }
                if not row["ok"]:
                    data_status[f"指数:{name}"] = stage["note"]
                    skipped += 1
                indexes.append(row)
            except Exception as e:  # noqa: BLE001
                skipped += 1
                data_status[f"指数:{name}"] = f"检测失败: {str(e)[:80]}"
                continue
        data_date = max((r["date"] for r in indexes), default=date)
        if not indexes:
            return {
                "date": date, "data_date": data_date, "agent": "市场周期",
                "signal": "震荡", "confidence": 0.0, "status": "degraded",
                "evidence": [f"无可用指数数据（{skipped}/{len(sources)} 跳过）"],
                "data_status": data_status, "resonance": _resonance([]),
                "vix_zone": _detect_vix_zone(vix, temp), "t_hits": [],
                "composite": None, "rag": self._rag(),
            }

        # 3. 周期共振 + VIX区 + T形底聚合
        stages = [r["stage"] for r in indexes if r["ok"]]
        reso = _resonance(stages)
        vix_zone = _detect_vix_zone(vix, temp)
        t_hits = [r["t_formation"] for r in indexes
                  if r["t_formation"].get("signal") == T_FORMATION_SIGNAL]

        # 4. 综合
        composite = _composite({
            "resonance": reso, "vix_zone": vix_zone,
            "t_hits": t_hits, "breadth": breadth,
        })
        signal, conf = composite["signal"], composite["confidence"]

        # 5. 证据链
        evidence = [
            f"周期共振: {reso['state']}（{reso['note']}）",
            f"VIX区: {vix_zone['zone']}（{vix_zone['reason']}）",
            f"综合: 主导阶段{reso['dominant']} → {composite['strategy']}",
            *composite["reasons"],
        ]
        for r in indexes[:6]:
            ev = (f"{r['name']}: {r['stage']}（置信{r['stage_confidence']:.2f}）"
                  f" MA20={fmt(r['metrics']['ma20'])} MA60={fmt(r['metrics']['ma60'])}"
                  f" MA20斜率{_fmt_pct(r['metrics']['ma20_slope_pct'])}"
                  f" ROC20={_fmt_pct(r['metrics']['roc20_pct'])}"
                  f" 20日振幅{_fmt_pct(r['metrics']['range20_pct'])}")
            if r["t_formation"].get("signal"):
                tf = r["t_formation"]
                ev += f" | {T_FORMATION_SIGNAL}({tf['date']}, 量比{tf['volume_ratio']}, 突破{tf['breakout_pct']}%)"
            evidence.append(ev)
        if breadth is not None:
            evidence.append(
                f"宽度({breadth['as_of']}): 新高20={breadth['high20']} 新低20={breadth['low20']} "
                f"新高60={breadth['high60']} 新低60={breadth['low60']}（比={breadth['ratio20']}）")
        if temp is not None:
            evidence.append(f"温度{temp:.0f}(as_of={temp_as_of}){'⚠️落后' if (temp_lag or 0) > 3 else ''}")
        rag = self._rag()
        evidence.append(f"RAG: {rag.get('basis', '检索不可用')}")

        status = "ok" if not data_status else "degraded"
        return {
            "date": date,
            "data_date": data_date,
            "agent": "市场周期",
            "temperature": temp, "temperature_as_of": temp_as_of,
            "vix": vix, "vix_as_of": vix_as_of, "vix_note": vix_note,
            "breadth": breadth,
            "indexes": indexes,
            "stages": stages,
            "resonance": reso,
            "vix_zone": vix_zone,
            "t_hits": t_hits,
            "composite": composite,
            "signal": signal,
            "confidence": conf,
            "evidence": evidence,
            "rag": rag,
            "data_status": data_status,
            "status": status,
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # ── report ─────────────────────────────────────────────
    def report(self, date: str | None = None, out_dir: Path | str | None = None) -> Path:
        date = norm_date(date)
        out = self.out_dir if out_dir is None else Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        r = self.detect(date)
        path = out / f"cycle_report_{r['data_date']}.md"
        path.write_text(_render_markdown(r), encoding="utf-8")
        return path

    # ── view ───────────────────────────────────────────────
    def view(self, date: str | None = None) -> dict:
        """多智能体视图: {agent, signal, view, confidence, evidence, weight, status, detail}。"""
        try:
            r = self.detect(date)
        except Exception as e:  # noqa: BLE001
            return {"agent": "市场周期", "signal": "震荡", "view": "震荡",
                    "confidence": 0.0, "evidence": [f"检测异常: {str(e)[:100]}"],
                    "status": "degraded", "weight": VIEW_WEIGHT,
                    "detail": {"date": norm_date(date)}, "rag_basis": "检测异常"}
        view = signal_view(r["signal"])
        comp = r.get("composite") or {}
        return {
            "agent": "市场周期",
            "signal": r["signal"],
            "view": view,
            "confidence": r["confidence"],
            "evidence": r["evidence"],
            "weight": VIEW_WEIGHT,
            "status": r["status"],
            "detail": {
                "date": r["data_date"],
                "cycle_state": comp.get("state", "无数据"),
                "strategy": comp.get("strategy", "观望"),
                "dominant_stage": (r.get("resonance") or {}).get("dominant"),
                "resonance": (r.get("resonance") or {}).get("state"),
                "vix_zone": (r.get("vix_zone") or {}).get("zone"),
                "t_hits": [h.get("index") for h in (r.get("t_hits") or [])],
                "temperature": r.get("temperature"),
                "vix": r.get("vix"),
            },
            "rag_basis": (r.get("rag") or {}).get("basis", "检索不可用"),
        }

    def _degraded_view(self, msg: str, rag_basis: str = "检索不可用",
                       date: str | None = None) -> dict:
        return {"agent": "市场周期", "signal": "震荡", "view": "震荡", "confidence": 0.0,
                "evidence": [msg, f"RAG: {rag_basis}"], "status": "degraded",
                "weight": VIEW_WEIGHT, "detail": {"date": date or ""},
                "rag_basis": rag_basis}


# ════════════════════════════════════════════════════════════
# 报告渲染
# ════════════════════════════════════════════════════════════
def _render_markdown(r: dict) -> str:
    comp = r.get("composite") or {}
    reso = r.get("resonance") or {}
    vix_zone = r.get("vix_zone") or {}
    temp_txt = f"{r['temperature']:.0f}" if r.get("temperature") is not None else "NA"
    vix_txt = f"{r['vix']:.1f}" if r.get("vix") is not None else "NA"
    lines = [
        f"# 市场周期报告 {r['data_date']}",
        "",
        f"> 体系11 · Appel 周期分割 / T形底部 / VIX 买入区 / 周期共振",
        f"> 数据日期 {r['data_date']}（分析基准 {r['date']}） | 状态: {r['status']}",
        "",
        "## 一、市场周期总览",
        "",
        f"- 周期状态: **{comp.get('state', '无数据')}**",
        f"- 策略建议: **{comp.get('strategy', '观望')}**",
        f"- 信号 / 置信: {r['signal']} / {r['confidence']:.2f}",
        f"- 周期共振: {reso.get('state', '无数据')}（{reso.get('note', '')}）",
        f"- VIX 区: {vix_zone.get('zone', '中性')}（{vix_zone.get('reason', '')}）",
        f"- VIX: {vix_txt}（as_of {r.get('vix_as_of') or '无'}） | 温度: {temp_txt}",
        "",
        "## 二、指数周期阶段",
        "",
        "| 指数 | 阶段 | 置信 | 收盘 | MA20 | MA60 | MA20斜率%/日 | ROC20% | 20日振幅% | T形底 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for idx in r.get("indexes", []):
        m = idx["metrics"]
        tf = idx["t_formation"]
        tf_txt = (f"✅{tf['date']} 量比{tf['volume_ratio']} 突破{tf['breakout_pct']}%"
                  if tf.get("signal") else "—")
        lines.append(
            f"| {idx['name']} | {idx['stage']} | {idx['stage_confidence']:.2f} "
            f"| {fmt(idx['close'])} | {fmt(m['ma20'])} | {fmt(m['ma60'])} "
            f"| {fmt(m['ma20_slope_pct'])} | {fmt(m['roc20_pct'])} "
            f"| {fmt(m['range20_pct'])} | {tf_txt} |")
    lines += ["", "## 三、T形底部信号", ""]
    t_hits = r.get("t_hits") or []
    if t_hits:
        for h in t_hits:
            lines.append(
                f"- **{h['index']}** {h['date']}: 横盘{h['base_days']}日振幅{h['range_pct']}%"
                f"（距60日低点{h['dist_low60_pct']}%）→ 放量突破（量比{h['volume_ratio']}）"
                f"，突破{h['breakout_pct']}%")
        lines.append("- 信号: **T形底突破**（Appel: 需价格/宽度确认，分批介入）")
    else:
        lines.append("- 近期无 T形底突破（底部横盘<5% + 量比>1.5 未同时满足）")
    lines += ["", "## 四、策略建议", ""]
    lines += [f"- {s}" for s in (comp.get("suggestions") or [])]
    lines += ["", "## 五、宽度与证据", ""]
    b = r.get("breadth")
    if b:
        lines.append(
            f"- 新高/新低({b['as_of']}): 20日 新高{b['high20']}/新低{b['low20']} "
            f"| 60日 新高{b['high60']}/新低{b['low60']}")
    lines += [f"- {e}" for e in (r.get("evidence") or [])]
    rag = r.get("rag") or {}
    lines += ["", "## 六、知识依据（RAG）", ""]
    lines.append(f"- 检索: `{rag.get('query', RAG_QUERY)}` → {rag.get('basis', '检索不可用')}")
    for h in (rag.get("hits") or [])[:RAG_K]:
        lines.append(f"- [{h.get('cat', '')}] {h.get('file', '')}（score {h.get('score')}）")
        lines.append(f"  - {(h.get('summary') or '')[:120].replace(chr(10), ' ')}")
    ds = r.get("data_status") or {}
    if ds:
        lines += ["", "## 七、数据状态", ""]
        lines += [f"- {k}: {v}" for k, v in ds.items()]
    lines += ["", f"生成于 {datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')} CST"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="体系11 市场周期系统")
    ap.add_argument("--date", type=str, default=None, help="分析日期 YYYY-MM-DD（默认今天）")
    ap.add_argument("--report", action="store_true", help="输出 generated/cycle_report_{date}.md")
    args = ap.parse_args()
    cs = CycleSystem()
    r = cs.detect(args.date)
    print(f"日期 {r['date']} | 数据日 {r['data_date']} | 状态 {r['status']}")
    print(f"周期状态: {(r.get('composite') or {}).get('state')} | "
          f"策略: {(r.get('composite') or {}).get('strategy')} | 信号: {r['signal']} "
          f"置信: {r['confidence']}")
    for idx in r.get("indexes", []):
        tf = idx["t_formation"]
        tf_txt = f" | T形底: {tf.get('date')} 量比{tf.get('volume_ratio')}" if tf.get("signal") else ""
        print(f"  {idx['name']}: {idx['stage']} 置信{idx['stage_confidence']:.2f} "
              f"MA20斜率{_fmt_pct(idx['metrics']['ma20_slope_pct'])} "
              f"ROC20={_fmt_pct(idx['metrics']['roc20_pct'])} "
              f"振幅{_fmt_pct(idx['metrics']['range20_pct'])}{tf_txt}")
    print(f"共振: {(r.get('resonance') or {}).get('state')} | "
          f"VIX区: {(r.get('vix_zone') or {}).get('zone')}")
    if args.report:
        path = cs.report(args.date)
        print(f"报告已写: {path}")


if __name__ == "__main__":
    main()
