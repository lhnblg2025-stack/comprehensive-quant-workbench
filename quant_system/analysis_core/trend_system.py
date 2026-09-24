#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trend_system — 体系2 三屏趋势系统（Elder 三重滤网 / Murphy 突破确认 / Trader Vic 趋势诊断）

方法论（skills/elder-triple-screen-system + murphy-technical-analysis-study-guide
        + trader-vic-trend-change-diagnosis）:
  第一屏（周线=大周期）: 市场潮汐方向 —— 周线重采样(W-FRI≈5交易日)
                          MA20/MA60 多空排列 + MA20 近5周斜率
  第二屏（日线=中周期）: 与潮汐同向的入场窗口 —— 日线 MACD(12,26,9) 柱状图方向
                          + 收盘 vs MA20
  第三屏（近5日=小周期）: 精确定时 —— 收盘 vs MA5 + 近5日 MA5 斜率（日内动量代理）
  综合评级: 三屏同向=强趋势(多/空) / 两屏同向=中等 / 混乱=震荡 / 任一屏不足=样本不足
  趋势线: pivot 高低点线性回归（独立实现 pattern_engine._trendline 思路）
          突破需 >3% 收盘确认 或 连续2日收盘在另一侧（Murphy）
          支撑阻力互换（Murphy）: 突破后原阻力转支撑 / 原支撑转阻力

输入:
  data_warehouse/kline/{6位代码}.parquet   个股日K（date/open/high/low/close）
  data_warehouse/market/index_daily*.parquet 沪深300 基准（同为三屏分析对象）

统一接口:
  from quant_system.analysis_core.trend_system import TrendSystem
  ts = TrendSystem()
  res = ts.detect(date=None, watch=["601899", "600519"])   # 全量检测
  path = ts.report(date=None, watch=["601899"])            # 写 generated/trend_report_{date}.md
  view = ts.view(date=None)                                # multi_agent 兼容 {agent, signal, confidence, evidence}

用法:
  python3 -m quant_system.analysis_core.trend_system --watch 601899,600519
  python3 -m quant_system.analysis_core.trend_system --report --watch 601899,600519
  python3 -m quant_system.analysis_core.trend_system --limit 200            # 全A抽样
  python3 -m quant_system.analysis_core.trend_system --date 2026-08-07 --watch 601899
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # noqa: F401
except Exception:  # pragma: no cover
    pq = None

DATA_ROOT = Path(__file__).resolve().parent.parent.parent          # workspace（data_warehouse 所在）
REPO_ROOT = Path(__file__).resolve().parent.parent                 # 仓库根（generated/ 输出目录）
KLINE_DIR = DATA_ROOT / "data_warehouse" / "kline"
MARKET_DIR = DATA_ROOT / "data_warehouse" / "market"
DEFAULT_OUT_DIR = REPO_ROOT / "generated"

CODE_RE = re.compile(r"^\d{6}$")

sys.path.insert(0, str(DATA_ROOT))
from quant_system.analysis_core.common import (  # noqa: E402
    kline_files,
    load_names,
    read_kline_window,
)


# 基准候选：优先显式沪深300（最新 2026-08-10），回退通用 index_daily 等
BENCH_CANDIDATES = (
    "index_daily_沪深300.parquet",
    "index_daily.parquet",
    "hs300.parquet",
    "000300.parquet",
    "sh000300.parquet",
)
BENCH_CODE = "000300"
BENCH_NAME = "沪深300"

# ── 参数 ─────────────────────────────────────────────────────────
LOOKBACK_DAYS = 480           # 日历回看窗口（≈340 交易日 ≈68 周，覆盖周线 MA60 + 斜率余量）
MIN_DAILY_BARS = 90           # 日线 MACD 屏最少样本
MIN_DAILY_BARS_WEEKLY = 310   # 周线屏最少日线样本（≈62 周）
MIN_WEEK_BARS = 62            # 周线最少根数（MA60=60 + 余量）
FLAT_SLOPE_WEEK = 0.15        # 周线 MA20 斜率阈值（%/周，低于视为走平）
CONFIRM_PCT = 0.03            # Murphy 突破确认: 超出线位 >3%
MIN_TREND_SLOPE = 0.0008      # 趋势线最小相对斜率（与 pattern_engine 口径一致）
RAG_QUERY = "三屏趋势 突破 确认"
RAG_K = 3
VIEW_WEIGHT = 1.0             # view() 默认权重（multi_agent 仲裁权重在 multi_agent.BASE_WEIGHTS 维护）
PROGRESS_EVERY = 500


# ────────────────────────────────────────────────────────────
# 数据读取 / 名称
# ────────────────────────────────────────────────────────────
def load_benchmark_df() -> tuple[pd.DataFrame | None, str]:
    """沪深300 日K（date/open/high/low/close/volume，date 为 datetime64）。"""
    for fn in BENCH_CANDIDATES:
        p = MARKET_DIR / fn
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if not {"date", "close"}.issubset(df.columns):
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
            if len(df) < MIN_DAILY_BARS:
                continue
            return df, f"market/{fn}"
        except Exception as e:
            logging.getLogger(__name__).error(f"[trend_system] 操作失败: {e}", exc_info=True)
            continue
    return None, "未找到沪深300基准文件"


def _norm_kline(df: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame | None:
    """K线列规整：date 转 datetime、去重、排序、截断至 target。"""
    if df is None or df.empty:
        return None
    df = df.copy()
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= target]
    if df.empty:
        return None
    return df


# ────────────────────────────────────────────────────────────
# 三屏检测（每屏返回 trend ∈ {多, 空, 震荡, 样本不足}）
# ────────────────────────────────────────────────────────────
def _screen_weekly(df: pd.DataFrame) -> dict:
    """第一屏 大周期（周线重采样≈5交易日）: MA20/MA60 多空排列 + MA20 近5周斜率。"""
    base = {"trend": "样本不足", "n_bars": 0, "ma20": np.nan, "ma60": np.nan,
            "slope_pct": np.nan, "note": ""}
    if df is None or len(df) < MIN_DAILY_BARS_WEEKLY:
        base["note"] = f"日线样本<{MIN_DAILY_BARS_WEEKLY}根(约62周)"
        return base
    try:
        d = df.set_index("date")[["open", "high", "low", "close"]]
        w = d.resample("W-FRI").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"})
    except Exception as e:
        base["note"] = f"周线重采样失败: {str(e)[:60]}"
        return base
    w = w.dropna(subset=["close"])
    wc = w["close"].astype(float)
    nw = len(wc)
    if nw < MIN_WEEK_BARS:
        base["note"] = f"周线样本<{MIN_WEEK_BARS}根(实际{nw})"
        return base
    ma20 = wc.rolling(20, min_periods=20).mean()
    ma60 = wc.rolling(60, min_periods=60).mean()
    ma20_t, ma60_t = float(ma20.iloc[-1]), float(ma60.iloc[-1])
    slope_pct = np.nan
    seg = ma20.iloc[-5:].to_numpy(dtype=float)
    if np.all(np.isfinite(seg)) and np.isfinite(ma20_t) and ma20_t > 0:
        slope_pct = float(np.polyfit(np.arange(5), seg, 1)[0]) / ma20_t * 100.0
    trend = "震荡"
    if np.isfinite(ma20_t) and np.isfinite(ma60_t):
        if not np.isfinite(slope_pct):
            trend = "多" if ma20_t > ma60_t else ("空" if ma20_t < ma60_t else "震荡")
            base["note"] = "MA20斜率不可用，仅按排列判断"
        elif ma20_t > ma60_t and slope_pct > -FLAT_SLOPE_WEEK:
            trend = "多"
        elif ma20_t < ma60_t and slope_pct < FLAT_SLOPE_WEEK:
            trend = "空"
    return {"trend": trend, "n_bars": nw, "ma20": ma20_t, "ma60": ma60_t,
            "slope_pct": slope_pct, "note": base["note"]}


def _screen_daily(close: pd.Series) -> dict:
    """第二屏 中周期（日线）: MACD(12,26,9) 柱状图方向 + 收盘 vs MA20。"""
    base = {"trend": "样本不足", "macd": np.nan, "signal": np.nan, "hist": np.nan,
            "hist_up": None, "ma20": np.nan, "close_vs_ma20": None, "note": ""}
    if close is None or len(close) < MIN_DAILY_BARS:
        base["note"] = f"日线样本<{MIN_DAILY_BARS}"
        return base
    c = pd.to_numeric(close, errors="coerce")
    if c.isna().all():
        base["note"] = "收盘数据缺失"
        return base
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    ma20 = c.rolling(20, min_periods=20).mean()
    hist_t, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    hist_up = bool(hist_t >= hist_prev)
    close_t = float(c.iloc[-1])
    ma20_t = float(ma20.iloc[-1])
    above = bool(np.isfinite(ma20_t) and close_t >= ma20_t)
    trend = "震荡"
    if hist_up and above:
        trend = "多"
    elif (not hist_up) and (not above):
        trend = "空"
    return {"trend": trend, "macd": float(macd.iloc[-1]), "signal": float(signal.iloc[-1]),
            "hist": hist_t, "hist_up": hist_up, "ma20": ma20_t,
            "close_vs_ma20": "上" if above else "下", "note": ""}


def _screen_short(close: pd.Series) -> dict:
    """第三屏 小周期（近5日）: 收盘 vs MA5 + 近5日斜率（日内动量代理）。"""
    base = {"trend": "样本不足", "ma5": np.nan, "slope_pct": np.nan,
            "close_vs_ma5": None, "note": ""}
    if close is None or len(close) < 6:
        base["note"] = "近5日样本不足"
        return base
    c = pd.to_numeric(close, errors="coerce")
    ma5 = c.rolling(5, min_periods=5).mean()
    close_t = float(c.iloc[-1])
    ma5_t = float(ma5.iloc[-1])
    seg = c.iloc[-5:].to_numpy(dtype=float)
    slope_pct = np.nan
    if np.all(np.isfinite(seg)) and np.isfinite(ma5_t) and ma5_t > 0:
        slope_pct = float(np.polyfit(np.arange(5), seg, 1)[0]) / ma5_t * 100.0
    above = bool(np.isfinite(ma5_t) and close_t >= ma5_t)
    trend = "震荡"
    if above and np.isfinite(slope_pct) and slope_pct >= 0:
        trend = "多"
    elif (not above) and np.isfinite(slope_pct) and slope_pct <= 0:
        trend = "空"
    return {"trend": trend, "ma5": ma5_t, "slope_pct": slope_pct,
            "close_vs_ma5": "上" if above else "下", "note": ""}


def _composite(s1: dict, s2: dict, s3: dict, tl: dict | None = None) -> dict:
    """综合评级: 三屏同向=强趋势 / 两屏同向=中等 / 混乱=震荡 / 任一屏不足=样本不足。"""
    votes = {"多": 0, "空": 0, "震荡": 0}
    for s in (s1, s2, s3):
        k = s["trend"] if s["trend"] in votes else "震荡"
        votes[k] += 1
    if any(s["trend"] == "样本不足" for s in (s1, s2, s3)):
        return {"rating": "样本不足", "votes": votes, "confidence": 0.0,
                "note": "任一屏样本不足，结果仅供参考"}
    m, b = votes["多"], votes["空"]
    if m == 3:
        rating, conf = "强趋势多", 0.85
    elif b == 3:
        rating, conf = "强趋势空", 0.85
    elif m == 2:
        rating, conf = "中等多", 0.65
    elif b == 2:
        rating, conf = "中等空", 0.65
    else:
        rating, conf = "震荡", 0.40
    note = ""
    if m == 2 and b == 1:
        note = "两屏多与一屏空冲突，潮汐偏多但存分歧"
    elif b == 2 and m == 1:
        note = "两屏空与一屏多冲突，潮汐偏空但存分歧"
    # 趋势线同向突破且已确认（Murphy）→ 上调置信
    if tl is not None:
        if rating.endswith("多") and tl.get("break_up_confirmed"):
            conf = min(conf + 0.10, 0.95)
        elif rating.endswith("空") and tl.get("break_down_confirmed"):
            conf = min(conf + 0.10, 0.95)
    return {"rating": rating, "votes": votes, "confidence": round(conf, 2), "note": note}


def _trendline(highs, lows, close=None) -> dict:
    """pivot 高低点 → 线性回归趋势线（复用 pattern_engine._trendline，避免阈值漂移）。

    pattern_engine._trendline 同签名（highs/lows/close），核心 pivot/回归/斜率阈值
    单点维护；此处仅补本模块需要的 level_prev 与 Murphy 突破确认字段。
    """
    from quant_system.analysis_core.pattern_engine import _trendline as _pe_trendline

    h = np.asarray(pd.to_numeric(highs, errors="coerce").to_numpy(dtype=float), dtype=float)
    n = len(h)
    out = {"up": {"slope": np.nan, "rel_slope": np.nan, "level": np.nan,
                  "level_prev": np.nan, "n": 0},
           "down": {"slope": np.nan, "rel_slope": np.nan, "level": np.nan,
                    "level_prev": np.nan, "n": 0},
           "break_up": False, "break_down": False,
           "break_up_confirmed": False, "break_down_confirmed": False,
           "break_up_gap_pct": np.nan, "break_down_gap_pct": np.nan,
           "note": ""}
    if n < 30:
        out["note"] = "趋势线样本不足(<30日)"
        return out

    tl = _pe_trendline(highs, lows, close)
    for key in ("up", "down"):
        for k, v in (tl.get(key) or {}).items():
            out[key][k] = v
        out[key].setdefault("rel_slope", np.nan)
        out[key].setdefault("anchor_idx", np.nan)
        if np.isfinite(out[key].get("level")) and np.isfinite(out[key].get("slope")):
            out[key]["level_prev"] = float(out[key]["level"] - out[key]["slope"])
    out["break_up"] = bool(tl.get("break_up"))
    out["break_down"] = bool(tl.get("break_down"))
    out["note"] = tl.get("note") or out["note"]

    if close is not None and n >= 2:
        cc = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
        if len(cc) == n:
            c_t, c_prev = float(cc[-1]), float(cc[-2])
            dl, du = out["down"], out["up"]
            if np.isfinite(dl["level"]) and dl["level"] > 0:
                gap = c_t / dl["level"] - 1.0
                out["break_up"] = bool(c_t > dl["level"])
                out["break_up_gap_pct"] = round(gap * 100, 2) if out["break_up"] else np.nan
                out["break_up_confirmed"] = bool(
                    out["break_up"] and (gap >= CONFIRM_PCT or
                                         (np.isfinite(dl["level_prev"]) and c_prev > dl["level_prev"])))
            if np.isfinite(du["level"]) and du["level"] > 0:
                gap = du["level"] / c_t - 1.0
                out["break_down"] = bool(c_t < du["level"])
                out["break_down_gap_pct"] = round(gap * 100, 2) if out["break_down"] else np.nan
                out["break_down_confirmed"] = bool(
                    out["break_down"] and (gap >= CONFIRM_PCT or
                                           (np.isfinite(du["level_prev"]) and c_prev < du["level_prev"])))
    return out

def _trendline_signals(code: str, name: str, tl: dict) -> list[dict]:
    """趋势线突破信号（含 Murphy 确认状态 + 支撑阻力互换说明）。"""
    sigs: list[dict] = []
    if tl.get("break_up") and np.isfinite(tl["down"].get("level")):
        confirmed = bool(tl.get("break_up_confirmed"))
        sigs.append({
            "type": "trendline_breakout_up", "direction": "多",
            "code": code, "name": name,
            "level": round(float(tl["down"]["level"]), 2),
            "rel_slope_pct": round(float(tl["down"]["rel_slope"]) * 100, 3),
            "gap_pct": tl.get("break_up_gap_pct"),
            "confirmed": confirmed,
            "note": "下降趋势线突破" + ("（>3%或2日收盘确认，原阻力转支撑）" if confirmed
                                    else "（待>3%或2日收盘确认）"),
        })
    if tl.get("break_down") and np.isfinite(tl["up"].get("level")):
        confirmed = bool(tl.get("break_down_confirmed"))
        sigs.append({
            "type": "trendline_breakout_down", "direction": "空",
            "code": code, "name": name,
            "level": round(float(tl["up"]["level"]), 2),
            "rel_slope_pct": round(float(tl["up"]["rel_slope"]) * 100, 3),
            "gap_pct": tl.get("break_down_gap_pct"),
            "confirmed": confirmed,
            "note": "上升趋势线跌破" + ("（>3%或2日收盘确认，原支撑转阻力）" if confirmed
                                    else "（待>3%或2日收盘确认）"),
        })
    return sigs


def detect_symbol(df: pd.DataFrame, code: str, name: str, is_index: bool = False) -> dict:
    """单标的（个股/指数）三屏检测。任何异常 → ok=False + error，不抛断。"""
    out = {"code": code, "name": name, "is_index": is_index, "ok": False,
           "error": "", "date": None, "close": np.nan, "pct_chg": np.nan,
           "weekly": None, "daily": None, "short": None, "trendline": None,
           "rating": "样本不足", "confidence": 0.0, "votes": {},
           "composite_note": "", "signals": [], "sample_note": ""}
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        if df.empty:
            out["error"] = "无数据"
            out["sample_note"] = "无数据"
            return out
        if len(df) < MIN_DAILY_BARS:
            out["error"] = "样本不足"
            out["sample_note"] = f"日线{len(df)}根<{MIN_DAILY_BARS}"
            return out
        close = pd.to_numeric(df["close"], errors="coerce")
        s1 = _screen_weekly(df)
        s2 = _screen_daily(close)
        s3 = _screen_short(close)
        tl = _trendline(df["high"], df["low"], df["close"])
        comp = _composite(s1, s2, s3, tl)
        close_t = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else np.nan
        pct = (close_t / prev - 1.0) * 100.0 if np.isfinite(prev) and prev > 0 else np.nan
        out.update({
            "ok": True,
            "date": str(df["date"].iloc[-1].date()),
            "close": round(close_t, 3),
            "pct_chg": round(pct, 2) if np.isfinite(pct) else np.nan,
            "weekly": s1, "daily": s2, "short": s3, "trendline": tl,
            "rating": comp["rating"], "confidence": comp["confidence"],
            "votes": comp["votes"], "composite_note": comp["note"],
            "signals": _trendline_signals(code, name, tl),
            "sample_note": "；".join(n for n in (s1["note"], s2["note"], s3["note"], tl["note"]) if n) or "",
        })
    except Exception as e:  # 单标的失败跳过不崩
        out["error"] = str(e)[:120]
        out["sample_note"] = f"检测异常: {str(e)[:80]}"
    return out


# ────────────────────────────────────────────────────────────
# TrendSystem 统一接口
# ────────────────────────────────────────────────────────────
class TrendSystem:
    """三屏趋势系统：detect() 检测 / report() 写 md / view() 输出 multi_agent 兼容观点。"""

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self._bench: tuple[pd.DataFrame | None, str] | None = None
        self._names: tuple[dict[str, str], dict[str, bool]] | None = None
        self._rag_cache: dict | None = None

    # ── 内部：懒加载 ─────────────────────────────────────
    def _benchmark(self) -> tuple[pd.DataFrame | None, str]:
        if self._bench is None:
            self._bench = load_benchmark_df()
        return self._bench

    def _name_map(self) -> dict[str, str]:
        if self._names is None:
            self._names = load_names()
        return self._names[0]

    def _rag(self) -> dict:
        """RAG 方法论依据: knowledge_rag.search('三屏趋势 突破 确认', k=3)，失败→检索不可用。"""
        if self._rag_cache is not None:
            return self._rag_cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
        except Exception as e:
            self._rag_cache = {"query": RAG_QUERY, "available": False, "hits": [],
                               "basis": "检索不可用", "error": str(e)[:120]}
        return self._rag_cache

    # ── detect ───────────────────────────────────────────
    def detect(self, date: str | None = None, watch: list[str] | None = None,
               limit: int | None = None) -> dict:
        """三屏检测。watch 非空→自选股；否则全A（limit 抽样）。返回 {date, rows, meta, rag}。"""
        t0 = time.time()
        bench_df, bench_note = self._benchmark()
        names = self._name_map()
        watch_codes = [str(c).strip().zfill(6) for c in (watch or []) if str(c).strip()]
        if watch_codes:
            files = [KLINE_DIR / f"{c}.parquet" for c in watch_codes]
            universe = "watch"
        else:
            files = kline_files(limit)
            universe = "fullA_sample" if limit else "fullA"

        target = self._resolve_target(date, files)
        window_start = target - pd.Timedelta(days=LOOKBACK_DAYS)

        meta = {
            "target_date": str(target.date()), "universe": universe,
            "files_total": len(files), "parsed": 0, "skip_read_error": 0,
            "skip_schema": 0, "skip_empty": 0, "skip_no_data": 0,
            "insufficient": 0, "benchmark": bench_note,
            "benchmark_available": False, "benchmark_as_of": None,
            "benchmark_lag_days": None, "elapsed_sec": 0.0,
        }
        rows: list[dict] = []
        for i, f in enumerate(files, 1):
            code = f.stem
            name = names.get(code, code)
            if not f.exists():
                meta["skip_read_error"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": "kline文件缺失"})
                continue
            try:
                df = read_kline_window(f, ["date", "open", "high", "low", "close"],
                                       pd.Timestamp(window_start))
            except Exception:
                meta["skip_read_error"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": "读取失败"})
                continue
            df = _norm_kline(df, target)
            if df is None:
                meta["skip_empty"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": "无数据"})
                continue
            if not (df["date"] == target).any():
                meta["skip_no_data"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": "目标日无交易"})
                continue
            res = detect_symbol(df, code, name)
            rows.append(res)
            meta["parsed"] += 1
            if not res["ok"]:
                meta["insufficient"] += 1
            if i % PROGRESS_EVERY == 0:
                print(f"[进度] {i}/{len(files)} | 已读 {meta['parsed']} | "
                      f"耗时 {time.time() - t0:.0f}s", flush=True)

        # 基准指数同样三屏检测（每只/每个指数）
        if bench_df is not None and len(bench_df):
            bdf = bench_df[bench_df["date"] <= target]
            if len(bdf):
                bench_asof = bdf["date"].max()
                if pd.notna(bench_asof):
                    meta["benchmark_as_of"] = str(pd.Timestamp(bench_asof).date())
                    meta["benchmark_lag_days"] = int(max(0, (target - pd.Timestamp(bench_asof)).days))
            if len(bdf) >= MIN_DAILY_BARS:
                bdf = bdf.tail(LOOKBACK_DAYS * 2)  # 控制窗口，避免超长历史
                bres = detect_symbol(bdf, BENCH_CODE, BENCH_NAME, is_index=True)
                rows.append(bres)
                meta["benchmark_available"] = True
                if not bres["ok"]:
                    meta["insufficient"] += 1

        meta["elapsed_sec"] = round(time.time() - t0, 1)
        return {"date": str(target.date()), "rows": rows, "meta": meta, "rag": self._rag()}

    def _resolve_target(self, date: str | None, files: list[Path]) -> pd.Timestamp:
        """目标日期：--date 优先；否则基准与样本K线的最新共同交易日。"""
        if date:
            return pd.Timestamp(date).normalize()
        bench_df, _ = self._benchmark()
        bench_max = bench_df["date"].max() if bench_df is not None and len(bench_df) else None
        kline_max = None
        probe = files[: min(30, len(files))] if files else []
        for f in probe:
            try:
                d = pd.read_parquet(f, columns=["date"])
                m = pd.to_datetime(d["date"], errors="coerce").max()
                if m is not None and (kline_max is None or m > kline_max):
                    kline_max = m
            except Exception as e:
                logging.getLogger(__name__).error(f"[trend_system] 操作失败: {e}", exc_info=True)
                continue
        cands = [x for x in (bench_max, kline_max) if x is not None]
        if not cands:
            return pd.Timestamp("today").normalize()
        return min(cands).normalize()

    # ── report ───────────────────────────────────────────
    def report(self, date: str | None = None, watch: list[str] | None = None,
               limit: int | None = None, out_dir: str | Path | None = None) -> Path:
        """检测并写 generated/trend_report_{date}.md，返回文件路径。"""
        res = self.detect(date=date, watch=watch, limit=limit)
        out = Path(out_dir) if out_dir else self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"trend_report_{res['date']}.md"
        path.write_text(render_markdown(res), encoding="utf-8")
        return path

    # ── view ─────────────────────────────────────────────
    def view(self, date: str | None = None, watch: list[str] | None = None,
             code: str | None = None) -> dict:
        """multi_agent 兼容观点 {agent, signal, confidence, evidence}。

        默认输出市场潮汐观点（沪深300 三屏 + 自选股同向率）；
        code 指定时输出该标的的三屏观点。任何异常降级为震荡，不崩溃。
        """
        try:
            res = self.detect(date=date, watch=watch,
                              limit=None if (watch or code) else 0)
        except Exception as e:
            return self._degraded_view(f"检测异常: {str(e)[:120]}")
        rows = res["rows"]
        rag_basis = res.get("rag", {}).get("basis", "检索不可用")

        if code:
            row = next((r for r in rows if r.get("code") == str(code).strip().zfill(6)), None)
            if row is None or not row.get("ok"):
                return self._degraded_view(f"{code} 无有效三屏结果", rag_basis=rag_basis,
                                           date=res["date"])
            signal = _rating_signal(row["rating"])
            evidence = self._evidence_from_row(row)
            evidence.append(f"RAG: {rag_basis}")
            return {
                "agent": "三屏趋势", "signal": signal, "view": signal,
                "confidence": float(row.get("confidence", 0.4)),
                "evidence": evidence, "status": "ok",
                "weight": VIEW_WEIGHT, "object_code": row["code"],
                "object_name": row["name"], "detail": {"date": res["date"], "rating": row["rating"]},
                "rag_basis": rag_basis,
            }

        bench_row = next((r for r in rows if r.get("is_index") and r.get("ok")), None)
        if bench_row is None:
            ok_rows = [r for r in rows if r.get("ok")]
            if not ok_rows:
                return self._degraded_view("无可用三屏样本", rag_basis=rag_basis, date=res["date"])
            cnt: dict[str, int] = {}
            for r in ok_rows:
                s = _rating_signal(r.get("rating"))
                cnt[s] = cnt.get(s, 0) + 1
            signal = max(cnt, key=cnt.get)
            share = cnt[signal] / len(ok_rows)
            conf = round(0.3 + 0.4 * share, 2)
            evidence = [f"基准不可用，按 {len(ok_rows)} 只样本多数票: {signal} "
                        f"({cnt[signal]}/{len(ok_rows)})", f"RAG: {rag_basis}"]
            return {"agent": "三屏趋势", "signal": signal, "view": signal,
                    "confidence": conf, "evidence": evidence, "status": "degraded",
                    "weight": VIEW_WEIGHT, "detail": {"date": res["date"],
                                                      "universe": res["meta"].get("universe")},
                    "rag_basis": rag_basis}

        signal = _rating_signal(bench_row["rating"])
        conf = float(bench_row.get("confidence", 0.4))
        stock_rows = [r for r in rows if not r.get("is_index") and r.get("ok")]
        agree = [r for r in stock_rows if _rating_signal(r.get("rating")) == signal]
        pct = (len(agree) / len(stock_rows)) if stock_rows else 0.5
        conf = round(min(0.95, conf + 0.1 * pct), 2)
        evidence = [
            f"市场潮汐({bench_row['name']}): 大周期{fmt_trend(bench_row['weekly']['trend'])} "
            f"| 中周期{fmt_trend(bench_row['daily']['trend'])} "
            f"| 小周期{fmt_trend(bench_row['short']['trend'])} → {bench_row['rating']}",
            f"自选/样本 {len(stock_rows)} 只中 {len(agree)} 只与潮汐同向（{pct:.0%}）",
        ]
        evidence.extend(self._evidence_from_row(bench_row)[:4])
        bench_lag = res["meta"].get("benchmark_lag_days") or 0
        if bench_lag > 0:
            evidence.append(f"⚠️ 基准 as_of 落后目标日 {bench_lag} 天"
                            f"（基准 {res['meta'].get('benchmark_as_of')} / 目标 {res['date']}）")
        evidence.append(f"RAG: {rag_basis}")
        return {
            "agent": "三屏趋势", "signal": signal, "view": signal,
            "confidence": conf, "evidence": evidence, "status": "ok",
            "weight": VIEW_WEIGHT,
            "detail": {"date": res["date"], "universe": res["meta"].get("universe"),
                       "benchmark": res["meta"].get("benchmark"), "rows": len(rows),
                       "rating": bench_row["rating"],
                       "benchmark_as_of": res["meta"].get("benchmark_as_of"),
                       "benchmark_lag_days": bench_lag},
            "rag_basis": rag_basis,
        }

    def _degraded_view(self, msg: str, rag_basis: str = "检索不可用",
                       date: str | None = None) -> dict:
        return {"agent": "三屏趋势", "signal": "震荡", "view": "震荡", "confidence": 0.0,
                "evidence": [msg, f"RAG: {rag_basis}"], "status": "degraded",
                "weight": VIEW_WEIGHT,
                "detail": {"date": date or ""}, "rag_basis": rag_basis}

    @staticmethod
    def _evidence_from_row(row: dict) -> list[str]:
        ev: list[str] = []
        w, d, s = row.get("weekly"), row.get("daily"), row.get("short")
        if w:
            ev.append(f"大周期(周线): MA20w={_fmt_num(w.get('ma20'))} MA60w={_fmt_num(w.get('ma60'))} "
                      f"斜率{_fmt_slope(w.get('slope_pct'))}%/周 → {fmt_trend(w.get('trend'))}")
        if d:
            ev.append(f"中周期(日线): MACD柱{'+' if d.get('hist_up') else '-'}"
                      f"({_fmt_num(d.get('hist'))}) 收盘vs MA20: {d.get('close_vs_ma20')} "
                      f"→ {fmt_trend(d.get('trend'))}")
        if s:
            ev.append(f"小周期(近5日): 收盘vs MA5: {s.get('close_vs_ma5')} "
                      f"斜率{_fmt_slope(s.get('slope_pct'))}%/日 → {fmt_trend(s.get('trend'))}")
        ev.append(f"综合评级: {row.get('rating')}（置信 {row.get('confidence')}）")
        for sig in (row.get("signals") or [])[:3]:
            ev.append(f"趋势线: {sig['note']}（线位 {sig['level']}）")
        return ev


# ────────────────────────────────────────────────────────────
# 输出
# ────────────────────────────────────────────────────────────
def _rating_signal(rating: str) -> str:
    if rating in ("强趋势多", "中等多"):
        return "多"
    if rating in ("强趋势空", "中等空"):
        return "空"
    return "震荡"


def fmt_trend(t) -> str:
    return str(t) if t in ("多", "空", "震荡", "样本不足") else "—"


def _fmt_num(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"


def _fmt_slope(v) -> str:
    return f"{v:+.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"


def _fmt_gap(v) -> str:
    return f"{v:+.1f}%" if isinstance(v, (int, float)) and np.isfinite(v) else "—"


def _screen_brief(row: dict, key: str) -> str:
    s = row.get(key) or {}
    return fmt_trend(s.get("trend"))


def _weekly_detail(w: dict) -> str:
    return (f"MA20w {_fmt_num(w.get('ma20'))} vs MA60w {_fmt_num(w.get('ma60'))}, "
            f"MA20斜率 {_fmt_slope(w.get('slope_pct'))}%/周")


def _daily_detail(d: dict) -> str:
    return (f"MACD柱{'上行' if d.get('hist_up') else '下行'}({_fmt_num(d.get('hist'))}), "
            f"收盘{'上' if d.get('close_vs_ma20') == '上' else '下'}MA20 {_fmt_num(d.get('ma20'))}")


def _short_detail(s: dict) -> str:
    return (f"收盘{'上' if s.get('close_vs_ma5') == '上' else '下'}MA5 {_fmt_num(s.get('ma5'))}, "
            f"MA5斜率 {_fmt_slope(s.get('slope_pct'))}%/日")


def render_markdown(res: dict) -> str:
    """渲染 trend_report_{date}.md。"""
    meta = res["meta"]
    rows = res["rows"]
    date = res["date"]
    bench_row = next((r for r in rows if r.get("is_index")), None)
    lines = [f"# 三屏趋势系统（体系2）— {date}", ""]
    lines += [
        f"- 基准: {meta['benchmark']}（{'可用' if meta['benchmark_available'] else '不可用'}"
        + (f"，as_of 落后 {meta['benchmark_lag_days']} 天" if (meta.get('benchmark_lag_days') or 0) > 0 else "") + "）",
        f"- 范围: {meta['universe']}（{meta['files_total']} 只）→ 有效 {meta['parsed']} | "
        f"读失败 {meta['skip_read_error']} | 无数据 {meta['skip_empty']} | "
        f"当日无交易 {meta['skip_no_data']} | 样本不足 {meta['insufficient']}",
        f"- 耗时: {meta['elapsed_sec']}s",
        "",
        "## 方法论依据（RAG）", "",
    ]
    rag = res.get("rag") or {}
    lines.append(f"查询: `{rag.get('query', RAG_QUERY)}` → {rag.get('basis', '检索不可用')}")
    if rag.get("basis") == "检索可用":
        for h in rag.get("hits", []):
            lines.append(f"- [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')} — "
                         f"{str(h.get('summary') or '')[:120].replace(chr(10), ' ')}")
    elif rag.get("error"):
        lines.append(f"- 检索异常: {rag.get('error')}")
    lines.append("")

    lines += ["## 市场潮汐（沪深300 三屏）", ""]
    if bench_row is None or not bench_row.get("ok"):
        lines.append("- 基准不可用/样本不足，潮汐方向以个股多数票替代。", "")
    else:
        bw, bd, bs = bench_row.get("weekly") or {}, bench_row.get("daily") or {}, bench_row.get("short") or {}
        lines += [
            "| 层次 | 方向 | 依据 |",
            "|---|---:|---|",
            f"| 大周期(周线) | {fmt_trend(bw.get('trend'))} | {_weekly_detail(bw)} |",
            f"| 中周期(日线) | {fmt_trend(bd.get('trend'))} | {_daily_detail(bd)} |",
            f"| 小周期(近5日) | {fmt_trend(bs.get('trend'))} | {_short_detail(bs)} |",
        ]
        lines += ["", "> 综合评级: **%s**（置信 %.2f）%s" % (
            bench_row["rating"], bench_row.get("confidence", 0.0),
            f"，{bench_row['composite_note']}" if bench_row.get("composite_note") else "")]
        lines.append("")

    lines += ["## 标的（自选/抽样）三屏状态", "",
              "| 代码 | 名称 | 收盘 | 涨跌% | 大周期(周线) | 中周期(日线) | 小周期(近5日) | "
              "综合评级 | 置信 | 趋势线信号 |", "|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---|"]
    stock_rows = [r for r in rows if not r.get("is_index")]
    for r in stock_rows:
        if r.get("missing"):
            lines.append(f"| {r['code']} | {r['name']} | — | — | — | — | — | 无数据 | — | {r.get('error', '')} |")
            continue
        tl_txt = ""
        for sig in r.get("signals", []):
            tl_txt += f"{'确认' if sig['confirmed'] else '待确认'}突破({sig['level']}) "
        if r.get("sample_note"):
            tl_txt += f"⚠{r['sample_note'][:40]}"
        lines.append(
            f"| {r['code']} | {r['name']} | {_fmt_num(r.get('close'))} | {_fmt_slope(r.get('pct_chg'))} | "
            f"{_screen_brief(r, 'weekly')} | {_screen_brief(r, 'daily')} | {_screen_brief(r, 'short')} | "
            f"{r.get('rating', '—')} | {r.get('confidence', 0.0):.2f} | {tl_txt.strip() or '—'} |"
        )
    lines.append("")

    lines += ["## 趋势线突破信号（Murphy 确认）", ""]
    sig_rows = [sig for r in rows if r.get("ok") for sig in r.get("signals", [])]
    if not sig_rows:
        lines.append("- 无突破信号。")
    else:
        for sig in sig_rows:
            lines.append(f"- {sig['name']}({sig['code']}): {sig['note']}，"
                         f"线位 {sig['level']}（斜率 {_fmt_slope(sig['rel_slope_pct'])}%/日，"
                         f"超出 {_fmt_gap(sig.get('gap_pct'))}）")
    lines.append("")

    insufficient = [r for r in stock_rows if r.get("ok") and r.get("rating") == "样本不足"]
    if insufficient:
        lines += ["## 样本不足标注", ""]
        for r in insufficient:
            lines.append(f"- {r['name']}({r['code']}): {r.get('sample_note') or '样本不足'}")
        lines.append("")

    lines += ["---",
               "*方法: 大周期=周线重采样MA20/MA60排列+MA20近5周斜率；中周期=日线MACD(12,26,9)柱方向"
               "+收盘vs MA20；小周期=近5日收盘vs MA5+斜率；三屏同向=强趋势/两屏=中等/混乱=震荡；"
               "趋势线=pivot高低点线性回归，突破需>3%或2日收盘确认(Murphy)，突破后支撑阻力互换。*",
               ""]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="三屏趋势系统（Elder三重滤网 / Murphy突破确认 / Trader Vic趋势诊断）")
    ap.add_argument("--watch", default="", help="自选股池，逗号分隔代码，如 601899,600519")
    ap.add_argument("--limit", type=int, default=0, help="全A抽样前 N 个文件（0=全量；无 --watch 时生效）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最近共同交易日）")
    ap.add_argument("--report", action="store_true", help="写入 generated/trend_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    ap.add_argument("--view", action="store_true", help="输出 multi_agent 兼容 view JSON")
    args = ap.parse_args()

    ts = TrendSystem(out_dir=args.out_dir)
    watch_codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
    res = ts.detect(date=args.date, watch=watch_codes or None, limit=args.limit or None)
    meta = res["meta"]
    print(f"[信息] 目标日期 {res['date']} | 基准 {meta['benchmark']} | "
          f"{meta['universe']} {meta['files_total']} 只 | 有效 {meta['parsed']} | "
          f"耗时 {meta['elapsed_sec']}s", flush=True)

    stock_rows = [r for r in res["rows"] if not r.get("is_index")]
    print("\n[三屏状态] 每标的大周期/中周期/小周期/综合评级:")
    for r in stock_rows:
        if r.get("missing"):
            print(f"  {r['code']} {r['name']:<8} 无数据/目标日无交易（{r.get('error', '')}）")
            continue
        print(f"  {r['code']} {r['name']:<8} 收{_fmt_num(r.get('close'))} "
              f"大周期:{_screen_brief(r, 'weekly')} 中周期:{_screen_brief(r, 'daily')} "
              f"小周期:{_screen_brief(r, 'short')} → {r.get('rating')}(置信{r.get('confidence'):.2f})"
              + (f" ⚠{r.get('sample_note', '')[:50]}" if r.get("sample_note") else ""))

    bench_row = next((r for r in res["rows"] if r.get("is_index")), None)
    if bench_row is not None:
        print(f"\n[市场潮汐] {bench_row['name']}: 大周期:{_screen_brief(bench_row, 'weekly')} "
              f"中周期:{_screen_brief(bench_row, 'daily')} 小周期:{_screen_brief(bench_row, 'short')} "
              f"→ {bench_row.get('rating')}(置信{bench_row.get('confidence'):.2f})")

    sigs = [sig for r in stock_rows if r.get("ok") for sig in r.get("signals", [])]
    if sigs:
        print("\n[趋势线突破] Murphy 确认:")
        for sig in sigs:
            print(f"  {sig['name']}({sig['code']}): {sig['note']} 线位 {sig['level']} "
                  f"(超出 {_fmt_gap(sig.get('gap_pct'))})")

    rag = res.get("rag") or {}
    print(f"\n[RAG] {rag.get('query', '')} → {rag.get('basis', '检索不可用')}")
    for h in rag.get("hits", [])[:RAG_K]:
        print(f"  [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')}")

    if args.report:
        path = ts.report(date=args.date, watch=watch_codes or None,
                         limit=args.limit or None, out_dir=args.out_dir)
        print(f"\n已保存: {path}")

    if args.view:
        v = ts.view(date=args.date, watch=watch_codes or None)
        print(f"\n[VIEW] {json.dumps(v, ensure_ascii=False, indent=2, default=str)}")


if __name__ == "__main__":
    main()
