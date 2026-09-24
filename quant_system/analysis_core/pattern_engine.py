#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern_engine — 自主学习者·规律引擎（五层架构中的 ②规律层/③验证层/④知识层）

架构（见 项目文档/量化交易系统/2026-08-11_自主学习者体系设计.md）:
  ① 信号层(现有模块) → ② 规律层 detect_patterns() 提取可复述规律
  → ③ 验证层 validate_pattern() 历史回放胜率
  → ④ 知识层 data_warehouse/patterns/knowledge_base.parquet 持久化+新鲜度
  → ⑤ 决策层 multi_agent 消费规律库（外部接入）

规律类型（首批 8 种，各配 skill 依据）:
  stagflation        滞涨避险模式            animal-spirits-risk-forecasting
  dense_zone         交易密集区博弈          volume-price-analysis + trade-survival
  support_resistance 支撑压力位(三重确认)    livermore-pivotal-point-breakout + elder
  trendline_breakout 趋势线突破             trader-vic-trend-change-diagnosis
  divergence         量价背离               volume-price-analysis
  style_switch       风格切换               a-share-market-state-monitor
  breakout_volume    新高放量突破           master-trading-carter
  sector_resonance   板块共振               a-share-market-state-monitor

复用模块（import 不重写）:
  macro_overseas.get_macro_overseas        — 滞涨模式
  volume_profile.load_kline_60d / compute_volume_profile — 密集区/支撑压力
  breakout_watch.scan / load_names        — 新高放量突破
  rs_strength.load_benchmark / compute_rs_metrics — 趋势确认
  style_spread.load_index / index_stats / judge_style — 风格切换
  fund_flow_divergence.local_pct_chg / fetch_fund_flow_by_code / judge_divergence — 量价背离
  resonance_scorer.score_day              — 板块共振

数据契约:
  data_warehouse/kline/{6位}.parquet        日K（只读）
  data_warehouse/market/index_daily_*.parquet 指数日线
  data_warehouse/realtime_snapshot/{YYYYMMDD}/*.parquet 当日全市场快照（宇宙选样）
  data_warehouse/patterns/knowledge_base.parquet  规律库（增量 upsert）
  data_warehouse/patterns/feedback.jsonl         人工标注通道

用法:
  python3 -m quant_system.analysis_core.pattern_engine --report            # 今日检测+验证+报告
  python3 -m quant_system.analysis_core.pattern_engine --date 2026-08-11 --report
  python3 -m quant_system.analysis_core.pattern_engine --validate <pattern_id>
  python3 -m quant_system.analysis_core.pattern_engine --weekly           # 周度规律库复盘
  python3 -m quant_system.analysis_core.pattern_engine --mark <pattern_id> --action confirm --note "..."
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import (  # noqa: E402
    breakout_watch,
    fund_flow_divergence,
    macro_overseas,
    rs_strength,
    style_spread,
    volume_profile,
)
from quant_system.analysis_core.common import fmt, today  # noqa: E402

CST = timezone(timedelta(hours=8))
_WINDOW_CACHE: dict[tuple[str, str, str], pd.DataFrame | None] = {}
_SCORE_DAY_CACHE: dict[str, pd.DataFrame] = {}
KLINE_DIR = ROOT / "data_warehouse" / "kline"
MARKET_DIR = ROOT / "data_warehouse" / "market"
SNAPSHOT_DIR = ROOT / "data_warehouse" / "realtime_snapshot"
PATTERNS_DIR = ROOT / "data_warehouse" / "patterns"
KB_PATH = PATTERNS_DIR / "knowledge_base.parquet"
FEEDBACK_PATH = PATTERNS_DIR / "feedback.jsonl"
EXPLANATION_CACHE_PATH = PATTERNS_DIR / "pattern_explanations.json"
OUT_DIR = ROOT / "generated"

# ── RAG 规律逻辑解释 ───────────────────────────────────────
RAG_QUERY_K = 3                       # 每类规律检索 top-k 命中
_DIRECTION_CN = {"up": "看多向上", "down": "看空向下", "risk_off": "风险规避防守",
                 "defensive": "防守", "sideways": "震荡横盘", "neutral": "中性震荡",
                 "oscillation": "震荡", "small": "小盘风格", "large": "大盘风格"}
WATCHLIST_PATH = ROOT / "quant_system" / "config" / "watchlist.json"

# ── 规律类型登记表 ────────────────────────────────────────────
PATTERN_SPECS: list[dict] = [
    {"type": "stagflation", "name": "滞涨避险模式(金油同涨+美债高)",
     "skill_ref": "animal-spirits-risk-forecasting", "scope": "market"},
    {"type": "dense_zone", "name": "交易密集区博弈(上沿突破/假突破)",
     "skill_ref": "volume-price-analysis + trade-survival", "scope": "stock"},
    {"type": "support_resistance", "name": "支撑压力位(前高前低+密集区+趋势线三重确认)",
     "skill_ref": "livermore-pivotal-point-breakout + elder", "scope": "stock"},
    {"type": "trendline_breakout", "name": "趋势线突破(升降趋势线+量能确认)",
     "skill_ref": "trader-vic-trend-change-diagnosis", "scope": "stock"},
    {"type": "divergence", "name": "量价背离(涨但主力流出/跌但主力流入)",
     "skill_ref": "volume-price-analysis", "scope": "stock"},
    {"type": "style_switch", "name": "风格切换(大小盘剪刀差极端+反转)",
     "skill_ref": "a-share-market-state-monitor", "scope": "market"},
    {"type": "breakout_volume", "name": "新高放量突破(250日新高+量能1.5x)",
     "skill_ref": "master-trading-carter", "scope": "market"},
    {"type": "sector_resonance", "name": "板块共振(行业广度+领涨领跌一致性)",
     "skill_ref": "a-share-market-state-monitor", "scope": "market"},
]
TYPE_SPEC = {s["type"]: s for s in PATTERN_SPECS}

# ── 阈值（与现有模块口径对齐）─────────────────────────────────
CONFIRM_SAMPLES = 10        # confirmed 最少样本
CONFIRM_WIN_RATE = 0.55     # confirmed 5日胜率门槛
VALIDATE_SAMPLES = 3        # validating 最少样本
DEPRECATE_WIN_RATE = 0.50   # 周度：近期胜率低于此且样本足够 → deprecated
NEAR_PCT = 0.015            # 支撑/压力贴近判定 1.5%
CONFLUENT_PCT = 0.008       # 多层级汇合判定 0.8%
TREND_VOL_MULT = 1.2        # 趋势线突破量能确认
ZONE_VOL_MULT = 1.5         # 密集区放量阈值
ZONE_SHRINK = 0.8           # 密集区缩量阈值
DEFAULT_LOOKBACK_DAYS = 120
SAMPLE_EVERY = 5            # 回放采样间隔（交易日）
DEFAULT_UNIVERSE_LIMIT = 120
DEFAULT_SCAN_LIMIT = 2500   # breakout_watch 全A抽样上限
DIVERGENCE_TIMEOUT = 4.0

KB_COLUMNS = [
    "pattern_id", "pattern_type", "name", "signal_date", "confidence",
    "evidence", "skill_ref", "status", "backtest_stats",
    "object_code", "object_name", "direction", "source_note",
    "first_seen", "last_seen", "updated_at", "feedback",
    "n_obs_5", "win_rate_5", "avg_ret_5", "n_obs_10", "win_rate_10", "avg_ret_10",
]


# ────────────────────────────────────────────────────────────
# 基础工具
# ────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now(CST)


def _snap_date(target: pd.Timestamp) -> str:
    return target.strftime("%Y%m%d")


def _load_kb() -> pd.DataFrame:
    if not KB_PATH.exists():
        return pd.DataFrame(columns=KB_COLUMNS)
    try:
        df = pd.read_parquet(KB_PATH)
        for c in KB_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[KB_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=KB_COLUMNS)


def load_knowledge_base(date: str | None = None) -> pd.DataFrame:
    """公开读取规律库（供 multi_agent 规律面 / 决策层消费）。

    date 指定时仅返回该 signal_date 的规律；默认返回全库。
    """
    kb = _load_kb()
    if date is not None and not kb.empty:
        kb = kb[kb["signal_date"].astype(str) == str(date)]
    return kb


def _save_kb(df: pd.DataFrame) -> None:
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
    for c in KB_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[KB_COLUMNS]
    _STR_COLS = ["pattern_id", "pattern_type", "name", "signal_date", "evidence",
                 "skill_ref", "status", "backtest_stats", "object_code", "object_name",
                 "direction", "source_note", "first_seen", "last_seen", "updated_at",
                 "feedback"]
    df[_STR_COLS] = df[_STR_COLS].fillna("")
    for c in ("confidence", "n_obs_5", "win_rate_5", "avg_ret_5",
              "n_obs_10", "win_rate_10", "avg_ret_10"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(KB_PATH, index=False)


def _pattern_id(ptype: str, date: str, code: str | None = None) -> str:
    obj = code or "market"
    return f"{ptype}_{date}_{obj}"


def _upsert_patterns(df: pd.DataFrame, patterns: list[dict]) -> pd.DataFrame:
    """按 pattern_id 幂等 upsert：已存在则刷新证据/状态，保留 first_seen。"""
    rows = []
    for p in patterns:
        row = {c: p.get(c) for c in KB_COLUMNS}
        if not row.get("skill_ref"):
            row["skill_ref"] = TYPE_SPEC.get(row.get("pattern_type"), {}).get("skill_ref", "")
        row["evidence"] = json.dumps(p.get("evidence", []), ensure_ascii=False)
        row["backtest_stats"] = json.dumps(p.get("backtest_stats", {}), ensure_ascii=False,
                                           default=str)
        bs = p.get("backtest_stats") or {}
        row["n_obs_5"] = bs.get("samples_5")
        row["win_rate_5"] = bs.get("win_rate_5")
        row["avg_ret_5"] = bs.get("avg_ret_5")
        row["n_obs_10"] = bs.get("samples_10")
        row["win_rate_10"] = bs.get("win_rate_10")
        row["avg_ret_10"] = bs.get("avg_ret_10")
        row["first_seen"] = p.get("first_seen") or _now().isoformat(timespec="seconds")
        row["last_seen"] = p.get("signal_date")
        row["updated_at"] = _now().isoformat(timespec="seconds")
        row["feedback"] = json.dumps(p.get("feedback") or {}, ensure_ascii=False)
        rows.append(row)
    if df.empty:
        df = pd.DataFrame(columns=KB_COLUMNS)
    new = pd.DataFrame(rows, columns=KB_COLUMNS)
    if df.empty:
        return new
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # 空/全NA列 dtype 推断告警
        key = df["pattern_id"].isin(new["pattern_id"])
    if key.any():
        old = df.loc[key].set_index("pattern_id")
        for _, r in new.iterrows():
            if r["pattern_id"] in old.index:
                o = old.loc[r["pattern_id"]]
                new.loc[new["pattern_id"] == r["pattern_id"], "first_seen"] = (
                    o["first_seen"] or r["first_seen"])
        rest = df.loc[~key]
        if rest.empty:
            df = new
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                df = pd.concat([rest, new], ignore_index=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df = pd.concat([df, new], ignore_index=True)
    return df


def _load_universe(target: pd.Timestamp, limit: int = DEFAULT_UNIVERSE_LIMIT) -> list[dict]:
    """有界宇宙 = 自选股 + 当日成交额 TOP-N（realtime_snapshot 单文件全市场快照）。

    历史日期无快照 → 回退最近快照；均缺 → 回退自选。剔除 ST/北交所。
    """
    names, st_map = breakout_watch.load_names()
    out: list[dict] = []
    seen: set[str] = set()

    def _add(code: str) -> None:
        code = str(code).zfill(6)
        if code in seen or code.startswith(("4", "8", "92")):
            return
        name = names.get(code, code)
        if st_map.get(code, "ST" in name.upper()):
            return
        seen.add(code)
        out.append({"code": code, "name": name})

    try:
        if WATCHLIST_PATH.exists():
            wl = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
            for c in (wl.get("watch") or [])[:20]:
                _add(c)
    except Exception as e:
        logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)

    snap_dir = SNAPSHOT_DIR / _snap_date(target)
    if not snap_dir.exists():
        cands = sorted(SNAPSHOT_DIR.glob("*")) if SNAPSHOT_DIR.exists() else []
        snap_dir = cands[-1] if cands else None
    if snap_dir is not None:
        try:
            files = sorted(snap_dir.glob("*.parquet"))
            if files:
                df = pd.read_parquet(files[0])
                if {"code", "amount_wan"}.issubset(df.columns):
                    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})")[0]
                    df = df.dropna(subset=["code"])
                    df = df.sort_values("amount_wan", ascending=False)
                    for c in df["code"].head(max(0, limit - len(out))):
                        _add(c)
        except Exception as e:
            logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)

    if not out:  # 最后回退：kline 文件表头抽样
        files = breakout_watch.kline_files(limit)
        for f in files:
            _add(f.stem)
    return out[: max(limit, 1)]


# ────────────────────────────────────────────────────────────
# 纯计算：趋势线 / 支撑压力整合（不依赖 I/O）
# ────────────────────────────────────────────────────────────
def _vol_ratio(df: pd.DataFrame) -> float:
    """当日量 / 此前20日均量（与 breakout_watch 口径一致）。"""
    vol = pd.to_numeric(df["volume"], errors="coerce")
    if len(vol) < 21:
        return np.nan
    ma20 = float(vol.iloc[-21:-1].mean())
    if not np.isfinite(ma20) or ma20 <= 0:
        return np.nan
    return float(vol.iloc[-1]) / ma20


def _trendline(highs, lows, close=None) -> dict:
    """pivot 高低点 → 线性回归趋势线。

    输入: highs/lows（array-like，按时间升序；长度 n，只允许末尾即"当前日"）。
    输出: up/down 两条候选线的斜率与今日位、突破状态（收盘穿越+斜率方向）。
    """
    h = np.asarray(pd.to_numeric(highs, errors="coerce").to_numpy(dtype=float), dtype=float)
    l = np.asarray(pd.to_numeric(lows, errors="coerce").to_numpy(dtype=float), dtype=float)
    n = len(h)
    out = {"up": {"slope": np.nan, "level": np.nan, "n": 0},
           "down": {"slope": np.nan, "level": np.nan, "n": 0},
           "break_up": False, "break_down": False, "note": ""}
    if n < 8:
        out["note"] = "样本不足"
        return out
    c = None
    if close is not None:
        cc = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
        if len(cc) == n:
            c = cc

    hi_idx = [i for i in range(2, n - 2)
              if np.isfinite(h[i]) and h[i] == float(np.nanmax(h[i - 2:i + 3]))]
    lo_idx = [i for i in range(2, n - 2)
              if np.isfinite(l[i]) and l[i] == float(np.nanmin(l[i - 2:i + 3]))]

    def _fit(vals: np.ndarray, idxs: list[int], key: str) -> None:
        # 优先最近 3 个 pivot；pivot 不足时回退最近 10 根 K 线回归（robust）
        seg = idxs[-3:] if len(idxs) >= 2 else list(range(max(0, n - 10), n))
        x = np.asarray(seg, dtype=float)      # 绝对 K 线坐标（pivot 间距≠1）
        y = np.array([vals[i] for i in seg], dtype=float)
        m = np.isfinite(y)
        if int(m.sum()) < 2:
            return
        x, y = x[m], y[m]
        if np.nanstd(y) == 0:
            return
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            return
        base = float(np.mean(y))
        rel = slope / base if base != 0 else 0.0
        if key == "up" and rel < 0.0008:      # 上升趋势线：支撑点上行(≥0.08%/bar)
            return
        if key == "down" and rel > -0.0008:   # 下降趋势线：压力点下行
            return
        level = intercept + slope * (n - 1)   # 外推至最新一根 K 线
        out[key] = {"slope": float(slope), "rel_slope": float(rel), "level": float(level),
                    "n": int(len(seg)), "anchor_idx": int(seg[-1])}

    _fit(l, lo_idx, "up")
    _fit(h, hi_idx, "down")

    if c is not None:
        c_t = float(c[-1])
        dl = out["down"]["level"]
        ul = out["up"]["level"]
        if np.isfinite(dl) and dl > 0:
            out["break_up"] = bool(c_t > dl)
        if np.isfinite(ul) and ul > 0:
            out["break_down"] = bool(c_t < ul)
    return out


def _support_resistance(df: pd.DataFrame) -> dict:
    """支撑压力整合 = 前高前低(60日) + 密集区上下沿 + 趋势线位。

    返回 levels（含来源标注）、贴近方向、汇合数（三重确认用）。
    """
    n = len(df)
    res: dict[str, float] = {}
    sup: dict[str, float] = {}
    if n >= 2:
        win = n  # 实际窗口长度（正常为60日，回放/样本不足时更短）
        prev_high = float(pd.to_numeric(df["high"], errors="coerce").iloc[:-1].max())
        prev_low = float(pd.to_numeric(df["low"], errors="coerce").iloc[:-1].min())
        if np.isfinite(prev_high):
            res[f"前高{win}日"] = prev_high
        if np.isfinite(prev_low):
            sup[f"前低{win}日"] = prev_low
    try:
        vp = volume_profile.compute_volume_profile(df)
        if vp.get("ok") and vp.get("upper") and vp.get("lower"):
            res["密集区上沿"] = float(vp["upper"])
            sup["密集区下沿"] = float(vp["lower"])
    except Exception as e:
        logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
    tl = _trendline(df["high"], df["low"], df["close"])
    if np.isfinite(tl["down"]["level"]):
        res["下降趋势线"] = float(tl["down"]["level"])
    if np.isfinite(tl["up"]["level"]):
        sup["上升趋势线"] = float(tl["up"]["level"])

    close_t = float(df["close"].iloc[-1])
    near = []
    for side, levels in (("resistance", res), ("support", sup)):
        for src, lv in levels.items():
            if lv <= 0:
                continue
            dist = (close_t / lv - 1.0) * 100.0
            if abs(dist) <= NEAR_PCT * 100.0:
                near.append({"side": side, "source": src, "level": round(lv, 3),
                             "dist_pct": round(dist, 2)})

    confluent = 0
    for side, levels in (("resistance", res), ("support", sup)):
        lvs = [v for v in levels.values() if np.isfinite(v) and v > 0]
        if not lvs:
            continue
        for lv in lvs:
            cnt = sum(1 for o in lvs if abs(o / lv - 1.0) <= CONFLUENT_PCT)
            if cnt >= 2:
                confluent = max(confluent, cnt)
    return {"levels": {"resistance": res, "support": sup},
            "near": near, "confluent": confluent, "close": close_t,
            "vol_ratio": _vol_ratio(df), "trendline": tl}


# ────────────────────────────────────────────────────────────
# 个股级规律检测（纯函数，输入为截断至目标日的 60 日窗口 df）
# ────────────────────────────────────────────────────────────
def _detect_dense_zone(df: pd.DataFrame, code: str, name: str) -> dict | None:
    if len(df) < 20:
        return None
    vp = volume_profile.compute_volume_profile(df)
    if not vp.get("ok"):
        return None
    vr = _vol_ratio(df)
    label = vp.get("pos_label", "")
    if label not in ("上方", "下方"):
        return None
    if not np.isfinite(vr):
        return None
    if not (0.2 <= vr <= 8.0):
        return None
    if label == "上方":
        if vr >= ZONE_VOL_MULT:
            pname, direction, conf, note = "密集区上沿放量突破", "up", 0.70, "突破放量"
        elif vr <= ZONE_SHRINK:
            pname, direction, conf, note = "密集区上沿假突破(缩量)", "down", 0.60, "假突破缩量"
        else:
            return None
    else:
        if vr >= ZONE_VOL_MULT:
            pname, direction, conf, note = "密集区下沿放量跌破", "down", 0.70, "跌破放量"
        elif vr <= ZONE_SHRINK:
            pname, direction, conf, note = "密集区下沿假跌破(缩量)", "up", 0.60, "假跌破缩量"
        else:
            return None
    evidence = [
        f"{name}({code}) 收{vp['close']:.2f} 进入密集区{label}",
        f"密集区 POC={vp['poc']:.2f} 上沿={vp['upper']:.2f} 下沿={vp['lower']:.2f} "
        f"位置={vp['pos_pct']:.0f}%",
        f"量比 {vr:.2f}（20日均量基准；{note}）",
        f"密集区峰值占比 {vp['peak_share']:.0f}%",
    ]
    return {"pattern_type": "dense_zone", "name": pname, "confidence": conf,
            "direction": direction, "evidence": evidence,
            "object_code": code, "object_name": name,
            "source_note": "volume_profile 60日简化VP"}


def _detect_support_resistance(df: pd.DataFrame, code: str, name: str) -> dict | None:
    sr = _support_resistance(df)
    if not sr["near"] and sr["confluent"] < 2:
        return None
    vr = sr["vol_ratio"]
    near_side = sr["near"][0]["side"] if sr["near"] else (
        "resistance" if sr["confluent"] >= 2 else "")
    if not near_side:
        return None
    has_vol = bool(np.isfinite(vr) and vr >= 1.3)
    if not has_vol and sr["confluent"] < 2:
        return None
    direction = "up" if near_side == "support" else "down"
    # 贴近支撑=买点（向上）、贴近压力=卖点/突破观察（向下），三重确认时方向随汇合
    conf = round(min(0.9, 0.55 + 0.15 * sr["confluent"] + (0.1 if has_vol else 0.0)), 2)
    ev = [f"{name}({code}) 收{sr['close']:.2f} 贴近{near_side}（三重确认汇合数={sr['confluent']}）"]
    for r in sr["near"]:
        ev.append(f"{r['source']} {r['level']:.2f}（距离 {r['dist_pct']:+.2f}%）")
    ev.append(f"量比 {fmt(vr)}（{'放量确认' if has_vol else '无放量'}）")
    return {"pattern_type": "support_resistance", "name": f"支撑压力位·{near_side}贴近(三重确认)",
            "confidence": conf, "direction": direction, "evidence": ev,
            "object_code": code, "object_name": name,
            "source_note": "前高前低60日+密集区+趋势线"}


def _detect_trendline_breakout(df: pd.DataFrame, code: str, name: str) -> dict | None:
    if len(df) < 30:
        return None
    tl = _trendline(df["high"], df["low"], df["close"])
    vr = _vol_ratio(df)
    if not np.isfinite(vr) or not (0.2 <= vr <= 8.0):
        return None
    if tl.get("break_up") and np.isfinite(tl["down"]["level"]):
        if vr < TREND_VOL_MULT:
            return None
        pname, direction, conf = "下降趋势线放量突破", "up", 0.72
        line = tl["down"]
    elif tl.get("break_down") and np.isfinite(tl["up"]["level"]):
        if vr < TREND_VOL_MULT:
            return None
        pname, direction, conf = "上升趋势线放量跌破", "down", 0.72
        line = tl["up"]
    else:
        return None
    ev = [
        f"{name}({code}) 收{float(df['close'].iloc[-1]):.2f} {pname}",
        f"趋势线斜率 {line['rel_slope'] * 100:+.3f}%/日（pivot 数 {line['n']}）",
        f"今日线位 {line['level']:.2f}，量比 {vr:.2f}（≥{TREND_VOL_MULT} 确认）",
    ]
    return {"pattern_type": "trendline_breakout", "name": pname, "confidence": conf,
            "direction": direction, "evidence": ev,
            "object_code": code, "object_name": name,
            "source_note": "pivot线性回归趋势线+量能确认"}


def _detect_breakout_volume_on_df(df: pd.DataFrame) -> dict | None:
    """单股切片上的 250日新高+量能1.5x 判定（供历史回放复用，口径与 breakout_watch 一致）。"""
    close = pd.to_numeric(df["close"], errors="coerce")
    if len(close) < 251:
        return None
    close_t = float(close.iloc[-1])
    prev_max = float(close.iloc[-251:-1].max())
    if not (close_t > prev_max):
        return None
    vr = _vol_ratio(df)
    if not np.isfinite(vr) or vr < 1.5:
        return None
    return {"vol_ratio": vr, "close": close_t}


# ────────────────────────────────────────────────────────────
# 市场级规律检测
# ────────────────────────────────────────────────────────────
def _detect_stagflation(date: str) -> list[dict]:
    res = macro_overseas.get_macro_overseas(date)
    mode = res.get("mode", "")
    flags = res.get("flags") or {}
    if mode != "滞胀避险模式":
        return []
    conf = 0.6
    ev = list(res.get("evidence") or [])
    if bool(flags.get("gold_up")):
        conf += 0.1
    if bool(flags.get("oil_up")):
        conf += 0.1
    if bool(flags.get("yield_high")):
        conf += 0.1
    return [{"pattern_type": "stagflation", "name": "滞涨避险模式(金油同涨+美债高)",
             "confidence": round(min(conf, 0.95), 2), "direction": "risk_off",
             "evidence": ev + [f"模式判定: {mode}（macro_overseas）"],
             "object_code": None, "object_name": "海外宏观",
             "source_note": "macro_overseas 缓存/当日抓取"}]


def _detect_style_switch(date: str, target: pd.Timestamp | None = None) -> list[dict]:
    target = target or pd.Timestamp(date).normalize()
    rows: dict[str, dict] = {}
    notes: list[str] = []
    for name, cat, local, ak_symbol in style_spread.INDEX_SPECS:
        s, source = style_spread.load_index(name, local, ak_symbol, target)
        if s is None:
            notes.append(f"{name}:{source}")
            continue
        st = style_spread.index_stats(s)
        st["source"] = source
        st["category"] = cat
        rows[name] = st
    if len(rows) < 2:
        return []
    judge = style_spread.judge_style(rows)
    d5, d1 = judge.get("diff5"), judge.get("diff1")
    if d5 is None or d1 is None or not judge.get("extreme"):
        return []
    reversal = (d5 * d1 < 0) or (abs(d1) < abs(d5) * 0.5)
    if not reversal:
        return []
    direction = "small" if d5 < 0 else "large"
    ev = [
        f"风格判定: {judge['style']}（5日剪刀差 {d5:+.2f}pp）",
        f"当日剪刀差 {d1:+.2f}pp（{'反向' if d5 * d1 < 0 else '收敛'}，疑似切换）",
        f"{judge['breadth']} | 极端分化={judge['extreme']}",
    ] + ([f"小盘组回退: {'/'.join(judge.get('small_names', []))}" ] if judge.get("fallback") else [])
    return [{"pattern_type": "style_switch", "name": f"风格切换·{judge['style']}反转",
             "confidence": 0.72, "direction": direction, "evidence": ev,
             "object_code": None, "object_name": "大小盘指数",
             "source_note": f"style_spread（{'；'.join(notes) if notes else '全部指数可用'}）"}]


def _detect_sector_resonance(date: str) -> list[dict]:
    from quant_system.analysis_core import resonance_scorer
    df = resonance_scorer.score_day(date)
    if df is None or df.empty:
        return []
    total = len(df)
    active = int((df["zt_cnt"] >= 3).sum())
    breadth = active / total if total else 0.0
    main_lines = int((df["level"] == "主线").sum())
    top = df.head(3)[["board_name", "score", "zt_cnt", "flow_yi"]].to_dict("records")
    bot = df[df["score"] < 0].tail(3)[["board_name", "score"]].to_dict("records") if (df["score"] < 0).any() else []
    if breadth < 0.3 and main_lines < 2:
        return []
    conf = round(min(0.9, 0.5 + breadth * 0.4), 2)
    ev = [
        f"行业广度: {active}/{total} 个板块有涨停合力（{breadth * 100:.0f}%）",
        f"主线板块 {main_lines} 个 | 领涨: " + "、".join(f"{r['board_name']}(分{r['score']})" for r in top[:3]),
    ]
    if bot:
        ev.append("领跌: " + "、".join(f"{r['board_name']}(分{r['score']})" for r in bot))
    return [{"pattern_type": "sector_resonance", "name": f"板块共振·广度{breadth * 100:.0f}%",
             "confidence": conf, "direction": "up", "evidence": ev,
             "object_code": None, "object_name": "行业/概念板块",
             "source_note": f"resonance_scorer（{len(df)} 个板块）"}]


def _detect_breakout_volume(target: pd.Timestamp, scan_limit: int = DEFAULT_SCAN_LIMIT) -> list[dict]:
    days = [60, 120, 250]
    result, meta = breakout_watch.scan(days, target, limit=scan_limit)
    rows = result.get("vol_break_rows") or []
    n250 = sum(1 for r in rows if (r.get("new_high") or {}).get(250))
    if n250 == 0:
        return []
    top = sorted(rows, key=lambda r: -r.get("amount_yi", 0))[:5]
    ev = [
        f"全A（抽样{meta.get('universe', '?')}只）放量突破 {len(rows)} 只，其中250日新高+放量 {n250} 只",
        "代表: " + "、".join(f"{r['name']}({r['code']})量比{r['vol_ratio']:.1f}成交{r['amount_yi']:.0f}亿"
                             for r in top),
        f"新高/下跌比(60日)={meta.get('new_high_low_ratio', {}).get(60)}（Elder 宽度）",
    ]
    return [{"pattern_type": "breakout_volume", "name": f"新高放量突破·250日新高{n250}只",
             "confidence": 0.75, "direction": "up", "evidence": ev,
             "object_code": None, "object_name": "全A市场",
             "source_note": f"breakout_watch scan（days=60/120/250, limit={scan_limit}）"}]


def _detect_divergence(code: str, target: pd.Timestamp) -> dict | None:
    pct, _ = fund_flow_divergence.local_pct_chg(code, target)
    ff = fund_flow_divergence.fetch_fund_flow_by_code(code, target, timeout=DIVERGENCE_TIMEOUT)
    if not ff.get("ok"):
        raise RuntimeError(f"资金流不可用: {str(ff.get('error', ''))[:60]}")
    signal = fund_flow_divergence.judge_divergence(pct, ff.get("main_net"))
    if signal.startswith("⚠️"):
        pname, direction, conf = "量价背离(涨但主力流出)", "down", 0.80
    elif signal.startswith("✅"):
        pname, direction, conf = "主力承接(跌但主力流入)", "up", 0.75
    else:
        return None
    ev = [
        f"{code} 当日 {fmt(pct)}% vs 主力净额 {ff.get('main_net', 0) / 1e8:+.2f}亿",
        f"超大单 {ff.get('super_net', 0) / 1e8:+.2f}亿 | 判定: {signal}",
    ]
    return {"pattern_type": "divergence", "name": pname, "confidence": conf,
            "direction": direction, "evidence": ev,
            "object_code": code, "object_name": code,
            "source_note": "fund_flow_divergence(东财资金流)"}


# ────────────────────────────────────────────────────────────
# 规律检测主入口
# ────────────────────────────────────────────────────────────
def detect_patterns(date: str | None = None, universe_limit: int = DEFAULT_UNIVERSE_LIMIT,
                    scan_limit: int = DEFAULT_SCAN_LIMIT, verbose: bool = False) -> dict:
    """检测 8 种规律在当前日是否触发。任一信号源失败 → 该规律跳过（不崩溃）。"""
    date = date or today()
    target = pd.Timestamp(date).normalize()
    t0 = time.time()
    patterns: list[dict] = []
    skipped: list[str] = []
    base = {"pattern_id": "", "name": "", "signal_date": date, "status": "new",
            "backtest_stats": {}}

    # ── 市场级 ──
    for fn, key in ((_detect_stagflation, "stagflation"),
                    (_detect_style_switch, "style_switch"),
                    (_detect_sector_resonance, "sector_resonance")):
        try:
            got = fn(date) if key != "style_switch" else fn(date, target)
            for p in got:
                patterns.append({**base, **p, "pattern_id": _pattern_id(p["pattern_type"], date, None)})
        except Exception as e:
            skipped.append(f"{key}: {type(e).__name__}: {str(e)[:100]}")
    try:
        for p in _detect_breakout_volume(target, scan_limit):
            patterns.append({**base, **p, "pattern_id": _pattern_id(p["pattern_type"], date, None)})
    except Exception as e:
        skipped.append(f"breakout_volume: {type(e).__name__}: {str(e)[:100]}")

    # ── 个股级（有界宇宙）──
    universe = _load_universe(target, universe_limit)
    for u in universe:
        df, note = volume_profile.load_kline_60d(u["code"], target)
        if df is None:
            continue
        for det in (_detect_dense_zone, _detect_support_resistance, _detect_trendline_breakout):
            try:
                p = det(df, u["code"], u["name"])
                if p:
                    patterns.append({**base, **p,
                                     "pattern_id": _pattern_id(p["pattern_type"], date, u["code"])})
            except Exception as e:
                skipped.append(f"{det.__name__}({u['code']}): {type(e).__name__}: {str(e)[:80]}")

    # ── 量价背离（仅自选池，联网源，失败即跳过）──
    for code in [str(c).zfill(6) for c in _watchlist_codes()]:
        try:
            p = _detect_divergence(code, target)
            if p:
                patterns.append({**base, **p, "pattern_id": _pattern_id("divergence", date, code)})
        except Exception as e:
            skipped.append(f"divergence({code}): {type(e).__name__}: {str(e)[:80]}")

    return {"date": date, "target": target.isoformat(), "patterns": patterns,
            "skipped": skipped, "universe": len(universe),
            "elapsed_sec": round(time.time() - t0, 1),
            "generated_at": _now().isoformat(timespec="seconds")}


def _watchlist_codes() -> list[str]:
    try:
        if WATCHLIST_PATH.exists():
            wl = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
            return [str(c) for c in (wl.get("watch") or [])]
    except Exception as e:
        logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
    return []


# ────────────────────────────────────────────────────────────
# 验证层：历史回放 + 每日增量验证
# ────────────────────────────────────────────────────────────
def _fwd_rets(closes: np.ndarray, idx: int) -> tuple[float | None, float | None]:
    """给定收盘序列与信号日下标 → (5日收益, 10日收益)。不足则 None。"""
    r5 = r10 = None
    if idx + 5 < len(closes) and closes[idx] > 0:
        r5 = closes[idx + 5] / closes[idx] - 1.0
    if idx + 10 < len(closes) and closes[idx] > 0:
        r10 = closes[idx + 10] / closes[idx] - 1.0
    return r5, r10


def _agg_rets(r5s: list[float], r10s: list[float]) -> dict:
    def _one(rs: list[float]) -> dict:
        a = np.asarray([x for x in rs if np.isfinite(x)], dtype=float)
        if not len(a):
            return {"samples": 0, "win_rate": 0.0, "avg_ret": 0.0, "median_ret": 0.0}
        return {"samples": int(len(a)), "win_rate": float((a > 0).mean()),
                "avg_ret": float(a.mean()), "median_ret": float(np.median(a))}

    return {"samples_5": _one(r5s)["samples"], "win_rate_5": _one(r5s)["win_rate"],
            "avg_ret_5": _one(r5s)["avg_ret"], "median_ret_5": _one(r5s)["median_ret"],
            "samples_10": _one(r10s)["samples"], "win_rate_10": _one(r10s)["win_rate"],
            "avg_ret_10": _one(r10s)["avg_ret"], "median_ret_10": _one(r10s)["median_ret"]}


def _decide_status(stats: dict) -> str:
    n = stats.get("samples_5", 0)
    wr = stats.get("win_rate_5", 0.0)
    if n >= CONFIRM_SAMPLES and wr > CONFIRM_WIN_RATE:
        return "confirmed"
    if n >= VALIDATE_SAMPLES:
        return "validating"
    return "new"


def _market_bench(target: pd.Timestamp, horizon_days: int = 20) -> pd.Series:
    """沪深300 收盘序列（截断至 target + horizon 日历日，供前瞻收益）。"""
    for fn in ("index_daily_沪深300.parquet", "index_daily.parquet"):
        p = MARKET_DIR / fn
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            s = (df.dropna(subset=["date", "close"]).drop_duplicates("date")
                 .set_index("date")["close"].sort_index())
            s = s[s.index <= target + pd.Timedelta(days=horizon_days)]
            if len(s) >= 30:
                return s
        except Exception as e:
            logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
            continue
    return pd.Series(dtype=float)


def _load_window(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    key = (code, start.date().isoformat(), end.date().isoformat())
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    if len(_WINDOW_CACHE) > 300:
        _WINDOW_CACHE.clear()
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        _WINDOW_CACHE[key] = None
        return None
    try:
        df = breakout_watch.read_kline_window(path, ["date", "open", "high", "low", "close", "volume"],
                                              start - pd.Timedelta(days=5))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    except Exception:
        df = None
    _WINDOW_CACHE[key] = df
    return df


def _replay_stock(ptype: str, date: str, lookback_days: int, sample_every: int,
                  universe: list[dict], verbose: bool = False) -> tuple[dict, str]:
    """个股级规律历史回放：窗口内每 sample_every 交易日检测一次，统计前瞻收益。"""
    end = pd.Timestamp(date).normalize()
    start = end - pd.Timedelta(days=lookback_days)
    need_250 = ptype == "breakout_volume"
    load_start = start - pd.Timedelta(days=430 if need_250 else 100)

    bench: pd.Series | None = None
    r5s: list[float] = []
    r10s: list[float] = []
    sample_dates: list[pd.Timestamp] = []
    for code in [u["code"] for u in universe[:45]]:
        df = _load_window(code, load_start, end + pd.Timedelta(days=20))
        if df is None or len(df) < (251 if need_250 else 40):
            continue
        dates = df["date"].to_numpy()
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        idxs = np.where((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))[0]
        idxs = idxs[::sample_every]
        for i in idxs:
            d = pd.Timestamp(dates[i])
            if len(sample_dates) < 400 and d not in sample_dates:
                sample_dates.append(d)
            if need_250:
                sub = df.iloc[: i + 1]              # 250日新高需要长窗口
            else:
                sub = df.iloc[max(0, i - 59): i + 1]  # 与 live 60日窗口口径一致
            try:
                if ptype == "dense_zone":
                    hit = _detect_dense_zone(sub, code, "") is not None
                elif ptype == "support_resistance":
                    hit = _detect_support_resistance(sub, code, "") is not None
                elif ptype == "trendline_breakout":
                    hit = _detect_trendline_breakout(sub, code, "") is not None
                elif ptype == "breakout_volume":
                    hit = _detect_breakout_volume_on_df(sub) is not None
                else:
                    hit = False
            except Exception:
                hit = False
            if not hit:
                continue
            r5, r10 = _fwd_rets(closes, i)
            if r5 is not None:
                r5s.append(r5)
            if r10 is not None:
                r10s.append(r10)

    stats = _agg_rets(r5s, r10s)
    stats.update({"replay_available": True, "replay_window_days": lookback_days,
                  "replay_sample_days": len(sample_dates),
                  "replay_note": f"个股级回放（{len(universe[:45])}只宇宙样本，"
                                 f"每{sample_every}交易日采样）"})
    return stats, _decide_status(stats)


def _replay_style(date: str, lookback_days: int, sample_every: int) -> tuple[dict, str]:
    end = pd.Timestamp(date).normalize()
    start = end - pd.Timedelta(days=lookback_days)
    bench = _market_bench(end)
    r5s, r10s, sample_dates = [], [], []
    # 预载全部指数序列，逐日切片（避免重复读盘 + 无前视）
    series: dict[str, pd.Series] = {}
    for name, cat, local, ak in style_spread.INDEX_SPECS:
        s, _ = style_spread.load_index(name, local, ak, end)
        if s is not None:
            series[name] = s
    for name, s in series.items():
        dates = s.index.to_numpy()
        idxs = np.where((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))[0][::sample_every]
        for i in idxs:
            d = pd.Timestamp(dates[i])
            if d in sample_dates:
                continue
            sample_dates.append(d)
            rows: dict[str, dict] = {}
            for nm, ss in series.items():
                sub = ss[ss.index <= d]
                if len(sub) < 6:
                    continue
                rows[nm] = style_spread.index_stats(sub)
            judge = style_spread.judge_style(rows)
            d5, d1 = judge.get("diff5"), judge.get("diff1")
            if d5 is None or d1 is None or not judge.get("extreme"):
                continue
            if not ((d5 * d1 < 0) or (abs(d1) < abs(d5) * 0.5)):
                continue
            pos = np.searchsorted(bench.index.to_numpy(), np.datetime64(d))
            if pos >= len(bench) or bench.index[pos] != d:
                continue
            r5, r10 = _fwd_rets(bench.to_numpy(dtype=float), pos)
            if r5 is not None:
                r5s.append(r5)
            if r10 is not None:
                r10s.append(r10)
    stats = _agg_rets(r5s, r10s)
    stats.update({"replay_available": True, "replay_window_days": lookback_days,
                  "replay_sample_days": len(sample_dates),
                  "replay_note": "沪深300 前瞻收益"})
    return stats, _decide_status(stats)


def _replay_sector(date: str, lookback_days: int, sample_every: int,
                   verbose: bool = False) -> tuple[dict, str]:
    from quant_system.analysis_core import resonance_scorer
    end = pd.Timestamp(date).normalize()
    start = end - pd.Timedelta(days=lookback_days)
    bench = _market_bench(end)
    r5s, r10s, sample_dates = [], [], []
    # 用 theme_cycle 交易日历采样（避免逐日空跑）
    try:
        tc = pd.read_parquet(ROOT / "data_warehouse" / "market" / "zt_daily_stats.parquet",
                             columns=["date"])
        all_days = sorted(pd.to_datetime(tc["date"]).dt.normalize().unique())
        sample_dates = [d for d in all_days if start <= d <= end][::sample_every]
    except Exception:
        sample_dates = []
        d = start
        while d <= end:
            sample_dates.append(d)
            d += pd.Timedelta(days=7)
    for d in sample_dates:
        ds = d.strftime("%Y-%m-%d")
        try:
            if ds not in _SCORE_DAY_CACHE:
                _SCORE_DAY_CACHE[ds] = resonance_scorer.score_day(ds)
            df = _SCORE_DAY_CACHE[ds]
            if df is None or df.empty:
                continue
            total = len(df)
            active = int((df["zt_cnt"] >= 3).sum())
            breadth = active / total if total else 0.0
            main_lines = int((df["level"] == "主线").sum())
            if breadth < 0.3 and main_lines < 2:
                continue
        except Exception as e:
            logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
            continue
        pos = np.searchsorted(bench.index.to_numpy(), np.datetime64(d))
        if pos >= len(bench) or bench.index[pos] != d:
            continue
        r5, r10 = _fwd_rets(bench.to_numpy(dtype=float), pos)
        if r5 is not None:
            r5s.append(r5)
        if r10 is not None:
            r10s.append(r10)
    stats = _agg_rets(r5s, r10s)
    stats.update({"replay_available": True, "replay_window_days": lookback_days,
                  "replay_sample_days": len(sample_dates),
                  "replay_note": "沪深300 前瞻收益（resonance_scorer 板块广度≥30% 或主线≥2）"})
    return stats, _decide_status(stats)


def validate_pattern(pattern: dict, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                     sample_every: int = SAMPLE_EVERY, verbose: bool = False) -> dict:
    """历史回放验证规律：过去 N 次出现后的 5/10 日收益分布 → 状态判定。"""
    ptype = pattern["pattern_type"]
    date = pattern["signal_date"]
    stats: dict = {}
    status = "new"
    try:
        if ptype in ("dense_zone", "support_resistance", "trendline_breakout", "breakout_volume"):
            univ = _load_universe(pd.Timestamp(date).normalize(), DEFAULT_UNIVERSE_LIMIT)
            stats, status = _replay_stock(ptype, date, lookback_days, sample_every, univ, verbose)
        elif ptype == "style_switch":
            stats, status = _replay_style(date, lookback_days, sample_every)
        elif ptype == "sector_resonance":
            stats, status = _replay_sector(date, lookback_days, sample_every, verbose)
        elif ptype == "stagflation":
            stats = {"replay_available": False, "replay_window_days": lookback_days,
                     "samples_5": 0, "win_rate_5": 0.0, "avg_ret_5": 0.0,
                     "samples_10": 0, "win_rate_10": 0.0, "avg_ret_10": 0.0,
                     "replay_note": "宏观缓存仅当日可得，历史回放受限（等待人工标注/数据补齐）"}
            status = "new"
        elif ptype == "divergence":
            stats = {"replay_available": False, "replay_window_days": lookback_days,
                     "samples_5": 0, "win_rate_5": 0.0, "avg_ret_5": 0.0,
                     "samples_10": 0, "win_rate_10": 0.0, "avg_ret_10": 0.0,
                     "replay_note": "资金流依赖联网源，历史回放受限"}
            status = "new"
        else:
            stats = {"replay_available": False, "replay_note": f"未知规律类型 {ptype}"}
    except Exception as e:
        stats = {"replay_available": False, "replay_note": f"回放异常: {type(e).__name__}: {str(e)[:100]}"}
        status = "new"
    pattern["backtest_stats"] = stats
    pattern["status"] = status
    return pattern


def validate_open_patterns(date: str | None = None) -> list[dict]:
    """每日增量验证：库中 new/validating 且 signal_date<date 的规律，直接用最新
    数据补算前瞻收益（不重新回放），窗口自然增长后更新胜率与状态。"""
    date = date or today()
    end = pd.Timestamp(date).normalize()
    kb = _load_kb()
    if kb.empty:
        return []
    open_mask = kb["status"].isin(["new", "validating"]) & (
        pd.to_datetime(kb["signal_date"], errors="coerce") < end)
    updated: list[dict] = []
    for _, r in kb[open_mask].iterrows():
        pid = r["pattern_id"]
        obj = r.get("object_code")
        try:
            if obj:
                df = _load_window(obj, end - pd.Timedelta(days=420), end)
                if df is None:
                    continue
                closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
                dates = df["date"].to_numpy()
                want = np.datetime64(pd.Timestamp(r["signal_date"]))
                pos = int(np.searchsorted(dates, want))
                if pos >= len(dates) or dates[pos] != want:
                    pos -= 1
                if pos < 0:
                    continue
                r5, r10 = _fwd_rets(closes, pos)
            else:
                bench = _market_bench(end)
                want = np.datetime64(pd.Timestamp(r["signal_date"]))
                pos = int(np.searchsorted(bench.index.to_numpy(), want))
                if pos >= len(bench) or bench.index[pos] != want:
                    pos -= 1
                if pos < 0:
                    continue
                r5, r10 = _fwd_rets(bench.to_numpy(dtype=float), pos)
            if r5 is None and r10 is None:
                continue
            bs = json.loads(r.get("backtest_stats") or "{}")
            hist5 = bs.get("_hist_5") or []
            hist10 = bs.get("_hist_10") or []
            if r5 is not None:
                hist5.append(r5)
            if r10 is not None:
                hist10.append(r10)
            agg = _agg_rets(hist5, hist10)
            bs.update(agg)
            bs["_hist_5"] = hist5[-200:]
            bs["_hist_10"] = hist10[-200:]
            bs["updated_at"] = _now().isoformat(timespec="seconds")
            status = _decide_status(bs)
            kb.loc[kb["pattern_id"] == pid, "backtest_stats"] = json.dumps(bs, ensure_ascii=False)
            kb.loc[kb["pattern_id"] == pid, "status"] = status
            for col, key in (("n_obs_5", "samples_5"), ("win_rate_5", "win_rate_5"),
                             ("avg_ret_5", "avg_ret_5"), ("n_obs_10", "samples_10"),
                             ("win_rate_10", "win_rate_10"), ("avg_ret_10", "avg_ret_10")):
                kb.loc[kb["pattern_id"] == pid, col] = bs.get(key)
            updated.append({"pattern_id": pid, "pattern_type": r["pattern_type"],
                            "signal_date": str(r["signal_date"]), "status": status,
                            "samples_5": bs.get("samples_5"), "win_rate_5": bs.get("win_rate_5")})
        except Exception as e:
            logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
            continue
    _save_kb(kb)
    return updated


# ────────────────────────────────────────────────────────────
# 周度迭代 + 人工反馈
# ────────────────────────────────────────────────────────────
def weekly_review(date: str | None = None) -> dict:
    """每周汇总规律库表现：失效规律自动 deprecated，出具周报。"""
    date = date or today()
    kb = _load_kb()
    summary: dict[str, dict] = {}
    deprecated: list[dict] = []
    if not kb.empty:
        for ptype, grp in kb.groupby("pattern_type"):
            recent = grp[pd.to_datetime(grp["signal_date"], errors="coerce")
                         >= pd.Timestamp(date) - pd.Timedelta(days=120)]
            n5 = pd.to_numeric(recent["n_obs_5"], errors="coerce").fillna(0)
            wr5 = pd.to_numeric(recent["win_rate_5"], errors="coerce").fillna(0)
            n = int(n5.sum())
            wr = float((n5 * wr5).sum() / n) if n else 0.0
            confirmed_n = int((grp["status"] == "confirmed").sum())
            summary[ptype] = {"total": len(grp), "confirmed": confirmed_n,
                              "recent_n": n, "recent_win_rate_5": round(wr, 3) if n else None,
                              "skill_ref": TYPE_SPEC.get(ptype, {}).get("skill_ref", "")}
            if n >= CONFIRM_SAMPLES and wr < DEPRECATE_WIN_RATE:
                ids = recent.loc[recent["status"].isin(["confirmed", "validating"]), "pattern_id"].tolist()
                for pid in ids:
                    kb.loc[kb["pattern_id"] == pid, "status"] = "deprecated"
                    deprecated.append({"pattern_id": pid, "pattern_type": ptype,
                                       "reason": f"近120日{n}次样本胜率{wr:.1%}<{DEPRECATE_WIN_RATE:.0%}"})
    _save_kb(kb)
    lines = [f"# 规律库周度复盘 {date}", "",
             f"- 规律总数 {len(kb)} | 本周废弃 {len(deprecated)} | 状态分布 "
             f"{kb['status'].value_counts().to_dict() if not kb.empty else {}}", ""]
    lines.append("| 类型 | 总数 | confirmed | 近120日样本 | 近120日5日胜率 | skill |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for t, s in summary.items():
        lines.append(f"| {t} | {s['total']} | {s['confirmed']} | {s['recent_n']} | "
                     f"{fmt(s['recent_win_rate_5'] * 100, 1) if s['recent_win_rate_5'] is not None else '—'}% | {s['skill_ref']} |")
    if deprecated:
        lines.append("")
        lines.append("## 本周废弃规律")
        for d in deprecated:
            lines.append(f"- `{d['pattern_id']}` — {d['reason']}")
    out = OUT_DIR / f"pattern_weekly_{date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"date": date, "patterns_total": len(kb), "deprecated": deprecated,
            "summary": summary, "report_path": str(out)}


def mark_pattern(pattern_id: str, action: str, note: str = "") -> dict:
    """人工反馈通道：action ∈ {confirm, deprecate, note}，写 feedback.jsonl 并改库。"""
    if action not in ("confirm", "deprecate", "note"):
        raise ValueError("action 必须是 confirm/deprecate/note")
    kb = _load_kb()
    if pattern_id not in set(kb["pattern_id"]):
        return {"ok": False, "error": f"pattern_id {pattern_id} 不在规律库"}
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"pattern_id": pattern_id, "action": action, "note": note,
           "ts": _now().isoformat(timespec="seconds")}
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if action in ("confirm", "deprecate"):
        kb.loc[kb["pattern_id"] == pattern_id, "status"] = action
    kb.loc[kb["pattern_id"] == pattern_id, "feedback"] = json.dumps(
        {"action": action, "note": note, "ts": rec["ts"]}, ensure_ascii=False)
    _save_kb(kb)
    return {"ok": True, **rec}


# ────────────────────────────────────────────────────────────
# 报告
# ────────────────────────────────────────────────────────────
def _render_report(res: dict, kb: pd.DataFrame, inc: list[dict],
                 explanations: dict | None = None) -> str:
    date = res["date"]
    pats = res["patterns"]
    lines = [f"# 规律检测报告 {date}", "",
             f"- 触发规律 {len(pats)} 个 | 跳过/降级 {len(res['skipped'])} | "
             f"宇宙 {res['universe']} 只 | 耗时 {res['elapsed_sec']}s", ""]

    if pats:
        lines.append("## 今日触发规律")
        lines.append("")
        lines.append("| 规律 | 置信 | 方向 | 标的 | 状态 | 5日样本/胜率 | 10日样本/胜率 |")
        lines.append("|---|---:|---:|---|---|---:|---:|")
        for p in pats:
            bs = p.get("backtest_stats") or {}
            obj = p.get("object_name") or p.get("object_code") or "市场"
            lines.append(f"| {p['name']} | {p['confidence']:.2f} | {p.get('direction', '—')} | {obj} "
                         f"| {p['status']} | {bs.get('samples_5', 0)}/{fmt(bs.get('win_rate_5', 0) * 100, 1)}% "
                         f"| {bs.get('samples_10', 0)}/{fmt(bs.get('win_rate_10', 0) * 100, 1)}% |")
        lines.append("")
        for p in pats:
            lines.append(f"### {p['name']} (`{p['pattern_id']}`)")
            lines.append("")
            lines.append(f"- 置信 {p['confidence']:.2f} | 方向 {p.get('direction', '—')} | "
                         f"状态 **{p['status']}** | skill: {p.get('skill_ref', '')}")
            lines.append(f"- 信号源: {p.get('source_note', '—')}")
            for e in p.get("evidence", []):
                lines.append(f"- {e}")
            bs = p.get("backtest_stats") or {}
            if bs:
                lines.append("")
                lines.append(f"- 历史验证: 5日 {bs.get('samples_5', 0)} 样本 胜率 "
                             f"{fmt(bs.get('win_rate_5', 0) * 100, 1)}% 均收益 "
                             f"{fmt(bs.get('avg_ret_5', 0) * 100, 2)}% | 10日 {bs.get('samples_10', 0)} 样本 胜率 "
                             f"{fmt(bs.get('win_rate_10', 0) * 100, 1)}% 均收益 "
                             f"{fmt(bs.get('avg_ret_10', 0) * 100, 2)}%")
                if bs.get("replay_note"):
                    lines.append(f"- 回放说明: {bs['replay_note']}")
            lines.append("")

    if inc:
        lines.append("## 每日增量验证（库中待确认规律）")
        lines.append("")
        for r in inc:
            lines.append(f"- `{r['pattern_id']}` → {r['status']} "
                         f"(5日样本 {r.get('samples_5', 0)}, 胜率 {fmt(r.get('win_rate_5', 0) * 100, 1)}%)")
        lines.append("")

    if res.get("skipped"):
        lines.append("## 跳过/降级说明")
        lines.append("")
        for s in res["skipped"]:
            lines.append(f"- {s}")
        lines.append("")

    if not kb.empty:
        cnt = kb["status"].value_counts().to_dict()
        lines.append("## 规律库快照")
        lines.append("")
        lines.append(f"- 总条数 {len(kb)} | 状态分布 {cnt}")
        lines.append(f"- 路径: `data_warehouse/patterns/knowledge_base.parquet`")
        lines.append("")
        lines.append("| 类型 | 总数 | confirmed | validating | 近120日5日胜率(加权) |")
        lines.append("|---|---:|---:|---:|---:|")
        for t, grp in kb.groupby("pattern_type"):
            recent = grp[pd.to_datetime(grp["signal_date"], errors="coerce")
                         >= pd.Timestamp(date) - pd.Timedelta(days=120)]
            n5 = pd.to_numeric(recent["n_obs_5"], errors="coerce").fillna(0)
            wr5 = pd.to_numeric(recent["win_rate_5"], errors="coerce").fillna(0)
            n = int(n5.sum())
            wr = float((n5 * wr5).sum() / n) if n else 0.0
            lines.append(f"| {t} | {len(grp)} | {(grp['status'] == 'confirmed').sum()} | "
                         f"{(grp['status'] == 'validating').sum()} | "
                         f"{fmt(wr * 100, 1) if n else '—'}% |")

    if explanations is not None:
        lines.append("")
        lines.append(_render_explanation_section(explanations))
    return "\n".join(lines)


def report(date: str | None = None, universe_limit: int = DEFAULT_UNIVERSE_LIMIT,
           scan_limit: int = DEFAULT_SCAN_LIMIT, do_validate: bool = True,
           verbose: bool = False) -> dict:
    """完整流水线：检测 → 验证 → 入库 → 增量验证 → 报告。"""
    res = detect_patterns(date, universe_limit, scan_limit, verbose)
    date = res["date"]
    kb = _load_kb()
    for p in res["patterns"]:
        if do_validate:
            p = validate_pattern(p, verbose=verbose)
        else:
            # 幂等：本次未验证时保留库中已有统计/状态，不覆盖
            prev = kb[kb["pattern_id"] == p["pattern_id"]]
            if not prev.empty:
                p["status"] = prev.iloc[0]["status"]
                p["backtest_stats"] = json.loads(prev.iloc[0].get("backtest_stats") or "{}")
        kb = _upsert_patterns(kb, [p])
    _save_kb(kb)
    inc = validate_open_patterns(date) if do_validate else []
    expl = explain_patterns(date, patterns=res["patterns"])
    out = OUT_DIR / f"pattern_report_{date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_report(res, kb, inc, expl), encoding="utf-8")
    res["report_path"] = str(out)
    res["kb_total"] = len(kb)
    res["open_validated"] = inc
    res["explanations"] = expl
    return res


# ────────────────────────────────────────────────────────────
# RAG 规律逻辑解释（为什么有效）
# ────────────────────────────────────────────────────────────
_RAG_SEARCH_CACHE: dict[tuple[str, int], list[dict]] = {}


def _rag_search(query: str, k: int) -> list[dict]:
    """knowledge_rag.search 进程级 memo：同进程重复检索同一 query 不重算。

    knowledge_rag 每次调用会重新加载向量模型（约 10-20s），
    本 memo 保证 explain_patterns 在报告/决策同进程内多次调用不重复付这笔开销。
    """
    try:
        from quant_system.analysis_core import knowledge_rag as _kr
    except ImportError as e:
        raise RuntimeError(f"knowledge_rag 不可用: {e}") from e
    key = (query, k)
    if key not in _RAG_SEARCH_CACHE:
        _RAG_SEARCH_CACHE[key] = _kr.search(query, k=k)
    return _RAG_SEARCH_CACHE[key]


def _load_explanations() -> dict:
    if not EXPLANATION_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(EXPLANATION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_explanations(data: dict) -> str | None:
    """写解释缓存；失败返回错误信息（不抛异常，调用方优雅降级）。"""
    tmp = EXPLANATION_CACHE_PATH.with_suffix(".json.tmp")
    try:
        EXPLANATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        os.replace(tmp, EXPLANATION_CACHE_PATH)  # 原子替换, 避免并发读到半截 JSON
        return None
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception as e:
            logging.getLogger(__name__).error(f"[pattern_engine] 操作失败: {e}", exc_info=True)
        return f"缓存写失败: {str(e)[:120]}"


def explain_patterns(date: str | None = None,
                     patterns: list[dict] | None = None) -> dict:
    """RAG 规律逻辑解释：按 pattern_type 分组，检索 skill/知识库片段说明规律为何有效。

    - patterns 为空时读当日规律库（confirmed/validating/new 全量参与分组）；
    - 每组以 f'{规律中文名} {方向} 交易逻辑' 调 knowledge_rag.search(query, k=3)，
      top 命中（file/cat/score/摘要）作为该类型的"逻辑解释"；
    - 结果按 {date: {type: ...}} 缓存到 pattern_explanations.json，当日不重复检索；
    - knowledge_rag 检索失败 → 解释字段 '(检索不可用)'，不崩溃。
    """
    date = date or today()
    if patterns is None:
        kb = load_knowledge_base(date)
        patterns = []
        for _, r in kb.iterrows():
            p = dict(r)
            try:
                p["evidence"] = json.loads(p.get("evidence") or "[]")
            except Exception:
                p["evidence"] = []
            try:
                p["backtest_stats"] = json.loads(p.get("backtest_stats") or "{}")
            except Exception:
                p["backtest_stats"] = {}
            patterns.append(p)

    groups: dict[str, list[dict]] = {}
    for p in patterns:
        groups.setdefault(p.get("pattern_type") or "unknown", []).append(p)

    cache = _load_explanations()
    day = cache.setdefault(date, {})
    out: dict[str, dict] = {}
    # 本地向量模型优先走离线缓存，避免联网重试拖慢检索（不可用则 knowledge_rag 自动降级关键词）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for ptype, items in groups.items():
        if ptype in day:  # 当日已缓存 → 不重复检索
            out[ptype] = {**day[ptype], "cached": True}
            continue
        name = items[0].get("name") or TYPE_SPEC.get(ptype, {}).get("name", ptype)
        direction = items[0].get("direction") or ""
        dircn = _DIRECTION_CN.get(str(direction), str(direction) or "")
        query = f"{name} {dircn} 交易逻辑".strip()
        try:
            hits = _rag_search(query, k=RAG_QUERY_K)
            if hits:
                out[ptype] = {"query": query, "pattern_count": len(items),
                              "hits": [{"file": h.get("file"), "cat": h.get("cat", ""),
                                        "score": h.get("score"),
                                        "summary": (h.get("text") or "")[:200]}
                                       for h in hits],
                              "cached": False}
            else:
                out[ptype] = {"query": query, "pattern_count": len(items),
                              "hits": [], "logic": "(检索无命中)", "cached": False}
        except Exception as e:
            out[ptype] = {"query": query, "pattern_count": len(items),
                          "hits": [], "logic": "(检索不可用)", "error": str(e)[:120],
                          "cached": False}
    day.update(out)
    write_err = _save_explanations(cache)
    return {"date": date, "explanations": out,
            "cached_types": [t for t, v in out.items() if v.get("cached")],
            "cache_path": str(EXPLANATION_CACHE_PATH),
            "cache_write": write_err or "ok"}


def _render_explanation_section(expl: dict) -> str:
    """规律逻辑解释 markdown 章节（类型 → 检索到的 skill 依据 + 摘要）。"""
    lines = ["## 规律逻辑解释（RAG）", ""]
    exps = expl.get("explanations") or {}
    if not exps:
        lines.append("- 当日无规律类型可解释")
        return "\n".join(lines)
    for ptype, e in exps.items():
        spec = TYPE_SPEC.get(ptype, {})
        lines.append(f"### {spec.get('name', ptype)}（`{ptype}`）")
        lines.append("")
        lines.append(f"- 检索式: `{e.get('query', '')}`（{e.get('pattern_count', 0)} 条规律）"
                     + (" | ✅ 当日缓存未重复检索" if e.get("cached") else ""))
        hits = e.get("hits") or []
        if not hits:
            lines.append(f"- 逻辑解释: {e.get('logic', '(检索无命中)')}")
            if e.get("error"):
                lines.append(f"- 检索异常: {e['error']}")
        else:
            lines.append(f"- 逻辑解释: top{RAG_QUERY_K} 命中 skill/知识库依据")
            for h in hits:
                lines.append(f"  - 📚 [{h.get('cat', '')}] `{h.get('file')}` "
                             f"(score {h.get('score')})")
                lines.append(f"    - {h.get('summary', '')[:160]}")
        lines.append("")
    return "\n".join(lines)


def _render_explanation_report(expl: dict) -> str:
    """独立规律逻辑解释报告（--explain）。"""
    head = [f"# 规律逻辑解释报告 {expl.get('date', '')}",
            "", f"- 缓存: `{expl.get('cache_path', '')}` | 写缓存: {expl.get('cache_write', '')}", ""]
    return "\n".join(head) + _render_explanation_section(expl)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="自主学习者·规律引擎")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD")
    ap.add_argument("--report", action="store_true", help="检测+验证+入库+报告")
    ap.add_argument("--detect", action="store_true", help="仅检测（不验证不入库）")
    ap.add_argument("--validate", metavar="PATTERN_ID", default=None,
                    help="对指定规律执行历史回放验证")
    ap.add_argument("--weekly", action="store_true", help="周度规律库复盘")
    ap.add_argument("--mark", metavar="PATTERN_ID", default=None, help="人工标注规律")
    ap.add_argument("--action", choices=["confirm", "deprecate", "note"], default="note")
    ap.add_argument("--note", default="", help="标注说明")
    ap.add_argument("--universe", type=int, default=DEFAULT_UNIVERSE_LIMIT)
    ap.add_argument("--scan-limit", type=int, default=DEFAULT_SCAN_LIMIT)
    ap.add_argument("--no-validate", action="store_true", help="报告跳过验证")
    ap.add_argument("--explain", action="store_true",
                    help="单独生成规律逻辑解释报告(RAG，当日缓存不重复检索)")
    args = ap.parse_args()

    if args.weekly:
        r = weekly_review(args.date)
        print(json.dumps({k: v for k, v in r.items() if k != "summary"},
                         ensure_ascii=False, indent=2, default=str))
        return
    if args.mark:
        r = mark_pattern(args.mark, args.action, args.note)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    if args.validate:
        kb = _load_kb()
        row = kb[kb["pattern_id"] == args.validate]
        if row.empty:
            print(f"规律 {args.validate} 不在库中", file=sys.stderr)
            sys.exit(1)
        p = row.iloc[0].to_dict()
        p["evidence"] = json.loads(p.get("evidence") or "[]")
        p["backtest_stats"] = json.loads(p.get("backtest_stats") or "{}")
        p = validate_pattern(p, verbose=True)
        print(json.dumps({k: p[k] for k in ("pattern_id", "status", "backtest_stats")},
                         ensure_ascii=False, indent=2, default=str))
        return
    if args.explain:
        r = explain_patterns(args.date)
        path = OUT_DIR / f"pattern_explanations_{r['date']}.md"
        write_note = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render_explanation_report(r), encoding="utf-8")
        except Exception as e:
            write_note = f"报告写失败: {str(e)[:120]}"
        r["report_path"] = str(path)
        r["report_write"] = write_note or "ok"
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        return
    if args.report:
        r = report(args.date, args.universe, args.scan_limit, not args.no_validate, verbose=True)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("patterns", "skipped")}, ensure_ascii=False, indent=2, default=str))
        return
    # 默认：仅检测（可读性输出）
    r = detect_patterns(args.date, args.universe, args.scan_limit, verbose=True)
    print(f"[规律检测] {r['date']} | 触发 {len(r['patterns'])} | 跳过 {len(r['skipped'])} "
          f"| 宇宙 {r['universe']} | 耗时 {r['elapsed_sec']}s")
    for p in r["patterns"]:
        print(f"- {p['name']} | 置信 {p['confidence']:.2f} | {p.get('direction', '—')} "
              f"| {p.get('object_name') or p.get('object_code') or '市场'}")
        for e in p["evidence"]:
            print(f"    · {e}")
    for s in r["skipped"]:
        print(f"[跳过] {s}")


if __name__ == "__main__":
    main()
