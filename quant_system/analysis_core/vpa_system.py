#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vpa_system — 体系3 量价筹码系统（Anna Coulling VPA + 威科夫 + Livermore 关键价位确认）

方法论（skills/volume-price-analysis + mainline-volume-capital-flow-trading +
livermore-pivotal-point-breakout）:
  - 威科夫三定律: 供求（量价配合）/ 因果（吸筹→上涨）/ 投入产出（价格变动需要量能）。
  - 吸筹: 下跌末端缩量 + 测试（下探收回）+ 震仓；派发: 上涨末端放量滞涨 + 上冲回落。
  - 止跌量: 大跌后巨量长下影；买入高潮: 天量滞涨；量价背离: 价涨量缩=动能衰竭、价跌量增=恐慌/承接。
  - Livermore: 关键价位（密集区上沿/POC）突破需市场确认——放量突破=真突破，无量突破=假突破。

每只标的五层检测:
  1. 量价关系: 5日价量相关性 + 量比分布 → 放量涨/缩量涨/放量跌/缩量跌/中性。
  2. 威科夫阶段: 规则引擎（60日价格分位 + 量能形态序列）→
     吸筹/派发/测试/止跌量/买入高潮/中性。
  3. 密集区博弈: 当前价 vs POC/上沿/下沿（复用 volume_profile）→ 突破/假突破/区间内/跌破。
  4. 主力行为: 复用 fund_flow_divergence.check_divergence → 背离/承接/正常/数据不足。
  5. 综合: 阶段×量价×密集区（主力作置信修正）→ 看多/看空/中性 + confidence。

防御:
  - 单标的任一步骤失败 → 跳过该股，不影响整体。
  - 量比 0.2–8 过滤异常（停牌/数据错误导致的极端量比不参与统计）。
  - 东财资金流接口失败 → 主力行为标注"数据不足"，置信度下调。
  - RAG 检索失败 → 方法论依据标注不可用，不中断。

数据契约:
  data_warehouse/kline/{6位代码}.parquet  日K（date/open/high/low/close/volume）
  东财 push2delay 个股资金流日线（fund_flow_divergence 复用）

用法:
  python3 -m quant_system.analysis_core.vpa_system --watch 601899,600519
  python3 -m quant_system.analysis_core.vpa_system --date 2026-08-07
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # RAG 本地向量模型，避免网络重试

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.common import load_names, read_kline_window  # noqa: E402
from quant_system.analysis_core.volume_profile import compute_volume_profile, fmt_pos  # noqa: E402
from quant_system.analysis_core.fund_flow_divergence import (  # noqa: E402
    check_divergence,
    fmt_amount,
)

KLINE_DIR = ROOT / "data_warehouse" / "kline"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
WATCHLIST_FILE = CONFIG_DIR / "watchlist.json"
WATCHLIST_TEMPLATE = {"watch": ["600519", "000001"]}

CODE_RE = re.compile(r"^\d{6}$")
LOOKBACK_DAYS = 300         # 日历回看窗口（≈200 交易日，覆盖 60日分位/VP60/MA20 余量）
MIN_DAYS = 61               # 威科夫 60 日分位所需最少样本
PRICE_WIN = 60              # 价格位置窗口（60日分位）
VOL_AVG_N = 20              # 均量窗口
CORR_N = 5                  # 5日价量相关性
RATIO_FLOOR, RATIO_CEIL = 0.2, 8.0   # 量比过滤区间（异常量比剔除）
VOL_UP, VOL_DOWN = 1.5, 0.7          # 放量/缩量量比阈值
VOL_UP_STRONG = 1.8         # 强放量（止跌量/滞涨）
VOL_CLIMAX = 2.5            # 高潮量比（买入高潮）
WICK_LOW = 0.03             # 长下影 (close-low)/close
WICK_HIGH = 0.03            # 长上影 (high-close)/close
STAGE_LOOKBACK = 10         # 威科夫模式序列回看
STAGE_RECENT = 5            # 近期模式窗口（止跌量/买入高潮）
TEST_RECENT = 3             # 测试窗口
ZONE_UP_RATIO = 1.3         # 放量突破/跌破量比阈值

POS_LOW = 0.40              # 低位分位阈值（吸筹）
POS_HIGH = 0.72             # 高位分位阈值（派发）
POS_TEST_LOW = 0.18         # 低点测试位置阈值

RAG_QUERY = "量价分析 吸筹 派发 威科夫"
RAG_K = 3


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def load_kline(code: str, target: pd.Timestamp | None = None) -> tuple[pd.DataFrame | None, str]:
    """最近约 200 个交易日的日K（date/open/high/low/close/volume），截断至 target。"""
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "kline文件缺失"
    window = (pd.Timestamp(target) if target is not None else pd.Timestamp.now()).normalize()
    try:
        df = read_kline_window(path, ["date", "open", "high", "low", "close", "volume"],
                               window - pd.Timedelta(days=LOOKBACK_DAYS))
    except Exception as e:
        return None, f"读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "无数据"
    if "close" not in df.columns:
        return None, "缺关键列"
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= window]
    if df.empty:
        return None, "目标日无交易"
    return df.reset_index(drop=True), ""


def resolve_target(codes: list[str], date: str | None = None) -> pd.Timestamp:
    """目标交易日：--date 指定，否则取所有 watch 股票本地 kline 的最新共同日期。"""
    if date:
        return pd.Timestamp(date).normalize()
    cands = []
    for code in codes:
        p = KLINE_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        try:
            d = pd.read_parquet(p, columns=["date"])
            m = pd.to_datetime(d["date"], errors="coerce").max()
            if m is not None and pd.notna(m):
                cands.append(m)
        except Exception as e:
            logging.getLogger(__name__).error(f"[vpa_system] 操作失败: {e}", exc_info=True)
            continue
    return pd.Timestamp(min(cands)).normalize() if cands else pd.Timestamp.now().normalize()


def _vol_ratio_series(vols: pd.Series, n: int = VOL_AVG_N) -> pd.Series:
    """抗单位切换的日量比序列：量比 = 当日量 / 前n日中位量。

    前n日按"单位一致片段"回溯——相邻日量跳变 >20 倍视为 股/手 等单位切换，
    截断后再取中位数（最少 3 个一致样本，否则该日量比记 NaN）。
    """
    vals = vols.to_numpy(dtype=float)
    m = len(vals)
    out = np.full(m, np.nan)
    for i in range(m):
        if not np.isfinite(vals[i]) or vals[i] <= 0:
            continue
        tail: list[float] = []
        j = i - 1
        while j >= 0 and len(tail) < n:
            v = vals[j]
            if not np.isfinite(v) or v <= 0:
                j -= 1
                continue
            if tail and (v / tail[-1] > 20.0 or tail[-1] / v > 20.0):
                break
            tail.append(v)
            j -= 1
        if len(tail) >= 3:
            out[i] = vals[i] / float(np.median(tail))
    return pd.Series(out, index=vols.index)


def _last_vol_ratio(df: pd.DataFrame) -> float:
    """当日量比（抗单位切换）；过滤异常（0.2–8）后返回。"""
    vols = pd.to_numeric(df["volume"], errors="coerce")
    ratio = _vol_ratio_series(vols).iloc[-1]
    if not np.isfinite(ratio) or ratio <= 0:
        return 1.0
    return float(np.clip(ratio, RATIO_FLOOR, RATIO_CEIL))


def _pos60_series(closes: pd.Series, win: int = PRICE_WIN) -> pd.Series:
    """滚动 60 日价格分位（0~1，当日收盘在窗口 [min,max] 的位置）。"""
    def _pct(x: np.ndarray) -> float:
        if len(x) < 2 or not (x.max() > x.min()):
            return 0.5
        return float((x[-1] - x.min()) / (x.max() - x.min()))
    return closes.rolling(win, min_periods=win).apply(_pct, raw=True)


# ────────────────────────────────────────────────────────────
# 1. 量价关系
# ────────────────────────────────────────────────────────────
def price_volume_relation(df: pd.DataFrame) -> dict:
    """5日价量相关性 + 量比分布 → 放量涨/缩量涨/放量跌/缩量跌/中性。

    量比 = 当日量/前20日中位量（单位一致片段回溯，抗 股/手 切换）；量比 <0.2 或 >8 剔除。
    相关 >0 表示价涨量增（供求正常）；<0 表示价量背离（动能衰竭/恐慌承接）。
    """
    out = {"ok": False, "label": "样本不足", "score": 0.0, "corr": np.nan,
           "cum_ret": np.nan, "mean_ratio": np.nan, "up_days": 0, "down_days": 0,
           "filtered": 0, "note": ""}
    closes = pd.to_numeric(df["close"], errors="coerce")
    vols = pd.to_numeric(df["volume"], errors="coerce")
    ratio = _vol_ratio_series(vols)
    rets = closes.pct_change() * 100.0
    cur = pd.concat([closes, ratio, rets], axis=1)
    cur.columns = ["close", "ratio", "ret"]
    cur = cur.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cur) < CORR_N + 1:
        return out
    tail = cur.tail(CORR_N + 1)
    ratios = tail["ratio"].values[1:]
    rets5 = tail["ret"].values[1:]
    ok_mask = (ratios >= RATIO_FLOOR) & (ratios <= RATIO_CEIL)
    r5 = rets5[ok_mask]
    q5 = ratios[ok_mask]
    cum_ret = float(tail["close"].iloc[-1] / tail["close"].iloc[0] - 1.0) * 100.0
    corr = np.nan
    if len(r5) >= 3 and r5.std() > 0 and q5.std() > 0:
        corr = float(np.corrcoef(r5, q5)[0, 1])
    mean_ratio = float(q5.mean()) if len(q5) else np.nan
    out.update({"ok": True, "corr": corr, "cum_ret": cum_ret, "mean_ratio": mean_ratio,
                "up_days": int((q5 >= VOL_UP).sum()), "down_days": int((q5 <= VOL_DOWN).sum()),
                "filtered": int((~ok_mask).sum())})
    if not np.isfinite(mean_ratio):
        out.update({"label": "中性", "note": "量比全部被过滤(异常)"})
        return out
    up = cum_ret > 0.5
    down = cum_ret < -0.5
    if up and mean_ratio >= 1.2:
        out["label"], out["score"] = "放量涨", 1.0
    elif up and mean_ratio <= 0.9:
        out["label"], out["score"] = "缩量涨", 0.4
    elif down and mean_ratio >= 1.2:
        out["label"], out["score"] = "放量跌", -1.0
    elif down and mean_ratio <= 0.9:
        out["label"], out["score"] = "缩量跌", -0.4
    else:
        out["label"], out["score"] = "中性", 0.0
    if out["filtered"]:
        out["note"] = f"量比过滤异常 {out['filtered']} 日(0.2–8)"
    return out


# ────────────────────────────────────────────────────────────
# 2. 威科夫阶段（规则引擎）
# ────────────────────────────────────────────────────────────
def wyckoff_stage(df: pd.DataFrame) -> dict:
    """规则引擎: 60日价格分位 + 量能形态序列 → 吸筹/派发/测试/止跌量/买入高潮/中性。

    模式:
      止跌量: 放量 + 长下影 + 收跌（大跌后巨量长下影）。
      抛售高潮: 跌幅>5% + 量比>2.5（恐慌性抛售，底部构筑开始）。
      买入高潮: 量比>2.5 + 涨幅<1% + 长上影（天量滞涨/上冲回落）。
      放量滞涨: 量比>1.8 + |涨跌|<1.5% + 长上影。
      测试: 价格回到 60日低点附近（分位<0.18）——低量=供应枯竭，高量=供应仍存。
      震仓: 低点击穿前5日低点但收盘收回前收（下探收回）+ 放量。
      无需求/无供给: 反弹/回调无量。
    """
    out = {"ok": False, "label": "样本不足", "score": 0.0, "pos60": np.nan,
           "patterns": [], "desc": ""}
    closes = pd.to_numeric(df["close"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")
    highs = pd.to_numeric(df["high"], errors="coerce")
    vols = pd.to_numeric(df["volume"], errors="coerce")
    n = len(df)
    if n < MIN_DAYS:
        return out

    win = closes.iloc[-PRICE_WIN:]
    lo60, hi60 = float(win.min()), float(win.max())
    pos60 = float((closes.iloc[-1] - lo60) / (hi60 - lo60)) if hi60 > lo60 else 0.5
    pos60_s = _pos60_series(closes)
    ratio = _vol_ratio_series(vols)
    wick_low = (closes - lows) / closes
    wick_high = (highs - closes) / closes
    pct = closes.pct_change() * 100.0

    idxs = list(range(max(1, n - STAGE_LOOKBACK), n))
    patterns: list[dict] = []
    dts = pd.to_datetime(df["date"]).tolist()
    for i in idxs:
        vr = ratio.iloc[i]
        if not np.isfinite(vr) or vr <= 0:
            continue
        vr = float(np.clip(vr, RATIO_FLOOR, RATIO_CEIL))
        pc, wl, wh = float(pct.iloc[i]), float(wick_low.iloc[i]), float(wick_high.iloc[i])
        day_pos = float(pos60_s.iloc[i]) if np.isfinite(pos60_s.iloc[i]) else 0.5
        d = df["date"].iloc[i]
        if vr >= VOL_UP_STRONG and wl >= WICK_LOW and pc < 0:
            patterns.append({"date": str(d), "idx": i, "kind": "止跌量",
                             "desc": f"放量{vr:.1f}x+长下影+收跌"})
        elif pc <= -5.0 and vr >= VOL_CLIMAX:
            patterns.append({"date": str(d), "idx": i, "kind": "抛售高潮",
                             "desc": f"恐慌放量{vr:.1f}x 跌{pc:.1f}%"})
        elif vr >= VOL_CLIMAX and pc <= 1.0 and wh >= WICK_HIGH * 0.5:
            patterns.append({"date": str(d), "idx": i, "kind": "买入高潮",
                             "desc": f"天量{vr:.1f}x滞涨(上冲回落)"})
        elif vr >= VOL_UP_STRONG and abs(pc) <= 1.5 and wh >= WICK_HIGH:
            patterns.append({"date": str(d), "idx": i, "kind": "放量滞涨",
                             "desc": f"量{vr:.1f}x 涨跌{pc:+.1f}% 长上影"})
        elif vr >= 1.5 and pc <= -3.5 and wl < 0.02:
            patterns.append({"date": str(d), "idx": i, "kind": "放量长阴",
                             "desc": f"高位放量{vr:.1f}x 长阴跌{pc:.1f}%(派发/恐慌)"})
        elif day_pos <= POS_TEST_LOW:
            kind = "低量测试" if vr <= 0.8 else ("高量测试" if vr >= 1.5 else "测试")
            patterns.append({"date": str(d), "idx": i, "kind": kind,
                             "desc": f"回试低点(分位{day_pos:.0%}, 量比{vr:.1f})"})
        elif day_pos >= 1 - POS_TEST_LOW and vr <= 0.8:
            patterns.append({"date": str(d), "idx": i, "kind": "高位低量测试",
                             "desc": f"回试高点无量(分位{day_pos:.0%})"})
        if i >= 5 and float(lows.iloc[i]) < float(lows.iloc[i - 5:i].min()) \
                and float(closes.iloc[i]) > float(closes.iloc[i - 1]) and vr >= 1.2:
            patterns.append({"date": str(d), "idx": i, "kind": "震仓",
                             "desc": f"下探收回(击穿前5日低点后收复, 量{vr:.1f}x)"})
        if pc > 0 and vr <= VOL_DOWN:
            patterns.append({"date": str(d), "idx": i, "kind": "无需求", "desc": "反弹无量"})
        if pc < 0 and vr <= VOL_DOWN:
            patterns.append({"date": str(d), "idx": i, "kind": "无供给", "desc": "回调无量"})

    def _recent(days: int) -> set[str]:
        # 交易日窗口：模式日与最新交易日的间隔按交易日计数（周末/节假日不计）
        return {p["kind"] for p in patterns
                if len(dts) - 1 - _trading_pos(p["date"]) <= days}

    def _trading_pos(d: object) -> int:
        pos = int(pd.DatetimeIndex(dts).searchsorted(pd.Timestamp(d)))
        return min(pos, len(dts) - 1)

    recent5 = _recent(5)      # 近5个交易日
    recent3 = _recent(3)      # 近3个交易日
    recent_all = {p["kind"] for p in patterns}

    stage, score = "中性", 0.0
    if recent5 & {"买入高潮"} and pos60 > 0.55:
        stage, score = "买入高潮", -0.8
    elif recent5 & {"止跌量", "抛售高潮"} and pos60 < 0.6:
        stage, score = "止跌量", 0.7
    elif recent3 & {"低量测试"}:
        stage, score = "测试", 0.5
    elif recent3 & {"高量测试"}:
        stage, score = "测试", -0.5
    elif pos60 < POS_LOW and (recent_all & {"止跌量", "抛售高潮", "震仓", "低量测试"}):
        stage, score = "吸筹", 0.8
    elif pos60 > POS_HIGH and (recent_all & {"买入高潮", "放量滞涨", "高位低量测试", "放量长阴"}):
        stage, score = "派发", -0.9

    desc = ""
    if stage == "吸筹":
        desc = "低位+止跌/震仓/低量测试序列，筹码由弱转强"
    elif stage == "派发":
        desc = "高位+放量滞涨/买入高潮/放量长阴，筹码由强转弱"
    elif stage == "测试":
        desc = "低量→供应枯竭(成功)；高量→供应仍存(失败)"
    elif stage == "止跌量":
        desc = "大跌后巨量长下影，主力恐慌中吸收卖盘"
    elif stage == "买入高潮":
        desc = "天量滞涨，公众狂热买入/主力借流动性派发"
    else:
        desc = "无明确吸筹/派发证据"
    out.update({"ok": True, "label": stage, "score": score, "pos60": pos60,
                "patterns": [{"kind": p["kind"], "date": p["date"], "desc": p["desc"]}
                             for p in patterns], "desc": desc})
    return out


# ────────────────────────────────────────────────────────────
# 3. 密集区博弈
# ────────────────────────────────────────────────────────────
def dense_zone_game(vp: dict, vol_ratio: float, pct_today: float = 0.0) -> dict:
    """当前价 vs POC/上沿/下沿（复用 volume_profile）→ 突破/假突破/区间内/跌破。

    Livermore: 上沿突破需放量市场确认——放量=真突破，缩量=假突破/陷阱。
    当日收跌时上方放量视为"上冲回落"而非突破。
    """
    out = {"ok": False, "zone": "无密集区", "label": vp.get("error", "样本不足"),
           "score": 0.0, "detail": vp.get("error", "样本不足"),
           "poc": vp.get("poc"), "upper": vp.get("upper"), "lower": vp.get("lower")}
    if not vp.get("ok"):
        return out
    label = vp.get("pos_label") or "区间内"
    pos_pct = vp.get("pos_pct")
    vr = vol_ratio if np.isfinite(vol_ratio) else 1.0
    out["ok"] = True
    out["detail"] = fmt_pos(vp)
    if label == "上方":
        if vr >= ZONE_UP_RATIO and pct_today > 0:
            out.update({"zone": "突破", "label": "放量突破", "score": 0.8})
        elif vr >= ZONE_UP_RATIO:
            out.update({"zone": "突破", "label": "上沿上方放量回落", "score": -0.3})
        else:
            out.update({"zone": "假突破", "label": "无量突破(需确认)", "score": -0.6})
    elif label == "下方":
        if vr >= ZONE_UP_RATIO:
            out.update({"zone": "跌破", "label": "放量跌破", "score": -0.8})
        else:
            out.update({"zone": "回踩", "label": "缩量回踩(测试)", "score": 0.3})
    elif label == "贴POC":
        out.update({"zone": "区间内", "label": "贴POC", "score": 0.15})
    else:  # 区间内
        if pos_pct is not None and pos_pct >= 60:
            out.update({"zone": "区间内", "label": "区间内·上方", "score": 0.2})
        elif pos_pct is not None and pos_pct <= 40:
            out.update({"zone": "区间内", "label": "区间内·下方", "score": -0.2})
        else:
            out.update({"zone": "区间内", "label": "区间内·中部", "score": 0.0})
    return out


# ────────────────────────────────────────────────────────────
# 4. 主力行为（复用 fund_flow_divergence）
# ────────────────────────────────────────────────────────────
def fund_behavior(code: str, target: pd.Timestamp | None = None) -> dict:
    """主力净流方向 vs 价格方向 → 背离/承接/正常/数据不足。

    复用 fund_flow_divergence.check_divergence；接口/解析失败自动标注"数据不足"。
    """
    try:
        d = check_divergence(code, target)
    except Exception as e:
        return {"state": "数据不足", "note": f"接口异常 {str(e)[:40]}", "score": 0.0,
                "signal": "unavailable", "main_net": None, "date": None}
    sig = d.get("signal") or ""
    note = d.get("note") or ""
    if not sig or sig == "unavailable" or "跳过" in sig or "数据不足" in sig:
        return {"state": "数据不足", "note": (note or sig or "接口失败")[:60], "score": 0.0,
                "signal": sig, "main_net": d.get("main_net"), "date": d.get("date")}
    if "背离" in sig:
        state, score = "背离", -0.7
    elif "承接" in sig:
        state, score = "承接", 0.7
    else:
        state, score = "正常", 0.15
    return {"state": state, "note": str(sig)[:60], "score": score, "signal": sig,
            "main_net": d.get("main_net"), "date": d.get("date")}


# ────────────────────────────────────────────────────────────
# 5. 综合
# ────────────────────────────────────────────────────────────
def _compose(price: dict, stage: dict, zone: dict, fund: dict) -> dict:
    """阶段×量价×密集区 加权 → 看多/看空/中性 + confidence（主力作修正）。"""
    score = (0.30 * price["score"] + 0.35 * stage["score"]
             + 0.25 * zone["score"] + 0.10 * fund["score"])
    if score > 0.2:
        signal = "看多"
    elif score < -0.2:
        signal = "看空"
    else:
        signal = "中性"
    conf = 0.35 + 0.5 * abs(score)
    if fund["state"] == "数据不足":
        conf *= 0.9
    conf = round(min(conf, 0.95), 2)
    pos60 = stage.get("pos60")
    pos_txt = f"（60日分位 {pos60:.0%}）" if np.isfinite(pos60) else ""
    corr = price.get("corr")
    corr_txt = f"{corr:.2f}" if np.isfinite(corr) else "—"
    mean_ratio = price.get("mean_ratio")
    ratio_txt = f"量比均值 {mean_ratio:.2f}" if np.isfinite(mean_ratio) else ""
    evidence = [
        f"威科夫阶段: {stage['label']}{pos_txt}——{stage.get('desc', '')}",
        f"量价关系: {price['label']}（5日价量相关 {corr_txt}，{ratio_txt}）",
        f"密集区博弈: {zone['label']}（{zone['detail']}）",
        f"主力行为: {fund['state']}（{fund['note']}）",
    ]
    return {"signal": signal, "confidence": conf, "score": round(float(score), 3),
            "evidence": evidence}


# ────────────────────────────────────────────────────────────
# VpaSystem 统一接口
# ────────────────────────────────────────────────────────────
class VpaSystem:
    """量价筹码系统统一接口: detect(date)/report(date)/view(date)。"""

    def __init__(self, watch: list[str] | None = None,
                 out_dir: Path | None = None) -> None:
        self.watch = [str(c).zfill(6) for c in (watch or WATCHLIST_TEMPLATE["watch"])]
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self.names, _ = load_names()
        self._detect_cache: dict | None = None
        self._cache_key: str | None = None
        self._rag: list[dict] | None = None

    # ---- RAG 方法论依据（懒加载 + 缓存） ----
    def _rag_basis(self) -> list[dict]:
        if self._rag is None:
            try:
                from quant_system.analysis_core.knowledge_rag import search
                self._rag = search(RAG_QUERY, k=RAG_K)
            except Exception as e:
                self._rag = [{"file": "(rag不可用)", "cat": "—", "score": 0.0,
                              "text": f"RAG检索失败: {str(e)[:80]}"}]
        return self._rag

    # ---- 单标的分析 ----
    def _analyze_one(self, code: str, target: pd.Timestamp) -> dict:
        row = {"code": code, "name": self.names.get(code, ""), "ok": False,
               "error": "", "signal": "中性", "confidence": 0.0, "score": 0.0,
               "evidence": []}
        try:
            df, note = load_kline(code, target)
            if df is None or len(df) < MIN_DAYS:
                row["error"] = note or f"样本不足(<{MIN_DAYS}日)"
                return row
            price = price_volume_relation(df)
            stage = wyckoff_stage(df)
            vp = compute_volume_profile(df)
            closes = pd.to_numeric(df["close"], errors="coerce")
            pct_today = float(closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0 \
                if len(closes) >= 2 and closes.iloc[-2] > 0 else 0.0
            zone = dense_zone_game(vp, _last_vol_ratio(df), pct_today)
            fund = fund_behavior(code, target)
            comp = _compose(price, stage, zone, fund)
            row.update({"ok": True, "price": price, "stage": stage, "vp": vp,
                        "zone": zone, "fund": fund, "signal": comp["signal"],
                        "confidence": comp["confidence"], "score": comp["score"],
                        "evidence": comp["evidence"]})
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return row

    def _run_detect(self, date: str | None) -> dict:
        target = resolve_target(self.watch, date)
        t0 = time.time()
        rows = [self._analyze_one(c, target) for c in self.watch]
        elapsed = time.time() - t0
        ok_rows = [r for r in rows if r["ok"]]
        signals = {"看多": 0, "看空": 0, "中性": 0}
        for r in ok_rows:
            signals[r["signal"]] = signals.get(r["signal"], 0) + 1
        return {"date": str(target.date()), "target": target, "rows": rows,
                "summary": {"date": str(target.date()), "total": len(rows),
                            "ok": len(ok_rows), "failed": len(rows) - len(ok_rows),
                            "signals": signals, "elapsed": round(elapsed, 1)},
                "rag": self._rag_basis()}

    def _detect_cached(self, date: str | None) -> dict:
        key = date or "auto"
        if self._cache_key != key:
            self._detect_cache = self._run_detect(date)
            self._cache_key = key
        return self._detect_cache

    def detect(self, date: str | None = None) -> dict:
        """全量检测：每只标的 量价关系/威科夫阶段/密集区博弈/主力行为/综合。"""
        return self._detect_cached(date)

    def view(self, date: str | None = None) -> dict:
        """多空观点（供 multi_agent 使用）: {agent, signal, confidence, evidence}。"""
        det = self._detect_cached(date)
        rows = [r for r in det["rows"] if r["ok"]]
        if not rows:
            return {"agent": "量价筹码", "signal": "中性", "view": "震荡",
                    "confidence": 0.0, "evidence": ["无有效标的（全部跳过）"],
                    "weight": 1.0, "status": "degraded", "detail": det}
        score_avg = float(np.mean([r["score"] for r in rows]))
        if score_avg > 0.15:
            signal = "看多"
        elif score_avg < -0.15:
            signal = "看空"
        else:
            signal = "中性"
        conf = min(0.95, 0.35 + 0.45 * abs(score_avg) + 0.05 * min(1.0, len(rows) / 5.0))
        conf = round(conf, 2)
        top = sorted(rows, key=lambda r: -abs(r["score"]))[:5]
        evidence = [
            f"{r['name'] or r['code']}({r['code']}) {r['signal']} 置信{r['confidence']:.2f}："
            f"威科夫{r['stage']['label']}/量价{r['price']['label']}/"
            f"密集区{r['zone']['label']}/主力{r['fund']['state']}"
            for r in top
        ]
        evidence.insert(0, f"有效标的 {len(rows)}/{len(det['rows'])} 只，"
                           f"综合分 {score_avg:+.2f}")
        return {"agent": "量价筹码", "signal": signal,
                "view": {"看多": "多", "看空": "空", "中性": "震荡"}.get(signal, "震荡"),
                "confidence": conf, "evidence": evidence,
                "weight": 1.0, "status": "ok", "detail": det}

    # ---- Markdown 报告 ----
    def render_markdown(self, det: dict) -> str:
        s = det["summary"]
        lines = [f"# 量价筹码系统 VPA — {s['date']}", "",
                 f"- 标的: {s['total']} 只 | 可用: {s['ok']} 只 | 跳过: {s['failed']} 只 | "
                 f"耗时 {s['elapsed']}s",
                 "- 方法论: Anna Coulling 量价分析(VPA) + 威科夫三定律（供求/因果/投入产出）"
                 " + Livermore 关键价位突破确认",
                 "- 数据源: 本地日K(OHLCV) + 60日简化Volume Profile + 东财主力资金流（估算口径）", "",
                 "## 方法论依据（RAG: 量价分析 吸筹 派发 威科夫）", ""]
        for i, r in enumerate(det.get("rag") or [], 1):
            lines.append(f"{i}. `{r.get('file', '')}` ({r.get('cat', '—')}, "
                         f"score {r.get('score', 0)}) — "
                         f"{str(r.get('text', ''))[:120].replace(chr(10), ' ')}")
        if not det.get("rag"):
            lines.append("_无检索结果_")

        lines += ["", "## 汇总", "",
                  "| 代码 | 名称 | 威科夫阶段 | 量价关系 | 密集区博弈 | 主力行为 | 信号 | 置信 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in det["rows"]:
            code, name = r["code"], r["name"]
            if not r["ok"]:
                lines.append(f"| {code} | {name} | — | — | — | — | 跳过 | "
                             f"{r['error'][:30]} |")
                continue
            net = r["fund"].get("main_net")
            fund_txt = f"{r['fund']['state']} {fmt_amount(net)}" if net is not None \
                else r["fund"]["state"]
            lines.append(f"| {code} | {name} | {r['stage']['label']} | "
                         f"{r['price']['label']} | {r['zone']['label']} | "
                         f"{fund_txt} | {r['signal']} | {r['confidence']:.2f} |")

        lines += ["", "## 明细", ""]
        for r in det["rows"]:
            code, name = r["code"], r["name"]
            title = f"{name}({code}) — {r['signal']}" if r["ok"] else \
                f"{name}({code}) — 跳过"
            lines.append(f"### {title}")
            if not r["ok"]:
                lines.append(f"- 错误: {r['error']}")
                lines.append("")
                continue
            st = r["stage"]
            pos60 = st.get("pos60")
            pos_txt = f"（60日分位 {pos60:.0%}）" if np.isfinite(pos60) else ""
            lines.append(f"- 威科夫阶段: **{st['label']}**{pos_txt} — {st['desc']}")
            if st.get("patterns"):
                pat = "；".join(f"{str(p['date'])[:10]} {p['kind']}({p['desc']})"
                                for p in st["patterns"][-5:])
                lines.append(f"  - 模式序列: {pat}")
            pr = r["price"]
            corr = pr.get("corr")
            corr_txt = f"{corr:.2f}" if np.isfinite(corr) else "—"
            cum_ret = pr.get("cum_ret")
            cum_txt = f"{cum_ret:+.2f}%" if np.isfinite(cum_ret) else "—"
            mean_ratio = pr.get("mean_ratio")
            ratio_txt = f"{mean_ratio:.2f}" if np.isfinite(mean_ratio) else "—"
            lines.append(f"- 量价关系: **{pr['label']}** — 5日价量相关 {corr_txt}，"
                         f"5日累计 {cum_txt}，量比均值 {ratio_txt}"
                         + (f"（过滤异常 {pr['filtered']} 日）" if pr.get("filtered") else ""))
            zn = r["zone"]
            if zn.get("poc"):
                lines.append(f"- 密集区博弈: **{zn['label']}**（{zn['detail']}）"
                             f" — POC {zn.get('poc'):.2f} / 上沿 {zn.get('upper'):.2f} / "
                             f"下沿 {zn.get('lower'):.2f}")
            else:
                lines.append(f"- 密集区博弈: **{zn['label']}**（{zn['detail']}）")
            fd = r["fund"]
            lines.append(f"- 主力行为: **{fd['state']}** — {fd['note']}")
            lines.append(f"- 结论: **{r['signal']}**（置信 {r['confidence']:.0%}）")
            for ev in r["evidence"]:
                lines.append(f"  - {ev}")
            lines.append("")
        lines.append("---")
        lines.append("*口径: 量比=当日量/前20日中位量(单位一致片段回溯, 0.2–8过滤)；60日分位=收盘在60日"
                     "[min,max]的位置；主力资金流为东财按单笔金额分桶估算，仅供参考。*")
        return "\n".join(lines) + "\n"

    def report(self, date: str | None = None) -> Path:
        """生成并保存 generated/vpa_report_{date}.md，返回文件路径。"""
        det = self._detect_cached(date)
        md = self.render_markdown(det)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / f"vpa_report_{det['date']}.md"
        out.write_text(md, encoding="utf-8")
        return out


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def _load_watchlist() -> list[str]:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not WATCHLIST_FILE.exists():
            WATCHLIST_FILE.write_text(
                json.dumps(WATCHLIST_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            return [str(c) for c in WATCHLIST_TEMPLATE["watch"]]
        data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("watch") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(c).strip().zfill(6) for c in raw if str(c).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="体系3 量价筹码系统（VPA + 威科夫 + Livermore）")
    ap.add_argument("--watch", default=None, help="股票代码，逗号分隔（缺省读 config/watchlist.json）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认 kline 最新交易日）")
    ap.add_argument("--no-report", action="store_true", help="不写 generated/vpa_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录")
    args = ap.parse_args()

    if args.watch:
        codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
        bad = [c for c in codes if not CODE_RE.match(c)]
        if bad:
            print(f"[错误] 非法代码: {','.join(bad)}（需6位数字）")
            sys.exit(1)
        if not codes:
            print("[错误] --watch 不能为空")
            sys.exit(1)
    else:
        codes = _load_watchlist()
        if not codes:
            print("[错误] config/watchlist.json 缺失/格式非法，且未提供 --watch")
            sys.exit(1)
        print(f"[提示] --watch 缺省，读取 {WATCHLIST_FILE}")

    system = VpaSystem(watch=codes, out_dir=Path(args.out_dir))
    det = system.detect(args.date)
    s = det["summary"]
    print(f"[量价筹码] {s['date']} | {s['total']} 只（可用 {s['ok']}，跳过 {s['failed']}）"
          f" | 信号 {s['signals']} | 耗时 {s['elapsed']}s")
    print()
    print(f"  {'代码':<6} {'名称':<8} {'威科夫':<6} {'量价':<6} {'密集区':<12} "
          f"{'主力':<6} {'信号':<4} 置信")
    for r in det["rows"]:
        if not r["ok"]:
            print(f"  {r['code']:<6} {r['name'][:8]:<8} {'跳过':<6} {r['error'][:40]}")
            continue
        print(f"  {r['code']:<6} {r['name'][:8]:<8} {r['stage']['label']:<6} "
              f"{r['price']['label']:<6} {r['zone']['label']:<12} "
              f"{r['fund']['state']:<6} {r['signal']:<4} {r['confidence']:.2f}")

    if not args.no_report:
        system.report(args.date)
        out = system.out_dir / f"vpa_report_{det['date']}.md"
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
