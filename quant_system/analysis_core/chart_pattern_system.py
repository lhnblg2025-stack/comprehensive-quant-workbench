#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chart_pattern_system — 体系7 图表形态系统（规则引擎 + 斐波那契位 + 突破确认）

方法论（skills/chart-patterns-recognition + fibonacci-ratio-patterns）:
  1. 形态识别（规则引擎）: 用摆动点(pivot highs/lows)序列识别
     头肩顶/底、双顶/双底、上升/下降三角形、旗形
     → 输出 形态名 + 方向 + 颈线位 + 量能确认（右肩量缩/第二底缩量/突破放量）
  2. 斐波那契位: 识别最近主要摆动(60日高低) → 0.382/0.5/0.618/0.786 回撤位
     + 1.27/1.618 扩展位 → 当前价相对位置（支撑/阻力/突破区）
  3. 突破信号: 收盘破颈线(超 3%, Murphy 过滤) + 量比 > 1.5 → '形态突破'
  4. 综合: 形态方向 × 斐波那契支撑/阻力 × 密集区 → 多/空/震荡/防守

输入:
  data_warehouse/kline/{6位代码}.parquet  日K（date/open/high/low/close/volume）
  密集区: volume_profile.compute_volume_profile(近60日) 复用体系5口径

统一接口:
  from quant_system.analysis_core.chart_pattern_system import ChartPatternSystem
  cps = ChartPatternSystem()
  res = cps.detect(date=None, watch=["601899", "600519"])   # 全量检测
  path = cps.report(date=None, watch=["601899"])            # 写 generated/chart_report_{date}.md
  v = cps.view(date=None, watch=["601899", "600519"], code="601899")

防御:
  - 形态样本 < 60 日 → 标注 '形态样本<60日'，置信封顶
  - 单标的检测失败 → 跳过并计数，不阻塞其余标的

用法:
  python3 -m quant_system.analysis_core.chart_pattern_system --watch 601899,600519
  python3 -m quant_system.analysis_core.chart_pattern_system --watch 601899,600519 --report
  python3 -m quant_system.analysis_core.chart_pattern_system --watch 601899,600519 --view --date 2026-08-11
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

ROOT = Path(__file__).resolve().parent.parent.parent          # workspace（data_warehouse 所在）
REPO_ROOT = Path(__file__).resolve().parent.parent            # 仓库根（generated/ 输出目录）
KLINE_DIR = ROOT / "data_warehouse" / "kline"
MARKET_DIR = ROOT / "data_warehouse" / "market"
DEFAULT_OUT_DIR = REPO_ROOT / "generated"

CODE_RE = re.compile(r"^\d{6}$")

sys.path.insert(0, str(ROOT))
from quant_system.analysis_core.common import (  # noqa: E402
    kline_files,
    load_names,
    read_kline_window,
)

AGENT_NAME = "图表形态"
VIEW_WEIGHT = 1.0            # view() 默认权重（multi_agent 仲裁权重独立维护）

# ── 参数 ─────────────────────────────────────────────
LOOKBACK_DAYS = 300          # 日历回看窗口（≈200 交易日，覆盖形态+60日摆动）
MAX_PATTERN_BARS = 120       # 形态回看窗口（交易日）
MIN_PATTERN_BARS = 60        # 防御: 少于 60 交易日 → 形态标注样本不足
FIB_WINDOW = 60              # 斐波那契主要摆动窗口（交易日）
PIVOT_K_PATTERN = 2          # 形态摆动点 fractal 半径（±2 日）
PIVOT_K_FIB = 3              # 斐波那契主要摆动 fractal 半径（±3 日）
CONFIRM_PCT = 0.03           # Murphy 3% 收盘确认
VOL_RATIO_CONFIRM = 1.5      # 突破量比阈值
VOL_RATIO_FLOOR = 0.2
VOL_RATIO_CEIL = 8.0
VP_MIN_SAMPLES = 20          # 密集区最少样本

RAG_QUERY = "图表形态 头肩顶 突破 斐波那契"
RAG_K = 3

# 形态方向权重（综合评分用）
PATTERN_SCORE = {
    "头肩顶": 1.6, "头肩底": 1.6,
    "双顶": 1.5, "双底": 1.5,
    "上升三角形": 1.2, "下降三角形": 1.2,
    "旗形": 0.9,
}
BREAKOUT_CONFIRMED_BONUS = 0.8
BREAKOUT_PENDING_BONUS = 0.2

BULL_THRESHOLD = 1.2         # 综合分 ≥ → 多
BEAR_THRESHOLD = -1.2        # 综合分 ≤ → 空
DEFENSE_THRESHOLD = -2.0     # 强空 + 确认破位 → 防守
MAX_CONF = 0.92

FIB_RATIOS = {"0.382": 0.382, "0.5": 0.5, "0.618": 0.618, "0.786": 0.786}
FIB_EXTS = {"1.27": 1.27, "1.618": 1.618}


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def _norm_kline(df: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame | None:
    """K线列规整：date 转 datetime、去重、排序、截断至 target、补齐 OHLCV。"""
    if df is None or df.empty:
        return None
    df = df.copy()
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= target]
    if df.empty:
        return None
    # NaN OHLC 以收盘价回填，保证 pivot 计算稳定
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    return df.reset_index(drop=True)


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
            logging.getLogger(__name__).error(f"[chart_pattern_system] 操作失败: {e}", exc_info=True)
            continue
    return pd.Timestamp(min(cands)).normalize() if cands else pd.Timestamp.now().normalize()


# ────────────────────────────────────────────────────────────
# 摆动点 + 量能工具
# ────────────────────────────────────────────────────────────
def pivots(highs: np.ndarray, lows: np.ndarray, k: int = PIVOT_K_PATTERN) -> list[tuple[int, float, str]]:
    """fractal 摆动点序列 [(idx, price, 'H'|'L')]，同类型相邻取更极端值。"""
    n = len(highs)
    pts: list[tuple[int, float, str]] = []
    for i in range(k, n - k):
        h = float(highs[i])
        if all(float(highs[i - j]) <= h for j in range(1, k + 1)) and \
           all(float(highs[i + j]) <= h for j in range(1, k + 1)):
            pts.append((i, h, "H"))
        l = float(lows[i])
        if all(float(lows[i - j]) >= l for j in range(1, k + 1)) and \
           all(float(lows[i + j]) >= l for j in range(1, k + 1)):
            pts.append((i, l, "L"))
    pts.sort(key=lambda x: x[0])
    out: list[tuple[int, float, str]] = []
    for idx, price, typ in pts:
        if out and out[-1][2] == typ:
            p_i, p_price, _ = out[-1]
            if (typ == "H" and price >= p_price) or (typ == "L" and price <= p_price):
                out[-1] = (idx, price, typ)
        else:
            out.append((idx, price, typ))
    return out


def vol_ratio_last(df: pd.DataFrame, n: int = 20) -> float:
    """当日量比（抗股/手等单位切换）：最近 20 日一致单位中位量，跳变 >20 倍截断。

    与 vpa_system._last_vol_ratio 口径一致（n=20 / >20x 单位切换截断 / 最少3样本 /
    clip [0.2, 8]），此处仅取末点避免整列计算。
    """
    vols = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float)
    m = len(vols)
    if m < 4:
        return 1.0
    last = vols[-1]
    if not np.isfinite(last) or last <= 0:
        return 1.0
    tail: list[float] = []
    j = m - 2
    while j >= 0 and len(tail) < n:
        v = vols[j]
        if not np.isfinite(v) or v <= 0:
            j -= 1
            continue
        if tail and (v / tail[-1] > 20.0 or tail[-1] / v > 20.0):
            break
        tail.append(v)
        j -= 1
    if len(tail) < 3:
        return 1.0
    r = last / float(np.median(tail))
    return float(np.clip(r, VOL_RATIO_FLOOR, VOL_RATIO_CEIL))


def _vol_at(df: pd.DataFrame, idx: int) -> float:
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float)
    if not np.isfinite(v[idx]) or v[idx] <= 0:
        return float("nan")
    return float(v[idx])


def _median_vol(df: pd.DataFrame, lo: int, hi: int) -> float:
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float)[lo:hi]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else float("nan")


# ────────────────────────────────────────────────────────────
# 形态识别（规则引擎）
# ────────────────────────────────────────────────────────────
def _pattern_result(name: str, direction: str, neckline: float, note: str,
                    piv_idx: list[int]) -> dict:
    return {"name": name, "direction": direction, "neckline": float(neckline),
            "note": note, "pivots": piv_idx}


def _detect_head_shoulders(seq: list[tuple[int, float, str]], df: pd.DataFrame) -> list[dict]:
    """头肩顶/头肩底：H-L-H-L-H（顶）或 L-H-L-H-L（底），中间点最极端，两肩对称。"""
    out: list[dict] = []
    for i in range(len(seq) - 4):
        a, b, c, d, e = seq[i], seq[i + 1], seq[i + 2], seq[i + 3], seq[i + 4]
        if a[2] == c[2] == e[2] == "H" and b[2] == d[2] == "L":
            h1, h2, h3 = a[1], c[1], e[1]
            l1, l2 = b[1], d[1]
            if not (h2 > h1 and h2 > h3):
                continue
            if h1 / h2 < 0.82 or h3 / h2 < 0.82:
                continue                     # 肩部需接近头部高度
            if abs(h1 - h3) / max(h1, h3) > 0.12:
                continue                     # 两肩基本对称
            if abs(l1 - l2) / max(l1, l2) > 0.06:
                continue                     # 颈线接近水平
            if not (12 <= e[0] - a[0] <= MAX_PATTERN_BARS):
                continue
            neck = (l1 + l2) / 2.0
            v_head = _vol_at(df, c[0])
            v_rs = _vol_at(df, e[0])
            vol_note = "右肩量缩(动能衰竭)✅" if (np.isfinite(v_head) and np.isfinite(v_rs)
                                                    and v_rs < v_head * 0.95) else "右肩量能未确认"
            out.append(_pattern_result("头肩顶", "空", neck,
                                       f"{vol_note}；跌破颈线需收盘确认+放量",
                                       [a[0], b[0], c[0], d[0], e[0]]))
        elif a[2] == c[2] == e[2] == "L" and b[2] == d[2] == "H":
            l1, l2, l3 = a[1], c[1], e[1]
            h1, h2 = b[1], d[1]
            if not (l2 < l1 and l2 < l3):
                continue
            if l1 / l2 > 1.18 or l3 / l2 > 1.18:
                continue
            if abs(l1 - l3) / max(l1, l3) > 0.12:
                continue
            if abs(h1 - h2) / max(h1, h2) > 0.06:
                continue
            if not (12 <= e[0] - a[0] <= MAX_PATTERN_BARS):
                continue
            neck = (h1 + h2) / 2.0
            v_ls = _vol_at(df, a[0])
            v_head = _vol_at(df, c[0])
            v_rs = _vol_at(df, e[0])
            vol_note = "突破颈线需明显放量"
            if np.isfinite(v_ls) and np.isfinite(v_head) and np.isfinite(v_rs):
                vol_note += f"（左肩{v_ls:,.0f}/头{v_head:,.0f}/右肩{v_rs:,.0f}）"
            out.append(_pattern_result("头肩底", "多", neck, vol_note,
                                       [a[0], b[0], c[0], d[0], e[0]]))
    return out


def _detect_double(seq: list[tuple[int, float, str]], df: pd.DataFrame) -> list[dict]:
    """双顶/双底：L-H-L（底）/ H-L-H（顶），两端等高，中间点为触发颈线。"""
    out: list[dict] = []
    for i in range(len(seq) - 2):
        a, b, c = seq[i], seq[i + 1], seq[i + 2]
        if a[2] == c[2] == "L" and b[2] == "H":
            l1, l2 = a[1], c[1]
            neck = b[1]
            if abs(l1 - l2) / min(l1, l2) > 0.05:
                continue
            if neck <= max(l1, l2) * 1.04:
                continue
            if c[0] - a[0] < 10:
                continue
            v1, v2 = _vol_at(df, a[0]), _vol_at(df, c[0])
            vol_note = "第二底量能萎缩✅" if (np.isfinite(v1) and np.isfinite(v2)
                                              and v2 < v1 * 0.95) else "两底量能接近"
            out.append(_pattern_result("双底", "多", neck,
                                       f"{vol_note}；突破颈线需收盘确认+放量", [a[0], b[0], c[0]]))
        elif a[2] == c[2] == "H" and b[2] == "L":
            h1, h2 = a[1], c[1]
            neck = b[1]
            if abs(h1 - h2) / min(h1, h2) > 0.05:
                continue
            if neck >= min(h1, h2) * 0.96:
                continue
            if c[0] - a[0] < 10:
                continue
            v1, v2 = _vol_at(df, a[0]), _vol_at(df, c[0])
            vol_note = "第二顶量能萎缩✅" if (np.isfinite(v1) and np.isfinite(v2)
                                              and v2 < v1 * 0.95) else "两顶量能接近"
            out.append(_pattern_result("双顶", "空", neck,
                                       f"{vol_note}；跌破颈线需收盘确认+放量", [a[0], b[0], c[0]]))
    return out


def _detect_triangles(seq: list[tuple[int, float, str]]) -> list[dict]:
    """上升三角形（上边水平+下边抬高）/ 下降三角形（下边水平+上边降低）。"""
    out: list[dict] = []
    for i in range(len(seq) - 3):
        a, b, c, d = seq[i], seq[i + 1], seq[i + 2], seq[i + 3]
        if a[2] == c[2] == "H" and b[2] == d[2] == "L":
            if abs(a[1] - c[1]) / max(a[1], c[1]) > 0.025:
                continue                     # 上边（阻力）水平
            if d[1] <= b[1] * 1.004:
                continue                     # 下边（支撑）抬高
            if d[0] - a[0] < 8:
                continue
            out.append(_pattern_result("上升三角形", "多", max(a[1], c[1]),
                                       f"水平阻力 {max(a[1], c[1]):.2f} + 抬高下边 "
                                       f"({b[1]:.2f}→{d[1]:.2f})；突破上边看涨", [a[0], b[0], c[0], d[0]]))
        elif a[2] == c[2] == "L" and b[2] == d[2] == "H":
            if abs(a[1] - c[1]) / max(a[1], c[1]) > 0.025:
                continue                     # 下边（支撑）水平
            if d[1] >= b[1] * 0.996:
                continue                     # 上边（阻力）降低
            if d[0] - a[0] < 8:
                continue
            out.append(_pattern_result("下降三角形", "空", min(a[1], c[1]),
                                       f"水平支撑 {min(a[1], c[1]):.2f} + 降低上边 "
                                       f"({b[1]:.2f}→{d[1]:.2f})；跌破下边看空", [a[0], b[0], c[0], d[0]]))
    return out


def _detect_flag(df: pd.DataFrame) -> dict | None:
    """旗形：急涨/急跌（旗杆,≥7%）后逆势小幅整理（幅度<4%,≥3日），量能萎缩。"""
    hi = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    n = len(cl)
    best = None
    for pole_len in range(4, 9):
        for j in range(min(n - 4, n - 1), max(n - 25, 5), -1):
            s = j - pole_len
            if s < 0 or not np.isfinite(cl[j]) or not np.isfinite(cl[s]):
                continue
            move = (cl[j] / cl[s] - 1.0) * 100.0
            if abs(move) < 7.0:
                continue
            body_lo_i, body_hi_i = j + 1, n
            if body_hi_i - body_lo_i < 3:
                continue
            body = slice(body_lo_i, body_hi_i)
            if not np.all(np.isfinite(hi[body])) or not np.all(np.isfinite(lo[body])):
                continue
            amp = (hi[body].max() / lo[body].min() - 1.0) * 100.0
            if amp > 4.0:
                continue
            drift = (cl[-1] / cl[j] - 1.0) * 100.0
            direction = "多" if move > 0 else "空"
            if direction == "多" and drift > 2.0:
                continue
            if direction == "空" and drift < -2.0:
                continue
            pole_vol = _median_vol(df, s, j + 1)
            body_vol = _median_vol(df, j + 1, n)
            vol_note = "整理缩量✅" if (np.isfinite(pole_vol) and np.isfinite(body_vol)
                                        and body_vol < pole_vol) else "整理量能未确认"
            span = n - 1 - s
            if not (8 <= span <= 30):
                continue
            best = _pattern_result("旗形", direction, float(cl[j]),
                                   f"旗杆 {move:+.1f}% → 整理幅度 {amp:.1f}%；{vol_note}，"
                                   f"突破后量度移动=旗杆长度", [s, j])
            break
        if best:
            break
    return best


def detect_patterns(df: pd.DataFrame) -> tuple[list[dict], str]:
    """形态识别主入口（最近 MAX_PATTERN_BARS 交易日）。

    返回 (patterns, sample_note)；样本 < MIN_PATTERN_BARS 时标注 '形态样本<60日'。
    """
    n = len(df)
    if n < MIN_PATTERN_BARS:
        sample_note = f"形态样本<60日（仅{n}日）"
    else:
        sample_note = ""
    w = df.tail(min(n, MAX_PATTERN_BARS))
    hi = pd.to_numeric(w["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(w["low"], errors="coerce").to_numpy(float)
    seq = pivots(hi, lo, k=PIVOT_K_PATTERN)
    if len(seq) < 3:
        return [], sample_note
    patterns: list[dict] = []
    patterns.extend(_detect_head_shoulders(seq, w))
    patterns.extend(_detect_double(seq, w))
    patterns.extend(_detect_triangles(seq))
    flag = _detect_flag(w)
    if flag:
        patterns.append(flag)
    # 只保留形态完成于最近 60 日内的（避免陈旧形态）；pivot 索引相对窗口 w
    last_idx = len(w) - 1
    patterns = [p for p in patterns if last_idx - max(p["pivots"]) <= FIB_WINDOW]
    # 去重：同形态名+方向只保留完成时间最近的一个（避免滑动窗口重复检出）
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for p in sorted(patterns, key=lambda x: -max(x["pivots"])):
        key = (p["name"], p["direction"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique, sample_note


# ────────────────────────────────────────────────────────────
# 突破信号（Murphy 3% 收盘确认 + 量比）
# ────────────────────────────────────────────────────────────
def detect_breakouts(df: pd.DataFrame, patterns: list[dict], vratio: float) -> list[dict]:
    """收盘破颈线 >3%（Murphy）+ 量比 >1.5 → 形态突破；0.5%~3% 记待确认。"""
    close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    out: list[dict] = []
    for p in patterns:
        neck = float(p["neckline"])
        if neck <= 0 or not np.isfinite(neck):
            continue
        gap = (close / neck - 1.0) * 100.0
        if p["direction"] == "多":
            if gap > CONFIRM_PCT * 100:
                confirmed = vratio > VOL_RATIO_CONFIRM
                out.append({"name": f"{p['name']}突破", "direction": "多", "level": neck,
                            "gap_pct": round(gap, 2), "vol_ratio": round(vratio, 2),
                            "confirmed": confirmed,
                            "note": f"收盘超颈线 {gap:.1f}% 量比 {vratio:.2f}"
                                    f"（{'✅确认' if confirmed else '⚠放量不足'}，Murphy3%+量比>1.5）"})
            elif gap > 0.5:
                out.append({"name": f"{p['name']}突破", "direction": "多", "level": neck,
                            "gap_pct": round(gap, 2), "vol_ratio": round(vratio, 2),
                            "confirmed": False,
                            "note": f"收盘逼近颈线 +{gap:.1f}%（待 3% 确认）"})
        elif p["direction"] == "空":
            if gap < -CONFIRM_PCT * 100:
                confirmed = vratio > VOL_RATIO_CONFIRM
                out.append({"name": f"{p['name']}跌破", "direction": "空", "level": neck,
                            "gap_pct": round(gap, 2), "vol_ratio": round(vratio, 2),
                            "confirmed": confirmed,
                            "note": f"收盘破颈线 {gap:.1f}% 量比 {vratio:.2f}"
                                    f"（{'✅确认' if confirmed else '⚠放量不足'}，Murphy3%+量比>1.5）"})
            elif gap < -0.5:
                out.append({"name": f"{p['name']}跌破", "direction": "空", "level": neck,
                            "gap_pct": round(gap, 2), "vol_ratio": round(vratio, 2),
                            "confirmed": False,
                            "note": f"收盘逼近颈线 {gap:.1f}%（待 3% 确认）"})
    return out


# ────────────────────────────────────────────────────────────
# 斐波那契位（fibonacci-ratio-patterns）
# ────────────────────────────────────────────────────────────
def _major_swing(w: pd.DataFrame) -> dict | None:
    """最近主要摆动：最近 60 日内的最近一个主要高点/低点对。

    k=3 fractal 摆动；up = 低→高（回撤测支撑），down = 高→低（回撤测阻力）。
    """
    hi = pd.to_numeric(w["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(w["low"], errors="coerce").to_numpy(float)
    seq = pivots(hi, lo, k=PIVOT_K_FIB)
    last = seq[-1] if seq else None
    if last is not None:
        if last[2] == "H":
            lows_before = [p for p in seq if p[2] == "L" and p[0] < last[0]]
            if lows_before:
                a = lows_before[-1]
                if last[1] > a[1]:
                    return {"swing": "up", "a_idx": a[0], "b_idx": last[0],
                            "a": a[1], "b": last[1]}
        else:
            highs_before = [p for p in seq if p[2] == "H" and p[0] < last[0]]
            if highs_before:
                a = highs_before[-1]
                if a[1] > last[1]:
                    return {"swing": "down", "a_idx": a[0], "b_idx": last[0],
                            "a": a[1], "b": last[1]}
    # 兜底：窗口高低点（按发生顺序）
    hi_i = int(np.nanargmax(hi))
    lo_i = int(np.nanargmin(lo))
    if hi_i > lo_i:
        return {"swing": "up", "a_idx": lo_i, "b_idx": hi_i, "a": float(lo[lo_i]), "b": float(hi[hi_i])}
    return {"swing": "down", "a_idx": hi_i, "b_idx": lo_i, "a": float(hi[hi_i]), "b": float(lo[lo_i])}


def fib_analysis(df: pd.DataFrame, window: int = FIB_WINDOW) -> dict | None:
    """斐波那契位：主要摆动 → 回撤 0.382/0.5/0.618/0.786 + 扩展 1.27/1.618。

    返回 {swing, a, b, r382..r786, x127, x1618, close, position, bias, span_days}。
    bias>0 偏多、<0 偏空（供综合评分）。
    """
    if df is None or len(df) < VP_MIN_SAMPLES:
        return None
    w = df.tail(window)
    swing = _major_swing(w)
    if swing is None:
        return None
    a, b = swing["a"], swing["b"]
    rng = abs(b - a)
    if rng <= 0 or not np.isfinite(rng):
        return None
    close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    if swing["swing"] == "up":
        r = {f"r{k}": b - ratio * rng for k, ratio in FIB_RATIOS.items()}
        x = {f"x{k.replace('.', '')}": a + ratio * rng for k, ratio in FIB_EXTS.items()}
        # 当前价相对位置（回撤测支撑）
        if close >= x["x1618"]:
            pos, bias = "扩展1.618上方(强势突破)", 0.8
        elif close >= x["x127"]:
            pos, bias = "扩展1.27-1.618(强势区)", 0.6
        elif close >= b:
            pos, bias = "突破前高(回撤上方)", 0.6
        elif close >= r["r0.382"]:
            pos, bias = "回撤0-0.382(强势回撤)", 0.3
        elif close >= r["r0.5"]:
            pos, bias = "回撤0.382-0.5", 0.1
        elif close >= r["r0.618"]:
            pos, bias = "回撤0.5-0.618(黄金区)", -0.2
        elif close >= r["r0.786"]:
            pos, bias = "回撤0.618-0.786(深回撤)", -0.6
        else:
            pos, bias = "跌破0.786(趋势转弱)", -1.0
    else:
        r = {f"r{k}": b + ratio * rng for k, ratio in FIB_RATIOS.items()}
        x = {f"x{k.replace('.', '')}": b - ratio * rng for k, ratio in FIB_EXTS.items()}
        if close <= x["x1618"]:
            pos, bias = "扩展1.618下方(弱势破位)", -0.8
        elif close <= x["x127"]:
            pos, bias = "扩展1.27-1.618(弱势区)", -0.6
        elif close <= b:
            pos, bias = "跌破前低(回撤下方)", -0.6
        elif close <= r["r0.382"]:
            pos, bias = "回撤0-0.382(弱势反抽)", -0.3
        elif close <= r["r0.5"]:
            pos, bias = "回撤0.382-0.5", -0.1
        elif close <= r["r0.618"]:
            pos, bias = "回撤0.5-0.618(黄金区)", 0.2
        elif close <= r["r0.786"]:
            pos, bias = "回撤0.618-0.786(深回撤)", 0.6
        else:
            pos, bias = "突破0.786(趋势转强)", 1.0
    return {"swing": swing["swing"], "a": round(a, 2), "b": round(b, 2),
            "r382": round(r["r0.382"], 2), "r500": round(r["r0.5"], 2),
            "r618": round(r["r0.618"], 2), "r786": round(r["r0.786"], 2),
            "x127": round(x["x127"], 2), "x1618": round(x["x1618"], 2),
            "close": round(close, 2), "position": pos, "bias": bias,
            "span_days": int(swing["b_idx"] - swing["a_idx"])}


def volume_profile_note(code: str, target: pd.Timestamp) -> dict:
    """密集区（复用 volume_profile 体系5口径）：poc/上沿/下沿/位置。"""
    try:
        from quant_system.analysis_core.volume_profile import load_kline_60d, compute_volume_profile
        df60, err = load_kline_60d(code, target)
        if df60 is None or len(df60) < VP_MIN_SAMPLES:
            return {"ok": False, "note": f"密集区不可用({err})"}
        vp = compute_volume_profile(df60)
        if not vp.get("ok"):
            return {"ok": False, "note": f"密集区不可用({vp.get('note', '')})"}
        cur = vp.get("close")
        upper, lower = vp.get("upper"), vp.get("lower")
        label = vp.get("pos_label") or ""
        pos_txt = label
        if label == "上方" and cur is not None and upper:
            pos_txt = f"高于上沿 {(cur / upper - 1.0) * 100.0:.1f}%"
        elif label == "下方" and cur is not None and lower:
            pos_txt = f"低于下沿 {(lower / cur - 1.0) * 100.0:.1f}%"
        elif vp.get("pos_pct") is not None:
            pos_txt = f"{label} {vp['pos_pct']:.0f}%"
        return {"ok": True, "poc": vp.get("poc"), "upper": upper,
                "lower": lower, "pos_label": label,
                "pos_pct": vp.get("pos_pct"), "pos_txt": pos_txt}
    except Exception as e:
        return {"ok": False, "note": f"密集区不可用({str(e)[:60]})"}


# ────────────────────────────────────────────────────────────
# 综合评分 → 多/空/震荡/防守
# ────────────────────────────────────────────────────────────
def compose_view(row: dict) -> dict:
    """形态方向 × 斐波那契支撑/阻力 × 密集区 → 综合分 → 多/空/震荡/防守。"""
    score = 0.0
    # 综合评分只取最近完成的 3 个形态，按新旧加权（越新权重越大，避免箱体反复检出互相抵消）
    recency_w = (1.0, 0.6, 0.4)
    for rank, p in enumerate(sorted(row.get("patterns") or [],
                                    key=lambda x: -max(x["pivots"]))[:3]):
        wgt = PATTERN_SCORE.get(p["name"], 1.0) * recency_w[min(rank, len(recency_w) - 1)]
        score += wgt if p["direction"] == "多" else -wgt
    for b in row.get("breakouts") or []:
        bonus = BREAKOUT_CONFIRMED_BONUS if b["confirmed"] else BREAKOUT_PENDING_BONUS
        score += bonus if b["direction"] == "多" else -bonus
    fib = row.get("fib")
    if fib:
        score += fib.get("bias", 0.0)
    vp = row.get("vp") or {}
    vp_bias = 0.0
    if vp.get("ok"):
        label = vp.get("pos_label") or ""
        if label == "上方":
            vp_bias = 0.2
        elif label == "下方":
            vp_bias = -0.2
    score += vp_bias

    if score >= BULL_THRESHOLD:
        view_ = "多"
    elif score <= BEAR_THRESHOLD:
        bearish_break = any(b.get("confirmed") and b.get("direction") == "空"
                            for b in row.get("breakouts") or [])
        view_ = "防守" if score <= DEFENSE_THRESHOLD and bearish_break else "空"
    else:
        view_ = "震荡"

    conf = 0.35 + 0.12 * abs(score)
    if any(b.get("confirmed") for b in row.get("breakouts") or []):
        conf += 0.10
    if row.get("sample_note"):
        conf = min(conf, 0.40)
    if not row.get("fib"):
        conf -= 0.05
    conf = round(float(np.clip(conf, 0.15, MAX_CONF)), 2)
    return {"score": round(score, 2), "view": view_, "confidence": conf, "vp_bias": vp_bias}


def _symbol_evidence(code: str, name: str, row: dict) -> list[str]:
    ev = [f"{code} {name} 收{row.get('close'):.2f}（{row.get('date')}）样本{row.get('samples')}日"
          + (f" ⚠{row['sample_note']}" if row.get("sample_note") else "")]
    for p in row.get("patterns") or []:
        ev.append(f"形态[{p['name']}·{p['direction']}] 颈线 {p['neckline']:.2f} — {p['note']}")
    for b in row.get("breakouts") or []:
        ev.append(f"信号[{b['name']}·{b['direction']}] 收盘超颈线 {b['gap_pct']:+.1f}% "
                  f"量比 {b['vol_ratio']:.2f} → {'形态突破(确认)' if b['confirmed'] else '待确认'}")
    fib = row.get("fib")
    if fib:
        ev.append(f"斐波那契(主要摆动 {fib['swing']} {fib['a']:.2f}→{fib['b']:.2f}) "
                  f"回撤 382/500/618/786 = {fib['r382']:.2f}/{fib['r500']:.2f}/{fib['r618']:.2f}"
                  f"/{fib['r786']:.2f}；扩展 127/1618 = {fib['x127']:.2f}/{fib['x1618']:.2f}；"
                  f"当前价 → {fib['position']}")
    vp = row.get("vp") or {}
    if vp.get("ok"):
        ev.append(f"密集区 POC {vp['poc']:.2f} [{vp['lower']:.2f}, {vp['upper']:.2f}] → {vp.get('pos_txt', vp['pos_label'])}")
    elif vp.get("note"):
        ev.append(vp["note"])
    return ev


# ────────────────────────────────────────────────────────────
# 单标的检测
# ────────────────────────────────────────────────────────────
def detect_symbol(df: pd.DataFrame, code: str, name: str) -> dict:
    """单标的图表形态全量检测（形态/突破/斐波那契/密集区/综合）。"""
    close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    last_date = str(pd.to_datetime(df["date"]).iloc[-1].date())
    patterns, sample_note = detect_patterns(df)
    vratio = vol_ratio_last(df)
    breakouts = detect_breakouts(df, patterns, vratio)
    fib = fib_analysis(df)
    row = {"code": code, "name": name, "ok": True, "date": last_date,
           "close": close, "samples": len(df), "sample_note": sample_note,
           "patterns": patterns, "breakouts": breakouts, "fib": fib,
           "vp": volume_profile_note(code, pd.Timestamp(last_date)), "vol_ratio": round(vratio, 2)}
    comp = compose_view(row)
    row.update(comp)
    row["evidence"] = _symbol_evidence(code, name, row)
    return row


# ────────────────────────────────────────────────────────────
# 体系入口
# ────────────────────────────────────────────────────────────
class ChartPatternSystem:
    """图表形态系统：detect() 检测 / report() 写 md / view() 输出 multi_agent 兼容观点。"""

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self._names: dict[str, str] | None = None
        self._rag_cache: dict | None = None

    def _name_map(self) -> dict[str, str]:
        if self._names is None:
            self._names = load_names()[0]  # common.load_names 返回 (name_map, st_map)
        return self._names

    def _rag(self) -> dict:
        """RAG 方法论依据: knowledge_rag.search('图表形态 头肩顶 突破 斐波那契', k=3)。"""
        if self._rag_cache is not None:
            return self._rag_cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from quant_system.analysis_core import knowledge_rag
            hits = knowledge_rag.search(RAG_QUERY, k=RAG_K)
            if hits:
                self._rag_cache = {"query": RAG_QUERY, "available": True, "basis": "检索可用",
                                   "hits": [{"file": h.get("file"), "cat": h.get("cat", ""),
                                             "score": h.get("score"),
                                             "summary": (h.get("text") or "")[:200]} for h in hits]}
            else:
                self._rag_cache = {"query": RAG_QUERY, "available": False,
                                   "basis": "检索无命中", "hits": []}
        except Exception as e:
            self._rag_cache = {"query": RAG_QUERY, "available": False, "hits": [],
                               "basis": "检索不可用", "error": str(e)[:120]}
        return self._rag_cache

    def detect(self, date: str | None = None, watch: list[str] | None = None,
               limit: int | None = None) -> dict:
        """图表形态检测。watch 非空→自选股；否则全A（limit 抽样）。返回 {date, rows, meta, rag}。"""
        t0 = time.time()
        names = self._name_map()
        watch_codes = [str(c).strip().zfill(6) for c in (watch or []) if str(c).strip()]
        if watch_codes:
            files = [KLINE_DIR / f"{c}.parquet" for c in watch_codes]
            universe = "watch"
        else:
            files = kline_files(limit)
            universe = "fullA_sample" if limit else "fullA"

        target = resolve_target(watch_codes or [f.stem for f in files], date)
        window_start = target - pd.Timedelta(days=LOOKBACK_DAYS)

        meta = {"target_date": str(target.date()), "universe": universe,
                "files_total": len(files), "parsed": 0, "skip_read_error": 0,
                "skip_empty": 0, "skip_no_data": 0, "skip_failed": 0,
                "insufficient": 0, "elapsed_sec": 0.0}
        rows: list[dict] = []
        for f in files:
            code = f.stem
            name = names.get(code, code)
            if not f.exists():
                meta["skip_read_error"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": "kline文件缺失"})
                continue
            try:
                df = read_kline_window(f, ["date", "open", "high", "low", "close", "volume"],
                                       pd.Timestamp(window_start))
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
                row = detect_symbol(df, code, name)
                rows.append(row)
                meta["parsed"] += 1
                if row.get("sample_note"):
                    meta["insufficient"] += 1
            except Exception as e:
                meta["skip_failed"] += 1
                if watch_codes:
                    rows.append({"code": code, "name": name, "missing": True,
                                 "ok": False, "error": f"检测失败: {str(e)[:80]}"})
                continue
        meta["elapsed_sec"] = round(time.time() - t0, 2)
        return {"date": str(target.date()), "rows": rows, "meta": meta,
                "rag": self._rag()}

    def report(self, date: str | None = None, watch: list[str] | None = None,
               limit: int | None = None, out_dir: str | Path | None = None) -> Path:
        """检测并写 generated/chart_report_{date}.md，返回文件路径。"""
        res = self.detect(date=date, watch=watch, limit=limit)
        out = Path(out_dir) if out_dir else self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"chart_report_{res['date']}.md"
        path.write_text(render_markdown(res), encoding="utf-8")
        return path

    def view(self, date: str | None = None, watch: list[str] | None = None,
             code: str | None = None) -> dict:
        """multi_agent 兼容观点 {agent, signal, view, confidence, evidence, weight, status, detail}。

        code 指定 → 单标的观点；否则按 watch/样本多数票合成（多/空/震荡/防守）。
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
                return self._degraded_view(f"{code} 无有效图表形态结果", rag_basis=rag_basis,
                                           date=res["date"])
            ev = row.get("evidence") or []
            ev = ev + [f"RAG: {rag_basis}"]
            return {"agent": AGENT_NAME, "signal": row["view"], "view": row["view"],
                    "confidence": float(row.get("confidence", 0.4)),
                    "evidence": ev, "status": "ok", "weight": VIEW_WEIGHT,
                    "object_code": row["code"], "object_name": row["name"],
                    "detail": {"date": res["date"], "score": row.get("score"),
                               "patterns": row.get("patterns", []),
                               "breakouts": row.get("breakouts", []),
                               "fib": row.get("fib"), "vp": row.get("vp"),
                               "sample_note": row.get("sample_note")},
                    "rag_basis": rag_basis}

        ok_rows = [r for r in rows if r.get("ok")]
        if not ok_rows:
            return self._degraded_view("无可用图表形态样本", rag_basis=rag_basis, date=res["date"])
        votes: dict[str, float] = {}
        for r in ok_rows:
            v = r.get("view", "震荡")
            votes[v] = votes.get(v, 0) + float(r.get("confidence", 0.4))
        signal = max(votes, key=votes.get)
        n_bull = sum(1 for r in ok_rows if r.get("view") == "多")
        n_bear = sum(1 for r in ok_rows if r.get("view") in ("空", "防守"))
        n_neutral = len(ok_rows) - n_bull - n_bear
        conf = round(min(MAX_CONF, 0.35 + 0.2 * (votes[signal] / sum(votes.values()))), 2)
        ev = [f"图表形态扫描 {len(ok_rows)} 只: 多 {n_bull} / 空+防守 {n_bear} / 震荡 {n_neutral} "
              f"→ 加权多数票 {signal}（置信 {conf:.0%}）", f"RAG: {rag_basis}"]
        top = sorted(ok_rows, key=lambda r: -float(r.get("confidence", 0)))[:8]
        for r in top:
            ps = "、".join(f"{p['name']}{p['direction']}" for p in (r.get("patterns") or [])[:3]) or "无"
            ev.append(f"  {r['code']} {r['name']}: {r.get('view')} 置信{r.get('confidence'):.0%} [{ps}]")
        detail = {"date": res["date"], "universe": res["meta"].get("universe"),
                  "n_symbols": len(ok_rows), "bull": n_bull, "bear": n_bear,
                  "neutral": n_neutral, "votes": votes,
                  "insufficient": res["meta"].get("insufficient", 0)}
        return {"agent": AGENT_NAME, "signal": signal, "view": signal,
                "confidence": conf, "evidence": ev, "status": "ok",
                "weight": VIEW_WEIGHT, "detail": detail, "rag_basis": rag_basis}

    def _degraded_view(self, msg: str, rag_basis: str = "检索不可用",
                       date: str | None = None) -> dict:
        return {"agent": AGENT_NAME, "signal": "震荡", "view": "震荡", "confidence": 0.0,
                "evidence": [msg, f"RAG: {rag_basis}"], "status": "degraded",
                "weight": VIEW_WEIGHT, "detail": {"date": date or ""}, "rag_basis": rag_basis}


# ────────────────────────────────────────────────────────────
# Markdown 报告
# ────────────────────────────────────────────────────────────
def _fmt_levels(fib: dict) -> str:
    if not fib:
        return "—"
    if fib["swing"] == "up":
        return (f"回撤(自高点) 382={fib['r382']:.2f} 500={fib['r500']:.2f} "
                f"618={fib['r618']:.2f} 786={fib['r786']:.2f}；扩展 127={fib['x127']:.2f} "
                f"1618={fib['x1618']:.2f}")
    return (f"回撤(自低点) 382={fib['r382']:.2f} 500={fib['r500']:.2f} "
            f"618={fib['r618']:.2f} 786={fib['r786']:.2f}；扩展 127={fib['x127']:.2f} "
            f"1618={fib['x1618']:.2f}")


def render_markdown(res: dict) -> str:
    rows = res["rows"]
    meta = res["meta"]
    ok_rows = [r for r in rows if r.get("ok")]
    rag = res.get("rag") or {}
    lines = [f"# 图表形态报告 — {res['date']}", "",
             f"- 扫描: {meta.get('universe', '')} {meta.get('files_total', 0)} 只 | "
             f"有效 {meta.get('parsed', 0)} | 样本不足 {meta.get('insufficient', 0)} | "
             f"失败 {meta.get('skip_failed', 0)} | 耗时 {meta.get('elapsed_sec', 0)}s", "",
             "- 方法: 摆动点规则引擎（头肩/双顶底/三角/旗形）→ 颈线 + 量能确认；"
             "斐波那契 0.382/0.5/0.618/0.786 回撤 + 1.27/1.618 扩展；"
             "突破=收盘破颈线>3%(Murphy)+量比>1.5；综合=形态×斐波那契×密集区。", "",
             f"- RAG: {rag.get('basis', '检索不可用')} ({rag.get('query', '')})"]
    for h in rag.get("hits", [])[:RAG_K]:
        lines.append(f"  - [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')}")
    if ok_rows:
        lines += ["", "## 标的全览", "",
                  "| 代码 | 名称 | 观点 | 置信 | 形态 | 突破信号 | 斐波那契位置 |", "|---|---|---|---|---|---|---|"]
        for r in ok_rows:
            ps = "、".join(f"{p['name']}({p['direction']})" for p in (r.get("patterns") or [])[:3]) or "无"
            bs = "、".join(f"{b['name']}{'✅' if b['confirmed'] else '⏳'}"
                           for b in (r.get("breakouts") or [])[:3]) or "无"
            fib_pos = (r.get("fib") or {}).get("position", "—")
            lines.append(f"| {r['code']} | {r['name']} | **{r.get('view', '震荡')}** | "
                         f"{r.get('confidence', 0):.0%} | {ps} | {bs} | {fib_pos} |")
    lines += ["", "## 明细", ""]
    for r in ok_rows:
        lines.append(f"### {r['code']} {r['name']} — **{r.get('view', '震荡')}** "
                     f"置信 {r.get('confidence', 0):.0%}（综合分 {r.get('score', 0):+.2f}）")
        lines.append("")
        for e in r.get("evidence") or []:
            lines.append(f"- {e}")
        if r.get("fib"):
            lines += ["", f"- 斐波那契位: {_fmt_levels(r['fib'])}"]
        lines.append("")
    for r in rows:
        if not r.get("ok"):
            lines.append(f"### {r['code']} {r['name']} — 跳过（{r.get('error', '')}）")
        lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="图表形态系统（chart-patterns-recognition + fibonacci-ratio-patterns）")
    ap.add_argument("--watch", default="", help="自选股池，逗号分隔代码，如 601899,600519")
    ap.add_argument("--limit", type=int, default=0, help="全A抽样前 N 个文件（0=全量；无 --watch 时生效）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最近共同交易日）")
    ap.add_argument("--report", action="store_true", help="写入 generated/chart_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    ap.add_argument("--view", action="store_true", help="输出 multi_agent 兼容 view JSON")
    args = ap.parse_args()

    cps = ChartPatternSystem(out_dir=args.out_dir)
    watch_codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
    res = cps.detect(date=args.date, watch=watch_codes or None, limit=args.limit or None)
    meta = res["meta"]
    print(f"[信息] 目标日期 {res['date']} | {meta['universe']} {meta['files_total']} 只 | "
          f"有效 {meta['parsed']} | 样本不足 {meta['insufficient']} | 失败 {meta['skip_failed']} | "
          f"耗时 {meta['elapsed_sec']}s", flush=True)

    print("\n[图表形态] 每标的 观点/置信/形态/突破/斐波那契:")
    for r in res["rows"]:
        if not r.get("ok"):
            print(f"  {r['code']} {r['name']:<8} 跳过（{r.get('error', '')}）")
            continue
        ps = "、".join(f"{p['name']}{p['direction']}" for p in r.get("patterns") or []) or "无"
        bs = "、".join(f"{b['name']}{'✅' if b['confirmed'] else '⏳'}"
                       for b in r.get("breakouts") or []) or "无"
        fib_pos = (r.get("fib") or {}).get("position", "—")
        print(f"  {r['code']} {r['name']:<8} 收{r.get('close'):.2f} → {r.get('view')}"
              f"(置信{r.get('confidence'):.0%}, 分{r.get('score'):+.2f}) "
              f"形态[{ps}] 突破[{bs}] fib[{fib_pos}]"
              + (f" ⚠{r['sample_note']}" if r.get("sample_note") else ""))

    rag = res.get("rag") or {}
    print(f"\n[RAG] {rag.get('query', '')} → {rag.get('basis', '检索不可用')}")
    for h in rag.get("hits", [])[:RAG_K]:
        print(f"  [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')}")

    if args.report:
        path = cps.report(date=args.date, watch=watch_codes or None,
                          limit=args.limit or None, out_dir=args.out_dir)
        print(f"\n已保存: {path}")

    if args.view:
        v = cps.view(date=args.date, watch=watch_codes or None)
        print(f"\n[VIEW] {json.dumps(v, ensure_ascii=False, indent=2, default=str)}")


if __name__ == "__main__":
    main()
