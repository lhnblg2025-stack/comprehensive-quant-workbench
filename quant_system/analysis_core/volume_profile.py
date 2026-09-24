#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""volume_profile — 交易密集区（简化 Volume Profile）

方法（skills/mainline-volume-capital-flow-trading 的"支撑/承接"思想落地为量化口径）:
  1. 取个股最近 60 个交易日 OHLCV（缺列回退，样本 < 20 日 → "样本不足"）。
  2. 以 0.5% 步长做几何价格网格（lo = 60日最低, hi = 60日最高, 格宽 = 1.005 倍）。
  3. 每个交易日把 [low, high] 区间覆盖到的格子按等分累加当日成交量
     （日线无逐笔/分时量，无法精确到价位，等分是日线口径的标准简化）。
  4. POC = 累计量最大的格子价格（几何中心）；密集区上沿/下沿 = 从 POC 格向两侧
     扩展、量 > POC 量 70% 的连续格带边界。
  5. 当前价相对位置: 当前收盘相对密集区带 [下沿, 上沿] 的百分比位置与上/中/下标注。

数据契约:
  data_warehouse/kline/{6位代码}.parquet  日K（date/open/high/low/close/volume）

用法:
  python3 -m quant_system.analysis_core.volume_profile --watch 600519
  python3 -m quant_system.analysis_core.volume_profile --watch 600519,000001 --date 2026-08-07
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.common import (  # noqa: E402
    fmt,
    load_names,
    read_kline_window,
)

KLINE_DIR = ROOT / "data_warehouse" / "kline"
CODE_RE = re.compile(r"^\d{6}$")

VP_DAYS = 60            # 回看交易日数
MIN_SAMPLES = 20        # 样本不足阈值
STEP_PCT = 0.005        # 0.5% 步长（几何网格）
POC_RATIO = 0.70        # 密集区阈值：量 > POC量 × 70%


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def load_kline_60d(code: str, target: pd.Timestamp | None = None) -> tuple[pd.DataFrame | None, str]:
    """最近 60 个交易日的日K（date/open/high/low/close/volume），截断至 target。"""
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "kline文件缺失"
    window = (pd.Timestamp(target) if target is not None else pd.Timestamp.now())
    window = window.normalize()
    try:
        df = read_kline_window(path, ["date", "open", "high", "low", "close", "volume"],
                               window - pd.Timedelta(days=VP_DAYS * 3))
    except Exception as e:
        return None, f"读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "无数据"
    if not {"date", "close"}.issubset(df.columns):
        return None, "缺关键列"
    for c in ("open", "high", "low", "volume"):
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= window]
    if df.empty:
        return None, "目标日无交易"
    return df.tail(VP_DAYS).reset_index(drop=True), ""


# ────────────────────────────────────────────────────────────
# 简化 Volume Profile
# ────────────────────────────────────────────────────────────
def _vp_insufficient(n: int, note: str) -> dict:
    return {"ok": False, "error": note, "note": note, "days": n, "n": n,
            "poc": None, "upper": None, "lower": None, "close": None,
            "pos": None, "pos_label": "—", "pos_pct": None, "peak_share": None}


def fmt_pos(vp: dict) -> str:
    """密集区位置文本：上方→'高于上沿 X%'（(cur/upper-1)*100），
    下方→'低于下沿 X%'（(lower/cur-1)*100）；区间内/贴POC 保留原百分比位置。"""
    label = vp.get("pos_label") or "—"
    cur = vp.get("close")
    if label == "上方":
        upper = vp.get("upper")
        if (cur is not None and upper is not None
                and np.isfinite(cur) and np.isfinite(upper) and upper > 0):
            return f"高于上沿 {(cur / upper - 1.0) * 100.0:.1f}%"
        return label
    if label == "下方":
        lower = vp.get("lower")
        if (cur is not None and lower is not None
                and np.isfinite(cur) and np.isfinite(lower) and cur > 0):
            return f"低于下沿 {(lower / cur - 1.0) * 100.0:.1f}%"
        return label
    pos_pct = vp.get("pos_pct")
    if pos_pct is not None and np.isfinite(pos_pct):
        return f"{label} {pos_pct:.0f}%"
    return label


def compute_volume_profile(df: pd.DataFrame) -> dict:
    """60 日简化 Volume Profile。

    返回 dict: ok/error/note/days(=n)/poc/upper/lower/close/pos/pos_label/pos_pct/
    peak_share。poc 在样本不足时为 None（watch_card 契约：poc None → 样本不足）。
    pos = 当前价在密集区带内的位置（小数，>1 表示上方、<0 表示下方）。
    """
    if df is None or len(df) < MIN_SAMPLES:
        n = len(df) if df is not None else 0
        return _vp_insufficient(n, "样本不足" if n < MIN_SAMPLES else "无数据")
    close = pd.to_numeric(df["close"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    rows = pd.concat([close, low, high, vol], axis=1).dropna()
    if len(rows) < MIN_SAMPLES:
        return _vp_insufficient(len(rows), "样本不足")
    rows = rows.tail(VP_DAYS)
    lo = float(rows["low"].min())
    hi = float(rows["high"].max())
    cur = float(rows["close"].iloc[-1])
    days = len(rows)

    if not (lo > 0 and hi > lo):
        return {"ok": True, "error": "", "note": "", "days": days, "n": days,
                "poc": cur, "upper": cur, "lower": cur, "close": cur,
                "pos": 1.0, "pos_label": "贴POC", "pos_pct": 100.0, "peak_share": 100.0}

    n_cells = max(1, int(np.ceil(np.log(hi / lo) / np.log(1.0 + STEP_PCT))))
    edges = lo * (1.0 + STEP_PCT) ** np.arange(n_cells + 1)
    vols = np.zeros(n_cells, dtype=float)

    for _, r in rows.iterrows():
        d_low, d_high, d_vol = float(r["low"]), float(r["high"]), float(r["volume"])
        if not np.isfinite(d_vol) or d_vol <= 0 or not (d_high >= d_low):
            continue
        i0 = max(0, int(np.searchsorted(edges, d_low, side="right")) - 1)
        i1 = min(n_cells - 1, int(np.searchsorted(edges, d_high, side="left")) - 1)
        if i1 < i0:
            i1 = i0
        vols[i0:i1 + 1] += d_vol / float(i1 - i0 + 1)

    peak_i = int(np.argmax(vols))
    poc = float(np.sqrt(edges[peak_i] * edges[peak_i + 1]))
    total = float(vols.sum())
    peak_share = float(vols[peak_i] / total * 100.0) if total > 0 else np.nan

    thresh = POC_RATIO * float(vols[peak_i])
    left = peak_i
    while left - 1 >= 0 and vols[left - 1] >= thresh:
        left -= 1
    right = peak_i
    while right + 1 < n_cells and vols[right + 1] >= thresh:
        right += 1
    lower = float(edges[left])
    upper = float(edges[right + 1])

    if upper > lower:
        pos_pct = (cur - lower) / (upper - lower) * 100.0
        if cur > upper:
            pos_label = "上方"
        elif cur < lower:
            pos_label = "下方"
        else:
            pos_label = "区间内"
    else:
        pos_pct = 100.0
        pos_label = "贴POC"

    return {"ok": True, "error": "", "note": "", "days": days, "n": days,
            "poc": poc, "upper": upper, "lower": lower, "close": cur,
            "pos": pos_pct / 100.0, "pos_label": pos_label,
            "pos_pct": pos_pct, "peak_share": peak_share}


# watch_card 兼容别名（契约: poc=None→样本不足, pos=0~1 位置小数, n=样本数）
calc_volume_profile = compute_volume_profile


def render_row(code: str, name: str, vp: dict) -> str:
    if not vp.get("ok"):
        return f"  {code:<6} {name:<8} 日期— | {vp.get('error', '无数据')}"
    return (f"  {code:<6} {name:<8} {vp['days']:>3}日 | POC {fmt(vp['poc']):>10} | "
            f"上沿 {fmt(vp['upper']):>10} | 下沿 {fmt(vp['lower']):>10} | "
            f"当前价 {fmt(vp['close']):>10} | {fmt_pos(vp)} | 峰格占量 {fmt(vp['peak_share'], 1)}%")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="交易密集区（60日简化 Volume Profile）")
    ap.add_argument("--watch", required=True, help="股票代码，逗号分隔，如 600519,000001")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认 kline 最新）")
    args = ap.parse_args()

    codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
    bad = [c for c in codes if not CODE_RE.match(c)]
    if bad:
        print(f"[错误] 非法代码: {','.join(bad)}（需6位数字）")
        sys.exit(1)
    if not codes:
        print("[错误] --watch 不能为空")
        sys.exit(1)

    target = pd.Timestamp(args.date).normalize() if args.date else None
    names, _ = load_names()
    print(f"[交易密集区] 口径: 最近{VP_DAYS}日 × 0.5%步长几何网格，日内量等分到覆盖格，"
          f"密集带=量>POC量×{int(POC_RATIO * 100)}%")
    for code in codes:
        df, err = load_kline_60d(code, target)
        if df is None:
            print(f"  {code:<6} {names.get(code, ''):<8} {err}")
            continue
        vp = compute_volume_profile(df)
        print(render_row(code, names.get(code, code), vp))


if __name__ == "__main__":
    main()
