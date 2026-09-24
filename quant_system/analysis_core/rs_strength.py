#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rs_strength — 全A RS 相对强弱线复盘（Elder 体系）

方法（依据 skills/elder-technical-analysis 的 RS / 趋势强度方法论）：
  RS        = 个股收盘 / 基准收盘（同日期对齐）
              基准 = 沪深300（data_warehouse/market/index_daily.parquet，已验证与
              index_pe.parquet 的沪深300点位完全一致）；若该基准文件缺失，
              则回退用 kline 目录全部个股收盘等权合成基准。
  RS_MA20   = RS 的 20 日均线
  RS_slope  = RS_MA20 最近 5 个交易日的线性回归斜率，归一化为 % / 交易日
              （斜率 ÷ 当期 RS_MA20 × 100，便于跨股票横向比较）
  RS60日新高 = 当日 RS > 此前 60 个交易日 RS 最大值（需 ≥61 个对齐样本）
  评级规则：
    强 = RS_slope>0 且 RS > RS_MA20
    中 = 仅满足其一
    弱 = 均不满足

数据契约：
  data_warehouse/kline/{6位代码}.parquet  日K（只读 date/close 两列，按日期窗口裁剪）
  data_warehouse/market/index_daily.parquet  沪深300 日线
  data_warehouse/market/stock_names.parquet  代码-名称-ST 映射（缺失时回退
  quant_system/stock_name_map.json）

用法：
  python3 -m quant_system.analysis_core.rs_strength --report                 # 全A扫描 + 写报告
  python3 -m quant_system.analysis_core.rs_strength --watch 600519,000001    # 自选股池
  python3 -m quant_system.analysis_core.rs_strength --report --watch 600519,000001
  python3 -m quant_system.analysis_core.rs_strength --date 2026-08-07 --limit 200  # 指定日期/抽样
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # noqa: F401
except Exception:
    pq = None

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace（与 analysis_core.config 同约定）
KLINE_DIR = ROOT / "data_warehouse" / "kline"
MARKET_DIR = ROOT / "data_warehouse" / "market"
DEFAULT_OUT_DIR = ROOT / "generated"

CODE_RE = re.compile(r"^\d{6}$")

sys.path.insert(0, str(ROOT))
from quant_system.analysis_core.common import (  # noqa: E402
    kline_files,
    load_names,
    read_kline_window,
)
from quant_system.analysis_core.data_contract import format_pct_value  # noqa: E402



# 基准候选文件：优先 index_daily.parquet（沪深300），其余为可能的别名
BENCH_CANDIDATES = (
    "index_daily.parquet",
    "hs300.parquet",
    "000300.parquet",
    "sh000300.parquet",
    "a_equal_weight.parquet",
    "a_share_equal.parquet",
)

MIN_RS_WATCH = 21   # 自选股最少对齐点数（仅需 RS_MA20 = 20 点）
MIN_RS_RANK = 61    # TOP 排名最少对齐点数（20 MA + 60 日新高窗口）
LOOKBACK_DAYS = 170  # 读入日K的日历回看窗口（≈115 个交易日，覆盖 MA20 + 60日新高 + 余量）
PROGRESS_EVERY = 500


# ────────────────────────────────────────────────────────────
# 股票名称 / ST 标记
# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
# 基准
# ────────────────────────────────────────────────────────────
def load_benchmark() -> tuple[pd.Series | None, str]:
    """沪深300 收盘序列（date index -> close）。缺失时返回 None，由调用方回退等权合成。"""
    for fn in BENCH_CANDIDATES:
        p = MARKET_DIR / fn
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["date", "close"])
            if not {"date", "close"}.issubset(df.columns):
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            s = (df.dropna(subset=["date", "close"])
                   .drop_duplicates("date")
                   .set_index("date")["close"]
                   .sort_index())
            if len(s) >= 60:
                return s, f"market/{fn}（沪深300）"
        except Exception as e:
            logging.getLogger(__name__).error(f"[rs_strength] 操作失败: {e}", exc_info=True)
            continue
    return None, "未找到沪深300/全A等权基准文件，将回退等权合成"


def build_equal_weight_bench(files: list[Path], target: pd.Timestamp,
                             window_start: pd.Timestamp) -> tuple[pd.Series | None, str]:
    """全A等权合成基准：每个交易日对全部个股收盘取等权平均。"""
    frames = []
    for f in files:
        try:
            df = read_kline_window(f, ["date", "close"], pd.Timestamp(window_start))
        except Exception as e:
            logging.getLogger(__name__).error(f"[rs_strength] 操作失败: {e}", exc_info=True)
            continue
        if df is None or df.empty or not {"date", "close"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date")
        if not df.empty:
            frames.append(df)
    if not frames:
        return None, "等权合成失败：无可用K线"
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    s = all_df.groupby("date")["close"].mean().sort_index()
    s = s[s.index <= target]
    if len(s) < 60:
        return None, "等权合成失败：有效交易日过少"
    return s, "等权合成（kline 全A收盘均值）"


# ────────────────────────────────────────────────────────────
# RS 计算
# ────────────────────────────────────────────────────────────
def compute_rs_metrics(close: pd.Series, bench: pd.Series) -> dict:
    """纯计算：个股收盘序列 vs 基准序列 → RS 指标（与 scan 口径一致）。

    同日期对齐后 RS=个股/基准，返回 rs/rs_ma20/rs_slope/rating/rs_60d_high，
    供外部模块（watch_card）复用。样本不足时返回默认占位值，不抛异常。
    """
    base = {"rs": np.nan, "rs_ma20": np.nan, "rs_slope": np.nan,
            "rs_60d_high": False, "rating": "样本不足", "n_points": 0,
            "rs_slope_raw": np.nan}
    close = close.dropna()
    if len(close) < MIN_RS_WATCH:
        return base
    aligned = close.to_frame("close").join(bench.rename("bench"), how="inner")
    n = len(aligned)
    if n < MIN_RS_WATCH:
        return base
    rs = aligned["close"] / aligned["bench"]
    rsv = rs.to_numpy(dtype=float)
    mav = rs.rolling(20, min_periods=20).mean().to_numpy()
    rs_t = float(rsv[-1])
    ma20_t = float(mav[-1]) if n >= 20 and np.isfinite(mav[-1]) else float("nan")
    slope_raw = float("nan")
    slope_pct = float("nan")
    last5 = mav[-5:] if n >= 24 else np.array([])
    if len(last5) == 5 and np.all(np.isfinite(last5)) and np.isfinite(ma20_t) and ma20_t != 0:
        slope_raw = float(np.polyfit(np.arange(5), last5, 1)[0])
        slope_pct = slope_raw / ma20_t * 100.0
    new60 = bool(n >= MIN_RS_RANK and rs_t > float(np.max(rsv[-61:-1])))
    if slope_pct > 0 and rs_t > ma20_t:
        rating = "强"
    elif slope_pct > 0 or rs_t > ma20_t:
        rating = "中"
    else:
        rating = "弱"
    return {"rs": rs_t, "rs_ma20": ma20_t, "rs_slope": slope_pct,
            "rs_slope_raw": slope_raw, "rs_60d_high": new60,
            "rating": rating, "n_points": n}


# ────────────────────────────────────────────────────────────
# 主扫描
# ────────────────────────────────────────────────────────────
def scan(target: pd.Timestamp, limit: int | None = None) -> tuple[list[dict], dict]:
    """全A扫描：逐文件循环读（列裁剪 date/close + 列表缓存），宽表向量化算 RS。"""
    names, st_map = load_names()
    bench, bench_note = load_benchmark()
    bench_fallback = bench is None
    files = kline_files(limit)
    meta = {
        "files_total": len(files),
        "parsed": 0,
        "skip_read_error": 0,
        "skip_schema": 0,
        "skip_empty": 0,
        "skip_no_data": 0,
        "skip_short": 0,
        "st_in_universe": 0,
        "bj_in_universe": 0,
        "eligible": 0,
        "benchmark": bench_note,
    }
    if bench is None:
        bench, note = build_equal_weight_bench(files, target, target - pd.Timedelta(days=LOOKBACK_DAYS))
        if bench is None:
            return [], {**meta, "benchmark": note, "benchmark_available": False}
        meta["benchmark"] = note
    bench = bench[bench.index <= target]
    if len(bench) < MIN_RS_WATCH:
        return [], {**meta, "benchmark_available": False, "benchmark_error": "基准样本过少"}

    window_start = target - pd.Timedelta(days=LOOKBACK_DAYS)
    closes: list[pd.Series] = []
    t0 = time.time()

    for i, f in enumerate(files, 1):
        code = f.stem
        try:
            df = read_kline_window(f, ["date", "close"], pd.Timestamp(window_start))
        except Exception:
            meta["skip_read_error"] += 1
            continue
        if df is None or df.empty:
            meta["skip_empty"] += 1
            continue
        if not {"date", "close"}.issubset(df.columns):
            meta["skip_schema"] += 1
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        df = df[df["date"] <= target]
        if df.empty or not (df["date"] == target).any():
            meta["skip_no_data"] += 1
            continue
        meta["parsed"] += 1
        closes.append(df.set_index("date")["close"].rename(code))
        if i % PROGRESS_EVERY == 0:
            print(f"[进度] {i}/{len(files)} | 已读 {meta['parsed']} | 耗时 {time.time() - t0:.0f}s", flush=True)

    if not closes:
        meta["benchmark_available"] = True
        meta["benchmark_fallback"] = bench_fallback
        meta["elapsed_sec"] = round(time.time() - t0, 1)
        return [], meta

    wide = pd.concat(closes, axis=1)
    rs = wide.div(bench.reindex(wide.index), axis=0)

    rows: list[dict] = []
    for code in wide.columns:
        # 按个股自身交易日计算（dropna 剔除并集轴上的停牌/缺数日，与逐股算法一致）
        col = rs[code].dropna()
        rsv = col.to_numpy()
        mav = col.rolling(20, min_periods=20).mean().to_numpy()
        n = len(rsv)
        rs_t = float(rsv[-1])
        ma20_t = float(mav[-1]) if n >= 20 and np.isfinite(mav[-1]) else float("nan")

        slope_raw = float("nan")
        slope_pct = float("nan")
        last5 = mav[-5:] if n >= 24 else np.array([])
        if len(last5) == 5 and np.all(np.isfinite(last5)) and np.isfinite(ma20_t) and ma20_t != 0:
            slope_raw = float(np.polyfit(np.arange(5), last5, 1)[0])
            slope_pct = slope_raw / ma20_t * 100.0

        new60 = bool(n >= MIN_RS_RANK and rs_t > float(np.max(rsv[-61:-1])))

        if n < MIN_RS_WATCH:
            rating = "样本不足"
        elif slope_pct > 0 and rs_t > ma20_t:
            rating = "强"
        elif slope_pct > 0 or rs_t > ma20_t:
            rating = "中"
        else:
            rating = "弱"

        name = names.get(code, code)
        is_st = st_map.get(code, "ST" in name.upper())
        is_bj = code.startswith(("4", "8", "92"))
        if is_st:
            meta["st_in_universe"] += 1
        if is_bj:
            meta["bj_in_universe"] += 1

        rows.append({
            "code": code,
            "name": name,
            "close": float(wide[code].iloc[-1]),
            "rs": rs_t,
            "rs_ma20": ma20_t,
            "rs_slope": slope_pct,
            "rs_slope_raw": slope_raw,
            "rs_60d_high": new60,
            "rating": rating,
            "n_points": n,
        })
        if n >= MIN_RS_RANK and np.isfinite(slope_pct):
            meta["eligible"] += 1

    meta["benchmark_available"] = True
    meta["benchmark_fallback"] = bench_fallback
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    return rows, meta


def top_rows(rows: list[dict], top_n: int) -> list[dict]:
    """TOP30：RS_slope>0 且 RS 创60日新高 优先，组内按 RS_slope 降序。"""
    cands = [r for r in rows if r["n_points"] >= MIN_RS_RANK and np.isfinite(r["rs_slope"])]
    for r in cands:
        r["priority"] = bool(r["rs_slope"] > 0 and r["rs_60d_high"])
    cands.sort(key=lambda r: (0 if r["priority"] else 1, -r["rs_slope"]))
    return cands[:top_n]


def watch_rows(rows: list[dict], codes: list[str]) -> list[dict | None]:
    by_code = {r["code"]: r for r in rows}
    out = []
    for c in codes:
        r = by_code.get(c)
        if r is None:
            out.append({"code": c, "name": "(kline无数据/当日无交易)", "missing": True})
        else:
            out.append(r)
    return out


# ────────────────────────────────────────────────────────────
# 输出
# ────────────────────────────────────────────────────────────
def fmt_rs(v: float) -> str:
    return f"{v:.4f}" if np.isfinite(v) else "—"


def fmt_slope(v: float) -> str:
    return format_pct_value(v, digits=3, unit="pct")


def render_markdown(target: pd.Timestamp, rows: list[dict], meta: dict,
                    watch: list[dict | None], top_n: int) -> str:
    date = target.strftime("%Y-%m-%d")
    lines = [
        f"# RS 相对强弱线（Elder 体系）— {date}",
        "",
        f"- 基准: {meta['benchmark']}（{'可用' if meta.get('benchmark_available') else '不可用'}）",
        f"- 样本: kline {meta['files_total']} 只 → 目标日有数据 {meta['parsed']} | "
        f"有效RS(≥{MIN_RS_RANK}点) {meta['eligible']} | 不足样本 {meta['skip_short']} | "
        f"当日无数据 {meta['skip_no_data']}",
        f"- 样本内 ST {meta['st_in_universe']} 只 / 北交所 {meta['bj_in_universe']} 只（按规格含于全A池）",
        f"- 耗时: {meta.get('elapsed_sec', '?')}s",
        "",
        f"## 全A RS TOP{top_n}（RS_slope>0 且 RS创60日新高 优先）",
        "",
        "| # | 代码 | 名称 | 收盘 | RS | RS_MA20 | RS_slope(%/日) | RS60日新高 | 评级 |",
        "|---:|---:|:---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for i, r in enumerate(top_rows(rows, top_n), 1):
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['close']:.2f} | {fmt_rs(r['rs'])} | "
            f"{fmt_rs(r['rs_ma20'])} | {fmt_slope(r['rs_slope'])} | "
            f"{'是' if r['rs_60d_high'] else '否'} | {r['rating']} |"
        )
    lines += ["", "## 自选股池（--watch）", "", "| 代码 | 名称 | RS | RS_MA20 | RS_slope(%/日) | RS60日新高 | 评级 |", "|---:|---:|---:|---:|---:|:---:|:---:|"]
    for r in watch:
        if r.get("missing"):
            lines.append(f"| {r['code']} | {r['name']} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['code']} | {r['name']} | {fmt_rs(r['rs'])} | {fmt_rs(r['rs_ma20'])} | "
            f"{fmt_slope(r['rs_slope'])} | {'是' if r['rs_60d_high'] else '否'} | {r['rating']} |"
        )
    lines += ["", "---", "*方法: RS=个股收盘/基准收盘(同日期对齐); RS_MA20=RS的20日均线; "
                   "RS_slope=RS_MA20最近5日线性回归斜率(归一化%/交易日); "
                   "评级: 强=slope>0且RS>RS_MA20, 中=满足其一, 弱=均不满足。*"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="RS 相对强弱线（Elder 体系）复盘")
    ap.add_argument("--report", action="store_true", help="写入 generated/rs_strength_{date}.md")
    ap.add_argument("--watch", default="", help="自选股池，逗号分隔代码，如 600519,000001")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最近共同交易日）")
    ap.add_argument("--top", type=int, default=30, help="TOP N（默认30）")
    ap.add_argument("--limit", type=int, default=0, help="抽样处理前 N 个文件（开发验证用，0=全量）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认 workspace/generated）")
    args = ap.parse_args()

    # 目标日期：基准与kline样本的最新共同交易日
    bench, _ = load_benchmark()
    bench_max = bench.index.max() if bench is not None else None
    kline_max = None
    try:
        for f in kline_files(30):
            d = pd.read_parquet(f, columns=["date"])
            m = pd.to_datetime(d["date"], errors="coerce").max()
            kline_max = m if kline_max is None else max(kline_max, m)
    except Exception as e:
        logging.getLogger(__name__).error(f"[rs_strength] 操作失败: {e}", exc_info=True)
    target = min(x for x in (bench_max, kline_max) if x is not None)
    if args.date:
        target = pd.Timestamp(args.date)
    target = pd.Timestamp(target).normalize()

    print(f"[信息] 目标日期 {target.date()} | kline {len(kline_files(args.limit))} 只 | 基准as-of "
          f"{bench_max.date() if bench_max is not None else 'N/A'}", flush=True)
    t0 = time.time()
    rows, meta = scan(target, limit=args.limit or None)
    print(f"[完成] 解析 {meta['parsed']} | 有效RS {meta['eligible']} | 耗时 {meta['elapsed_sec']}s")

    top = top_rows(rows, args.top)
    print(f"\n[TOP{args.top}] 全A RS 排名（RS_slope>0 且 RS创60日新高 优先）:")
    for i, r in enumerate(top, 1):
        print(f"  {i:>2}. {r['code']} {r['name']:<10} 收{r['close']:>9.2f} RS {fmt_rs(r['rs'])} "
              f"MA20 {fmt_rs(r['rs_ma20'])} 斜率 {fmt_slope(r['rs_slope'])}/日 "
              f"60日新高{'是' if r['rs_60d_high'] else '否'} 评级{r['rating']}")

    watch_codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
    watch = watch_rows(rows, watch_codes) if watch_codes else []
    if watch:
        print("\n[自选股池]")
        for r in watch:
            if r.get("missing"):
                print(f"  {r['code']} {r['name']}")
                continue
            print(f"  {r['code']} {r['name']:<10} RS {fmt_rs(r['rs'])} MA20 {fmt_rs(r['rs_ma20'])} "
                  f"斜率 {fmt_slope(r['rs_slope'])}/日 60日新高{'是' if r['rs_60d_high'] else '否'} "
                  f"评级{r['rating']}")

    if args.report:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"rs_strength_{target:%Y-%m-%d}.md"
        out.write_text(render_markdown(target, rows, meta, watch, args.top), encoding="utf-8")
        print(f"\n已保存: {out}")

    print(f"\n[基准] {meta['benchmark']}（fallback={'是' if meta.get('benchmark_fallback') else '否'}）")


if __name__ == "__main__":
    main()
