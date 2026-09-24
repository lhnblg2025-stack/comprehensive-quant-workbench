#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""breakout_watch — 创N日新高扫描（Carter 突破确认 / Murphy 确认规则）

方法（依据 skills/master-trading-carter 与 murphy-technical-analysis-study-guide）：
  - 创N日新高：当日收盘 > 此前 N 个交易日收盘最大值（60/120/250 三档并列输出）
  - 放量突破：当日创新高（任一档）且 当日量 > 20日均量×1.5（20日均量=此前20日量均值）
  - 年线突破(待确认)：收盘 > MA250 且 距年线 < 3% 且 250日新高。
    Murphy 规则「突破需超过3%或2天收盘在另一侧」→ 距年线<3% 尚未达标，标记为待确认
  - 新高/下跌比 = 各档新高家数 ÷ 当日下跌家数（Elder new_high_low_ratio 情绪含义）：
    >1 多头宽度占优（情绪偏热），<1 宽度偏弱（追涨容错率低）

过滤（与规格一致）：
  - 剔除 ST/*ST（market/stock_names.parquet is_st，缺失时按名称含ST）
  - 剔除北交所（8/4/92 开头）
  - 剔除上市不足 250 个交易日的新股（以窗口内、target 之前交易日行数 < 250 判断）

数据契约：
  data_warehouse/kline/{6位代码}.parquet  日K（只读 date/open/high/low/close/volume/amount，
                                          按日期窗口裁剪）
  data_warehouse/market/stock_names.parquet  代码-名称-ST 映射

用法：
  python3 -m quant_system.analysis_core.breakout_watch --report
  python3 -m quant_system.analysis_core.breakout_watch --report --days 60,120,250 --top 50
  python3 -m quant_system.analysis_core.breakout_watch --date 2026-08-07 --limit 200  # 指定日期/抽样
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
SNAPSHOT_DIR = ROOT / "data_warehouse" / "realtime_snapshot"
DEFAULT_OUT_DIR = ROOT / "generated"

CODE_RE = re.compile(r"^\d{6}$")

sys.path.insert(0, str(ROOT))
from quant_system.analysis_core.common import (  # noqa: E402
    kline_files as _kline_files_common,
    load_names as _load_names_common,
    read_kline_window,
)

MIN_HIST_ROWS = 250          # 上市满 250 个交易日（窗口内交易日行数门槛）
LOOKBACK_DAYS = 400          # 日历回看窗口（≈275 个交易日，覆盖 250日新高 + MA250 + 余量）
VOL_MULT = 1.5               # 放量突破：当日量 > 20日均量 × 1.5
YEARLINE_GAP_PCT = 3.0       # 距年线 < 3% 视为待确认（Murphy: 需 >3% 或 2天收盘确认）
PROGRESS_EVERY = 500


# ────────────────────────────────────────────────────────────
# 股票名称 / ST 标记
# ────────────────────────────────────────────────────────────
def load_amount_map(target: pd.Timestamp) -> dict[str, float]:
    """目标日成交额真值：realtime_snapshot/{YYYYMMDD}/ 快照的 amount_wan（万元→元）。

    仅当快照目录日期 == 目标日时返回真值；否则返回空，由调用方回退估算
    （避免把非目标日的成交额当作真值展示/排序）。
    """
    if not SNAPSHOT_DIR.is_dir():
        return {}
    try:
        dirs = sorted(d.name for d in SNAPSHOT_DIR.iterdir()
                      if d.is_dir() and d.name.isdigit())
    except OSError:
        return {}
    if not dirs or target.strftime("%Y%m%d") not in dirs:
        return {}
    amount_map: dict[str, float] = {}
    frames = []
    try:
        for f in sorted((SNAPSHOT_DIR / target.strftime("%Y%m%d")).glob("*.parquet")):
            df = pd.read_parquet(f)
            if not {"code", "amount_wan"}.issubset(df.columns):
                continue
            df["code"] = df["code"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
            frames.append(df[["code", "amount_wan", "ts"]])
    except Exception:
        return {}
    if not frames:
        return {}
    all_df = pd.concat(frames, ignore_index=True)
    if "ts" in all_df.columns:
        all_df = all_df.sort_values("ts").drop_duplicates("code", keep="last")
    amount_map = dict(zip(all_df["code"], all_df["amount_wan"].astype(float) * 1e4))
    return amount_map




def kline_files(limit: int | None = None) -> list[Path]:
    """全A K线文件（6 位代码过滤）；limit 时确定性等距抽样（common 实现 + 本模块 KLINE_DIR）。"""
    return _kline_files_common(limit, kline_dir=KLINE_DIR)


def load_names() -> tuple[dict[str, str], dict[str, bool]]:
    """code -> (name, is_st)。优先 market/stock_names.parquet，回退 stock_name_map.json（common 实现 + 本模块 MARKET_DIR）。"""
    return _load_names_common(market_dir=MARKET_DIR)

# ────────────────────────────────────────────────────────────
# 主扫描
# ────────────────────────────────────────────────────────────
def scan(days: list[int], target: pd.Timestamp, limit: int | None = None) -> tuple[dict, dict]:
    names, st_map = load_names()
    amount_map = load_amount_map(target)
    files = kline_files(limit)
    window_start = target - pd.Timedelta(days=LOOKBACK_DAYS)

    meta = {
        "files_total": len(files),
        "parsed": 0,
        "skip_read_error": 0,
        "skip_schema": 0,
        "skip_empty": 0,
        "skip_no_data": 0,
        "skip_st": 0,
        "skip_bj": 0,
        "skip_new": 0,
        "universe": 0,
        "advancing": 0,
        "declining": 0,
        "flat": 0,
        "vol_breakout": 0,
        "yearline_breakout": 0,
    }
    new_high_cnt = {n: 0 for n in days}
    vol_break_rows: list[dict] = []
    yearline_rows: list[dict] = []
    list_rows = {n: [] for n in days}
    t0 = time.time()

    for i, f in enumerate(files, 1):
        code = f.stem
        try:
            df = read_kline_window(f, ["date", "open", "high", "low", "close", "volume", "amount"],
                                   pd.Timestamp(window_start))
        except Exception:
            meta["skip_read_error"] += 1
            continue
        if df is None or df.empty:
            meta["skip_empty"] += 1
            continue
        if not {"date", "close"}.issubset(df.columns):
            meta["skip_schema"] += 1
            continue
        for c in ("open", "high", "low", "volume", "amount"):
            if c not in df.columns:
                df[c] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
        df = df[df["date"] <= target]
        if df.empty or not (df["date"] == target).any():
            meta["skip_no_data"] += 1
            continue
        meta["parsed"] += 1

        # 过滤：ST / 北交所 / 上市<250交易日（窗口内交易日行数判断，避免 --date 历史场景
        # 把 target 之后的 kline 全文件行数计入而误判次新）
        name = names.get(code, code)
        is_st = st_map.get(code, "ST" in name.upper())
        is_bj = code.startswith(("4", "8", "92"))
        if is_st:
            meta["skip_st"] += 1
            continue
        if is_bj:
            meta["skip_bj"] += 1
            continue
        close = df["close"].astype(float)
        if len(close) < MIN_HIST_ROWS:
            meta["skip_new"] += 1
            continue
        meta["universe"] += 1
        prev_close = close.iloc[-2] if len(close) >= 2 else np.nan
        close_t = float(close.iloc[-1])
        pct_chg = (close_t / prev_close - 1.0) * 100.0 if np.isfinite(prev_close) and prev_close > 0 else np.nan
        if np.isfinite(prev_close):
            if close_t > prev_close:
                meta["advancing"] += 1
            elif close_t < prev_close:
                meta["declining"] += 1
            else:
                meta["flat"] += 1

        new_high = {n: False for n in days}
        for n in days:
            if len(close) >= n + 1:
                new_high[n] = bool(close_t > float(close.iloc[-(n + 1):-1].max()))
                if new_high[n]:
                    new_high_cnt[n] += 1

        # 量能确认：当日量 > 20日均量（此前20日均值）×1.5
        vol_ratio = np.nan
        vol = df["volume"].astype(float)
        if len(vol) >= 21 and vol.iloc[-1] == vol.iloc[-1]:
            ma20_vol = float(vol.iloc[-21:-1].mean())
            if ma20_vol > 0:
                vol_ratio = float(vol.iloc[-1]) / ma20_vol
        vol_break = bool(vol_ratio > VOL_MULT and any(new_high[n] for n in days))

        # 年线突破（待确认）：收盘>MA250 且 距年线<3% 且 250日新高
        # 固定绑定 250 档：--days 不含 250 时显式计算 250 日新高，避免错用 days[-1] 档
        ma250 = np.nan
        gap_pct = np.nan
        yearline = False
        if len(close) >= MIN_HIST_ROWS:
            ma250 = float(close.iloc[-MIN_HIST_ROWS:].mean())
            if ma250 > 0:
                gap_pct = (close_t / ma250 - 1.0) * 100.0
                if 250 in days:
                    nh250 = new_high[250]
                else:
                    nh250 = len(close) >= MIN_HIST_ROWS + 1 and bool(
                        close_t > float(close.iloc[-(MIN_HIST_ROWS + 1):-1].max()))
                yearline = bool(close_t > ma250 and 0 <= gap_pct < YEARLINE_GAP_PCT and nh250)

        # 成交额：优先 realtime_snapshot 目标日真值；缺失时按个股历史 amount/volume
        # 单位推断用 close×volume 估算（近端 kline amount 常缺失且 volume 单位按股票不一致）
        amount_raw = amount_map.get(code, np.nan)
        amount_est = False
        if not np.isfinite(amount_raw):
            vol_t = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else np.nan
            if np.isfinite(vol_t) and vol_t > 0:
                unit = 1.0
                hist = df.dropna(subset=["amount"])
                if len(hist) >= 10:
                    unit = float(np.median(hist["amount"] / (hist["close"] * hist["volume"])))
                    if not np.isfinite(unit) or unit <= 0:
                        unit = 1.0
                amount_raw = close_t * vol_t * unit
                amount_est = True
            else:
                amount_raw = 0.0
        row = {
            "code": code,
            "name": name,
            "close": close_t,
            "pct_chg": pct_chg,
            "amount_yi": amount_raw / 1e8,
            "amount_est": amount_est,
            "vol_ratio": vol_ratio,
            "vol_break": vol_break,
            "ma250": ma250,
            "gap_pct": gap_pct,
            "yearline": yearline,
            "new_high": new_high,
        }
        if vol_break:
            meta["vol_breakout"] += 1
            vol_break_rows.append(row)
        if yearline:
            meta["yearline_breakout"] += 1
            yearline_rows.append(row)
        for n in days:
            if new_high[n]:
                list_rows[n].append(row)

        if i % PROGRESS_EVERY == 0:
            print(f"[进度] {i}/{len(files)} | 合格 {meta['universe']} | 60日新高 {new_high_cnt.get(days[0])} "
                  f"| 耗时 {time.time() - t0:.0f}s", flush=True)

    meta["new_high_cnt"] = new_high_cnt
    # 新高/下跌家数比（Elder new_high_low_ratio 情绪含义）
    decl = meta["declining"]
    ratio = {}
    for n in days:
        ratio[n] = round(new_high_cnt[n] / decl, 3) if decl > 0 else None
    meta["new_high_low_ratio"] = ratio
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    meta["amount_source"] = "realtime_snapshot(目标日快照 amount_wan)" if amount_map else "kline估算(close×volume)"
    meta["amount_est_used"] = any(r.get("amount_est") for rows in list_rows.values() for r in rows)

    for n in days:
        list_rows[n].sort(key=lambda r: -r["amount_yi"])
    vol_break_rows.sort(key=lambda r: -r["amount_yi"])
    yearline_rows.sort(key=lambda r: -r["amount_yi"])
    return {"new_high_lists": list_rows, "vol_break_rows": vol_break_rows,
            "yearline_rows": yearline_rows}, meta


# ────────────────────────────────────────────────────────────
# 输出
# ────────────────────────────────────────────────────────────
def fmt_num(v: float) -> str:
    return f"{v:,.2f}" if np.isfinite(v) else "—"


def fmt_ratio(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "无下跌"


def render_markdown(target: pd.Timestamp, result: dict, meta: dict, days: list[int], top_n: int) -> str:
    date = target.strftime("%Y-%m-%d")
    lines = [
        f"# 创N日新高扫描（Carter/Murphy 突破确认）— {date}",
        "",
        f"- 合格样本（剔除 ST/北交所/上市<250交易日）: {meta['universe']} 只 | "
        f"剔除: ST {meta['skip_st']} / 北交所 {meta['skip_bj']} / 新股 {meta['skip_new']}",
        f"- 当日涨跌分布: 上涨 {meta['advancing']} | 下跌 {meta['declining']} | 平盘 {meta['flat']}",
        f"- 新高家数: " + " | ".join(f"{n}日 {meta['new_high_cnt'][n]}" for n in days),
        f"- 新高/下跌家数比（Elder new_high_low_ratio 情绪含义）: " + " | ".join(
            f"{n}日 {fmt_ratio(meta['new_high_low_ratio'][n])}" for n in days),
        f"- 放量突破（新高+量>20日均量×1.5）: {meta['vol_breakout']} 只 | "
        f"年线突破(距MA250<3%,待2日/3%确认): {meta['yearline_breakout']} 只",
        f"- 耗时: {meta['elapsed_sec']}s",
        ""]
    lines.append(f"- 成交额来源: {meta['amount_source']}"
                 + ("（少量个股缺失快照，按历史单位 close×volume 估算）" if meta.get("amount_est_used") else ""))
    lines += ["",
        "**情绪解读**：新高/下跌比 > 1 → 多头宽度占优（情绪偏热）；< 1 → 宽度偏弱，追涨容错率低。"
        "年线突破距年线 < 3% 属 Murphy「突破需 >3% 或 2 天收盘确认」的待确认状态。",
        "",
    ]
    for n in days:
        lines += [f"## {n}日新高 TOP{top_n}（按成交额降序）", "",
                  "| # | 代码 | 名称 | 收盘 | 涨跌% | 成交额(亿) | 放量突破 | 年线突破 | 距年线% |",
                  "|---:|---:|:---|---:|---:|---:|:---:|:---:|---:|"]
        rows = result["new_high_lists"][n][:top_n]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r['code']} | {r['name']} | {r['close']:.2f} | {fmt_num(r['pct_chg'])} | "
                f"{r['amount_yi']:.2f} | {'是' if r['vol_break'] else '否'} | "
                f"{'是' if r['yearline'] else '否'} | {fmt_num(r['gap_pct'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="创N日新高扫描（Carter/Murphy 突破确认）")
    ap.add_argument("--report", action="store_true", help="写入 generated/breakout_{date}.md")
    ap.add_argument("--days", default="60,120,250", help="新高窗口，逗号分隔（默认 60,120,250）")
    ap.add_argument("--top", type=int, default=50, help="每档 TOP N（默认50）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认最近交易日）")
    ap.add_argument("--limit", type=int, default=0, help="抽样处理前 N 个文件（开发验证用，0=全量）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认 workspace/generated）")
    args = ap.parse_args()

    try:
        days = sorted({int(x) for x in args.days.split(",") if x.strip()})
    except ValueError:
        print("[错误] --days 需为逗号分隔整数，如 60,120,250")
        sys.exit(2)
    days = [d for d in days if d >= 2]
    if not days:
        print("[错误] 至少需要一个 >=2 的新高窗口")
        sys.exit(2)

    # 目标日期：kline 抽样最新共同交易日
    target = None
    try:
        for f in kline_files(30):
            d = pd.read_parquet(f, columns=["date"])
            m = pd.to_datetime(d["date"], errors="coerce").max()
            target = m if target is None else max(target, m)
    except Exception as e:
        logging.getLogger(__name__).error(f"[breakout_watch] 操作失败: {e}", exc_info=True)
    if args.date:
        target = pd.Timestamp(args.date)
    if target is None:
        target = pd.Timestamp.now().normalize()
    target = pd.Timestamp(target).normalize()

    print(f"[信息] 目标日期 {target.date()} | 窗口 {days} | kline {len(kline_files(args.limit))} 只",
          flush=True)
    result, meta = scan(days, target, limit=args.limit or None)
    print(f"[完成] 合格 {meta['universe']} | 上涨 {meta['advancing']} 下跌 {meta['declining']} "
          f"| 新高 " + " ".join(f"{n}日:{meta['new_high_cnt'][n]}" for n in days)
          + f" | 放量突破 {meta['vol_breakout']} | 年线突破(待确认) {meta['yearline_breakout']} "
          f"| 耗时 {meta['elapsed_sec']}s")

    for n in days:
        rows = result["new_high_lists"][n][:args.top]
        print(f"\n[{n}日新高 TOP{args.top}] 按成交额降序（合计 {meta['new_high_cnt'][n]} 只）:")
        if not rows:
            print("  (无)")
        for i, r in enumerate(rows, 1):
            print(f"  {i:>2}. {r['code']} {r['name']:<10} 收{r['close']:>8.2f} "
                  f"涨{fmt_num(r['pct_chg']):>6}% 额{r['amount_yi']:>8.2f}亿 "
                  f"放量{'是' if r['vol_break'] else '否'} 年线{'是' if r['yearline'] else '否'}")

    if args.report:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"breakout_{target:%Y-%m-%d}.md"
        out.write_text(render_markdown(target, result, meta, days, args.top), encoding="utf-8")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
