#!/usr/bin/env python3
"""
update_lhb_daily.py — 龙虎榜日更（2026-08-14 审计P0修复）
======================================================
审计发现: 龙虎榜数据本机/云端均停 2026-08-10, 无任何更新任务。
本脚本交易日 18:00 拉取当日龙虎榜明细(akshare stock_lhb_detail_em),
按季度段滚动落盘 lhb_YYYYMMDD_YYYYMMDD.parquet（与既有文件命名一致）。

用法:
  python3 scripts/update_lhb_daily.py            # 拉当日(交易日自动跳过非交易日)
  python3 scripts/update_lhb_daily.py --date 2026-08-13
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
MARKET_DIR = ROOT / "data_warehouse" / "market"
LHB_HISTORY_DIR = ROOT / "data_warehouse" / "lhb_hist"


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _quarter_file(date_str: str) -> Path:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    q = (d.month - 1) // 3 + 1
    start = f"{d.year}{q*3-2:02d}01"
    end = f"{d.year}{q*3:02d}31"
    return MARKET_DIR / f"lhb_{start}_{end}.parquet"


def update(date: str | None = None) -> int:
    date = date or _today()
    try:
        import akshare as ak
        # akshare stock_lhb_detail_em 参数为 YYYYMMDD 无横线格式
        d_compact = date.replace("-", "")
        df = ak.stock_lhb_detail_em(start_date=d_compact, end_date=d_compact)
    except Exception as e:
        print(f"[lhb] 抓取失败(可能非交易日): {e}", flush=True)
        return 0
    if df is None or df.empty:
        print(f"[lhb] {date} 无龙虎榜数据(非交易日?)", flush=True)
        return 0
    # 列名统一: 上榜日期→上榜日
    if "上榜日期" in df.columns:
        df = df.rename(columns={"上榜日期": "上榜日"})
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    out = _quarter_file(date)
    old = None
    if out.exists():
        try:
            old = pd.read_parquet(out)
        except Exception:  # noqa: BLE001
            old = None
    merged = pd.concat([old, df], ignore_index=True) if old is not None else df
    merged = merged.drop_duplicates(subset=["代码", "上榜日"], keep="last")
    tmp = out.with_suffix(".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(out)
    # Keep the canonical quarterly store and daily historical partition aligned.
    LHB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history = LHB_HISTORY_DIR / f"lhb_{date.replace('-', '')}.parquet"
    history_tmp = history.with_suffix(".tmp")
    df.to_parquet(history_tmp, index=False)
    history_tmp.replace(history)
    print(f"[lhb] {date} 新增 {len(df)} 行 → {out.name} / {history.name}", flush=True)
    return len(df)


def update_seats(date: str | None = None, top_n: int = 30) -> int:
    """抓当日龙虎榜个股席位明细(买5/卖5 双榜), 累积 lhb_stock_detail_daily.parquet,
    供 broker_gaming.stock_gaming / stock_lens 龙虎榜席位深度使用。"""
    try:
        from quant_system.analysis_core.broker_gaming import fetch_stock_detail
        d = (date or _today()).replace("-", "")
        n = fetch_stock_detail(d, top_n=top_n)
        print(f"[lhb] {date or _today()} 席位明细: {n} 只入库", flush=True)
        return n
    except Exception as e:
        print(f"[lhb] 席位明细抓取失败: {e}", flush=True)
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="龙虎榜日更")
    ap.add_argument("--date", default=None, help="日期 YYYY-MM-DD（默认今天）")
    ap.add_argument("--seat", action="store_true", help="同时抓个股席位明细(买5/卖5)")
    args = ap.parse_args()
    target = args.date or _today()
    n = update(args.date)
    if args.seat:
        update_seats(args.date)
    # V12.3 审计 P2-4: 退出口径——非交易日(=周末, 精确日历不可用/异常时按周末)且确无
    # LHB 数据 → 0(正常); 交易日却无数据(网络/接口抽风或异常) → 非 0, 让 schtasks
    # 的 Last Result 能暴露失败, 而非恒 0 掩盖。注: 若 akshare 抓取本身抛异常,
    # update() 内已返回 0 且打日志; 这里用交易日历区分"今天该不该有数据"。
    try:
        from quant_system.market_clock import is_trading_day
        trading_day = is_trading_day(target)
    except Exception:  # noqa: BLE001 - 日历不可用则退化为周末粗筛
        d = datetime.strptime(target, "%Y-%m-%d")
        trading_day = d.weekday() < 5
    if trading_day and n == 0:
        # 交易日却无龙虎榜数据 → 视为失败(非交易日空返回在前已 0)
        raise SystemExit(1)
    raise SystemExit(0)
