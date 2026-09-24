#!/usr/bin/env python3
"""update_margin_detail.py — 两融个股明细日更(云端每日任务)

akshare stock_margin_detail_sse/szse 抓当日融资融券明细
→ 写 data_warehouse/market/margin_detail_{sh,sz}/{YYYYMMDD}.parquet,
供 stock_lens 中线「融资盘」维度使用。

用法:
  python3 scripts/update_margin_detail.py [--date 2026-08-14]
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


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def update(date: str | None = None, backfill_days: int = 0) -> int:
    """抓取两融明细。两融数据 T+1 发布，默认回退尝试最近 5 个自然日；
    --backfill N 时从指定日期起往前补 N 天缺口（用于一次性回填）。"""
    import akshare as ak
    total = 0
    found_existing = 0  # 已存在的数据日期数（数据已最新时也视为成功）
    if date:
        dates = [date]
    else:
        from datetime import timedelta
        today = datetime.now(CST)
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
        if backfill_days:
            # 回填：从 backfill_days 天前到今天逐日尝试（已存在文件自动跳过）
            dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d")
                     for i in range(backfill_days, -1, -1)]
    done: set[str] = set()
    for date in dates:
        d_compact = date.replace("-", "")
        frames = []
        for market, fn in (("sh", ak.stock_margin_detail_sse),
                           ("sz", ak.stock_margin_detail_szse)):
            out = MARKET_DIR / f"margin_detail_{market}" / f"{d_compact}.parquet"
            if out.exists():
                found_existing += 1
                continue
            try:
                df = fn(date=d_compact)
            except Exception as e:  # noqa: BLE001
                print(f"[margin] {market} {date} 抓取失败: {str(e)[:70]}", flush=True)
                continue
            if df is None or df.empty:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            # V12.3 审计 P2-3: 写盘走 tmp+os.replace 原子替换, 防半程被杀留下半截
            tmp = out.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            frames.append((market, len(df)))
            print(f"[margin] {market} {date} 写入 {len(df)} 行 → {out.name}", flush=True)
        if frames:
            total += sum(n for _, n in frames)
            done.add(date)
            if not backfill_days:
                break  # 默认模式：抓到最近一个有数据的日期即可
    return total if total else found_existing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="两融明细日更")
    ap.add_argument("--date", default=None, help="日期 YYYY-MM-DD")
    ap.add_argument("--backfill", type=int, default=0, help="回填 N 天缺口（从 N 天前起补）")
    args = ap.parse_args()
    n = update(args.date, backfill_days=args.backfill)
    print(f"[margin] 完成: {n} 行", flush=True)
    raise SystemExit(0 if n else 1)
