#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件域刷新（2026-08-21 —— 修复 events 基座停更 08-04）

akshare 东财源拉大宗交易(A股 含折溢率) + 更新 events/block.parquet。
与原表 schema 兼容(序号/交易日期/证券代码/证券简称/成交价/成交量/成交额/营业部)，
新增列(涨跌幅/收盘价/折溢率) 前端/决策可直接用。

用法:
  python3 scripts/refresh_events.py [--days 3]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data_warehouse" / "events"
CST = timezone(timedelta(hours=8))


def refresh_block(days: int = 3) -> dict:
    """拉近 N 日大宗交易 → 覆盖写入 events/block.parquet。"""
    import pandas as pd
    try:
        import akshare as ak
    except ImportError:
        return {"ok": False, "note": "akshare 缺失"}
    today = datetime.now(CST)
    start = (today - timedelta(days=days)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    try:
        df = ak.stock_dzjy_mrmx(symbol="A股", start_date=start, end_date=end)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "note": f"akshare 拉取失败: {str(e)[:60]}"}
    if df is None or not len(df):
        return {"ok": False, "note": "无大宗数据"}
    # 统一列名: 交易日期/成交额 等
    df = df.rename(columns={"交易日期": "交易日期"})
    EVENTS.mkdir(parents=True, exist_ok=True)
    out = EVENTS / "block.parquet"
    df.to_parquet(out, index=False)
    latest = str(df["交易日期"].max())[:10]
    amt = float(df["成交额"].sum() or 0) / 1e8
    dis = float(df["折溢率"].mean() or 0) if "折溢率" in df.columns else None
    return {"ok": True, "note": f"大宗更新 {latest}: {len(df)}笔 {amt:.1f}亿"
            + (f" 均折溢率{dis:+.2f}%" if dis is not None else "")}


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()
    r = refresh_block(args.days)
    print(r["note"])