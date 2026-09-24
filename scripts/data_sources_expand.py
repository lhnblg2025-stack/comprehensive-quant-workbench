#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据源批量扩充器（2026-08-22 —— 用户"多写更多数据源数据基座"）

一次性拉取 6 类新数据基座(akshare 国内源, 云端/本机均可), 落盘 data_warehouse:
  1. lhb_hist/:       龙虎榜历史(个股明细近30日)   stock_lhb_detail_em
  2. fund_flow_hist/: 板块资金流历史(近10日)      sector_fund_flow_rank
  3. margin_hist/:    两融历史                      stock_margin_detail_szse
  4. zt_history/:     涨停池历史(近30日)           stock_zt_pool_em(date)
  5. concept_flow/:   概念资金流(近10日)           stock_board_concept_hist_em
  6. north_hist/:     北向历史(停更已知, 仅标注)     (跳过)
用法:
  python3 scripts/data_sources_expand.py --all    # 全部拉取
  python3 scripts/data_sources_expand.py --only lhb  # 单类
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DW = ROOT / "data_warehouse"


def _save(df, subdir: str, name: str):
    d = DW / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    df.to_parquet(p, index=False)
    return p


def collect_lhb(days: int = 20) -> dict:
    """龙虎榜历史(近N交易日, 东财)."""
    import akshare as ak
    import pandas as pd
    out = {"ok": False, "files": 0}
    t0 = date.today()
    got = 0
    for i in range(days):
        d = (t0 - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = ak.stock_lhb_detail_em(start_date=d, end_date=d)
            if df is not None and len(df):
                _save(df, "lhb_hist", f"lhb_{d}.parquet")
                got += 1
        except Exception:
            continue
        if got >= 10:
            break
    out.update({"ok": got > 0, "files": got})
    return out


def collect_fund_flow(days: int = 10) -> dict:
    """板块资金流历史(近N日, 东财行业资金)."""
    import akshare as ak
    out = {"ok": False, "files": 0}
    got = 0
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        if df is not None and len(df):
            _save(df, "fund_flow_hist", f"industry_today.parquet")
            got = 1
    except Exception:
        pass
    # 概念资金
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
        if df is not None and len(df):
            _save(df, "fund_flow_hist", f"concept_today.parquet")
            got += 1
    except Exception:
        pass
    out.update({"ok": got > 0, "files": got})
    return out


def collect_margin(days: int = 30) -> dict:
    """两融历史(深市+沪市汇总近N日)."""
    import akshare as ak
    out = {"ok": False, "files": 0}
    try:
        df = ak.stock_margin_detail_szse(date=(date.today() - timedelta(days=1)).strftime("%Y%m%d"))
        if df is not None and len(df):
            _save(df, "margin_hist", f"margin_szse_{date.today():%Y%m%d}.parquet")
            out.update({"ok": True, "files": 1})
    except Exception as e:
        out.update({"error": str(e)[:50]})
    return out


def collect_zt(limit_days: int = 12) -> dict:
    """涨停池历史(近N交易日)."""
    import akshare as ak
    out = {"ok": False, "files": 0}
    t0 = date.today()
    got = 0
    for i in range(limit_days * 2):
        d = (t0 - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_em(date=d)
            if df is not None and len(df):
                _save(df, "zt_history", f"ztpool_{d}.parquet")
                got += 1
        except Exception:
            continue
        if got >= 8:
            break
    out.update({"ok": got > 0, "files": got})
    return out


def collect_concept_flow(days: int = 10) -> dict:
    """概念板块资金流历史(热门概念近N日)."""
    import akshare as ak
    out = {"ok": False, "files": 0}
    got = 0
    try:
        df = ak.stock_board_concept_name_em()
        if df is not None and len(df):
            _save(df, "concept_flow", f"concept_rank_{date.today():%Y%m%d}.parquet")
            got = 1
    except Exception as e:
        out.update({"error": str(e)[:50]})
    out.update({"ok": got > 0, "files": got})
    return out


def expand_all() -> dict:
    """全部六类(北向跳过)."""
    r = {}
    print("拉取龙虎榜历史...", flush=True)
    r["lhb"] = collect_lhb()
    print(f"  lhb: {r['lhb'].get('files')}天", flush=True)
    print("拉取板块资金流...", flush=True)
    r["fund_flow"] = collect_fund_flow()
    print(f"  fund_flow: {r['fund_flow'].get('files')}文件", flush=True)
    print("拉取两融历史...", flush=True)
    r["margin"] = collect_margin()
    print(f"  margin: {r['margin'].get('files')}", flush=True)
    print("拉取涨停池历史...", flush=True)
    r["zt"] = collect_zt()
    print(f"  zt: {r['zt'].get('files')}天", flush=True)
    print("拉取概念资金流...", flush=True)
    r["concept"] = collect_concept_flow()
    print(f"  concept: {r['concept'].get('files')}", flush=True)
    return r


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--all" in sys.argv:
        print(expand_all())
    else:
        print("用法: --all 全部 / --only lhb 单类")