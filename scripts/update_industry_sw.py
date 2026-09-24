#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_industry_sw.py — 申万行业分级 + 行情快照（行业分析数据源）
===================================================================
2026-08-11 新增（用户"中证/申万行业分级"方向）:
  - 申万一级 31 行业 / 二级 131 行业（含估值/股息/成分数）
  - 申万一级成分股（index_component_sw）
  - 申万二级实时行情（index_realtime_sw）
  - 申万一级历史 K 线（index_hist_sw，增量合并）

输出（data_warehouse/industry/）:
  - sw_first.parquet      一级行业信息（代码/名称/成分数/PE/PB/股息率）
  - sw_second.parquet     二级行业信息（含上级一级）
  - sw_first_cons.parquet 一级行业成分股（code/name/权重）
  - sw_second_spot.parquet 二级行业实时行情快照（含涨跌幅/成交额）
  - sw_first_hist.parquet 一级行业历史 K 线（增量）
  - industry_meta.json    抓取时间/新鲜度标注

用法:
  python3 scripts/update_industry_sw.py            # 全量（信息+成分+快照+历史）
  python3 scripts/update_industry_sw.py --only info,cons,spot,hist
  python3 scripts/update_industry_sw.py --no-hist  # 跳过历史K线（快）
"""
from __future__ import annotations
import logging

import argparse
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[update_industry_sw] 操作失败: {e}", exc_info=True)

ROOT = Path(__file__).resolve().parent.parent
IND_DIR = ROOT / "data_warehouse" / "industry"
IND_DIR.mkdir(parents=True, exist_ok=True)
META = IND_DIR / "industry_meta.json"

FIRST_FILE = IND_DIR / "sw_first.parquet"
SECOND_FILE = IND_DIR / "sw_second.parquet"
FIRST_CONS_FILE = IND_DIR / "sw_first_cons.parquet"
SECOND_SPOT_FILE = IND_DIR / "sw_second_spot.parquet"
FIRST_HIST_FILE = IND_DIR / "sw_first_hist.parquet"


def load_meta() -> dict:
    if META.exists():
        try:
            return json.loads(META.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_meta(m: dict) -> None:
    m["_generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    META.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_first() -> pd.DataFrame:
    """申万一级行业信息。"""
    import akshare as ak
    df = ak.sw_index_first_info()
    if df is None or df.empty:
        raise RuntimeError("sw_index_first_info 空")
    return df


def fetch_second() -> pd.DataFrame:
    """申万二级行业信息。"""
    import akshare as ak
    df = ak.sw_index_second_info()
    if df is None or df.empty:
        raise RuntimeError("sw_index_second_info 空")
    return df


def fetch_first_cons(first_codes: list[str]) -> pd.DataFrame:
    """申万一级成分股（全部一级行业）。"""
    import akshare as ak
    rows = []
    fails = []
    for i, code in enumerate(first_codes, 1):
        try:
            df = ak.index_component_sw(symbol=code)
            if df is not None and not df.empty:
                df = df.copy()
                df["行业代码"] = code
                rows.append(df)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{code}:{type(e).__name__}")
        if i % 10 == 0:
            print(f"  [cons] {i}/{len(first_codes)} 累计 {sum(len(r) for r in rows)} 行", flush=True)
        time.sleep(0.2)
    if fails:
        print(f"  ⚠️ 成分失败 {len(fails)}: {fails[:5]}", flush=True)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fetch_second_spot() -> pd.DataFrame:
    """申万二级行业实时行情快照。"""
    import akshare as ak
    df = ak.index_realtime_sw(symbol="二级行业")
    if df is None or df.empty:
        raise RuntimeError("index_realtime_sw 空")
    return df


def fetch_first_hist(first_codes: list[str], force_all: bool = False) -> pd.DataFrame:
    """申万一级历史 K 线（增量合并；force_all 全量重建）。"""
    import akshare as ak
    old = None
    if FIRST_HIST_FILE.exists() and not force_all:
        try:
            old = pd.read_parquet(FIRST_HIST_FILE)
        except Exception:
            old = None
    rows = []
    fails = []
    for i, code in enumerate(first_codes, 1):
        try:
            df = ak.index_hist_sw(symbol=code, period="day")
            if df is not None and not df.empty:
                df = df.copy()
                df["代码"] = str(df["代码"].iloc[0]) if "代码" in df else code
                rows.append(df)
        except Exception as e:  # noqa: BLE001
            fails.append(f"{code}:{type(e).__name__}")
        if i % 10 == 0:
            print(f"  [hist] {i}/{len(first_codes)} 累计 {sum(len(r) for r in rows)} 行", flush=True)
        time.sleep(0.3)
    if fails:
        print(f"  ⚠️ 历史失败 {len(fails)}: {fails[:5]}", flush=True)
    if not rows:
        return pd.DataFrame()
    new = pd.concat(rows, ignore_index=True)
    if old is not None and len(old):
        cols = [c for c in old.columns if c in new.columns]
        merged = pd.concat([old[cols], new[cols]], ignore_index=True)
        merged = merged.drop_duplicates(subset=["代码", "日期"], keep="last")
        return merged.sort_values(["代码", "日期"]).reset_index(drop=True)
    return new.sort_values(["代码", "日期"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔: info,cons,spot,hist")
    ap.add_argument("--no-hist", action="store_true", help="跳过历史K线")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    meta = load_meta()
    today = date.today().isoformat()

    if not only or "info" in only:
        print("=== 申万一级/二级行业信息 ===", flush=True)
        first = fetch_first()
        first.to_parquet(FIRST_FILE)
        print(f"✅ 一级行业 {len(first)} 个 → {FIRST_FILE.name}", flush=True)
        second = fetch_second()
        second.to_parquet(SECOND_FILE)
        print(f"✅ 二级行业 {len(second)} 个 → {SECOND_FILE.name}", flush=True)
        meta["first_date"] = today
        meta["first_count"] = len(first)
        meta["second_count"] = len(second)
        save_meta(meta)
    else:
        first = pd.read_parquet(FIRST_FILE) if FIRST_FILE.exists() else fetch_first()
        second = pd.read_parquet(SECOND_FILE) if SECOND_FILE.exists() else fetch_second()

    first_codes = [str(c).split(".")[0] for c in first["行业代码"]]
    name_map = dict(zip(first["行业代码"].astype(str).str.split(".").str[0], first["行业名称"]))

    if not only or "cons" in only:
        print("=== 申万一级成分股 ===", flush=True)
        cons = fetch_first_cons(first_codes)
        if not cons.empty:
            cons.to_parquet(FIRST_CONS_FILE)
            print(f"✅ 一级成分 {len(cons)} 行 → {FIRST_CONS_FILE.name}", flush=True)
            meta["cons_date"] = today
            meta["cons_rows"] = len(cons)
            save_meta(meta)

    if not only or "spot" in only:
        print("=== 申万二级实时行情 ===", flush=True)
        try:
            spot = fetch_second_spot()
            spot.to_parquet(SECOND_SPOT_FILE)
            print(f"✅ 二级行情 {len(spot)} 行 → {SECOND_SPOT_FILE.name}", flush=True)
            meta["spot_date"] = today
            save_meta(meta)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 二级行情失败: {e}", flush=True)

    if (not only or "hist" in only) and not args.no_hist:
        print("=== 申万一级历史 K 线（增量）===", flush=True)
        hist = fetch_first_hist(first_codes)
        if not hist.empty:
            hist.to_parquet(FIRST_HIST_FILE)
            print(f"✅ 一级历史 {len(hist)} 行 → {FIRST_HIST_FILE.name}", flush=True)
            meta["hist_date"] = today
            meta["hist_rows"] = len(hist)
            save_meta(meta)

    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
