#!/usr/bin/env python3
"""全市场截面 Feature Store 生成器（数据资产化 V2-2，2026-08-07）

把 data_warehouse 的个股级 parquet（K线/估值）聚合成按交易日的截面特征文件：
    data_warehouse/feature_store/{YYYYMMDD}.parquet

首批特征组（全部来自仓库，零现场抓数）：
  1. 估值类：peTTM / pbMRQ / psTTM / pcfNcfTTM 及 252 日分位
  2. 动量类：20/60/120 日收益、距 52 周高点、20 日振幅
  3. 量价类：换手率(缺则 NaN)、成交额(缺则 NaN)、20 日均量比

消费方：因子生产 / IC 分析 / 组合构建 / 日报 / 前端特征库视图 统一从 feature_store 读截面，
不再各自临时计算。用 DataStore 门面读取（V2-1 统一 Data Access Layer）。

用法:
    python3 scripts/build_feature_store.py --date 20260807          # 指定交易日
    python3 scripts/build_feature_store.py --latest                 # 仓库最新共同交易日
    python3 scripts/build_feature_store.py --update                 # 滚动更新：最新共同交易日，已存在则跳过（幂等）
    python3 scripts/build_feature_store.py --latest --limit 200     # 测试限200只
"""
from __future__ import annotations
import logging

import argparse
import os
import sys
import time
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

os.nice(10)  # 降低优先级（用户硬规则）


def _latest_common_trade_date(ds) -> str:
    """从 K线/估值仓库取最近共同交易日（取两者最新日期的较小者）。"""
    ws = ds.warehouse_root()
    kline_dates = []
    for f in sorted((ws / "kline").glob("*.parquet"))[:200]:
        try:
            df = pd.read_parquet(f, columns=["date"])
            kline_dates.append(str(df["date"].max())[:10])
        except Exception as e:
            logging.getLogger(__name__).error(f"[build_feature_store] 操作失败: {e}", exc_info=True)
            continue
    val_dates = []
    for f in sorted((ws / "valuation").glob("*.parquet"))[:200]:
        try:
            df = pd.read_parquet(f, columns=["date"])
            val_dates.append(str(df["date"].max())[:10])
        except Exception as e:
            logging.getLogger(__name__).error(f"[build_feature_store] 操作失败: {e}", exc_info=True)
            continue
    if not kline_dates or not val_dates:
        raise RuntimeError("K线/估值仓库为空，无法确定交易日")
    return min(max(kline_dates), max(val_dates)).replace("-", "")


def _build_single(ds, code: str, tdate: str, val_cache: dict | None = None, snap_df: pd.DataFrame | None = None) -> dict | None:
    """单只股票在 tdate 的截面特征（全部从仓库读）。"""
    try:
        # 2026-08-10 审计：特征窗口起点原写死 20250101，改为 tdate 前 2 年动态计算
        _start = (_date(int(tdate[:4]), int(tdate[4:6]), int(tdate[6:]))
                  - pd.Timedelta(days=730)).strftime("%Y%m%d")
        k = ds.get(code, start=_start, adjust=None)
    except Exception:
        try:
            k = pd.read_parquet(ROOT / "data_warehouse" / "kline" / f"{code}.parquet")
            k["date"] = pd.to_datetime(k["date"])
        except Exception:
            return None
    if k is None or k.empty:
        return None
    k = k.sort_values("date")
    # 估值缓存：首次按 code 读取，整个运行期复用
    if val_cache is None:
        val_cache = {}
    v = val_cache.get(code)
    if v is None:
        try:
            v = pd.read_parquet(ROOT / "data_warehouse" / "valuation" / f"{code}.parquet")
            v["date"] = pd.to_datetime(v["date"])
        except Exception:
            v = None
        val_cache[code] = v

    # 该交易日行
    t = pd.Timestamp(tdate)
    row = k[k["date"] == t]
    if row.empty:
        return None
    r = row.iloc[-1]
    close = float(r["close"]) if pd.notna(r["close"]) else np.nan

    # ── 涨跌幅：优先 pct_chg，缺失用 close 前收计算 ──
    pct_chg = np.nan
    if "pct_chg" in k.columns and pd.notna(r.get("pct_chg")):
        pct_chg = float(r["pct_chg"])
    else:
        prev = k[k["date"] < t]
        if not prev.empty and close and pd.notna(prev.iloc[-1]["close"]):
            pct_chg = close / float(prev.iloc[-1]["close"]) - 1

    # ── 估值（取 <= t 的最新）──
    pe = pb = ps = pcf = np.nan
    pe_pct = pb_pct = np.nan
    if v is not None and not v.empty:
        vv = v[v["date"] <= t]
        if not vv.empty:
            vrow = vv.iloc[-1]
            pe = float(vrow["peTTM"]) if pd.notna(vrow["peTTM"]) else np.nan
            pb = float(vrow["pbMRQ"]) if pd.notna(vrow["pbMRQ"]) else np.nan
            ps = float(vrow["psTTM"]) if pd.notna(vrow["psTTM"]) else np.nan
            pcf = float(vrow["pcfNcfTTM"]) if pd.notna(vrow["pcfNcfTTM"]) else np.nan
            # 252 日分位（用估值历史自身）
            vhist = v[v["date"] <= t]["peTTM"].dropna()
            if len(vhist) > 20:
                pe_pct = float((vhist <= pe).mean()) if pd.notna(pe) else np.nan
            vhist_pb = v[v["date"] <= t]["pbMRQ"].dropna()
            if len(vhist_pb) > 20:
                pb_pct = float((vhist_pb <= pb).mean()) if pd.notna(pb) else np.nan

    # ── 动量 ──
    def ret_days(n: int) -> float:
        hist = k[k["date"] <= t].tail(n + 1)
        if len(hist) < 2:
            return np.nan
        c0 = float(hist["close"].iloc[0])
        c1 = float(hist["close"].iloc[-1])
        return c1 / c0 - 1 if c0 else np.nan

    mom20 = ret_days(20)
    mom60 = ret_days(60)
    mom120 = ret_days(120)
    hist252 = k[k["date"] <= t].tail(252)
    high52 = float(hist252["high"].max()) if len(hist252) and pd.notna(hist252["high"].max()) else np.nan
    dist_52w = close / high52 - 1 if high52 and close else np.nan
    amp20 = float((k[k["date"] <= t].tail(20)["high"] / k[k["date"] <= t].tail(20)["low"] - 1).mean()) if len(k[k["date"] <= t].tail(20)) else np.nan

    # ── 量价：优先 K线 turnover/amount，缺失用 realtime 快照兜底（V2-2）──
    turnover = float(r["turnover"]) if "turnover" in k.columns and pd.notna(r.get("turnover")) else np.nan
    amount = float(r["amount"]) if "amount" in k.columns and pd.notna(r.get("amount")) else np.nan
    if (pd.isna(turnover) or pd.isna(amount)) and snap_df is not None and not snap_df.empty:
        sn = snap_df[snap_df["code"].astype(str).str.contains(code, regex=False)]
        if not sn.empty:
            srow = sn.iloc[0]
            if pd.isna(turnover) and "turnover" in srow and pd.notna(srow.get("turnover")):
                turnover = float(srow["turnover"])
            if pd.isna(amount) and "amount_wan" in srow and pd.notna(srow.get("amount_wan")):
                amount = float(srow["amount_wan"]) * 1e4  # 万元 -> 元
    vol_ratio20 = np.nan
    vhist = k[k["date"] <= t].tail(21)
    if len(vhist) >= 5 and "volume" in vhist.columns:
        v20 = float(vhist["volume"].iloc[:-1].mean()) if len(vhist) > 1 else np.nan
        vcur = float(vhist["volume"].iloc[-1])
        vol_ratio20 = vcur / v20 if v20 else np.nan

    return {
        "code": code, "date": tdate,
        "close": close, "pct_chg": pct_chg,
        "peTTM": pe, "pbMRQ": pb, "psTTM": ps, "pcfNcfTTM": pcf,
        "pe_pct252": pe_pct, "pb_pct252": pb_pct,
        "mom20": mom20, "mom60": mom60, "mom120": mom120,
        "dist_52w": dist_52w, "amp20": amp20,
        "turnover": turnover, "amount": amount, "vol_ratio20": vol_ratio20,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="目标交易日 YYYYMMDD")
    ap.add_argument("--latest", action="store_true", help="用仓库最近共同交易日")
    ap.add_argument("--update", action="store_true",
                    help="滚动更新：用仓库最近共同交易日，已存在该日文件则跳过（--force 强制重建）")
    ap.add_argument("--force", action="store_true", help="--update 时强制重建已存在的日期文件")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（测试）")
    args = ap.parse_args()

    from quant_system.data_store import DataStore
    ds = DataStore()

    tdate = args.date
    if not tdate:
        if args.latest or args.update:
            tdate = _latest_common_trade_date(ds)
        else:
            tdate = _date.today().strftime("%Y%m%d")
    tdate = tdate.replace("-", "")
    print(f"目标交易日: {tdate}", flush=True)
    out_dir = ds.warehouse_root() / "feature_store"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tdate}.parquet"
    if args.update and out_path.exists() and not args.force:
        print(f"✅ 已存在 {out_path}，跳过（--force 强制重建）", flush=True)
        return 0

    codes = sorted(p.stem for p in (ds.warehouse_root() / "kline").glob("*.parquet")
                   if p.stem.isdigit() and len(p.stem) == 6)
    if args.limit:
        codes = codes[: args.limit]
    print(f"标的数: {len(codes)}", flush=True)

    t0 = time.time()
    rows = []
    val_cache: dict = {}
    # 预加载 realtime 快照（turnover/amount 兜底）
    snap_df = None
    snap_dir = ds.warehouse_root() / "realtime_snapshot" / tdate
    if snap_dir.exists():
        try:
            snap_df = pd.concat([pd.read_parquet(f) for f in sorted(snap_dir.glob("*.parquet"))], ignore_index=True)
        except Exception:
            snap_df = None
    for i, code in enumerate(codes):
        try:
            feat = _build_single(ds, code, tdate, val_cache, snap_df)
            if feat:
                rows.append(feat)
        except Exception as e:
            logging.getLogger(__name__).error(f"[build_feature_store] 操作失败: {e}", exc_info=True)
            continue
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(codes)} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        print(f"✅ 写出 {out_path} rows={len(df)} cols={len(df.columns)} {time.time()-t0:.0f}s", flush=True)
    else:
        print("⚠️ 无有效行，未写出", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
