#!/usr/bin/env python3
"""增量抓取行业商品期货主力连续日线。

输出统一到 data_warehouse/market/commodity__<tag>.parquet。单品种失败不会
覆盖旧数据；命令返回非零仅当所有指定品种都失败。数据源为 AkShare 新浪主力连续。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.industry_commodity import COMMODITY_SPECS, COLLECTOR_TAGS

OUT = ROOT / "data_warehouse" / "market"
# 采集清单与画像层共享同一份合约定义，避免两处新增品种后静默分叉。
CONTRACTS = {
    tag: {
        "symbol": COMMODITY_SPECS[tag]["contract"],
        "name": COMMODITY_SPECS[tag]["name"].removesuffix("期货"),
        "exchange": COMMODITY_SPECS[tag]["exchange"],
    }
    for tag in COLLECTOR_TAGS
    if COMMODITY_SPECS.get(tag, {}).get("contract")
}


def _normalize(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    rename = {"日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low", "收盘价": "close", "成交量": "volume"}
    out = frame.rename(columns=rename).copy()
    required = {"date", "close"}
    if not required.issubset(out.columns):
        raise ValueError(f"{tag} 缺少字段 {sorted(required - set(out.columns))}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "持仓量", "动态结算价"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["tag"] = tag
    out["contract"] = CONTRACTS[tag]["symbol"]
    out["commodity_name"] = CONTRACTS[tag]["name"]
    out["exchange"] = CONTRACTS[tag]["exchange"]
    out["source"] = "akshare.futures_main_sina"
    out["fetched_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return out.dropna(subset=["date", "close"]).query("close > 0").drop_duplicates("date", keep="last").sort_values("date")


def collect(tag: str) -> dict:
    import akshare as ak
    spec = CONTRACTS[tag]
    raw = ak.futures_main_sina(symbol=spec["symbol"])
    if raw is None or raw.empty:
        raise RuntimeError(f"{spec['symbol']} 返回空")
    fresh = _normalize(raw, tag)
    path = OUT / f"commodity__{tag}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = _normalize(old, tag) if "tag" not in old.columns else old
        fresh = pd.concat([old, fresh], ignore_index=True)
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
        fresh = fresh.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fresh.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {"ok": True, "tag": tag, "symbol": spec["symbol"], "rows": len(fresh), "as_of": str(fresh["date"].max().date()), "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default=",".join(CONTRACTS), help="逗号分隔商品 tag")
    args = parser.parse_args()
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    unknown = [tag for tag in tags if tag not in CONTRACTS]
    if unknown:
        raise SystemExit(f"未知 tag: {unknown}")
    result = {}
    for tag in tags:
        try:
            result[tag] = collect(tag)
        except Exception as exc:
            result[tag] = {"ok": False, "error": str(exc)[:200]}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if any(item.get("ok") for item in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
