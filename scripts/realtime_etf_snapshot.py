#!/usr/bin/env python3
"""Collect five-minute ETF quote snapshots through Tencent's batch endpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from realtime_snapshot import fetch_all, is_trading_minute, now_cst  # noqa: E402

ETF_STATE = ROOT / "data_warehouse" / "events" / "etf_state.parquet"
OUT = ROOT / "data_warehouse" / "realtime_etf_snapshot"


def _symbols() -> list[str]:
    if not ETF_STATE.exists():
        return []
    frame = pd.read_parquet(ETF_STATE, columns=["code", "name"])
    codes = frame.dropna(subset=["code"]).drop_duplicates("code", keep="last")["code"].astype(str).str.zfill(6)
    return sorted(("sh" if code.startswith(("5", "6")) else "sz") + code for code in codes)


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取ETF五分钟批量快照")
    parser.add_argument("--keep", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not is_trading_minute() and not args.force:
        print(f"[{now_cst():%H:%M:%S}] 非交易时段，ETF快照退出")
        return 0
    symbols = _symbols()
    if not symbols:
        print("ETF代码池为空", file=sys.stderr)
        return 1
    frame, errors = fetch_all(symbols)
    if frame.empty or len(frame) < max(100, int(len(symbols) * 0.8)):
        print(f"ETF快照覆盖不足 rows={len(frame)} expected={len(symbols)} errors={errors}", file=sys.stderr)
        return 1
    names = pd.read_parquet(ETF_STATE, columns=["code", "name"]).drop_duplicates("code", keep="last")
    names["code"] = names["code"].astype(str).str.zfill(6)
    frame["code6"] = frame["code"].astype(str).str[-6:]
    frame = frame.merge(names.rename(columns={"code": "code6", "name": "warehouse_name"}), on="code6", how="left")
    frame["name"] = frame["warehouse_name"].fillna(frame["name"])
    frame = frame.drop(columns=["warehouse_name"])

    day_dir = OUT / now_cst().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_cst().strftime("%H%M%S")
    path = day_dir / f"{stamp}.parquet"
    frame.to_parquet(path, index=False)
    snapshots = sorted(day_dir.glob("*.parquet"))
    for old in snapshots[:-args.keep]:
        old.unlink(missing_ok=True)
    print(f"[{now_cst():%H:%M:%S}] ETF快照 {len(frame)}/{len(symbols)}只 -> {path.name} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
