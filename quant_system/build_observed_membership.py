"""Build auditable broad-universe availability intervals from observed history.

This is a broad A-share research-universe contract, not index membership. Start
is the first observed tradable price; end is the earlier of last observed price
and sourced exchange delisting date where available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root); frames = []
    for path in sorted((root / "raw" / "prices").glob("*.parquet")):
        data = pd.read_parquet(path, columns=["date", "code", "raw_open", "raw_close"])
        data["date"] = pd.to_datetime(data["date"])
        valid = data[pd.to_numeric(data["raw_open"], errors="coerce").gt(0) & pd.to_numeric(data["raw_close"], errors="coerce").gt(0)]
        if not valid.empty:
            frames.append(valid.groupby("code", as_index=False)["date"].agg(start_date="min", end_date="max"))
    membership = pd.concat(frames, ignore_index=True).groupby("code", as_index=False).agg(start_date=("start_date", "min"), end_date=("end_date", "max"))
    delist_path = root / "delisting_reference.parquet"
    if delist_path.exists():
        delisted = pd.read_parquet(delist_path); delisted["code"] = delisted.code.astype(str).str.zfill(6); delisted["delist_date"] = pd.to_datetime(delisted["delist_date"], errors="coerce")
        membership = membership.merge(delisted[["code", "delist_date"]], on="code", how="left")
        membership["end_date"] = membership[["end_date", "delist_date"]].min(axis=1)
        membership = membership.drop(columns="delist_date")
    membership["universe_id"] = "broad_ashare_observed_price_universe"
    membership["membership_type"] = "observed_tradable_interval_plus_exchange_delisting"
    membership["source_as_of"] = "tushare_daily_raw_plus_exchange_delisting_reference"
    membership.to_csv(root / "membership.csv", index=False)
    report = {"schema": "observed_membership/v1", "symbols": int(len(membership)), "start_min": str(membership.start_date.min().date()), "end_max": str(membership.end_date.max().date()), "delisting_reference_used": delist_path.exists(), "limitations": ["not_index_membership", "listing_date_is_first_observed_tradable_price"]}
    (root / "membership_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
