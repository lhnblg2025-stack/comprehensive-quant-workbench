"""Materialize broad PIT price, financial, industry, membership and benchmark tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .pit_fundamentals import attach_pit_fundamentals, attach_pit_industry


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_parts(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {directory}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    prices = _read_parts(root / "raw" / "prices")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.dropna(subset=["date", "code", "close", "raw_open", "raw_close"])
    prices = prices.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    calendar = pd.read_csv(args.calendar)
    sessions = pd.to_datetime(calendar.iloc[:, 0], errors="coerce").dropna()
    financials = _read_parts(root / "raw" / "financials")
    panel = attach_pit_fundamentals(prices, financials, sessions, lag_sessions=1)
    industry_path = root / "sw_l1_industry_history.parquet"
    panel = attach_pit_industry(panel, pd.read_parquet(industry_path))
    panel.to_parquet(root / "historical_panel.parquet", index=False)
    membership = panel.groupby("code", as_index=False)["date"].agg(start_date="min", end_date="max")
    membership["universe_id"] = "baostock_broad_500"
    membership["source_as_of"] = "price_observed_interval"
    membership.to_csv(root / "membership.csv", index=False)
    benchmark = pd.read_parquet(args.benchmark)
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date")
    benchmark["return"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change()
    benchmark = benchmark[["date", "return"]].dropna()
    benchmark = benchmark[benchmark["date"].isin(panel["date"].unique())]
    benchmark.to_csv(root / "benchmark_csi300.csv", index=False)
    coverage = panel.groupby("date")["code"].nunique()
    report = {
        "schema": "long_pit_materialized/v1",
        "panel_rows": int(len(panel)),
        "symbols": int(panel["code"].nunique()),
        "dates": int(panel["date"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "min_cross_section": int(coverage.min()),
        "median_cross_section": int(coverage.median()),
        "financial_announcement_coverage": float(panel["announcement_date"].notna().mean()),
        "industry_coverage": float(panel["industry"].notna().mean()),
        "benchmark_rows": int(len(benchmark)),
        "panel_sha256": _hash(root / "historical_panel.parquet"),
    }
    (root / "materialization_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
