"""Materialize annual PIT value/quality research panel from dual-price shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pit_fundamentals import add_fundamental_factors, attach_pit_fundamentals, attach_pit_industry, coverage_summary, industry_neutralize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--industry-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-stocks", type=int, default=30)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.80)
    args = parser.parse_args(argv)
    root = Path(args.root)
    prices = pd.concat([pd.read_parquet(p) for p in sorted((root / "raw" / "prices").glob("*.parquet"))], ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    # Signal aliases are adjusted prices while raw fields remain execution prices.
    prices[["open", "high", "low", "close"]] = prices[["hfq_open", "hfq_high", "hfq_low", "hfq_close"]]
    prices = prices.dropna(subset=["close", "raw_open", "raw_close"])
    releases = pd.read_parquet(root / "annual_fundamental_releases.parquet")
    calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
    panel = attach_pit_fundamentals(prices, releases, calendar, lag_sessions=1)
    industries = pd.read_parquet(args.industry_history)
    panel = attach_pit_industry(panel, industries)
    panel = add_fundamental_factors(panel)
    base = ("value_book_to_price", "quality_low_leverage", "value_quality_composite")
    panel = industry_neutralize(panel, base, min_industry_size=5)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); panel.to_parquet(output, index=False)
    factors = tuple(f"{name}_industry_neutral" for name in base)
    limitations = ["annual_pit_only", "historical_trade_state_unverified"]
    if not (root / "corporate_actions.parquet").is_file():
        limitations.append("corporate_actions_missing")
    report = {"schema": "annual_pit_research_panel/v2", "status": "research_only_pending_execution_data", "symbols": int(panel.code.nunique()), "rows": int(len(panel)), "date_min": str(panel.date.min().date()), "date_max": str(panel.date.max().date()), "factor_coverage": coverage_summary(panel, factors, min_stocks=args.min_stocks, min_coverage_ratio=args.min_coverage_ratio), "limitations": limitations}
    (output.parent / "annual_pit_research_panel_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
