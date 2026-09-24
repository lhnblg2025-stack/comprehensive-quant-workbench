"""Build a derived trade-state QA table, never an authoritative production source.

Prices and exchange rules can identify candidates for limit locks and missing
bars; they cannot prove suspension reasons or historical ST status.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _limit_rate(code: str) -> float:
    if code.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root); price_files = sorted((root / "raw" / "prices").glob("*.parquet"))
    # Market calendar = union of every observed price date across the universe.
    market_dates = pd.DatetimeIndex(sorted({ts for path in price_files for ts in pd.read_parquet(path, columns=["date"])["date"].dropna()}), dtype="datetime64[ns]")
    rows = []
    for path in price_files:
        d = pd.read_parquet(path).sort_values("date").copy(); d["date"] = pd.to_datetime(d["date"]); code = path.stem.zfill(6)
        close = pd.to_numeric(d["raw_close"], errors="coerce"); pre = close.shift(1)
        pct = close / pre - 1.0
        rate = _limit_rate(code)
        limit_up_price = (pre * (1.0 + rate)).round(2)
        limit_down_price = (pre * (1.0 - rate)).round(2)
        observed = set(d["date"])
        first, last = d["date"].iloc[0], d["date"].iloc[-1]
        expected = market_dates[(market_dates >= first) & (market_dates <= last)]
        suspended = ~pd.Series(expected).isin(observed)
        suspended_map = dict(zip(expected, suspended.to_numpy()))
        row = pd.DataFrame({
            "code": code, "date": d["date"], "suspended": d["date"].map(suspended_map).fillna(False).astype(bool),
            "suspension_reason": pd.NA, "st_flag": pd.NA,
            "limit_up_price": limit_up_price, "limit_down_price": limit_down_price,
            "limit_up_locked": pct.ge(rate - 0.005) & pd.to_numeric(d["raw_open"], errors="coerce").eq(pd.to_numeric(d["raw_high"], errors="coerce")),
            "limit_down_locked": pct.le(-(rate - 0.005)) & pd.to_numeric(d["raw_open"], errors="coerce").eq(pd.to_numeric(d["raw_low"], errors="coerce")),
            "source_document_id": "derived_calendar_and_board_rules_candidate", "source_as_of": d["date"], "state_confidence": "research_candidate_not_authoritative",
        })
        rows.append(row)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_parquet(root / "trade_state_derived_candidates.parquet", index=False)
    report = {"schema": "trade_state_derived_candidates/v2", "status": "not_production_eligible", "rows": int(len(out)), "symbols": int(out.code.nunique()) if len(out) else 0, "blocked_fields": ["suspension_reason", "st_flag", "official_lock_status"], "limit_price_basis": "board_rule_10_or_20_pct_without_st_adjustment", "source": "derived_calendar_board_rules_ohlcv"}
    (root / "trade_state_derived_candidates_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
