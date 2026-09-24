"""Build non-authoritative tradeability candidates from raw daily OHLCV.

This artifact supports data QA only. It explicitly cannot satisfy the production
historical suspension, ST, and price-limit data contract.
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
    root = Path(args.root); rows = []
    for path in sorted((root / "raw" / "prices").glob("*.parquet")):
        data = pd.read_parquet(path, columns=["date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount"])
        data = data.sort_values("date").copy()
        close = pd.to_numeric(data["raw_close"], errors="coerce")
        prior = close.shift(1)
        change = close / prior - 1.0
        candidate = pd.DataFrame({
            "date": pd.to_datetime(data["date"]), "code": data["code"].astype(str).str.zfill(6),
            "zero_volume_candidate": pd.to_numeric(data["volume"], errors="coerce").fillna(0).le(0),
            "zero_amount_candidate": pd.to_numeric(data["amount"], errors="coerce").fillna(0).le(0),
            "limit_up_price_change_candidate": change.ge(.095),
            "limit_down_price_change_candidate": change.le(-.095),
            "state_source": "inferred_candidate_from_ohlcv_not_exchange_verified",
        })
        rows.append(candidate)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_parquet(root / "tradeability_candidates.parquet", index=False)
    report = {"schema": "tradeability_candidates/v1", "status": "not_production_eligible", "rows": int(len(out)), "symbols": int(out.code.nunique()) if len(out) else 0, "zero_volume_rows": int(out.zero_volume_candidate.sum()) if len(out) else 0, "limit_up_candidates": int(out.limit_up_price_change_candidate.sum()) if len(out) else 0, "limit_down_candidates": int(out.limit_down_price_change_candidate.sum()) if len(out) else 0, "blocked_for_execution": ["suspension_reason_missing", "st_history_missing", "official_price_limit_missing"]}
    (root / "tradeability_candidates_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
