"""Collect source-backed annual-report actual disclosure dates from AkShare."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--first-report-year", type=int, default=2015)
    parser.add_argument("--last-report-year", type=int, default=2025)
    args = parser.parse_args(argv)
    import akshare as ak
    frames = []; failures = []
    for year in range(args.first_report_year, args.last_report_year + 1):
        try:
            data = ak.stock_report_disclosure(market="沪深京", period=f"{year}年报")
            out = data.rename(columns={"股票代码": "code", "股票简称": "name", "实际披露": "announcement_date"})
            out["code"] = out["code"].astype(str).str.zfill(6)
            out["report_period"] = pd.Timestamp(f"{year}-12-31")
            out["announcement_date"] = pd.to_datetime(out["announcement_date"], errors="coerce")
            out["source_document_id"] = f"akshare_stock_report_disclosure:{year}annual"
            out["source_as_of"] = "actual_disclosure_date"
            frames.append(out[["code", "name", "report_period", "announcement_date", "source_document_id", "source_as_of"]])
        except Exception as exc:
            failures.append({"report_year": year, "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_parquet(root / "annual_disclosure_dates.parquet", index=False)
    manifest = {"schema": "annual_disclosure_dates/v1", "source": "akshare_stock_report_disclosure", "rows": int(len(result)), "report_years": sorted(result.report_period.dt.year.unique().tolist()) if len(result) else [], "failures": failures}
    (root / "annual_disclosure_dates_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
