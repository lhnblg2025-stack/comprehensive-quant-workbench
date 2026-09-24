"""Collect source-backed annual financial statements for the frozen research universe.

Raw statements are persisted per security before any cross-industry field mapping.
This preserves the source schema and lets PIT construction reject unavailable fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _market_code(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    import akshare as ak
    root = Path(args.root); outdir = root / "raw" / "annual_financials"; outdir.mkdir(parents=True, exist_ok=True)
    codes = sorted(path.stem.zfill(6) for path in (root / "raw" / "prices").glob("*.parquet"))
    if args.limit:
        codes = codes[:args.limit]
    failures = []
    for index, code in enumerate(codes, 1):
        path = outdir / f"{code}.parquet"
        if path.exists():
            continue
        parts = []
        for statement in ("资产负债表", "利润表", "现金流量表"):
            try:
                frame = ak.stock_financial_report_sina(stock=_market_code(code), symbol=statement)
                if frame.empty or "报告日" not in frame:
                    continue
                frame = frame.copy()
                frame["code"] = code; frame["statement_type"] = statement
                frame["report_period"] = pd.to_datetime(frame["报告日"], format="%Y%m%d", errors="coerce")
                parts.append(frame)
            except Exception as exc:
                failures.append({"code": code, "statement": statement, "reason": f"{type(exc).__name__}:{str(exc)[:160]}"})
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        else:
            failures.append({"code": code, "statement": "all", "reason": "no_financial_statements"})
        if index % 25 == 0:
            print(json.dumps({"processed": index, "stored": len(list(outdir.glob('*.parquet'))), "total": len(codes)}), flush=True)
    manifest = {"schema": "sina_annual_financials_raw/v1", "source": "akshare_stock_financial_report_sina", "requested": len(codes), "stored": len(list(outdir.glob("*.parquet"))), "failures": failures}
    (root / "annual_financials_collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
