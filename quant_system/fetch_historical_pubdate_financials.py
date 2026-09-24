"""Resumably fetch actual Baostock pubDate releases for a fixed long-history universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .build_long_pit_dataset import fetch_financials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    panel = pd.read_parquet(args.price_panel, columns=["code"])
    codes = sorted(panel.code.astype(str).str.zfill(6).unique())
    codes = codes[args.offset:]
    if args.limit:
        codes = codes[:args.limit]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.json"
    done = set(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed", [])) if checkpoint.exists() else set()
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock_login:{login.error_msg}")
    try:
        for index, code in enumerate(codes, 1):
            path = out / f"{code}.parquet"
            if code not in done:
                frame = fetch_financials(bs, code, range(args.start_year, args.end_year + 1))
                if not frame.empty:
                    frame.to_parquet(path, index=False)
                done.add(code)
                checkpoint.write_text(json.dumps({"completed": sorted(done), "requested": len(codes), "years": [args.start_year, args.end_year]}, ensure_ascii=True, indent=2), encoding="utf-8")
            if index % 5 == 0:
                print(json.dumps({"completed": index, "total": len(codes), "files": len(list(out.glob("*.parquet")))}, ensure_ascii=True), flush=True)
    finally:
        bs.logout()
    print(json.dumps({"status": "complete", "codes": len(codes), "files": len(list(out.glob("*.parquet")))}, ensure_ascii=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
