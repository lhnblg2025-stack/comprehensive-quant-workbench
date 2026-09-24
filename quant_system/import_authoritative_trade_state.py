"""Import an authoritative trade-state extract into the production contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .trade_state_contract import validate_trade_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV or Parquet authoritative daily state extract")
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-id", required=True, help="immutable vendor/exchange dataset identifier")
    parser.add_argument("--source-as-of", required=True)
    args = parser.parse_args(argv)
    source = Path(args.input); root = Path(args.root)
    data = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    data = data.rename(columns={"symbol": "code", "trade_date": "date", "is_suspended": "suspended", "is_st": "st_flag", "limit_up": "limit_up_price", "limit_down": "limit_down_price"})
    data["source_document_id"] = data.get("source_document_id", args.source_id).fillna(args.source_id) if isinstance(data.get("source_document_id", args.source_id), pd.Series) else args.source_id
    data["source_as_of"] = data.get("source_as_of", args.source_as_of).fillna(args.source_as_of) if isinstance(data.get("source_as_of", args.source_as_of), pd.Series) else args.source_as_of
    audit = validate_trade_state(data)
    report = {"schema": "authoritative_trade_state_import/v1", "input": str(source), "source_id": args.source_id, "audit": audit}
    if audit["status"] != "PASS":
        (root / "trade_state_import_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True)); return 2
    target = root / "trade_state.parquet"; data.to_parquet(target, index=False)
    report["output"] = str(target); (root / "trade_state_import_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
