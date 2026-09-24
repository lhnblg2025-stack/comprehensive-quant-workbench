"""Finalize collected long-PIT shards into auditable campaign input tables.

This command never manufactures missing history. It consolidates genuinely
collected PIT releases and, when supplied, copies source-backed membership,
corporate-action, and industry history tables into the campaign root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _combine(directory: Path, output: Path, *, keys: list[str]) -> dict[str, object]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return {"status": "missing_source", "files": 0, "rows": 0}
    data = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if data.empty:
        return {"status": "empty_source", "files": len(files), "rows": 0}
    for column in ("code",):
        if column in data:
            data[column] = data[column].astype(str).str.replace(r"^(sh|sz)\\.", "", regex=True).str.zfill(6)
    data = data.drop_duplicates(keys, keep="last").sort_values(keys)
    data.to_parquet(output, index=False)
    return {"status": "materialized", "files": len(files), "rows": int(len(data)), "output": output.name}


def _copy_source(source: Path | None, destination: Path) -> dict[str, object]:
    if source is None or not source.is_file():
        return {"status": "missing_source"}
    data = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    if destination.suffix == ".csv":
        data.to_csv(destination, index=False)
    else:
        data.to_parquet(destination, index=False)
    return {"status": "copied", "rows": int(len(data)), "output": destination.name}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--industry-source")
    parser.add_argument("--membership-source")
    parser.add_argument("--actions-source")
    args = parser.parse_args(argv)
    root = Path(args.root)
    industry_destination = root / "sw_l1_industry_history.parquet"
    report = {
        "schema": "long_pit_finalize/v1",
        "fundamentals": _combine(root / "raw" / "financials", root / "fundamental_releases.parquet", keys=["code", "report_period", "announcement_date"]),
        "prices": _combine(root / "raw" / "prices", root / "prices_raw_hfq.parquet", keys=["code", "date"]),
        "industry": _copy_source(Path(args.industry_source), industry_destination) if args.industry_source else ({"status": "existing", "output": industry_destination.name} if industry_destination.exists() else {"status": "missing_source"}),
        "membership": _copy_source(Path(args.membership_source), root / "membership.csv") if args.membership_source else ({"status": "existing", "output": "membership.csv"} if (root / "membership.csv").exists() else {"status": "missing_source"}),
        "corporate_actions": _copy_source(Path(args.actions_source), root / "corporate_actions.parquet") if args.actions_source else ({"status": "existing", "output": "corporate_actions.parquet"} if (root / "corporate_actions.parquet").exists() else {"status": "missing_source"}),
    }
    (root / "finalization_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
