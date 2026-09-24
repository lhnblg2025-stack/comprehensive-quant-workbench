"""Collect source-backed security master, delisting and corporate-action references.

Each source is written separately; collection failures are retained in the
manifest so the campaign gate can block rather than silently infer history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _codes(root: Path) -> list[str]:
    return sorted(path.stem.zfill(6) for path in (root / "raw" / "prices").glob("*.parquet"))


def _normalize_code(value: object) -> str:
    return str(value).replace("sh.", "").replace("sz.", "").zfill(6)


def _collect_listing_membership(bs, codes: list[str], root: Path, report: dict[str, object]) -> None:
    rows = []
    for code in codes:
        try:
            market = "sh" if code.startswith(("5", "6", "9")) else "sz"
            query = bs.query_stock_basic(code=f"{market}.{code}")
            while query.error_code == "0" and query.next():
                row = dict(zip(query.fields, query.get_row_data()))
                row["code"] = _normalize_code(row.get("code", code))
                rows.append(row)
        except Exception as exc:
            report["failures"].append({"source": "stock_basic", "code": code, "error": str(exc)})
    if not rows:
        report["membership_status"] = "missing_source"
        return
    data = pd.DataFrame(rows).drop_duplicates("code")
    list_col = "ipoDate" if "ipoDate" in data else "listDate" if "listDate" in data else None
    if list_col is None:
        report["failures"].append({"source": "stock_basic", "error": "listing_date_column_missing"})
        return
    data["start_date"] = pd.to_datetime(data[list_col], errors="coerce")
    data["end_date"] = pd.NaT
    data["membership_type"] = "source_listing_interval"
    data["source"] = "baostock_stock_basic"
    data[["code", "start_date", "end_date", "membership_type", "source"]].to_csv(root / "membership.csv", index=False)
    report["membership_status"] = "materialized_listing_intervals"
    report["membership_rows"] = int(len(data))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--actions", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    codes = _codes(root); codes = codes[:args.limit] if args.limit else codes
    import akshare as ak
    import baostock as bs
    report: dict[str, object] = {"schema": "campaign_reference_data/v2", "requested_codes": len(codes), "failures": []}
    login = bs.login()
    try:
        if login.error_code == "0":
            _collect_listing_membership(bs, codes, root, report)
        else:
            report["failures"].append({"source": "baostock_login", "error": login.error_msg})
    finally:
        bs.logout()
    delisted = []
    for getter, exchange in ((ak.stock_info_sh_delist, "SH"), (ak.stock_info_sz_delist, "SZ")):
        try:
            frame = getter().copy(); frame["exchange"] = exchange; delisted.append(frame)
        except Exception as exc:
            report["failures"].append({"source": f"{exchange}_delist", "error": str(exc)})
    if delisted:
        data = pd.concat(delisted, ignore_index=True)
        data.to_parquet(root / "delisting_reference.parquet", index=False)
        report["delisting_rows"] = int(len(data))
    if args.actions:
        rows = []
        for code in codes:
            try:
                frame = ak.stock_fhps_detail_em(symbol=code)
                if not frame.empty:
                    frame["code"] = code; frame["source"] = "stock_fhps_detail_em"; rows.append(frame)
            except Exception as exc:
                report["failures"].append({"source": "corporate_actions", "code": code, "error": str(exc)})
        if rows:
            data = pd.concat(rows, ignore_index=True)
            data.to_parquet(root / "corporate_actions_raw.parquet", index=False)
            report["corporate_action_raw_rows"] = int(len(data))
    (root / "reference_collection_manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
