"""Validation contract for authoritative historical A-share trade state."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "code", "date", "suspended", "suspension_reason", "st_flag",
    "limit_up_price", "limit_down_price", "limit_up_locked", "limit_down_locked",
    "source_document_id", "source_as_of",
}


def validate_trade_state(frame: pd.DataFrame, *, expected_pairs: set[tuple[str, pd.Timestamp]] | None = None) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    errors: list[str] = []
    if missing:
        errors.append("missing_columns:" + ",".join(missing))
        return {"status": "BLOCK", "errors": errors, "rows": int(len(frame)), "symbols": 0}
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data[["code", "date"]].duplicated().any():
        errors.append("duplicate_code_date")
    if data["date"].isna().any():
        errors.append("invalid_dates")
    if data["source_document_id"].isna().any() or data["source_document_id"].astype(str).str.strip().eq("").any():
        errors.append("missing_source_document_id")
    if data["source_as_of"].isna().any():
        errors.append("missing_source_as_of")
    source = data["source_document_id"].astype(str).str.lower()
    if source.str.contains("unverified|inferred|proxy|placeholder|derived|candidate", regex=True).any():
        errors.append("non_authoritative_source")
    if expected_pairs is not None:
        actual = set(zip(data["code"], data["date"]))
        missing_pairs = expected_pairs - actual
        if missing_pairs:
            errors.append(f"coverage_missing_pairs:{len(missing_pairs)}")
    return {"status": "PASS" if not errors else "BLOCK", "errors": errors, "rows": int(len(data)), "symbols": int(data["code"].nunique()), "date_min": str(data["date"].min().date()) if len(data) else None, "date_max": str(data["date"].max().date()) if len(data) else None}


def audit_trade_state_file(path: str | Path, price_pairs: set[tuple[str, pd.Timestamp]] | None = None) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        return {"status": "BLOCK", "errors": [f"missing:{file}"], "rows": 0, "symbols": 0}
    frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
    return validate_trade_state(frame, expected_pairs=price_pairs)
