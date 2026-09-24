"""Read-only completeness and point-in-time audit for frozen backtest bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .dual_price_quality import validate_dual_price_panel

REQUIRED_ASSETS = (
    "prices_hfq_raw.parquet",
    "corporate_actions.parquet",
    "universe_membership.parquet",
    "trade_calendar.parquet",
    "benchmark_csi300.parquet",
    "factor_registry.json",
    "data_manifest.json",
)
VALID_ACTION_TYPES = {"dividend", "split"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_hash_checks(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {str(item.get("name")): str(item.get("sha256")) for item in manifest.get("assets", [])}
    mismatches = []
    missing_manifest_hashes = []
    for name in REQUIRED_ASSETS:
        if name == "data_manifest.json":
            continue
        if name not in expected:
            missing_manifest_hashes.append(name)
            continue
        path = root / name
        if path.exists():
            actual = _sha256(path)
            if actual != expected[name]:
                mismatches.append({"asset": name, "expected": expected[name], "actual": actual})
    return {"mismatches": mismatches, "missing_manifest_hashes": missing_manifest_hashes}


def audit_bundle(
    bundle_dir: str | Path,
    *,
    min_symbols: int,
    min_days: int,
    write_report: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    missing = [name for name in REQUIRED_ASSETS if not (root / name).exists()]
    if missing:
        return {"status": "BLOCK", "bundle": str(root), "missing_assets": missing, "errors": ["missing_assets"]}

    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    prices = pd.read_parquet(root / "prices_hfq_raw.parquet").copy()
    actions = pd.read_parquet(root / "corporate_actions.parquet").copy()
    membership = pd.read_parquet(root / "universe_membership.parquet").copy()
    calendar = pd.read_parquet(root / "trade_calendar.parquet").copy()
    benchmark = pd.read_parquet(root / "benchmark_csi300.parquet").copy()

    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    dual = validate_dual_price_panel(prices, min_symbols=min_symbols)
    per_symbol = prices.groupby("code").date.nunique()
    short = per_symbol[per_symbol < min_days]

    membership["code"] = membership["code"].astype(str).str.zfill(6)
    membership["start_date"] = pd.to_datetime(membership.get("start_date"), errors="coerce")
    membership["end_date"] = pd.to_datetime(membership.get("end_date"), errors="coerce")
    price_codes = set(prices.code)
    membership_codes = set(membership.code)

    actions["symbol"] = actions["symbol"].astype(str).str.zfill(6)
    actions["date"] = pd.to_datetime(actions.get("date"), errors="coerce")
    action_in_period = actions[actions.date.between(prices.date.min(), prices.date.max(), inclusive="both")]
    invalid_action_types = sorted(set(actions.action_type.astype(str)) - VALID_ACTION_TYPES)
    invalid_action_sources = int(actions.get("source", pd.Series(dtype=str)).astype(str).str.strip().eq("").sum())
    invalid_action_status = int((~actions.get("status", pd.Series(index=actions.index, dtype=str)).astype(str).str.contains("实施", na=False)).sum()) if len(actions) else 0

    calendar_dates = set(pd.to_datetime(calendar.iloc[:, 0], errors="coerce").dropna().dt.normalize())
    price_dates = set(prices.date.dropna().dt.normalize())
    missing_calendar_dates = sorted(str(day.date()) for day in price_dates - calendar_dates)
    benchmark_dates = set(pd.to_datetime(benchmark["date"], errors="coerce").dropna().dt.normalize())
    missing_benchmark_dates = sorted(str(day.date()) for day in price_dates - benchmark_dates)

    membership_invalid_bounds = int(membership.start_date.isna().sum())
    membership_reversed = int(((membership.end_date.notna()) & (membership.end_date < membership.start_date)).sum())
    outside_membership = prices.merge(membership[["code", "start_date", "end_date"]], on="code", how="left")
    outside_mask = outside_membership.start_date.isna() | (outside_membership.date < outside_membership.start_date) | (
        outside_membership.end_date.notna() & (outside_membership.date > outside_membership.end_date)
    )

    hashes = _asset_hash_checks(root, manifest)
    manifest_counts = {
        "collected_symbols": int(manifest.get("collected_symbols", -1)),
        "price_rows": int(manifest.get("price_rows", -1)),
        "action_rows": int(manifest.get("action_rows", -1)),
        "membership_rows": int(manifest.get("membership_rows", -1)),
    }
    actual_counts = {
        "collected_symbols": int(prices.code.nunique()),
        "price_rows": int(len(prices)),
        "action_rows": int(len(actions)),
        "membership_rows": int(len(membership)),
    }
    count_mismatches = {key: {"manifest": manifest_counts[key], "actual": value} for key, value in actual_counts.items() if manifest_counts[key] != value}

    checks = {
        "schema_version": manifest.get("schema_version"),
        "symbols": actual_counts["collected_symbols"],
        "rows": len(prices),
        "date_min": str(prices.date.min().date()),
        "date_max": str(prices.date.max().date()),
        "short_history_symbols": short.index.astype(str).tolist(),
        "membership_missing_symbols": sorted(price_codes - membership_codes),
        "membership_invalid_bounds": membership_invalid_bounds,
        "membership_reversed": membership_reversed,
        "price_rows_outside_membership": int(outside_mask.sum()),
        "corporate_action_rows": len(actions),
        "corporate_actions_in_period": len(action_in_period),
        "invalid_action_dates": int(actions.date.isna().sum()),
        "invalid_action_types": invalid_action_types,
        "invalid_action_sources": invalid_action_sources,
        "invalid_action_status": invalid_action_status,
        "calendar_rows": len(calendar),
        "missing_calendar_dates": missing_calendar_dates,
        "benchmark_rows": len(benchmark),
        "missing_benchmark_dates": missing_benchmark_dates,
        "manifest_hashes": hashes,
        "manifest_count_mismatches": count_mismatches,
        "dual_price": dual,
    }

    errors = []
    if dual.get("status") == "BLOCK": errors.append("dual_price")
    if len(price_codes) < min_symbols: errors.append("insufficient_symbols")
    if len(short): errors.append("short_history")
    if checks["membership_missing_symbols"]: errors.append("membership_coverage")
    if membership_invalid_bounds or membership_reversed or checks["price_rows_outside_membership"]: errors.append("membership_pit")
    if checks["invalid_action_dates"] or invalid_action_types or invalid_action_sources or invalid_action_status: errors.append("corporate_actions")
    if missing_calendar_dates: errors.append("calendar_alignment")
    if hashes["mismatches"] or hashes["missing_manifest_hashes"] or count_mismatches: errors.append("manifest_integrity")

    warnings = []
    if missing_benchmark_dates:
        warnings.append("benchmark_alignment")
    result = {"status": "BLOCK" if errors else "PASS", "bundle": str(root), "checks": checks, "errors": errors, "warnings": warnings}
    if write_report:
        (root / "quality_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
