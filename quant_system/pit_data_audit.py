"""Read-only audit for the configured A-share PIT research panel."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_ASSETS = {
    "panel": "historical_panel.parquet",
    "membership": "membership.csv",
    "corporate_actions": "corporate_actions.parquet",
    "liquidity_capacity": "liquidity_capacity.parquet",
    "fundamental_releases": "fundamental_releases.parquet",
    "industry_history": "sw_l1_industry_history.parquet",
    "benchmark": "benchmark_csi300.csv",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _coverage(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": int(len(frame)), "columns": list(frame.columns)}
    for field in ("code", "symbol"):
        if field in frame:
            result["symbols"] = int(frame[field].astype(str).nunique())
            break
    for field in ("date", "trade_date", "announcement_date", "effective_date", "start_date"):
        if field in frame:
            values = pd.to_datetime(frame[field], errors="coerce").dropna()
            if len(values):
                result["date_min"] = str(values.min().date())
                result["date_max"] = str(values.max().date())
            break
    return result


def audit(panel_root: str | Path, output: str | Path | None = None, *, min_symbols: int = 500, min_years: int = 10) -> dict[str, Any]:
    root = Path(panel_root)
    assets: dict[str, Any] = {}
    missing: list[str] = []
    for domain, filename in REQUIRED_ASSETS.items():
        path = root / filename
        if not path.is_file():
            missing.append(filename)
            assets[domain] = {"path": str(path), "status": "MISSING"}
            continue
        try:
            assets[domain] = {"path": str(path), "status": "PRESENT", **_coverage(_read(path))}
        except Exception as exc:
            missing.append(filename)
            assets[domain] = {"path": str(path), "status": "UNREADABLE", "error": f"{type(exc).__name__}: {exc}"}
    price = assets.get("panel", {})
    finance = assets.get("fundamental_releases", {})
    capacity = assets.get("liquidity_capacity", {})
    industry = assets.get("industry_history", {})
    symbol_counts = {"panel": int(price.get("symbols", 0)), "financials": int(finance.get("symbols", 0)), "capacity": int(capacity.get("symbols", 0)), "industry": int(industry.get("symbols", 0))}
    date_min, date_max = price.get("date_min"), price.get("date_max")
    years = 0.0
    if date_min and date_max:
        years = (pd.Timestamp(date_max) - pd.Timestamp(date_min)).days / 365.25
    errors = []
    if missing:
        errors.append("required_assets_missing_or_unreadable")
    if symbol_counts["panel"] < min_symbols:
        errors.append("panel_symbol_coverage_below_minimum")
    if symbol_counts["financials"] < min_symbols:
        errors.append("fundamental_symbol_coverage_below_minimum")
    if symbol_counts["capacity"] < min_symbols:
        errors.append("capacity_symbol_coverage_below_minimum")
    if symbol_counts["industry"] < min_symbols:
        errors.append("industry_symbol_coverage_below_minimum")
    if years < min_years:
        errors.append("panel_history_below_minimum")
    result = {
        "schema": "pit_data_audit/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "DATA_BLOCKED" if errors else "READY_FOR_MATRIX",
        "panel_root": str(root),
        "requirements": {"min_symbols": min_symbols, "min_years": min_years},
        "assets": assets,
        "symbol_coverage": symbol_counts,
        "panel_history_years": years,
        "errors": errors,
    }
    destination = Path(output) if output else root / "pit_data_audit_current.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-symbols", type=int, default=500)
    parser.add_argument("--min-years", type=int, default=10)
    args = parser.parse_args()
    result = audit(args.panel_root, args.output, min_symbols=args.min_symbols, min_years=args.min_years)
    print(json.dumps({"status": result["status"], "errors": result["errors"], "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
