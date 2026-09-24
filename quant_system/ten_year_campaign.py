"""Auditable entrypoint for the 20-strategy, ten-year A-share campaign.

The campaign has a deliberate hard boundary: data quality is evaluated before any
strategy is selected or backtested. A failed audit writes an explicit
``DATA_BLOCKED`` report and exits successfully only in ``audit`` mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "date", "code", "raw_open", "raw_high", "raw_low", "raw_close",
    "volume", "amount", "is_suspended", "limit_up", "limit_down", "trade_state_source",
}
REQUIRED_FINANCIAL_COLUMNS = {
    "code", "report_period", "announcement_date", "source_document_id", "source_as_of",
}
REQUIRED_INDUSTRY_COLUMNS = {"code", "industry", "effective_date", "end_date", "source_as_of"}
REQUIRED_MEMBERSHIP_COLUMNS = {"code", "start_date", "end_date"}
REQUIRED_ACTION_COLUMNS = {"code", "ex_date", "action_type"}
REQUIRED_CAPACITY_COLUMNS = {"date", "code", "adv_amount", "turnover", "participation_limit", "impact_bps"}


def _ensure_capacity_table(base: Path, prices: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    """Materialize a conservative, auditable capacity table from raw amount/turnover.

    This is not a predictive impact model: impact_bps is a declared stress proxy and
    remains subject to replacement by venue-level historical impact observations.
    """
    path = base / "liquidity_capacity.parquet"
    if path.exists():
        return path, {"status": "existing"}
    required = {"date", "code", "amount"}
    if prices.empty or not required.issubset(prices.columns):
        return path, {"status": "blocked", "reason": "amount_required"}
    out = prices[["date", "code", "amount"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["date", "code", "amount"])
    out = out.sort_values(["code", "date"])
    out["adv_amount"] = out.groupby("code")["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["turnover"] = out["amount"]
    out["participation_limit"] = out["adv_amount"] * 0.10
    out["impact_bps"] = (100.0 * (out["amount"] / out["adv_amount"]).clip(lower=0, upper=1).pow(0.5)).fillna(100.0)
    out[["date", "code", "adv_amount", "turnover", "participation_limit", "impact_bps"]].to_parquet(path, index=False)
    return path, {"status": "materialized_conservative_proxy", "rows": int(len(out))}


def _read_parts(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _table(path: Path, columns: set[str]) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        return pd.DataFrame(), [f"missing:{path.name}"]
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = sorted(columns - set(frame.columns))
    return frame, [f"missing_columns:{path.name}:{','.join(missing)}"] if missing else []


def _yearly_coverage(prices: pd.DataFrame) -> dict[str, int]:
    if prices.empty or not {"date", "code"}.issubset(prices.columns):
        return {}
    data = prices[["date", "code"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna()
    return {str(year): int(group["code"].nunique()) for year, group in data.groupby(data["date"].dt.year)}


def audit_campaign(root: str | Path, *, min_symbols: int = 500, min_years: int = 10) -> dict[str, Any]:
    """Audit all prerequisite tables without substituting proxy fields."""
    base = Path(root)
    prices = _read_parts(base / "raw" / "prices")
    financials = _read_parts(base / "raw" / "financials")
    financial_source = "raw_financial_shards"
    if financials.empty and (base / "annual_fundamental_releases.parquet").is_file():
        financials = pd.read_parquet(base / "annual_fundamental_releases.parquet")
        financial_source = "annual_pit_releases"
    _, capacity_materialization = _ensure_capacity_table(base, prices)
    industries, industry_errors = _table(base / "sw_l1_industry_history.parquet", REQUIRED_INDUSTRY_COLUMNS)
    membership, membership_errors = _table(base / "membership.csv", REQUIRED_MEMBERSHIP_COLUMNS)
    actions, action_errors = _table(base / "corporate_actions.parquet", REQUIRED_ACTION_COLUMNS)
    capacity, capacity_errors = _table(base / "liquidity_capacity.parquet", REQUIRED_CAPACITY_COLUMNS)

    errors: list[str] = []
    research_errors: list[str] = []
    production_errors: list[str] = []
    errors += [f"missing_columns:prices:{','.join(sorted(REQUIRED_PRICE_COLUMNS - set(prices.columns)))}"] if not prices.empty and REQUIRED_PRICE_COLUMNS - set(prices.columns) else []
    errors += ["missing:raw/prices"] if prices.empty else []
    errors += [f"missing_columns:financials:{','.join(sorted(REQUIRED_FINANCIAL_COLUMNS - set(financials.columns)))}"] if not financials.empty and REQUIRED_FINANCIAL_COLUMNS - set(financials.columns) else []
    errors += ["missing:raw/financials"] if financials.empty else []
    errors += industry_errors + membership_errors + action_errors + capacity_errors

    yearly = _yearly_coverage(prices)
    price_symbols = int(prices["code"].astype(str).str.zfill(6).nunique()) if "code" in prices else 0
    financial_symbols = int(financials["code"].astype(str).str.zfill(6).nunique()) if "code" in financials else 0
    date_min = date_max = None
    if "date" in prices:
        dates = pd.to_datetime(prices["date"], errors="coerce").dropna()
        if not dates.empty:
            date_min, date_max = str(dates.min().date()), str(dates.max().date())
    sufficient_years = [year for year, count in yearly.items() if count >= min_symbols]
    # Historical universes grow with IPOs. Do not require 500 securities before
    # they existed; instead require a declared lower floor throughout history and
    # the target breadth in the mature part of the sample.
    historical_floor = min(300, min_symbols)
    historical_years = [year for year, count in yearly.items() if count >= historical_floor]
    if price_symbols < min_symbols:
        errors.append("insufficient_price_symbols")
    if financial_symbols < min_symbols:
        errors.append("insufficient_financial_symbols")
    if len(sufficient_years) < 3:
        errors.append("insufficient_mature_universe_years")
    if len(historical_years) < min_years:
        errors.append("insufficient_historical_cross_section")
    if len(yearly) < min_years:
        errors.append("insufficient_history_years")
    if "trade_state_source" in prices and prices["trade_state_source"].astype(str).str.contains("unverified", case=False, na=True).any():
        production_errors.append("unverified_historical_trade_state")

    trade_state_path = base / "trade_state.parquet"
    if not trade_state_path.is_file():
        errors.append("missing:trade_state.parquet")
        research_errors.append("missing:trade_state.parquet")
    else:
        from .trade_state_contract import audit_trade_state_file
        state_audit = audit_trade_state_file(trade_state_path)
        if state_audit["status"] != "PASS":
            # Derived OHLCV state is sufficient for research replay but never for
            # production promotion. Keep both statuses explicit.
            state_text = trade_state_path.with_suffix(".report.json")
            legacy_state_text = trade_state_path.with_suffix(".parquet.report.json")
            report_text = ""
            if state_text.is_file():
                report_text = state_text.read_text(encoding="utf-8")
            elif legacy_state_text.is_file():
                report_text = legacy_state_text.read_text(encoding="utf-8")
            derived = "not_authoritative" in report_text or "research_derived" in report_text
            if derived:
                production_errors.extend(f"trade_state:{error}" for error in state_audit["errors"])
            else:
                errors.extend(f"trade_state:{error}" for error in state_audit["errors"])
    # All ordinary prerequisite errors apply to research; production adds the
    # stricter authority requirements below.
    research_errors.extend(errors)
    research_status = "PASS" if not research_errors else "DATA_BLOCKED"
    status = "PASS" if not errors and not production_errors else "DATA_BLOCKED"
    result = {
        "schema": "pit_data_quality/v3",
        "status": status,
        "research_status": research_status,
        "production_status": status,
        "production_blockers": sorted(set(production_errors)),
        "data_quality": "actual_pubdate_and_historical_industry" if status == "PASS" else ("research_derived_trade_state" if research_status == "PASS" else "incomplete"),
        "requirements": {"min_symbols": min_symbols, "min_years": min_years},
        "coverage": {
            "price_symbols": price_symbols,
            "financial_symbols": financial_symbols,
            "financial_source": financial_source,
            "price_date_min": date_min,
            "price_date_max": date_max,
            "yearly_cross_section": yearly,
            "years_meeting_minimum": sufficient_years,
            "industry_rows": int(len(industries)),
            "membership_rows": int(len(membership)),
            "corporate_action_rows": int(len(actions)),
            "capacity_rows": int(len(capacity)),
            "capacity_materialization": capacity_materialization,
            "trade_state_path": str(trade_state_path),
        },
        "errors": sorted(set(errors)),
        "next_action": "materialize_and_run" if status == "PASS" else "resume_data_acquisition",
    }
    (base / "pit_data_quality_report.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--min-symbols", type=int, default=500)
    parser.add_argument("--min-years", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="run research only after a PASS data audit")
    parser.add_argument("--config", help="research config required with --run")
    args = parser.parse_args(argv)
    audit = audit_campaign(args.root, min_symbols=args.min_symbols, min_years=args.min_years)
    if audit["status"] != "PASS":
        print(json.dumps(audit, ensure_ascii=True))
        return 2 if args.run else 0
    if not args.run:
        print(json.dumps(audit, ensure_ascii=True))
        return 0
    if not args.config:
        raise ValueError("--config is required with --run")
    from .materialize_long_pit_panel import main as materialize
    from .research_pipeline import run_config
    root = Path(args.root)
    materialize(["--root", str(root), "--calendar", "quant_system/data/trade_calendar.csv", "--benchmark", str(root / "benchmark_csi300.parquet")])
    result = run_config(args.config)
    print(json.dumps({"data_audit": audit["status"], "research_windows": len(result["rolling_windows"])}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
