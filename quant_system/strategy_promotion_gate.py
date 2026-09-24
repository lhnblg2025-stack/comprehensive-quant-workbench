"""Deterministic promotion gate for factor research candidates."""
from __future__ import annotations

from typing import Any


def evaluate_candidate(candidate: dict[str, Any], *, data_quality: str, min_oos_windows: int = 3) -> dict[str, Any]:
    """Return a non-negotiable lifecycle decision with every failed condition."""
    failures: list[str] = []
    if data_quality != "actual_pubdate_and_historical_industry":
        failures.append("data_quality_not_production_pit")
    if candidate.get("return_basis") != "raw_next_open_to_following_open":
        failures.append("signal_return_basis_not_next_open")
    if int(candidate.get("oos_windows", 0)) < min_oos_windows:
        failures.append("insufficient_rolling_oos_windows")
    if float(candidate.get("oos_excess_annual", 0.0)) <= 0:
        failures.append("non_positive_oos_excess")
    if float(candidate.get("cost_x2_oos_total_return", 0.0)) <= 0:
        failures.append("fails_double_cost_stress")
    if int(candidate.get("consistent_quantiles", 0)) < 2:
        failures.append("quantile_instability")
    if not bool(candidate.get("industry_cap_pass", False)):
        failures.append("industry_cap_breach_or_infeasible")
    if not bool(candidate.get("execution_reconciled", False)):
        failures.append("execution_not_reconciled")
    return {"status": "candidate" if not failures else "research_only", "failures": failures, "promotable": not failures}
