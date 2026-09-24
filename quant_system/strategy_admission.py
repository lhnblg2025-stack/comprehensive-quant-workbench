"""Research admission gate for predeclared Strategy Lab candidates.

This module does not select a winner. It creates an immutable, auditable decision
record and fails closed whenever PIT universe/trade-state, OOS evidence, complete
trial accounting, or capacity data are missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .overfitting_tests import deflated_sharpe_ratio
from .research_contracts import QualityResult, sha256_file, write_manifest
from .research_pipeline import rolling_windows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "price_discovery_panel.parquet"
DEFAULT_TRADE_STATE = (ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_merged" / "trade_state_pit.parquet") if (ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_merged" / "trade_state_pit.parquet").is_file() else ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "trade_state.parquet"
DEFAULT_QUALITY = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_data_quality_report.json"
DEFAULT_EXECUTION = ROOT / "generated" / "strategy_lab" / "template_full_execution.json"
DEFAULT_REGISTRY = ROOT / "quant_system" / "configs" / "research" / "strategy_lab_admission_candidates.json"
DEFAULT_SOURCE_CAPABILITIES = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_source_capabilities.json"
DEFAULT_SOURCE_CATALOG = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_source_catalog.json"
DEFAULT_QLIB_MANIFEST = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "qlib_provider_staging" / "manifest.json"
DEFAULT_CLOUD_PIT_MANIFEST = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "cloud_pit_tradability" / "manifest.json"
DEFAULT_QLIB_OOS = ROOT / "generated" / "strategy_lab" / "qlib_candidate_oos.json"
DEFAULT_ORDER_RECONCILIATION = ROOT / "generated" / "strategy_lab" / "candidate_order_reconciliation.json"


@dataclass(frozen=True)
class AdmissionSpec:
    candidate_id: str
    template: str
    top_n: int = 20
    rebalance_sessions: int = 20
    label_horizon_sessions: int = 20
    min_oos_windows: int = 4
    min_oos_observations: int = 252
    max_capacity_participation: float = 0.10


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> tuple[AdmissionSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("candidates", [])
    specs = tuple(AdmissionSpec(**item) for item in items)
    if not specs:
        raise ValueError("admission_candidate_registry_empty")
    ids = [item.candidate_id for item in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("admission_candidate_registry_duplicate_id")
    return specs


def _load_quality(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _trade_state_gate(trade_state: Path, quality: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {"path": str(trade_state), "exists": trade_state.is_file()}
    if not trade_state.is_file():
        return False, ["authoritative_historical_trade_state_missing"], summary
    frame = pd.read_parquet(trade_state, columns=None)
    required = {"date", "code", "suspended", "limit_up_locked", "limit_down_locked"}
    missing = sorted(required - set(frame.columns))
    summary.update({"rows": int(len(frame)), "columns": list(frame.columns), "missing_columns": missing})
    if missing:
        blockers.append("historical_trade_state_schema_incomplete")
    if str(quality.get("research_status") or quality.get("status")) == "DATA_BLOCKED":
        blockers.append("historical_trade_state_not_authoritative")
    return not blockers, blockers, summary


def _source_capability_gate(path: Path = DEFAULT_SOURCE_CAPABILITIES) -> tuple[bool, list[str], dict[str, Any]]:
    report = _load_quality(path)
    if not report:
        return False, ["pit_source_capability_report_missing"], {"path": str(path), "status": "missing"}
    unavailable = report.get("unavailable_or_limited", {})
    summary = {"path": str(path), "status": report.get("status"), "unavailable_or_limited": unavailable, "next_action": report.get("next_action")}
    blockers = []
    if unavailable.get("suspend_d") == "permission_denied":
        blockers.append("source_permission_missing_suspend_d")
    if unavailable.get("stk_limit") == "permission_denied":
        blockers.append("source_permission_missing_stk_limit")
    if unavailable.get("stock_basic") in {"rate_limited", "permission_denied"}:
        blockers.append("source_limited_stock_basic")
    if unavailable.get("namechange") in {"rate_limited", "permission_denied"}:
        blockers.append("source_limited_namechange")
    return not blockers, blockers, summary


def _qlib_provider_gate(path: Path = DEFAULT_QLIB_MANIFEST) -> tuple[bool, list[str], dict[str, Any]]:
    manifest = _load_quality(path)
    if not manifest:
        return False, ["qlib_provider_manifest_missing"], {"status": "missing", "path": str(path)}
    blockers = []
    if manifest.get("binary_dump_status") != "READY":
        blockers.append("qlib_binary_provider_not_ready")
    if manifest.get("status") not in {"READY_RESEARCH_STAGING", "PRODUCTION_READY"}:
        blockers.append("qlib_provider_not_research_ready")
    return not blockers, blockers, {"path": str(path), "status": manifest.get("status"), "binary_dump_status": manifest.get("binary_dump_status"), "instruments": manifest.get("instruments"), "sessions": manifest.get("sessions")}


def _pit_universe_gate(panel: pd.DataFrame, quality: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    coverage = quality.get("coverage", {}) if isinstance(quality, dict) else {}
    summary = {
        "symbols": int(panel["code"].nunique()),
        "date_min": str(pd.to_datetime(panel["date"]).min().date()),
        "date_max": str(pd.to_datetime(panel["date"]).max().date()),
        "yearly_cross_section": coverage.get("yearly_cross_section", {}),
        "membership_rows": coverage.get("membership_rows", 0),
    }
    blockers = []
    # A fixed membership count is not dated point-in-time membership.
    if int(coverage.get("membership_rows", 0) or 0) <= int(panel["code"].nunique()):
        blockers.append("historical_point_in_time_universe_missing")
    return not blockers, blockers, summary


def _capacity_gate(panel: pd.DataFrame, spec: AdmissionSpec) -> tuple[bool, list[str], dict[str, Any]]:
    if "amount" not in panel.columns:
        return False, ["capacity_amount_missing"], {}
    amount = pd.to_numeric(panel["amount"], errors="coerce")
    summary = {
        "non_null_amount_ratio": float(amount.notna().mean()),
        "non_positive_amount_ratio": float((amount.fillna(0) <= 0).mean()),
        "median_daily_amount": float(amount.dropna().median()) if amount.notna().any() else None,
        "max_participation": spec.max_capacity_participation,
    }
    blockers = []
    if summary["non_null_amount_ratio"] < 0.95 or summary["non_positive_amount_ratio"] > 0.05:
        blockers.append("capacity_data_incomplete")
    # Amount availability alone is not a capacity study. Until an explicit AUM,
    # ADV horizon, impact model, and per-order participation simulation are fixed,
    # capacity must remain a promotion blocker.
    summary["status"] = "preliminary_amount_coverage_only"
    blockers.append("capacity_impact_model_missing")
    return False, blockers, summary


def _execution_record(execution: dict[str, Any], template: str) -> dict[str, Any] | None:
    return (execution.get("templates") or {}).get(template)


def _bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values for the complete trial ledger."""
    if not p_values:
        return []
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n, 0, -1):
        index = int(order[rank - 1])
        running = min(running, float(p_values[index]) * n / rank)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _oos_plan(panel: pd.DataFrame, specs: tuple[AdmissionSpec, ...]) -> dict[str, Any]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    plan = rolling_windows(len(dates), train=504, validation=126, oos=126, step=126, purge=20, embargo=20)
    return {
        "status": "planned_not_run",
        "protocol": "expanding_train_validation_oos_with_20_session_purge_and_embargo",
        "candidate_count": len(specs),
        "windows": [
            {
                "train": [str(dates[w["train"][0]].date()), str(dates[w["train"][1] - 1].date())],
                "validation": [str(dates[w["validation"][0]].date()), str(dates[w["validation"][1] - 1].date())],
                "oos": [str(dates[w["oos"][0]].date()), str(dates[w["oos"][1] - 1].date())],
            }
            for w in plan
        ],
        "blocker": "authoritative_pit_trade_state_required_before_canonical_oos_execution",
    }


def _trial_metrics(record: dict[str, Any] | None, trial_count: int) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not record or record.get("status") != "executed":
        return {"status": "missing"}, ["full_execution_missing"]
    metrics = record.get("metrics") or {}
    observations = int(metrics.get("observations") or 0)
    annual_sharpe = metrics.get("sharpe")
    if annual_sharpe is None or observations < 2:
        return {"status": "insufficient"}, ["execution_metrics_insufficient"]
    daily_sharpe = float(annual_sharpe) / np.sqrt(252.0)
    dsr, p_value = deflated_sharpe_ratio(daily_sharpe, trial_count, observations)
    return {
        "status": "computed",
        "annual_sharpe": float(annual_sharpe),
        "daily_sharpe": daily_sharpe,
        "observations": observations,
        "dsr": float(dsr),
        "dsr_p_value": float(p_value),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
    }, blockers


def _load_candidate_evidence(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_quality(path)
    rows = payload.get("candidates", payload.get("results", []))
    return {str(row.get("candidate")): row for row in rows if isinstance(row, dict)}


def _oos_evidence(path: Path, template: str) -> dict[str, Any] | None:
    payload = _load_quality(path)
    for row in payload.get("candidates", []):
        if row.get("candidate") == template:
            return row
    return None


def run_admission(output_root: str | Path, *, registry_path: str | Path = DEFAULT_REGISTRY, panel_path: str | Path = DEFAULT_PANEL, trade_state_path: str | Path = DEFAULT_TRADE_STATE, quality_path: str | Path = DEFAULT_QUALITY, execution_path: str | Path = DEFAULT_EXECUTION) -> dict[str, Any]:
    """Generate a fail-closed candidate admission decision and immutable manifest."""
    out = Path(output_root)
    panel_file, state_file, quality_file, execution_file = map(Path, (panel_path, trade_state_path, quality_path, execution_path))
    specs = load_registry(registry_path)
    panel = pd.read_parquet(panel_file)
    qlib_oos = _load_quality(DEFAULT_QLIB_OOS)
    order_evidence = _load_candidate_evidence(DEFAULT_ORDER_RECONCILIATION)
    quality = _load_quality(quality_file)
    execution = _load_quality(execution_file)
    trade_ok, trade_blockers, trade_summary = _trade_state_gate(state_file, quality)
    source_ok, source_blockers, source_summary = _source_capability_gate()
    qlib_ok, qlib_blockers, qlib_summary = _qlib_provider_gate()
    universe_ok, universe_blockers, universe_summary = _pit_universe_gate(panel, quality)
    rows = []
    trial_count = len(specs)
    for spec in specs:
        capacity_ok, capacity_blockers, capacity_summary = _capacity_gate(panel, spec)
        metrics, metric_blockers = _trial_metrics(_execution_record(execution, spec.template), trial_count)
        blockers = trade_blockers + source_blockers + qlib_blockers + universe_blockers + capacity_blockers + metric_blockers
        oos_summary = {"status": "missing"}
        order_summary = {"status": "missing"}
        oos = _oos_evidence(DEFAULT_QLIB_OOS, spec.template)
        order = order_evidence.get(spec.template)
        if not oos:
            blockers.append("canonical_qlib_oos_missing")
        else:
            oos_windows = oos.get("windows", [])
            complete_oos = [w for w in oos_windows if (w.get("oos_metrics") or {}).get("status") == "complete"]
            positive_oos = sum(((w.get("oos_metrics") or {}).get("annual_return") or -1) > 0 for w in complete_oos)
            oos_summary = {"windows": len(oos_windows), "complete_windows": len(complete_oos), "positive_windows": positive_oos}
            if len(complete_oos) < spec.min_oos_windows:
                blockers.append("oos_complete_window_count_below_minimum")
            if positive_oos < spec.min_oos_windows:
                blockers.append("oos_positive_window_count_below_minimum")
        if not order or order.get("status") != "executed":
            blockers.append("backtrader_order_reconciliation_missing")
            order_summary = {"status": "missing"}
        else:
            order_summary = {"status": order.get("status"), "execution_status": order.get("execution_status"), "orders": order.get("orders"), "rejected_orders": order.get("rejected_orders"), "metrics": order.get("metrics", {})}
        rows.append({
            "candidate": asdict(spec),
            "execution": metrics,
            "qlib_oos": oos_summary,
            "order_reconciliation": order_summary,
            "capacity": capacity_summary,
            "data_gates": {"trade_state": trade_summary, "source_capabilities": source_summary, "qlib_provider": qlib_summary, "pit_universe": universe_summary},
            "classification": "blocked" if blockers else "eligible_for_review",
            "admitted": False,
            "blockers": sorted(set(blockers)),
        })
    computed = [row for row in rows if row["execution"].get("status") == "computed"]
    q_values = _bh_fdr([float(row["execution"]["dsr_p_value"]) for row in computed])
    for row, q_value in zip(computed, q_values):
        row["execution"]["bh_fdr_q_value"] = q_value
        if q_value > 0.05:
            row["blockers"].append("dsr_bh_fdr_not_significant")
            row["blockers"] = sorted(set(row["blockers"]))
    oos_plan = _oos_plan(panel, specs)
    quality_result = QualityResult(
        status="BLOCK" if any(row["blockers"] for row in rows) else "PASS",
        checks={"candidates": len(rows), "trade_state_ok": trade_ok, "source_capabilities_ok": source_ok, "pit_universe_ok": universe_ok},
        errors=tuple(sorted(set(item for row in rows for item in row["blockers"]))),
    )
    payload = {
        "schema": "strategy_admission/v1",
        "status": "DATA_BLOCKED" if quality_result.status == "BLOCK" else "REVIEW",
        "candidate_count": len(rows),
        "admitted_count": 0,
        "trial_ledger": {"declared_trials": trial_count, "registry_path": str(Path(registry_path)), "complete": True, "multiple_testing": "deflated_sharpe_plus_bh_fdr"},
        "canonical_oos": oos_plan,
        "candidates": rows,
        "dataset": {"path": str(panel_file), "sha256": sha256_file(panel_file)},
    }
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "admission_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = write_manifest(out, config={"experiment": {"id": "strategy_lab_admission"}, "registry": str(registry_path)}, inputs=[panel_file, state_file, quality_file, execution_file, registry_path, DEFAULT_SOURCE_CAPABILITIES, DEFAULT_SOURCE_CATALOG, DEFAULT_QLIB_MANIFEST, DEFAULT_CLOUD_PIT_MANIFEST], quality=quality_result, root=ROOT, artifacts=[report_path])
    payload["manifest"] = str(out / "experiment_manifest.json")
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload
