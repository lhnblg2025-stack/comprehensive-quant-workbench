"""Canonical backtest service shared by Web/API and research callers.

Qlib owns factor/model research and walk-forward evaluation. Backtrader owns the
canonical order-level execution semantics. Other engines may sweep or cross-check,
but they must not emit admission decisions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .backtest_protocol import BacktestRequest, UnifiedBacktestResult
from .backtrader_engine import run_backtrader, targets_from_signal

ROOT = Path(__file__).resolve().parents[1]
QUALITY_PATH = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "pit_data_quality_report.json"
REQUIRED_EXECUTION_COLUMNS = {"date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "volume"}
REQUIRED_TRADE_STATE_COLUMNS = {"date", "code", "suspended", "limit_up_locked", "limit_down_locked"}


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def data_gate(panel: pd.DataFrame, *, require_production: bool = False,
              trade_state: pd.DataFrame | None = None, quality_path: Path | None = None) -> dict[str, Any]:
    quality = _json(quality_path or QUALITY_PATH)
    blockers: list[str] = []
    warnings: list[str] = []
    status = str(quality.get("production_status") or quality.get("status") or "UNVERIFIED")
    missing = sorted(REQUIRED_EXECUTION_COLUMNS - set(panel.columns))
    if missing:
        blockers.append("panel_missing:" + ",".join(missing))
    if panel.empty:
        blockers.append("panel_empty")
    elif {"date", "code"}.issubset(panel.columns):
        duplicate_count = int(panel.duplicated(["date", "code"]).sum())
        if duplicate_count:
            blockers.append(f"duplicate_date_code:{duplicate_count}")
        invalid_dates = int(pd.to_datetime(panel["date"], errors="coerce").isna().sum())
        if invalid_dates:
            blockers.append(f"invalid_dates:{invalid_dates}")
    numeric_required = sorted(REQUIRED_EXECUTION_COLUMNS - {"date", "code"})
    for column in numeric_required:
        if column in panel:
            values = pd.to_numeric(panel[column], errors="coerce")
            invalid = int(values.isna().sum())
            if invalid:
                blockers.append(f"invalid_{column}:{invalid}")
            if column != "volume" and int((values <= 0).sum()):
                blockers.append(f"nonpositive_{column}:{int((values <= 0).sum())}")
    state_ready = trade_state is not None and REQUIRED_TRADE_STATE_COLUMNS.issubset(set(trade_state.columns))
    if not state_ready:
        warnings.append("historical_trade_state_missing_or_incomplete")
    if status == "DATA_BLOCKED":
        blockers.extend(quality.get("production_blockers") or quality.get("errors") or ["data_quality_blocked"])
    elif status == "UNVERIFIED":
        warnings.append("quality_report_unverified")
    if require_production:
        if not quality:
            blockers.append("quality_report_missing")
        if not state_ready:
            blockers.append("authoritative_trade_state_required")
        if status not in {"PASS", "PRODUCTION_READY"}:
            blockers.append(f"production_quality_status:{status}")
    return {
        "status": "BLOCK" if blockers else "PASS",
        "classification": "production_eligible" if require_production and not blockers else "research_only" if warnings or blockers else "research_eligible",
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "quality_status": status,
        "trade_state_available": state_ready,
        "coverage": quality.get("coverage", {}),
        "panel_rows": int(len(panel)),
        "panel_symbols": int(panel.code.nunique()) if "code" in panel else 0,
    }


def spec_hash(params: Mapping[str, Any]) -> str:
    payload = json.dumps(params, ensure_ascii=True, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_targets(panel: pd.DataFrame, request: BacktestRequest,
                   explicit_targets: Mapping[object, Mapping[str, float]] | None) -> Mapping[object, Mapping[str, float]]:
    strategy = request.strategy
    if strategy.kind == "target_weights":
        if explicit_targets is None:
            raise ValueError("target_weights_required")
        return explicit_targets
    assert strategy.signal is not None
    if strategy.signal not in panel.columns:
        raise ValueError(f"signal_not_found:{strategy.signal}")
    return targets_from_signal(panel, strategy.signal, direction=int(strategy.direction),
                               quantile=float(strategy.quantile),
                               rebalance_sessions=int(strategy.rebalance_sessions))


def execute(request: BacktestRequest | Mapping[str, Any], panel: pd.DataFrame, *,
            trade_state: pd.DataFrame | None = None,
            targets: Mapping[object, Mapping[str, float]] | None = None,
            quality_path: Path | None = None) -> dict[str, Any]:
    """Validate and execute one canonical request through Backtrader."""
    req = request if isinstance(request, BacktestRequest) else BacktestRequest.from_dict(request)
    req.validate()
    production = req.mode in {"paper", "production"}
    gate = data_gate(panel, require_production=production, trade_state=trade_state, quality_path=quality_path)
    if production and gate["status"] != "PASS":
        return {"ok": False, "status": "DATA_BLOCKED", "canonical": True, "schema": req.schema,
                "gate": gate, "spec": req.to_dict(), "spec_hash": spec_hash(req.to_dict()),
                "error": "production_data_gate_blocked"}
    try:
        built_targets = _build_targets(panel, req, targets)
    except ValueError as exc:
        return {"ok": False, "status": "INVALID_REQUEST", "canonical": True, "schema": req.schema,
                "gate": gate, "spec": req.to_dict(), "spec_hash": spec_hash(req.to_dict()), "error": str(exc)}
    costs, risk = req.costs, req.risk
    result, equity, trades = run_backtrader(
        panel, built_targets, capital=float(req.initial_capital), trade_state=trade_state,
        commission_bps=float(costs.commission_bps), stamp_duty_bps=float(costs.stamp_duty_bps),
        transfer_fee_bps=float(costs.transfer_fee_bps), slippage_bps=float(costs.slippage_bps),
        min_commission=float(costs.min_commission), max_adv_participation=float(costs.max_adv_participation),
        max_position_weight=float(risk.max_position_weight), max_gross_exposure=float(risk.max_gross_exposure),
        max_names=int(risk.max_names), lot_size=int(risk.lot_size),
    )
    request_dict = req.to_dict()
    digest = spec_hash(request_dict)
    result.metadata.update({"canonical": True, "schema": req.schema, "spec_hash": digest,
                            "data_gate": gate, "strategy_family": req.strategy.family,
                            "research_framework": "microsoft_qlib", "execution_engine": "backtrader"})
    return {"ok": True, "status": result.status, "engine": result.engine, "canonical": True,
            "schema": req.schema, "spec": request_dict, "spec_hash": digest, "gate": gate,
            "result": result, "equity": equity, "trades": trades}


def run_canonical(panel: pd.DataFrame, signal: str, *, quantile: float = .2,
                  rebalance_sessions: int = 5, capital: float = 1_000_000.0,
                  production: bool = False, trade_state: pd.DataFrame | None = None, **kwargs) -> dict[str, Any]:
    """Compatibility wrapper; all execution is delegated to :func:`execute`."""
    family = str(kwargs.pop("family", "cross_section"))
    request = BacktestRequest.from_dict({
        "strategy": {"kind": "cross_sectional_signal", "family": family, "signal": signal,
                     "direction": kwargs.pop("direction", 1), "quantile": quantile,
                     "rebalance_sessions": rebalance_sessions},
        "costs": {name: kwargs.pop(name) for name in list(kwargs) if name in {
            "commission_bps", "stamp_duty_bps", "transfer_fee_bps", "slippage_bps",
            "min_commission", "max_adv_participation"}},
        "risk": {name: kwargs.pop(name) for name in list(kwargs) if name in {
            "max_position_weight", "max_gross_exposure", "max_names", "lot_size"}},
        "initial_capital": capital,
        "mode": "production" if production else "research",
    })
    if kwargs:
        raise ValueError("unsupported_canonical_options:" + ",".join(sorted(kwargs)))
    return execute(request, panel, trade_state=trade_state)


def serializable_run(run: dict[str, Any]) -> dict[str, Any]:
    if not run.get("ok"):
        return dict(run)
    result: UnifiedBacktestResult = run["result"]
    equity = run["equity"].copy()
    equity["date"] = equity["date"].astype(str)
    trades = run["trades"].copy()
    if "date" in trades:
        trades["date"] = trades["date"].astype(str)
    return {"ok": True, "status": run["status"], "engine": run["engine"],
            "canonical": run["canonical"], "schema": run["schema"], "spec": run["spec"],
            "spec_hash": run["spec_hash"], "gate": run["gate"], "metrics": result.to_dict(),
            "equity_curve": equity.to_dict("records"), "trades": trades.to_dict("records")}
