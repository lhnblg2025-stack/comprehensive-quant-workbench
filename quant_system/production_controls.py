"""Production control-plane contracts for PIT, OMS, ledger and admission.

This module contains deterministic checks and evidence builders. It never treats a
local cache, broker stub, or paper ledger as production evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd

SCHEMA = "production-control/v1"
CST = timezone.utc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


@dataclass(frozen=True)
class PITRelease:
    release_id: str
    as_of: str
    provider: str
    source_files: tuple[dict[str, Any], ...]
    trade_state_authoritative: bool
    instrument_master_authoritative: bool
    corporate_actions_authoritative: bool
    independent_attestation: str | None = None
    schema: str = "pit-release/v1"

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        if not self.release_id or not self.provider:
            errors.append("release_id_and_provider_required")
        try:
            pd.Timestamp(self.as_of)
        except Exception:
            errors.append("invalid_release_as_of")
        if not self.source_files:
            errors.append("source_files_required")
        for item in self.source_files:
            if not item.get("path") or not item.get("sha256"):
                errors.append("source_file_hash_missing")
        if not self.trade_state_authoritative:
            errors.append("authoritative_trade_state_missing")
        if not self.instrument_master_authoritative:
            errors.append("authoritative_instrument_master_missing")
        if not self.corporate_actions_authoritative:
            errors.append("authoritative_corporate_actions_missing")
        if not self.independent_attestation:
            errors.append("independent_release_attestation_missing")
        return {"status": "PASS" if not errors else "BLOCK", "errors": _unique(errors), "release_id": self.release_id}


def build_pit_release(release_id: str, as_of: str, provider: str, paths: Iterable[str | Path], **flags: Any) -> dict[str, Any]:
    files = tuple({"path": str(Path(path)), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size} for path in paths if Path(path).is_file())
    release = PITRelease(release_id=release_id, as_of=as_of, provider=provider, source_files=files,
                        trade_state_authoritative=bool(flags.get("trade_state_authoritative")),
                        instrument_master_authoritative=bool(flags.get("instrument_master_authoritative")),
                        corporate_actions_authoritative=bool(flags.get("corporate_actions_authoritative")),
                        independent_attestation=flags.get("independent_attestation"))
    validation = release.validate()
    payload = {"schema": release.schema, "release": asdict(release), "validation": validation, "created_at": _now()}
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode()
    payload["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def freeze_research_release(output: str | Path, paths: Iterable[str | Path], *, as_of: str, provider: str = "local-versioned-parquet", limitations: Iterable[str] = ()) -> dict[str, Any]:
    """Freeze research inputs without granting a live-trading attestation.

    The release is content addressed and immutable. This is the production boundary
    needed by factor research: every result can be replayed, while live execution
    remains blocked unless a separate authoritative PIT/OMS preflight passes.
    """
    target = Path(output)
    files = tuple(sorted(({"path": str(Path(path).resolve()), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size} for path in paths if Path(path).is_file()), key=lambda item: item["path"]))
    if not files:
        raise ValueError("research_release_inputs_missing")
    release_id = "research-" + hashlib.sha256(json.dumps({"files": files, "as_of": as_of, "provider": provider}, sort_keys=True).encode()).hexdigest()[:24]
    release = {"schema": "research-release/v1", "release_id": release_id, "as_of": str(pd.Timestamp(as_of).date()), "provider": provider, "status": "FROZEN", "source_files": files, "limitations": _unique(limitations), "live_execution": {"status": "BLOCK", "reason": "research_release_is_not_trading_attestation"}}
    release["evidence_sha256"] = hashlib.sha256(json.dumps(release, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("evidence_sha256") != release["evidence_sha256"]:
            raise ValueError(f"research_release_immutable:{target}")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(release, ensure_ascii=True, indent=2), encoding="utf-8")
    return release


def verify_research_release(release: Mapping[str, Any]) -> dict[str, Any]:
    """Verify release digest and all source files before a research run consumes it."""
    errors: list[str] = []
    payload = dict(release)
    supplied = payload.pop("evidence_sha256", None)
    expected = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode()).hexdigest()
    if supplied != expected: errors.append("release_digest_mismatch")
    if payload.get("status") != "FROZEN": errors.append("release_not_frozen")
    for item in payload.get("source_files", ()):
        path = Path(item["path"])
        if not path.is_file(): errors.append(f"missing_source:{path}")
        elif sha256_file(path) != item.get("sha256"): errors.append(f"source_hash_mismatch:{path}")
    return {"status": "PASS" if not errors else "BLOCK", "release_id": payload.get("release_id"), "errors": errors}


def validate_trade_state_release(frame: pd.DataFrame, *, release: Mapping[str, Any], expected_pairs: set[tuple[str, pd.Timestamp]] | None = None) -> dict[str, Any]:
    from .trade_state_contract import validate_trade_state
    base = validate_trade_state(frame, expected_pairs=expected_pairs)
    errors = list(base.get("errors", []))
    release_data = dict(release.get("release") or release)
    if not release_data.get("release_id"):
        errors.append("pit_release_id_missing")
    if (release.get("validation") or {}).get("status") == "BLOCK":
        errors.append("pit_release_validation_blocked")
    if not release_data.get("trade_state_authoritative"):
        errors.append("pit_release_trade_state_not_authoritative")
    if release_data.get("schema") not in {"pit-release/v1", "production-pit-release/v1"}:
        errors.append("pit_release_schema_invalid")
    return {**base, "status": "PASS" if not errors else "BLOCK", "errors": _unique(errors), "release_id": release_data.get("release_id")}


REQUIRED_ACTION_COLUMNS = {"code", "ex_date", "action_type", "source_document_id", "source_as_of"}


def validate_corporate_actions(actions: pd.DataFrame, *, release_as_of: str, expected_codes: set[str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_ACTION_COLUMNS - set(actions.columns))
    if missing:
        errors.append("missing_columns:" + ",".join(missing))
        return {"status": "BLOCK", "errors": errors, "warnings": warnings, "rows": int(len(actions))}
    data = actions.copy()
    data["code"] = data["code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    data["ex_date"] = pd.to_datetime(data["ex_date"], errors="coerce").dt.normalize()
    data["source_as_of"] = pd.to_datetime(data["source_as_of"], errors="coerce").dt.normalize()
    cutoff = pd.Timestamp(release_as_of).normalize()
    errors.extend(["invalid_ex_date"] if data["ex_date"].isna().any() else [])
    errors.extend(["invalid_source_as_of"] if data["source_as_of"].isna().any() else [])
    if (data["source_as_of"] > cutoff).any():
        errors.append("source_observed_after_release_as_of")
    if data.duplicated(["code", "ex_date", "action_type"]).any():
        errors.append("duplicate_corporate_actions")
    if data[["code", "ex_date", "action_type"]].duplicated().any():
        errors.append("duplicate_action_key")
    source = data["source_document_id"].astype(str)
    if source.str.strip().eq("").any() or source.str.contains("unverified|proxy|derived|inferred", case=False, regex=True).any():
        errors.append("non_authoritative_action_source")
    if expected_codes is not None:
        missing_codes = expected_codes - set(data["code"].dropna())
        if missing_codes:
            warnings.append(f"codes_without_actions:{len(missing_codes)}")
    allowed = {"dividend", "split", "rights_issue", "conversion", "delisting"}
    unknown = sorted(set(data["action_type"].astype(str)) - allowed)
    if unknown:
        errors.append("unknown_action_type:" + ",".join(unknown))
    return {"status": "PASS" if not errors else "BLOCK", "errors": _unique(errors), "warnings": _unique(warnings), "rows": int(len(data)), "symbols": int(data["code"].nunique()), "release_as_of": str(cutoff.date())}


def apply_pit_corporate_actions(panel: pd.DataFrame, actions: pd.DataFrame, *, as_of: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only actions known by the release date; reject future observations."""
    report = validate_corporate_actions(actions, release_as_of=as_of)
    if report["status"] != "PASS":
        raise ValueError("corporate_actions_gate_blocked:" + ",".join(report["errors"]))
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    action = actions.copy()
    action["ex_date"] = pd.to_datetime(action["ex_date"], errors="coerce").dt.normalize()
    action = action[action["ex_date"] <= pd.Timestamp(as_of).normalize()]
    applied = 0
    for row in action.sort_values(["ex_date", "code"]).to_dict("records"):
        mask = (data["code"] == str(row["code"]).zfill(6)) & (data["date"] >= row["ex_date"])
        ratio = float(row.get("ratio") or 1.0)
        if row["action_type"] in {"split", "rights_issue", "conversion"} and ratio > 0:
            for col in ("raw_open", "raw_high", "raw_low", "raw_close"):
                if col in data:
                    data.loc[mask, col] = pd.to_numeric(data.loc[mask, col], errors="coerce") / ratio
        if row["action_type"] == "dividend" and "raw_close" in data:
            data.loc[mask, "raw_close"] = pd.to_numeric(data.loc[mask, "raw_close"], errors="coerce") + float(row.get("cash_per_share") or 0.0)
        applied += int(mask.sum() > 0)
    return data, {"status": "PASS", "validated": report, "applied_actions": applied, "as_of": str(pd.Timestamp(as_of).date())}


class OMS(Protocol):
    name: str
    def health(self) -> Mapping[str, Any]: ...
    def submit(self, order: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def cancel(self, order_id: str) -> Mapping[str, Any]: ...
    def sync(self) -> Mapping[str, Any]: ...


@dataclass
class ProductionPreflight:
    pit_release: Mapping[str, Any]
    oms_health: Mapping[str, Any]
    independent_ledger: bool
    corporate_actions: Mapping[str, Any]
    approval: Mapping[str, Any]

    def evaluate(self) -> dict[str, Any]:
        blockers: list[str] = []
        if (self.pit_release.get("validation") or {}).get("status") != "PASS": blockers.append("pit_release_blocked")
        if not self.oms_health.get("connected") or not self.oms_health.get("authenticated") or not self.oms_health.get("order_callbacks"):
            blockers.append("oms_not_production_ready")
        if not self.independent_ledger: blockers.append("independent_ledger_missing")
        if self.corporate_actions.get("status") != "PASS": blockers.append("corporate_actions_blocked")
        if self.approval.get("status") != "APPROVED": blockers.append("strategy_not_approved")
        return {"schema": "production-preflight/v1", "status": "PASS" if not blockers else "BLOCK", "blockers": _unique(blockers), "checked_at": _now()}


class UnavailableOMS:
    """Explicit fail-closed OMS used until a licensed broker adapter is installed."""
    name = "unavailable"
    def health(self) -> Mapping[str, Any]:
        return {"connected": False, "authenticated": False, "order_callbacks": False, "reason": "licensed_broker_adapter_not_configured"}
    def submit(self, order: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"status": "BLOCKED", "error": "production_oms_not_configured"}
    def cancel(self, order_id: str) -> Mapping[str, Any]:
        return {"status": "BLOCKED", "error": "production_oms_not_configured"}
    def sync(self) -> Mapping[str, Any]:
        return {"status": "BLOCKED", "error": "production_oms_not_configured"}


def append_ledger_event(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append a tamper-evident ledger event with a chained digest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if target.is_file():
        last = target.read_text(encoding="utf-8").splitlines()[-1:]
        if last:
            previous = str(json.loads(last[0]).get("event_sha256") or previous)
    payload = {"schema": "independent-ledger-event/v1", "previous_sha256": previous,
               "event": dict(event), "recorded_at": _now()}
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode()
    payload["event_sha256"] = hashlib.sha256(canonical).hexdigest()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str) + "\\n")
    return payload


def verify_ledger_chain(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    errors: list[str] = []
    previous = "0" * 64
    rows = []
    if not target.is_file():
        return {"schema": "independent-ledger-chain/v1", "status": "BLOCK", "errors": ["ledger_missing"], "events": 0}
    for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line); supplied = row.pop("event_sha256", "")
            if row.get("previous_sha256") != previous: errors.append(f"chain_break:{index}")
            canonical = json.dumps(row, ensure_ascii=True, sort_keys=True, default=str).encode()
            if hashlib.sha256(canonical).hexdigest() != supplied: errors.append(f"digest_mismatch:{index}")
            previous = supplied; rows.append(row)
        except (ValueError, TypeError):
            errors.append(f"invalid_json:{index}")
    return {"schema": "independent-ledger-chain/v1", "status": "PASS" if not errors and rows else "BLOCK", "errors": _unique(errors), "events": len(rows), "last_sha256": previous}


def build_three_way_reconciliation(strategy_rows: pd.DataFrame, broker_rows: pd.DataFrame, ledger_rows: pd.DataFrame) -> dict[str, Any]:
    required = {"symbol", "shares"}
    errors = []
    for name, frame in (("strategy", strategy_rows), ("broker", broker_rows), ("ledger", ledger_rows)):
        if not required.issubset(frame.columns): errors.append(f"{name}_columns_missing")
    if errors:
        return {"schema": "three-way-reconciliation/v1", "status": "BLOCK", "errors": errors}
    def aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = frame.copy(); out["symbol"] = out["symbol"].astype(str).str.zfill(6); out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0)
        return out.groupby("symbol", as_index=False)["shares"].sum().rename(columns={"shares": prefix})
    merged = aggregate(strategy_rows, "strategy").merge(aggregate(broker_rows, "broker"), on="symbol", how="outer").merge(aggregate(ledger_rows, "ledger"), on="symbol", how="outer").fillna(0)
    for col in ("strategy", "broker", "ledger"): merged[col] = merged[col].astype(float)
    merged["broker_delta"] = merged["broker"] - merged["strategy"]; merged["ledger_delta"] = merged["ledger"] - merged["strategy"]
    merged["status"] = merged.apply(lambda r: "matched" if r.broker_delta == 0 and r.ledger_delta == 0 else "mismatch", axis=1)
    mismatches = merged[merged["status"] != "matched"]
    return {"schema": "three-way-reconciliation/v1", "status": "PASS" if mismatches.empty else "BLOCK", "rows": json.loads(merged.to_json(orient="records")), "summary": {"symbols": len(merged), "matched": int(merged.status.eq("matched").sum()), "mismatched": int(len(mismatches))}, "checked_at": _now()}


APPROVAL_STATES = {"DRAFT", "REVIEW", "APPROVED", "REJECTED", "SUSPENDED"}


def transition_approval(current: str, target: str, *, actor: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    if current not in APPROVAL_STATES or target not in APPROVAL_STATES: raise ValueError("invalid_approval_state")
    allowed = {"DRAFT": {"REVIEW"}, "REVIEW": {"APPROVED", "REJECTED"}, "APPROVED": {"SUSPENDED"}, "REJECTED": {"DRAFT"}, "SUSPENDED": {"REVIEW"}}
    if target not in allowed[current]: raise ValueError(f"invalid_approval_transition:{current}->{target}")
    if not actor or not evidence: raise ValueError("approval_actor_and_evidence_required")
    return {"schema": "strategy-approval-event/v1", "from": current, "to": target, "actor": actor, "evidence": dict(evidence), "at": _now()}


def build_research_evidence(*, release: Mapping[str, Any], quality: Mapping[str, Any], pit: Mapping[str, Any], corporate_actions: Mapping[str, Any], factors: Iterable[str]) -> dict[str, Any]:
    """Build the auditable evidence package for a research factor run."""
    blockers: list[str] = []
    if release.get("status") != "FROZEN":
        blockers.append("data_release_not_frozen")
    if quality.get("status") == "BLOCK":
        blockers.append("market_panel_quality_blocked")
    if pit and pit.get("status") not in {"PASS", "not_configured"}:
        blockers.append("pit_validation_blocked")
    if corporate_actions.get("status") == "BLOCK":
        blockers.append("corporate_actions_blocked")
    return {
        "schema": "strategy-research-evidence/v1",
        "status": "PASS" if not blockers else "BLOCK",
        "release_id": release.get("release_id"),
        "release_evidence_sha256": release.get("evidence_sha256"),
        "factor_names": list(factors),
        "quality": dict(quality),
        "pit": dict(pit),
        "corporate_actions": dict(corporate_actions),
        "blockers": blockers,
        "live_execution": {"status": "BLOCK", "reason": "no_qmt_or_broker_adapter_requested"},
    }
