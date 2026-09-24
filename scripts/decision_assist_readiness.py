#!/usr/bin/env python3
"""Generate scoped readiness for analysis and decision assistance only."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exists(*parts: str) -> bool:
    return ROOT.joinpath(*parts).exists()


def _latest(pattern: str) -> str | None:
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(files[-1].relative_to(ROOT)) if files else None


def _git_count(path: Path) -> int | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return len((result.stdout or "").splitlines())


def build_snapshot() -> dict:
    evidence = {
        "latest_review_json": _latest("generated/review_*.json"),
        "latest_manifest": _latest("generated/runs/*/manifest.json"),
        "latest_delivery_receipt": _latest("generated/delivery_receipts/review_*.json"),
        "latest_wave_status": _latest("generated/wave_status/latest.json"),
        "dirty_inventory": _exists("generated", "workspace_hygiene", "dirty_inventory.json"),
        "quant_system_dirty_report": _exists("generated", "workspace_hygiene", "quant_system_repo.json"),
        "root_dirty_count": _git_count(ROOT),
        "quant_system_dirty_count": _git_count(ROOT / "quant_system"),
    }
    required = {
        "review_generation": bool(evidence["latest_review_json"]),
        "run_manifest": bool(evidence["latest_manifest"]),
        "delivery_receipt_or_skip_receipt": bool(evidence["latest_delivery_receipt"]),
        "wave_status": bool(evidence["latest_wave_status"]),
        "dirty_inventory": evidence["dirty_inventory"],
        "quant_system_dirty_report": evidence["quant_system_dirty_report"],
        "decision_chain_code": _exists("scripts", "daily_review_chain.py") and _exists("scripts", "fusion_decision.py"),
        "prediction_observer_code": _exists("scripts", "prediction_verify.py"),
    }
    blockers = [name for name, ok in required.items() if not ok]
    observations = [
        {"id": "next_market_prediction_verification", "why_not_blocking": "requires a future market close", "validator": "python3 scripts/prediction_verify.py"},
        {"id": "cron_stability_samples", "why_not_blocking": "requires scheduled runs over time", "validator": "generated/runs/*/manifest.json plus cron logs"},
        {"id": "decision_quality_trend", "why_not_blocking": "requires repeated decisions and outcomes", "validator": "prediction_verify plus calibration report"},
    ]
    excluded = ["broker adapter", "real order placement", "real fill reports", "live limits/circuit breakers", "broker audit trail"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "analysis_decision_assist_only",
        "status": "ready" if not blockers else "blocked",
        "summary": {"immediate_engineering_blockers": len(blockers), "run_observation_items": len(observations), "excluded_live_items": len(excluded)},
        "required_evidence": required,
        "evidence": evidence,
        "immediate_engineering_blockers": blockers,
        "run_observation_items": observations,
        "excluded_live_items": excluded,
        "decision": "analysis/decision assistance is deliverable now" if not blockers else "fix listed blockers first",
    }


def main() -> int:
    snapshot = build_snapshot()
    output_dir = ROOT / "generated" / "decision_assist_readiness"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "latest.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0 if snapshot["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
