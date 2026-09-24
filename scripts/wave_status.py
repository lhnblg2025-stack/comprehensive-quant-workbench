#!/usr/bin/env python3
"""Generate evidence-based analysis-assist readiness for Wave1/2/3.

Scope is research/paper/simulation decision assistance only; broker-live is
excluded and never inferred as complete from code presence alone."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CST = timezone.utc


def exists(*parts: str) -> bool:
    return (ROOT.joinpath(*parts)).exists()


def latest(pattern: str) -> str | None:
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(files[-1].relative_to(ROOT)) if files else None


def git_dirty() -> int:
    out = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True, check=False).stdout
    return len(out.splitlines())


def has_recent_manifest() -> bool:
    import time
    files = list(ROOT.glob("generated/runs/*/manifest.json"))
    return bool(files) and max(p.stat().st_mtime for p in files) >= time.time() - 2 * 86400


def main() -> int:
    status = {
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "source": "scripts/wave_status.py",
        "scope": "analysis_decision_assist_only",
        "excluded_live_scope": ["broker adapter", "real order placement", "real fill reports", "live limits/circuit breakers", "broker audit trail"],
        "waves": {
            "wave1": {
                "status": "ready_with_observation",
                "done": ["review_generation", "review_manifest_receipt_code", "health_degraded_visibility", "local_wrapper_regression"],
                "run_observation": ["prediction_verified_after_next_close", "cron_stability_samples", "decision_quality_trend"],
                "evidence": {
                    "manifest_code": exists("scripts", "daily_review_chain.py"),
                    "latest_review": latest("generated/review_*.json"),
                    "latest_health": latest("generated/data_health_*.json"),
                    "receipt_dir": exists("generated", "delivery_receipts"),
                    "recent_manifest": has_recent_manifest(),
                    "review_wrapper": exists("scripts", "run_review_cron.sh"),
                },
            },
            "wave2": {
                "status": "ready_with_observation",
                "done": ["oos_rule_code_and_tests", "engine_adapter_code", "backtest_cross_check_code", "quant_time_contract", "pytest_collection", "targeted_regression"],
                "run_observation": ["engine_rate_trend", "multi_origin_oos", "unified_tradability_backtest", "paper_order_fill_chain"],
                "evidence": {
                    "oos_rule": exists("quant_system", "ic_factors", "oos_pool_rule.py"),
                    "oos_report": exists("generated", "ic_report", "IC_OOS_REPORT.json"),
                    "backtest_cross_check": exists("scripts", "backtest_cross_check.py"),
                    "dirty_files": git_dirty(),
                },
            },
            "wave3": {
                "status": "ready_with_observation",
                "done": ["journal_rag_module", "three_endpoint_panel_code", "difficulty_cost_router", "workspace_hygiene", "workflow_sop_documents"],
                "run_observation": ["dsh_codex_claude_subagents", "three_endpoint_history_slo", "token_cost_ledger", "production_cross_review"],
                "evidence": {
                    "journal_rag": exists("quant_system", "analysis_core", "trading_journal_rag.py"),
                    "endpoint_panel": exists("scripts", "three_endpoint_panel.py"),
                    "endpoint_latest": latest("generated/endpoint_health/latest.json"),
                },
            },
        },
    }
    out_dir = ROOT / "generated" / "wave_status"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
