from __future__ import annotations

import json

import pytest

from scripts import after_close_orchestrator as orchestrator


def test_run_stage_records_spawn_failure():
    result = orchestrator._run_stage("missing", ["/definitely/missing/runtime"], 1)
    assert result["status"] == "failed"
    assert result["returncode"] is None


def test_atomic_status_write_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "GEN", tmp_path / "generated")
    path = orchestrator._status_dir() / "after_close.json"
    orchestrator._write_status({"run_id": "r1", "status": "running"})
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "r1"
    assert not list(path.parent.glob("*.tmp"))


def test_explicit_non_trading_day_is_rejected(monkeypatch):
    monkeypatch.setattr(orchestrator, "_target_date", lambda value: "2026-01-01")
    monkeypatch.setattr("quant_system.market_clock.is_trading_day", lambda value: False)
    monkeypatch.setattr(orchestrator.sys, "argv", ["after_close_orchestrator.py", "--date", "2026-01-01"])
    with pytest.raises(SystemExit, match="not a trading day"):
        orchestrator.main()


def test_run_context_reaches_every_child(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "GEN", tmp_path / "generated")
    monkeypatch.setattr(orchestrator, "_target_date", lambda value: "2026-01-05")
    monkeypatch.setattr("quant_system.market_clock.is_trading_day", lambda value: True)
    seen = []

    def fake_stage(name, command, timeout, *, env=None):
        seen.append((name, env["QUANT_RUN_ID"], env["QUANT_AS_OF"]))
        return {"name": name, "status": "ok", "returncode": 0,
                "duration_seconds": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(orchestrator, "_run_stage", fake_stage)
    monkeypatch.setattr(
        orchestrator.sys,
        "argv",
        ["after_close_orchestrator.py", "--date", "2026-01-05", "--run-id", "run-test-1", "--skip-weekly"],
    )
    assert orchestrator.main() == 0
    assert seen
    assert all(run_id == "run-test-1" and as_of == "2026-01-05" for _, run_id, as_of in seen)
    manifest = json.loads((tmp_path / "generated" / "runs" / "run-test-1" / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["current_stage"] is None
