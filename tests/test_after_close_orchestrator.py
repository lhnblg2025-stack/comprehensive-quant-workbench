from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import after_close_orchestrator as orchestrator


def test_orchestrator_stage_records_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "GEN", tmp_path / "generated")
    result = orchestrator._run_stage("测试阶段", ["/bin/sh", "-c", "exit 3"], 10)
    assert result["status"] == "failed"
    assert result["returncode"] == 3


def test_orchestrator_has_explicit_manifest_schema():
    source = Path(orchestrator.__file__).read_text(encoding="utf-8")
    assert '"schema": "after_close_orchestrator/v3"' in source
    assert '"external_delivery_requested"' in source
    assert '"blocked_downstream"' in source
    assert '"status_file"' in source


def test_target_date_rejects_invalid_value():
    with pytest.raises(SystemExit, match="invalid --date"):
        orchestrator._target_date("2026-02-30")


def test_orchestrator_blocks_dependent_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "GEN", tmp_path / "generated")
    calls = []

    def fake_stage(name, command, timeout, **kwargs):
        calls.append(name)
        return {"name": name, "status": "failed" if name == "统一决策复盘" else "ok",
                "returncode": 3 if name == "统一决策复盘" else 0,
                "duration_seconds": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(orchestrator, "_run_stage", fake_stage)
    monkeypatch.setattr(orchestrator, "_target_date", lambda value: "2026-01-05")
    monkeypatch.setattr("quant_system.market_clock.is_trading_day", lambda value: True)
    monkeypatch.setattr(orchestrator.sys, "argv", ["after_close_orchestrator.py"])
    assert orchestrator.main() == 2
    assert calls == ["统一决策复盘"]
    status = json.loads((tmp_path / "generated" / "runtime_status" / "after_close.json").read_text())
    assert status["status"] == "blocked"
    assert "统一决策快照" in status["blocked_downstream"]
