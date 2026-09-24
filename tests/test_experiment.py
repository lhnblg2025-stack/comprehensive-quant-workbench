from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.experiment import Experiment


def test_experiment_produces_immutable_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("research.experiment.RUNS", tmp_path / "runs")
    exp = Experiment(name="ic_oos", as_of="2026-08-30", parameters={"n": 150, "seed": 42}, strategy_ref={"name": "factor", "parameters": {"x": 1}}, run_id="run-1")
    manifest = exp.start(inputs=[__file__])
    assert manifest["schema"] == "quant-experiment/v2"
    assert manifest["run_id"] == "run-1"
    assert manifest["strategy_params_sha256"]
    assert exp.manifest_path.is_file()
    exp.checkpoint("step 1 done")
    result = exp.finalize(status="ok")
    payload = json.loads(exp.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["checkpoints"][0]["message"] == "step 1 done"
    assert payload["finished_at"]
    assert not list((tmp_path / "runs").glob("*.tmp"))


def test_experiment_requires_start_before_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("research.experiment.RUNS", tmp_path / "runs")
    exp = Experiment(name="x", as_of="2026-08-30", run_id="run-2")
    with pytest.raises(RuntimeError):
        exp.finalize()


def test_params_hash_is_stable_and_sensitive_to_order(tmp_path, monkeypatch):
    monkeypatch.setattr("research.experiment.RUNS", tmp_path / "runs")
    a = Experiment(name="x", as_of="2026-08-30", strategy_ref={"parameters": {"a": 1, "b": 2}}, run_id="r-a")
    b = Experiment(name="x", as_of="2026-08-30", strategy_ref={"parameters": {"b": 2, "a": 1}}, run_id="r-b")
    c = Experiment(name="x", as_of="2026-08-30", strategy_ref={"parameters": {"a": 1, "b": 3}}, run_id="r-c")
    assert a._params()["strategy_params_sha256"] == b._params()["strategy_params_sha256"]
    assert a._params()["strategy_params_sha256"] != c._params()["strategy_params_sha256"]
