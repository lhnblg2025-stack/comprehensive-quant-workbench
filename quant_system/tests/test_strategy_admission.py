from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.strategy_admission import load_registry, run_admission


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    panel = tmp_path / "panel.parquet"
    dates = pd.bdate_range("2024-01-01", periods=300)
    frame = pd.DataFrame([
        {"date": day, "code": f"{code:06d}", "close": 10.0, "amount": 1_000_000.0}
        for day in dates for code in range(1, 5)
    ])
    frame.to_parquet(panel, index=False)
    state = tmp_path / "state.parquet"
    pd.DataFrame({"date": dates.repeat(4), "code": [f"{code:06d}" for _ in dates for code in range(1, 5)], "suspended": False, "limit_up_locked": False, "limit_down_locked": False}).to_parquet(state, index=False)
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"research_status": "DATA_BLOCKED", "coverage": {"membership_rows": 4}}), encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({"templates": {"demo": {"status": "executed", "metrics": {"observations": 260, "sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.1}}}}), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"candidates": [{"candidate_id": "demo-candidate", "template": "demo"}]}), encoding="utf-8")
    return panel, state, quality, execution, registry


def test_registry_requires_unique_predeclared_ids(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"candidates": [{"candidate_id": "x", "template": "a"}, {"candidate_id": "x", "template": "b"}]}), encoding="utf-8")
    try:
        load_registry(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate registry id accepted")


def test_admission_fails_closed_on_non_authoritative_trade_state(tmp_path: Path):
    panel, state, quality, execution, registry = _write_inputs(tmp_path)
    result = run_admission(tmp_path / "out", registry_path=registry, panel_path=panel, trade_state_path=state, quality_path=quality, execution_path=execution)
    assert result["status"] == "DATA_BLOCKED"
    assert result["admitted_count"] == 0
    candidate = result["candidates"][0]
    assert "historical_trade_state_not_authoritative" in candidate["blockers"]
    assert "historical_point_in_time_universe_missing" in candidate["blockers"]
    assert "canonical_qlib_oos_missing" in candidate["blockers"]
    assert candidate["execution"]["dsr_p_value"] > 0.05
    assert candidate["execution"]["bh_fdr_q_value"] > 0.05
    assert "dsr_bh_fdr_not_significant" in candidate["blockers"]
    assert "capacity_impact_model_missing" in candidate["blockers"]
    assert result["canonical_oos"]["status"] == "planned_not_run"
    # 300 synthetic dates cannot meet the 504/126/126 OOS protocol; the gate
    # must keep the plan empty instead of shortening windows for convenience.
    assert result["canonical_oos"]["windows"] == []
    assert (tmp_path / "out" / "experiment_manifest.json").is_file()
