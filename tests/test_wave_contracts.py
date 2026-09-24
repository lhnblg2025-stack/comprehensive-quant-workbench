from datetime import date

import pytest

from scripts.quant_time_contract import TimeWindow, assert_feature_asof, split_metadata
from scripts.task_difficulty_router import route_task
from scripts.three_endpoint_panel import _state
from scripts.decision_assist_readiness import build_snapshot


def test_endpoint_health_states():
    assert _state([True, True]) == "healthy"
    assert _state([True, False]) == "degraded"
    assert _state([None, None]) == "unknown"
    assert _state([False, False]) == "failed"


def test_time_contract_accepts_non_overlapping_windows():
    meta = split_metadata(
        TimeWindow(date(2020, 1, 1), date(2023, 12, 31), date(2024, 2, 1), date(2024, 6, 30), 5),
        feature_asof=date(2024, 1, 31), label_end=date(2024, 1, 20), code_sha="test",
    )
    assert meta["embargo_days"] == 5
    assert meta["oos_start"] == "2024-02-01"


def test_time_contract_rejects_future_feature():
    with pytest.raises(ValueError, match="future_feature"):
        assert_feature_asof(feature_asof=date(2024, 2, 2), signal_date=date(2024, 2, 1))


def test_time_contract_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        TimeWindow(date(2020, 1, 1), date(2024, 2, 1), date(2024, 2, 1), date(2024, 3, 1)).validate()


def test_task_router_escalates_execution_and_security():
    assert route_task(files=1)["level"] == "L1"
    assert route_task(files=2)["level"] == "L2"
    assert route_task(touches_execution=True)["level"] == "L3"
    assert route_task(security=True)["level"] == "L4"


def test_task_router_budget_is_fail_closed():
    with pytest.raises(RuntimeError, match="budget_exceeded"):
        route_task(security=True, budget_units=8)


def test_decision_assist_readiness_scope():
    snapshot = build_snapshot()
    assert snapshot["scope"] == "analysis_decision_assist_only"
    assert "broker adapter" in snapshot["excluded_live_items"]
    for required in ("review_generation", "decision_chain_code", "prediction_observer_code"):
        assert snapshot["required_evidence"][required] is True
