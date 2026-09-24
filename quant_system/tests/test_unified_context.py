from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import pytest

from quant_system.context import AsOfContext, RunContext, coerce_as_of, coerce_run_context


def test_as_of_context_normalizes_legacy_values():
    assert AsOfContext("2026-08-30").iso_date == "2026-08-30"
    assert AsOfContext(date(2026, 8, 30)).iso_date == "2026-08-30"
    assert AsOfContext(datetime(2026, 8, 30, 9, 0)).iso_date == "2026-08-30"
    assert str(coerce_as_of("2026-08-30")) == "2026-08-30"


def test_as_of_context_rejects_invalid_dates():
    with pytest.raises(ValueError):
        AsOfContext("2026-02-30")
    with pytest.raises(ValueError):
        AsOfContext("20260830")


def test_run_context_propagates_environment_without_mutating_base():
    context = RunContext(run_id="run-abc", as_of="2026-08-30", metadata={"source": "test"})
    base = {"EXISTING": "1"}
    child = context.child_env(base)
    assert child["EXISTING"] == "1"
    assert child["QUANT_RUN_ID"] == "run-abc"
    assert child["QUANT_AS_OF"] == "2026-08-30"
    assert "QUANT_RUN_ID" not in base
    assert context.as_dict()["metadata"] == {"source": "test"}


def test_run_context_from_env_and_create():
    context = RunContext.from_env({"QUANT_RUN_ID": "env-run", "QUANT_AS_OF": "2026-08-29"})
    assert context.run_id == "env-run"
    assert context.iso_date == "2026-08-29"
    generated = RunContext.create(as_of="2026-08-30")
    assert generated.run_id and generated.iso_date == "2026-08-30"


def test_run_context_rejects_unsafe_run_id():
    with pytest.raises(ValueError):
        RunContext(run_id="bad/run", as_of="2026-08-30")


def test_coerce_existing_context_keeps_identity():
    original = RunContext(run_id="run-1", as_of="2026-08-30")
    assert coerce_run_context(original) is original
    merged = coerce_run_context(original, as_of="2026-08-29")
    assert merged.run_id == "run-1"
    assert merged.iso_date == "2026-08-29"
