from pathlib import Path

import pandas as pd
import pytest

from quant_system.closed_loop import LoopConfig, candidate_library, factor_diagnostics, replay_exit, run_loop
import sqlite3


def _panel(days=12, stocks=10):
    rows = []
    for day in range(days):
        for stock in range(stocks):
            price = 100 + day * (1 + stock / 100)
            rows.append({
                "code": f"{stock:06d}",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                "close": price,
                "open": price,
                "high": price + 2,
                "low": price - 1,
                "quality": stock + day / 100,
            })
    return pd.DataFrame(rows)


def test_candidate_labels_are_point_in_time_and_ranked():
    data = candidate_library(_panel(), LoopConfig(("quality",), top_n=2, horizons=(1, 3)))
    first = data[(data.code == "000009") & (data.date == pd.Timestamp("2024-01-01"))].iloc[0]
    assert first.forward_1d == pytest.approx((101.09 / 100) - 1)
    assert int(data[data.date == pd.Timestamp("2024-01-01")].selected.sum()) == 2
    assert data.factor_version.iloc[0]
    assert "factor_coverage" in data.columns


def test_factor_diagnostics_reports_monotonicity():
    data = candidate_library(_panel(), LoopConfig(("quality",), horizons=(1,)))
    report = factor_diagnostics(data, LoopConfig(("quality",), horizons=(1,)))
    item = report["factors"][0]
    assert item["status"] == "ok"
    assert item["monotonicity"] > 0
    assert item["quality_gate"] in {"PASS", "HOLD"}


def test_replay_exit_is_conservative_when_both_levels_hit():
    bars = pd.DataFrame([{"date": "2024-01-02", "high": 110, "low": 90, "close": 100}])
    result = replay_exit(bars, 100, stop_price=95, target_price=105)
    assert result["reason"] == "stop"
    assert result["exit_price"] == 95


def test_run_loop_writes_auditable_artifacts(tmp_path: Path):
    source = tmp_path / "panel.parquet"
    _panel().to_parquet(source)
    result = run_loop(source, tmp_path / "out", LoopConfig(("quality",), top_n=2))
    assert result["manifest"]["selected_rows"] > 0
    assert (tmp_path / "out" / "candidate_library.parquet").exists()
    assert (tmp_path / "out" / "candidate_performance.parquet").exists()
    assert (tmp_path / "out" / "paper_plan.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    with sqlite3.connect(tmp_path / "out" / "candidate_store.db") as db:
        assert db.execute("select count(*) from candidates").fetchone()[0] == result["manifest"]["performance_rows"]
    assert result["paper_plan"]["status"] == "BLOCKED_BY_QUALITY_GATE"
