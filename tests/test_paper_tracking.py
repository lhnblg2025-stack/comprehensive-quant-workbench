from __future__ import annotations

import json
from pathlib import Path

from scripts import paper_equity as pe
from scripts import benchmark_compare as bc


def _points(dates_values):
    return [{"date": d, "net_value": v} for d, v in dates_values]


def test_realized_cash_flow_signs():
    trades = [
        {"trade_type": "buy", "total_amount": 1000},
        {"trade_type": "sell", "total_amount": 1200},
    ]
    assert pe.realized_cash_flow(trades) == 200.0


def test_build_snapshot_reports_missing_market_value():
    snapshot = pe.build_snapshot(date="2026-08-30", initial_cash=1_000_000, trades=[], summary={})
    assert snapshot["cash"] == 1_000_000
    assert snapshot["market_value"] == 0
    assert snapshot["net_value"] == 1_000_000
    assert snapshot["cumulative_return_pct"] == 0.0


def test_max_drawdown():
    points = _points([("d1", 100), ("d2", 120), ("d3", 90), ("d4", 110)])
    assert pe.max_drawdown(points) == -25.0


def test_strategy_metrics_compute_from_curve():
    points = _points([("d1", 100), ("d2", 110), ("d3", 121)])
    metrics = bc.strategy_metrics(points)
    assert metrics["points"] == 3
    assert metrics["annual_return_pct"] is not None


def test_compare_marks_missing_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "ROOT", tmp_path)
    result = bc.compare(_points([("d1", 100), ("d2", 110)]))
    assert result["strategy"]["points"] == 2
    assert all(b["status"] == "missing" for b in result["benchmarks"].values())
