from __future__ import annotations

import json

import pandas as pd

from quant_system.strategy_data_gate import evaluate_strategy


def _panel(path, *, include_amount=True):
    rows = []
    for code in ("000001", "000002"):
        for day in pd.date_range("2024-01-01", periods=3, freq="D"):
            row = {
                "date": day,
                "code": code,
                "close": 10.0,
                "high": 10.5,
                "low": 9.5,
                "volume": 100.0,
                "turnover": 0.1,
                "synthetic_proxy": False,
            }
            if include_amount:
                row["amount"] = 1000.0
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _indexes(directory):
    directory.mkdir()
    frame = pd.DataFrame(
        {
            "index_code": ["000001", "000001"],
            "index_name": ["上证指数", "上证指数"],
            "trade_date": pd.date_range("2024-01-01", periods=2),
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.01],
            "volume": [1.0, 1.0],
            "amount": [1.0, 1.0],
        }
    )
    frame.to_parquet(directory / "000001_index_daily.parquet", index=False)
    manifest = directory.parent / "manifest.json"
    manifest.write_text(json.dumps({"datasets": [{"status": "PASS"}]}), encoding="utf-8")
    return manifest


def test_price_strategy_is_researchable_but_not_formal(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel)
    gate = evaluate_strategy("trend", panel_path=panel, index_dir=tmp_path / "none", index_manifest=tmp_path / "none.json")
    assert gate.status == "READY_RESEARCH"
    assert gate.allowed_for_research
    assert not gate.allowed_for_formal_backtest


def test_market_timing_uses_official_index_gate(tmp_path):
    index_dir = tmp_path / "normalized"
    manifest = _indexes(index_dir)
    gate = evaluate_strategy(
        "market_timing",
        panel_path=tmp_path / "none.parquet",
        index_dir=index_dir,
        index_manifest=manifest,
    )
    assert gate.allowed_for_research
    assert gate.status == "READY_RESEARCH"
    assert gate.universe["files"] == 1


def test_missing_amount_blocks_volume_flow(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel, include_amount=False)
    gate = evaluate_strategy("volume_flow", panel_path=panel, index_dir=tmp_path / "none", index_manifest=tmp_path / "none.json")
    assert gate.status == "BLOCKED"
    assert any(item.startswith("missing_field:amount") for item in gate.blockers)


def test_pit_and_execution_are_explicitly_blocked(tmp_path):
    panel = tmp_path / "panel.parquet"
    _panel(panel)
    for family in ("pit_value_quality", "execution_audit", "unbiased_market"):
        gate = evaluate_strategy(family, panel_path=panel)
        assert gate.status == "BLOCKED"
        assert not gate.allowed_for_research
