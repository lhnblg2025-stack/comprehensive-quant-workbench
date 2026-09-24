from __future__ import annotations

import pandas as pd

from quant_system.backtrader_adapter import run_backtrader_targets


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []
    for i, day in enumerate(dates):
        for code, price in (("000001", 10 + i * .1), ("000002", 20 + i * .1)):
            rows.append({
                "date": day, "code": code, "raw_open": price,
                "raw_high": price * 1.01, "raw_low": price * .99,
                "raw_close": price, "volume": 100000,
            })
    return pd.DataFrame(rows)


def test_backtrader_runs_target_weights_with_next_open_and_lots():
    panel = _panel()
    date = pd.Timestamp("2024-01-01")
    result, equity = run_backtrader_targets(panel, {date: {"000001": .5, "000002": .5}}, capital=100_000)
    assert result.engine == "backtrader"
    assert result.observations == 8
    assert result.trades >= 1
    assert result.final_value > 0
    assert equity.date.is_monotonic_increasing


def test_backtrader_rejects_missing_required_fields():
    panel = _panel().drop(columns=["raw_open"])
    try:
        run_backtrader_targets(panel, {})
    except ValueError as exc:
        assert "raw_open" in str(exc)
    else:
        raise AssertionError("missing raw_open was not rejected")
