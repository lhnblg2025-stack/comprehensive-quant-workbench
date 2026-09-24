from __future__ import annotations

import pandas as pd

from quant_system.factor_backtest_runner import _prepare_panel


def test_prepare_panel_next_open_labels_are_idempotent():
    dates = pd.bdate_range("2024-01-01", periods=5)
    panel = pd.DataFrame({
        "date": list(dates) * 2,
        "code": ["000001"] * 5 + ["000002"] * 5,
        "close": [10.0] * 10,
        "raw_open": [10.0, 11.0, 12.0, 13.0, 14.0] * 2,
    })
    first = _prepare_panel(panel)
    second = _prepare_panel(first)
    assert "entry_open" in second.columns
    assert "exit_open" in second.columns
    assert second["forward_return"].equals(first["forward_return"])
    assert second["return_basis"].eq("raw_next_open_to_following_open").all()
