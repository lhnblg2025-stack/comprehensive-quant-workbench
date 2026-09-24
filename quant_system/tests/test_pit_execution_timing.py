from __future__ import annotations

import pandas as pd

from quant_system.factor_backtest_runner import _prepare_panel
from quant_system.run_pit_order_validation import _targets


def test_raw_signal_label_enters_after_signal_close():
    dates = pd.bdate_range("2024-01-02", periods=4)
    panel = pd.DataFrame({
        "date": dates,
        "code": ["000001"] * 4,
        "close": [10.0, 10.0, 10.0, 10.0],
        "raw_open": [10.0, 11.0, 12.0, 13.0],
    })
    result = _prepare_panel(panel)
    assert result.loc[0, "entry_date"] == dates[1]
    assert result.loc[0, "exit_date"] == dates[2]
    assert result.loc[0, "forward_return"] == 12.0 / 11.0 - 1.0
    assert result.loc[0, "return_basis"] == "raw_next_open_to_following_open"


def test_targets_execute_on_next_session_not_signal_session():
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for date in dates:
        for code, score in (("000001", 2.0), ("000002", 1.0), ("000003", 0.0)):
            rows.append({"date": date, "code": code, "score": score, "industry": "A", "raw_open": 10.0, "raw_close": 10.0})
    targets = _targets(pd.DataFrame(rows), "score", 1, 0.34, "daily", 0.0)
    assert dates[0] not in targets
    assert dates[1] in targets
    assert "000001" in targets[dates[1]]
