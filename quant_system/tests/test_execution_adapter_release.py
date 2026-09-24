from __future__ import annotations

import pandas as pd

from quant_system.execution_adapter import run_order_level_backtest


def test_order_level_adapter_preserves_release_id():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    panel = pd.DataFrame({
        "date": list(dates) * 2, "code": ["000001"] * 3 + ["000002"] * 3,
        "open": [10.0] * 6, "high": [11.0] * 6, "low": [9.0] * 6,
        "close": [10.0] * 6, "volume": [1_000_000] * 6,
    })
    result = run_order_level_backtest(panel, {dates[0]: {"000001": 0.5, "000002": 0.5}}, release_id="release-test", candidate_ids={dates[0]: {"000001": "c1", "000002": "c2"}})
    assert result.release_id == "release-test"
    assert hasattr(result, "candidate_attribution")
