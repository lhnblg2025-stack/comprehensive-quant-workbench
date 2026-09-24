from __future__ import annotations

import pandas as pd
import pytest

from scripts.paper_execution_reconciliation import reconcile


def test_reconcile_partial_fills_and_adverse_slippage():
    orders = pd.DataFrame([
        {"order_id": "o1", "symbol": "600000", "side": "buy", "requested_shares": 1000, "suggested_price": 10.0},
        {"order_id": "o2", "symbol": "000001", "side": "sell", "requested_shares": 500, "suggested_price": 20.0},
    ])
    fills = pd.DataFrame([
        {"order_id": "o1", "filled_shares": 400, "filled_price": 10.1},
        {"order_id": "o1", "filled_shares": 300, "filled_price": 10.2},
        {"order_id": "o2", "filled_shares": 500, "filled_price": 19.8},
    ])
    result, summary = reconcile(orders, fills)
    assert result.loc[result.order_id == "o1", "fill_status"].iloc[0] == "partial"
    assert result.loc[result.order_id == "o2", "fill_status"].iloc[0] == "filled"
    assert summary["filled_shares"] == 1200
    assert summary["fill_rate"] == pytest.approx(1200 / 1500)
    assert summary["mean_adverse_slippage_bps"] > 0
