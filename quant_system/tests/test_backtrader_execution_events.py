from __future__ import annotations

import pandas as pd

from quant_system.backtrader_engine import run_backtrader


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=4)
    return pd.DataFrame([
        {"date": day, "code": code, "raw_open": 10.0, "raw_high": 10.0,
         "raw_low": 10.0, "raw_close": 10.0, "volume": 100_000.0, "amount": 1_000_000.0}
        for day in dates for code in ("000001", "000002")
    ])


def test_blocked_buy_has_explicit_reason_and_reconciled_counts():
    panel = _panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    state = pd.DataFrame([{
        "date": dates[1], "code": "000001", "suspended": False,
        "limit_up_locked": True, "limit_down_locked": False,
    }])
    result, _, events = run_backtrader(panel, {dates[0]: {"000001": 1.0}}, trade_state=state)
    assert result.orders == result.trades + result.blocked_orders + result.rejected_orders
    assert result.blocked_orders == 1
    assert result.rejected_orders == 0
    blocked = events[events.status == "blocked"].iloc[0]
    assert blocked.reason == "limit_up_locked"
    assert result.metadata["order_count_reconciliation"]["reconciles"] is True


def test_filled_order_counts_are_not_mixed_with_rejections():
    panel = _panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    result, _, events = run_backtrader(panel, {dates[0]: {"000001": 1.0}})
    assert result.trades == int(events.status.eq("filled").sum())
    assert result.blocked_orders == 0
    assert result.orders == len(events)
    assert set(events.status).issubset({"filled", "blocked", "rejected", "canceled", "skipped"})
