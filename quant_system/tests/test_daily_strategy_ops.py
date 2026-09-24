from __future__ import annotations

import pandas as pd
import pytest

from quant_system.daily_strategy_ops import build_order_advice, promotion_state, reconcile_fills, strategy_health
from quant_system.research_enhancements import neutralize_factor, reconcile_signal_execution


def test_order_advice_sells_before_buys_and_keeps_cash_buffer():
    orders = build_order_advice({"A": 1000}, {"B": 0.5}, {"A": 10.0, "B": 20.0}, 10_000, lot_size=100, cash_buffer=.1)
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0].phase == 1 and orders[1].phase == 2


def test_fill_reconciliation_and_health_gate():
    orders = build_order_advice({}, {"A": .5}, {"A": 10}, 100_000)
    result = reconcile_fills(orders, [{"symbol": "A", "side": "buy", "shares": orders[0].shares, "price": 10, "fee": 5}], {"A": 11}, 100_000)
    assert result["fill_rate"] == pytest.approx(1.0)
    assert strategy_health(rolling_excess_return=-.01, max_drawdown=-.05, fill_rate=1, tracking_error=.01, data_fresh=True)["status"] == "degraded"
    assert promotion_state({"oos_windows": 3}) == "RESEARCH"


def test_neutralization_and_reconciliation():
    frame = pd.DataFrame({"factor": [1., 2., 3., 4.], "market_cap": [10., 20., 30., 40.]})
    residual = neutralize_factor(frame, "factor", industry_col=None)
    assert abs(residual.dropna().mean()) < 1e-9
    rec = reconcile_signal_execution(pd.Series([.01, -.005]), pd.Series([100_000, 100_400]), initial_capital=100_000)
    assert "unexplained_gap" in rec
