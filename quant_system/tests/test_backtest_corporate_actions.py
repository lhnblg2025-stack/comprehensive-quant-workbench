from __future__ import annotations

import pandas as pd
import pytest

from quant_system.backtest_engine import BacktestEngine, StrategyTemplate


class BuyOnce(StrategyTemplate):
    def __init__(self):
        super().__init__(); self.done = False
    def on_bar(self, bar):
        if not self.done:
            self.buy(bar.close, 100, symbol=bar.symbol); self.done = True


def _engine():
    dates = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"date": dates, "open": 10., "high": 11., "low": 9., "close": 10., "volume": 100_000., "amount": 1_000_000.})
    e = BacktestEngine(); e.set_capital(100_000); e.set_commission(0); e.min_commission = 0; e.transfer_fee_rate = 0; e.set_slippage(0, "none"); e.add_data("600000", frame); e.add_strategy(BuyOnce); return e, dates


def test_mark_to_market_keeps_open_position_value():
    e, _ = _engine(); e.set_end_mode("mark_to_market"); result = e.run()
    assert result.equity_curve.iloc[-1].position_value == pytest.approx(1000)
    assert [trade.offset for trade in result.trades] == ["open"]


def test_dividend_and_split_are_applied_to_raw_account():
    e, dates = _engine(); e.set_end_mode("mark_to_market")
    e.add_corporate_actions(pd.DataFrame([
        {"date": dates[2], "symbol": "600000", "action_type": "dividend", "cash_per_share": .5},
        {"date": dates[3], "symbol": "600000", "action_type": "split", "ratio": 2.0},
    ]))
    result = e.run(); position = e.positions["600000"]
    assert position.volume == pytest.approx(200)
    assert len(e.applied_corporate_actions) == 2
    assert result.equity_curve.iloc[-1].total_value > 100_000
