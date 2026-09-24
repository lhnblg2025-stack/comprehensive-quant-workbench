from __future__ import annotations

import pandas as pd

from quant_system.portfolio_holding_backtest import HoldingConfig, run_portfolio_backtest


def _panel(days: int = 14) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    for i, day in enumerate(dates):
        for code, signal in (("000001", 2.0), ("000002", 1.0)):
            rows.append({
                "date": day, "code": code, "raw_open": 10 + i,
                "raw_close": 10 + i, "amount": 1_000_000,
                "adv_amount_20": 1_000_000, "signal": signal,
                "suspended": False, "limit_up_locked": False,
                "limit_down_locked": False,
            })
    return pd.DataFrame(rows)


def test_holding_periods_are_non_overlapping_and_use_next_open():
    result = run_portfolio_backtest(_panel(), HoldingConfig(holding_sessions=3, rebalance_sessions=3, initial_capital=100_000, quantile=.5))
    periods = result["equity"][["signal_date", "entry_date", "exit_date"]].drop_duplicates()
    assert len(periods) == 3
    assert (periods.entry_date > periods.signal_date).all()
    assert periods.entry_date.is_monotonic_increasing
    assert periods.exit_date.iloc[0] < periods.entry_date.iloc[1]


def test_adv_participation_limits_position_size():
    panel = _panel()
    panel["adv_amount_20"] = 20_000.0
    result = run_portfolio_backtest(panel, HoldingConfig(holding_sessions=3, rebalance_sessions=3, initial_capital=100_000, quantile=.5))
    buys = result["trades"][result["trades"].side == "buy"]
    assert not buys.empty
    assert (buys.notional <= 2_000.0 + 1e-8).all()


def test_limit_up_blocks_entry_and_limit_down_blocks_exit():
    panel = _panel()
    first_entry = pd.bdate_range("2024-01-01", periods=2)[1]
    last_exit = pd.bdate_range("2024-01-01", periods=5)[-1]
    panel.loc[(panel.date == first_entry) & (panel.code == "000001"), "limit_up_locked"] = True
    entry_result = run_portfolio_backtest(panel, HoldingConfig(holding_sessions=3, rebalance_sessions=3, initial_capital=100_000, quantile=.5))
    first_buys = entry_result["trades"].query("side == 'buy' and entry_date == '2024-01-02'")
    assert "000001" not in set(first_buys.code)

    exit_panel = _panel()
    exit_panel.loc[(exit_panel.date == last_exit) & (exit_panel.code == "000001"), "limit_down_locked"] = True
    exit_result = run_portfolio_backtest(exit_panel, HoldingConfig(holding_sessions=3, rebalance_sessions=3, initial_capital=100_000, quantile=.5))
    assert exit_result["trades"].query("side == 'sell' and code == '000001' and blocked == True").shape[0] >= 1
