from __future__ import annotations

import pandas as pd

from quant_system.backtest_crosscheck import bounded_crosscheck


def test_bounded_crosscheck_uses_same_window_and_target_dates():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for i, date in enumerate(dates):
        for code in ["000001", "000002", "000003", "000004", "000005"]:
            price = 10 + i * .05
            rows.append({"date": date, "code": code, "raw_open": price, "raw_high": price * 1.01, "raw_low": price * .99, "raw_close": price, "volume": 100000, "amount": 1000000, "short_reversal_5": float(i), "suspended": False, "limit_up_locked": False, "limit_down_locked": False})
    result = bounded_crosscheck(pd.DataFrame(rows), "short_reversal_5", days=25, symbols=5, holding_sessions=5, quantile=.2)
    assert result["window"]["symbols"] == 5
    assert result["window"]["target_dates"] > 0
    assert result["comparison"]["primary_engine"] == "portfolio_holding"
    assert result["comparison"]["secondary_engine"] == "backtrader"
