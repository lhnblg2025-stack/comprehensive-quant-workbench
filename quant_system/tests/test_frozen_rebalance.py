from __future__ import annotations

import pandas as pd


def test_monthly_rebalance_schedule_is_not_daily():
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    weekly = set(pd.Series(dates, index=dates).groupby(dates.to_period("W")).max())
    monthly = set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    assert len(monthly) < len(weekly) < len(dates)
