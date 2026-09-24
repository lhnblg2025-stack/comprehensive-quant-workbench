from __future__ import annotations

import pandas as pd

from quant_system.pit_fundamentals import coverage_summary


def test_coverage_summary_uses_daily_universe_ratio_not_fixed_full_count():
    dates = pd.bdate_range("2024-01-01", periods=2)
    rows = []
    for day in dates:
        for code in range(100):
            rows.append({"date": day, "code": f"{code:06d}", "factor": 1.0 if code < 85 else None})
    result = coverage_summary(pd.DataFrame(rows), ["factor"], min_stocks=30, min_coverage_ratio=0.80)
    assert result["factors"]["factor"]["usable_dates"] == 2
    assert result["factors"]["factor"]["median_coverage_ratio"] == 0.85
    assert result["status"] == "PASS"


def test_coverage_summary_blocks_below_ratio_even_if_absolute_floor_passes():
    rows = [{"date": "2024-01-02", "code": f"{code:06d}", "factor": 1.0 if code < 50 else None} for code in range(100)]
    result = coverage_summary(pd.DataFrame(rows), ["factor"], min_stocks=30, min_coverage_ratio=0.80)
    assert result["factors"]["factor"]["usable_dates"] == 0
    assert result["status"] == "BLOCK"
