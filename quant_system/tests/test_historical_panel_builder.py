from __future__ import annotations

import pandas as pd

from quant_system.historical_panel_builder import compute_price_factors


def test_compute_price_factors_is_asof_and_emits_expected_columns():
    dates = pd.bdate_range("2020-01-01", periods=260)
    close = pd.Series(range(100, 360), dtype=float)
    frame = pd.DataFrame({"date": dates, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0, "turnover": 0.01})
    result = compute_price_factors(frame)
    assert {"mom20", "mom60", "mom120", "dist_52w", "amp20", "vol_ratio20", "turnover"}.issubset(result.columns)
    assert pd.isna(result.loc[19, "mom20"])
    assert result.loc[20, "mom20"] == close.iloc[20] / close.iloc[0] - 1
    assert result.loc[259, "dist_52w"] <= 0
