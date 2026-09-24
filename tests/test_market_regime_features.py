from __future__ import annotations

import pandas as pd
import pytest

from quant_system.market_regime_features import build_market_regime_features


def _frame(periods=70):
    dates = pd.date_range("2020-01-01", periods=periods, freq="D")
    rows = []
    for code, multiplier in (("000001", 1.0), ("000300", 1.1), ("000852", 1.2)):
        for i, day in enumerate(dates):
            close = (100.0 + i) * multiplier
            rows.append(
                {
                    "index_code": code,
                    "index_name": code,
                    "trade_date": day,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000.0,
                    "amount": 10000.0,
                }
            )
    return pd.DataFrame(rows)


def test_features_are_causal_and_have_expected_fields():
    full = build_market_regime_features(_frame())
    truncated = build_market_regime_features(_frame().query("trade_date <= '2020-03-10'"))
    fields = {"return_1d", "return_5d", "return_20d", "return_60d", "volatility_20d", "ma_60d", "drawdown_60d", "relative_strength_20d", "style_regime"}
    assert fields.issubset(full.columns)
    left = full.query("index_code == '000300' and trade_date <= '2020-03-10'").reset_index(drop=True)
    right = truncated.query("index_code == '000300'").reset_index(drop=True)
    pd.testing.assert_frame_equal(left[sorted(fields)], right[sorted(fields)], check_dtype=False)
    assert full["feature_asof"].equals(full["trade_date"])


def test_insufficient_history_is_nan_not_backfilled():
    result = build_market_regime_features(_frame(periods=10))
    assert result["return_20d"].isna().all()
    assert result["volatility_20d"].isna().all()


def test_missing_columns_fail_closed():
    with pytest.raises(ValueError, match="missing_columns"):
        build_market_regime_features(_frame().drop(columns=["amount"]))
