from __future__ import annotations

import pandas as pd
import pytest

from quant_system.frozen_bundle_long_backtest import LongRunConfig, _annual_performance, _block_reason


def test_annual_performance_aligns_strategy_benchmark_and_excess():
    ledger = pd.DataFrame({
        "date": ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"],
        "return": [0.0, 0.10, 0.0, 0.10],
    })
    benchmark = pd.Series(
        [100.0, 105.0, 105.0, 110.25],
        index=pd.to_datetime(ledger["date"]),
    )

    result = _annual_performance(ledger, benchmark)

    assert result["year"].tolist() == [2024, 2025]
    assert result.loc[0, "strategy_total_return"] == pytest.approx(0.10)
    assert result.loc[0, "benchmark_total_return"] == pytest.approx(0.05)
    assert result.loc[0, "annual_excess_return"] > 0
    assert result.loc[1, "annual_excess_return"] > 0


def test_block_reason_covers_price_volume_and_limit_proxies():
    cfg = LongRunConfig(bundle_dir="bundle", output_dir="out")
    assert _block_reason(pd.Series({"raw_open": 0.0, "raw_volume": 100}), "buy", cfg) == "missing_or_invalid_open_price"
    assert _block_reason(pd.Series({"raw_open": 10.0, "raw_volume": 0}), "buy", cfg) == "suspended_or_zero_volume"
    assert _block_reason(pd.Series({"raw_open": 11.0, "raw_close": 10.0, "raw_volume": 100}), "buy", cfg) == "limit_up_buy_blocked_proxy"
    assert _block_reason(pd.Series({"raw_open": 9.0, "raw_close": 10.0, "raw_volume": 100}), "sell", cfg) == "limit_down_sell_blocked_proxy"
    assert _block_reason(pd.Series({"raw_open": 10.0, "raw_close": 10.0, "raw_volume": 100}), "buy", cfg) is None
