from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest import _perf_metrics
from quant_system.metrics_calculator import MetricsCalculator


def test_backtest_perf_metrics_uses_risk_free_adjusted_sharpe():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008, 0.012])
    result = _perf_metrics(returns)
    expected = MetricsCalculator.sharpe(returns, rf=0.02, ddof=1, trading_days=252)
    assert result["sharpe"] == pytest.approx(round(expected, 3))
    assert result["sharpe"] < MetricsCalculator.sharpe(returns, rf=0.0)


def test_financial_wide_frames_align_report_periods_without_duplicate_metrics():
    from scripts.update_financial_quarterly import merge_financial_frames

    old = pd.DataFrame({
        "选项": ["净资产收益率", "每股收益"],
        "2024-12-31": [10.0, 1.0],
        "2025-03-31": [11.0, 1.1],
    })
    current = pd.DataFrame({
        "选项": ["净资产收益率", "每股收益"],
        "2025-03-31": [12.0, np.nan],
        "2025-06-30": [13.0, 1.3],
    })
    result = merge_financial_frames(old, current)
    assert not result["选项"].duplicated().any()
    roe = result.set_index("选项").loc["净资产收益率"]
    eps = result.set_index("选项").loc["每股收益"]
    assert roe["2024-12-31"] == 10.0
    assert roe["2025-03-31"] == 12.0
    assert roe["2025-06-30"] == 13.0
    assert eps["2025-03-31"] == 1.1


def test_factor_rotation_momentum_uses_longer_window_than_validation_return():
    from quant_system.analysis_core.factor_rotation_system import _factor_cross_section

    stocks = {
        "000001": {"close": np.linspace(10, 20, 80), "share": 1e8, "fin": {}},
        "000002": {"close": np.linspace(20, 10, 80), "share": 1e8, "fin": {}},
        "000003": {"close": np.linspace(10, 11, 80), "share": 1e8, "fin": {}},
    }
    factors, coverage = _factor_cross_section(stocks)
    assert "momentum" in factors
    # 60日动量需要足够长历史；它不应退化成20日窗口的同窗值。
    assert coverage["mom60"] == 3
    assert factors["momentum"].notna().sum() == 3


def test_latest_expected_trade_date_uses_market_calendar(monkeypatch):
    import scripts.update_kline_tencent as mod

    monkeypatch.setattr(mod, "datetime", mod.datetime)
    monkeypatch.setattr("quant_system.market_clock.is_trading_day", lambda d: d.strftime("%Y-%m-%d") == "2026-08-21")
    now = mod.datetime(2026, 8, 23, tzinfo=mod.timezone(mod.timedelta(hours=8)))
    assert mod.latest_expected_trade_date(now) == "2026-08-21"
