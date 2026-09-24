from __future__ import annotations

from datetime import date

import pandas as pd

from quant_system import strategy_engine, trade_executor


def _frame(days: list[str], close: list[float] | None = None) -> pd.DataFrame:
    values = close or list(range(1, len(days) + 1))
    return pd.DataFrame({"date": pd.to_datetime(days), "close": values})


def test_factor_klines_drop_future_and_use_completed_day(monkeypatch, tmp_path):
    monkeypatch.setattr(strategy_engine.FactorStrategy, "KLINE_DIR", tmp_path)
    monkeypatch.setattr(strategy_engine, "latest_completed_trading_day", lambda _: date(2026, 8, 28))
    days = pd.bdate_range("2026-06-01", "2026-08-31").strftime("%Y-%m-%d").tolist()
    _frame(days).to_parquet(tmp_path / "600000.parquet")

    result = strategy_engine.FactorStrategy._load_klines(["600000"], as_of="2026-08-29 10:00:00")

    assert result["600000"]["date"].max() == pd.Timestamp("2026-08-28")


def test_factor_klines_reject_stale_completed_data(monkeypatch, tmp_path):
    monkeypatch.setattr(strategy_engine.FactorStrategy, "KLINE_DIR", tmp_path)
    monkeypatch.setattr(strategy_engine, "latest_completed_trading_day", lambda _: date(2026, 8, 28))
    days = pd.bdate_range("2026-04-01", "2026-08-20").strftime("%Y-%m-%d").tolist()
    _frame(days).to_parquet(tmp_path / "600000.parquet")

    assert strategy_engine.FactorStrategy._load_klines(["600000"], as_of="2026-08-29") == {}


def test_market_temperature_filters_future_and_rejects_stale(monkeypatch, tmp_path):
    path = tmp_path / "fusion.parquet"
    monkeypatch.setattr(trade_executor, "MARKET_FUSION_FILE", path)
    monkeypatch.setattr(
        trade_executor,
        "latest_completed_trading_day",
        lambda value: date(2026, 9, 10) if str(value).startswith("2026-09-10") else date(2026, 8, 28),
    )
    pd.DataFrame({
        "date": pd.to_datetime(["2026-08-27", "2026-08-31"]),
        "temperature": [35, 90],
    }).to_parquet(path)

    assert trade_executor._market_temperature("2026-08-29") == 35
    assert trade_executor._market_temperature("2026-09-10") is None
    assert trade_executor._max_new_buys(None) == 0


def test_market_temperature_accepts_latest_completed_day(monkeypatch, tmp_path):
    path = tmp_path / "fusion.parquet"
    monkeypatch.setattr(trade_executor, "MARKET_FUSION_FILE", path)
    monkeypatch.setattr(trade_executor, "latest_completed_trading_day", lambda _: date(2026, 8, 28))
    pd.DataFrame({"date": ["2026-08-28"], "temperature": [58]}).to_parquet(path)

    assert trade_executor._market_temperature("2026-08-29") == 58
    assert trade_executor._max_new_buys(58) == 4
