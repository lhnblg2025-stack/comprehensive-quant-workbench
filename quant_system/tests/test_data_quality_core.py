"""data_quality 核心单元测试（域 W 双轨制）。

轨道1 功能矩阵：交易日历、日期填充、周末判断、复权校验、异常价格检测、幸存者偏差。
轨道2 已知 bug 回归：节假日不填充、缺失日期不向前视。

全部使用合成日历与合成 K 线，不触网、不读真实数据库。
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd
import pytest

import quant_system.data_quality as dq


def business_days(start, end):
    return {
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start, end, freq="B")
    }


def setup_calendar(monkeypatch, calendar):
    monkeypatch.setattr(dq, "_TRADE_CALENDAR_CACHE", None)
    monkeypatch.setattr(dq, "_get_trade_calendar", lambda: set(calendar))


def make_bar(date, close=10.0):
    return [date, close, close + 0.1, close - 0.1, close, 1000.0]


class TestGetTradeCalendar:
    def test_returns_weekday_only_set(self, monkeypatch):
        calendar = business_days("2026-08-03", "2026-08-14")
        setup_calendar(monkeypatch, calendar)

        result = dq.get_trade_calendar()

        assert isinstance(result, set)
        assert result
        for value in result:
            assert datetime.strptime(value, "%Y-%m-%d").weekday() < 5


class TestFillMissingDates:
    def test_fills_middle_days_without_overrunning_bounds(self, monkeypatch):
        calendar = business_days("2026-08-10", "2026-08-14")
        setup_calendar(monkeypatch, calendar)
        bars = [make_bar("2026-08-10", 10.0), make_bar("2026-08-14", 12.0)]

        result = dq.fill_missing_dates(bars)

        assert [bar[0] for bar in result] == [
            "2026-08-10",
            "2026-08-11",
            "2026-08-12",
            "2026-08-13",
            "2026-08-14",
        ]
        assert result[0][4] == pytest.approx(10.0)
        assert result[-1][4] == pytest.approx(12.0)
        for filled in result[1:4]:
            assert filled[4] == pytest.approx(10.0)
            assert filled[5] == 0

    def test_empty_input_does_not_crash(self):
        assert dq.fill_missing_dates([]) == []


class TestWeekendHelpers:
    def test_is_weekend(self):
        assert dq._is_weekend(datetime(2026, 8, 15)) is True
        assert dq._is_weekend(datetime(2026, 8, 16)) is True
        assert dq._is_weekend(datetime(2026, 8, 14)) is False

    def test_trade_calendar_contains_only_weekdays(self, monkeypatch):
        calendar = business_days("2026-08-03", "2026-08-14")
        setup_calendar(monkeypatch, calendar)

        result = dq.get_trade_calendar()

        assert result == calendar
        assert all(
            datetime.strptime(value, "%Y-%m-%d").weekday() < 5
            for value in result
        )

    def test_get_missing_trading_days_excludes_weekends_in_span(self, monkeypatch):
        all_days = {
            d.strftime("%Y-%m-%d")
            for d in pd.date_range("2026-08-10", "2026-08-16", freq="D")
        }
        setup_calendar(monkeypatch, all_days)

        result = dq._get_missing_trading_days(
            datetime(2026, 8, 10),
            datetime(2026, 8, 16),
        )

        assert result == ["2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"]


class TestValidateAdjustedPrices:
    def test_synthetic_kline_returns_complete_structure(self, monkeypatch):
        dates = ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14"]
        closes = [10.0, 10.1, 10.2, 10.15, 10.25]

        class FakeAkshare:
            def stock_zh_a_hist(self, symbol=None, adjust=""):
                return pd.DataFrame(
                    {
                        "日期": dates,
                        "股票代码": [str(symbol)] * len(dates),
                        "开盘": [closes[i - 1] if i else closes[0] for i in range(len(closes))],
                        "收盘": closes,
                        "最高": [c * 1.005 for c in closes],
                        "最低": [c * 0.995 for c in closes],
                    }
                )

        monkeypatch.setitem(sys.modules, "akshare", FakeAkshare())
        monkeypatch.setattr(dq, "get_price_limit_pct", lambda symbol: 10.0)

        result = dq.validate_adjusted_prices("600000")

        assert result["symbol"] == "600000"
        assert result["ok"] is True
        assert result["adjust_factor_ok"] is True
        assert result["anomalies"] == []


class TestDetectPriceAnomalies:
    def test_extreme_price_move_is_detected(self):
        bars = []
        for i in range(19):
            bars.append([f"2026-08-{i + 3:02d}", 10.0, 10.1, 9.9, 10.0, 1000.0])
        bars.append(["2026-08-22", 13.5, 13.6, 13.4, 13.5, 5000.0])

        result = dq.detect_price_anomalies("600000", bars, z_threshold=4.0)

        assert len(result) == 1
        assert result[0]["date"] == "2026-08-22"
        assert result[0]["pct_chg"] == pytest.approx(35.0)
        assert result[0]["likely_error"] is True


class TestCheckSurvivorshipBias:
    def test_delisted_and_new_stock_sample(self, monkeypatch):
        monkeypatch.setattr(
            dq,
            "get_delisted_stocks",
            lambda: [
                {
                    "symbol": "000001",
                    "name": "退市甲",
                    "delist_date": "2026-03-01",
                    "last_price": 1.0,
                    "reason": "退市",
                },
                {
                    "symbol": "000002",
                    "name": "退市乙",
                    "delist_date": "",
                    "last_price": 1.0,
                    "reason": "",
                },
                {
                    "symbol": "000003",
                    "name": "退市丙",
                    "delist_date": "2025-01-01",
                    "last_price": 1.0,
                    "reason": "",
                },
            ],
        )

        result = dq.check_survivorship_bias(
            ["000001", "000002", "000003", "000004"],
            "2026-01-01",
        )

        assert len(result) == 2
        assert {item["symbol"] for item in result} == {"000001", "000002"}
        assert all(item["note"] for item in result)


class TestRegression:
    def test_national_day_holiday_is_not_filled(self, monkeypatch):
        setup_calendar(
            monkeypatch,
            {"2026-09-30", "2026-10-08", "2026-10-09"},
        )
        bars = [make_bar("2026-09-30", 10.0), make_bar("2026-10-08", 11.0)]

        result = dq.fill_missing_dates(bars)

        assert [bar[0] for bar in result] == ["2026-09-30", "2026-10-08"]

    def test_fill_missing_dates_does_not_look_into_future(self, monkeypatch):
        calendar = business_days("2026-08-10", "2026-08-14")
        setup_calendar(monkeypatch, calendar)
        bars = [make_bar("2026-08-10", 10.0), make_bar("2026-08-12", 11.0)]

        result = dq.fill_missing_dates(bars)

        assert [bar[0] for bar in result] == ["2026-08-10", "2026-08-11", "2026-08-12"]
        assert result[-1][0] == "2026-08-12"
