"""fundamental_analysis 核心单元测试（域 W 双轨制）。

轨道1 功能矩阵：增速归一化、安全浮点转换、DCF 手算、三阶段估值、比例与格式化。
轨道2 已知 bug 回归：growth_rates 与 projection_years 长度不一致时按默认增速补齐，
且较长输入截断后不越界。

全部使用合成数值，不触网。
"""

from __future__ import annotations

import pytest

from quant_system.fundamental_analysis import (
    DCFValuation,
    _first_growth_value,
    _format_multiple,
    _normalize_growth,
    _positive_ratio,
    _to_float,
)


class TestNormalizeGrowth:
    def test_boundaries_follow_implementation_semantics(self):
        assert _normalize_growth(None) is None
        assert _normalize_growth("5%") is None
        assert _normalize_growth("0.05") == pytest.approx(0.05)
        assert _normalize_growth(-0.2) == pytest.approx(-0.2)
        assert _normalize_growth(0) == pytest.approx(0.0)
        assert _normalize_growth(5) == pytest.approx(0.05)
        assert _normalize_growth(0.3) == pytest.approx(0.3)


class TestToFloat:
    def test_numeric_string_none_and_invalid_inputs(self):
        assert _to_float("12.3") == pytest.approx(12.3)
        assert _to_float(None) is None
        assert _to_float("abc") is None
        assert _to_float(True) is None
        assert _to_float(float("nan")) is None


def reference_two_stage_value(fcf_current, growth_rates, wacc, terminal_growth, shares=1.0, net_debt=0.0):
    fcf = fcf_current
    forecast = []
    pv_fcf = 0.0
    for i, growth in enumerate(growth_rates, start=1):
        fcf *= 1.0 + growth
        forecast.append(fcf)
        pv_fcf += fcf / (1.0 + wacc) ** i
    terminal_value = forecast[-1] * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1.0 + wacc) ** len(growth_rates)
    equity = pv_fcf + pv_terminal - net_debt
    return equity / max(shares, 1.0), pv_fcf + pv_terminal, forecast


class TestDCF:
    def test_two_stage_dcf_hand_calculation(self):
        model = DCFValuation(wacc=0.10, terminal_growth=0.03, projection_years=5)
        growth_rates = [0.05] * 5
        expected_per_share, expected_enterprise, expected_forecast = reference_two_stage_value(
            100.0, growth_rates, 0.10, 0.03
        )

        result = model.two_stage_dcf(100.0, growth_rates)

        assert result["enterprise_value"] == pytest.approx(round(expected_enterprise, 2), abs=1e-6)
        assert result["per_share"] == pytest.approx(round(expected_per_share, 2), abs=1e-6)
        assert result["phase_values"]["fcf_forecast"] == [
            pytest.approx(round(value, 2), abs=1e-6) for value in expected_forecast
        ]
        assert result["terminal_pct"] > 0

    def test_higher_growth_increases_valuation(self):
        model = DCFValuation(wacc=0.10, terminal_growth=0.03, projection_years=5)

        low = model.two_stage_dcf(100.0, [0.05] * 5)
        high = model.two_stage_dcf(100.0, [0.10] * 5)

        assert high["per_share"] > low["per_share"]

    def test_three_stage_dcf_is_reasonable(self):
        model = DCFValuation(wacc=0.10, terminal_growth=0.03, projection_years=5)

        result = model.three_stage_dcf(
            100.0,
            high_growth_rates=[0.20, 0.15],
            transition_growth_rates=[0.10, 0.08, 0.05],
        )

        assert result["per_share"] > 0
        assert result["enterprise_value"] > 0
        assert len(result["phase_values"]["fcf_forecast"]) == 5
        assert 0.0 <= result["terminal_pct"] <= 100.0


class TestHelpers:
    def test_positive_ratio_boundaries(self):
        assert _positive_ratio(10.0, 2.0) == pytest.approx(5.0)
        assert _positive_ratio(10.0, 0.0) is None
        assert _positive_ratio(10.0, -1.0) is None
        assert _positive_ratio(10.0, None) is None

    def test_first_growth_value_uses_first_non_none(self):
        value = _first_growth_value(
            {"revenue_growth": None, "revenue_yoy": "0.20"},
            {"revenue_growth": "0.30"},
            keys=("revenue_growth", "revenue_yoy"),
        )
        assert value == pytest.approx(0.20)

        value = _first_growth_value(
            {"revenue_growth": "0.10"},
            {"revenue_yoy": "0.40"},
            keys=("revenue_growth", "revenue_yoy"),
        )
        assert value == pytest.approx(0.10)

    def test_format_multiple_outputs(self):
        assert _format_multiple(None) == "N/A"
        assert _format_multiple(123.456) == "123.46"
        assert _format_multiple(-12.3) == "-12.30"


class TestRegression:
    def test_short_growth_rates_use_default_growth_not_dead_branch(self):
        model = DCFValuation(wacc=0.10, terminal_growth=0.03, projection_years=3)

        result = model.two_stage_dcf(100.0, [0.10])

        padded = [0.10, 0.05, 0.05]
        fcf = 100.0
        expected = []
        for growth in padded:
            fcf *= 1.0 + growth
            expected.append(round(fcf, 2))
        assert len(result["phase_values"]["fcf_forecast"]) == len(expected)
        assert all(
            result["phase_values"]["fcf_forecast"][i] == pytest.approx(expected[i], abs=1e-12)
            for i in range(len(expected))
        )

    def test_long_growth_rates_are_truncated_without_index_error(self):
        model = DCFValuation(wacc=0.10, terminal_growth=0.03, projection_years=2)

        result = model.two_stage_dcf(100.0, [0.10, 0.20, 0.30])

        assert len(result["phase_values"]["fcf_forecast"]) == 2
        assert result["phase_values"]["fcf_forecast"] == [
            pytest.approx(110.0),
            pytest.approx(132.0),
        ]
