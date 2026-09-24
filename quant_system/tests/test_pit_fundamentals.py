from __future__ import annotations

import pandas as pd
import pytest

from quant_system.pit_fundamentals import (
    add_fundamental_factors,
    attach_pit_fundamentals,
    attach_pit_industry,
    coverage_summary,
    industry_neutralize,
)


def _calendar():
    return pd.bdate_range("2024-04-01", periods=6)


def test_fundamentals_are_unavailable_until_after_announcement_lag():
    calendar = _calendar()
    panel = pd.DataFrame({
        "date": list(calendar) * 2,
        "code": ["000001"] * len(calendar) + ["000002"] * len(calendar),
        "close": [10.0] * (len(calendar) * 2),
    })
    fundamentals = pd.DataFrame({
        "code": ["000001", "000002"],
        "report_period": ["2024-03-31", "2024-03-31"],
        "announcement_date": ["2024-04-02", "2024-04-02"],
        "eps_ttm": [1.0, 2.0],
        "book_value_per_share": [5.0, 4.0],
    })
    result = attach_pit_fundamentals(panel, fundamentals, calendar, lag_sessions=1)
    before = result[(result.code == "000001") & (result.date <= pd.Timestamp("2024-04-02"))]
    after = result[(result.code == "000001") & (result.date >= pd.Timestamp("2024-04-03"))]
    assert before["eps_ttm"].isna().all()
    assert after["eps_ttm"].eq(1.0).all()


def test_missing_announcement_date_is_rejected_not_backfilled_from_report_period():
    panel = pd.DataFrame({"date": _calendar(), "code": ["000001"] * 6, "close": [10.0] * 6})
    invalid = pd.DataFrame({"code": ["000001"], "report_period": ["2024-03-31"], "eps_ttm": [1.0]})
    with pytest.raises(ValueError, match="fundamentals_missing_columns:announcement_date"):
        attach_pit_fundamentals(panel, invalid, _calendar())


def test_industry_history_is_attached_by_effective_interval_and_neutralizes_within_date():
    date = pd.Timestamp("2024-04-04")
    panel = pd.DataFrame({
        "date": [date] * 6,
        "code": [f"00000{i}" for i in range(1, 7)],
        "close": [10.0] * 6,
        "value_composite": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
    })
    history = pd.DataFrame({
        "code": panel.code,
        "industry": ["A", "A", "A", "B", "B", "B"],
        "effective_date": ["2024-01-01"] * 6,
    })
    result = attach_pit_industry(panel, history)
    neutral = industry_neutralize(result, ["value_composite"], min_industry_size=3)
    assert neutral.groupby("industry")["value_composite_industry_neutral"].mean().abs().max() < 1e-12


def test_value_and_quality_composites_require_all_declared_components():
    frame = pd.DataFrame({
        "date": ["2024-04-01"] * 3,
        "code": ["000001", "000002", "000003"],
        "close": [10.0, 20.0, 30.0],
        "eps_ttm": [1.0, 1.0, 1.0],
        "book_value_per_share": [5.0, 5.0, 5.0],
        "operating_cashflow_per_share": [1.0, 1.0, 1.0],
        "roe": [0.1, 0.2, 0.3],
        "net_margin": [0.1, 0.2, 0.3],
        "cfo_to_net_income": [0.8, 0.9, 1.0],
        "debt_to_assets": [0.5, 0.4, 0.3],
    })
    result = add_fundamental_factors(frame)
    assert result["value_composite"].notna().all()
    assert result["quality_composite"].notna().all()
    assert result["value_composite"].iloc[0] > result["value_composite"].iloc[-1]


def test_coverage_blocks_factors_without_requested_breadth():
    frame = pd.DataFrame({
        "date": ["2024-04-01"] * 3,
        "code": ["000001", "000002", "000003"],
        "value_composite": [1.0, 2.0, 3.0],
    })
    summary = coverage_summary(frame, ["value_composite"], min_stocks=4)
    assert summary["status"] == "BLOCK"
    assert summary["factors"]["value_composite"]["usable_dates"] == 0
