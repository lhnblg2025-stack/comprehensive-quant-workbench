from __future__ import annotations

import pandas as pd

from quant_system.factor_backtest_runner import _select_with_industry_cap
from quant_system.research_pipeline import _targets_from_factor


def test_industry_cap_limits_equal_weight_selection_even_when_one_industry_dominates_scores():
    ranked = pd.DataFrame({
        "code": [f"{item:06d}" for item in range(8)],
        "industry": ["A"] * 5 + ["B"] * 3,
        "_score": list(range(8, 0, -1)),
    })
    selected = _select_with_industry_cap(ranked, n=4, max_industry_weight=0.5)
    industries = ranked.set_index("code").loc[list(selected), "industry"]
    assert industries.value_counts().max() == 2


def test_order_targets_apply_same_industry_cap_as_signal_backtest():
    date = pd.Timestamp("2024-04-30")
    panel = pd.DataFrame({
        "date": [date] * 8,
        "code": [f"{item:06d}" for item in range(8)],
        "close": [10.0] * 8,
        "industry": ["A"] * 5 + ["B"] * 3,
        "score": list(range(8, 0, -1)),
    })
    targets = _targets_from_factor(panel, "score", [date], .5, rebalance="monthly", max_industry_weight=.5)
    codes = list(targets[date])
    industries = panel.set_index("code").loc[codes, "industry"]
    assert industries.value_counts().max() == 2
    assert sum(targets[date].values()) == 1.0
