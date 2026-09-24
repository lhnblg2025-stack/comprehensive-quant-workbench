from __future__ import annotations

import pandas as pd
import pytest

from quant_system.backtest_dataset_builder import normalize_corporate_actions


def test_normalize_dividend_and_bonus_per_ten_shares():
    raw = pd.DataFrame([{
        "除权除息日": "2025-06-12", "方案进度": "实施分配",
        "现金分红-现金分红比例": 3.6, "送转股份-送转总比例": 2.0,
    }, {
        "除权除息日": None, "方案进度": "预案",
        "现金分红-现金分红比例": 5.0, "送转股份-送转总比例": 0.0,
    }])
    actions = normalize_corporate_actions("000001", raw)
    dividend = actions[actions.action_type == "dividend"].iloc[0]
    split = actions[actions.action_type == "split"].iloc[0]
    assert dividend.cash_per_share == pytest.approx(.36)
    assert split.ratio == pytest.approx(1.2)
    assert len(actions) == 2
