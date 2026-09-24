from __future__ import annotations

import pandas as pd

from quant_system.research_pipeline import _apply_membership


def test_membership_provider_removes_out_of_membership_rows():
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "code": ["1", "1"], "close": [10, 11]})
    membership = pd.DataFrame({"code": ["1"], "start_date": ["2024-01-02"], "end_date": [None]})
    filtered, summary = _apply_membership(panel, membership)
    assert filtered["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert summary["removed_rows"] == 1
