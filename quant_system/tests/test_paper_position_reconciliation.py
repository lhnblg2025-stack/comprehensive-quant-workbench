from __future__ import annotations

import pandas as pd

from scripts.paper_position_reconciliation import reconcile


def test_position_reconciliation_reports_missing_positions():
    actual = pd.DataFrame([{"symbol": "600000", "shares": 100}, {"symbol": "000001", "shares": 50}])
    out, summary = reconcile(actual, [{"symbol": "600000", "shares": 100}, {"symbol": "000002", "shares": 30}])
    assert summary["matched"] == 1
    assert summary["mismatched"] == 2
    assert set(out.status) == {"matched", "missing_in_system", "missing_in_actual"}
