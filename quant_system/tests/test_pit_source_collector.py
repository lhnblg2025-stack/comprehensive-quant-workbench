from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.pit_source_collector import materialize_local_security_lifecycle, materialize_observed_calendar


def test_materialize_security_lifecycle_marks_snapshot_not_pit(tmp_path: Path):
    source = tmp_path / "security_master_source.parquet"
    pd.DataFrame({
        "ts_code": ["000001.SZ", "600000.SH"],
        "symbol": ["000001", "600000"],
        "list_date": ["19910403", "19991110"],
        "delist_date": [None, None],
    }).to_parquet(source, index=False)
    result = materialize_local_security_lifecycle(tmp_path)
    assert result["status"] == "materialized_snapshot_only"
    out = pd.read_parquet(tmp_path / "pit_security_lifecycle_snapshot.parquet")
    assert (out["source_as_of"] == "security_master_snapshot_not_dated_membership").all()


def test_observed_calendar_is_explicitly_not_official(tmp_path: Path):
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "code": "000001"}).to_parquet(tmp_path / "price_discovery_panel.parquet", index=False)
    result = materialize_observed_calendar(tmp_path)
    assert result["sessions"] == 5
    frame = pd.read_parquet(tmp_path / "observed_trade_calendar.parquet")
    assert (frame["source_as_of"] == "observed_price_panel_sessions_not_exchange_calendar").all()
