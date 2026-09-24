from __future__ import annotations

import pandas as pd

from quant_system.pit_data_audit import audit


def test_audit_fails_closed_for_missing_required_assets(tmp_path):
    result = audit(tmp_path, tmp_path / "report.json", min_symbols=2, min_years=1)

    assert result["status"] == "DATA_BLOCKED"
    assert "required_assets_missing_or_unreadable" in result["errors"]
    assert (tmp_path / "report.json").is_file()


def test_audit_reports_ready_when_all_minimal_assets_present(tmp_path):
    dates = pd.date_range("2020-01-01", periods=370, freq="D")
    panel = pd.DataFrame({"date": dates.repeat(2), "code": ["000001", "000002"] * len(dates), "close": 10.0})
    panel.to_parquet(tmp_path / "historical_panel.parquet", index=False)
    pd.DataFrame({"code": ["000001", "000002"], "start_date": [dates[0], dates[0]]}).to_csv(tmp_path / "membership.csv", index=False)
    pd.DataFrame({"date": dates.repeat(2), "code": ["000001", "000002"] * len(dates)}).to_parquet(tmp_path / "corporate_actions.parquet", index=False)
    pd.DataFrame({"date": dates.repeat(2), "code": ["000001", "000002"] * len(dates)}).to_parquet(tmp_path / "liquidity_capacity.parquet", index=False)
    pd.DataFrame({"announcement_date": dates.repeat(2), "code": ["000001", "000002"] * len(dates)}).to_parquet(tmp_path / "fundamental_releases.parquet", index=False)
    pd.DataFrame({"effective_date": dates.repeat(2), "code": ["000001", "000002"] * len(dates)}).to_parquet(tmp_path / "sw_l1_industry_history.parquet", index=False)
    pd.DataFrame({"date": dates, "return": 0.0}).to_csv(tmp_path / "benchmark_csi300.csv", index=False)

    result = audit(tmp_path, tmp_path / "report.json", min_symbols=2, min_years=1)

    assert result["status"] == "READY_FOR_MATRIX"
    assert result["errors"] == []
