from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.ten_year_campaign import audit_campaign


def _make_price(path: Path, n_symbols: int = 3, years: int = 2) -> None:
    dates = pd.bdate_range("2020-01-01", periods=252 * years)
    rows = []
    for date in dates:
        for i in range(n_symbols):
            rows.append({
                "date": date, "code": f"{i + 1:06d}", "raw_open": 10.0,
                "raw_high": 11.0, "raw_low": 9.0, "raw_close": 10.5,
                "volume": 1000, "amount": 10000, "is_suspended": False,
                "limit_up": False, "limit_down": False,
            })
    (path / "raw" / "prices").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "raw" / "prices" / "prices.parquet", index=False)


def test_audit_writes_data_blocked_report_for_incomplete_inputs(tmp_path: Path):
    _make_price(tmp_path)
    result = audit_campaign(tmp_path, min_symbols=500, min_years=10)
    assert result["status"] == "DATA_BLOCKED"
    assert "insufficient_price_symbols" in result["errors"]
    report = json.loads((tmp_path / "pit_data_quality_report.json").read_text())
    assert report["status"] == "DATA_BLOCKED"


def test_audit_requires_real_capacity_schema(tmp_path: Path):
    _make_price(tmp_path, n_symbols=500, years=10)
    (tmp_path / "raw" / "financials").mkdir(parents=True)
    pd.DataFrame({"code": ["000001"], "report_period": ["2020-01-01"], "announcement_date": ["2020-02-01"], "source_document_id": ["doc"], "source_as_of": ["2020-02-02"]}).to_parquet(tmp_path / "raw" / "financials" / "financials.parquet", index=False)
    result = audit_campaign(tmp_path, min_symbols=500, min_years=10)
    assert result["status"] == "DATA_BLOCKED"
    assert result["coverage"]["capacity_rows"] > 0
    assert result["coverage"]["capacity_materialization"]["status"] == "materialized_conservative_proxy"
