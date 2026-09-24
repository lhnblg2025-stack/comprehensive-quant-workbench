from __future__ import annotations

import json

import pandas as pd

from quant_system.public_data_gap_catalog import build_catalog


def test_catalog_is_fail_closed_and_writes_manifest(tmp_path):
    staging = tmp_path / "staging"
    source = staging / "example"
    source.mkdir(parents=True)
    frame = pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=2), "close": [1.0, 1.1]})
    frame.to_parquet(source / "trade_calendar.parquet", index=False)
    (source / "manifest.json").write_text(
        json.dumps({"datasets": [{"status": "PASS"}]}),
        encoding="utf-8",
    )

    output = tmp_path / "catalog"
    report = build_catalog(staging, output)
    assert report["counts"]["RESEARCH_ONLY"] >= 1
    assert "pit_financials" in report["admission_blockers"]
    assert (output / "manifest.json").is_file()
    assert (output / "README.md").is_file()


def test_missing_staging_domain_is_explicit(tmp_path):
    report = build_catalog(tmp_path / "empty", tmp_path / "catalog")
    item = next(item for item in report["domains"] if item["key"] == "pit_index_membership")
    assert item["status"] == "MISSING"
    assert item["evidence"]["files"] == 0
