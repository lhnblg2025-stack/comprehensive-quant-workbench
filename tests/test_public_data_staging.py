from __future__ import annotations

import json

from quant_system.public_data_staging import FetchResponse, PublicDatasetSpec, collect_dataset, collect_many


def _fetch_json(spec):
    return FetchResponse(
        body=json.dumps({"data": [{"trade_date": "2024-01-01", "value": 1.0}]}).encode(),
        status=200,
        content_type="application/json",
        final_url=spec.url,
    )


def test_staging_preserves_raw_and_normalized_artifacts(tmp_path):
    spec = PublicDatasetSpec(
        key="calendar",
        url="https://www.sse.com.cn/example",
        data_domain="exchange_calendar",
    )
    record = collect_dataset(spec, tmp_path, fetcher=_fetch_json)
    assert record["status"] == "PASS"
    assert record["rows"] == 1
    assert record["date_min"] == "2024-01-01"
    assert (tmp_path / record["raw_file"]).is_file()
    assert (tmp_path / record["normalized_file"]).is_file()


def test_non_official_source_is_rejected_without_network(tmp_path):
    spec = PublicDatasetSpec(
        key="bad",
        url="https://example.com/data.json",
        data_domain="unknown",
    )
    record = collect_dataset(spec, tmp_path, fetcher=_fetch_json)
    assert record["status"] == "FAILED"
    assert record["failure_reason"] == "non_official_domain_rejected"


def test_collection_manifest_keeps_failed_dataset(tmp_path):
    specs = [
        PublicDatasetSpec("ok", "https://sse.com.cn/a", "exchange_calendar"),
        PublicDatasetSpec("bad", "https://example.com/b", "exchange_calendar"),
    ]
    manifest = collect_many(specs, tmp_path, fetcher=_fetch_json)
    assert len(manifest["datasets"]) == 2
    assert manifest["datasets"][0]["status"] == "PASS"
    assert manifest["datasets"][1]["status"] == "FAILED"
    assert (tmp_path / "manifest.json").is_file()
