from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.data_release import DomainSpec, build_release, load_release, prune_releases


def _write_prices(path: Path, day: str):
    pd.DataFrame({"date": [day], "close": [1.0]}).to_parquet(path, index=False)


def test_release_is_immutable_and_loadable(tmp_path):
    (tmp_path / "data_warehouse/market").mkdir(parents=True)
    (tmp_path / "quant_system/data").mkdir(parents=True)
    _write_prices(tmp_path / "data_warehouse/market/a.parquet", "2026-08-28")
    pd.DataFrame({"trade_date": ["2026-08-28", "2026-12-31"]}).to_csv(tmp_path / "quant_system/data/calendar.csv", index=False)
    specs = (
        DomainSpec("market", "data_warehouse/market/a.parquet", 0),
        DomainSpec("calendar", "quant_system/data/calendar.csv", 0, mode="calendar"),
    )
    first = build_release(tmp_path, "2026-08-28", specs)
    second = build_release(tmp_path, "2026-08-28", specs)
    assert first["status"] == "PASS"
    assert first["release_id"] == second["release_id"]
    assert load_release(tmp_path, expected_day="2026-08-28")["release_id"] == first["release_id"]


def test_release_prune_keeps_referenced_manifest(tmp_path):
    (tmp_path / "generated").mkdir()
    for name in ("old", "new"):
        target = tmp_path / "generated/data_releases" / name
        target.mkdir(parents=True)
        (target / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "generated/review_2026-08-28.json").write_text('{"release_id":"old"}', encoding="utf-8")
    (tmp_path / "generated/data_releases/latest.json").write_text('{"release_id":"new"}', encoding="utf-8")
    result = prune_releases(tmp_path, keep=1)
    assert (tmp_path / "generated/data_releases/old").exists()
    assert result["removed"] == []


def test_release_blocks_stale_core_market(tmp_path):
    (tmp_path / "data_warehouse/market").mkdir(parents=True)
    _write_prices(tmp_path / "data_warehouse/market/a.parquet", "2026-08-27")
    result = build_release(tmp_path, "2026-08-28", (DomainSpec("market", "data_warehouse/market/a.parquet", 0),))
    assert result["status"] == "BLOCK"
    assert result["errors"] == ["market"]
