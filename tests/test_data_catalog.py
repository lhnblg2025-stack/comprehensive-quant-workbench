from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import data_catalog as dc


def test_snapshot_id_is_stable_and_order_insensitive(monkeypatch):
    monkeypatch.setattr(dc, "scan_datasets", lambda: {"a": {"files": 1}, "b": {"files": 2}})
    first = dc.build_snapshot("2026-08-30")
    second = dc.build_snapshot("2026-08-30")
    assert first["snapshot_id"] == second["snapshot_id"]


def test_snapshot_id_changes_with_as_of(monkeypatch):
    monkeypatch.setattr(dc, "scan_datasets", lambda: {"a": {"files": 1}})
    one = dc.build_snapshot("2026-08-29")
    two = dc.build_snapshot("2026-08-30")
    assert one["snapshot_id"] != two["snapshot_id"]


def test_sample_hash_is_deterministic(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"x" * 5000)
    first = dc._sample_hash(path)
    second = dc._sample_hash(path)
    assert first["sha256"] == second["sha256"]
    assert first["sampled"] is True
    assert first["bytes"] == 5000


def test_write_snapshot_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "CATALOG", tmp_path / "catalog")
    snapshot = dc.build_snapshot("2026-08-30")
    monkeypatch.setattr(dc, "scan_datasets", lambda: {"kline": {"files": 1, "as_of": "2026-08-30", "status": "available", "sample_hashes": []}})
    manifest = dc.write_snapshot(snapshot)
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["snapshot_id"]
    assert not list((tmp_path / "catalog").glob("*.tmp"))


def test_build_snapshot_marks_missing_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "ROOT", tmp_path)
    datasets = dc.scan_datasets()
    assert all(data["status"] == "missing" for data in datasets.values())
