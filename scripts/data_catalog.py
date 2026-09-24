#!/usr/bin/env python3
"""DatasetCatalog: versioned snapshots of the local data warehouse.

Large Parquet files are fingerprinted by sampling head/middle/tail bytes rather
than a full read; the manifest explicitly records that the hash is sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog"
CST = timezone(timedelta(hours=8))

DATASETS = {
    "kline": ("data_warehouse/kline", "*.parquet"),
    "industry": ("data_warehouse/industry", "*.parquet"),
    "market": ("data_warehouse/market", "*.parquet"),
    "classification": ("data_warehouse/classification", "*.parquet"),
}

_SAMPLE_SIZE = 1_000_000


def _sample_hash(path: Path, sample_size: int = _SAMPLE_SIZE) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as fh:
        digest.update(fh.read(sample_size))
        if size > sample_size * 2:
            fh.seek(size // 2 - sample_size // 2)
            digest.update(fh.read(sample_size))
        fh.seek(max(0, size - sample_size))
        digest.update(fh.read(sample_size))
    return {"sha256": digest.hexdigest(), "bytes": size, "sampled": True, "sample_size": sample_size}


def _date_from_name(name: str) -> str | None:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None


def scan_datasets() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name, (rel_dir, pattern) in DATASETS.items():
        directory = ROOT / rel_dir
        if not directory.is_dir():
            datasets[name] = {"files": 0, "rows": None, "as_of": None, "status": "missing"}
            continue
        files = sorted(directory.glob(pattern))
        total_rows = None
        latest = None
        hashes = []
        for path in files[:500]:
            hashes.append({"file": str(path.relative_to(ROOT)), **_sample_hash(path)})
            date = _date_from_name(path.name)
            if date and (latest is None or date > latest):
                latest = date
        datasets[name] = {
            "files": len(files),
            "as_of": latest,
            "status": "available" if files else "empty",
            "sample_hashes": hashes,
        }
    return datasets


def build_snapshot(as_of: str, trading_day: str | None = None) -> dict[str, Any]:
    datasets = scan_datasets()
    trading_day = trading_day or as_of
    identity = json.dumps({"as_of": as_of, "trading_day": trading_day, "sources": sorted(datasets)}, sort_keys=True, ensure_ascii=False)
    snapshot_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return {
        "schema": "quant-data-snapshot/v1",
        "snapshot_id": snapshot_id,
        "as_of": as_of,
        "trading_day": trading_day,
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "datasets": datasets,
    }


def write_snapshot(snapshot: dict[str, Any]) -> Path:
    directory = CATALOG / "snapshots" / snapshot["snapshot_id"]
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest)

    checksums = directory / "checksums.txt"
    lines = []
    for name, data in snapshot["datasets"].items():
        for entry in data.get("sample_hashes", []):
            lines.append(f"{entry['sha256']}  {entry['file']} (sampled={entry['sampled']}, bytes={entry['bytes']})")
    checksums.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest


def write_index(snapshot: dict[str, Any]) -> Path:
    index_path = CATALOG / "datasets.json"
    tmp = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"schema": "quant-data-catalog/v1", "latest_snapshot": snapshot["snapshot_id"], "as_of": snapshot["as_of"], "generated_at": snapshot["created_at"], "datasets": snapshot["datasets"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, index_path)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description="build a data warehouse snapshot")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--trading-day")
    args = parser.parse_args()
    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of must be YYYY-MM-DD")
    snapshot = build_snapshot(args.as_of, args.trading_day)
    manifest = write_snapshot(snapshot)
    index = write_index(snapshot)
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "manifest": str(manifest), "index": str(index)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
