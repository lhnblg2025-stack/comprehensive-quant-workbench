"""Create immutable revisions of frozen bundles after deterministic PIT fixes."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revise_fixed_universe_membership(source: str | Path, target: str | Path) -> dict:
    src, dst = Path(source), Path(target)
    if dst.exists():
        raise FileExistsError(f"target revision already exists: {dst}")
    dst.mkdir(parents=True)
    for path in src.iterdir():
        if path.is_file() and path.name not in {"data_manifest.json", "quality_report.json", "universe_membership.parquet"}:
            shutil.copy2(path, dst / path.name)

    prices = pd.read_parquet(src / "prices_hfq_raw.parquet").copy()
    membership = pd.read_parquet(src / "universe_membership.parquet").copy()
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    membership["code"] = membership["code"].astype(str).str.zfill(6)
    membership["original_start_date"] = pd.to_datetime(membership["start_date"], errors="coerce")
    first_price = prices.groupby("code", as_index=False)["date"].min().rename(columns={"date": "first_price_date"})
    membership = membership.merge(first_price, on="code", how="left")
    membership["start_date"] = membership["first_price_date"]
    membership["membership_type"] = "fixed_research_universe"
    membership["source"] = "bundle_first_price; revised_from_original_start_date"
    membership.to_parquet(dst / "universe_membership.parquet", index=False)
    return _write_manifest(src, dst, "membership_pit_fix")


def revise_benchmark(source: str | Path, target: str | Path, benchmark_path: str | Path) -> dict:
    """Create a revision with the refreshed benchmark, preserving the source bundle."""
    src, dst, benchmark = Path(source), Path(target), Path(benchmark_path)
    if dst.exists():
        raise FileExistsError(f"target revision already exists: {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("data_manifest.json", "quality_report.json", "benchmark_csi300.parquet"))
    frame = pd.read_parquet(benchmark) if benchmark.suffix == ".parquet" else pd.read_csv(benchmark)
    frame.to_parquet(dst / "benchmark_csi300.parquet", index=False)
    return _write_manifest(src, dst, "benchmark_refresh")


def _write_manifest(src: Path, dst: Path, revision_type: str) -> dict:
    old = json.loads((src / "data_manifest.json").read_text(encoding="utf-8"))
    asset_names = [item["name"] for item in old.get("assets", []) if item["name"] != "data_manifest.json"]
    assets = []
    for name in asset_names:
        path = dst / name
        if path.exists():
            assets.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        **{key: value for key, value in old.items() if key != "assets"},
        "schema_version": "1.2",
        "revision": {
            "type": revision_type,
            "source_bundle": str(src.resolve()),
            "source_manifest_sha256": _sha256(src / "data_manifest.json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "assets": assets,
    }
    (dst / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
