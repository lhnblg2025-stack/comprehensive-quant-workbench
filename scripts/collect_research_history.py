#!/usr/bin/env python3
"""Resumable RAW/HFQ collection into an isolated archive, including late listings.

The inventory is a contemporary code inventory, NOT a historical market census.
No minimum-history filter is applied. The manifest keeps this limitation explicit.
Each request runs in a child process so a hung upstream request has a hard timeout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def board(code: str) -> str:
    if code.startswith("30"):
        return "growth"
    if code.startswith("68"):
        return "star"
    if code.startswith(("4", "8", "92")):
        return "beijing"
    return "main"


def universe(inventory: Path, count: int) -> list[str]:
    codes = sorted({p.stem for p in inventory.glob("*.parquet")
                    if len(p.stem) == 6 and p.stem.isdigit()
                    and p.stem.startswith(("00", "30", "60", "68"))
                    and p.stem not in {"000300", "000905", "000852"}})
    # Stable hash sampling avoids choosing only the oldest / lowest stock codes.
    ranked = sorted(codes, key=lambda c: hashlib.sha256(("research-v1:" + c).encode()).hexdigest())
    quotas = {"main": int(count * .65), "growth": int(count * .22)}
    quotas["star"] = count - sum(quotas.values())
    chosen = [c for b, n in quotas.items() for c in [x for x in ranked if board(x) == b][:n]]
    chosen += [c for c in ranked if c not in set(chosen)][:max(count - len(chosen), 0)]
    # Interleave boards so a partially completed download has broad coverage.
    return sorted(chosen, key=lambda c: hashlib.sha256(c.encode()).hexdigest())


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def fetch_one(root: Path, code: str, start: str, end: str) -> None:
    import numpy as np
    import pandas as pd
    from scripts.build_eastmoney_800_8y import fetch_sina
    from quant_system.research_contracts import sha256_file

    raw, hfq, factors = fetch_sina(code, start.replace("-", ""), end.replace("-", ""))
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    for frame in (raw, hfq):
        if frame.empty or not set(required).issubset(frame):
            raise ValueError("empty_or_missing_price_columns")
        if frame.date.duplicated().any() or frame.date.isna().any():
            raise ValueError("invalid_or_duplicate_dates")
        prices = frame[["open", "high", "low", "close"]].to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("invalid_price")
        if (frame.high < frame[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError("invalid_high")
        if (frame.low > frame[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError("invalid_low")
    if not raw.date.equals(hfq.date):
        raise ValueError("raw_hfq_date_mismatch")
    if not np.allclose(hfq.close, raw.close * hfq.adjust_factor, rtol=1e-8):
        raise ValueError("adjustment_mismatch")
    outputs = {}
    for dirname, frame in (("kline_raw", raw), ("kline_hfq", hfq), ("adjust_factors", factors)):
        dest = root / dirname / f"{code}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        tmp.replace(dest)
        outputs[dirname] = {"path": str(dest), "sha256": sha256_file(dest)}
    atomic_json(root / "receipts" / f"{code}.json", {
        "code": code, "board": board(code), "rows": len(raw),
        "observed_start": str(raw.date.min().date()), "observed_end": str(raw.date.max().date()),
        "requested_start": start, "requested_end": end, "files": outputs,
        "source": "sina_via_akshare_raw_and_hfq_factor",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "listing_date_verified": False, "delisting_date_verified": False,
    })


def collect(args: argparse.Namespace) -> dict:
    from quant_system.research_contracts import sha256_file

    root = Path(args.output).resolve()
    (root / "receipts").mkdir(parents=True, exist_ok=True)
    codes = universe(Path(args.inventory), args.count)
    manifest_path = root / "collection_manifest.json"
    manifest = {"schema": "research_history/v1", "requested_start": args.start,
                "requested_end": args.end, "requested_symbols": len(codes), "codes": codes,
                "status": "RUNNING", "completed": {}, "failures": {},
                "limitations": ["contemporary_inventory_not_historical_census",
                                "beijing_not_supported_by_this_provider",
                                "observed_end_is_not_delisting_date"]}
    atomic_json(manifest_path, manifest)

    def task(code: str) -> tuple[str, str | None]:
        receipt = root / "receipts" / f"{code}.json"
        if receipt.exists():
            r = json.loads(receipt.read_text())
            if r.get("requested_start") == args.start and r.get("requested_end") == args.end:
                if all(Path(x["path"]).is_file() and sha256_file(x["path"]) == x["sha256"]
                       for x in r["files"].values()):
                    return code, None
        error = "unknown"
        for _ in range(args.retries):
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", code,
                     "--output", str(root), "--start", args.start, "--end", args.end],
                    capture_output=True, text=True, timeout=args.timeout, cwd=ROOT,
                )
                if result.returncode == 0:
                    return code, None
                error = result.stderr.strip().splitlines()[-1][:240] if result.stderr else "worker_failed"
            except subprocess.TimeoutExpired:
                error = "upstream_timeout"
        return code, error

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, c) for c in codes]
        for i, f in enumerate(as_completed(futures), 1):
            code, error = f.result()
            if error:
                manifest["failures"][code] = error
            else:
                manifest["completed"][code] = json.loads((root / "receipts" / f"{code}.json").read_text())
            if i % 25 == 0 or i == len(codes):
                atomic_json(manifest_path, manifest)
                print(json.dumps({"processed": i, "total": len(codes),
                                  "ok": len(manifest["completed"]),
                                  "failed": len(manifest["failures"])}), flush=True)
    manifest["status"] = "COMPLETE" if not manifest["failures"] else "PARTIAL"
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", default="data_warehouse/kline")
    p.add_argument("--output", default="data_warehouse/research_history_10y")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", required=True)
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--worker")
    a = p.parse_args()
    if a.worker:
        fetch_one(Path(a.output), a.worker, a.start, a.end)
    else:
        if min(a.count, a.workers, a.timeout, a.retries) < 1 or a.workers > 8:
            p.error("positive parameters required; workers must be <= 8")
        collect(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
