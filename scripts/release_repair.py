#!/usr/bin/env python3
"""Bounded domain repair and revalidation for a blocked Data Release Gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("QUANT_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
REPAIRS = {
    "market_index": ["fetch_index_daily.py"],
    "benchmark_csi300": ["fetch_index_daily.py"],
    "valuation": ["update_valuation_baostock.py", "--days", "10"],
    "financial": ["update_financial_quarterly.py"],
    "lhb_hist": ["update_lhb_daily.py", "--date", "{day}"],
    "zt_history": ["backfill_zt_history.py"],
    "overseas": ["overseas_collector.py", "--save"],
}


def repair(day: str, domains: list[str]) -> dict:
    outcomes = {}
    for domain in dict.fromkeys(domains):
        spec = REPAIRS.get(domain)
        if not spec:
            outcomes[domain] = {"status": "no_repairer"}
            continue
        args = [part.format(day=day) for part in spec]
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], cwd=ROOT,
                              capture_output=True, text=True, timeout=900)
        outcomes[domain] = {"status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode,
                            "output_tail": ((proc.stdout or "") + (proc.stderr or ""))[-1000:]}
    from quant_system.data_release import build_release
    release = build_release(ROOT, day)
    return {"schema": "quant-release-repair/v1", "day": day, "repairs": outcomes,
            "release_id": release["release_id"], "release_status": release["status"],
            "errors": release.get("errors", []), "warnings": release.get("warnings", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("day"); parser.add_argument("domains", nargs="+")
    args = parser.parse_args(); result = repair(args.day, args.domains); print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["release_status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
