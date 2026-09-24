#!/usr/bin/env python3
"""Materialize one chronological research partition with audit metadata.

Development and validation can be read directly.  Holdout rows require a
freeze manifest whose boundary hash matches the requested configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_system.research_upgrade import ResearchBoundaries, audit_point_in_time, evaluate_holdout, split_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="generated/long_price_panel/price_panel.parquet")
    parser.add_argument("--partition", choices=("development", "validation", "holdout"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--holdout-end")
    args = parser.parse_args()
    boundaries = ResearchBoundaries(holdout_end=args.holdout_end)
    frame = pd.read_parquet(ROOT / args.panel)
    if args.partition == "holdout":
        if not args.manifest:
            raise SystemExit("holdout_requires_frozen_manifest")
        selected = evaluate_holdout(frame, boundaries, frozen_manifest=ROOT / args.manifest)
    else:
        selected = split_frame(frame, boundaries, partition=args.partition)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out, index=False)
    report = {"partition": args.partition, "rows": int(len(selected)), "symbols": int(selected["code"].nunique()) if "code" in selected else None, "date_min": str(pd.to_datetime(selected["date"]).min().date()) if len(selected) else None, "date_max": str(pd.to_datetime(selected["date"]).max().date()) if len(selected) else None, "quality": audit_point_in_time(selected, boundaries)}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
