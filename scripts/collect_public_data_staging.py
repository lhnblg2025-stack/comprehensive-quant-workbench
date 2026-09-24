#!/usr/bin/env python3
"""Collect explicitly listed official public datasets into isolated staging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_system.public_data_staging import PublicDatasetSpec, collect_many


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON list of dataset specifications")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-non-official", action="store_true")
    args = parser.parse_args()
    raw = json.loads(args.spec.read_text(encoding="utf-8"))
    specs = [PublicDatasetSpec(**item) for item in raw]
    manifest = collect_many(specs, args.output_root, allow_non_official=args.allow_non_official)
    failed = sum(item["status"] == "FAILED" for item in manifest["datasets"])
    print(f"manifest={args.output_root / 'manifest.json'} datasets={len(specs)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
