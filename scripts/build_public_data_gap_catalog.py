#!/usr/bin/env python3
"""Build the auditable public-data gap catalog without downloading data."""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_system.public_data_gap_catalog import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGING_ROOT,
    build_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = build_catalog(args.staging_root, args.output_root)
    print(
        f"catalog={args.output_root / 'manifest.json'} "
        f"blockers={len(report['admission_blockers'])} counts={report['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
