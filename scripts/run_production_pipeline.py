#!/usr/bin/env python3
"""Run the canonical production pipeline for the latest completed session."""
from __future__ import annotations

import json
from pathlib import Path

from build_data_release import _latest_completed_day
from quant_system.production_pipeline import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    day = _latest_completed_day().isoformat()
    result = run(day)
    print(json.dumps({"status": result["status"], "release_id": result["release_id"], "order_level_audit": result["order_level_audit"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] in ("PASS", "WARN") else 2)
