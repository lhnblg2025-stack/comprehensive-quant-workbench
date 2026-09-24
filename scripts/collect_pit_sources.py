#!/usr/bin/env python3
"""Probe and materialize available PIT source inputs without exposing credentials."""
from __future__ import annotations

import json
from quant_system.pit_source_collector import materialize_permitted_tushare_sources, run

if __name__ == "__main__":
    report = run()
    materialized = materialize_permitted_tushare_sources()
    print(json.dumps({"status": report["status"], "next_action": report["next_action"], "unavailable_or_limited": report["unavailable_or_limited"], "materialized": materialized["domains"]}, ensure_ascii=False))
