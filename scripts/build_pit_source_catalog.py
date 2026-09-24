#!/usr/bin/env python3
from __future__ import annotations
import json
from quant_system.pit_source_catalog import build
if __name__ == "__main__":
    report = build()
    print(json.dumps({"status": report["status"], "research_ready_domains": report["research_ready_domains"], "admission_blocking_domains": report["admission_blocking_domains"]}, ensure_ascii=False))
