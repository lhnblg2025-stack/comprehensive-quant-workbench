#!/usr/bin/env python3
from __future__ import annotations
import json
from quant_system.build_qlib_provider_staging import build
if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "rows": result["rows"], "instruments": result["instruments"], "sessions": result["sessions"], "binary_dump_status": result["binary_dump_status"]}, ensure_ascii=False))
