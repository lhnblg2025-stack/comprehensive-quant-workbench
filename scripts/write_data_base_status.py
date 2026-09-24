#!/usr/bin/env python3
"""将所有数据源的真实状态写入统一数据基座状态快照。"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "data_base_status.json"
CST = timezone(timedelta(hours=8))


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from data_contract_registry import run_check
    summary = run_check(verbose=False)
    payload = {
        "schema": "data_base_status/v1",
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "source_count": summary.get("total", 0),
        "healthy": summary.get("ok", 0),
        "degraded": summary.get("degraded", 0),
        "missing": summary.get("missing", 0),
        "stale": summary.get("stale", 0),
        "degraded_ids": summary.get("degraded_ids", []),
        "sources": summary.get("results", []),
        "policy": "所有源保留真实状态；未到有效更新节点不伪造新日期，缺失不补零。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({k: payload[k] for k in ("source_count", "healthy", "degraded", "missing", "stale")}, ensure_ascii=False))
    return 0 if not payload["degraded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
