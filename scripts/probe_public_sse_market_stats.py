#!/usr/bin/env python3
"""Probe (without claiming availability) the public SSE daily market-stat endpoint."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data_warehouse/staging/public_sse_market_stats_probe_20260923"
URL = "https://query.sse.com.cn/commonSoaQuery.do"
DATES = ["2026-09-23", "2022-01-03"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    (OUTPUT / "original").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "quant-workbench-public-data/1.0", "Referer": "https://www.sse.com.cn/"})
    results = []
    for date in DATES:
        params = {
            "sqlId": "COMMON_SSE_SJ_GPSJ_CJGK_DAYCJGK_C",
            "stockType": "90",
            "searchDate": date,
            "pageHelp.pageSize": "20",
            "pageHelp.pageNo": "1",
            "_": "1",
        }
        retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = OUTPUT / "original" / f"{date}.response"
        try:
            response = session.get(URL, params=params, timeout=30)
            path.write_bytes(response.content)
            text = response.text[:500]
            results.append({
                "query_date": date,
                "url": response.url,
                "params": params,
                "retrieved_at_utc": retrieved,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type"),
                "bytes": len(response.content),
                "sha256": sha256(path),
                "response_prefix": text,
                "status": "SUCCESS_JSON" if response.text.lstrip().startswith("{") else "NON_JSON_OR_JSONP",
            })
        except Exception as exc:
            results.append({
                "query_date": date,
                "url": URL,
                "params": params,
                "retrieved_at_utc": retrieved,
                "status": "FAILED",
                "error": f"{type(exc).__name__}:{str(exc)[:240]}",
            })
    manifest = {
        "schema": "public_sse_market_stats_probe/v1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "上海证券交易所 commonSoaQuery",
        "results": results,
        "data_use_gate": (
            "PROBE_ONLY: no structured daily market-stat dataset was admitted. "
            "A successful HTTP response is not sufficient; the endpoint must return "
            "a stable, parseable historical schema before use in strategy research."
        ),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
