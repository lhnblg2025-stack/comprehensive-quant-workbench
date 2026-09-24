#!/usr/bin/env python3
"""Collect full daily bars for public SSE index endpoints into isolated staging.

The endpoint is an official SSE quotation service. This collector deliberately
does not merge the result into stock panels: index bars are benchmarks and
market-regime inputs, not point-in-time stock membership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_warehouse/staging/public_sse_index_daily_20260923"
ENDPOINT = "https://yunhq.sse.com.cn:32042/v1/sh1/dayk/{code}"
INDEXES = {
    "000001": "上证指数",
    "000010": "上证180",
    "000016": "上证50",
    "000300": "沪深300",
    "000688": "科创50",
    "000680": "科创板相关指数",
    "000852": "中证1000",
    "000905": "中证500",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    price_cols = ["open", "high", "low", "close"]
    prices = frame[price_cols].apply(pd.to_numeric, errors="coerce")
    high_expected = prices[["open", "low", "close"]].max(axis=1)
    low_expected = prices[["open", "high", "close"]].min(axis=1)
    return {
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_dates": int(frame["trade_date"].duplicated().sum()),
        "not_strictly_increasing_dates": int((dates.diff().dropna() <= pd.Timedelta(0)).sum()),
        "nonpositive_or_nonfinite_prices": int((~prices.notna().all(axis=1) | (prices <= 0).any(axis=1)).sum()),
        "ohlc_range_violations": int(((prices["high"] < high_expected) | (prices["low"] > low_expected)).sum()),
        "negative_volume_rows": int((pd.to_numeric(frame["volume"], errors="coerce") < 0).sum()),
        "negative_amount_rows": int((pd.to_numeric(frame["amount"], errors="coerce") < 0).sum()),
        "future_date_rows": int((dates > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()).sum()),
    }


def collect(code: str, name: str, output: Path, session: requests.Session, retries: int = 3) -> dict[str, Any]:
    original = output / "original" / f"{code}_dayk.json"
    normalized = output / "normalized" / f"{code}_index_daily.parquet"
    url = ENDPOINT.format(code=code)
    params = {"begin": 0, "end": -1, "period": "day"}
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    last_error: str | None = None
    response: requests.Response | None = None
    payload: Any = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("kline"), list):
                raise ValueError("response has no kline list")
            break
        except Exception as exc:  # network and payload failures remain in manifest
            last_error = f"{type(exc).__name__}:{str(exc)[:240]}"
            if attempt < retries:
                time.sleep(2 * attempt)
    if response is None or not isinstance(payload, dict) or not isinstance(payload.get("kline"), list):
        return {
            "index_code": code,
            "index_name": name,
            "url": url,
            "params": params,
            "retrieved_at_utc": retrieved,
            "status": "FAILED",
            "error": last_error or "unknown fetch failure",
        }

    original.write_bytes(response.content)
    rows = payload["kline"]
    frame = pd.DataFrame(rows, columns=["trade_date_raw", "open", "high", "low", "close", "volume", "amount"])
    frame["trade_date"] = pd.to_datetime(frame.pop("trade_date_raw").astype(str), format="%Y%m%d", errors="coerce")
    frame.insert(0, "index_code", code)
    frame.insert(1, "index_name", name)
    frame["source"] = "SSE_YUNHQ_OFFICIAL"
    frame["source_url"] = response.url
    frame["retrieved_at_utc"] = retrieved
    frame = frame[
        ["index_code", "index_name", "trade_date", "open", "high", "low", "close", "volume", "amount",
         "source", "source_url", "retrieved_at_utc"]
    ]
    quality = validate(frame)
    frame.to_parquet(normalized, index=False)
    return {
        "index_code": code,
        "index_name": name,
        "url": url,
        "params": params,
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "retrieved_at_utc": retrieved,
        "original_file": str(original.relative_to(output)),
        "normalized_file": str(normalized.relative_to(output)),
        "original_sha256": sha256(original),
        "normalized_sha256": sha256(normalized),
        "bytes": len(response.content),
        "rows": int(len(frame)),
        "date_min": frame["trade_date"].min().date().isoformat(),
        "date_max": frame["trade_date"].max().date().isoformat(),
        "columns": list(frame.columns),
        "quality": quality,
        "status": "PASS" if not any(quality.values()) else "QUALITY_REVIEW",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()
    output = args.output
    (output / "original").mkdir(parents=True, exist_ok=True)
    (output / "normalized").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "quant-workbench-public-data/1.0", "Accept": "application/json"})
    datasets = []
    for code, name in INDEXES.items():
        result = collect(code, name, output, session)
        datasets.append(result)
        print(code, result["status"], result.get("rows", 0), result.get("date_min"), result.get("date_max"))
        time.sleep(args.sleep_seconds)
    manifest = {
        "schema": "public_sse_index_daily_staging/v1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "上海证券交易所 / SSE YunHQ",
        "endpoint_template": ENDPOINT,
        "datasets": datasets,
        "data_use_gate": (
            "STAGING_ONLY: official index daily bars may be used for benchmark, "
            "market-regime and index-factor research. They are not historical stock "
            "membership, listing lifecycle, suspension, ST or point-in-time tradability."
        ),
        "survivorship_bias_risk": True,
    }
    manifest_path = output / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    failed = sum(item["status"] == "FAILED" for item in datasets)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
