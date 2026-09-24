"""Small, auditable collector for explicitly approved public data sources.

This collector is deliberately bounded: callers provide one URL and one
dataset specification, the response is stored verbatim, and normalization is
best-effort with a visible failure record.  It does not merge data into any
research panel and it rejects non-official domains by default.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data_warehouse" / "staging" / "public_data_collect_20260923"
OFFICIAL_DOMAIN_SUFFIXES = (
    "sse.com.cn",
    "szse.cn",
    "cninfo.com.cn",
    "csindex.com.cn",
    "csrc.gov.cn",
    "csrc.gov.cn",
    "chinaclear.cn",
    "hkex.com.hk",
)


@dataclass(frozen=True)
class PublicDatasetSpec:
    key: str
    url: str
    data_domain: str
    params: Mapping[str, str] = ()
    expected_format: str = "auto"
    source_name: str = ""
    max_bytes: int = 50_000_000


@dataclass(frozen=True)
class FetchResponse:
    body: bytes
    status: int
    content_type: str
    final_url: str


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _official_url(url: str) -> bool:
    hostname = (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    return any(hostname == suffix or hostname.endswith("." + suffix) for suffix in OFFICIAL_DOMAIN_SUFFIXES)


def _request(spec: PublicDatasetSpec, timeout: int = 30) -> FetchResponse:
    query = urllib.parse.urlencode(dict(spec.params))
    url = spec.url + (("&" if "?" in spec.url else "?") + query if query else "")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "quant-workbench-public-data/1.0", "Accept": "*/*"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(spec.max_bytes + 1)
        if len(body) > spec.max_bytes:
            raise ValueError(f"response_exceeds_max_bytes:{spec.max_bytes}")
        return FetchResponse(
            body=body,
            status=int(getattr(response, "status", 200)),
            content_type=str(response.headers.get("Content-Type", "")),
            final_url=str(response.geturl()),
        )


def _normalise(body: bytes, content_type: str, expected_format: str) -> tuple[str, bytes, list[str], int, str | None, str | None]:
    fmt = expected_format.lower()
    text = body.decode("utf-8-sig", errors="replace")
    if fmt == "auto":
        lowered = content_type.lower()
        if "json" in lowered or text.lstrip().startswith(("{", "[")):
            fmt = "json"
        elif "csv" in lowered or (text.splitlines() and "," in text.splitlines()[0]):
            fmt = "csv"
        else:
            fmt = "bytes"
    if fmt == "json":
        value = json.loads(text)
        if isinstance(value, dict):
            rows = value.get("data") if isinstance(value.get("data"), list) else [value]
        elif isinstance(value, list):
            rows = value
        else:
            rows = [{"value": value}]
        frame = pd.json_normalize(rows)
        payload = frame.to_json(orient="records", force_ascii=False, date_format="iso").encode("utf-8")
        return "json", payload, list(frame.columns), len(frame), *_date_range(frame)
    if fmt == "csv":
        frame = pd.read_csv(io.BytesIO(body))
        payload = frame.to_json(orient="records", force_ascii=False, date_format="iso").encode("utf-8")
        return "json", payload, list(frame.columns), len(frame), *_date_range(frame)
    if fmt == "parquet":
        frame = pd.read_parquet(io.BytesIO(body))
        sink = io.BytesIO()
        frame.to_parquet(sink, index=False)
        return "parquet", sink.getvalue(), list(frame.columns), len(frame), *_date_range(frame)
    return "bytes", body, [], 0, None, "unparsed_format"


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    for name in ("date", "trade_date", "cal_date", "announced_at", "effective_date"):
        if name not in frame.columns:
            continue
        values = pd.to_datetime(frame[name], errors="coerce").dropna()
        if not values.empty:
            return values.min().date().isoformat(), values.max().date().isoformat()
    return None, None


def collect_dataset(
    spec: PublicDatasetSpec,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    fetcher: Callable[[PublicDatasetSpec], FetchResponse] | None = None,
    allow_non_official: bool = False,
) -> dict[str, Any]:
    """Collect one bounded dataset and always return a manifest record."""
    output = Path(output_root)
    raw_dir = output / "original"
    normalized_dir = output / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        **asdict(spec),
        "params": dict(spec.params),
        "requested_at_utc": retrieved,
        "status": "FAILED",
        "failure_reason": None,
    }
    if not allow_non_official and not _official_url(spec.url):
        record["failure_reason"] = "non_official_domain_rejected"
        return record
    try:
        response = (fetcher or _request)(spec)
        raw_path = raw_dir / f"{spec.key}.response"
        raw_path.write_bytes(response.body)
        record.update({
            "http_status": response.status,
            "content_type": response.content_type,
            "final_url": response.final_url,
            "raw_file": str(raw_path.relative_to(output)),
            "raw_sha256": _sha256_bytes(response.body),
            "raw_bytes": len(response.body),
        })
        if response.status < 200 or response.status >= 300:
            record["failure_reason"] = f"http_status:{response.status}"
            return record
        fmt, payload, columns, rows, date_min, date_max = _normalise(
            response.body, response.content_type, spec.expected_format
        )
        normalized_path = normalized_dir / f"{spec.key}.{fmt}"
        normalized_path.write_bytes(payload)
        record.update({
            "normalized_file": str(normalized_path.relative_to(output)),
            "normalized_sha256": _sha256_bytes(payload),
            "normalized_bytes": len(payload),
            "normalized_format": fmt,
            "columns": columns,
            "rows": rows,
            "date_min": date_min,
            "date_max": date_max,
            "status": "PASS" if fmt != "bytes" else "RESEARCH_ONLY",
            "failure_reason": None,
        })
        if record["failure_reason"] == "unparsed_format":
            record["failure_reason"] = None
        return record
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}:{str(exc)[:300]}"
        return record


def collect_many(
    specs: list[PublicDatasetSpec],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    fetcher: Callable[[PublicDatasetSpec], FetchResponse] | None = None,
    allow_non_official: bool = False,
) -> dict[str, Any]:
    output = Path(output_root)
    records = [
        collect_dataset(
            spec,
            output,
            fetcher=fetcher,
            allow_non_official=allow_non_official,
        )
        for spec in specs
    ]
    manifest = {
        "schema": "public_data_staging/v1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": records,
        "policy": {
            "staging_only": True,
            "main_panel_mutation": False,
            "official_domain_allowlist": list(OFFICIAL_DOMAIN_SUFFIXES),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
