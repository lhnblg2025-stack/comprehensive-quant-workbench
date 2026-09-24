"""Immutable data release gate for production research consumers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .product_contract import PRODUCT_VERSION, release_metadata

CST = timezone(timedelta(hours=8))
SCHEMA = "quant-data-release/v1"


@dataclass(frozen=True)
class DomainSpec:
    name: str
    path: str
    max_lag_days: int
    required: bool = True
    min_coverage: float = 1.0
    mode: str = "file"  # file | directory | quarterly_directory | calendar | overseas


DEFAULT_SPECS = (
    DomainSpec("market_index", "data_warehouse/market/index_daily_上证指数.parquet", 0),
    DomainSpec("benchmark_csi300", "data_warehouse/market/index_daily_沪深300.parquet", 0),
    DomainSpec("market_breadth", "data_warehouse/market/zt_daily_stats.parquet", 0),
    DomainSpec("trade_calendar", "quant_system/data/trade_calendar.csv", 0, mode="calendar"),
    DomainSpec("valuation", "data_warehouse/valuation", 5, min_coverage=0.95, mode="directory"),
    DomainSpec("financial", "data_warehouse/financial", 120, min_coverage=0.95, mode="quarterly_directory"),
    # Overseas is a decision-support sleeve, not a hard blocker for domestic
    # close reports; its own scorer remains fail-closed when coverage is poor.
    DomainSpec("overseas", "generated", 3, required=False, min_coverage=1.0, mode="overseas"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) and value > 1_000_000_000:
            return datetime.fromtimestamp(value, timezone.utc).date()
        text = str(value).strip()
        if len(text) >= 8 and text[:8].isdigit() and "-" not in text[:8]:
            return datetime.strptime(text[:8], "%Y%m%d").date()
        return pd.Timestamp(text).date()
    except Exception:
        return None


def _weekday_lag(observed: date | None, expected: date) -> int | None:
    if observed is None:
        return None
    if observed > expected:
        return -1
    lag, cursor = 0, observed
    while cursor < expected:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            lag += 1
    return lag


def _frame_date(path: Path) -> date | None:
    """Read an observation date cheaply from parquet statistics/schema."""
    try:
        if path.suffix == ".csv":
            frame = pd.read_csv(path, usecols=lambda c: c in {"date", "trade_date", "日期"})
            columns = list(frame.columns)
            for column in ("date", "trade_date", "日期"):
                if column in columns:
                    values = pd.to_datetime(frame[column], errors="coerce").dropna()
                    if not values.empty:
                        return values.max().date()
            return None
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        schema = list(parquet.schema_arrow.names)
        # Financial snapshots encode quarter-end periods as column names.
        period_columns = [_parse_date(column) for column in schema if str(column).isdigit() and len(str(column)) == 8]
        period_columns = [value for value in period_columns if value]
        if period_columns:
            return max(period_columns)
        column = next((c for c in ("date", "trade_date", "日期", "报告期", "report_date", "公告日期") if c in schema), None)
        if column is None:
            return None
        index = schema.index(column)
        values = []
        for row_group_index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(row_group_index).column(index).statistics
            if stats and stats.max is not None:
                values.append(stats.max)
        if values:
            parsed = [_parse_date(value) for value in values]
            parsed = [value for value in parsed if value]
            if parsed:
                return max(parsed)
        frame = pd.read_parquet(path, columns=[column])
        parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
        return parsed.max().date() if not parsed.empty else None
    except Exception:
        return None


def _file_state(root: Path, spec: DomainSpec, expected: date) -> dict[str, Any]:
    path = root / spec.path
    if not path.exists() or not path.is_file():
        return {"status": "BLOCK" if spec.required else "WARN", "reason": "missing", "path": spec.path}
    observed = _frame_date(path)
    lag = _weekday_lag(observed, expected)
    status = "PASS" if lag is not None and 0 <= lag <= spec.max_lag_days else "BLOCK"
    return {
        "status": status, "path": spec.path, "observed_at": observed.isoformat() if observed else None,
        "lag_weekdays": lag, "max_lag_days": spec.max_lag_days,
        "bytes": path.stat().st_size, "sha256": _sha256(path), "missing_ratio": 0.0,
    }


def _directory_state(root: Path, spec: DomainSpec, expected: date) -> dict[str, Any]:
    directory = root / spec.path
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not files:
        return {"status": "BLOCK" if spec.required else "WARN", "reason": "missing_or_empty", "path": spec.path}
    observed_dates: list[date] = []
    hashes = hashlib.sha256()
    valid = 0
    for path in files:
        # Thousands of per-symbol files make full content hashing too expensive
        # for a daily gate. Use the file's own observation date when present;
        # mtime is only a legacy fallback and never overrides point-in-time data.
        stat = path.stat()
        observed = _frame_date(path) or datetime.fromtimestamp(stat.st_mtime, CST).date()
        observed_dates.append(observed)
        lag = _weekday_lag(observed, expected)
        if lag is not None and 0 <= lag <= spec.max_lag_days:
            valid += 1
        hashes.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    sample = files[:3] + files[-3:] if len(files) > 6 else files
    for path in sample:
        hashes.update(path.name.encode()); hashes.update(_sha256(path).encode())
    coverage = valid / len(files)
    latest = max(observed_dates) if observed_dates else None
    status = "PASS" if coverage >= spec.min_coverage else "BLOCK"
    return {
        "status": status, "path": spec.path, "observed_at": latest.isoformat() if latest else None,
        "max_lag_days": spec.max_lag_days, "files": len(files), "valid_files": valid,
        "coverage": round(coverage, 6), "min_coverage": spec.min_coverage,
        "missing_ratio": round(1 - coverage, 6), "sha256": hashes.hexdigest(),
    }


def _calendar_state(root: Path, spec: DomainSpec, expected: date) -> dict[str, Any]:
    path = root / spec.path
    if not path.exists():
        return {"status": "BLOCK", "reason": "missing", "path": spec.path}
    try:
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        column = next((c for c in ("trade_date", "date", "日期") if c in frame.columns), None)
        dates = set(pd.to_datetime(frame[column], errors="coerce").dropna().dt.date) if column else set()
    except Exception as exc:
        return {"status": "BLOCK", "reason": f"read_error:{type(exc).__name__}", "path": spec.path}
    return {
        "status": "PASS" if expected in dates else "BLOCK", "path": spec.path,
        "expected_day_present": expected in dates, "calendar_min": min(dates).isoformat() if dates else None,
        "calendar_max": max(dates).isoformat() if dates else None, "rows": len(frame),
        "missing_ratio": 0.0 if expected in dates else 1.0, "sha256": _sha256(path),
    }


def _quarterly_directory_state(root: Path, spec: DomainSpec, expected: date) -> dict[str, Any]:
    directory = root / spec.path
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not files:
        return {"status": "BLOCK", "reason": "missing_or_empty", "path": spec.path}
    valid = 0; readable = 0; latest_periods: list[date] = []; fingerprint = hashlib.sha256()
    for path in files:
        try:
            import pyarrow.parquet as pq
            columns = list(pq.ParquetFile(path).schema_arrow.names)
            readable += 1
            periods = [_parse_date(c) for c in columns if str(c).isdigit() and len(str(c)) == 8]
            periods = [period for period in periods if period and period <= expected]
            latest = max(periods) if periods else None
        except Exception:
            latest = None
        if latest:
            latest_periods.append(latest)
            if (expected - latest).days <= spec.max_lag_days:
                valid += 1
        elif readable:
            # A readable legacy financial snapshot without an explicit period
            # remains usable only as a degraded, auditable observation.
            valid += 1
        stat = path.stat(); fingerprint.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    coverage = valid / len(files)
    sample = files[:3] + files[-3:] if len(files) > 6 else files
    for path in sample:
        fingerprint.update(path.name.encode()); fingerprint.update(_sha256(path).encode())
    return {
        "status": "PASS" if coverage >= spec.min_coverage else "BLOCK", "path": spec.path,
        "observed_at": max(latest_periods).isoformat() if latest_periods else None,
        "files": len(files), "readable_files": readable, "valid_files": valid, "coverage": round(coverage, 6),
        "min_coverage": spec.min_coverage, "max_lag_days": spec.max_lag_days,
        "missing_ratio": round(1 - coverage, 6), "sha256": fingerprint.hexdigest(),
    }


def _overseas_state(root: Path, spec: DomainSpec, expected: date) -> dict[str, Any]:
    candidates = sorted((root / spec.path).glob("overseas_20??-??-??.json"))
    eligible = [path for path in candidates if _parse_date(path.stem.replace("overseas_", "")) <= expected]
    if not eligible:
        return {"status": "BLOCK", "reason": "missing", "path": spec.path}
    path = eligible[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "BLOCK", "reason": f"read_error:{type(exc).__name__}", "path": str(path.relative_to(root))}
    quotes = payload.get("quotes") or []
    mapped = []
    invalid = []
    for quote in quotes:
        observed = _parse_date(quote.get("observed_at") or quote.get("date"))
        # US/commodity close is consumed by the next China session; HK is same-day.
        label = str(quote.get("label") or "")
        is_hk = label in {"恒生指数", "恒生科技", "腾讯控股", "阿里巴巴", "美团", "京东"}
        china_session = observed if is_hk else (observed + timedelta(days=1) if observed else None)
        while china_session and china_session.weekday() >= 5:
            china_session += timedelta(days=1)
        lag = _weekday_lag(china_session, expected)
        item = {"label": label, "observed_at": observed.isoformat() if observed else None,
                "china_session": china_session.isoformat() if china_session else None, "lag_weekdays": lag}
        if lag is None or lag < 0 or lag > spec.max_lag_days:
            invalid.append(item)
        else:
            mapped.append(item)
    coverage = len(mapped) / len(quotes) if quotes else 0.0
    score_eligible = payload.get("score_eligible")
    status = "PASS" if quotes and coverage >= spec.min_coverage and score_eligible is not False else ("WARN" if not spec.required else "BLOCK")
    return {
        "status": status, "path": str(path.relative_to(root)), "observed_at": path.stem.replace("overseas_", ""),
        "quotes": len(quotes), "valid_quotes": len(mapped), "coverage": round(coverage, 6),
        "min_coverage": spec.min_coverage, "missing_ratio": round(1 - coverage, 6),
        "invalid_quotes": invalid, "cross_timezone_mapping": "US/commodity close -> next China weekday; HK -> same day",
        "sha256": _sha256(path),
    }


def build_release(root: str | Path, expected_day: str, specs: tuple[DomainSpec, ...] = DEFAULT_SPECS) -> dict[str, Any]:
    workspace = Path(root).resolve()
    expected = pd.Timestamp(expected_day).date()
    domains: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.mode == "directory":
            state = _directory_state(workspace, spec, expected)
        elif spec.mode == "quarterly_directory":
            state = _quarterly_directory_state(workspace, spec, expected)
        elif spec.mode == "calendar":
            state = _calendar_state(workspace, spec, expected)
        elif spec.mode == "overseas":
            state = _overseas_state(workspace, spec, expected)
        else:
            state = _file_state(workspace, spec, expected)
        domains[spec.name] = state
    errors = [name for name, state in domains.items() if state.get("status") == "BLOCK"]
    warnings = [name for name, state in domains.items() if state.get("status") == "WARN"]
    fingerprint = {
        "schema": SCHEMA, "gate_revision": "r3", "product_version": PRODUCT_VERSION, "expected_day": expected.isoformat(),
        "sources": {name: state.get("sha256") for name, state in sorted(domains.items())},
    }
    release_id = f"{expected.isoformat()}-{hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]}"
    manifest = {
        **release_metadata(), "schema": SCHEMA, "release_id": release_id,
        "expected_day": expected.isoformat(), "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "status": "PASS" if not errors else "BLOCK", "domains": domains, "errors": errors, "warnings": warnings,
        "fingerprint": fingerprint,
    }
    release_dir = workspace / "generated" / "data_releases" / release_id
    manifest_path = release_dir / "manifest.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(f"immutable release collision: {release_id}")
        manifest = existing
    else:
        _atomic_write(manifest_path, serialized)
    if manifest["status"] == "PASS":
        _atomic_write(workspace / "generated" / "data_releases" / "latest.json", json.dumps({
            "schema": "quant-data-release-pointer/v1", "release_id": release_id,
            "expected_day": expected.isoformat(), "manifest": str(manifest_path.relative_to(workspace)),
        }, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_release(root: str | Path, *, release_id: str | None = None, expected_day: str | None = None) -> dict[str, Any]:
    workspace = Path(root).resolve()
    if release_id:
        path = workspace / "generated" / "data_releases" / release_id / "manifest.json"
    else:
        pointer = json.loads((workspace / "generated" / "data_releases" / "latest.json").read_text(encoding="utf-8"))
        path = workspace / pointer["manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"data release blocked: {manifest.get('errors')}")
    if expected_day and manifest.get("expected_day") != pd.Timestamp(expected_day).date().isoformat():
        raise RuntimeError(f"data release date mismatch: {manifest.get('expected_day')} != {expected_day}")
    return manifest


def prune_releases(root: str | Path, *, keep: int = 90) -> dict[str, Any]:
    """Remove only old, unreferenced release manifests; never touch research data."""
    workspace = Path(root).resolve(); base = workspace / "generated" / "data_releases"
    releases = sorted((path for path in base.iterdir() if path.is_dir() and (path / "manifest.json").exists()),
                      key=lambda path: path.stat().st_mtime, reverse=True) if base.is_dir() else []
    latest = _read_latest_id(base)
    removed = []
    for path in releases[keep:]:
        if path.name == latest:
            continue
        # A release referenced from any retained review is evidence and cannot be pruned.
        referenced = any(path.name in review.read_text(encoding="utf-8", errors="ignore")
                         for review in (workspace / "generated").glob("review_*.json"))
        if referenced:
            continue
        for child in path.iterdir():
            child.unlink()
        path.rmdir(); removed.append(path.name)
    return {"kept": len(releases) - len(removed), "removed": removed, "latest": latest}


def _read_latest_id(base: Path) -> str | None:
    try:
        return json.loads((base / "latest.json").read_text(encoding="utf-8")).get("release_id")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", nargs="?")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--prune", type=int, metavar="KEEP", help="keep newest unreferenced releases")
    args = parser.parse_args()
    if args.prune is not None:
        print(json.dumps(prune_releases(args.root, keep=args.prune), ensure_ascii=False)); return 0
    if not args.day:
        parser.error("day is required unless --prune is used")
    release = build_release(args.root, args.day)
    print(json.dumps({"release_id": release["release_id"], "status": release["status"], "errors": release["errors"], "warnings": release.get("warnings", [])}, ensure_ascii=False))
    return 0 if release["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
