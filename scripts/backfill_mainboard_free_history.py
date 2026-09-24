"""Backfill 2016-2017 daily data for the user's cash-equity universe.

Scope is deliberately limited to Shanghai/Shenzhen main-board codes. The
script does not require QMT or broker credentials. Cloud files remain the
primary dataset; free-provider rows are stored separately with provenance and
must pass validation before any merge.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant_system.market_data_providers import MarketDataError, ProviderConfig, fetch_with_fallback


MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
EXCLUDED_PREFIXES = ("300", "301", "688", "689", "8", "4")


@dataclass(frozen=True)
class BackfillConfig:
    start: str = "2016-01-01"
    end: str = "2017-12-31"
    source: str = "free_provider_fallback"
    min_rows: int = 350
    request_interval: float = 0.8
    max_codes: int | None = None
    offset: int = 0

def is_mainboard_code(code: str) -> bool:
    text = str(code).strip().zfill(6)
    return text.startswith(MAINBOARD_PREFIXES) and not text.startswith(EXCLUDED_PREFIXES)


def load_codes(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"code": str})
    if "code" not in frame.columns:
        raise ValueError("universe_missing_code_column")
    codes = sorted({str(code).zfill(6) for code in frame["code"].dropna() if is_mainboard_code(str(code))})
    return codes


def validate_frame(frame: pd.DataFrame, code: str, config: BackfillConfig) -> list[str]:
    errors: list[str] = []
    if frame.empty:
        return ["empty_frame"]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append("missing_columns:" + ",".join(missing))
        return errors
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        errors.append("invalid_or_duplicate_dates")
    if len(frame) < config.min_rows:
        errors.append(f"too_few_rows:{len(frame)}<{config.min_rows}")
    if not ((dates >= pd.Timestamp(config.start)).all() and (dates <= pd.Timestamp(config.end)).all()):
        errors.append("date_outside_requested_window")
    for col in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or (values < 0).any():
            errors.append(f"invalid_numeric:{col}")
    if (pd.to_numeric(frame["high"], errors="coerce") < pd.to_numeric(frame["low"], errors="coerce")).any():
        errors.append("high_below_low")
    if frame.get("code", pd.Series(dtype=str)).astype(str).ne(code).any():
        errors.append("code_mismatch")
    return errors


def _write_atomic(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    frame.to_parquet(temp, index=False)
    temp.replace(target)


def run_backfill(
    universe_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    config: BackfillConfig = BackfillConfig(),
) -> dict:
    codes = load_codes(universe_path)
    if config.offset < 0:
        raise ValueError("offset_must_be_nonnegative")
    if config.offset:
        codes = codes[config.offset:]
    if config.max_codes is not None:
        codes = codes[: config.max_codes]
    output = Path(output_dir)
    manifest_file = Path(manifest_path)
    records: list[dict] = []
    provider_config = ProviderConfig(min_interval=config.request_interval, retries=3)
    for index, code in enumerate(codes, start=1):
        target = output / f"{code}.parquet"
        record = {"code": code, "status": "FAILED", "path": str(target), "attempt": index}
        try:
            if target.exists():
                existing = pd.read_parquet(target)
                errors = validate_frame(existing, code, config)
                if not errors:
                    record.update(status="EXISTING_VALID", rows=len(existing), source=existing.get("source", pd.Series(["unknown"])).iloc[0])
                    records.append(record)
                    continue
            frame = fetch_with_fallback(code, config.start, config.end, "raw", provider_config)
            frame = frame.copy()
            frame["code"] = code
            frame["requested_start"] = config.start
            frame["requested_end"] = config.end
            errors = validate_frame(frame, code, config)
            if errors:
                record["status"] = "INVALID"
                record["errors"] = errors
            else:
                _write_atomic(frame, target)
                record.update(status="DOWNLOADED", rows=len(frame), source=str(frame["source"].iloc[0]), errors=[])
        except Exception as exc:
            record["errors"] = [f"{type(exc).__name__}:{exc}"]
        records.append(record)
        if index % 25 == 0:
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_file.write_text(json.dumps({"config": asdict(config), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.05)
    report = {
        "schema": "mainboard-free-backfill/v1",
        "config": asdict(config),
        "universe_path": str(universe_path),
        "requested_codes": len(codes),
        "downloaded": sum(r["status"] == "DOWNLOADED" for r in records),
        "existing_valid": sum(r["status"] == "EXISTING_VALID" for r in records),
        "invalid": sum(r["status"] == "INVALID" for r in records),
        "failed": sum(r["status"] == "FAILED" for r in records),
        "records": records,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="data_warehouse/raw_hfq_universe.csv")
    parser.add_argument("--output", default="data_warehouse/kline_raw_free_backfill")
    parser.add_argument("--manifest", default="data_warehouse/mainboard_free_backfill_manifest.json")
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.8)
    args = parser.parse_args()
    report = run_backfill(args.universe, args.output, args.manifest, config=BackfillConfig(max_codes=args.max_codes, offset=args.offset, request_interval=args.interval))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
