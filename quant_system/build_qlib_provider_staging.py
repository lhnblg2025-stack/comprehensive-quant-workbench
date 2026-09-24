"""Materialize an auditable Qlib provider staging release from local PIT panels.

This builds Qlib-compatible source CSVs plus calendar/instrument metadata. The
binary feature dump remains a separate pyqlib step so missing Qlib dependencies
cannot be mistaken for a completed provider deployment.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .research_contracts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"
SOURCE_PANEL = PANEL_ROOT / "annual_pit_research_panel.parquet"
OUTPUT = PANEL_ROOT / "qlib_provider_staging"

FIELDS = [
    "open", "high", "low", "close", "raw_open", "raw_high", "raw_low", "raw_close",
    "volume", "amount", "value_book_to_price_industry_neutral",
    "quality_low_leverage_industry_neutral", "value_quality_composite_industry_neutral",
]


def build(source: Path = SOURCE_PANEL, output: Path = OUTPUT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"qlib_source_panel_missing:{source}")
    output.mkdir(parents=True, exist_ok=True)
    csv_dir = output / "source_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(source, columns=["date", "code", *FIELDS])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel = panel.dropna(subset=["date", "code", "raw_open", "raw_close"]).sort_values(["code", "date"])
    # Qlib CN instruments convention: SHxxxxxx / SZxxxxxx.
    panel["instrument"] = panel["code"].map(lambda code: ("SH" if code.startswith(("5", "6", "9")) else "SZ") + code)
    for instrument, frame in panel.groupby("instrument", sort=True):
        out = frame[["date", *FIELDS]].rename(columns={"date": "date"})
        out.insert(0, "symbol", instrument)
        out.to_csv(csv_dir / f"{instrument}.csv", index=False)
    calendars = output / "calendars"; calendars.mkdir(exist_ok=True)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    (calendars / "day.txt").write_text("\n".join(day.strftime("%Y-%m-%d") for day in dates) + "\n", encoding="utf-8")
    instruments = output / "instruments"; instruments.mkdir(exist_ok=True)
    intervals = panel.groupby("instrument")["date"].agg(["min", "max"]).reset_index()
    # These intervals are observed price availability, not authoritative lifecycle.
    lines = [f"{row.instrument}\t{row['min'].strftime('%Y-%m-%d')}\t{row['max'].strftime('%Y-%m-%d')}" for _, row in intervals.iterrows()]
    (instruments / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    field_coverage = {field: float(panel[field].notna().mean()) for field in FIELDS}
    manifest = {
        "schema": "qlib_provider_staging/v1",
        "status": "STAGING_RESEARCH_ONLY",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "rows": int(len(panel)),
        "instruments": int(panel["instrument"].nunique()),
        "sessions": int(len(dates)),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "fields": FIELDS,
        "field_coverage": field_coverage,
        "calendar_source": "observed_price_panel_sessions_not_exchange_calendar",
        "instrument_source": "observed_price_intervals_not_authoritative_lifecycle",
        "binary_dump_status": "NOT_RUN_PYQLIB_NOT_INSTALLED",
        "admission_blockers": ["authoritative_instruments_missing", "authoritative_trade_state_missing", "pyqlib_runtime_missing"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
