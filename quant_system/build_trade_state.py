"""Build a deterministic research-grade trade-state table from daily OHLCV.

This is deliberately *not* an authoritative exchange source. It derives:

- suspension: a trading-calendar date with no bar for a symbol inside its
  observed [first_bar, last_bar] window;
- price-limit bounds: board rule (10% main board, 20% ChiNext/STAR) applied to
  the previous close, rounded to 0.01, without ST-specific 5% adjustment;
- limit locks: close pinned at the computed limit price (one-price board).

It unblocks *research* backtest execution (blocked buy at limit-up, blocked sell
at limit-down / suspension). Production paper/live promotion still requires an
authoritative suspension/ST/limit source, which remains recorded as a blocker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "trade_state_derived/v3"
SOURCE_ID = "derived_ohlcv_calendar_board_rules_v3"


def board_rate(code: str) -> float:
    code = str(code).zfill(6)
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10


def build_trade_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a trade-state frame keyed by (code, date)."""
    required = {"date", "code", "raw_open", "raw_high", "raw_low", "raw_close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel_missing:{','.join(missing)}")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for col in ("raw_open", "raw_high", "raw_low", "raw_close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "code"]).sort_values(["code", "date"])

    # Market calendar = union of all observed dates. A symbol is "suspended" on a
    # calendar date that falls between its first and last observed bar but has no bar.
    calendar = pd.DatetimeIndex(sorted(data["date"].unique()))
    cal_series = pd.Series(calendar)

    rows: list[pd.DataFrame] = []
    for code, group in data.groupby("code", sort=False):
        group = group.sort_values("date").set_index("date")
        close = group["raw_close"]
        prior = close.shift(1)
        rate = board_rate(code)
        limit_up_price = (prior * (1.0 + rate)).round(2)
        limit_down_price = (prior * (1.0 - rate)).round(2)
        first, last = close.index[0], close.index[-1]
        # Emit one row per calendar date in [first, last], including suspended
        # dates where the symbol has no bar.
        expected = pd.DatetimeIndex(cal_series[(cal_series >= first) & (cal_series <= last)].to_numpy())
        frame = pd.DataFrame({"code": code, "date": expected})
        frame = frame.join(group[["raw_close"]].rename(columns={"raw_close": "_close"}), on="date")
        frame["suspended"] = frame["_close"].isna()
        frame["suspension_reason"] = np.where(frame["suspended"], "missing_bar_inferred", None)
        frame["st_flag"] = False  # ST history unavailable; 5% ST limit not applied.
        # Rebuild limit prices on the aligned calendar so suspended dates carry the
        # previous close forward, matching exchange price-limit computation.
        aligned_close = frame["_close"].ffill()
        aligned_prior = aligned_close.shift(1)
        frame["limit_up_price"] = (aligned_prior * (1.0 + rate)).round(2)
        frame["limit_down_price"] = (aligned_prior * (1.0 - rate)).round(2)
        frame["limit_up_locked"] = aligned_close.ge(frame["limit_up_price"] - 0.005)
        frame["limit_down_locked"] = aligned_close.le(frame["limit_down_price"] + 0.005)
        frame["source_document_id"] = SOURCE_ID
        frame["source_as_of"] = frame["date"]
        frame = frame.drop(columns=["_close"])
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def report_for(frame: pd.DataFrame) -> dict:
    lock_up = int(frame["limit_up_locked"].sum()) if "limit_up_locked" in frame else 0
    lock_down = int(frame["limit_down_locked"].sum()) if "limit_down_locked" in frame else 0
    suspended = int(frame["suspended"].sum()) if "suspended" in frame else 0
    return {
        "schema": SCHEMA,
        "status": "research_derived_not_authoritative",
        "rows": int(len(frame)),
        "symbols": int(frame["code"].nunique()) if len(frame) else 0,
        "date_min": str(frame["date"].min().date()) if len(frame) else None,
        "date_max": str(frame["date"].max().date()) if len(frame) else None,
        "suspended_rows": suspended,
        "limit_up_locked_rows": lock_up,
        "limit_down_locked_rows": lock_down,
        "source_document_id": SOURCE_ID,
        "blocked_for_production": ["suspension_reason_not_authoritative", "st_history_missing", "official_price_limit_missing"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, help="price panel parquet (date/code/raw_* columns)")
    parser.add_argument("--output", required=True, help="output trade_state.parquet path")
    args = parser.parse_args(argv)
    panel = pd.read_parquet(args.panel)
    frame = build_trade_state(panel)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    rep = report_for(frame)
    report_text = json.dumps(rep, ensure_ascii=True, indent=2)
    out.with_suffix(".report.json").write_text(report_text, encoding="utf-8")
    # Keep the sidecar name stable for a parquet artifact as well.
    out.parent.joinpath(f"{out.stem}.report.json").write_text(report_text, encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
