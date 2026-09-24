"""Build a lifecycle-aware RAW/HFQ price panel.

Unlike the legacy eight-year builder, this module does not require every
security to exist for the whole sample.  New listings and securities whose
history ends before the latest date remain in the panel with explicit
``start_date``/``end_date`` and membership fields.  ``kline`` (QFQ) is never
read; only the genuine unadjusted and backward-adjusted shard directories are
accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE = ("open", "high", "low", "close")


def _read_pair(raw_path: Path, hfq_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    hfq = pd.read_parquet(hfq_path)
    for frame in (raw, hfq):
        if "date" not in frame.columns:
            raise ValueError(f"missing_date:{frame}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        if frame.date.isna().any() or frame.date.duplicated().any():
            raise ValueError("invalid_or_duplicate_price_dates")
    raw = raw.sort_values("date")
    hfq = hfq.sort_values("date")
    raw_dates = set(raw.loc[raw.date.between(start, end), "date"])
    hfq_dates = set(hfq.loc[hfq.date.between(start, end), "date"])
    if raw_dates != hfq_dates:
        raise ValueError("raw_hfq_date_mismatch")
    common = sorted(set(raw["date"]) & set(hfq["date"]))
    if not common:
        return pd.DataFrame()
    rcols = ["date", *[c for c in PRICE if c in raw.columns], "turnover", "volume", "amount"]
    rcols = [c for c in rcols if c in raw.columns]
    hcols = ["date", *[c for c in PRICE if c in hfq.columns], "adjust_factor"]
    hcols = [c for c in hcols if c in hfq.columns]
    r = raw[rcols].rename(columns={**{c: f"raw_{c}" for c in PRICE if c in raw.columns}, "turnover": "raw_turnover"})
    h = hfq[hcols].rename(columns={c: f"hfq_{c}" for c in PRICE if c in hfq.columns})
    frame = h.merge(r, on="date", how="inner").sort_values("date")
    frame = frame[frame["date"].between(start, end)].copy()
    if frame.empty:
        return frame
    code = raw_path.stem.zfill(6)
    frame["code"] = code
    frame["open"] = frame.get("hfq_open")
    frame["high"] = frame.get("hfq_high")
    frame["low"] = frame.get("hfq_low")
    frame["close"] = frame.get("hfq_close")
    frame["turnover_rate"] = pd.to_numeric(frame.get("raw_turnover"), errors="coerce")
    frame["raw_close"] = frame.get("raw_close")
    frame["is_tradable"] = (pd.to_numeric(frame.get("raw_open"), errors="coerce") > 0) & (pd.to_numeric(frame.get("raw_close"), errors="coerce") > 0) & (pd.to_numeric(frame.get("volume"), errors="coerce") > 0)
    frame["is_suspended"] = ~frame["is_tradable"].fillna(False)
    frame["price_adjustment_source"] = "raw_and_hfq_paired_archive"
    frame["start_date"] = frame["date"].min()
    frame["end_date"] = frame["date"].max()
    return frame


def build_lifecycle_panel(raw_dir: str | Path, hfq_dir: str | Path, *, start: str = "2010-01-01", end: str = "2100-01-01", codes: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict]:
    raw_root, hfq_root = Path(raw_dir), Path(hfq_dir)
    if raw_root.name != "kline_raw" or hfq_root.name != "kline_hfq":
        raise ValueError("only_kline_raw_and_kline_hfq_are_allowed")
    if not raw_root.is_dir() or not hfq_root.is_dir():
        raise FileNotFoundError("raw_or_hfq_directory_missing")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    wanted = {str(c).zfill(6) for c in codes} if codes is not None else None
    paths = sorted(raw_root.glob("*.parquet"))
    frames, rejected = [], {}
    for raw_path in paths:
        code = raw_path.stem.zfill(6)
        if wanted is not None and code not in wanted:
            continue
        hfq_path = hfq_root / f"{code}.parquet"
        if not hfq_path.is_file():
            rejected[code] = "missing_hfq_pair"
            continue
        try:
            part = _read_pair(raw_path, hfq_path, start_ts, end_ts)
        except Exception as exc:
            rejected[code] = f"read_error:{type(exc).__name__}:{exc}"
            continue
        if part.empty:
            rejected[code] = "empty_intersection"
            continue
        frames.append(part)
    if not frames:
        raise ValueError("no_lifecycle_rows")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    global_end = panel["date"].max()
    membership = panel.groupby("code", as_index=False).agg(start_date=("date", "min"), end_date=("date", "max"))
    membership["listing_status"] = membership["end_date"].ge(global_end - pd.Timedelta(days=30)).map({True: "active_at_tail", False: "history_ended"})
    panel = panel.drop(columns=["start_date", "end_date"], errors="ignore").merge(membership, on="code", how="left")
    panel["universe_id"] = "lifecycle_raw_hfq"
    report = {"schema": "lifecycle_price_panel.v2", "rows": int(len(panel)), "symbols": int(panel["code"].nunique()), "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()), "active_at_tail": int((membership["listing_status"] == "active_at_tail").sum()), "history_ended": int((membership["listing_status"] == "history_ended").sum()), "rejected": rejected, "limitations": ["observed_history_is_not_official_listing_membership", "history_ended_is_not_verified_delisting", "current_inventory_survivorship_bias_not_eliminated"]}
    return panel, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data_warehouse/kline_raw")
    parser.add_argument("--hfq-dir", default="data_warehouse/kline_hfq")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2100-01-01")
    args = parser.parse_args()
    panel, report = build_lifecycle_panel(args.raw_dir, args.hfq_dir, start=args.start, end=args.end)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
