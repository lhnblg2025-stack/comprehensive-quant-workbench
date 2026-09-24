#!/usr/bin/env python3
"""Update daily K-line and valuation stores from a validated close snapshot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data_warehouse"
KLINE = WAREHOUSE / "kline"
VALUATION = WAREHOUSE / "valuation"


def _close_snapshot(day: str) -> Path:
    directory = WAREHOUSE / "realtime_snapshot" / day.replace("-", "")
    files = sorted(directory.glob("*.parquet"))
    eligible = [path for path in files if path.stem[:6] >= "145500"]
    if not eligible:
        raise RuntimeError(f"缺少{day} 14:55后的收盘快照")
    path = eligible[-1]
    rows = len(pd.read_parquet(path, columns=["code"]))
    if rows < 5_000:
        raise RuntimeError(f"收盘快照覆盖不足: {rows} < 5000 ({path.name})")
    return path


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _merge_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    new = pd.DataFrame([row])
    new["date"] = pd.to_datetime(new["date"])
    if path.exists():
        old = pd.read_parquet(path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        for column in columns:
            if column not in old.columns:
                old[column] = pd.NA
        merged = pd.concat([old[columns], new[columns]], ignore_index=True)
    else:
        merged = new[columns]
    merged = merged.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    _atomic_parquet(merged, path)


def update(day: str, *, snapshot: Path | None = None, limit: int = 0) -> dict[str, Any]:
    path = snapshot or _close_snapshot(day)
    frame = pd.read_parquet(path)
    required = {"code", "price", "pre_close", "open", "high", "low", "volume_lot", "amount_wan", "pct_chg", "turnover", "pe_ttm", "pb"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"收盘快照字段缺失: {missing}")
    frame = frame.copy()
    frame["code6"] = frame["code"].astype(str).str[-6:].str.zfill(6)
    frame = frame.drop_duplicates("code6", keep="last")
    if limit:
        frame = frame.head(limit)

    kline_columns = ["date", "open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover", "pct_chg"]
    valuation_columns = ["date", "close", "total_mv", "float_mv", "pe_ttm", "pe_lyr", "pb", "peg", "pcf", "ps", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    kline_ok = valuation_ok = 0
    failures: list[dict[str, str]] = []
    for item in frame.to_dict("records"):
        code = str(item["code6"])
        try:
            close = pd.to_numeric(item.get("price"), errors="coerce")
            if pd.isna(close) or float(close) <= 0:
                continue
            kline_row = {
                "date": day,
                "open": pd.to_numeric(item.get("open"), errors="coerce"),
                "high": pd.to_numeric(item.get("high"), errors="coerce"),
                "low": pd.to_numeric(item.get("low"), errors="coerce"),
                "close": close,
                "volume": pd.to_numeric(item.get("volume_lot"), errors="coerce") * 100.0,
                "amount": pd.to_numeric(item.get("amount_wan"), errors="coerce") * 10_000.0,
                "outstanding_share": pd.NA,
                "turnover": pd.to_numeric(item.get("turnover"), errors="coerce"),
                "pct_chg": pd.to_numeric(item.get("pct_chg"), errors="coerce"),
            }
            _merge_row(KLINE / f"{code}.parquet", kline_row, kline_columns)
            kline_ok += 1

            pe = pd.to_numeric(item.get("pe_ttm"), errors="coerce")
            pb = pd.to_numeric(item.get("pb"), errors="coerce")
            valuation_row = {column: pd.NA for column in valuation_columns}
            valuation_row.update({
                "date": day, "close": close, "pe_ttm": pe, "pb": pb,
                "peTTM": pe, "pbMRQ": pb,
            })
            _merge_row(VALUATION / f"{code}.parquet", valuation_row, valuation_columns)
            valuation_ok += 1
        except Exception as exc:
            failures.append({"code": code, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})

    result = {
        "ok": not failures and kline_ok >= (min(5_000, len(frame)) if not limit else 1),
        "date": day,
        "snapshot": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "snapshot_rows": len(frame),
        "kline_updated": kline_ok,
        "valuation_updated": valuation_ok,
        "failures": failures[:30],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="从收盘全市场快照批量更新日线和估值")
    parser.add_argument("--date", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = update(args.date, snapshot=Path(args.snapshot) if args.snapshot else None, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
