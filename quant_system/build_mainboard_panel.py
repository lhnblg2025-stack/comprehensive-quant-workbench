"""Build a broad main-board Qlib input panel from local Tencent HFQ shards.

Universe definition: Shanghai main board (60/6) and Shenzhen main board (00),
excluding ChiNext/STAR/Beijing. This is a research universe. Historical ST
intervals and authoritative delisting/trade-state records are not inferred from
current names; the manifest records that limitation explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_price_discovery_panel import _features

ROOT = Path(__file__).resolve().parents[1]


def is_mainboard(code: str) -> bool:
    return str(code).zfill(6).startswith(("00", "60"))


def build(input_dir: str | Path, output_dir: str | Path, *, min_date: str = "2016-01-01") -> dict:
    source = Path(input_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(min_date)
    failures: list[dict] = []
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    target = outdir / "mainboard_price_factor_panel.parquet"
    part_paths: list[Path] = []
    for path in sorted(source.glob("*.parquet")):
        code = path.stem.zfill(6)
        if not is_mainboard(code):
            continue
        try:
            raw = pd.read_parquet(path)
            required = {"date", "open", "high", "low", "close", "volume", "amount"}
            missing = required - set(raw.columns)
            if missing:
                failures.append({"code": code, "error": "missing:" + ",".join(sorted(missing))})
                continue
            data = raw.rename(columns={"open": "hfq_open", "high": "hfq_high", "low": "hfq_low", "close": "hfq_close"}).copy()
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data[data["date"] >= cutoff].dropna(subset=["date", "hfq_open", "hfq_high", "hfq_low", "hfq_close"])
            if len(data) < 252 * 5:
                failures.append({"code": code, "error": "less_than_5_years", "rows": int(len(data))})
                continue
            data["code"] = code
            data["raw_open"] = pd.to_numeric(data["hfq_open"], errors="coerce")
            data["raw_high"] = pd.to_numeric(data["hfq_high"], errors="coerce")
            data["raw_low"] = pd.to_numeric(data["hfq_low"], errors="coerce")
            data["raw_close"] = pd.to_numeric(data["hfq_close"], errors="coerce")
            # Tencent HFQ volume/amount are scaled units. Ratios and rolling
            # participation are scale-invariant; absolute execution remains HFQ
            # research-only and is not treated as live cash accounting.
            panel_part = _features(data)
            panel_part["trade_state_source"] = "derived_missing_bar_and_board_rule_research"
            panel_part["is_suspended"] = False
            panel_part["limit_up"] = np.nan
            panel_part["limit_down"] = np.nan
            table = pa.Table.from_pandas(panel_part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(target), table.schema, compression="zstd")
            writer.write_table(table)
            part_paths.append(path)
            del raw, data, panel_part, table
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "error": f"{type(exc).__name__}:{str(exc)[:160]}"})
    if writer is not None:
        writer.close()
    if not part_paths:
        raise RuntimeError("no_mainboard_shards")
    # Read only the compact date/code columns to produce the manifest.
    coverage_frame = pd.read_parquet(target, columns=["date", "code"])
    yearly = coverage_frame.groupby(coverage_frame["date"].dt.year)["code"].nunique().to_dict()
    panel_rows = len(coverage_frame)
    panel_symbols = coverage_frame["code"].nunique()
    date_min = coverage_frame["date"].min()
    date_max = coverage_frame["date"].max()
    del coverage_frame
    report = {
        "schema": "mainboard_price_factor_panel/v1",
        "status": "research_only",
        "universe": "mainboard_60_00",
        "new_account_filter": "main_board_only; historical_ST_and_authoritative_delisting_not_available",
        "symbols": int(panel_symbols),
        "rows": int(panel_rows),
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "yearly_cross_section": {str(k): int(v) for k, v in yearly.items()},
        "failed_symbols": failures,
        "signal_price": "tencent_hfq_frozen_snapshot",
        "execution_price": "tencent_hfq_research_proxy",
        "blocked_for_production": ["authoritative_trade_state_missing", "historical_ST_intervals_missing", "raw_10y_execution_price_missing"],
    }
    (outdir / "manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data_warehouse" / "kline_hfq"))
    parser.add_argument("--output", default=str(ROOT / "data_warehouse" / "research_panels" / "mainboard_10y"))
    parser.add_argument("--min-date", default="2016-01-01")
    args = parser.parse_args(argv)
    build(args.input, args.output, min_date=args.min_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
