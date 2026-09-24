#!/usr/bin/env python3
"""Calendar statistics for day / week / month timing questions.

Computes mean daily returns and Newey-West t-statistics by weekday, by calendar
month, by turn-of-month window and by month-phase, for both the CSI 300 index
and the equal-weight market portfolio built from the 800-name panel.  This is
descriptive evidence for "when" rather than "which stock".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stats_by(returns: pd.Series, labels: pd.Series, universe: str, dimension: str) -> list[dict]:
    rows = []
    frame = pd.DataFrame({"ret": returns, "label": labels}).dropna()
    for label, block in frame.groupby("label", sort=True):
        values = block["ret"].to_numpy(dtype=float)
        if len(values) < 5:
            continue
        model = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({
            "universe": universe, "dimension": dimension, "bucket": str(label),
            "observations": int(len(values)),
            "mean_daily_return": float(values.mean()),
            "annualized_mean_return": float(values.mean() * 252.0),
            "t_stat": float(model.tvalues[0]),
            "positive_ratio": float((values > 0).mean()),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="generated/strategy_study_v2_20260922/signal_frame.parquet")
    parser.add_argument("--benchmark", default="data_warehouse/market/index_daily_沪深300.parquet")
    parser.add_argument("--out", default="generated/strategy_study_v2_20260922/calendar_effects.csv")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    index = pd.read_parquet(ROOT / args.benchmark)
    index["date"] = pd.to_datetime(index["date"], errors="coerce")
    index = index.dropna(subset=["date"]).sort_values("date")
    index = index[index["date"].between(start, end)]
    index_ret = pd.to_numeric(index["close"], errors="coerce").pct_change().dropna()
    index_dates = index["date"].iloc[1:][: len(index_ret)]
    index_ret.index = pd.DatetimeIndex(index_dates)

    frame = pd.read_parquet(ROOT / args.frame, columns=["date", "code", "ret_1d"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].between(start, end)]
    market = frame.groupby("date", sort=True)["ret_1d"].mean().dropna()

    rows: list[dict] = []
    for universe, series in (("CSI300", index_ret), ("EW800", market)):
        dates = pd.DatetimeIndex(series.index)
        weekday = dates.dayofweek.to_series(index=series.index).map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        )
        rows += stats_by(series, weekday, universe, "weekday")
        rows += stats_by(series, dates.month.to_series(index=series.index), universe, "month")
        phase = ((dates.day - 1) // 5 + 1).to_series(index=series.index)
        rows += stats_by(series, phase, universe, "month_phase")
        turn = dates.day.to_series(index=series.index).map(lambda d: "turn_of_month" if d <= 5 or d >= 25 else "rest")
        rows += stats_by(series, turn, universe, "turn_of_month")

    out = pd.DataFrame(rows)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False, max_rows=60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
