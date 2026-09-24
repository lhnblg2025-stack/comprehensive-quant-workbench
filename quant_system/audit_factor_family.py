"""Audit a factor-family result with BH-FDR, DSR, and split stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .overfitting_tests import deflated_sharpe_ratio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True); parser.add_argument("--output", required=True); parser.add_argument("--trials", type=int, default=27)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.matrix).read_text())
    rows = []
    for item in report["results"]["results"]:
        if "long_only" not in item: continue
        oos = item["long_only"]["oos"]; series_path = Path(args.matrix).parent / "factor" / f"{item['factor']}_daily_returns.csv"
        if not series_path.exists(): continue
        daily = pd.read_csv(series_path, index_col=0, parse_dates=True)["long_only"].dropna()
        sr = float(oos["sharpe"] or 0.0); daily_sr = sr / np.sqrt(252.0)
        dsr, dsr_p = deflated_sharpe_ratio(daily_sr, args.trials, len(daily), float(daily.skew()), float(daily.kurtosis() + 3.0))
        half = len(daily) // 2; first = float(daily.iloc[:half].mean()) if half else 0.0; second = float(daily.iloc[half:].mean()) if half else 0.0
        rows.append({"factor": item["factor"], "oos_sharpe": sr, "oos_excess_annual": item["vs_benchmark"]["oos"]["annual_return"], "ic_mean": item["diagnostics"]["ic_mean"], "daily_observations": len(daily), "dsr": dsr, "dsr_p": dsr_p, "first_half_mean": first, "second_half_mean": second, "same_sign_halves": bool(first * second > 0), "cost_x2_total_return": item.get("cost_x2_total_return")})
    rows.sort(key=lambda x: x["dsr_p"])
    pvals = np.array([x["dsr_p"] for x in rows], dtype=float); m = len(pvals); order = np.argsort(pvals); qvals = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]; running = min(running, pvals[idx] * m / (rank + 1)); qvals[idx] = running
    for row, q in zip(rows, qvals): row["bh_fdr_q"] = float(q); row["family_pass"] = bool(q <= .10 and row["same_sign_halves"])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({"schema": "factor_family_audit/v1", "trials": args.trials, "rows": rows, "family_pass_count": sum(x["family_pass"] for x in rows)}, ensure_ascii=True, indent=2), encoding="utf-8"); print(json.dumps({"rows": len(rows), "family_pass_count": sum(x["family_pass"] for x in rows), "output": str(out)}, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
