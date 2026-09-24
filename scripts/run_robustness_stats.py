#!/usr/bin/env python3
"""Statistical robustness for the expanded portfolio matrix.

Reads the simulated equity curves and produces, without any further
simulation:

* block-bootstrap Sharpe confidence intervals and ``P(Sharpe <= 0)``,
* per-calendar-year returns,
* concentration diagnostics (share of the total return coming from the best
  5% of sessions),
* a White Reality Check across all strategies inside each frequency/mode
  group (joint max-Sharpe null distribution from the same bootstrap draws),
* the deflated Sharpe ratio (Bailey / Lopez de Prado) that penalises the
  number of trials the best cell was chosen from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOCK = 21
NBOOT = 1000
RC_BOOT = 500
EULER = 0.5772156649015329


def sharpe(returns: np.ndarray, periods: float = 252.0) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods))


def block_bootstrap_sharpe(returns: np.ndarray, rng: np.random.Generator,
                           n_boot: int = NBOOT, block: int = BLOCK) -> dict:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < block * 3:
        return {}
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, blocks))
    offsets = np.arange(block)
    index = starts[:, :, None] + offsets[None, None, :]
    sample = returns[index].reshape(n_boot, -1)[:, :n]
    means = sample.mean(axis=1)
    stds = sample.std(axis=1, ddof=1)
    valid = stds > 0
    draws = np.full(n_boot, np.nan)
    draws[valid] = means[valid] / stds[valid] * np.sqrt(252.0)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {}
    return {
        "sharpe_boot_mean": float(draws.mean()),
        "sharpe_ci_low": float(np.percentile(draws, 2.5)),
        "sharpe_ci_high": float(np.percentile(draws, 97.5)),
        "probability_sharpe_positive": float((draws > 0).mean()),
        "bootstrap_draws": int(len(draws)),
    }


def reality_check(matrix: np.ndarray, rng: np.random.Generator, n_boot: int = RC_BOOT) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
    if matrix.shape[1] == 0 or matrix.shape[0] < BLOCK * 3:
        return {}
    observed = np.array([sharpe(matrix[:, k]) for k in range(matrix.shape[1])])
    observed_max = float(np.nanmax(observed))
    demeaned = matrix - matrix.mean(axis=0, keepdims=True)
    n = demeaned.shape[0]
    blocks = int(np.ceil(n / BLOCK))
    offsets = np.arange(BLOCK)
    maxima = np.empty(n_boot)
    for draw in range(n_boot):
        starts = rng.integers(0, n - BLOCK + 1, size=blocks)
        index = (starts[:, None] + offsets[None, :]).reshape(-1)[:n]
        sample = demeaned[index]
        stats = np.empty(sample.shape[1])
        for k in range(sample.shape[1]):
            stats[k] = sharpe(sample[:, k])
        maxima[draw] = np.nanmax(stats)
    p_value = float((maxima >= observed_max).mean())
    return {
        "trials": int(matrix.shape[1]),
        "sessions": int(matrix.shape[0]),
        "observed_max_sharpe": observed_max,
        "reality_check_p_value": p_value,
        "null_max_sharpe_median": float(np.median(maxima)),
        "null_max_sharpe_p95": float(np.percentile(maxima, 95)),
    }


def deflated_sharpe(returns: np.ndarray, trials: int, trial_sharpe_variance: float) -> dict:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    sr = sharpe(returns) / np.sqrt(252.0)
    if not np.isfinite(sr) or n < 30 or trial_sharpe_variance <= 0 or trials < 2:
        return {}
    gamma3 = float(pd.Series(returns).skew())
    gamma4 = float(pd.Series(returns).kurt()) + 3.0
    sr0 = np.sqrt(trial_sharpe_variance) * (
        (1.0 - EULER) * norm.ppf(1.0 - 1.0 / trials) + EULER * norm.ppf(1.0 - 1.0 / (trials * np.e))
    )
    denominator = np.sqrt(max(1e-12, 1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr * sr))
    z = (sr - sr0) * np.sqrt(n - 1) / denominator
    return {
        "sharpe_nonannual": float(sr),
        "expected_max_sharpe_under_null": float(sr0),
        "deflated_sharpe_prob": float(norm.cdf(z)),
        "trials": int(trials),
        "trial_sharpe_variance": float(trial_sharpe_variance),
        "skew": gamma3, "kurtosis": float(gamma4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="generated/strategy_study_v2_20260922")
    args = parser.parse_args()
    out_dir = ROOT / args.dir
    curves = pd.read_parquet(out_dir / "portfolio_equity_curves.parquet")
    curves["date"] = pd.to_datetime(curves["date"])
    curves = curves[(curves["window"] == "test_2023_2025") & (curves["mode"] != "raw_gross")]
    matrix = curves.pivot_table(index="date", columns=["strategy", "frequency", "mode"], values="return")
    rng = np.random.default_rng(20260922)

    rows: list[dict] = []
    for column in matrix.columns:
        series = matrix[column].dropna()
        if len(series) < 100:
            continue
        stats = block_bootstrap_sharpe(series.to_numpy(dtype=float), rng)
        yearly = series.groupby(series.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))
        top_share = float(series.nlargest(max(1, int(len(series) * 0.05))).sum() / series.sum()) if series.sum() > 0 else None
        rows.append({
            "strategy": column[0], "frequency": column[1], "mode": column[2],
            "sessions": int(len(series)),
            "sharpe": sharpe(series.to_numpy(dtype=float)),
            "annual_return": float((1.0 + series).prod() ** (252.0 / len(series)) - 1.0),
            "total_return": float((1.0 + series).prod() - 1.0),
            "max_drawdown": float(((1.0 + series).cumprod() / (1.0 + series).cumprod().cummax() - 1.0).min()),
            "year_2023": float(yearly.get(2023, np.nan)), "year_2024": float(yearly.get(2024, np.nan)),
            "year_2025": float(yearly.get(2025, np.nan)),
            "top5pct_return_share": top_share,
            **stats,
        })
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(out_dir / "robustness_bootstrap.csv", index=False)

    rc_rows: list[dict] = []
    group_rows: list[dict] = []
    for (frequency, mode), block in matrix.groupby(level=[1, 2], axis=1):
        valid = block.dropna(how="all", axis=1)
        data = valid.dropna()
        if data.shape[1] < 2:
            continue
        result = reality_check(data.to_numpy(dtype=float), rng)
        if not result:
            continue
        sharpes = np.array([sharpe(data.iloc[:, k].to_numpy(dtype=float)) for k in range(data.shape[1])])
        best_index = int(np.nanargmax(sharpes))
        trial_variance = float(np.nanvar(sharpes, ddof=1)) / 252.0
        dsr = deflated_sharpe(data.iloc[:, best_index].to_numpy(dtype=float), data.shape[1], trial_variance)
        rc_rows.append({"frequency": frequency, "mode": mode, **result, **dsr})
        for k, name in enumerate(data.columns):
            group_rows.append({
                "strategy": name[0], "frequency": frequency, "mode": mode,
                "group_sharpe": float(sharpes[k]),
                "group_rank": int(np.sum(sharpes > sharpes[k]) + 1),
                "reality_check_p_value": result["reality_check_p_value"],
            })
    pd.DataFrame(rc_rows).to_csv(out_dir / "robustness_reality_check.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out_dir / "robustness_group_ranking.csv", index=False)
    print(json.dumps({"bootstrap_rows": int(len(bootstrap)), "reality_check_rows": int(len(rc_rows))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
