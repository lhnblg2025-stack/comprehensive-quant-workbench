"""Consolidated research-grade validation suite.

Runs non-overlapping rolling OOS for the frozen PIT candidates, DSR/FDR for the
OHLCV factor family, and aggregates the corrected ML walk-forward. All outputs
are labelled research_only; production promotion still requires the authoritative
trade-state gate and strict cross-engine reconciliation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "annual_pit_research_panel.parquet"
MATRIX = ROOT / "generated" / "runs" / "broad_ohlcv_factor_matrix_10y" / "broad_factor_matrix.json"
FAMILY_AUDIT = ROOT / "generated" / "runs" / "broad_ohlcv_factor_matrix_10y" / "factor_family_audit.json"
ML = ROOT / "generated" / "runs" / "ml_ohlcv_walk_forward_10y_corrected" / "result.json"
ML_DETAIL = ROOT / "generated" / "runs" / "ml_ohlcv_walk_forward_10y_corrected" / "result.parquet"

FROZEN = ("value_book_to_price_industry_neutral", "quality_low_leverage_industry_neutral", "value_quality_composite_industry_neutral")


def _rolling_oos_frozen() -> list[dict]:
    from quant_system.factor_backtest_runner import _factor_returns_variants
    if not PANEL.is_file():
        return [{"status": "unavailable", "reason": "annual_pit_research_panel_missing"}]
    panel = pd.read_parquet(PANEL)
    rows = []
    for factor in FROZEN:
        variants = _factor_returns_variants(panel, factor, (0.20,), 15.0, 300, 1, "monthly")
        daily = variants.get(0.20, pd.DataFrame())
        if daily.empty or "long_only" not in daily:
            rows.append({"factor": factor, "status": "insufficient_cross_section", "windows": 0, "positive_windows": 0, "median_excess": None, "mean_excess": None})
            continue
        series = daily["long_only"].dropna()
        benchmark = daily["benchmark"].reindex(series.index)
        n = len(series); window = 252; start = 0; window_metrics = []
        while start + window <= n:
            chunk = series.iloc[start:start + window]; bench = benchmark.iloc[start:start + window]
            total_excess = float((1 + (chunk - bench)).prod() - 1)
            window_metrics.append(total_excess)
            start += window
        if window_metrics:
            arr = np.array(window_metrics)
            rows.append({"factor": factor, "status": "complete", "windows": int(len(arr)), "positive_windows": int((arr > 0).sum()), "median_excess": float(np.median(arr)), "mean_excess": float(arr.mean()), "per_window_excess": [float(x) for x in arr]})
        else:
            rows.append({"factor": factor, "status": "insufficient_history", "windows": 0, "positive_windows": 0, "median_excess": None, "mean_excess": None})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8")) if MATRIX.is_file() else {}
    family = json.loads(FAMILY_AUDIT.read_text(encoding="utf-8")) if FAMILY_AUDIT.is_file() else {}
    ml = json.loads(ML.read_text(encoding="utf-8")) if ML.is_file() else {}
    ml_windows = []
    if ML_DETAIL.is_file():
        detail = pd.read_parquet(ML_DETAIL)
        if {"window_start", "window_end", "return"}.issubset(detail.columns):
            ml_windows = detail.groupby(["window_start", "window_end"])["return"].apply(lambda s: float(s.mean())).reset_index(name="mean_return").to_dict("records")
    matrix_trials = 30
    results_config = matrix.get("results", {}).get("config", {}) if isinstance(matrix.get("results"), dict) else {}
    factors_value = results_config.get("factors", [])
    if isinstance(factors_value, (list, tuple)):
        matrix_trials = len(factors_value)
    elif isinstance(factors_value, (int, float)):
        matrix_trials = int(factors_value)
    report = {
        "schema": "research_validation_suite/v1",
        "status": "research_only",
        "frozen_pit_rolling_oos": _rolling_oos_frozen(),
        "ohlcv_factor_family": {
            "matrix_trials": matrix_trials,
            "family_audit": family,
        },
        "ml_challenger": {
            "summary": ml,
            "windows": ml_windows,
        },
        "production_blockers": ["authoritative_trade_state_missing", "strict_cross_engine_reconciliation_block"],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    md = ["# 研究级验证套件", "", f"- 状态: `{report['status']}`", "", "## 冻结 PIT 候选 · 非重叠滚动 OOS", "", "| 因子 | 窗口数 | 正窗口 | 中位超额 | 均值超额 |", "|---|---:|---:|---:|---:|"]
    for row in report["frozen_pit_rolling_oos"]:
        median = row.get("median_excess")
        mean = row.get("mean_excess")
        md.append(f"| {row.get('factor','')} | {row.get('windows',0)} | {row.get('positive_windows',0)} | {'--' if median is None else f'{median:.4f}'} | {'--' if mean is None else f'{mean:.4f}'} |")
    md += ["", "## 生产门禁", ""] + [f"- {item}" for item in report["production_blockers"]]
    output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frozen_windows": [r.get("windows") for r in report["frozen_pit_rolling_oos"]], "blockers": report["production_blockers"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
