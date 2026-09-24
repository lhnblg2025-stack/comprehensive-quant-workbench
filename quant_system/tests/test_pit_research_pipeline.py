from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_system.research_pipeline import run_config


def _write_config(tmp_path: Path, panel_dir: Path, fundamental_path: Path, industry_path: Path, benchmark_path: Path) -> Path:
    path = tmp_path / "pit.yaml"
    path.write_text(f"""schema_version: '1.0'
experiment:
  id: pit_pipeline
  random_seed: 7
data:
  panel_dir: {panel_dir}
  min_stocks: 6
pit:
  required: true
  fundamentals: {fundamental_path}
  industry_history: {industry_path}
  availability_lag_sessions: 1
  min_industry_size: 3
factors:
  names: [value_composite_industry_neutral, quality_composite_industry_neutral]
  directions: {{value_composite_industry_neutral: 1, quality_composite_industry_neutral: 1}}
  neutralization: industry
  industry_neutralize: [value_composite, quality_composite]
labels:
  horizons: [1]
validation:
  train_days: 3
  validation_days: 1
  oos_days: 2
  step_days: 1
  purge_days: 0
  embargo_days: 0
  min_oos_windows: 1
portfolio:
  default_quantile: 0.3
  quantiles: [0.3]
  rebalance: monthly
execution:
  commission_bps: 0
  slippage_bps: 0
benchmark:
  type: custom_return_series
  source: {benchmark_path}
  require_alignment: true
outputs:
  root: {tmp_path / 'runs'}
""", encoding="utf-8")
    return path


def test_pit_pipeline_emits_fundamental_industry_coverage_and_aligned_benchmark(tmp_path: Path):
    dates = pd.bdate_range("2024-04-01", periods=9)
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    rows = []
    for day, date in enumerate(dates):
        for stock in range(6):
            rows.append({"date": date, "code": f"{stock + 1:06d}", "close": 10 + stock + day})
    pd.DataFrame(rows).to_parquet(panel_dir / "panel.parquet", index=False)
    fundamentals = pd.DataFrame({
        "code": [f"{stock + 1:06d}" for stock in range(6)],
        "report_period": ["2024-03-31"] * 6,
        "announcement_date": ["2024-04-01"] * 6,
        "eps_ttm": [1, 2, 3, 4, 5, 6],
        "book_value_per_share": [5, 5, 5, 5, 5, 5],
        "operating_cashflow_per_share": [1, 2, 3, 4, 5, 6],
        "roe": [1, 2, 3, 4, 5, 6],
        "net_margin": [1, 2, 3, 4, 5, 6],
        "cfo_to_net_income": [1, 2, 3, 4, 5, 6],
        "debt_to_assets": [6, 5, 4, 3, 2, 1],
    })
    fundamental_path = tmp_path / "fundamentals.parquet"
    fundamentals.to_parquet(fundamental_path, index=False)
    industries = pd.DataFrame({
        "code": [f"{stock + 1:06d}" for stock in range(6)],
        "industry": ["A"] * 3 + ["B"] * 3,
        "effective_date": ["2024-01-01"] * 6,
    })
    industry_path = tmp_path / "industries.parquet"
    industries.to_parquet(industry_path, index=False)
    benchmark_path = tmp_path / "benchmark.csv"
    pd.DataFrame({"date": dates, "return": [0.0] * len(dates)}).to_csv(benchmark_path, index=False)
    result = run_config(_write_config(tmp_path, panel_dir, fundamental_path, industry_path, benchmark_path))
    assert result["pit_summary"]["status"] == "PASS"
    assert result["pit_summary"]["coverage"]["factors"]["value_composite_industry_neutral"]["usable_dates"] > 0
    assert result["benchmark_summary"]["missing"] == 0
    assert result["factor_specs"][0]["name"] == "value_composite_industry_neutral"


def test_pit_pipeline_rejects_misaligned_required_benchmark(tmp_path: Path):
    dates = pd.bdate_range("2024-04-01", periods=9)
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    rows = [{"date": date, "code": f"{stock + 1:06d}", "close": 10 + stock} for date in dates for stock in range(6)]
    pd.DataFrame(rows).to_parquet(panel_dir / "panel.parquet", index=False)
    fundamental_path = tmp_path / "fundamentals.parquet"
    pd.DataFrame({"code": [f"{stock + 1:06d}" for stock in range(6)], "report_period": ["2024-03-31"] * 6, "announcement_date": ["2024-04-01"] * 6, "eps_ttm": range(1, 7), "book_value_per_share": range(1, 7), "operating_cashflow_per_share": range(1, 7), "roe": range(1, 7), "net_margin": range(1, 7), "cfo_to_net_income": range(1, 7), "debt_to_assets": range(1, 7)}).to_parquet(fundamental_path, index=False)
    industry_path = tmp_path / "industries.parquet"
    pd.DataFrame({"code": [f"{stock + 1:06d}" for stock in range(6)], "industry": ["A"] * 3 + ["B"] * 3, "effective_date": ["2024-01-01"] * 6}).to_parquet(industry_path, index=False)
    benchmark_path = tmp_path / "benchmark.csv"
    pd.DataFrame({"date": dates[:-1], "return": [0.0] * (len(dates) - 1)}).to_csv(benchmark_path, index=False)
    with pytest.raises(ValueError, match="benchmark_alignment_required"):
        run_config(_write_config(tmp_path, panel_dir, fundamental_path, industry_path, benchmark_path))
