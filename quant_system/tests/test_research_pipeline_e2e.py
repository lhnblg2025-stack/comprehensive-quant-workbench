from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system.research_pipeline import run_config


def test_config_pipeline_writes_manifest_and_marks_short_history(tmp_path: Path):
    data_dir = tmp_path / "panel"
    data_dir.mkdir()
    rows = []
    for day in range(8):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for stock in range(4):
            rows.append({"code": f"{stock:06d}", "date": date.strftime("%Y%m%d"), "close": 10 + day + stock / 10, "mom20": stock})
    pd.DataFrame(rows).to_parquet(data_dir / "snapshot.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(f"""schema_version: '1.0'
experiment:
  id: pipeline_test
  random_seed: 7
data:
  panel_dir: {data_dir}
  quality_gate: warn
factors:
  names: [mom20]
validation:
  train_days: 3
  validation_days: 1
  oos_days: 2
  step_days: 1
  purge_days: 1
  embargo_days: 1
  min_oos_windows: 1
portfolio:
  default_quantile: 0.2
execution:
  commission_bps: 8.5
  slippage_bps: 10
outputs:
  root: {tmp_path / 'runs'}
""", encoding="utf-8")
    result = run_config(config)
    assert result["rolling_oos_status"] == "validated"
    output = tmp_path / "runs" / "pipeline_test"
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"]
    assert manifest["input_files"][0]["sha256"]
    assert (output / "research_report.json").exists()
