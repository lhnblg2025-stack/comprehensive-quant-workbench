from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_health_module():
    path = ROOT / "quant_system" / "analysis_core" / "data_health_check.py"
    spec = importlib.util.spec_from_file_location("recovery_health_tested", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recovered_file_mtime_does_not_fake_fresh_business_date(tmp_path):
    module = _load_health_module()
    domain = tmp_path / "hot_rank"
    domain.mkdir()
    path = domain / "hot_rank_20260823.parquet"
    pd.DataFrame({"date": ["2026-08-23"], "score": [1]}).to_parquet(path, index=False)
    now = datetime.now(timezone.utc).timestamp()
    os.utime(path, (now, now))
    assert module._latest_business_date(domain) == pd.Timestamp("2026-08-23")


def test_filename_date_is_used_when_parquet_has_no_date_column(tmp_path):
    module = _load_health_module()
    domain = tmp_path / "fund_flow"
    domain.mkdir()
    pd.DataFrame({"value": [1]}).to_parquet(domain / "fund_flow_20260915.parquet", index=False)
    assert module._latest_business_date(domain) == pd.Timestamp("2026-09-15")
