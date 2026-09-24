"""stress_baseline 域 X 测试。

全部使用合成 parquet / 合成 2000 bar 数据，不触发真实全量温度回放。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_system import stress_baseline as sb


def _write_replay_fixture(root: Path) -> None:
    """构造最小 data_warehouse，供 temperature_replay 小样本回放。"""
    warehouse = root / "data_warehouse"
    kline_dir = warehouse / "kline"
    market_dir = warehouse / "market"
    kline_dir.mkdir(parents=True)
    market_dir.mkdir(parents=True)

    dates = pd.bdate_range("2023-01-01", periods=45)
    for symbol in ["000001", "000002", "000003", "000004", "000005"]:
        close = 10.0 + np.arange(len(dates)) * 0.2 + (int(symbol) % 5)
        df = pd.DataFrame({"date": dates, "close": close})
        df.to_parquet(kline_dir / f"{symbol}.parquet", index=False)

    index_df = pd.DataFrame(
        {
            "date": dates,
            "open": 3000.0 + np.arange(len(dates)) * 5.0,
            "high": 3000.0 + np.arange(len(dates)) * 5.0 + 10.0,
            "low": 3000.0 + np.arange(len(dates)) * 5.0 - 10.0,
            "close": 3000.0 + np.arange(len(dates)) * 5.0,
            "volume": 100000.0,
        }
    )
    index_df.to_parquet(market_dir / "index_daily.parquet", index=False)

    margin = pd.DataFrame(
        {
            "日期": dates,
            "融资余额": np.linspace(1e12, 1.01e12, len(dates)),
        }
    )
    margin.to_parquet(market_dir / "market_margin_sh.parquet", index=False)
    margin.to_parquet(market_dir / "market_margin_sz.parquet", index=False)


def test_backtest_2000bar_runs_and_returns_metrics():
    result = sb.measure_backtest()

    assert result["bar_count"] == 2000
    assert result["sec"] > 0


def test_kline_io_sample(tmp_path):
    kline_dir = tmp_path / "kline"
    kline_dir.mkdir()
    dates = pd.bdate_range("2026-01-01", periods=20)
    for idx in range(10):
        df = pd.DataFrame(
            {
                "date": dates,
                "close": np.linspace(10.0, 11.0, len(dates)) + idx,
            }
        )
        df.to_parquet(kline_dir / f"{idx:06d}.parquet", index=False)

    result = sb.measure_realtime_snapshot(
        kline_dir=kline_dir,
        sample_n=10,
    )

    assert result["files"] == 10
    assert result["sec"] > 0


def test_baseline_record_and_check_roundtrip(tmp_path):
    _write_replay_fixture(tmp_path)
    baseline_path = tmp_path / "stress_baseline.json"

    baseline = sb.record_baseline(
        path=baseline_path,
        data_root=tmp_path,
        limit=250,
        sample_n=500,
    )
    loaded = sb.load_baseline(baseline_path)

    assert baseline["version"] == 1
    assert baseline["baseline_date"]
    assert loaded["version"] == 1
    assert set(loaded["items"]) == {
        "temp_replay_60f",
        "kline_io",
        "backtest_2000bar",
    }
    assert loaded["items"]["temp_replay_60f"]["sec"] >= 0
    assert loaded["items"]["temp_replay_60f"]["mem_mb"] >= 0
    assert loaded["items"]["kline_io"]["sec"] >= 0
    assert loaded["items"]["backtest_2000bar"]["sec"] > 0
    assert loaded["items"]["backtest_2000bar"]["bar_count"] == 2000


def test_load_baseline_maps_legacy_keys(tmp_path):
    baseline_path = tmp_path / "stress_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "version": 1,
                "baseline_date": "2026-08-13",
                "items": {
                    "temp_replay_250d": {"sec": 1.0, "mem_mb": 1.0},
                    "kline_io_20": {"sec": 0.1, "files": 20},
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = sb.load_baseline(baseline_path)

    assert loaded["items"]["kline_io"]["files"] == 20
    assert "kline_io_20" not in loaded["items"]
    assert loaded["items"]["temp_replay_60f"]["sec"] == 1.0
    assert "temp_replay_250d" not in loaded["items"]


def test_check_degrades_when_threshold_exceeded():
    baseline = {
        "version": 1,
        "items": {
            "backtest_2000bar": {"sec": 1.0, "bar_count": 2000},
            "kline_io": {"sec": 1.0, "files": 20},
            "temp_replay_60f": {"sec": 1.0, "mem_mb": 1.0, "files": 60},
        },
    }
    current = {
        "version": 1,
        "items": {
            "backtest_2000bar": {"sec": 3.0, "bar_count": 999999},
            "kline_io": {"sec": 1.0, "files": 999999},
            "temp_replay_60f": {"sec": 1.0, "mem_mb": 1.0, "files": 999999},
        },
    }

    warnings = sb.check_baseline(current, baseline, threshold=2.5)

    assert warnings
    assert len(warnings) == 1
    assert warnings[0]["item"] == "backtest_2000bar"
    assert warnings[0]["metric"] == "sec"
    assert warnings[0]["ratio"] == 3.0


def test_temp_replay_limit_small(tmp_path):
    _write_replay_fixture(tmp_path)
    out_path = tmp_path / "temperature_replay_small.parquet"

    result = sb.measure_temperature_replay(
        data_root=tmp_path,
        limit=30,
        out_path=out_path,
    )

    assert result["sec"] > 0
    assert result["mem_mb"] > 0
    assert out_path.exists()
