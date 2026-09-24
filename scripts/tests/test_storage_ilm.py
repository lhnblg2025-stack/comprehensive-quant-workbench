"""storage_ilm.py 单元测试：只用 tmp_path 合成小 parquet，不访问真实 data_warehouse。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import storage_ilm as ilm


def _make_kline(
    directory: Path,
    code: str = "000001",
    start: str = "2020-01-01",
    end: str = "2021-12-31",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + (i % 5) * 0.1 for i in range(n)],
            "high": [10.5 + (i % 5) * 0.1 for i in range(n)],
            "low": [9.8 + (i % 5) * 0.1 for i in range(n)],
            "close": [10.2 + (i % 5) * 0.1 for i in range(n)],
            "volume": [1000000.0 + i for i in range(n)],
            "amount": [10000000.0 + i for i in range(n)],
            "outstanding_share": [1e9 + i for i in range(n)],
            "turnover": [0.01 for _ in range(n)],
            "pct_chg": [0.0 for _ in range(n)],
        }
    )
    path = directory / f"{code}.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_scan_counts_files_and_size(tmp_path):
    kline_dir = tmp_path / "kline"
    _make_kline(kline_dir, "000001", "2020-01-01", "2020-03-31")
    _make_kline(kline_dir, "000002", "2020-01-01", "2020-03-31")

    other_dir = tmp_path / "other"
    _make_kline(other_dir, "other", "2020-01-01", "2020-01-31")

    result = ilm.scan(str(tmp_path))

    assert result["file_count"] == 3
    assert result["kline"]["count"] == 2
    expected_kline_mb = sum(
        p.stat().st_size for p in kline_dir.glob("*.parquet")
    ) / (1024 * 1024)
    assert result["kline"]["total_mb"] == pytest.approx(expected_kline_mb, abs=1e-3)
    assert result["total_gb"] > 0


def test_dry_run_classifies_hot_warm_cold(tmp_path):
    today = pd.Timestamp("2026-08-13")
    kline_dir = tmp_path / "kline"

    _make_kline(kline_dir, "000001", "2026-07-01", "2026-08-13")  # hot
    _make_kline(kline_dir, "000002", "2025-08-01", "2025-08-31")  # warm
    _make_kline(kline_dir, "000003", "2020-01-01", "2020-12-31")  # cold

    report = ilm.dry_run(str(tmp_path), today=today)

    assert report["summary"]["hot_files"] == 1
    assert report["summary"]["warm_files"] == 1
    assert report["summary"]["cold_files"] == 1
    by_name = {item["file"]: item for item in report["files"]}
    assert by_name["000001.parquet"]["category"] == "hot"
    assert by_name["000002.parquet"]["category"] == "warm"
    assert by_name["000003.parquet"]["category"] == "cold"
    assert by_name["000001.parquet"]["hot_rows"] == by_name["000001.parquet"]["rows"]
    assert by_name["000003.parquet"]["cold_rows"] == by_name["000003.parquet"]["rows"]


def test_threshold_triggered_with_small_threshold(tmp_path):
    _make_kline(tmp_path / "kline", "000001", "2020-01-01", "2020-03-31")

    scan_result = ilm.scan(str(tmp_path), threshold_gb=0.000001)
    dry_result = ilm.dry_run(str(tmp_path), threshold_gb=0.000001)

    assert scan_result["over_threshold"] is True
    assert scan_result["alert_message"] and scan_result["alert_message"].startswith("⚠️")
    assert dry_result["over_threshold"] is True
    assert dry_result["alert_message"] and dry_result["alert_message"].startswith("⚠️")


def test_archive_cold_monthly_aggregation_correct(tmp_path):
    today = pd.Timestamp("2026-08-13")
    kline_dir = tmp_path / "kline"
    _make_kline(kline_dir, "000001", "2020-01-01", "2021-12-31")

    result = ilm.archive_cold(str(tmp_path), today=today)

    output_path = tmp_path / "kline_monthly" / "000001.parquet"
    assert result["files_written"] == 1
    assert output_path.exists()
    frame = pd.read_parquet(output_path)
    assert len(frame) == 24
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert (kline_dir / "000001.parquet").exists()


def test_no_alert_below_threshold(tmp_path):
    _make_kline(tmp_path / "kline", "000001", "2020-01-01", "2020-03-31")

    result = ilm.scan(str(tmp_path), threshold_gb=2.0)

    assert result["over_threshold"] is False
    assert result["alert_message"] is None


def test_missing_directory_does_not_crash(tmp_path):
    missing = tmp_path / "not_exists"

    assert ilm.scan(str(missing))["file_count"] == 0
    assert ilm.dry_run(str(missing))["files"] == []
    assert ilm.archive_cold(str(missing))["files_written"] == 0


def test_cli_scan_runs(tmp_path, capsys):
    _make_kline(tmp_path / "kline", "000001", "2020-01-01", "2020-03-31")

    code = ilm.main(["--scan", "--data-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "数据仓库" in captured.out
    assert "kline:" in captured.out


def test_cli_apply_requires_dry_run(tmp_path):
    code = ilm.main(["--apply", "--data-root", str(tmp_path)])

    assert code == 2
