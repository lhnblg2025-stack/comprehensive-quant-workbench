"""P0 IPO lockout tests for factor_model.get_universe().

These tests exercise the ``min_trading_days`` IPO cool-down filter with synthetic
parquet klines and a fake akshare module.  No network access and no real warehouse
data are used.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_system.factor_model as fm
from quant_system.factor_model import FactorModel


def write_kline(kline_dir: Path, code: str, rows: int) -> None:
    """Write a minimal parquet file with exactly ``rows`` rows."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-02", periods=rows, freq="B")})
    df.to_parquet(kline_dir / f"{code}.parquet", index=False)


def patch_akshare(monkeypatch: pytest.MonkeyPatch, codes: list[str]) -> None:
    """Return the supplied codes from the akshare spot endpoint."""
    df = pd.DataFrame(
        {
            "代码": codes,
            "名称": [f"stock_{code}" for code in codes],
        }
    )
    fake_akshare = types.SimpleNamespace(stock_zh_a_spot_em=lambda: df)
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)


@pytest.fixture
def kline_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "kline"
    path.mkdir()
    monkeypatch.setattr(fm, "KLINE_DIR", path)
    fm._get_kline_row_counts.cache_clear()
    return path


def test_get_universe_filters_recent_ipo_codes(kline_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row_counts = {
        "000010": 10,
        "000030": 30,
        "000059": 59,
        "000060": 60,
        "000200": 200,
    }
    for code, rows in row_counts.items():
        write_kline(kline_dir, code, rows)
    patch_akshare(monkeypatch, list(row_counts))

    universe = FactorModel(universe_size=100).get_universe(100, min_trading_days=60)

    assert universe == ["000060", "000200"]


def test_boundary_exactly_60_kept_59_dropped(kline_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_kline(kline_dir, "000059", 59)
    write_kline(kline_dir, "000060", 60)
    patch_akshare(monkeypatch, ["000059", "000060"])

    universe = FactorModel(universe_size=100).get_universe(100, min_trading_days=60)

    assert universe == ["000060"]


@pytest.mark.parametrize("min_trading_days", [0, None])
def test_min_trading_days_zero_or_none_disables_filter(
    kline_dir: Path, monkeypatch: pytest.MonkeyPatch, min_trading_days: int | None
) -> None:
    codes = ["000010", "000030", "000059", "000060", "000200"]
    for index, code in enumerate(codes):
        write_kline(kline_dir, code, [10, 30, 59, 60, 200][index])
    patch_akshare(monkeypatch, codes)

    universe = FactorModel(universe_size=100).get_universe(100, min_trading_days=min_trading_days)

    assert universe == codes


def test_missing_kline_file_is_kept_conservatively(kline_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codes = ["000010", "000020", "000030"]
    write_kline(kline_dir, "000010", 100)
    write_kline(kline_dir, "000020", 10)
    # 000030 intentionally has no kline file.
    patch_akshare(monkeypatch, codes)

    universe = FactorModel(universe_size=100).get_universe(100, min_trading_days=60)

    assert universe == ["000010", "000030"]


def test_fallback_universe_under_30_does_not_scan_kline(
    kline_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_scan(*args, **kwargs):
        raise AssertionError("fallback path must not read kline files")

    def fail_filter(*args, **kwargs):
        raise AssertionError("fallback path must not apply IPO filter")

    monkeypatch.setattr(fm, "_scan_kline_row_counts", fail_scan)
    monkeypatch.setattr(FactorModel, "_filter_ipo_lockout", fail_filter)
    fm._get_kline_row_counts.cache_clear()

    universe = FactorModel(universe_size=30).get_universe(30, min_trading_days=60)

    assert len(universe) == 30
    assert all(isinstance(code, str) and len(code) == 6 and code.isdigit() for code in universe)


def test_row_count_scan_is_cached_between_calls(
    kline_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codes = ["000010", "000030", "000059", "000060", "000200"]
    for code, rows in zip(codes, [10, 30, 59, 60, 200]):
        write_kline(kline_dir, code, rows)
    patch_akshare(monkeypatch, codes)

    original_scan = fm._scan_kline_row_counts
    calls: list[str] = []

    def counting_scan(path):
        calls.append(str(path))
        return original_scan(path)

    monkeypatch.setattr(fm, "_scan_kline_row_counts", counting_scan)
    fm._get_kline_row_counts.cache_clear()

    model = FactorModel(universe_size=100)
    first = model.get_universe(100, min_trading_days=60)
    second = model.get_universe(100, min_trading_days=60)

    assert first == second == ["000060", "000200"]
    assert len(calls) == 1
