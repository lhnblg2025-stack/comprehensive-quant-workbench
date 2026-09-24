"""macro_proxy 高频代理模型测试。

全部使用 tmp_path 合成 parquet，通过 MACRO_DATA_DIR 重定向数据仓库，不访问
真实 data_warehouse，也不发起网络请求。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_system.macro_proxy as mp


@pytest.fixture
def proxy_env(tmp_path, monkeypatch):
    """构造指向 tmp_path 的数据仓库，并简化 PROXY_MAP 为单代理。"""
    warehouse = tmp_path / "data_warehouse"
    (warehouse / "macro").mkdir(parents=True, exist_ok=True)
    (warehouse / "market").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MACRO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        mp,
        "PROXY_MAP",
        {"ppi_yearly": ["commodity__crude"]},
    )
    return warehouse


def _write_linear_crude(warehouse: Path, start: str = "2018-01-01",
                        end: str = "2025-03-31") -> pd.Series:
    dates = pd.bdate_range(start, end)
    close = 100.0 + 0.025 * np.arange(len(dates), dtype=float)
    pd.DataFrame({"日期": dates, "收盘价": close}).to_parquet(
        warehouse / "market" / "commodity__crude.parquet", index=False
    )
    return mp.load_proxy_series("commodity__crude")


def _write_linear_ppi(warehouse: Path, proxy_daily: pd.Series,
                      months: int = 36) -> pd.DatetimeIndex:
    proxy_monthly = proxy_daily.resample("ME").last().dropna()
    target_dates = proxy_monthly.index[-months:]
    target_values = 0.6 * proxy_monthly.loc[target_dates] + 1.5
    pd.DataFrame({
        "日期": target_dates,
        "当月同比增长": target_values.to_numpy(),
    }).to_parquet(warehouse / "macro" / "ppi_yearly.parquet", index=False)
    return target_dates


def _make_linear_fixture(warehouse: Path) -> pd.DatetimeIndex:
    proxy_daily = _write_linear_crude(warehouse)
    return _write_linear_ppi(warehouse, proxy_daily)


def test_build_proxy_model_linear_r2_gt_08(proxy_env):
    _make_linear_fixture(proxy_env)
    model = mp.build_proxy_model("ppi_yearly", lookback_years=3)

    assert model is not None
    assert model["target"] == "ppi_yearly"
    assert model["proxies"] == ["commodity__crude"]
    assert model["r2"] > 0.8
    assert model["reliability_score"] == model["r2"]
    assert model["is_proxy"] is True
    assert model["n"] >= 12


def test_daily_estimate_daily_frequency_and_attrs(proxy_env):
    _make_linear_fixture(proxy_env)
    estimate = mp.daily_estimate("ppi_yearly")

    assert isinstance(estimate, pd.Series)
    assert len(estimate) > 0
    assert estimate.attrs.get("is_proxy") is True
    assert estimate.attrs.get("reliability_score", -1) > 0.8
    assert "ppi_yearly" in estimate.name


def test_insufficient_months_returns_none(proxy_env):
    proxy_daily = _write_linear_crude(proxy_env)
    _write_linear_ppi(proxy_env, proxy_daily, months=6)

    model = mp.build_proxy_model("ppi_yearly")
    assert model is None


def test_missing_proxy_file_returns_none(proxy_env):
    proxy_daily = _write_linear_crude(proxy_env)
    _write_linear_ppi(proxy_env, proxy_daily)
    (proxy_env / "market" / "commodity__crude.parquet").unlink()

    assert mp.load_proxy_series("commodity__crude").empty
    assert mp.build_proxy_model("ppi_yearly") is None


def test_coefficient_direction_positive(proxy_env):
    _make_linear_fixture(proxy_env)
    model = mp.build_proxy_model("ppi_yearly")

    assert model is not None
    assert model["coef"]["commodity__crude"] > 0


def test_as_of_truncates_future_dates(proxy_env):
    target_dates = _make_linear_fixture(proxy_env)
    cutoff = target_dates[-24]

    estimate = mp.daily_estimate("ppi_yearly", as_of=cutoff)

    assert len(estimate) > 0
    assert estimate.index.max() <= cutoff


def test_cli_print_runs(proxy_env, capsys):
    _make_linear_fixture(proxy_env)

    exit_code = mp.main(["--target", "ppi_yearly", "--print"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "r2=" in output
    assert "ppi_yearly" in output
    assert "reliability_score=" in output
