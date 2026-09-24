"""退市股接入回测测试（数据层 → 回测层打通）。

全部使用 tmp_path 构造合成 delisted.parquet 与小型 kline parquet，
不读真实数据、不触网。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest_pro import (
    SurvivalBiasCorrector,
    integrate_delisted,
    load_delisted_prices,
)


def _write_kline(kline_dir: Path, code: str, rows: list[tuple[str, float]]) -> None:
    kline_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "close"]).to_parquet(
        kline_dir / f"{code}.parquet", index=False
    )


def _write_delisted_table(path: Path, rows) -> None:
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    rows.to_parquet(path, index=False)


def test_load_delisted_prices_normal_parquet_structure(tmp_path: Path) -> None:
    kline_dir = tmp_path / "kline"
    _write_kline(kline_dir, "000001", [
        ("2026-01-02", 10.0),
        ("2026-01-03", 11.0),
        ("2026-01-06", 12.0),
    ])
    delisted_path = tmp_path / "delisted.parquet"
    _write_delisted_table(delisted_path, [{
        "code": "1",
        "name": "测试退市",
        "delist_date": "2026-01-06",
        "reason": "测试",
    }])

    result = load_delisted_prices(
        delisted_path=delisted_path,
        kline_dir=kline_dir,
        max_days_after_delist=0,
    )

    assert result == {
        "000001": {
            "2026-01-02": 10.0,
            "2026-01-03": 11.0,
            "2026-01-06": 12.0,
        },
    }


def test_load_delisted_prices_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_delisted_prices(
        delisted_path=tmp_path / "missing.parquet",
        kline_dir=tmp_path / "missing_kline",
    )

    assert result == {}


def test_load_delisted_prices_empty_parquet_returns_empty(tmp_path: Path) -> None:
    delisted_path = tmp_path / "delisted.parquet"
    _write_delisted_table(
        delisted_path,
        pd.DataFrame(columns=["code", "name", "delist_date", "reason"]),
    )

    result = load_delisted_prices(
        delisted_path=delisted_path,
        kline_dir=tmp_path / "kline",
    )

    assert result == {}


def test_load_delisted_prices_post_delist_is_liquidation(tmp_path: Path) -> None:
    """P2-3: 退市日后价格为清算价（默认0=close→0），回测实现损失而非0%。"""
    kline_dir = tmp_path / "kline"
    _write_kline(kline_dir, "000001", [
        ("2026-01-05", 20.0),
        ("2026-01-06", 21.0),
        ("2026-01-07", 22.0),
        ("2026-01-08", 23.0),
    ])
    delisted_path = tmp_path / "delisted.parquet"
    _write_delisted_table(delisted_path, [{
        "code": "000001",
        "name": "测试退市",
        "delist_date": "2026-01-06",
        "reason": "测试",
    }])

    result = load_delisted_prices(
        delisted_path=delisted_path,
        kline_dir=kline_dir,
        max_days_after_delist=2,
    )

    assert result["000001"]["2026-01-05"] == 20.0
    assert result["000001"]["2026-01-06"] == 21.0
    # 退市日(01-06)后无交易 → 默认清算价 0.0（close→0），引擎将记 -100% 损失
    assert result["000001"]["2026-01-07"] == 0.0
    assert result["000001"]["2026-01-08"] == 0.0


def test_integrate_delisted_injects_column_and_report(tmp_path: Path) -> None:
    kline_dir = tmp_path / "kline"
    _write_kline(kline_dir, "000003", [
        ("2026-01-05", 20.0),
        ("2026-01-06", 21.0),
        ("2026-01-07", 22.0),
    ])
    delisted_path = tmp_path / "delisted.parquet"
    _write_delisted_table(delisted_path, [{
        "code": "000003",
        "name": "测试退市",
        "delist_date": "2026-01-07",
        "reason": "测试",
    }])
    index = pd.date_range("2026-01-05", periods=3, freq="B")
    backtest_df = pd.DataFrame(
        {
            "600000": [100.0, 101.0, 102.0],
            "600001": [50.0, 51.0, 52.0],
        },
        index=index,
    )

    result_df, report = integrate_delisted(
        backtest_df,
        delisted_path=delisted_path,
        kline_dir=kline_dir,
    )

    assert result_df.shape[1] == 3
    assert "000003" in result_df.columns
    assert result_df["000003"].tolist() == [20.0, 21.0, 22.0]
    assert report["delisted_in_universe"] == ["000003"]
    assert report["injected_count"] == 1
    assert report["death_rate"] == pytest.approx(1 / 3)
    assert report["risk_level"] == "high"


@pytest.mark.parametrize(
    ("alive_count", "dead_count", "expected_level"),
    [
        (6, 4, "high"),      # 40% > 30%
        (8, 2, "medium"),    # 20% > 10%
        (9, 1, "low"),       # 10% 不触发 medium
    ],
)
def test_flag_survivorship_risk_thresholds(
    alive_count: int, dead_count: int, expected_level: str
) -> None:
    alive = [f"A{i}" for i in range(alive_count)]
    dead = [f"D{i}" for i in range(dead_count)]
    symbols = alive + dead

    result = SurvivalBiasCorrector.flag_survivorship_risk(symbols, alive)

    assert result["dead_count"] == dead_count
    assert result["death_rate"] == pytest.approx(dead_count / len(symbols))
    assert result["risk_level"] == expected_level


def test_add_delisted_stocks_fillna_does_not_overwrite_existing_values() -> None:
    backtest_df = pd.DataFrame(
        {"000001": [9.0, np.nan]},
        index=["2026-01-05", "2026-01-06"],
    )
    delisted_prices = {
        "000001": {
            "2026-01-05": 7.0,
            "2026-01-06": 8.0,
        },
    }

    result = SurvivalBiasCorrector.add_delisted_stocks(
        backtest_df, delisted_prices
    )

    assert result.loc["2026-01-05", "000001"] == 9.0
    assert result.loc["2026-01-06", "000001"] == 8.0
