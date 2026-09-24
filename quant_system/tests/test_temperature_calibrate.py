"""temperature_replay / calibrate_position_map / _get_strategy 校准链路测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "quant_system") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "quant_system"))

import quant_system.calibrate_position_map as cpm
import quant_system.market_temperature as mt
import quant_system.temperature_replay as tr


@pytest.fixture(autouse=True)
def _clear_position_phase_cache():
    """避免全量测试中其他用例污染 _load_position_phases 的 lru_cache。"""
    mt._load_position_phases.cache_clear()
    yield
    mt._load_position_phases.cache_clear()


def _write_replay_fixture(root: Path) -> None:
    """在临时根目录构造最小 data_warehouse 数据。"""
    warehouse = root / "data_warehouse"
    kline_dir = warehouse / "kline"
    market_dir = warehouse / "market"
    kline_dir.mkdir(parents=True)
    market_dir.mkdir(parents=True)

    dates = pd.bdate_range("2023-01-01", periods=45)
    for symbol in ["000001", "000002", "000003", "000004", "000005"]:
        close = 10 + np.arange(len(dates)) * 0.2 + (int(symbol) % 5)
        df = pd.DataFrame(
            {
                "date": dates,
                "close": close,
                "amount": np.linspace(1e8, 2e8, len(dates)),
            }
        )
        df.to_parquet(kline_dir / f"{symbol}.parquet", index=False)

    index_df = pd.DataFrame(
        {
            "date": dates,
            "open": 3000 + np.arange(len(dates)) * 5,
            "high": 3000 + np.arange(len(dates)) * 5 + 10,
            "low": 3000 + np.arange(len(dates)) * 5 - 10,
            "close": 3000 + np.arange(len(dates)) * 5,
            "volume": 100000,
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


def test_get_strategy_calibrated(tmp_path, monkeypatch):
    config_path = tmp_path / "position_map.json"
    config_path.write_text(
        json.dumps({"phases": {"恐慌": 0.7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mt, "_POSITION_MAP_PATHS", (config_path,))

    strategy = mt._get_strategy("恐慌/衰竭(Panic)", "分批回场", 5)

    assert strategy["position"] == "50-70%"
    assert strategy["position_source"] == "calibrated"
    assert strategy["action"] == "🟢 分批建仓"


def test_get_strategy_fallback(tmp_path, monkeypatch):
    missing = tmp_path / "missing_position_map.json"
    monkeypatch.setattr(mt, "_POSITION_MAP_PATHS", (missing,))

    strategy = mt._get_strategy("底部积累(Bottoming)", "选择性做多", 25)

    assert strategy["position"] == "20-30%"
    assert strategy["position_source"] == "default"


def test_replay_output_shape(tmp_path):
    _write_replay_fixture(tmp_path)
    out = tmp_path / "temperature_history.parquet"

    history = tr.replay(
        data_root=tmp_path,
        out_path=out,
        limit=30,
        rebuild=True,
    )

    assert list(history.columns) == tr.REQUIRED_COLUMNS
    assert not history.empty
    assert history["temperature"].between(0, 100).all()
    assert history["ret_next"].notna().sum() > 0
    assert pd.isna(history["ret_next"].iloc[-1])
    assert set(history["cycle_segment"].astype(str)) <= {
        "恐慌/衰竭(Panic)",
        "熊市下跌(Bear Decline)",
        "底部积累(Bottoming)",
        "后期狂热/派发(Late-cycle)",
        "成熟牛市(Mature Bull)",
        "早期牛市(Early Bull)",
        "温和正常",
    }


def test_calibrate_baseline():
    n = 70
    segments = []
    for i in range(n):
        segments.append(
            {
                "恐慌": "恐慌/衰竭(Panic)",
                "底部": "底部积累(Bottoming)",
                "早期": "早期牛市(Early Bull)",
                "成熟": "成熟牛市(Mature Bull)",
                "后期": "后期狂热/派发(Late-cycle)",
                "熊市": "熊市下跌(Bear Decline)",
                "其他": "温和正常",
            }[cpm.PHASES[i % len(cpm.PHASES)]]
        )
    returns = np.sin(np.arange(n) * 0.7) * 0.01 + 0.0002
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=n),
            "cycle_segment": segments,
            "temperature": np.arange(n) % 100,
            "risk_posture": "中性",
            "ret_next": returns,
            "margin_available": True,
        }
    )

    result = cpm.calibrate(history)

    assert set(result["phases"]) == set(cpm.PHASES)
    assert len(result["phases"]) == len(cpm.PHASES)
    for metric in ("baseline_sharpe", "oos_sharpe", "train_sharpe"):
        assert isinstance(result[metric], float)
        assert np.isfinite(result[metric])
    assert isinstance(result["note"], str)
    assert all(isinstance(v, float) for v in result["phases"].values())
