"""P2a 因子自动退役调度器测试。

全部使用合成 DataFrame，不读真实数据、不触网。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.factor_model import FactorModel


@pytest.fixture(autouse=True)
def _quiet_qv6_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把 ic_factors 日志指到临时文件，避免测试环境 /root/quant/logs 权限噪音。"""
    monkeypatch.setenv("QV6_LOG_FILE", str(tmp_path / "qv6.log"))


def _weekly_sign_panel(signs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造每个自然周 5 个交易日的单因子与次日收益面板。

    factor 每周为 [0,1,2,3,4]，return = sign * factor + 微小噪声；
    周内 Spearman IC 近似 +1 或 -1。
    """
    n_weeks = len(signs)
    if n_weeks == 0:
        idx = pd.DatetimeIndex([])
        return pd.DataFrame(index=idx), pd.DataFrame(index=idx)

    idx = pd.date_range("2026-01-05", periods=n_weeks * 5, freq="B")
    base = np.tile(np.arange(5, dtype=float), n_weeks)
    weekly_sign = np.repeat(np.asarray(signs, dtype=float), 5)
    noise = np.random.default_rng(42).normal(0.0, 1e-6, len(idx))
    factor = base
    forward_return = weekly_sign * base + noise
    factor_df = pd.DataFrame({"f1": factor}, index=idx)
    return_df = pd.DataFrame({"f1": forward_return}, index=idx)
    return factor_df, return_df


def test_strong_factor_not_retired() -> None:
    factor_df, return_df = _weekly_sign_panel([1.0] * 6)

    result = FactorModel.factor_retirement_scheduler(factor_df, return_df)

    assert result["retired"] == {}
    assert "f1" in result["active"]
    assert result["scan_date"]


def test_weak_factor_retired() -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0] * 5)

    result = FactorModel.factor_retirement_scheduler(factor_df, return_df)

    assert "f1" in result["retired"]
    assert result["retired"]["f1"]["status"] == "RETIRED"
    assert result["retired"]["f1"]["icir"] < 0.1
    assert result["retired"]["f1"]["win_rate"] < 0.45
    assert "f1" not in result["active"]


def test_intermittent_weak_not_retired() -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0, -1.0, 1.0, 1.0])

    result = FactorModel.factor_retirement_scheduler(factor_df, return_df)

    assert result["retired"] == {}
    assert "f1" in result["active"]


def test_insufficient_weeks_not_retired() -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0, -1.0])

    result = FactorModel.factor_retirement_scheduler(factor_df, return_df)

    assert result["retired"] == {}
    assert "f1" in result["active"]
    assert "f1" in result["insufficient"]
    assert "insufficient" in result["insufficient"]["f1"]["reason"]


def test_save_and_resave_are_idempotent(tmp_path: Path) -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0] * 5)
    scan = FactorModel.factor_retirement_scheduler(
        factor_df, return_df, as_of="2026-02-06"
    )
    out = tmp_path / "factor_retirement.json"

    FactorModel.save_retirement_state(scan, out)
    first = json.loads(out.read_text(encoding="utf-8"))
    FactorModel.save_retirement_state(scan, out)
    second = json.loads(out.read_text(encoding="utf-8"))

    assert list(first["retired"]) == ["f1"]
    assert list(second["retired"]) == ["f1"]
    assert second["retired"]["f1"]["retired_date"] == "2026-02-06"


def test_existing_retired_factor_keeps_original_retired_date(tmp_path: Path) -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0] * 5)
    out = tmp_path / "factor_retirement.json"

    first_scan = FactorModel.factor_retirement_scheduler(
        factor_df, return_df, as_of="2026-02-06"
    )
    FactorModel.save_retirement_state(first_scan, out)

    second_scan = FactorModel.factor_retirement_scheduler(
        factor_df, return_df, as_of="2026-02-13"
    )
    FactorModel.save_retirement_state(second_scan, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["retired"]["f1"]["retired_date"] == "2026-02-06"
    assert payload["scan_date"] == "2026-02-13"


def test_reactivated_factor_removed_from_retired_state(tmp_path: Path) -> None:
    factor_df, return_df = _weekly_sign_panel([-1.0] * 5)
    out = tmp_path / "factor_retirement.json"

    retired_scan = FactorModel.factor_retirement_scheduler(
        factor_df, return_df, as_of="2026-02-06"
    )
    FactorModel.save_retirement_state(retired_scan, out)

    recovered_info = {
        "retired": {},
        "active": ["f1"],
        "scan_date": "2026-02-20",
    }
    FactorModel.save_retirement_state(recovered_info, out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["retired"] == {}
    assert payload["active"] == ["f1"]


def test_empty_input_does_not_crash() -> None:
    empty = pd.DataFrame()

    result = FactorModel.factor_retirement_scheduler(empty, empty)

    assert result["retired"] == {}
    assert result["active"] == []
    assert isinstance(result["scan_date"], str)
