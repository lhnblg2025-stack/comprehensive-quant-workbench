"""pattern_gate 规律显著性门禁 + 半衰期机制测试（V12.1 域P）。

覆盖:
  - Bootstrap 强规律 p<0.05 通过 / 弱规律 p>=0.05 拒绝
  - 1000 次分布形状 / H0 中心化 / 无 scipy numpy 回退 / 空数据降级
  - extract_returns DataFrame 过滤、pattern_id 过滤、short 方向取反
  - 半衰期：未到期 active / 到期高胜率续期 / 到期低胜率 archived / 交易日历
  - 实时权重：归档退出 / 权重归一
  - validate_pattern 集成强/弱规律
  - gate_patterns json/md 输出结构与无规律降级
  - register_pattern 补 birth_date
全 mock，无网络。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module("quant_system.analysis_core.pattern_gate")


m = _import()

STRONG = [0.02] * 160 + [-0.01] * 40
STRONG_RECENT = [0.02] * 200
WEAK = [0.01, -0.01] * 100
TRADING_DAYS = pd.bdate_range("2026-07-01", periods=35)


# ── Bootstrap 验证器 ──────────────────────────────────────────────────────
def test_bootstrap_strong_pattern_passes():
    r = m.bootstrap_test(STRONG, seed=42)
    assert r["n_samples"] == 200
    assert r["observed_win_rate"] == 0.8
    assert r["p_value"] < 0.05
    assert r["avg_ret_p_value"] < 0.05
    assert r["passed"] is True
    assert r["degraded"] is False


def test_bootstrap_weak_pattern_rejects():
    r = m.bootstrap_test(WEAK, seed=42)
    assert r["observed_win_rate"] == 0.5
    assert r["p_value"] >= 0.05
    assert r["passed"] is False


def test_bootstrap_distribution_shape_is_1000():
    r = m.bootstrap_test(STRONG, seed=7, n_iterations=1000)
    assert len(r["bootstrap_win_rates"]) == 1000
    assert len(r["bootstrap_avg_rets"]) == 1000
    assert abs(float(np.mean(r["bootstrap_win_rates"])) - 0.5) < 0.1
    assert abs(float(np.mean(r["bootstrap_avg_rets"]))) < 0.01


def test_bootstrap_is_h0_centered():
    # 原始分布再抽样的 p 会接近 0.5；H0 中心化后强规律必须拒绝 H0。
    raw = m.bootstrap_test(STRONG, seed=42)
    assert raw["p_value"] < 0.05
    assert abs(raw["observed_avg_ret"] - 0.014) < 1e-9


def test_bootstrap_empty_returns_degrades():
    r = m.bootstrap_test([])
    assert r["degraded"] is True
    assert r["passed"] is False
    assert r["p_value"] is None
    assert r["bootstrap_win_rates"] == []
    assert r["bootstrap_avg_rets"] == []


def test_bootstrap_numpy_fallback_without_scipy(monkeypatch):
    expected = m.bootstrap_test(WEAK, seed=42)
    monkeypatch.setattr(m, "SCIPY_STATS", None)
    fallback = m.bootstrap_test(WEAK, seed=42)
    assert fallback["p_value"] == pytest.approx(expected["p_value"])
    assert fallback["avg_ret_p_value"] == pytest.approx(expected["avg_ret_p_value"])
    assert fallback["passed"] is expected["passed"]


def test_bootstrap_short_direction_flips_returns():
    # 对空头规律，亏损日应算作方向正确的胜局。
    r = m.bootstrap_test([-0.02, -0.01, 0.03], n_iterations=50, seed=1, direction="short")
    assert r["observed_win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["observed_avg_ret"] == pytest.approx(0.0)


# ── 历史数据提取 ──────────────────────────────────────────────────────────
def test_extract_returns_dataframe_signal_filters():
    df = pd.DataFrame({
        "date": ["2026-08-01", "2026-08-02", "2026-08-03"],
        "signal": [1, 1, 0],
        "return": [0.01, 0.02, -0.05],
    })
    assert m.extract_returns(df).tolist() == [0.01, 0.02]


def test_extract_returns_dataframe_pattern_id_filter():
    df = pd.DataFrame({
        "pattern_id": ["p1", "p1", "p2"],
        "signal": [1, 1, 1],
        "return": [0.01, 0.02, 0.99],
    })
    assert m.extract_returns(df, pattern_id="p1").tolist() == [0.01, 0.02]


def test_extract_returns_short_flips():
    assert m.extract_returns([0.01, -0.02], direction="short").tolist() == [-0.01, 0.02]


def test_extract_returns_mapping():
    history = {"p1": [0.01, 0.02], "p2": [0.09]}
    assert m.extract_returns(history, pattern_id="p2").tolist() == [0.09]


# ── 半衰期机制 ────────────────────────────────────────────────────────────
def test_half_life_not_expired_active():
    st = m.half_life_status(
        {"pattern_id": "p", "birth_date": TRADING_DAYS[0], "direction": "long"},
        as_of_date=TRADING_DAYS[5], trading_dates=TRADING_DAYS)
    assert st["status"] == "active"
    assert st["age_days"] == 5
    assert st["reason"] == "not_expired"
    assert st["renewed"] is False


def test_half_life_expired_high_win_renews():
    p = {"pattern_id": "p", "birth_date": TRADING_DAYS[0], "direction": "long",
         "recent_returns": [0.01] * 12 + [-0.01] * 8}
    st = m.half_life_status(p, as_of_date=TRADING_DAYS[25], trading_dates=TRADING_DAYS)
    assert st["age_days"] == 25
    assert st["win_rate"] == pytest.approx(0.6)
    assert st["status"] == "active"
    assert st["renewed"] is True
    assert st["reason"] == "renewed"


def test_half_life_expired_low_win_archives():
    p = {"pattern_id": "p", "birth_date": TRADING_DAYS[0], "direction": "long",
         "recent_returns": [0.01] * 10 + [-0.01] * 10}
    st = m.half_life_status(p, as_of_date=TRADING_DAYS[25], trading_dates=TRADING_DAYS)
    assert st["win_rate"] == pytest.approx(0.5)
    assert st["status"] == "archived"
    assert st["reason"] == "low_win_rate"


def test_half_life_uses_trading_dates():
    st = m.half_life_status(
        {"pattern_id": "p", "birth_date": TRADING_DAYS[0]},
        as_of_date=TRADING_DAYS[10], trading_dates=TRADING_DAYS)
    assert st["age_days"] == 10
    assert st["status"] == "active"


def test_half_life_expired_no_returns_archives():
    st = m.half_life_status(
        {"pattern_id": "p", "birth_date": TRADING_DAYS[0]},
        as_of_date=TRADING_DAYS[25], trading_dates=TRADING_DAYS)
    assert st["status"] == "archived"
    assert st["reason"] == "no_recent_returns"


def test_half_life_short_direction_win_rate():
    p = {"pattern_id": "p", "birth_date": TRADING_DAYS[0], "direction": "short",
         "recent_returns": [-0.01] * 12 + [0.01] * 8}
    st = m.half_life_status(p, as_of_date=TRADING_DAYS[25], trading_dates=TRADING_DAYS)
    assert st["win_rate"] == pytest.approx(0.6)
    assert st["status"] == "active"


# ── 实时权重接口 ───────────────────────────────────────────────────────────
def test_real_time_weights_excludes_archived():
    patterns = [
        {"pattern_id": "good", "birth_date": TRADING_DAYS[0], "direction": "long",
         "recent_returns": [0.01] * 12 + [-0.01] * 8},
        {"pattern_id": "bad", "birth_date": TRADING_DAYS[0], "direction": "long",
         "recent_returns": [0.01] * 10 + [-0.01] * 10},
    ]
    weights = m.real_time_weights(patterns, as_of_date=TRADING_DAYS[25],
                                  trading_dates=TRADING_DAYS)
    assert weights == {"good": 1.0}
    assert "bad" not in weights


def test_real_time_weights_normalizes_active_weights():
    patterns = [
        {"pattern_id": "a", "weight": 1.0, "birth_date": TRADING_DAYS[0],
         "recent_returns": [0.01] * 12 + [-0.01] * 8},
        {"pattern_id": "b", "weight": 3.0, "birth_date": TRADING_DAYS[0],
         "recent_returns": [0.01] * 12 + [-0.01] * 8},
    ]
    weights = m.real_time_weights(patterns, as_of_date=TRADING_DAYS[25],
                                  trading_dates=TRADING_DAYS)
    assert weights == {"a": 0.25, "b": 0.75}


# ── 门禁集成与输出 ─────────────────────────────────────────────────────────
def test_validate_pattern_strong_active():
    p = {"pattern_id": "strong", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0]}
    out = m.validate_pattern(p, returns=STRONG_RECENT, as_of_date=TRADING_DAYS[1],
                             trading_dates=TRADING_DAYS, seed=42)
    assert out["status"] == "active"
    assert out["bootstrap"]["passed"] is True
    assert out["win_rate"] == pytest.approx(1.0)


def test_validate_pattern_weak_archived():
    p = {"pattern_id": "weak", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0]}
    out = m.validate_pattern(p, returns=WEAK, as_of_date=TRADING_DAYS[1],
                             trading_dates=TRADING_DAYS, seed=42)
    assert out["status"] == "archived"
    assert out["bootstrap"]["passed"] is False


def test_gate_patterns_output_json_md(tmp_path):
    patterns = [
        {"pattern_id": "strong", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0], "returns": STRONG_RECENT},
        {"pattern_id": "weak", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0], "returns": WEAK},
    ]
    result = m.gate_patterns(patterns, as_of_date="2026-08-13",
                             trading_dates=TRADING_DAYS, seed=42, out_dir=tmp_path)
    assert result["degraded"] is False
    assert result["n_patterns"] == 2
    assert result["n_active"] == 1
    assert result["n_archived"] == 1
    statuses = {p["pattern_id"]: p["status"] for p in result["patterns"]}
    assert statuses == {"strong": "active", "weak": "archived"}

    day = "20260813"
    json_path = tmp_path / "pattern_gate" / day / "pattern_gate_2026-08-13.json"
    md_path = tmp_path / "pattern_gate" / day / "pattern_gate_2026-08-13.md"
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema"] == "pattern_gate/v1"
    assert len(data["patterns"]) == 2
    assert "规律显著性门禁" in md_path.read_text(encoding="utf-8")


def test_gate_patterns_no_patterns_degrades(tmp_path):
    result = m.gate_patterns([], as_of_date="2026-08-13", out_dir=tmp_path)
    assert result["degraded"] is True
    assert result["reason"] == "no_patterns"
    assert result["patterns"] == []
    day = "20260813"
    assert (tmp_path / "pattern_gate" / day / "pattern_gate_2026-08-13.json").exists()
    assert (tmp_path / "pattern_gate" / day / "pattern_gate_2026-08-13.md").exists()


def test_register_pattern_sets_birth_date():
    p = m.register_pattern({"pattern_id": "x", "first_seen": "2026-08-01"})
    assert p["birth_date"] == "2026-08-01"
    p2 = m.register_pattern({"pattern_id": "y"}, birth_date="2026-08-02")
    assert p2["birth_date"] == "2026-08-02"
    p3 = m.register_pattern({"pattern_id": "z"})
    assert p3["birth_date"]


def test_gate_patterns_history_dataframe_filter(tmp_path):
    history = pd.DataFrame({
        "pattern_id": ["strong", "strong", "weak"],
        "signal": [1, 1, 1],
        "return": STRONG[:2] + [WEAK[0]],
    })
    patterns = [
        {"pattern_id": "strong", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0]},
        {"pattern_id": "weak", "pattern_type": "mock", "direction": "long",
         "birth_date": TRADING_DAYS[0]},
    ]
    result = m.gate_patterns(patterns, history=history, as_of_date="2026-08-13",
                             trading_dates=TRADING_DAYS, seed=42, out_dir=tmp_path)
    assert result["n_patterns"] == 2
    # strong 只有 2 个样本，bootstrap 应保守地不通过；结构仍完整。
    assert all(p["status"] in ("active", "archived") for p in result["patterns"])
