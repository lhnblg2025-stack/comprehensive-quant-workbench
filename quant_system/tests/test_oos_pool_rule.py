"""W2.4 — OOS 入池硬规则单测（翻转自动降权/退池）。

覆盖：
  - 同号 keep
  - 翻转 downweight（×0.5）
  - 严重翻转 / 连续多期翻转 → retire
  - |OOS IC| 低于阈值 → retire
  - 缺失 OOS（None / NaN / 报告缺失 / 旧格式）→ 保守处理
  - 非法参数 → raise ValueError（不吞异常）
  - apply_oos_rule_to_weights 批量施加
  - DynamicFactorSelector 接入点验证
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_system.ic_factors.oos_pool_rule import (
    oos_flip_rule,
    load_oos_report,
    apply_oos_rule_to_weights,
    flip_streak,
)


# ── 核心规则 ─────────────────────────────────────────────────────────────
class TestOosFlipRule:
    def test_keep_same_sign_positive(self):
        r = oos_flip_rule(0.01, 0.02)
        assert r["action"] == "keep"
        assert r["weight_multiplier"] == 1.0
        assert r["flip"] is False

    def test_keep_same_sign_negative(self):
        r = oos_flip_rule(-0.01, -0.02)
        assert r["action"] == "keep"
        assert r["weight_multiplier"] == 1.0

    def test_flip_downweight(self):
        r = oos_flip_rule(0.01, -0.01)
        assert r["action"] == "downweight"
        assert r["weight_multiplier"] == 0.5
        assert r["flip"] is True

    def test_severe_flip_retire(self):
        """翻转且 |OOS IC| 达严重阈值 → 直接退池。"""
        r = oos_flip_rule(0.01, -0.05)
        assert r["action"] == "retire"
        assert r["weight_multiplier"] == 0.0

    def test_consecutive_flip_retire(self):
        """连续多期翻转（含当期 >= 2）→ 退池。"""
        r = oos_flip_rule(0.01, -0.01, flip_history=1)
        assert r["action"] == "retire"
        assert r["weight_multiplier"] == 0.0
        assert "consecutive_flip_streak" in r["reason"]

    def test_single_flip_history_zero_stays_downweight(self):
        r = oos_flip_rule(0.01, -0.01, flip_history=0)
        assert r["action"] == "downweight"

    def test_oos_ic_below_threshold_retire(self):
        """|OOS IC| < min_abs_oos_ic → 方向噪声，退池。"""
        r = oos_flip_rule(0.01, 0.001)
        assert r["action"] == "retire"
        assert r["weight_multiplier"] == 0.0

    def test_missing_oos_downweight_conservative(self):
        r = oos_flip_rule(0.01, None)
        assert r["action"] == "downweight"
        assert r["weight_multiplier"] == 0.5
        assert r["flip"] is False
        assert r["reason"] == "oos_missing"

    def test_missing_oos_nan_treated_missing(self):
        r = oos_flip_rule(0.01, float("nan"))
        assert r["action"] == "downweight"

    def test_missing_oos_retire_when_configured(self):
        r = oos_flip_rule(0.01, None, missing_oos_action="retire")
        assert r["action"] == "retire"

    def test_in_sample_ic_zero_downweight(self):
        r = oos_flip_rule(0.0, 0.01)
        assert r["action"] == "downweight"

    def test_in_sample_ic_missing_downweight(self):
        r = oos_flip_rule(None, 0.01)
        assert r["action"] == "downweight"

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            oos_flip_rule(0.01, 0.01, downweight_multiplier=1.5)
        with pytest.raises(ValueError):
            oos_flip_rule(0.01, 0.01, retire_flip_streak=0)
        with pytest.raises(ValueError):
            oos_flip_rule(0.01, 0.01, min_abs_oos_ic=-1)
        with pytest.raises(ValueError):
            oos_flip_rule(0.01, 0.01, missing_oos_action="bogus")


# ── 连续翻转计数 ──────────────────────────────────────────────────────────
class TestFlipStreak:
    def test_tail_consecutive(self):
        assert flip_streak([True, True, False, True]) == 1
        assert flip_streak([True, True, True]) == 3
        assert flip_streak([False, True, True]) == 2

    def test_empty(self):
        assert flip_streak([]) == 0


# ── 批量施加 ──────────────────────────────────────────────────────────────
class TestApplyOosRuleToWeights:
    def test_multipliers(self):
        w = pd.Series([1.0, 1.0, 1.0], index=["keep_f", "flip_f", "retire_f"])
        oos_map = {
            "keep_f": {"in_sample_ic": 0.01, "oos_ic": 0.02, "flip_history": 0},
            "flip_f": {"in_sample_ic": 0.01, "oos_ic": -0.01, "flip_history": 0},
            "retire_f": {"in_sample_ic": 0.01, "oos_ic": -0.05, "flip_history": 0},
        }
        out = apply_oos_rule_to_weights(w, oos_map, log_actions=False)
        assert out["keep_f"] == 1.0
        assert out["flip_f"] == 0.5
        assert out["retire_f"] == 0.0

    def test_factor_not_in_map_untouched(self):
        w = pd.Series([1.0], index=["unknown"])
        out = apply_oos_rule_to_weights(w, {}, log_actions=False)
        assert out["unknown"] == 1.0


# ── 报告加载 ──────────────────────────────────────────────────────────────
class TestLoadOosReport:
    def test_new_format_factors(self, tmp_path):
        p = tmp_path / "IC_OOS_REPORT.json"
        p.write_text(json.dumps({
            "factors": [
                {"factor": "a", "ic_mean_train": 0.01, "ic_mean_test": 0.02, "sign_keep": True},
                {"factor": "b", "ic_mean_train": 0.01, "ic_mean_test": -0.01, "sign_keep": False},
            ]
        }), encoding="utf-8")
        m = load_oos_report(p)
        assert m["a"]["in_sample_ic"] == 0.01
        assert m["a"]["oos_ic"] == 0.02
        assert m["b"]["flip"] is True

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_oos_report(tmp_path / "nope.json") == {}

    def test_legacy_flipped_factors(self, tmp_path):
        p = tmp_path / "IC_OOS_REPORT.json"
        p.write_text(json.dumps({"flipped_factors": ["x", "y"]}), encoding="utf-8")
        m = load_oos_report(p)
        assert set(m) == {"x", "y"}
        assert m["x"]["flip"] is True
        assert m["x"]["oos_ic"] is None  # 旧格式无 OOS IC → 规则保守降权


# ── 接入点：DynamicFactorSelector 权重路径 ────────────────────────────────
class TestSelectorIntegration:
    def test_selector_applies_oos_rule(self):
        from quant_system.ic_factors.composite import DynamicFactorSelector
        sel = DynamicFactorSelector(regime="oscill")
        icirs = pd.Series([0.02, 0.02, 0.02], index=["keep_f", "flip_f", "retire_f"])
        oos_rules = {
            "keep_f": {"in_sample_ic": 0.01, "oos_ic": 0.02, "flip_history": 0},
            "flip_f": {"in_sample_ic": 0.01, "oos_ic": -0.01, "flip_history": 0},
            "retire_f": {"in_sample_ic": 0.01, "oos_ic": -0.05, "flip_history": 0},
        }
        w = sel.weights(icirs, oos_rules=oos_rules)
        assert w.sum() == pytest.approx(1.0)
        assert "retire_f" not in w.index  # 退池
        assert w["flip_f"] < w["keep_f"]  # 翻转降权
        assert w["keep_f"] > 0 and w["flip_f"] > 0
