"""P3a 政策引擎测试：集中化魔法阈值 + macro_veto 接入。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system import policy
from quant_system.analysis_core import macro_veto as mv


@pytest.fixture(autouse=True)
def _clear_policy_cache():
    policy.clear_policy_cache()
    yield
    policy.clear_policy_cache()


def _patch_macro_veto(monkeypatch, *, tmp_path, pe=None, buffett=None, regime="震荡市",
                      zt_row=None, hs_ctx=None):
    (tmp_path / "generated").mkdir(exist_ok=True)
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    monkeypatch.setattr(mv, "get_regime", lambda date: {"regime": regime})
    monkeypatch.setattr(mv, "_pe_percentile", lambda date: pe)
    monkeypatch.setattr(mv, "_buffett_ratio", lambda date: buffett)
    monkeypatch.setattr(mv, "_load_zt_row", lambda date: zt_row)
    monkeypatch.setattr(mv, "_load_hs300_ctx", lambda date: hs_ctx)


def test_load_policy_reads_yaml_into_dict():
    data = policy.load_policy(policy.POLICY_PATH)

    assert isinstance(data, dict)
    assert data["veto"]["hard_coef"] == 0.3
    assert data["temperature"]["default_temp"] == 50
    assert data["multi_agent"]["devils_max"] == 0.65


def test_get_policy_reads_dot_path():
    assert policy.get_policy("veto.hard_coef") == 0.3
    assert policy.get_policy("veto.soft_coef") == 0.6
    assert policy.get_policy("multi_agent.devils_min") == 0.45


def test_get_policy_missing_key_returns_default():
    assert policy.get_policy("veto.not_exists", 123) == 123
    assert policy.get_policy("not_a_section.key", None) is None


def test_load_policy_missing_file_returns_empty_and_get_uses_default(monkeypatch, tmp_path):
    missing = tmp_path / "missing_policy.yaml"

    assert policy.load_policy(missing) == {}

    monkeypatch.setattr(policy, "POLICY_PATH", missing)
    policy.clear_policy_cache()
    assert policy.get_policy("veto.hard_coef", 0.3) == 0.3


def test_macro_veto_uses_configured_hard_coef(monkeypatch, tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        "veto:\n"
        "  hard_coef: 0.5\n"
        "  soft_coef: 0.6\n"
        "  pe_hard: 0.95\n"
        "  pe_soft: 0.80\n"
        "  buffett_hard: 1.10\n"
        "  buffett_soft: 0.95\n"
        "  r1_zb_rate: 0.40\n"
        "  r1_zt_cnt: 30\n"
        "  r2_zb_rate: 0.30\n"
        "  r3_zt_cnt: 20\n"
        "  r4_zt_cnt: 30\n"
        "  r5_weight: 1\n"
        "  r6_big_loss: 50\n"
        "  r6_zb_rate: 0.30\n"
        "  r8_panic: -0.04\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(policy, "POLICY_PATH", policy_file)
    policy.clear_policy_cache()
    _patch_macro_veto(monkeypatch, tmp_path=tmp_path, pe=0.96, regime="高波市")

    result = mv.veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["position_coef"] == 0.5


def test_macro_veto_falls_back_when_policy_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "POLICY_PATH", tmp_path / "no_such_policy.yaml")
    policy.clear_policy_cache()
    _patch_macro_veto(monkeypatch, tmp_path=tmp_path, pe=0.96, regime="高波市")

    result = mv.veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["position_coef"] == 0.3


def test_get_policy_caches_until_cache_clear(monkeypatch, tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("veto:\n  hard_coef: 0.25\n", encoding="utf-8")
    monkeypatch.setattr(policy, "POLICY_PATH", policy_file)
    policy.clear_policy_cache()

    assert policy.get_policy("veto.hard_coef") == 0.25
    policy_file.write_text("veto:\n  hard_coef: 0.75\n", encoding="utf-8")
    assert policy.get_policy("veto.hard_coef") == 0.25

    policy.clear_policy_cache()
    assert policy.get_policy("veto.hard_coef") == 0.75


def test_get_policy_resolves_env_specific_value(monkeypatch, tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        "veto:\n"
        "  hard_coef:\n"
        "    all: 0.30\n"
        "    prod: 0.35\n"
        "    backtest: 0.20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(policy, "POLICY_PATH", policy_file)
    policy.clear_policy_cache()

    monkeypatch.setenv("ENV", "prod")
    assert policy.get_policy("veto.hard_coef") == 0.35

    monkeypatch.setenv("ENV", "backtest")
    assert policy.get_policy("veto.hard_coef") == 0.20
