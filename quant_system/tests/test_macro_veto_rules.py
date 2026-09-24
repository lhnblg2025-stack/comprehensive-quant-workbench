"""macro_veto 交易级/情绪级规则扩展测试。

覆盖:
  - _check_trade_rules 8 条交易级规则触发与不触发边界
  - veto() 集成: macro_risk / trade_risk 分别累计后的 hard/soft/none 判定、trade_hits 输出
  - 数据缺失降级: zt/hs300 均缺失时规则全部跳过
  - get_veto 兼容旧 JSON（无 trade_hits 字段）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import macro_veto as mv


def _zt(**overrides):
    row = {
        "zt_cnt": 100,
        "zb_rate": 0.10,
        "dt_cnt": 5,
        "big_loss_cnt": 0,
        "jr1": 1.0,
        "jr1_prev": 1.0,
        "max_board": 5,
        "max_board_prev": 5,
    }
    row.update(overrides)
    return row


def _hs(**overrides):
    ctx = {"ret_1d": 0.01, "ret_5d": 0.01, "down3": False}
    ctx.update(overrides)
    return ctx


def _ids(hits):
    return [h["rule_id"] for h in hits]


def test_r1_blast_retreat_triggers():
    assert _ids(mv._check_trade_rules(_zt(zt_cnt=29, zb_rate=0.41), _hs())) == ["R1"]


def test_r2_idx3d_down_weak_triggers():
    assert _ids(mv._check_trade_rules(_zt(zb_rate=0.31), _hs(down3=True))) == ["R2"]


def test_r3_profit_drought_triggers():
    assert _ids(mv._check_trade_rules(_zt(zt_cnt=19), _hs(ret_5d=-0.031))) == ["R3"]


def test_r4_limit_down_surge_triggers():
    assert _ids(mv._check_trade_rules(_zt(zt_cnt=20, dt_cnt=21), _hs())) == ["R4"]


def test_r5_jr1_loss_2d_triggers():
    assert _ids(mv._check_trade_rules(_zt(jr1=-0.01, jr1_prev=-0.02), _hs())) == ["R5"]


def test_r6_big_loss_wave_triggers():
    assert _ids(mv._check_trade_rules(_zt(zb_rate=0.31, big_loss_cnt=51), _hs())) == ["R6"]


def test_r7_high_board_break_triggers():
    assert _ids(mv._check_trade_rules(_zt(max_board_prev=6, max_board=3), _hs())) == ["R7"]


def test_r8_panic_day_triggers():
    assert _ids(mv._check_trade_rules(_zt(), _hs(ret_1d=-0.041))) == ["R8"]


def test_r1_boundary_zb_rate_equal_does_not_trigger():
    assert _ids(mv._check_trade_rules(_zt(zt_cnt=29, zb_rate=0.40), _hs())) == []


def test_r5_only_one_day_loss_does_not_trigger():
    assert _ids(mv._check_trade_rules(_zt(jr1=-0.01, jr1_prev=0.01), _hs())) == []


def test_r7_prev_5_current_4_does_not_trigger():
    assert _ids(mv._check_trade_rules(_zt(max_board_prev=5, max_board=4), _hs())) == []


def test_missing_trade_fields_skip_all_rules():
    assert mv._check_trade_rules(None, None) == []


def _patch_veto(monkeypatch, *, zt_row, hs_ctx, regime="震荡市", pe=None, buffett=None):
    monkeypatch.setattr(mv, "get_regime", lambda date: {"regime": regime})
    monkeypatch.setattr(mv, "_pe_percentile", lambda date: pe)
    monkeypatch.setattr(mv, "_buffett_ratio", lambda date: buffett)
    monkeypatch.setattr(mv, "_load_zt_row", lambda date: zt_row)
    monkeypatch.setattr(mv, "_load_hs300_ctx", lambda date: hs_ctx)


def test_veto_weight2_hit_with_high_vol_is_hard(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(monkeypatch, zt_row=_zt(zt_cnt=29, zb_rate=0.41), hs_ctx=_hs(), regime="高波市")

    result = mv.veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["position_coef"] == 0.3
    assert [h["rule_id"] for h in result["trade_hits"]] == ["R1"]
    assert result["trade_hits"][0]["name"] == "blast_retreat"
    assert result["trade_hits"][0]["weight"] == 2


def test_veto_single_weight2_trade_hit_is_hard(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(monkeypatch, zt_row=_zt(zt_cnt=29, zb_rate=0.41), hs_ctx=_hs())

    result = mv.veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["position_coef"] == 0.3
    assert [h["rule_id"] for h in result["trade_hits"]] == ["R1"]


def test_veto_single_weight1_trade_hit_is_soft(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(monkeypatch, zt_row=_zt(zt_cnt=19), hs_ctx=_hs(ret_5d=-0.031))

    result = mv.veto("2026-08-12")

    assert result["level"] == "soft"
    assert result["position_coef"] == 0.6
    assert [h["rule_id"] for h in result["trade_hits"]] == ["R3"]


def test_veto_macro_plus_trade_sum_reaches_3_is_hard(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(
        monkeypatch,
        zt_row=_zt(zt_cnt=19),
        hs_ctx=_hs(ret_5d=-0.031),
        pe=0.96,
    )

    result = mv.veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["position_coef"] == 0.3
    assert [h["rule_id"] for h in result["trade_hits"]] == ["R3"]


def test_veto_macro1_plus_trade1_sum2_is_soft(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(
        monkeypatch,
        zt_row=_zt(zt_cnt=19),
        hs_ctx=_hs(ret_5d=-0.031),
        pe=0.81,
    )

    result = mv.veto("2026-08-12")

    assert result["level"] == "soft"
    assert result["position_coef"] == 0.6
    assert [h["rule_id"] for h in result["trade_hits"]] == ["R3"]


def test_veto_no_hit_is_none(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(monkeypatch, zt_row=_zt(), hs_ctx=_hs())

    result = mv.veto("2026-08-12")

    assert result["level"] == "none"
    assert result["position_coef"] == 1.0
    assert result["trade_hits"] == []


def test_veto_missing_data_degrades_without_risk_increase(monkeypatch, tmp_path):
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(mv, "ROOT", tmp_path)
    _patch_veto(monkeypatch, zt_row=None, hs_ctx=None)

    result = mv.veto("2026-08-12")

    assert result["level"] == "none"
    assert result["position_coef"] == 1.0
    assert result["trade_hits"] == []


def test_get_veto_reads_legacy_json_without_trade_hits(monkeypatch, tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    old = {"date": "2026-08-11", "level": "soft", "position_coef": 0.6, "reasons": ["旧数据"]}
    (generated / "macro_veto_2026-08-11.json").write_text(
        json.dumps(old, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(mv, "ROOT", tmp_path)
    result = mv.get_veto("2026-08-11")

    assert result["level"] == "soft"
    assert result["position_coef"] == 0.6
    assert "trade_hits" not in result


def test_get_veto_reads_trade_hits_from_new_json(monkeypatch, tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    trade_hits = [{
        "rule_id": "R1",
        "name": "blast_retreat",
        "reason": "炸板率>40% 且涨停<30家，情绪退潮",
        "evidence": "error_book特征: 炸板率高+涨停少=情绪退潮",
        "weight": 2,
    }]
    new = {
        "date": "2026-08-12",
        "level": "hard",
        "position_coef": 0.3,
        "reasons": ["R1 blast_retreat: 炸板率>40% 且涨停<30家，情绪退潮"],
        "trade_hits": trade_hits,
    }
    (generated / "macro_veto_2026-08-12.json").write_text(
        json.dumps(new, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(mv, "ROOT", tmp_path)
    result = mv.get_veto("2026-08-12")

    assert result["level"] == "hard"
    assert result["trade_hits"] == trade_hits
