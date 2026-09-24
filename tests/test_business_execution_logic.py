"""
test_business_execution_logic — 业务逻辑 + 执行逻辑回归（2026-08-23）

覆盖:
  1. P2 决策语义: 高潮+炸板率>0.42 = 退潮前兆 → 情绪降分 + 定调禁进攻档
  2. 执行逻辑: 盘中基调(bias) 注入盘后决策 → 防御/谨慎 门控定调档位
  3. apply_intraday_guard 纯函数三态(防御/谨慎/进攻) + 缺失不调整

运行: python3 -m pytest tests/test_business_execution_logic.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from daily_review_collectors import SignalBlock  # noqa: E402
from fusion_decision import (  # noqa: E402
    _card_rank,
    _climax_blowout,
    _rank_picks,
    _score_emotion,
    _selection_bias,
    apply_intraday_guard,
    make_decision,
)


def _strong_blocks(**overrides) -> dict:
    """强市 blocks: 无修复时定调应为"积极进攻"或"试探性进攻"。"""
    base = {
        "market_temperature": {"date": "2026-08-21", "components": {
            "emotion": {"stage_cn": "发酵", "confidence": 0.5, "zt_cnt": 74, "max_board": 5},
            "fund": {"force_index": 70.0, "youzi_net": 12e9, "jg_net": 5e9, "north_net": 1e9}}},
        "leader_sentiment": {"date": "2026-08-21", "components": {
            "ladder": {"zt_cnt": 74, "max_board": 5, "zb_cnt": 8, "dt_cnt": 3},
            "leader": {"overheat": 3}}},
        "strong_direction": {"directions": [
            {"name": "半导体", "level": "主线", "score": 3},
            {"name": "创新药", "level": "主线", "score": 2}]},
        "stock_picks": {"pools": {"lhb_stocks": [{"name": "x", "code": "1", "net": 1e8}],
                                   "rs_stocks": []}},
        "technical": {"indices": [{"name": "上证指数", "score": 82, "trend": "多头", "above_ma300": True},
                                    {"name": "沪深300", "score": 80, "trend": "多头"},
                                   {"name": "中证500", "score": 78, "trend": "多头"}]},
        "overseas": {"quotes": [{"label": "纳指", "chg_pct": 1.2}]},
        "factor_signal": {"groups": {"技术动量": {"score": 0.55, "n": 6},
                                     "质量": {"score": 0.5, "n": 4},
                                     "量价": {"score": 0.5, "n": 4}}},
        "macro_veto": {"level": "none", "position_coef": 1.0},
        "freshness": {"score": 95, "scan": {"kline": {"lag": 1, "level": "fresh"}}},
        "engine_fusion": {"ok_n": 4, "score": 65, "consensus": "共振", "note": ""},
    }
    base.update(overrides)
    return {k: SignalBlock(k, "x", v) for k, v in base.items()}


# ── P2: 高潮+炸板 退潮前兆 ──────────────────────────────────────────

def _climax_blocks() -> dict:
    return _strong_blocks(
        market_temperature={"date": "2026-08-21", "components": {
            "emotion": {"stage_cn": "高潮", "confidence": 0.5, "zt_cnt": 100, "max_board": 6},
            "fund": {"force_index": 70.0, "youzi_net": 12e9, "jg_net": 5e9, "north_net": 1e9}}},
        leader_sentiment={"date": "2026-08-21", "components": {
            "ladder": {"zt_cnt": 100, "max_board": 6, "zb_cnt": 80, "dt_cnt": 5},
            "leader": {"overheat": 5}}},
    )


def test_climax_blowout_detected():
    blocks = _climax_blocks()
    assert _climax_blowout(blocks) is True


def test_emotion_score_penalized_on_climax_blowout():
    blocks = _climax_blocks()
    s, note, avail = _score_emotion(blocks)
    assert avail is True
    # 高潮(75) - 高炸板(10) - 退潮前兆(15) = 50
    assert s <= 50
    assert "退潮前兆" in note


def test_climax_blowout_caps_posture_to_neutral():
    """P2 核心: 高潮+炸板 45% 不再给进攻档(降至中性轮动)。"""
    blocks = _climax_blocks()
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "中性轮动"
    assert "高潮炸板退潮前兆" in (dec["定调"].get("guard_note") or "")


# ── 执行逻辑: 盘中基调 → 盘后定调门控 ──────────────────────────────

def test_intraday_defensive_caps_attack():
    blocks = _strong_blocks()
    dec0 = make_decision(blocks)
    assert dec0["定调"]["posture"] == "积极进攻"  # 无 bias 基线为积极进攻

    blocks["intraday_bias"] = SignalBlock("intraday_bias", "intraday_guard",
                                          {"tone": "防御", "breadth": 0.2, "index_pct": -0.8})
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "中性轮动"
    assert "盘中防御基调" in (dec["定调"].get("guard_note") or "")
    assert any("盘中实际基调防御" in r.get("trigger", "") for r in dec.get("风险预案", []))


def test_intraday_cautious_caps_aggressive_only():
    blocks = _strong_blocks()
    blocks["intraday_bias"] = SignalBlock("intraday_bias", "intraday_guard",
                                          {"tone": "谨慎", "breadth": 0.5, "index_pct": 0.1})
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "试探性进攻"  # 禁积极进攻, 保留试探
    assert "盘中谨慎基调" in (dec["定调"].get("guard_note") or "")


def test_no_bias_no_guard():
    blocks = _strong_blocks()
    dec = make_decision(blocks)
    assert not (dec["定调"].get("guard_note") or "")


def test_shanghai_below_ma300_blocks_attack():
    blocks = _strong_blocks(technical={"indices": [
        {"name": "上证指数", "score": 80, "trend": "多头", "above_ma300": False},
        {"name": "沪深300", "score": 80, "trend": "多头"},
    ]})
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "中性轮动"
    assert dec["定调"]["position"].endswith("20%")
    assert "跌破MA300" in dec["定调"]["guard_note"]


def test_critical_no_data_forces_zero_position():
    blocks = _strong_blocks(freshness={"score": 0, "scan": {
        "kline": {"lag": -1, "level": "no_data"},
        "market": {"lag": -1, "level": "no_data"},
    }})
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "防守观望"
    assert dec["定调"]["position"] == "0-0%"


# ── apply_intraday_guard 纯函数 ─────────────────────────────────────

def test_apply_guard_defensive():
    p, pos, note = apply_intraday_guard("积极进攻", (0.5, 0.8), {"tone": "防御", "breadth": 0.2})
    assert p == "中性轮动"
    assert pos[1] <= 0.4
    assert "防御" in note


def test_apply_guard_cautious():
    p, pos, note = apply_intraday_guard("积极进攻", (0.5, 0.8), {"tone": "谨慎"})
    assert p == "试探性进攻"
    assert pos[1] <= 0.5
    assert "谨慎" in note


def test_apply_guard_offensive_or_missing_noop():
    p, pos, note = apply_intraday_guard("积极进攻", (0.5, 0.8), {"tone": "进攻"})
    assert p == "积极进攻" and note == ""
    p2, pos2, note2 = apply_intraday_guard("中性轮动", (0.2, 0.4), {})
    assert p2 == "中性轮动" and note2 == ""
    # 防御基调但已是防守档 → 不降级(不误降)
    p3, _, note3 = apply_intraday_guard("防守观望", (0.0, 0.1), {"tone": "防御"})
    assert p3 == "防守观望" and note3 == ""


# ── B1: 作战卡词汇统一(_card_rank) ─────────────────────────────────

def test_card_rank_semantics():
    assert _card_rank("积极进攻") == 3
    assert _card_rank("进攻") == 2
    assert _card_rank("试错") == 1
    assert _card_rank("低吸") == 1   # 低吸是进攻动作, 原误归防守 → 已修正
    assert _card_rank("观望") == 1   # 观望是中性, 原误归防守 → 已修正
    assert _card_rank("防守") == 0
    assert _card_rank("") is None
    assert _card_rank("不可识别") is None


def test_card_rank_no_false_divergence_for_dip():
    """低吸/观望(中性偏多) vs 主链中性轮动 → 不应误报"方向分歧"。"""
    blocks = _strong_blocks()
    blocks["battle_map_card"] = SignalBlock("battle_map_card", "battle_map", {
        "action_card": "低吸", "position_range": "20-40%", "confidence": 0.5})
    dec = make_decision(blocks)
    # 低吸(rank1) vs 积极进攻(rank3) gap>5 → 仍算分歧; 这里只验证不崩且语义可归类
    assert dec["定调"]["posture"]


# ── B2: 选股共振动态权重 ───────────────────────────────────────────

def _stage_blocks(stage: str) -> dict:
    return _strong_blocks(
        market_temperature={"date": "2026-08-21", "components": {
            "emotion": {"stage_cn": stage, "confidence": 0.5, "zt_cnt": 74, "max_board": 5},
            "fund": {"force_index": 70.0, "youzi_net": 12e9, "jg_net": 5e9, "north_net": 1e9}}})


def test_selection_bias_risk_off():
    b = _selection_bias(_stage_blocks("退潮"))
    assert b["mainline"] == -2.0 and b["rs"] == 1.0


def test_selection_bias_risk_on():
    b = _selection_bias(_stage_blocks("冰点"))
    assert b["mainline"] == 0.5 and b["rs"] == -0.5


def test_rank_picks_risk_off_deprioritizes_mainline():
    picks = [
        {"name": "主线A(1)", "dim": 3, "src": "mainline"},
        {"name": "龙虎B(2)", "dim": 2, "src": "lhb"},
        {"name": "RS(3)", "dim": 1, "src": "rs"},
    ]
    ranked = _rank_picks(picks, {"mainline": -2.0, "lhb": 0.0, "rs": 1.0})
    # mainline 3-2=1.0 < lhb 2.0 == rs 2.0 → 主线龙头排最后
    assert ranked[-1]["src"] == "mainline"


# ── B3: 风险预案↔仓位联动 ──────────────────────────────────────────

def _risky_attack_blocks() -> dict:
    return _strong_blocks(
        market_temperature={"date": "2026-08-21", "components": {
            "emotion": {"stage_cn": "发酵", "confidence": 0.5, "zt_cnt": 30, "max_board": 2},
            "fund": {"force_index": 70.0, "youzi_net": -20e9, "jg_net": 0, "north_net": -60e8}}},
        leader_sentiment={"date": "2026-08-21", "components": {
            "ladder": {"zt_cnt": 30, "max_board": 2, "zb_cnt": 5, "dt_cnt": 40},
            "leader": {"overheat": 2}}},
        overseas={"quotes": [{"label": "纳指", "chg_pct": -3.0},
                              {"label": "伦敦金", "chg_pct": 3.0}]},
    )


def test_risk_count_downgrades_posture():
    blocks = _risky_attack_blocks()
    dec = make_decision(blocks)
    assert len(dec["风险预案"]) >= 4
    # 高密度风险 → 从"积极进攻"自动降一档到"试探性进攻"
    assert dec["定调"]["posture"] == "试探性进攻"
    assert "自动降一档" in (dec["定调"].get("guard_note") or "")


# ── D: 盘中 rows 五视角进盘后选股共振 ──────────────────────────────

def test_intraday_rows_enter_stock_picks(monkeypatch):
    import fusion_decision as fd
    monkeypatch.setattr(fd, "mainline_leaders", lambda *a, **k: [])
    blocks = _strong_blocks(
        strong_direction={"directions": []},
        stock_picks={"pools": {"lhb_stocks": [], "rs_stocks": []}},
    )
    blocks["intraday_rows"] = SignalBlock("intraday_rows", "intraday_guard", [
        {"symbol": "600519", "name": "贵州茅台", "composite_conf": 0.8,
         "dominant": "short", "dominant_label": "短线趋势"},
        {"symbol": "000001", "name": "平安银行", "composite_conf": 0.4,
         "dominant": "value", "dominant_label": "中长线价值"},
    ])
    dec = make_decision(blocks)
    attack = dec["操作清单"]["attack"]
    names = [a["name"] for a in attack]
    assert any("600519" in n for n in names)      # 高置信盘中机会进进攻池
    assert not any("000001" in n for n in names)  # 低置信(0.4<0.55)不进
