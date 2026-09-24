"""
test_engine_fusion — 引擎→融合适配器单测（2026-08-22 融合升级）

覆盖: 高价值引擎信号提取、共识计算、<3引擎样本不足降级、
fusion_decision._score_engine 第7维接入（3元组契约 + available 规则）。

运行: python3 -m pytest tests/test_engine_fusion.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def fake_calls(monkeypatch):
    """替身 _call_engine：返回确定性引擎输出。"""
    import engine_fusion as ef
    canned = {
        "risk_system": {"ok": True, "value": {"temperature": 43, "stage": "accumulate"}},
        "meta_reviewer": {"ok": True, "value": {"conclusions": [
            {"stance": "neutral", "confidence": 0.5, "stale": False}]}},
        "decision_card": {"ok": True, "value": {"stance": "轻仓试错", "position_range": "10-20%"}},
        "multi_agent": {"ok": True, "value": {"consensus": "震荡", "confidence": 0.5}},
    }

    def fake(engine, entry, kwargs=None, timeout=18):
        return canned.get(engine, {"ok": False, "note": "missing"})

    monkeypatch.setattr(ef, "_call_engine", fake)

    def fake_trust(v):
        return {"score": 40, "note": "校准未闭环(0条已验证)", "ok": False}
    monkeypatch.setattr(ef, "_trust_signal", fake_trust)
    # 禁用产物优先路径 → expert 走 canned 确定性输出
    monkeypatch.setattr(ef, "_expert_from_artifact", lambda: None)
    return ef


def test_engine_fused_signals_consensus(fake_calls):
    ef = fake_calls
    r = ef.engine_fused_signals({}, date="2026-08-21")
    assert r["ok_n"] >= 3            # risk/meta/card OK
    # W2.5 修复后：只对 ok 维度加权平均（失败 trust=40 不再稀释共识分）
    assert r["score"] == pytest.approx(45.6, abs=0.2)  # 4 OK 维平均(risk/meta/card/expert)
    assert isinstance(r["score"], float)
    assert r["consensus"] in ("中性", "共振", "分歧", "样本不足")
    assert "risk" in r["dimensions"]
    assert r["dimensions"]["risk"]["ok"] is True


def test_engine_less_than_three_degraded(fake_calls, monkeypatch):
    ef = fake_calls

    def fake(engine, entry, kwargs=None, timeout=18):
        return {"ok": False, "note": "全挂"}
    monkeypatch.setattr(ef, "_call_engine", fake)
    r = ef.engine_fused_signals({}, date="2026-08-21")
    assert r["ok_n"] < 3
    assert r["consensus"] == "样本不足"
    assert r["score"] == 50.0


def test_score_engine_dimension_in_decision(monkeypatch):
    """引擎维度接入 make_decision：3元组契约 + ok_n<3 时 available=False。"""
    from fusion_decision import _score_engine, make_decision
    from daily_review_collectors import SignalBlock
    ef = {"date": "2026-08-21", "ok_n": 3, "score": 44.0, "consensus": "中性",
          "dimensions": {"risk": {"score": 44.0, "ok": True}}, "note": "x"}
    blocks = {"engine_fusion": SignalBlock("engine_fusion", "engine_fusion", ef)}
    s, note, avail = _score_engine(blocks)
    assert avail is True
    assert 0 <= s <= 100
    # 不足3引擎 → 不进加权
    ef2 = dict(ef, ok_n=2)
    blocks["engine_fusion"] = SignalBlock("engine_fusion", "engine_fusion", ef2)
    _, _, avail2 = _score_engine(blocks)
    assert avail2 is False
    # 无引擎块 → 不看
    _, _, avail3 = _score_engine({})
    assert avail3 is False
    # 全维度契约：make_decision 不炸
    dec = make_decision(blocks)
    assert "engine" in (dec.get("决策依据") or {})
    assert "engine" in (dec.get("fusion") or {}).get("dimensions", {})


def test_all_scorers_three_tuple_seven_dims():
    """七大维度 scorer 全部返回 (score, note, available) 3元组（2026-08-22 契约修复）。"""
    import fusion_decision as fd
    from daily_review_collectors import SignalBlock
    blocks = {}  # 空块：每个维度都应降级而不是抛错
    for name in ("technical", "emotion", "fund", "factor", "overseas", "mainline", "engine"):
        fn = getattr(fd, f"_score_{name}")
        r = fn(blocks)
        assert isinstance(r, tuple) and len(r) == 3, f"{name} 返回 {len(r)} 元组"
        assert isinstance(r[2], bool), f"{name} available 非 bool"
    # 全维度缺失时 make_decision 仍能产出（不炸链）
    dec = fd.make_decision(blocks)
    assert "定调" in dec

def test_battle_map_cross_check_detects_divergence():
    """业务逻辑审计①: 主链决策 vs 作战卡分歧必须可见（不静默覆盖）。"""
    from fusion_decision import make_decision
    from daily_review_collectors import SignalBlock

    base = {
        "market_temperature": {"date": "2026-08-21", "components": {
            "emotion": {"stage_cn": "发酵", "confidence": 0.5, "zt_cnt": 74, "max_board": 3},
            "fund": {"force_index": 60.0, "youzi_net": 2e9, "jg_net": 0, "north_net": 0}}},
        "leader_sentiment": {"date": "2026-08-21", "components": {
            "ladder": {"zt_cnt": 74, "max_board": 3, "zb_cnt": 11, "dt_cnt": 17},
            "leader": {"overheat": 3}}},
        "strong_direction": {"directions": [{"name": "半导体", "level": "主线", "score": 3}]},
        "stock_picks": {"pools": {"lhb_stocks": [{"name": "x", "code": "1", "net": 1e8}],
                                   "rs_stocks": []}},
        "technical": {"indices": [{"name": "沪深300", "score": 80, "trend": "多头"}]},
        "overseas": {"quotes": [{"label": "纳指", "chg_pct": 0.5}]},
        "factor_signal": {"groups": {"技术动量": {"score": 0.45, "n": 6},
                                     "质量": {"score": 0.45, "n": 4},
                                     "量价": {"score": 0.45, "n": 4}}},
        "macro_veto": {"level": "soft", "position_coef": 0.9, "pe_percentile": 0.6},
        "freshness": {"score": 90, "scan": {"kline": {"lag": 1, "level": "fresh"}}},
        "engine_fusion": {"ok_n": 3, "score": 60, "consensus": "共振", "note": ""},
    }
    blocks = {k: SignalBlock(k, "x", v) for k, v in base.items()}

    # 场景1: 作战卡保守(试错8-16%) vs 主链积极 → 分歧标注
    blocks["battle_map_card"] = SignalBlock("battle_map_card", "battle_map", {
        "action_card": "试错", "position_range": "8-16%", "confidence": 0.6})
    dec1 = make_decision(blocks)
    assert "决策分歧" in (dec1["定调"].get("cross_note") or ""), dec1["定调"]

    # 场景2: 作战卡一致(进攻50-70%) → 无分歧
    blocks["battle_map_card"] = SignalBlock("battle_map_card", "battle_map", {
        "action_card": "进攻", "position_range": "50-70%", "confidence": 0.6})
    dec2 = make_decision(blocks)
    assert not (dec2["定调"].get("cross_note") or ""), dec2["定调"]

    # 场景3: 无作战卡 → 不炸, 无 cross_note
    del blocks["battle_map_card"]
    dec3 = make_decision(blocks)
    assert dec3["定调"].get("posture")

    # 场景4: 作战卡更激进但主链被宏观否决压制 → 以否决为准
    blocks["battle_map_card"] = SignalBlock("battle_map_card", "battle_map", {
        "action_card": "进攻", "position_range": "60-80%", "confidence": 0.6})
    blocks["freshness"] = SignalBlock("freshness", "x", {"score": 10, "scan": {}})
    blocks["macro_veto"] = SignalBlock("macro_veto", "x", {"level": "hard", "position_coef": 0.2})
    dec4 = make_decision(blocks)
    assert "宏观否决压制" in (dec4["定调"].get("cross_note") or ""), dec4["定调"]
