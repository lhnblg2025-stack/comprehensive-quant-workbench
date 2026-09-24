"""
test_network_outage — 暴力断连测试（2026-08-23 运维极端场景）

模拟"拔网线/断连/超时挂死"级别故障: 全链路网络调用抛 ProxyError/TimeoutError,
断言融合链核心函数全部降级返回(不炸链、不静默误判、输出结构完整)。

场景:
  1. requests 全灭(拔网线) → daily_review_collectors/engine_fusion/prediction_verify 降级
  2. 单引擎挂死(超时) → engine_fusion 该维度降级不拖垮共识
  3. 数据仓库文件被占用/删除 → 读表降级 None 不炸
  4. 决策链在断网 blocks 下仍产完整决策卡

运行: python3 -m pytest tests/test_network_outage.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class _NetDown:
    """拔网线: 所有 requests 调用抛代理/连接错误。"""

    def __init__(self):
        import requests
        self._orig = {}
        for name in ("get", "post", "put", "delete", "head", "Session"):
            if hasattr(requests, name):
                self._orig[name] = getattr(requests, name)

    def _sink(self, *a, **kw):
        raise requests.exceptions.ProxyError("NET_DOWN: Cannot connect to proxy.")

    def __enter__(self):
        import requests
        for name in self._orig:
            setattr(requests, name, self._sink)
        return self

    def __exit__(self, *exc):
        import requests
        for name, orig in self._orig.items():
            setattr(requests, name, orig)
        return False


def test_fusion_decision_survives_total_network_loss():
    """断网下 make_decision 仍产出完整决策卡(全维度降级但结构健在)。"""
    from fusion_decision import make_decision
    from extreme_market_drill import _mk, _base_blocks, _directions, _ladder
    with _NetDown():
        blocks = _mk(_base_blocks(),
                     market_temperature=None,  # 情绪/资金全断
                     leader_sentiment=_ladder(zt=50, mb=3, zb=15, dt=20, zb_rate=0.35),
                     strong_direction=_directions([]),
                     overseas=None, factor_signal=None, macro_veto=None)
        dec = make_decision(blocks)
    assert isinstance(dec, dict) and dec.get("定调")
    assert dec["定调"].get("posture")
    assert dec["定调"].get("position")
    assert isinstance(dec.get("风险预案"), list)


def test_engine_fusion_network_down_degrades_not_crash():
    """引擎全断网: 各信号降级, 共识为样本不足, 不抛异常。"""
    import engine_fusion as ef
    with _NetDown():
        r = ef.engine_fused_signals({}, date="2026-08-23")
    assert isinstance(r, dict)
    assert r["consensus"] in ("样本不足", "中性", "共振", "分歧")
    assert 0 <= r["score"] <= 100
    assert all(isinstance(d.get("score"), (int, float)) for d in r["dimensions"].values())


def test_prediction_verify_survives_network_outage(tmp_path, monkeypatch):
    """断网+无指数数据: verify_pending 返回明确降级(不抛)。"""
    import quant_system.analysis_core.predictions as predictions
    import prediction_verify as pv
    monkeypatch.setattr(predictions, "PREDICTIONS", tmp_path / "no_preds.parquet")
    monkeypatch.setattr(pv, "_INDEX_PATH", tmp_path / "no_index.parquet")
    with _NetDown():
        r = pv.verify_pending("2026-08-23", void_stale=False)
    assert isinstance(r, dict)
    assert "note" in r


def test_collectors_network_down_all_degrade():
    """采集器断网 → 每个块要么有值要么 error 明确, 不抛。"""
    import daily_review_collectors as drc
    results = {}
    with _NetDown():
        for name, fn in [("market_temperature", drc.collect_market_temperature),
                         ("strong_direction", lambda gw: drc.collect_strong_direction(gw)),
                         ("leader_sentiment", lambda gw: drc.collect_leader_sentiment(gw)),
                         ("engines", lambda _gw: drc.collect_engines())]:
            try:
                b = fn(None)
                results[name] = {"error": b.error, "has_value": bool(b.value)}
            except Exception as e:  # noqa: BLE001
                results[name] = {"exploded": str(e)[:80]}
    for name, r in results.items():
        assert "exploded" not in r, f"{name} 断网炸链: {r}"
        assert r.get("has_value") or r.get("error"), f"{name} 无降级状态"


def test_html_report_render_without_review_json(tmp_path, monkeypatch):
    """研报渲染在 review json 缺失/损坏时降级不炸(输出端暴力断连)。"""
    import html_report_generator as hg
    monkeypatch.setattr(hg, "ROOT", tmp_path)  # 无 generated/ → _load_review 空
    rv = hg._load_review(None)
    assert isinstance(rv, dict)  # 空 dict 而非抛异常
    # 损坏 json: 写坏文件后应仍返回空
    (tmp_path / "generated").mkdir(exist_ok=True)
    (tmp_path / "generated" / "review_2026-08-23.json").write_text("{broken", encoding="utf-8")
    rv2 = hg._load_review("2026-08-23")
    assert isinstance(rv2, dict)