from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "unified_decision_snapshot", ROOT / "scripts" / "unified_decision_snapshot.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_board_scope_rules():
    assert MOD._board("600519") == "沪主板"
    assert MOD._board("000001") == "深主板"
    assert MOD._board("300750") == "创业板"
    assert MOD._board("688981") == "科创板"


def test_tradable_allows_main_board_only():
    allowed = MOD._tradable({"symbol": "600519", "name": "贵州茅台", "price": 1500})
    growth = MOD._tradable({"symbol": "300750", "name": "宁德时代", "price": 200})
    st = MOD._tradable({"symbol": "000001", "name": "*ST测试", "price": 3})
    assert allowed["trade_allowed"] is True
    assert growth["trade_allowed"] is False
    assert growth["trade_label"] == "观察"
    assert st["trade_allowed"] is False


def test_snapshot_contract():
    out = MOD.build_snapshot(mode="intraday", date="2026-08-25")
    assert out["ok"] is True
    assert out["schema_version"] == "decision-snapshot.v1"
    assert out["scope"]["analysis"] == "全市场"
    assert "概念/题材" in out["scope"]["analysis_detail"]
    assert "主板" in out["scope"]["execution"]
    assert isinstance(out["market"]["risk_flags"], list)
    assert isinstance(out["mainlines"], list)
    assert isinstance(out["opportunities"], list)
    assert isinstance(out["counts"]["execution_candidates"], int)
