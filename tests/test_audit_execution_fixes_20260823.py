from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from daily_review_collectors import SignalBlock  # noqa: E402
from fusion_decision import make_decision  # noqa: E402


def _strong_blocks(**overrides) -> dict:
    base = {
        "market_temperature": {"components": {
            "emotion": {"stage_cn": "发酵", "confidence": 0.5, "zt_cnt": 80, "max_board": 5},
            "fund": {"force_index": 80.0, "youzi_net": 12e9, "jg_net": 5e9, "north_net": 1e9}}},
        "leader_sentiment": {"components": {
            "ladder": {"zt_cnt": 80, "max_board": 5, "zb_cnt": 5, "dt_cnt": 2},
            "leader": {"overheat": 3}}},
        "strong_direction": {"directions": [{"name": "半导体", "level": "主线", "score": 3}]},
        "stock_picks": {"pools": {"lhb_stocks": [{"name": "x", "code": "1", "net": 1e8}], "rs_stocks": []}},
        "technical": {"indices": [{"name": "沪深300", "score": 85, "trend": "多头"}]},
        "overseas": {"quotes": [{"label": "纳指", "chg_pct": 1.0}]},
        "factor_signal": {"groups": {"技术动量": {"score": 0.6, "n": 6}, "质量": {"score": 0.55, "n": 4}}},
        "macro_veto": {"level": "none", "position_coef": 1.0},
        "freshness": {"score": 100, "scan": {"kline": {"lag": 1, "level": "fresh"}}},
        "engine_fusion": {"ok_n": 4, "score": 65, "consensus": "共振", "note": ""},
    }
    base.update(overrides)
    return {k: SignalBlock(k, "test", v) for k, v in base.items()}


def test_hard_macro_veto_forces_defensive_posture():
    blocks = _strong_blocks(macro_veto={"level": "hard", "position_coef": 0.3, "pe_percentile": 0.9})
    dec = make_decision(blocks)
    assert dec["定调"]["posture"] == "防守观望"
    assert "宏观否决hard" in dec["定调"].get("veto_note", "")


def test_trade_db_facade_uses_keyword_semantics(monkeypatch):
    import quant_system.trade_db as tdb

    calls = {}

    def fake_add_buy(symbol, name, shares, price, signal_type="", notes="", sector="", trade_date=""):
        calls["buy"] = {
            "symbol": symbol, "name": name, "shares": shares, "price": price,
            "signal_type": signal_type, "notes": notes, "sector": sector, "trade_date": trade_date,
        }
        return {"ok": True}

    def fake_add_sell(symbol, shares, price, notes="", trade_date=""):
        calls["sell"] = {"symbol": symbol, "shares": shares, "price": price, "notes": notes, "trade_date": trade_date}
        return {"ok": True}

    def fake_set_stop(symbol, stop_loss_price=None, take_profit_price=None, stop_pct=None, take_profit_pct=None):
        calls.setdefault("stops", []).append({
            "symbol": symbol, "stop_loss_price": stop_loss_price, "take_profit_price": take_profit_price,
            "stop_pct": stop_pct, "take_profit_pct": take_profit_pct,
        })
        return {"ok": True}

    monkeypatch.setattr(tdb, "add_buy", fake_add_buy)
    monkeypatch.setattr(tdb, "add_sell", fake_add_sell)
    monkeypatch.setattr(tdb, "set_stop", fake_set_stop)

    db = tdb.TradeDB()
    db.add_buy("600000", "浦发银行", 100, 10.0, signal_type="sig", notes="note", sector="银行", trade_date="2026-08-23", stop_loss=9.5)
    db.add_sell("600000", 100, 11.0, notes="sell-note", trade_date="2026-08-24")
    db.set_stop("600000", stop_loss_price=9.0, take_profit_price=12.0, stop_pct=-0.08, take_profit_pct=0.15)

    assert calls["buy"]["sector"] == "银行"
    assert calls["buy"]["trade_date"] == "2026-08-23"
    assert calls["stops"][0]["stop_loss_price"] == 9.5
    assert calls["sell"] == {"symbol": "600000", "shares": 100, "price": 11.0, "notes": "sell-note", "trade_date": "2026-08-24"}
    assert calls["stops"][1]["take_profit_price"] == 12.0
    assert calls["stops"][1]["stop_pct"] == -0.08


def test_stock_picks_uses_canonical_mainline_contract(tmp_path, monkeypatch):
    import daily_review_collectors as drc

    gen = tmp_path / "generated"
    gen.mkdir()
    (gen / "after_close_extra_2026-08-21.json").write_text(json.dumps({"value_picks": []}), encoding="utf-8")
    monkeypatch.setattr(drc, "ROOT", tmp_path)
    monkeypatch.setattr(drc, "_canonical_mainlines", lambda date=None, limit=10: {
        "status": "watch", "items": [{"name": "AI", "status": "watch", "level": "观察方向",
                                         "score": 52.0, "evidence": {"funding": {"ok": False}}}]})

    block = drc.collect_stock_picks(None, date="2026-08-21")
    assert block.error is None
    assert block.value["pools"]["short_term"][0]["level"] == "观察方向"
    assert block.value["pools"]["short_term"][0]["signal"] == "观察候选"


def test_order_qty_can_be_completed_before_confirm(tmp_path, monkeypatch):
    from quant_system import trade_executor as te

    monkeypatch.setattr(te, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(te, "ORDERS_FILE", tmp_path / "trade_orders.json")
    te._save_orders([{"id": "ORD1", "action": "buy", "qty": 0, "qty_needs_manual": True, "status": "pending"}])

    assert te.set_order_qty("ORD1", 155, price=12.3) is True
    saved = json.loads(te.ORDERS_FILE.read_text(encoding="utf-8"))
    assert saved[0]["qty"] == 100
    assert saved[0]["qty_needs_manual"] is False
    assert saved[0]["price"] == 12.3


def test_order_merge_preserves_concurrent_status_update(tmp_path, monkeypatch):
    from quant_system import trade_executor as te

    monkeypatch.setattr(te, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(te, "ORDERS_FILE", tmp_path / "trade_orders.json")
    te._save_orders([{"id": "GUI1", "stock": "600000", "strategy": "manual",
                      "action": "buy", "status": "confirmed"}])

    pending = te._merge_new_orders([
        {"id": "SCAN1", "stock": "000001", "strategy": "strategy_engine",
         "action": "buy", "status": "pending"},
    ])

    saved = json.loads(te.ORDERS_FILE.read_text(encoding="utf-8"))
    assert {row["id"]: row["status"] for row in saved} == {
        "GUI1": "confirmed", "SCAN1": "pending",
    }
    assert [row["id"] for row in pending] == ["SCAN1"]


def test_order_merge_deduplicates_pending_business_key(tmp_path, monkeypatch):
    from quant_system import trade_executor as te

    monkeypatch.setattr(te, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(te, "ORDERS_FILE", tmp_path / "trade_orders.json")
    te._save_orders([{"id": "OLD", "stock": "000001", "strategy": "strategy_engine",
                      "action": "buy", "status": "pending"}])
    te._merge_new_orders([{"id": "NEW", "stock": "000001", "strategy": "strategy_engine",
                           "action": "buy", "status": "pending"}])

    saved = json.loads(te.ORDERS_FILE.read_text(encoding="utf-8"))
    assert [row["id"] for row in saved] == ["OLD"]


def test_order_dispatcher_high_vol_stop(tmp_path, monkeypatch):
    from quant_system.analysis_core import order_dispatcher as od

    gen = tmp_path / "generated"
    gen.mkdir()
    date = "2026-08-23"
    (gen / f"battle_map_{date}.json").write_text(json.dumps({
        "position_range": "0-30%",
        "emotion_stage": "修复",
        "macro_veto": "none",
        "regime": "高波市",
        "attack_groups": {"core": [{"leader": {"code": "600000", "name": "浦发银行"}, "score": 80, "strategy": "趋势低吸"}]},
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(od, "ROOT", tmp_path)
    monkeypatch.setattr(od, "_correlation_constraint", lambda orders: {"note": "test", "scale": 1.0})

    res = od.dispatch(date=date, capital=100000)
    assert res["orders"][0]["stop"] == "-3.5%"


def test_frontend_uses_authoritative_review_decision():
    from quant_web.handlers.review import _frontend_model

    review = {"date": "2026-08-23", "generated_at": "now", "blocks": {}, "battle_map": {}, "decision": {"sentinel": True}}
    model = _frontend_model(review)
    assert model["decision"] == {"sentinel": True}


def test_quant_web_protected_endpoints_fail_closed_without_key(monkeypatch):
    import quant_web.server as server

    handler = object.__new__(server.QuantHandler)
    handler.path = "/api/backtest/run"
    handler.headers = {}
    monkeypatch.setattr(server, "_QUANT_WEB_API_KEY", "")
    monkeypatch.setattr(server, "_ALLOW_UNAUTH", False)
    monkeypatch.setattr(server, "_IS_LOOPBACK_BIND", False)
    assert handler._path_protected("/api/backtest/run") is True
    assert handler._require_key() is False

    monkeypatch.setattr(server, "_IS_LOOPBACK_BIND", True)
    monkeypatch.setattr(server, "_ALLOW_UNAUTH", True)
    assert handler._require_key() is True


def test_save_to_youdao_requires_env_key(monkeypatch, capsys):
    import scripts.save_to_youdao as youdao

    monkeypatch.setattr(youdao, "API_KEY", "")
    assert youdao.save_note("t", "content", retries=1) is False
    assert "YOUDAONOTE_API_KEY" in capsys.readouterr().err
