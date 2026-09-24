from __future__ import annotations

import sqlite3

import pytest

from quant_system.closed_loop import (
    append_candidates,
    attribute_candidate,
    get_candidate,
    create_linked_paper_order,
    init_candidate_store,
    link_paper_order,
    record_exit,
    record_fill,
    sync_trade_db_fill,
    batch_replay_exits,
    transition_candidate,
)


def _seed(path, status="PAPER_BLOCKED"):
    init_candidate_store(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO candidates (candidate_id,as_of,symbol,strategy_version,release_id,data_sha256,plan_json,status) VALUES (?,?,?,?,?,?,?,?)",
                   ("c1", "2024-01-01", "000001", "s1", "r1", "d1", "{}", status))


def test_candidate_lifecycle_rejects_skips_and_records_events(tmp_path):
    path = tmp_path / "candidates.db"
    _seed(path)
    with pytest.raises(ValueError):
        transition_candidate(path, "c1", "FILLED")
    transition_candidate(path, "c1", "PAPER_PENDING")
    transition_candidate(path, "c1", "CONFIRMED")
    link_paper_order(path, "c1", 42)
    record_fill(path, "c1", order_id=42, shares=100, price=10.5)
    record_exit(path, "c1", price=11.0, reason="target")
    final = attribute_candidate(path, "c1", payload={"net_return": 0.04})
    assert final["status"] == "ATTRIBUTED"
    assert final["filled_shares"] == 100
    assert final["exit_reason"] == "target"
    with sqlite3.connect(path) as db:
        assert db.execute("select count(*) from candidate_events where candidate_id='c1'").fetchone()[0] == 6


def test_create_linked_paper_order_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADE_DB_DIR", str(tmp_path / "trade"))
    path = tmp_path / "candidates.db"
    _seed(path, status="PAPER_PENDING")
    with pytest.raises(ValueError):
        create_linked_paper_order(path, "c1", direction="buy", shares=100, suggested_price=10)
    transition_candidate(path, "c1", "CONFIRMED")
    linked = create_linked_paper_order(path, "c1", direction="buy", shares=100, suggested_price=10)
    assert linked["order_id"] is not None


def test_sync_trade_db_fill_bridge(tmp_path, monkeypatch):
    trade_dir = tmp_path / "trade"
    monkeypatch.setenv("TRADE_DB_DIR", str(trade_dir))
    from quant_system import trade_db
    trade_db.init_db()
    order = trade_db.create_paper_order(symbol="000001", direction="buy", shares=100, suggested_price=10, release_id="r1", candidate_id="c1")
    trade_db.record_paper_fill(order["id"], filled_shares=100, filled_price=10.2)
    path = tmp_path / "candidates.db"
    _seed(path, status="CONFIRMED")
    link_paper_order(path, "c1", order["id"])
    synced = sync_trade_db_fill(path, "c1", order["id"])
    assert synced["status"] == "FILLED"
    assert synced["filled_price"] == pytest.approx(10.2)


def test_batch_replay_exits_and_attributes(tmp_path):
    path = tmp_path / "candidates.db"
    _seed(path, status="FILLED")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE candidates SET filled_shares=100, filled_price=10, plan_json=? WHERE candidate_id='c1'", ('{"stop_loss": 9, "take_profit": 11, "cost_bps": 15}',))
    panel = __import__('pandas').DataFrame([{"code": "000001", "date": "2024-01-02", "open": 10, "high": 11.5, "low": 9.5, "close": 11.2}])
    result = batch_replay_exits(path, panel)
    assert result["attributed"] == 1
    assert get_candidate(path, "c1")["status"] == "ATTRIBUTED"


def test_fill_requires_linked_order(tmp_path):
    path = tmp_path / "candidates.db"
    _seed(path, status="PAPER_PENDING")
    with pytest.raises(ValueError):
        record_fill(path, "c1", order_id=7, shares=100, price=10)
