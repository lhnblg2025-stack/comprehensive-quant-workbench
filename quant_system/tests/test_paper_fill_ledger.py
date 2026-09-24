from __future__ import annotations

from quant_system import trade_db


def test_fill_execution_id_is_idempotent_and_reconciliation_is_per_order(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_db, "_DB_PATH", tmp_path / "trade_log.db")
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    order = trade_db.create_paper_order(symbol="600000", direction="buy", shares=200,
                                        suggested_price=10.0, release_id="release-1")
    first = trade_db.record_paper_fill(order["id"], execution_id="fill-1", filled_shares=100,
                                       filled_price=10.1, commission=5, transfer_fee=.1)
    duplicate = trade_db.record_paper_fill(order["id"], execution_id="fill-1", filled_shares=100,
                                           filled_price=10.1, commission=5, transfer_fee=.1)
    assert first["filled_shares"] == 100
    assert duplicate["duplicate"] is True
    fills = trade_db.list_paper_fills(release_id="release-1")
    assert len(fills) == 1
    assert fills[0]["transfer_fee"] == .1
    report = trade_db.paper_reconciliation(release_id="release-1")
    assert report["schema"] == "paper-execution-reconciliation/v1"
    assert report["summary"]["partial_orders"] == 1
    assert report["summary"]["fill_rate"] == .5


def test_fill_rejects_overfill_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(trade_db, "_DB_PATH", tmp_path / "trade_log.db")
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    order = trade_db.create_paper_order(symbol="000001", direction="sell", shares=100,
                                        suggested_price=12.0, release_id="release-2")
    result = trade_db.record_paper_fill(order["id"], execution_id="too-much", filled_shares=200,
                                        filled_price=12.0)
    assert result["error"] == "fill exceeds requested shares"
    assert trade_db.list_paper_fills(release_id="release-2") == []
