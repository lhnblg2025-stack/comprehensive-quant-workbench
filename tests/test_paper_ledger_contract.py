from __future__ import annotations

from quant_system import trade_db
from quant_web import server


def test_trade_db_books_buy_and_sell_fees_in_cash_amount(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_log.db"
    monkeypatch.setattr(trade_db, "_DB_PATH", db_path)
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    trade_db.init_db()
    trade_db.add_buy("600519", "测试股", 100, 10.0, trade_date="2026-08-28", commission=5.0, transfer_fee=0.01)
    bought = trade_db.get_trades(days=3650, limit=10)[-1]
    assert bought["gross_amount"] == 1000.0
    assert bought["total_amount"] == 1005.01
    assert trade_db.get_position("600519")["total_cost"] == 1005.01

    trade_db.add_sell("600519", 100, 11.0, trade_date="2026-08-29", commission=5.0, stamp_tax=0.55, transfer_fee=0.01)
    sold = trade_db.get_trades(days=3650, limit=10)[0]
    assert sold["gross_amount"] == 1100.0
    assert sold["total_amount"] == 1094.44
    assert sold["commission"] == 5.0
    assert sold["stamp_tax"] == 0.55


def test_trade_db_rejects_non_finite_and_negative_fees(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_log.db"
    monkeypatch.setattr(trade_db, "_DB_PATH", db_path)
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    trade_db.init_db()

    for invalid in (float("nan"), float("inf"), -0.01):
        result = trade_db.add_buy("600000", "测试股", 100, 10.0, commission=invalid)
        assert "error" in result
    assert trade_db.get_position("600000") is None


def test_trade_db_rejects_sell_fees_above_gross_without_mutation(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_log.db"
    monkeypatch.setattr(trade_db, "_DB_PATH", db_path)
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    trade_db.init_db()
    trade_db.add_buy("600000", "测试股", 100, 10.0, trade_date="2026-08-28")

    result = trade_db.add_sell(
        "600000", 100, 10.0, trade_date="2026-08-29",
        commission=600.0, stamp_tax=401.0,
    )
    assert "费用合计不能超过成交额" in result["error"]
    assert trade_db.get_position("600000")["shares"] == 100
    assert len(trade_db.get_trades(days=3650, limit=10)) == 1


def test_trade_db_rejects_non_finite_price(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_log.db"
    monkeypatch.setattr(trade_db, "_DB_PATH", db_path)
    monkeypatch.setattr(trade_db, "_DB_DIR", tmp_path)
    trade_db.init_db()
    result = trade_db.add_buy("600000", "测试股", 100, float("nan"))
    assert "error" in result
    assert trade_db.get_position("600000") is None
    base = server._paper_order_fingerprint("buy", "600519", 100, 10.0)
    assert base != server._paper_order_fingerprint("sell", "600519", 100, 10.0)
    assert base != server._paper_order_fingerprint("buy", "600519", 200, 10.0)
    assert base != server._paper_order_fingerprint("buy", "600519", 100, 10.01)
