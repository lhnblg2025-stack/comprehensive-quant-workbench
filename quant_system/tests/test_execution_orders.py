"""execution.py 单元测试（域J：执行层订单语义）。

全 mock：trade_db 重定向 tmp、watchlist 行情桩，不触网。
覆盖：split_order 整手/零股守恒、部分成交状态机、滑点边界、
佣金/最低费用、check_orders 自动成交与降级、结构化错误。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import quant_system.execution as ex
import quant_system.trade_db as trade_db


@pytest.fixture
def trade_db_tmp(tmp_path, monkeypatch):
    """trade_db 重定向 tmp 并重置 execution 的列迁移缓存。"""
    monkeypatch.setattr(trade_db, "_DB_PATH", tmp_path / "trade_log.db")
    trade_db.init_db()
    monkeypatch.setattr(ex, "_fill_cost_cols_checked", False)
    return trade_db


@pytest.fixture
def no_quotes(monkeypatch):
    """行情桩：fetch_quotes 返回空，避免 estimate_slippage 触网。"""
    import quant_system.watchlist as watchlist
    monkeypatch.setattr(watchlist, "fetch_quotes", lambda symbols: [])
    return watchlist


@pytest.fixture
def in_market(monkeypatch):
    monkeypatch.setattr(ex, "_is_in_market_hours", lambda now=None: True)


# ════════════════════════════════════════════════════════════════
# 场景5：split_order 整手/零股语义（VWAP 拆分守恒）
# ════════════════════════════════════════════════════════════════

class TestSplitOrder:
    def _create(self, direction: str, shares: int, otype: str = "limit",
                price: float | None = 10.0) -> int:
        # F1.1: create_order 现会校验持仓，卖出订单需先建立足够持仓。
        if direction == "sell":
            trade_db.add_buy("600000", "测试", 10000, 10.0)
            # 买入记为当日，拆分路径后续会走自动成交/T+1，这里统一改为隔日以专注拆分语义
            with trade_db._conn() as conn:
                conn.execute("UPDATE trades SET trade_date='2000-01-01' WHERE symbol='600000'")
        r = ex.create_order("600000", otype, direction, shares, price=price)
        assert "error" not in r, r
        return r["id"]

    def test_split_conserves_lot_and_odd_lot(self, trade_db_tmp):
        # 卖出允许零股：1234 = 12 手 + 34 零股，拆 3 份 → 400/400/434
        oid = self._create("sell", 1234)
        subs = ex.split_order(oid, 3)
        assert len(subs) == 3
        assert [s["shares"] for s in subs] == [400, 400, 434]
        assert sum(s["shares"] for s in subs) == 1234  # 总量守恒
        for i, s in enumerate(subs):
            assert s["status"] == "sent"
            assert s["parent_order_id"] == oid
            assert s["direction"] == "sell"
            assert s["order_type"] == "limit"
            assert s["symbol"] == "600000"
            assert s["price"] == 10.0
            assert s["notes"].startswith(f"VWAP拆分 #{i + 1}/3")

    def test_split_even_lots(self, trade_db_tmp):
        oid = self._create("buy", 1200)
        subs = ex.split_order(oid, 3)
        assert [s["shares"] for s in subs] == [400, 400, 400]
        assert sum(s["shares"] for s in subs) == 1200

    def test_split_remainder_lots_and_odd_lot(self, trade_db_tmp):
        # 1350 = 13 手 + 50 零股，拆 4 份：base 3 手 + 前 1 份余 1 手 → 400/300/300/350
        oid = self._create("sell", 1350)
        subs = ex.split_order(oid, 4)
        assert [s["shares"] for s in subs] == [400, 300, 300, 350]
        assert sum(s["shares"] for s in subs) == 1350
        # 前 3 份均为整手，零股并入尾单
        assert all(s["shares"] % 100 == 0 for s in subs[:-1])
        assert subs[-1]["shares"] % 100 == 50

    def test_split_marks_parent_cancelled(self, trade_db_tmp):
        oid = self._create("sell", 1000)
        ex.split_order(oid, 2)
        parent = ex.get_order(oid)
        assert parent["status"] == "cancelled"
        assert "已拆分为2个子单" in parent["notes"]

    def test_split_invalid_slices(self, trade_db_tmp):
        oid = self._create("sell", 1000)
        r = ex.split_order(oid, 0)
        assert r == [{"error": "拆分份数必须 >= 1"}]

    def test_split_unknown_parent(self, trade_db_tmp):
        r = ex.split_order(99999, 2)
        assert "error" in r[0]

    def test_split_terminal_parent(self, trade_db_tmp):
        oid = self._create("sell", 1000)
        ex.cancel_order(oid)
        r = ex.split_order(oid, 2)
        assert "终态" in r[0]["error"]

    def test_split_fewer_lots_than_slices(self, trade_db_tmp):
        # 100 股拆 3 份 → 每份不足 1 手，子单跳过，仅保留尾单
        oid = self._create("sell", 100)
        subs = ex.split_order(oid, 3)
        assert [s["shares"] for s in subs] == [100]
        assert sum(s["shares"] for s in subs) == 100


# ════════════════════════════════════════════════════════════════
# 场景2：部分成交（fill_order / _fill_internal 状态机）
# ════════════════════════════════════════════════════════════════

class TestPartialFill:
    def test_partial_then_full_fill(self, trade_db_tmp):
        r = ex.create_order("600000", "limit", "buy", 1000, price=10.0)
        oid = r["id"]
        assert r["status"] == "pending"
        f1 = ex.fill_order(oid, 10.0, 400)
        assert f1["status"] == "partial_filled"
        assert f1["filled_shares"] == 400
        assert f1["shares"] == 1000
        f2 = ex.fill_order(oid, 10.5, 600)
        assert f2["status"] == "filled"
        assert f2["filled_shares"] == 1000
        # 成交明细落库
        detail = ex.get_order(oid)
        assert len(detail["fills"]) == 2
        assert detail["fills"][0]["filled_shares"] == 400

    def test_overfill_rejected_structured(self, trade_db_tmp):
        oid = ex.create_order("600000", "limit", "buy", 1000, price=10.0)["id"]
        ex.fill_order(oid, 10.0, 400)
        r = ex.fill_order(oid, 10.0, 700)  # 400+700 > 1000
        assert "error" in r and "超过剩余" in r["error"]

    def test_nonpositive_fill_rejected(self, trade_db_tmp):
        oid = ex.create_order("600000", "limit", "buy", 1000, price=10.0)["id"]
        r = ex.fill_order(oid, 10.0, 0)
        assert "error" in r and "必须 > 0" in r["error"]

    def test_fill_terminal_rejected(self, trade_db_tmp):
        oid = ex.create_order("600000", "limit", "buy", 1000, price=10.0)["id"]
        ex.fill_order(oid, 10.0, 1000)
        r = ex.fill_order(oid, 10.0, 100)
        assert "error" in r and "终态" in r["error"]

    def test_fill_unknown_order(self, trade_db_tmp):
        r = ex.fill_order(99999, 10.0, 100)
        assert "error" in r

    def test_fill_sell_t1_rejected(self, trade_db_tmp):
        trade_db.add_buy("600001", "测试", 100, 10.0)
        oid = ex.create_order("600001", "limit", "sell", 100, price=10.0)["id"]
        r = ex.fill_order(oid, 10.0, 100)
        assert "error" in r and "T+1" in r["error"]

    def test_partial_cancel_and_state_transitions(self, trade_db_tmp):
        oid = ex.create_order("600000", "limit", "buy", 1000, price=10.0)["id"]
        assert ex.approve_order(oid)["success"] is True
        assert ex.approve_order(oid)["success"] is False  # approved→approved 非法
        ex.fill_order(oid, 10.0, 400)
        assert ex.get_order(oid)["status"] == "partial_filled"
        # 部分成交后可撤单
        assert ex.cancel_order(oid)["success"] is True
        assert ex.get_order(oid)["status"] == "cancelled"
        assert ex.fill_order(oid, 10.0, 100)["error"]


# ════════════════════════════════════════════════════════════════
# 场景3：estimate_slippage 滑点边界（方向×市值×成交量）
# ════════════════════════════════════════════════════════════════

class TestEstimateSlippage:
    def test_buy_sell_direction_factor(self):
        assert ex.estimate_slippage("600000", "buy", 1000, market_cap_yi=800.0) == pytest.approx(0.1)
        # 卖出方向系数 1.1
        assert ex.estimate_slippage("600000", "sell", 1000, market_cap_yi=800.0) == pytest.approx(0.11)
        assert ex.estimate_slippage("600000", "sell", 1000, market_cap_yi=800.0) > \
               ex.estimate_slippage("600000", "buy", 1000, market_cap_yi=800.0)

    def test_market_cap_tiers(self):
        assert ex.estimate_slippage("s", "buy", 1000, 800) == 0.1
        assert ex.estimate_slippage("s", "buy", 1000, 200) == 0.2
        assert ex.estimate_slippage("s", "buy", 1000, 60) == 0.3
        assert ex.estimate_slippage("s", "buy", 1000, 49) == 0.5

    def test_volume_factor_boundaries(self):
        f = lambda sh: ex.estimate_slippage("s", "buy", sh, 800.0)
        assert f(10000) == 0.1
        assert f(10001) == 0.15  # >1 万 → ×1.5
        assert f(50000) == 0.15
        assert f(50001) == 0.2   # >5 万 → ×2.0
        assert f(100000) == 0.2
        assert f(100001) == 0.3  # >10 万 → ×3.0

    def test_missing_market_cap_falls_to_base(self, no_quotes):
        # 无市值/行情不可得 → 基础档 0.5
        assert ex.estimate_slippage("600000", "buy", 1000, market_cap_yi=None) == pytest.approx(0.5)
        assert ex.estimate_slippage("600000", "buy", 1000, market_cap_yi=0) == pytest.approx(0.5)


# ════════════════════════════════════════════════════════════════
# 场景6：佣金/最低费用（_fill_internal 落库）
# ════════════════════════════════════════════════════════════════

class TestFillCost:
    def test_fill_records_commission_and_stamp(self, trade_db_tmp):
        oid = ex.create_order("600000", "limit", "buy", 1000, price=10.0)["id"]
        ex.fill_order(oid, 10.0, 1000)
        detail = ex.get_order(oid)
        fill = detail["fills"][0]
        # 成交额 1 万 → 佣金最低 5 元；买入无印花税
        assert fill["commission"] == pytest.approx(5.0)
        assert fill["stamp_tax"] == pytest.approx(0.0)
        assert fill["transfer_fee"] == pytest.approx(10000.0 * ex.TRANSFER_FEE_RATE)

    def test_sell_fill_stamp_tax(self, trade_db_tmp):
        trade_db.add_buy("600001", "测试", 1000, 10.0)
        # 把买入交易日改为过去，模拟隔日持仓（绕开 T+1，专注费用断言）
        with trade_db._conn() as conn:
            conn.execute("UPDATE trades SET trade_date='2000-01-01' WHERE symbol='600001'")
        oid = ex.create_order("600001", "limit", "sell", 1000, price=10.0)["id"]
        ex.fill_order(oid, 10.0, 1000)
        fill = ex.get_order(oid)["fills"][0]
        assert fill["stamp_tax"] == pytest.approx(10000.0 * ex.STAMP_TAX_RATE)


# ════════════════════════════════════════════════════════════════
# 场景7：check_orders 自动成交 + 异常降级
# ════════════════════════════════════════════════════════════════

class TestCheckOrders:
    def test_market_order_auto_fill_with_slippage(self, trade_db_tmp, no_quotes, in_market):
        oid = ex.create_order("600000", "market", "buy", 100)["id"]
        assert ex.get_order(oid)["status"] == "sent"
        triggered = ex.check_orders({"600000": 10.0})
        assert len(triggered) == 1
        order = ex.get_order(oid)
        assert order["status"] == "filled"
        assert order["filled_shares"] == 100
        # 无市值 → 0.5% 滑点向上
        assert order["filled_price"] == pytest.approx(round(10.0 * 1.005, 4))
        pos = trade_db.get_position("600000")
        assert pos is not None and pos["shares"] == 100

    def test_limit_order_trigger(self, trade_db_tmp, no_quotes, in_market):
        oid = ex.create_order("600000", "limit", "buy", 100, price=10.0)["id"]
        ex.approve_order(oid)
        assert ex.check_orders({"600000": 9.0})  # 市价 ≤ 限价 → 触发
        order = ex.get_order(oid)
        assert order["status"] == "filled"
        assert order["filled_price"] <= 10.0  # 限价上限约束

    def test_limit_order_not_triggered(self, trade_db_tmp, no_quotes, in_market):
        oid = ex.create_order("600000", "limit", "buy", 100, price=8.0)["id"]
        ex.approve_order(oid)
        assert ex.check_orders({"600000": 9.0}) == []  # 市价 > 限价 → 不成交
        assert ex.get_order(oid)["status"] == "approved"

    def test_no_prices_no_fill(self, trade_db_tmp, in_market):
        oid = ex.create_order("600000", "market", "buy", 100)["id"]
        assert ex.check_orders({}) == []
        assert ex.get_order(oid)["status"] == "sent"

    def test_t1_degrades_to_rejected_with_note(self, trade_db_tmp, no_quotes, in_market):
        # F1.1 后不再存在"无持仓即创建卖单"；持仓以当日买入形式存在（持仓校验通过），
        # 自动成交路径中 T+1 校验失败 → 显式降级为 rejected + notes。
        trade_db.add_buy("600000", "测试", 100, 10.0)  # 当日买入 → 当日不可卖
        oid = ex.create_order("600000", "market", "sell", 100)["id"]
        assert ex.check_orders({"600000": 10.0}) == []
        order = ex.get_order(oid)
        assert order["status"] == "rejected"
        assert "T+1" in order["notes"]


# ════════════════════════════════════════════════════════════════
# create_order 校验 / 结构化错误（降级：不抛异常）
# ════════════════════════════════════════════════════════════════

class TestCreateOrderValidation:
    def test_invalid_order_type(self, trade_db_tmp):
        r = ex.create_order("600000", "bogus", "buy", 100)
        assert "error" in r and "无效订单类型" in r["error"]

    def test_invalid_direction(self, trade_db_tmp):
        r = ex.create_order("600000", "limit", "hold", 100, price=10.0)
        assert "error" in r and "无效方向" in r["error"]

    def test_nonpositive_shares(self, trade_db_tmp):
        assert "error" in ex.create_order("600000", "limit", "buy", 0, price=10.0)
        assert "error" in ex.create_order("600000", "limit", "buy", -100, price=10.0)

    def test_buy_must_be_lot_multiple(self, trade_db_tmp):
        r = ex.create_order("600000", "limit", "buy", 150, price=10.0)
        assert "error" in r and "100 股整数倍" in r["error"]

    def test_limit_requires_price(self, trade_db_tmp):
        r = ex.create_order("600000", "limit", "buy", 100)
        assert "error" in r and "限价" in r["error"]

    def test_trailing_buy_rejected(self, trade_db_tmp):
        r = ex.create_order("600000", "trailing_stop", "buy", 100, trailing_pct=5.0)
        assert "error" in r and "暂不支持" in r["error"]

    def test_market_order_sent(self, trade_db_tmp):
        r = ex.create_order("600000", "market", "buy", 100)
        assert r["status"] == "sent"
        assert "message" in r


# ════════════════════════════════════════════════════════════════
# K线边界（止损拒绝时段）
# ════════════════════════════════════════════════════════════════

class TestStopCutoff:
    def test_cutoff_window(self):
        t = datetime(2026, 8, 12, 14, 50, tzinfo=ex.CST)
        assert ex.is_stop_cutoff_period(t) is True
        t2 = datetime(2026, 8, 12, 10, 0, tzinfo=ex.CST)
        assert ex.is_stop_cutoff_period(t2) is False
        t3 = datetime(2026, 8, 12, 14, 44, tzinfo=ex.CST)
        assert ex.is_stop_cutoff_period(t3) is False
