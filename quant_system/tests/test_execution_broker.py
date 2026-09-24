"""execution_broker.py 单元测试（域J：执行层）。

全 mock：不连真实券商、不写真实用户目录（OrderDB 重定向到 tmp）。
覆盖：下单失败重试/异常不静默、部分成交状态机、滑点方向与 0/负值边界、
断连重连、整手/零股（SimBroker 100 股粒度）、佣金/最低费用、异常降级。
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

import quant_system.execution_broker as eb


@pytest.fixture
def order_db(tmp_path, monkeypatch):
    """OrderDB 重定向到 tmp，避免污染真实 ~/.quant_system。"""
    db = eb.OrderDB(db_path=str(tmp_path / "orders.db"))
    monkeypatch.setattr(eb, "OrderDB", lambda *a, **k: db)
    return db


@pytest.fixture
def sim_broker(order_db):
    return eb.SimBroker(account_id="sim", initial_cash=1_000_000.0)


class _StubBroker(eb.Broker):
    """可配置失败次数的桩 Broker：前 fail_times 次抛异常，之后成交。"""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None, **kwargs):
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("券商接口异常")
        self.calls = 0
        super().__init__(**kwargs)

    def _do_place_order(self, order: eb.Order, **kwargs) -> eb.Order:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        order.status = "filled"
        order.filled_volume = order.volume
        order.avg_price = order.price
        return order

    def _do_cancel_order(self, order_id: str) -> bool:
        return True

    def _do_get_positions(self) -> list[eb.Position]:
        return []

    def _do_get_account(self) -> eb.AccountInfo:
        return eb.AccountInfo(account_id=self.account_id)


# ════════════════════════════════════════════════════════════════
# 场景1：下单失败重试（异常不静默、无幻影订单）
# ════════════════════════════════════════════════════════════════

class TestPlaceOrderRetry:
    def test_failure_propagates_not_silent(self, order_db):
        b = _StubBroker(fail_times=1, account_id="acc")
        with pytest.raises(RuntimeError, match="券商接口异常"):
            b.place_order(symbol="600000", side="buy", volume=100, price=10.0)
        assert b.calls == 1
        # 无幻影订单：失败尝试不注册、不落库
        assert b._orders == {}
        assert b.get_orders() == []
        assert b.get_order("any") is None

    def test_retry_until_success(self, order_db):
        b = _StubBroker(fail_times=2, account_id="acc")
        order = None
        attempts = 0
        # 调用方重试契约：每次失败抛异常，可重试 N 次，直至成功
        for _ in range(3):
            attempts += 1
            try:
                order = b.place_order(symbol="600000", side="buy", volume=100, price=10.0)
                break
            except RuntimeError:
                continue
        assert attempts == 3
        assert b.calls == 3
        assert order is not None
        assert order.status == "filled"
        assert order.filled_volume == 100
        # 成功后注册并落库
        assert b._orders[order.order_id] is order
        assert b.get_order(order.order_id) is not None

    def test_retry_exhausted_raises_on_last_attempt(self, order_db):
        b = _StubBroker(fail_times=5, account_id="acc")
        last_exc = None
        attempts = 0
        for _ in range(3):
            attempts += 1
            try:
                b.place_order(symbol="600000", side="buy", volume=100, price=10.0)
            except RuntimeError as e:
                last_exc = e
        # 重试耗尽后异常必须冒泡到调用方（最终失败明确可见，不静默）
        assert last_exc is not None
        assert attempts == 3
        assert b.calls == 3
        assert b._orders == {}

    def test_place_order_success_path(self, order_db):
        b = _StubBroker(account_id="acc")
        order = b.place_order(symbol="600000", side="buy", volume=100,
                              order_type="market", price=10.0)
        assert order.order_id.startswith("acc_")
        assert order.status == "filled"
        assert order.symbol == "600000"
        # 持久化后可被新实例（模拟进程重启）通过 DB 读取
        b2 = _StubBroker(account_id="acc")
        loaded = b2.get_order(order.order_id)
        assert loaded is not None and loaded.status == "filled"


# ════════════════════════════════════════════════════════════════
# 场景3：滑点方向与边界（买=向上，卖=向下；0/负值字面公式）
# ════════════════════════════════════════════════════════════════

class TestSimBrokerSlippage:
    def test_buy_slips_up(self, sim_broker):
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="market", price=100.0)
        assert o.status == "filled"
        # 默认万10 滑点：买方向成交价上浮
        assert o.avg_price == pytest.approx(100.0 * 1.001, rel=1e-9)
        assert o.avg_price > 100.0

    def test_sell_slips_down(self, sim_broker):
        # 直接播种可卖持仓（last_buy_date 为过去，绕开 T+1）
        sim_broker._positions["600000"] = eb.Position(
            symbol="600000", volume=100, cost_price=100.0, last_buy_date="2000-01-01")
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "sell", 100, order_type="market", price=100.0)
        assert o.status == "filled"
        # 卖方向成交价下浮
        assert o.avg_price == pytest.approx(100.0 * 0.999, rel=1e-9)
        assert o.avg_price < 100.0
        assert sim_broker.get_positions() == []  # 持仓清零后移除

    def test_zero_slippage_exec_price_equals_market(self, sim_broker):
        class _ZeroImpact:
            def total_cost(self, **kw):
                return {"cost_rate": 0.0}

        sim_broker._impact = _ZeroImpact()
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="market",
                                   price=100.0, adv=1_000_000, sigma=0.02)
        assert o.avg_price == pytest.approx(100.0, rel=1e-9)

    def test_negative_slippage_boundary(self, sim_broker):
        class _NegImpact:
            def total_cost(self, **kw):
                return {"cost_rate": -0.01}

        sim_broker._impact = _NegImpact()
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="market",
                                   price=100.0, adv=1_000_000, sigma=0.02)
        # 边界记录：公式字面执行 price*(1+cost_rate)，负滑点不额外兜底
        assert o.avg_price == pytest.approx(100.0 * 0.99, rel=1e-9)

    def test_limit_order_wait_then_fill(self, sim_broker):
        sim_broker.update_market_price("600000", 10.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="limit", price=8.0)
        assert o.status == "pending"  # 市价 > 限价，挂单等待
        assert "限价未触发" in o.reason
        sim_broker.update_market_price("600000", 7.0)
        o2 = sim_broker.place_order("600000", "buy", 100, order_type="limit", price=8.0)
        assert o2.status == "filled"  # 触发后按市价 ± 滑点成交
        assert o2.avg_price == pytest.approx(7.0 * 1.001, rel=1e-9)

    def test_limit_sell_not_triggered(self, sim_broker):
        sim_broker.update_market_price("600000", 6.0)
        o = sim_broker.place_order("600000", "sell", 100, order_type="limit", price=8.0)
        assert o.status == "pending"
        assert "限价未触发" in o.reason

    def test_no_market_price_limit_stays_pending(self, sim_broker):
        o = sim_broker.place_order("600000", "buy", 100, order_type="limit", price=8.0)
        assert o.status == "pending"
        assert "等待行情" in o.reason


# ════════════════════════════════════════════════════════════════
# 场景5/6：SimBroker 整手粒度 + 佣金/最低费用应用
# ════════════════════════════════════════════════════════════════

class TestSimBrokerCostAndLots:
    def test_buy_cost_deducts_commission_and_transfer(self, sim_broker):
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="market", price=100.0)
        exec_price = 100.0 * 1.001
        gross = 100 * exec_price
        expected_cost = gross + 5.0 + gross * 1e-5  # 佣金最低5 + 过户费
        assert sim_broker.cash == pytest.approx(1_000_000 - expected_cost, rel=1e-9)
        t = sim_broker._trade_history[0]
        assert t["commission"] == pytest.approx(5.0, rel=1e-9)
        assert t["stamp_tax"] == 0.0  # 买入无印花税
        assert t["transfer_fee"] == pytest.approx(gross * 1e-5, rel=1e-9)

    def test_sell_cost_includes_stamp_tax(self, sim_broker):
        sim_broker._positions["600000"] = eb.Position(
            symbol="600000", volume=100, cost_price=100.0, last_buy_date="2000-01-01")
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "sell", 100, order_type="market", price=100.0)
        assert o.status == "filled"
        t = sim_broker._trade_history[-1]
        assert t["stamp_tax"] == pytest.approx(o.avg_price * 100 * eb.STAMP_TAX_RATE, rel=1e-9)
        assert t["stamp_tax"] > 0  # 卖出计印花税

    def test_lot_sized_fill_volume(self, sim_broker):
        # A股整手语义：成交股数即委托股数（100 股粒度由调用方保证）
        sim_broker.update_market_price("600000", 10.0)
        o = sim_broker.place_order("600000", "buy", 200, order_type="market", price=10.0)
        assert o.filled_volume == 200
        assert sim_broker._positions["600000"].volume == 200

    def test_average_cost_update(self, sim_broker):
        sim_broker.update_market_price("600000", 10.0)
        sim_broker.place_order("600000", "buy", 100, order_type="market", price=10.0)
        sim_broker.update_market_price("600000", 12.0)
        sim_broker.place_order("600000", "buy", 100, order_type="market", price=12.0)
        pos = sim_broker._positions["600000"]
        assert pos.volume == 200
        assert pos.cost_price == pytest.approx(11.0, rel=1e-3)
        # T+1 当日买入股数累计
        assert pos.today_buy_volume == 200

    def test_reset_clears_state(self, sim_broker):
        sim_broker.update_market_price("600000", 10.0)
        sim_broker.place_order("600000", "buy", 100, order_type="market", price=10.0)
        sim_broker.reset(cash=500_000.0)
        assert sim_broker.cash == 500_000.0
        assert sim_broker._positions == {}
        assert sim_broker._orders == {}
        assert sim_broker._market_prices == {}
        assert sim_broker._trade_history == []

    def test_sellable_volume_t1(self, sim_broker):
        pos = eb.Position(symbol="600000", volume=100, today_buy_volume=100,
                          last_buy_date=datetime.now().strftime("%Y-%m-%d"))
        assert sim_broker._sellable_volume(pos) == 0
        pos2 = eb.Position(symbol="600000", volume=100, today_buy_volume=100,
                           last_buy_date="2000-01-01")
        assert sim_broker._sellable_volume(pos2) == 100


# ════════════════════════════════════════════════════════════════
# 场景6：佣金/最低费用模型（estimate_trade_cost）
# ════════════════════════════════════════════════════════════════

class TestEstimateTradeCost:
    def test_min_commission_floor_5(self):
        c = eb.estimate_trade_cost("buy", 100, 100.0)  # 成交额 1 万
        assert c["gross"] == 10_000.0
        assert c["commission"] == pytest.approx(5.0)  # 万0.85 → 0.85 元，触发最低 5 元
        assert c["stamp_tax"] == 0.0
        assert c["transfer_fee"] == pytest.approx(0.1)  # 万0.1 双边

    def test_commission_rate_applies_when_above_floor(self):
        c = eb.estimate_trade_cost("buy", 100_000, 100.0)  # 成交额 1e7
        assert c["commission"] == pytest.approx(1e7 * eb.COMMISSION_RATE)  # 850 元
        assert c["commission"] > eb.MIN_COMMISSION

    def test_stamp_tax_only_on_sell(self):
        sell = eb.estimate_trade_cost("sell", 100, 100.0)
        assert sell["stamp_tax"] == pytest.approx(10_000.0 * eb.STAMP_TAX_RATE)
        buy = eb.estimate_trade_cost("buy", 100, 100.0)
        assert buy["stamp_tax"] == 0.0

    def test_min_commission_override_01(self):
        # 最低费用参数化：0.1 元保护同样生效（成交额 1000 → 佣金 0.085 < 0.1 → 取 0.1）
        c = eb.estimate_trade_cost("buy", 100, 10.0, min_commission=0.1)
        assert c["commission"] == pytest.approx(0.1)

    def test_zero_quantity_boundary(self):
        c = eb.estimate_trade_cost("sell", 0, 100.0)
        assert c["gross"] == 0.0
        assert c["commission"] == pytest.approx(eb.MIN_COMMISSION)
        assert c["stamp_tax"] == 0.0
        assert c["transfer_fee"] == 0.0


# ════════════════════════════════════════════════════════════════
# 场景4：断连重连（QMT / PTrade 桩）
# ════════════════════════════════════════════════════════════════

class TestBrokerReconnect:
    def test_qmt_disconnected_rejected(self, order_db):
        b = eb.QmtBroker(account_id="q1")
        o = b.place_order("600000", "buy", 100, price=10.0)
        assert o.status == "rejected"
        assert "未连接" in o.reason  # 结构化错误，不抛异常

    def test_qmt_connect_disconnect_reconnect(self, order_db):
        b = eb.QmtBroker(account_id="q1")
        assert b.connect() is True and b._connected is True
        o1 = b.place_order("600000", "buy", 100, price=10.0)
        assert o1.status == "submitted"
        assert o1.filled_volume == 0  # 未确认成交
        b.disconnect()
        assert b._connected is False
        o2 = b.place_order("600000", "buy", 100, price=10.0)
        assert o2.status == "rejected" and "未连接" in o2.reason
        # 重连后恢复下单
        assert b.connect() is True
        o3 = b.place_order("600000", "buy", 100, price=10.0)
        assert o3.status == "submitted"
        assert b.cancel_order(o1.order_id) is True

    def test_ptrade_reconnect(self, order_db):
        b = eb.PTradeBroker(account_id="p1")
        o0 = b.place_order("600000", "buy", 100, price=10.0)
        assert o0.status == "rejected" and "未连接" in o0.reason
        assert b.connect() is True
        o1 = b.place_order("600000", "buy", 100, price=10.0)
        assert o1.status == "submitted"
        # PTradeBroker 无 disconnect 方法：以置位 _connected 模拟断连
        b._connected = False
        assert b.place_order("600000", "buy", 100, price=10.0).status == "rejected"
        b.connect()
        assert b.place_order("600000", "buy", 100, price=10.0).status == "submitted"


    def test_qmt_account_positions_sync(self, order_db):
        b = eb.QmtBroker(account_id="q1")
        assert b.get_account().account_id == "q1"
        assert b.get_positions() == []
        b.sync()  # 桩同步不抛异常

    def test_cancel_active_order_marks_cancelled(self, order_db):
        b = eb.QmtBroker(account_id="q1")
        b.connect()
        o = b.place_order("600000", "buy", 100, price=10.0)
        assert o.status == "submitted"
        assert b.cancel_order(o.order_id) is True
        assert o.status == "cancelled"
        assert o.is_active() is False
        assert b.cancel_order(o.order_id) is False  # 终态不可重复撤
    def test_create_broker_factory(self, order_db):
        assert isinstance(eb.create_broker("sim"), eb.SimBroker)
        assert isinstance(eb.create_broker("qmt"), eb.QmtBroker)
        assert isinstance(eb.create_broker("ptrade"), eb.PTradeBroker)
        with pytest.raises(ValueError, match="未知 Broker 类型"):
            eb.create_broker("unknown")


# ════════════════════════════════════════════════════════════════
# 场景7：异常降级（结构化错误而非抛到上层）
# ════════════════════════════════════════════════════════════════

class TestDegradedBehavior:
    def test_invalid_price_rejected(self, sim_broker):
        o = sim_broker.place_order("600000", "buy", 100, price=0.0)
        assert o.status == "rejected"
        assert o.reason == "无效价格"

    def test_insufficient_cash_rejected(self, order_db):
        b = eb.SimBroker(account_id="sim", initial_cash=1000.0)
        b.update_market_price("600000", 100.0)
        o = b.place_order("600000", "buy", 100, order_type="market", price=100.0)
        assert o.status == "rejected"
        assert "资金不足" in o.reason

    def test_sell_without_position_rejected(self, sim_broker):
        sim_broker.update_market_price("600000", 100.0)
        o = sim_broker.place_order("600000", "sell", 100, order_type="market", price=100.0)
        assert o.status == "rejected"
        assert "持仓不足" in o.reason

    def test_sell_t1_rejected(self, sim_broker):
        sim_broker.update_market_price("600000", 100.0)
        sim_broker.place_order("600000", "buy", 100, order_type="market", price=100.0)
        o = sim_broker.place_order("600000", "sell", 100, order_type="market", price=100.0)
        assert o.status == "rejected"
        assert "T+1" in o.reason

    def test_cancel_unknown_and_terminal(self, sim_broker):
        assert sim_broker.cancel_order("nonexistent") is False
        sim_broker.update_market_price("600000", 10.0)
        o = sim_broker.place_order("600000", "buy", 100, order_type="market", price=10.0)
        assert sim_broker.cancel_order(o.order_id) is False  # 已成交终态
        assert o.status == "filled"


# ════════════════════════════════════════════════════════════════
# 场景2：部分成交状态机（Order / LiveOrder 语义）
# ════════════════════════════════════════════════════════════════

class TestOrderStates:
    def test_order_is_active_and_pnl(self):
        o = eb.Order(order_id="1", side="buy", volume=100, filled_volume=40,
                     avg_price=10.0, status="partial")
        assert o.is_active() is True
        assert o.pnl(12.0) == pytest.approx(80.0)  # (12-10)*40
        o.status = "filled"
        assert o.is_active() is False
        sell = eb.Order(order_id="2", side="sell", volume=100, filled_volume=50,
                        avg_price=10.0, status="partial")
        assert sell.pnl(8.0) == pytest.approx(100.0)  # (10-8)*50
        assert sell.pnl(0.0) == pytest.approx(0.0)  # 无市价 → 用成本价

    def test_live_order_partial_fill(self):
        from quant_system.live_trading_manager import LiveOrder
        o = LiveOrder("600000", "buy", 1000, "limit", 10.0)
        assert o.status == "pending" and o.remaining == 1000
        o.fill(400, 10.0)
        assert o.status == "partial"          # 部分成交
        assert o.filled_volume == 400
        assert o.remaining == 600             # 剩余量正确
        assert o.avg_price == pytest.approx(10.0)
        assert o.is_active is True
        o.fill(600, 12.0)
        assert o.status == "filled"           # 补足后终态
        assert o.filled_volume == 1000
        assert o.remaining == 0
        assert o.avg_price == pytest.approx(11.2)  # (400*10+600*12)/1000
        assert o.is_active is False


# ════════════════════════════════════════════════════════════════
# OrderDB 持久化 + StrategyRunner 调仓
# ════════════════════════════════════════════════════════════════

class TestOrderDBAndStrategyRunner:
    def test_orderdb_roundtrip_and_filter(self, tmp_path):
        db = eb.OrderDB(db_path=str(tmp_path / "o.db"))
        o1 = eb.Order(order_id="a1", symbol="600000", side="buy", volume=100,
                      status="filled", created_at="2026-08-12T10:00:00",
                      updated_at="2026-08-12T10:00:00")
        o2 = eb.Order(order_id="a2", symbol="000001", side="sell", volume=200,
                      status="partial", created_at="2026-08-12T10:01:00",
                      updated_at="2026-08-12T10:01:00")
        db.save_order(o1)
        db.save_order(o2)
        assert db.load_order("a1") == o1
        assert db.load_order("missing") is None
        loaded = db.load_orders(symbol="600000")
        assert [o.order_id for o in loaded] == ["a1"]
        assert len(db.load_orders()) == 2


    def test_orderdb_date_filters(self, tmp_path):
        db = eb.OrderDB(db_path=str(tmp_path / "o2.db"))
        for i, ts in enumerate(["2026-08-01T10:00:00", "2026-08-10T10:00:00", "2026-08-12T10:00:00"]):
            db.save_order(eb.Order(order_id=f"d{i}", symbol="600000", side="buy",
                                   volume=100, status="filled", created_at=ts, updated_at=ts))
        assert len(db.load_orders(date_from="2026-08-10T00:00:00")) == 2
        assert len(db.load_orders(date_to="2026-08-01T23:59:59")) == 1
        assert len(db.load_orders(date_from="2026-08-10T00:00:00", date_to="2026-08-10T23:59:59")) == 1
    def test_get_orders_merges_memory_and_db(self, order_db):
        b = _StubBroker(account_id="acc")
        b.place_order(symbol="600000", side="buy", volume=100, price=10.0)
        # 新实例模拟重启：内存空、DB 有记录 → get_orders 合并
        b2 = _StubBroker(account_id="acc")
        assert len(b2.get_orders()) == 1
        assert b2.get_orders(status="filled")[0].symbol == "600000"

    def test_strategy_runner_buy_and_skip(self, sim_broker):
        from quant_system.execution_broker import StrategyRunner
        r = StrategyRunner(sim_broker, strategy_name="t")
        r.set_target("600000", 100)
        orders = r.execute({"600000": 10.0})
        assert len(orders) == 1 and orders[0].status == "filled"
        # diff=0 → 不重复下单
        assert r.execute({"600000": 10.0}) == []
        # 市价不可用 → 跳过不抛异常
        r.set_target("000001", 100)
        assert r.execute({"000001": 0.0}) == []
        assert r.execute({"000001": -1.0}) == []

    def test_strategy_runner_close_all(self, sim_broker):
        from quant_system.execution_broker import StrategyRunner
        r = StrategyRunner(sim_broker)
        r.set_target("600000", 100)
        r.execute({"600000": 10.0})
        assert len(sim_broker.get_positions()) == 1
        # 播种为隔日持仓，绕开 T+1，使 close_all 的卖出可成交
        sim_broker._positions["600000"].last_buy_date = "2000-01-01"
        orders = r.close_all()
        assert len(orders) == 1 and orders[0].side == "sell"
        assert sim_broker.get_positions() == []
