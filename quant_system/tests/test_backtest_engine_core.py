"""backtest_engine 核心引擎单元测试（域 W 双轨制）。

轨道1 功能矩阵：配置、多空完整闭环、资金校验、持仓记录、绩效字段、空数据、连续交易。
轨道2 已知 bug 回归：做空滑点方向、最低佣金、停牌 volume=0、整手化。

全部使用合成 BarData/DataFrame，不读真实数据、不触网。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from quant_system.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    BarData,
    OrderData,
    PositionData,
    StrategyTemplate,
)


SYMBOL = "600000"


def make_bars(
    prices,
    symbol: str = SYMBOL,
    volume: float = 100000.0,
    start: str = "2026-08-03",
) -> pd.DataFrame:
    """构造单标的 OHLCV DataFrame。"""
    dates = pd.date_range(start, periods=len(prices), freq="D")
    prices = [float(x) for x in prices]
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [x + 0.1 for x in prices],
            "low": [x - 0.1 for x in prices],
            "close": prices,
            "volume": [float(volume)] * len(prices),
            "amount": [float(volume) * x for x in prices],
        }
    )


def make_engine(
    prices,
    *,
    capital: float = 100_000.0,
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
    slippage_mode: str = "none",
    fill_mode: str = "current_close",
) -> BacktestEngine:
    """构造一个适合精确手算的引擎。"""
    engine = BacktestEngine()
    engine.set_capital(capital)
    engine.set_commission(commission_rate)
    engine.set_slippage(slippage_rate, slippage_mode)
    engine.set_fill_mode(fill_mode)
    engine.set_market_rules(False)
    engine.limit_up = 1.0
    engine.limit_down = -1.0
    engine.min_commission = 0.0
    engine.transfer_fee_rate = 0.0
    engine.max_volume_pct = 0.0
    engine.order_timeout_bars = 100
    engine.add_data(SYMBOL, make_bars(prices))
    return engine


class BuyThenSellStrategy(StrategyTemplate):
    """第一根 bar 买入，第二根 bar 快照持仓并卖出。"""

    volume = 1000

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.position_snapshot = None

    def on_bar(self, bar: BarData) -> None:
        self.bars_seen += 1
        if self.bars_seen == 1:
            self.buy(bar.close, self.volume, symbol=bar.symbol)
        elif self.bars_seen == 2:
            pos = self.bg.positions[bar.symbol]
            self.position_snapshot = (pos.direction, pos.volume, pos.price)
            self.sell(bar.close, self.volume, symbol=bar.symbol)


class ShortThenCoverStrategy(StrategyTemplate):
    """第一根 bar 开空，第二根 bar 快照持仓并平空。"""

    volume = 1000

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.position_snapshot = None

    def on_bar(self, bar: BarData) -> None:
        self.bars_seen += 1
        if self.bars_seen == 1:
            self.short(bar.close, self.volume, symbol=bar.symbol)
        elif self.bars_seen == 2:
            pos = self.bg.positions[bar.symbol]
            self.position_snapshot = (pos.direction, pos.volume, pos.price)
            self.cover(bar.close, self.volume, symbol=bar.symbol)


class RoundTripStrategy(StrategyTemplate):
    """奇数根买入、偶数根卖出，连续做多轮。"""

    volume = 1000

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0

    def on_bar(self, bar: BarData) -> None:
        self.bars_seen += 1
        if self.bars_seen % 2 == 1:
            self.buy(bar.close, self.volume, symbol=bar.symbol)
        else:
            self.sell(bar.close, self.volume, symbol=bar.symbol)


class NoopStrategy(StrategyTemplate):
    """不产生任何交易，用于绩效字段与空回测对照。"""

    def on_bar(self, bar: BarData) -> None:
        return None


# ════════════════════════════════════════════════════════════════
# 轨道1：功能矩阵
# ════════════════════════════════════════════════════════════════


class TestEngineConfig:
    def test_setters_update_internal_state(self):
        engine = BacktestEngine()
        engine.set_capital(123_456.0)
        assert engine.initial_capital == pytest.approx(123_456.0)
        assert engine.capital == pytest.approx(123_456.0)

        engine.set_commission(0.0003)
        assert engine.commission_rate == pytest.approx(0.0003)

        engine.set_slippage(0.002, "percent")
        assert engine.slippage_rate == pytest.approx(0.002)
        assert engine.slippage_mode == "percent"

        engine.set_fill_mode("current_close")
        assert engine.fill_mode == "current_close"

    def test_invalid_config_rejects(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError):
            engine.set_capital(0)
        with pytest.raises(ValueError):
            engine.set_commission(1)
        with pytest.raises(ValueError):
            engine.set_slippage(0.01, "bad_mode")
        with pytest.raises(ValueError):
            engine.set_fill_mode("bad_mode")


class TestLongFlow:
    def test_buy_then_sell_full_closed_loop(self):
        engine = make_engine([10.0, 11.0])
        strategy = engine.add_strategy(BuyThenSellStrategy)
        result = engine.run()

        direction, volume, price = strategy.position_snapshot
        assert direction == "long"
        assert volume == pytest.approx(1000.0)
        assert price == pytest.approx(10.0)
        assert len(result.trades) == 2
        assert [t.offset for t in result.trades] == ["open", "close"]
        assert [t.direction for t in result.trades] == ["long", "short"]
        assert [o.status for o in result.orders] == ["filled", "filled"]

        # 卖出 11000×0.05% A股印花税=5.5，100000+1000-5.5=100994.5
        assert result.equity_curve["cash"].iloc[-1] == pytest.approx(100_994.5)
        assert result.total_return == pytest.approx(0.009945)

    def test_position_data_fields_and_validation(self):
        pos = PositionData(symbol=SYMBOL, direction="long", volume=200.0, price=9.5)
        assert pos.symbol == SYMBOL
        assert pos.direction == "long"
        assert pos.volume == pytest.approx(200.0)
        assert pos.frozen == pytest.approx(0.0)
        assert pos.pnl == pytest.approx(0.0)
        assert pos.pnl_ratio == pytest.approx(0.0)

        with pytest.raises(ValueError):
            PositionData(symbol=SYMBOL, direction="invalid")

    def test_floating_pnl_long_matches_last_close(self):
        engine = make_engine([10.0, 11.0])
        engine.positions[SYMBOL] = PositionData(
            symbol=SYMBOL, direction="long", volume=1000.0, price=10.0
        )
        engine.current_dt = engine.bars[SYMBOL][1].datetime
        engine._calculate_pnl()

        pos = engine.positions[SYMBOL]
        assert pos.pnl == pytest.approx(1000.0)
        assert pos.pnl_ratio == pytest.approx(0.1)

    def test_floating_pnl_short_matches_last_close(self):
        engine = make_engine([10.0, 9.0])
        engine.positions[SYMBOL] = PositionData(
            symbol=SYMBOL, direction="short", volume=1000.0, price=10.0
        )
        engine.current_dt = engine.bars[SYMBOL][1].datetime
        engine._calculate_pnl()

        pos = engine.positions[SYMBOL]
        assert pos.pnl == pytest.approx(1000.0)
        assert pos.pnl_ratio == pytest.approx(0.1)


class TestShortFlow:
    def test_short_then_cover_full_closed_loop(self):
        engine = make_engine([10.0, 9.0])
        strategy = engine.add_strategy(ShortThenCoverStrategy)
        result = engine.run()

        direction, volume, price = strategy.position_snapshot
        assert direction == "short"
        assert volume == pytest.approx(1000.0)
        assert price == pytest.approx(10.0)
        assert len(result.trades) == 2
        assert [t.offset for t in result.trades] == ["open", "close"]
        assert [t.direction for t in result.trades] == ["short", "long"]
        assert [o.status for o in result.orders] == ["filled", "filled"]

        # 做空 10 -> 9，盈利 1000，平空买入不收印花税。
        assert result.equity_curve["cash"].iloc[-1] == pytest.approx(101_000.0)
        assert result.total_return == pytest.approx(0.01)


class TestFundsAndEmpty:
    def test_insufficient_capital_rejects_open(self):
        engine = make_engine([10.0, 10.0], capital=500.0)
        bar = engine.bars[SYMBOL][0]
        order = OrderData(
            symbol=SYMBOL,
            order_id="too_poor",
            direction="long",
            offset="open",
            price=10.0,
            volume=100.0,
            datetime=bar.datetime,
            order_type="market",
        )
        engine._process_single_order(order, bar)

        assert engine.trades == []
        assert order.filled == pytest.approx(0.0)
        assert order.status == "submitted"

    def test_empty_add_data_does_not_register_bars(self):
        engine = BacktestEngine()
        with pytest.warns(UserWarning, match="Empty DataFrame"):
            engine.add_data(SYMBOL, pd.DataFrame())
        assert engine.bars == {}
        assert engine.positions == {}

    def test_no_overlap_returns_empty_result(self):
        engine = make_engine([10.0, 11.0])
        engine.add_strategy(BuyThenSellStrategy)
        result = engine.run(start="2020-01-01", end="2020-01-02")
        assert isinstance(result, BacktestResult)
        assert result.equity_curve.empty
        assert result.trades == []


class TestBacktestResultAndConservation:
    def test_result_metrics_for_small_sample(self):
        engine = make_engine([10.0, 11.0])
        engine.add_strategy(NoopStrategy)
        result = engine.run()

        assert isinstance(result.equity_curve, pd.DataFrame)
        assert {"datetime", "total_value", "cash", "position_value", "returns"} <= set(
            result.equity_curve.columns
        )
        assert result.total_return == pytest.approx(0.0)
        assert result.max_drawdown == pytest.approx(0.0)
        assert result.total_trades == 0
        assert isinstance(result.annual_return, float)
        assert isinstance(result.annual_volatility, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.profit_loss_ratio, float)

    def test_multiple_round_trips_cash_conservation(self):
        # 10->11 赚 1000，9->12 赚 3000；两次卖出印花税 5.5 + 6.0。
        engine = make_engine([10.0, 11.0, 9.0, 12.0])
        engine.add_strategy(RoundTripStrategy)
        result = engine.run()

        assert result.total_trades == 2
        assert engine.positions[SYMBOL].volume == pytest.approx(0.0)
        assert result.equity_curve["total_value"].iloc[-1] == pytest.approx(103_988.5)
        assert result.equity_curve["cash"].iloc[-1] == pytest.approx(103_988.5)
        assert result.total_return == pytest.approx(0.039885)


# ════════════════════════════════════════════════════════════════
# 轨道2：已知 bug 回归
# ════════════════════════════════════════════════════════════════


class TestSlippageDirectionRegression:
    def _engine(self) -> BacktestEngine:
        engine = BacktestEngine()
        engine.set_capital(100_000.0)
        engine.set_commission(0.0)
        engine.set_slippage(0.01, "fixed")
        engine.set_fill_mode("current_close")
        engine.set_market_rules(False)
        engine.min_commission = 0.0
        engine.transfer_fee_rate = 0.0
        engine.max_volume_pct = 0.0
        return engine

    def _bar(self, close: float, volume: float = 1000.0) -> BarData:
        return BarData(
            symbol=SYMBOL,
            datetime=datetime(2026, 8, 3, 10, 0),
            open=close,
            high=close + 0.1,
            low=close - 0.1,
            close=close,
            volume=volume,
        )

    def _fill(self, engine, bar, direction, offset, volume=100.0, price=10.0):
        order = OrderData(
            symbol=SYMBOL,
            order_id="regression",
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            datetime=bar.datetime,
            order_type="market",
        )
        engine._process_single_order(order, bar)
        return engine.trades[-1].price

    def test_buy_open_price_is_above_bar(self):
        engine = self._engine()
        assert self._fill(engine, self._bar(10.0), "long", "open") == pytest.approx(10.01)

    def test_short_open_price_is_below_bar(self):
        engine = self._engine()
        assert self._fill(engine, self._bar(10.0), "short", "open") == pytest.approx(9.99)

    def test_sell_close_price_is_below_bar(self):
        engine = self._engine()
        engine.positions[SYMBOL] = PositionData(
            symbol=SYMBOL, direction="long", volume=100.0, price=10.0
        )
        assert self._fill(engine, self._bar(10.0), "short", "close") == pytest.approx(9.99)

    def test_cover_close_price_is_above_bar(self):
        engine = self._engine()
        engine.positions[SYMBOL] = PositionData(
            symbol=SYMBOL, direction="short", volume=100.0, price=10.0
        )
        assert self._fill(engine, self._bar(10.0), "long", "close") == pytest.approx(10.01)


class TestCommissionFloorRegression:
    def test_small_notional_honors_min_commission(self):
        engine = BacktestEngine()
        engine.set_capital(100_000.0)
        engine.set_commission(0.00001)
        engine.set_slippage(0.0, "none")
        engine.set_fill_mode("current_close")
        engine.set_market_rules(False)
        engine.min_commission = 0.1
        engine.transfer_fee_rate = 0.0
        engine.max_volume_pct = 0.0

        bar = BarData(
            symbol=SYMBOL,
            datetime=datetime(2026, 8, 3, 10, 0),
            open=10.0,
            high=10.1,
            low=9.9,
            close=10.0,
            volume=1000.0,
        )
        order = OrderData(
            symbol=SYMBOL,
            order_id="commission",
            direction="long",
            offset="open",
            price=10.0,
            volume=100.0,
            datetime=bar.datetime,
            order_type="market",
        )
        engine._process_single_order(order, bar)

        assert len(engine.trades) == 1
        # cost=1000，费率佣金=0.01，被最低佣金 0.1 抬升。
        assert engine.capital == pytest.approx(100_000.0 - 1000.0 - 0.1)


class TestVolumeZeroNoFillRegression:
    def test_suspended_bar_volume_zero_does_not_fill(self):
        engine = BacktestEngine()
        engine.set_capital(100_000.0)
        engine.set_commission(0.0)
        engine.set_slippage(0.0, "none")
        engine.set_fill_mode("current_close")
        engine.set_market_rules(False)
        engine.min_commission = 0.0
        engine.transfer_fee_rate = 0.0
        engine.max_volume_pct = 0.0

        bar = BarData(
            symbol=SYMBOL,
            datetime=datetime(2026, 8, 3, 10, 0),
            open=10.0,
            high=10.1,
            low=9.9,
            close=10.0,
            volume=0.0,
        )
        order = OrderData(
            symbol=SYMBOL,
            order_id="volume_zero",
            direction="long",
            offset="open",
            price=10.0,
            volume=100.0,
            datetime=bar.datetime,
            order_type="market",
        )
        engine._process_single_order(order, bar)

        assert engine.trades == []
        assert order.filled == pytest.approx(0.0)
        assert order.status == "submitted"


class TestRoundLotRegression:
    def test_non_board_lot_order_is_floored_to_lot_size(self):
        engine = BacktestEngine()
        order_id = engine.send_order(
            "tester", "long", "open", 10.0, 250.0, order_type="market", symbol=SYMBOL
        )
        assert order_id
        assert engine.orders[-1].volume == pytest.approx(200.0)

        engine.send_order(
            "tester", "long", "open", 10.0, 150.0, order_type="market", symbol=SYMBOL
        )
        assert engine.orders[-1].volume == pytest.approx(100.0)

        with pytest.warns(UserWarning, match="min size"):
            rejected = engine.send_order(
                "tester", "long", "open", 10.0, 99.0, order_type="market", symbol=SYMBOL
            )
        assert rejected == ""
