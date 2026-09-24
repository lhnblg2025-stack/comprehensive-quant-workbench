"""Bridge target weights into the existing order-level BacktestEngine."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Mapping

import pandas as pd

from .backtest_engine import StrategyTemplate


class SignalPortfolioAdapter(StrategyTemplate):
    """Long-only target-weight strategy with next-bar order submission.

    ``targets`` maps a signal timestamp to ``{symbol: target_weight}``. Orders are
    submitted from ``on_bar`` and therefore follow the engine's configured fill
    mode (``next_open`` by default). The adapter intentionally does not fake fills;
    rejected and blocked orders remain visible in the engine result.
    """

    def __init__(self, targets: Mapping[object, Mapping[str, float]],
                 rebalance_tolerance: float = 0.002) -> None:
        super().__init__()
        self.targets = {
            pd.Timestamp(key).to_pydatetime(): {str(symbol): float(weight) for symbol, weight in value.items()}
            for key, value in targets.items()
        }
        self.rebalance_tolerance = float(rebalance_tolerance)
        self.submitted_dates: set[datetime] = set()
        self.target_history: list[dict] = []
        self.actual_weight_history: list[dict] = []
        self.latest_prices: dict[str, float] = {}
        self.last_symbol_by_date: dict[datetime, str] = {}
        self.pending_buy_target: dict[str, float] | None = None
        self.pending_buy_signal_dt: datetime | None = None

    def on_start(self) -> None:
        if self.bg is None:
            return
        symbols_by_date: dict[datetime, list[str]] = defaultdict(list)
        for symbol, bars in self.bg.bars.items():
            for item in bars:
                symbols_by_date[item.datetime].append(symbol)
        self.last_symbol_by_date = {date: max(symbols) for date, symbols in symbols_by_date.items()}

    def _portfolio_values(self, fallback_price: float) -> tuple[dict[str, float], float]:
        symbols = set(self.bg.positions) if self.bg is not None else set()
        current = {}
        for symbol in symbols:
            position = self.bg.positions.get(symbol)
            price = self.latest_prices.get(symbol, fallback_price)
            current[symbol] = float(position.volume * price) if position else 0.0
        return current, float(self.bg.capital if self.bg is not None else 0.0) + sum(current.values())

    def _submit_pending_buys(self, fallback_price: float) -> None:
        if self.bg is None or not self.pending_buy_target:
            return
        current, total_value = self._portfolio_values(fallback_price)
        for symbol in sorted(self.pending_buy_target):
            desired = total_value * max(0.0, self.pending_buy_target[symbol])
            held = current.get(symbol, 0.0)
            if desired - held <= total_value * self.rebalance_tolerance:
                continue
            price = self.latest_prices.get(symbol, fallback_price)
            volume = (desired - held) / price
            if volume >= self.bg.size:
                self.buy(price, volume, symbol=symbol)
        self.pending_buy_target = None
        self.pending_buy_signal_dt = None

    def on_bar(self, bar) -> None:
        dt = bar.datetime
        self.latest_prices[bar.symbol] = float(bar.close)
        # The engine calls on_bar once per symbol. Act only after every symbol for
        # the timestamp refreshed its latest price.
        if bar.symbol != self.last_symbol_by_date.get(dt, bar.symbol):
            return
        self._submit_pending_buys(float(bar.close))
        signal_dt = dt if dt in self.targets else None
        if signal_dt is None or signal_dt in self.submitted_dates or self.bg is None:
            return
        self.submitted_dates.add(signal_dt)
        target = self.targets[signal_dt]
        self.target_history.append({"date": dt.isoformat(), "targets": target.copy()})
        current_values, total_value = self._portfolio_values(float(bar.close))
        symbols = set(self.bg.positions) | set(target)
        desired_values = {symbol: total_value * max(0.0, weight) for symbol, weight in target.items()}
        for symbol in sorted(symbols):
            position = self.bg.positions.get(symbol)
            current = current_values.get(symbol, 0.0)
            desired = desired_values.get(symbol, 0.0)
            if position is None or current <= 0 or current - desired <= total_value * self.rebalance_tolerance:
                continue
            price = self.latest_prices.get(symbol, float(bar.close))
            volume = min(position.volume, (current - desired) / price)
            if volume >= self.bg.size:
                self.sell(price, volume, symbol=symbol)
        self.pending_buy_target = target.copy()
        self.pending_buy_signal_dt = signal_dt

    def record_actual_weights(self, dt: object, total_value: float) -> None:
        if self.bg is None or total_value <= 0:
            return
        weights = {}
        for symbol, position in self.bg.positions.items():
            weights[symbol] = float(position.volume * position.price / total_value) if position.price else 0.0
        self.actual_weight_history.append({"date": pd.Timestamp(dt).isoformat(), "weights": weights})


def run_order_level_backtest(panel: pd.DataFrame, targets: Mapping[object, Mapping[str, float]], *, capital: float = 1_000_000.0, commission_rate: float | None = None, slippage_rate: float | None = None, end_mode: str = "mark_to_market", corporate_actions: pd.DataFrame | None = None, release_id: str | None = None, candidate_ids: Mapping[object, Mapping[str, str]] | None = None):
    """Run target weights through the canonical order-level engine.

    The panel must contain ``date/code/open/high/low/close``. Optional execution
    costs are applied through the engine setters, keeping fills and rejections in
    the returned ``BacktestResult`` instead of approximating them in research code.
    """
    from .backtest_engine import BacktestEngine

    required = {"date", "code", "open", "high", "low", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"execution panel missing columns: {sorted(missing)}")
    execution_panel = panel.copy()
    if "volume" not in execution_panel:
        execution_panel["volume"] = 0.0
    execution_panel["volume"] = pd.to_numeric(execution_panel["volume"], errors="coerce").fillna(0.0)
    if "amount" not in execution_panel:
        execution_panel["amount"] = pd.to_numeric(execution_panel["close"], errors="coerce") * execution_panel["volume"]
    else:
        fallback_amount = pd.to_numeric(execution_panel["close"], errors="coerce") * execution_panel["volume"]
        execution_panel["amount"] = pd.to_numeric(execution_panel["amount"], errors="coerce").fillna(fallback_amount)
    engine = BacktestEngine()
    engine.set_capital(capital)
    engine.set_end_mode(end_mode)
    if corporate_actions is not None and not corporate_actions.empty:
        actions = corporate_actions.rename(columns={"code": "symbol", "ex_date": "date"})
        engine.add_corporate_actions(actions)
    if commission_rate is not None:
        engine.set_commission(commission_rate)
    if slippage_rate is not None:
        engine.set_slippage(slippage_rate, mode="percent")
    for symbol, frame in execution_panel.groupby("code", sort=True):
        engine.add_data(str(symbol), frame.sort_values("date"), datetime_col="date")
    class _ConfiguredSignalAdapter(SignalPortfolioAdapter):
        def __init__(self):
            super().__init__(targets)

    engine.add_strategy(_ConfiguredSignalAdapter, name="signal_portfolio")
    result = engine.run()
    # Preserve the immutable input release alongside the canonical engine result.
    result.release_id = release_id
    # Attach immutable research identity to emitted order/trade objects for attribution.
    candidate_ids = candidate_ids or {}
    for order in result.orders:
        date_map = candidate_ids.get(pd.Timestamp(order.datetime), {}) if order.datetime is not None else {}
        setattr(order, "candidate_id", date_map.get(str(order.symbol)))
    attribution = []
    for trade in result.trades:
        date_map = candidate_ids.get(pd.Timestamp(trade.datetime), {}) if trade.datetime is not None else {}
        candidate_id = date_map.get(str(trade.symbol))
        setattr(trade, "candidate_id", candidate_id)
        attribution.append({"trade_id": trade.trade_id, "order_id": trade.order_id, "symbol": trade.symbol,
                            "datetime": trade.datetime.isoformat(), "candidate_id": candidate_id})
    result.candidate_attribution = attribution
    return result


def targets_from_frame(frame: pd.DataFrame, date_col: str = "date", symbol_col: str = "code", weight_col: str = "target_weight") -> dict[datetime, dict[str, float]]:
    required = {date_col, symbol_col, weight_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"target frame missing columns: {sorted(missing)}")
    result: dict[datetime, dict[str, float]] = defaultdict(dict)
    for row in frame[[date_col, symbol_col, weight_col]].itertuples(index=False):
        result[pd.Timestamp(row[0]).to_pydatetime()][str(row[1])] = float(row[2])
    return dict(result)
