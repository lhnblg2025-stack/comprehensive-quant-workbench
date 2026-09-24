"""Canonical Backtrader execution engine for cross-sectional A-share research.

This module is the single source of truth for order-level execution semantics.
Signals are formed at session ``t`` close and executed at ``t+1`` open, in
100-share lots, with buy commission + transfer fee, sell commission + transfer
fee + stamp duty, slippage, T+1 (no same-day sell of shares bought today), and
blocked buy at limit-up / suspension and blocked sell at limit-down / suspension.

The engine reindexes every symbol onto a common union calendar and forward-fills
missing bars, so backtrader's multi-feed clock is deterministic. A bar that was
absent on the calendar is flagged ``suspended`` and its orders are rejected.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest_protocol import UnifiedBacktestResult
from .build_trade_state import build_trade_state
from .metrics_calculator import MetricsCalculator

LOT_SIZE = 100


class AShareCommission:
    """Backtrader commission info: commission + transfer on both sides, stamp duty on sells."""

    @staticmethod
    def create(commission_bps: float = .85, stamp_duty_bps: float = 5.0, transfer_fee_bps: float = .1, min_commission: float = 5.0):
        import backtrader as bt

        class _Commission(bt.CommInfoBase):
            params = dict(commission=commission_bps / 10000.0, stamp=stamp_duty_bps / 10000.0,
                          transfer=transfer_fee_bps / 10000.0, min_comm=min_commission,
                          stocklike=True, commtype=bt.CommInfoBase.COMM_PERC)

            def _getcommission(self, size, price, pseudoexec):
                notional = abs(size) * price
                commission = max(self.p.min_comm, notional * self.p.commission)
                transfer = notional * self.p.transfer
                stamp = notional * self.p.stamp if size < 0 else 0.0
                return commission + transfer + stamp

        return _Commission()


def _aligned_feeds(panel: pd.DataFrame, calendar: pd.DatetimeIndex, trade_state: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Reindex each symbol onto the union calendar with forward-filled OHLCV.

    Returns a dict {code: DataFrame indexed by calendar} with extra columns
    ``suspended``, ``limit_up_locked``, ``limit_down_locked``.
    """
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for col in ("raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount"):
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "code"]).sort_values(["code", "date"])

    state = None
    if trade_state is not None and len(trade_state):
        s = trade_state.copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s["code"] = s["code"].astype(str).str.zfill(6)
        s = s.drop_duplicates(["code", "date"])
        state = {(row.code, row.date): {"suspended": row.suspended, "limit_up_locked": row.limit_up_locked, "limit_down_locked": row.limit_down_locked} for row in s.itertuples(index=False)}

    feeds: dict[str, pd.DataFrame] = {}
    for code, group in data.groupby("code", sort=True):
        g = group.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]
        g = g.reindex(calendar)
        has_bar = g["raw_close"].notna()
        g["raw_open"] = g["raw_open"].ffill()
        g["raw_high"] = g["raw_high"].ffill()
        g["raw_low"] = g["raw_low"].ffill()
        g["raw_close"] = g["raw_close"].ffill()
        g["volume"] = g["volume"].ffill().fillna(0.0)
        g["amount"] = g["amount"].ffill().fillna(0.0)
        if "adv_amount_20" in g:
            g["adv_amount_20"] = g["adv_amount_20"].ffill().fillna(0.0)
        else:
            g["adv_amount_20"] = g["amount"]
        g["suspended"] = ~has_bar
        if state is not None:
            g["limit_up_locked"] = [bool(state.get((code, d), {}).get("limit_up_locked", False)) for d in g.index]
            g["limit_down_locked"] = [bool(state.get((code, d), {}).get("limit_down_locked", False)) for d in g.index]
        else:
            g["limit_up_locked"] = False
            g["limit_down_locked"] = False
        g = g.dropna(subset=["raw_open", "raw_close"])
        if len(g):
            feeds[code] = g
    return feeds


def run_backtrader(panel: pd.DataFrame, targets: Mapping[pd.Timestamp | object, Mapping[str, float]], *,
                   capital: float = 1_000_000.0, commission_bps: float = .85, stamp_duty_bps: float = 5.0,
                   transfer_fee_bps: float = .1, slippage_bps: float = 10.0, min_commission: float = 5.0,
                   max_adv_participation: float = 0.10, max_position_weight: float = 1.0,
                   max_gross_exposure: float = 1.0, max_names: int = 10000, lot_size: int = LOT_SIZE,
                   trade_state: pd.DataFrame | None = None) -> tuple[UnifiedBacktestResult, pd.DataFrame, pd.DataFrame]:
    """Run a target-weight long-only portfolio in Backtrader.

    ``targets`` maps signal date -> {code: weight}. The target sealed at signal
    date ``t`` close is executed at ``t+1`` open.
    """
    import backtrader as bt

    required = {"date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"backtrader_panel_missing:{','.join(missing)}")

    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"], errors="coerce").dropna().unique()))
    if len(calendar) < 3:
        raise ValueError("insufficient_calendar_for_backtrader")

    feeds = _aligned_feeds(panel, calendar, trade_state)
    if not feeds:
        raise ValueError("no_symbol_feeds_for_backtrader")

    if capital <= 0 or not math.isfinite(float(capital)):
        raise ValueError("capital_must_be_finite_positive")
    if not 0 < float(max_position_weight) <= 1 or not 0 <= float(max_gross_exposure) <= 1:
        raise ValueError("invalid_risk_exposure_limits")
    if int(max_names) < 1 or int(lot_size) < 1:
        raise ValueError("invalid_risk_count_or_lot_size")
    normalized = {}
    for day, values in targets.items():
        clean = []
        for code, raw_weight in values.items():
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid_target_weight:{code}")
            if weight > 0:
                clean.append((str(code).zfill(6), min(weight, float(max_position_weight))))
        clean.sort(key=lambda item: (-item[1], item[0]))
        clean = clean[:int(max_names)]
        gross = sum(weight for _, weight in clean)
        scale = min(1.0, float(max_gross_exposure) / gross) if gross > 0 else 1.0
        normalized[pd.Timestamp(day).normalize()] = {code: weight * scale for code, weight in clean}

    class _Feed(bt.feeds.PandasData):
        params = (("datetime", None), ("open", "raw_open"), ("high", "raw_high"), ("low", "raw_low"),
                  ("close", "raw_close"), ("volume", "volume"), ("openinterest", -1))

    class _Strategy(bt.Strategy):
        params = dict(targets=normalized, lot_size=int(lot_size), max_participation=max_adv_participation,
                      slippage=slippage_bps, commission_bps=commission_bps, min_commission=min_commission,
                      stamp_duty_bps=stamp_duty_bps, transfer_fee_bps=transfer_fee_bps)

        def __init__(self):
            self.pending = None
            self.bought_today: set[str] = set()
            self.equity: list[dict] = []
            # One row represents one order intent. Statuses are mutually exclusive:
            # filled, blocked, rejected, canceled, or skipped.
            self.order_events: list[dict] = []
            self.broker_rejections: list[dict] = []
            self.total_commission = 0.0

        def _event(self, *, day, code: str, side: str, shares: int, status: str,
                   reason: str | None = None, price: float | None = None,
                   notional: float = 0.0) -> None:
            self.order_events.append({"date": pd.Timestamp(day), "code": code, "side": side,
                                      "shares": int(shares), "price": price, "notional": float(notional),
                                      "status": status, "reason": reason, "blocked": status == "blocked"})

        def _lot(self, shares: int) -> int:
            return int(shares // self.p.lot_size * self.p.lot_size)

        def _buy_fee(self, notional: float) -> float:
            return max(self.p.min_commission, notional * self.p.commission_bps / 10000.0) + notional * self.p.transfer_fee_bps / 10000.0

        def _sell_fee(self, notional: float) -> float:
            return max(self.p.min_commission, notional * self.p.commission_bps / 10000.0) + notional * self.p.transfer_fee_bps / 10000.0 + notional * self.p.stamp_duty_bps / 10000.0

        def _submit(self, day):
            if self.pending is None:
                return
            target = self.pending
            total = float(self.broker.getvalue())
            cash = float(self.broker.getcash())
            # 1. sells first; project their proceeds so the buy phase sees freed cash.
            #    (Orders are submitted sell-then-buy and Backtrader settles them in
            #    submission order, so this projection matches the broker.)
            for data in self.datas:
                code = data._name
                row = feeds[code].loc[day] if day in feeds[code].index else None
                if row is None:
                    continue
                current = int(self.getposition(data).size)
                weight = target.get(code, 0.0)
                open_price = float(row["raw_open"]) if np.isfinite(row["raw_open"]) else 0.0
                desired = self._lot(int(total * max(weight, 0.0) / max(open_price, 1e-12)))
                if code in self.bought_today:
                    desired = max(desired, current)  # T+1: cannot reduce below shares bought today
                delta = desired - current
                if delta >= 0:
                    continue
                blocked = bool(row["suspended"] or row["limit_down_locked"])
                if blocked:
                    reason = "suspended" if bool(row["suspended"]) else "limit_down_locked"
                    self._event(day=day, code=code, side="sell", shares=-delta, status="blocked", reason=reason)
                    continue
                sell_price = open_price * (1.0 - self.p.slippage / 10000.0)
                notional = (-delta) * sell_price
                fee = self._sell_fee(notional)
                cash += notional - fee
                self.sell(data=data, size=-delta)
                self._event(day=day, code=code, side="sell", shares=-delta, status="filled", price=sell_price, notional=notional)
            # 2. buys within projected cash and ADV cap.
            for data in self.datas:
                code = data._name
                row = feeds[code].loc[day] if day in feeds[code].index else None
                if row is None:
                    continue
                current = int(self.getposition(data).size)
                weight = target.get(code, 0.0)
                open_price = float(row["raw_open"]) if np.isfinite(row["raw_open"]) else 0.0
                desired = self._lot(int(total * max(weight, 0.0) / max(open_price, 1e-12)))
                delta = desired - current
                if delta <= 0 or open_price <= 0:
                    continue
                blocked = bool(row["suspended"] or row["limit_up_locked"])
                if blocked:
                    reason = "suspended" if bool(row["suspended"]) else "limit_up_locked"
                    self._event(day=day, code=code, side="buy", shares=delta, status="blocked", reason=reason)
                    continue
                price = open_price * (1.0 + self.p.slippage / 10000.0)
                adv = float(row["adv_amount_20"]) if np.isfinite(row.get("adv_amount_20", 0.0)) else 0.0
                if adv > 0:
                    cap = self._lot(int(adv * self.p.max_participation / max(price, 1e-12)))
                    delta = min(delta, cap)
                notional = delta * price
                fee = self._buy_fee(notional)
                if notional + fee > cash + 1e-8:
                    delta = self._lot(int(max(0.0, cash - 5.0) / price))
                    notional = delta * price
                    fee = self._buy_fee(notional) if delta else 0.0
                if delta > 0:
                    self.buy(data=data, size=delta)
                    cash -= notional + fee
                    self.bought_today.add(code)
                    self._event(day=day, code=code, side="buy", shares=delta, status="filled", price=price, notional=notional)
                else:
                    self._event(day=day, code=code, side="buy", shares=0, status="rejected", reason="insufficient_cash_or_adv_lot")
            self.pending = None

        def next_open(self):
            day = self.datas[0].datetime.date(0)
            self.bought_today = set()
            self._submit(pd.Timestamp(day))

        def next(self):
            day = pd.Timestamp(self.datas[0].datetime.date(0))
            self.equity.append({"date": day, "total_value": float(self.broker.getvalue())})
            if day in self.p.targets:
                self.pending = self.p.targets[day]

        def notify_order(self, order):
            if order.status == order.Completed:
                self.total_commission += float(order.executed.comm or 0.0)
            elif order.status in (order.Canceled, order.Margin, order.Rejected):
                reason = {order.Canceled: "broker_canceled", order.Margin: "broker_margin", order.Rejected: "broker_rejected"}.get(order.status, "broker_rejected")
                self.broker_rejections.append({"code": str(order.data._name), "side": "buy" if order.isbuy() else "sell", "reason": reason})

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.broker.setcash(capital)
    cerebro.broker.addcommissioninfo(AShareCommission.create(commission_bps, stamp_duty_bps, transfer_fee_bps, min_commission))
    cerebro.broker.set_slippage_perc(slippage_bps / 10000.0)
    for code in sorted(feeds):
        cerebro.adddata(_Feed(dataname=feeds[code]), name=code)
    cerebro.addstrategy(_Strategy)
    strategy = cerebro.run(runonce=False, preload=True)[0]

    equity = pd.DataFrame(strategy.equity).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    events = pd.DataFrame(strategy.order_events) if strategy.order_events else pd.DataFrame(columns=["date", "code", "side", "shares", "price", "notional", "status", "reason", "blocked"])
    if strategy.broker_rejections:
        rejected = pd.DataFrame(strategy.broker_rejections)
        for row in rejected.itertuples(index=False):
            events = pd.concat([events, pd.DataFrame([{"date": pd.NaT, "code": row.code, "side": row.side, "shares": 0, "price": np.nan, "notional": 0.0, "status": "rejected", "reason": row.reason, "blocked": False}])], ignore_index=True)
    filled = events[events["status"] == "filled"].copy() if len(events) else events
    status_counts = events["status"].value_counts().to_dict() if len(events) else {}
    reason_counts = events.loc[events["status"].isin(["blocked", "rejected", "canceled"]), "reason"].value_counts().to_dict() if len(events) else {}
    final_value = float(cerebro.broker.getvalue())
    annual, sharpe, drawdown = _annual_metrics(equity["total_value"]) if len(equity) else (None, None, None)
    rejected_orders = int(status_counts.get("rejected", 0) + status_counts.get("canceled", 0))
    blocked_orders = int(status_counts.get("blocked", 0))
    result = UnifiedBacktestResult(
        engine="backtrader", status="complete" if not rejected_orders and not blocked_orders else "completed_with_exceptions",
        initial_capital=capital, final_value=final_value, total_return=final_value / capital - 1.0,
        annual_return=annual, sharpe=sharpe, max_drawdown=drawdown, observations=len(equity),
        orders=int(len(events)), trades=int(len(filled)), rejected_orders=rejected_orders, blocked_orders=blocked_orders,
        total_fees=strategy.total_commission, metadata={"fill_timing": "next_open", "lot_size": int(lot_size), "target_dates": len(normalized), "slippage_bps": slippage_bps, "commission_bps": commission_bps, "stamp_duty_bps": stamp_duty_bps, "transfer_fee_bps": transfer_fee_bps, "min_commission": min_commission, "max_adv_participation": max_adv_participation, "max_position_weight": max_position_weight, "max_gross_exposure": max_gross_exposure, "max_names": int(max_names), "order_status_counts": status_counts, "order_reason_counts": reason_counts, "order_count_reconciliation": {"orders": int(len(events)), "filled": int(len(filled)), "blocked": blocked_orders, "rejected": rejected_orders, "canceled": int(status_counts.get("canceled", 0)), "reconciles": int(len(events)) == int(len(filled)) + blocked_orders + rejected_orders}})
    trades = events
    return result, equity, trades


def _annual_metrics(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(values) < 2:
        return None, None, None
    returns = values.pct_change().dropna()
    years = len(returns) / 252.0
    total = values.iloc[-1] / values.iloc[0] - 1.0
    annual = (1 + total) ** (1 / years) - 1 if years > 0 and 1 + total > 0 else None
    std = returns.std(ddof=1)
    sharpe = MetricsCalculator.sharpe(returns, rf=MetricsCalculator.DEFAULT_RISK_FREE_RATE) if std > 0 else None
    drawdown = float((values / values.cummax() - 1).min())
    return annual, float(sharpe) if sharpe is not None else None, drawdown


def targets_from_signal(panel: pd.DataFrame, signal: str, *, direction: int = 1, quantile: float = 0.2,
                        rebalance_sessions: int = 5) -> dict[pd.Timestamp, dict[str, float]]:
    """Build equal-weight top-quantile targets on each rebalance date (t close)."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["signal"] = pd.to_numeric(data[signal], errors="coerce")
    dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    for day in dates[::max(1, rebalance_sessions)]:
        frame = data[data["date"].eq(day)].dropna(subset=["signal"])
        if frame.empty:
            continue
        ranked = frame.assign(_s=frame["signal"] * direction).sort_values("_s", ascending=False)
        n = max(1, int(len(ranked) * quantile))
        targets[pd.Timestamp(day)] = {str(c): 1.0 / n for c in ranked.head(n)["code"]}
    return targets


def run_signal(panel: pd.DataFrame, signal: str, *, direction: int = 1, quantile: float = 0.2,
               rebalance_sessions: int = 5, capital: float = 1_000_000.0, trade_state: pd.DataFrame | None = None, **kwargs):
    targets = targets_from_signal(panel, signal, direction=direction, quantile=quantile, rebalance_sessions=rebalance_sessions)
    return run_backtrader(panel, targets, capital=capital, trade_state=trade_state, **kwargs)
