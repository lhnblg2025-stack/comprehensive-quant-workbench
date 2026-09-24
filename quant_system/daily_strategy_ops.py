"""Daily strategy operations: advice, sizing, fill reconciliation and health gates."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SuggestedOrder:
    symbol: str
    side: str
    shares: int
    reference_price: float
    target_weight: float
    reason: str
    phase: int


def build_order_advice(current_shares: dict[str, int], target_weights: dict[str, float], raw_prices: dict[str, float], cash: float, *, lot_size: int = 100, cash_buffer: float = 0.05, tolerance: float = 0.002) -> list[SuggestedOrder]:
    position_value = sum(current_shares.get(s, 0) * raw_prices.get(s, 0.0) for s in current_shares)
    equity = cash + position_value
    investable = equity * (1.0 - cash_buffer)
    orders = []
    symbols = set(current_shares) | set(target_weights)
    for symbol in sorted(symbols):
        price = float(raw_prices.get(symbol, 0.0))
        if price <= 0:
            continue
        current = int(current_shares.get(symbol, 0)); target_weight = max(0.0, float(target_weights.get(symbol, 0.0)))
        target = math.floor((investable * target_weight) / price / lot_size) * lot_size
        delta = target - current
        if abs(delta * price) <= equity * tolerance:
            continue
        if delta < 0:
            orders.append(SuggestedOrder(symbol, "sell", abs(delta), price, target_weight, "reduce_to_target", 1))
        elif delta >= lot_size:
            orders.append(SuggestedOrder(symbol, "buy", delta, price, target_weight, "increase_to_target", 2))
    return sorted(orders, key=lambda order: (order.phase, order.symbol))


def resize_buy_orders_after_sells(orders: list[SuggestedOrder], available_cash: float, latest_raw_prices: dict[str, float], *, lot_size: int = 100) -> list[SuggestedOrder]:
    resized = []
    cash = float(available_cash)
    for order in orders:
        if order.side != "buy":
            resized.append(order); continue
        price = float(latest_raw_prices.get(order.symbol, order.reference_price))
        affordable = math.floor(cash / price / lot_size) * lot_size
        shares = min(order.shares, affordable)
        if shares >= lot_size:
            resized.append(SuggestedOrder(order.symbol, order.side, shares, price, order.target_weight, order.reason, order.phase))
            cash -= shares * price
    return resized


def reconcile_fills(orders: list[SuggestedOrder], fills: list[dict[str, Any]], raw_close: dict[str, float], initial_cash: float) -> dict[str, Any]:
    planned = {(o.symbol, o.side): o.shares for o in orders}; filled = {}; cash = float(initial_cash); shares: dict[str, int] = {}
    for fill in fills:
        symbol = str(fill["symbol"]); side = str(fill["side"]); qty = int(fill["shares"]); price = float(fill["price"]); fee = float(fill.get("fee", 0.0))
        filled[(symbol, side)] = filled.get((symbol, side), 0) + qty
        shares[symbol] = shares.get(symbol, 0) + (qty if side == "buy" else -qty)
        cash += (-qty * price - fee) if side == "buy" else (qty * price - fee)
    planned_total = sum(planned.values()); filled_total = sum(filled.values())
    value = sum(max(0, qty) * raw_close.get(symbol, 0.0) for symbol, qty in shares.items())
    return {"planned_shares": planned_total, "filled_shares": filled_total, "fill_rate": filled_total / planned_total if planned_total else 1.0, "cash": cash, "position_value": value, "total_equity": cash + value, "share_positions": shares}


def strategy_health(*, rolling_excess_return: float, max_drawdown: float, fill_rate: float, tracking_error: float, data_fresh: bool, rolling_ic: float | None = None) -> dict[str, Any]:
    flags = []
    if not data_fresh: flags.append("stale_data")
    if rolling_excess_return < 0: flags.append("negative_excess")
    if max_drawdown < -0.15: flags.append("drawdown_breach")
    if fill_rate < 0.90: flags.append("low_fill_rate")
    if tracking_error > 0.05: flags.append("weight_tracking_error")
    if rolling_ic is not None and rolling_ic < 0: flags.append("negative_ic")
    status = "halt" if any(x in flags for x in ("stale_data", "drawdown_breach")) else "degraded" if flags else "healthy"
    return {"status": status, "flags": flags}


def promotion_state(metrics: dict[str, Any]) -> str:
    if metrics.get("oos_windows", 0) < 8 or metrics.get("positive_window_rate", 0) < 0.60:
        return "RESEARCH"
    if metrics.get("relative_return", -1) <= 0 or not metrics.get("execution_consistent", False) or not metrics.get("corporate_actions_complete", False):
        return "VALIDATED"
    if metrics.get("paper_months", 0) < 3 or metrics.get("health_status") != "healthy":
        return "PAPER"
    return "DEPLOYABLE"


def write_daily_packet(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output
