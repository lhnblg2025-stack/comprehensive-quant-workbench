"""Execution engine v2: the audited v1 engine plus risk overlays.

``simulate_portfolio`` here is a superset of
``quant_system.strategy_matrix_backtest.simulate_portfolio``.  With the overlay
arguments left at their defaults the code path is identical, which the study
verifier proves by re-running matched cells through both engines and comparing
the equity curves, trade ledgers and metrics bit-for-bit.

Added controls (all optional, all information-safe):

``rebalance_dates``
    Explicit rebalance calendar.  Used for rebalance-date placebo tests
    (first / middle / last trading day of the month, Monday / Wednesday /
    Friday weekly, ...).
``exposure``
    A per-decision-date multiplier in ``[0, 1]`` applied to every target
    weight.  Unused capital stays in cash, so this is a real de-risking lever
    rather than a cosmetic rescaling.  The caller builds it from index
    volatility and trend state (see ``public_strategies.build_exposure_series``).
``drawdown_deleverage``
    ``(entry, multiplier, recovery)``: once the *simulated* equity drawdown
    breaches ``entry`` every later target is scaled by ``multiplier`` until the
    drawdown recovers past ``recovery``.
``no_trade_band``
    Skip a rebalance leg whose weight drift is smaller than the band, which
    cuts turnover and therefore the fee drag.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from quant_system.strategy_matrix_backtest import (
    FREQUENCY_HORIZON,
    MatrixConfig,
    _cap_weights,
    _fee,
    _floor_lot,
    _pivot,
    _rebalance_date_set,
    annualized_metrics,
)


def simulate_portfolio(
    frame: pd.DataFrame,
    config: MatrixConfig,
    frequency: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    rebalance_dates: Any = None,
    exposure: Any = None,
    drawdown_deleverage: tuple[float, float, float] | None = None,
    no_trade_band: float = 0.0,
) -> dict[str, Any]:
    """Run one strategy/frequency portfolio with full trade audit fields."""
    required = {
        "date", "code", "expected_ret", "raw_open", "hfq_open", "hfq_close",
        "amount_ma_20", "tradable", "limit_up_locked", "limit_down_locked",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("simulation_missing_columns:" + ",".join(missing))
    if frequency not in FREQUENCY_HORIZON:
        raise ValueError("unknown_frequency:" + frequency)

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for column in ("expected_ret", "raw_open", "hfq_open", "hfq_close", "amount_ma_20"):
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    if start is not None:
        data = data[data["date"] >= pd.Timestamp(start)]
    if end is not None:
        data = data[data["date"] <= pd.Timestamp(end)]
    data = data.dropna(subset=["date", "code"]).sort_values(["date", "code"])
    if data.empty:
        raise ValueError("empty_simulation_window")
    for flag in ("tradable", "limit_up_locked", "limit_down_locked"):
        data[flag] = pd.to_numeric(data[flag], errors="coerce").fillna(0.0).astype(float)
    has_select_score = "select_score" in data.columns
    if has_select_score:
        data["select_score"] = pd.to_numeric(data["select_score"], errors="coerce").astype(float)

    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    codes = sorted(data["code"].unique())
    if len(dates) < 3:
        raise ValueError("insufficient_simulation_dates")
    exposure_values: np.ndarray | None = None
    if exposure is not None:
        series = exposure if isinstance(exposure, pd.Series) else pd.Series(np.asarray(exposure, dtype=float))
        if isinstance(exposure, pd.Series) and isinstance(exposure.index, pd.DatetimeIndex):
            series = series.reindex(pd.DatetimeIndex(dates)).ffill()
            exposure_values = series.to_numpy(dtype=float)
        else:
            values = np.asarray(series.to_numpy(dtype=float), dtype=float)
            if len(values) != len(dates):
                raise ValueError("exposure_length_mismatch")
            exposure_values = values
        if not np.all(np.isfinite(exposure_values)):
            exposure_values = np.nan_to_num(exposure_values, nan=0.0)
        exposure_values = np.clip(exposure_values, 0.0, 1.0)
    code_position = {code: position for position, code in enumerate(codes)}

    expected = _pivot(data, "expected_ret", dates, codes)
    raw_open = _pivot(data, "raw_open", dates, codes)
    hfq_open = _pivot(data, "hfq_open", dates, codes)
    hfq_close = _pivot(data, "hfq_close", dates, codes, ffill=True)
    adv = _pivot(data, "amount_ma_20", dates, codes, fill=0.0)
    tradable = _pivot(data, "tradable", dates, codes, fill=False, dtype=bool)
    limit_up = _pivot(data, "limit_up_locked", dates, codes, fill=False, dtype=bool)
    limit_down = _pivot(data, "limit_down_locked", dates, codes, fill=False, dtype=bool)
    select_score = (
        _pivot(data, "select_score", dates, codes)
        if has_select_score
        else None
    )

    explicit_calendar = rebalance_dates is not None
    if rebalance_dates is None:
        active_rebalance_dates = _rebalance_date_set(dates, frequency)
    else:
        active_rebalance_dates = set(pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))))
    rebalance_dates = active_rebalance_dates
    slip_buy = 1.0 + config.slippage_bps / 10000.0
    slip_sell = 1.0 - config.slippage_bps / 10000.0
    cash = float(config.initial_capital)
    positions: dict[str, dict[str, float]] = {}
    pending: dict[str, float] | None = None
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    stale_codes_rows: list[set] = []
    blocked: dict[str, int] = {}
    blocked_sell_dates: set = set()
    blocked_sell_codes: set = set()
    active_target: set = set()
    cursor: dict[str, Any] = {"date": None}
    equity_peak = float(config.initial_capital)
    deleverage_active = False
    deleverage_scale = 1.0
    deleverage_events = 0
    band_skipped = 0
    last_total_value = float(config.initial_capital)

    def note(reason: str, code: str | None = None) -> None:
        blocked[reason] = blocked.get(reason, 0) + 1
        day = cursor["date"]
        if code is not None and reason.startswith("sell"):
            # A holding that is no longer in the target book can only survive
            # if at least one exit attempt for it was blocked.
            blocked_sell_dates.add(day)
            blocked_sell_codes.add(code)

    for i, date in enumerate(dates):
        cursor["date"] = date
        if pending is not None:
            open_value = 0.0
            for code, position in positions.items():
                price = hfq_open[i, code_position[code]]
                if not np.isfinite(price):
                    price = hfq_close[i, code_position[code]]
                if np.isfinite(price):
                    open_value += position["base_adj"] * price
            investable = max(0.0, (cash + open_value) * (1.0 - config.cash_buffer))
            targets = {code: investable * weight for code, weight in pending.items()}
            target_weights = dict(pending)
            pending = None

            # Sell legs first so sale proceeds can fund the buy legs.
            for code in list(positions):
                j = code_position[code]
                current = int(positions[code]["shares"])
                price_raw = raw_open[i, j]
                target_value = targets.get(code, 0.0)
                if not np.isfinite(price_raw) or price_raw <= 0:
                    note("sell_no_quote", code)
                    continue
                desired = _floor_lot(target_value, price_raw, config.lot_size)
                if desired >= current:
                    continue
                if no_trade_band > 0 and last_total_value > 0:
                    drift = (current * price_raw - target_value) / last_total_value
                    if drift < no_trade_band:
                        band_skipped += 1
                        continue
                if not bool(tradable[i, j]):
                    note("sell_not_tradable", code)
                    continue
                if bool(limit_down[i, j]):
                    note("sell_limit_down", code)
                    continue
                available_adv = float(adv[i, j])
                if not np.isfinite(available_adv) or available_adv <= 0:
                    note("sell_no_adv", code)
                    continue
                sell_shares = current - desired
                cap = int(available_adv * config.max_adv_participation / price_raw // config.lot_size) * config.lot_size
                sell_shares = int(min(sell_shares, max(cap, 0)))
                if sell_shares <= 0:
                    note("sell_adv_cap", code)
                    continue
                price_hfq = hfq_open[i, j]
                if not np.isfinite(price_hfq):
                    price_hfq = hfq_close[i, j]
                if not np.isfinite(price_hfq) or price_hfq <= 0:
                    note("sell_no_hfq", code)
                    continue
                fraction = sell_shares / current
                traded_value = positions[code]["base_adj"] * fraction * price_hfq
                raw_notional = sell_shares * price_raw
                fee = _fee(raw_notional, "sell", config)
                cash += traded_value * slip_sell - fee
                positions[code]["base_adj"] *= (1.0 - fraction)
                positions[code]["shares"] = current - sell_shares
                if positions[code]["shares"] <= 0 or positions[code]["base_adj"] <= 1e-12:
                    del positions[code]
                trade_rows.append({
                    "date": date, "code": code, "side": "sell", "shares": int(sell_shares),
                    "raw_price": float(price_raw), "hfq_price": float(price_hfq),
                    "raw_notional": float(raw_notional), "traded_value": float(traded_value),
                    "fee": float(fee), "slippage_cost": float(traded_value * (config.slippage_bps / 10000.0)),
                    "target_value": float(target_value), "target_weight": float(target_weights.get(code, 0.0)),
                    "blocked": False, "block_reason": "",
                })

            # Buy legs, largest target first, sized against live cash.
            for code, target_value in sorted(targets.items(), key=lambda item: item[1], reverse=True):
                if target_value <= 0:
                    continue
                j = code_position[code]
                price_raw = raw_open[i, j]
                if not np.isfinite(price_raw) or price_raw <= 0:
                    note("buy_no_quote")
                    continue
                if not bool(tradable[i, j]):
                    note("buy_not_tradable")
                    continue
                if bool(limit_up[i, j]):
                    note("buy_limit_up")
                    continue
                price_hfq = hfq_open[i, j]
                if not np.isfinite(price_hfq) or price_hfq <= 0:
                    note("buy_no_hfq")
                    continue
                price = float(price_raw) * slip_buy
                current = int(positions.get(code, {}).get("shares", 0))
                desired = _floor_lot(target_value, price, config.lot_size)
                delta = desired - current
                if delta <= 0:
                    continue
                if no_trade_band > 0 and last_total_value > 0 and current > 0:
                    drift = (target_value - current * price_raw) / last_total_value
                    if drift < no_trade_band:
                        band_skipped += 1
                        continue
                available_adv = float(adv[i, j])
                if not np.isfinite(available_adv) or available_adv <= 0:
                    note("buy_no_adv")
                    continue
                cap = int(available_adv * config.max_adv_participation / price // config.lot_size) * config.lot_size
                if cap <= 0:
                    note("buy_adv_cap")
                    continue
                delta = int(min(delta, cap))
                while delta > 0:
                    gross = delta * price
                    if gross + _fee(gross, "buy", config) <= cash:
                        break
                    delta -= config.lot_size
                if delta <= 0:
                    note("buy_insufficient_cash")
                    continue
                gross = delta * price
                fee = _fee(gross, "buy", config)
                cash -= gross + fee
                position = positions.setdefault(code, {"shares": 0.0, "base_adj": 0.0})
                position["shares"] += delta
                # Basis uses the unslipped raw consideration: the slipped
                # price is what cash actually pays, so the slippage shows up
                # immediately as a loss instead of inflating the position.
                position["base_adj"] += (delta * float(price_raw)) / float(price_hfq)
                trade_rows.append({
                    "date": date, "code": code, "side": "buy", "shares": int(delta),
                    "raw_price": float(price_raw), "hfq_price": float(price_hfq),
                    "raw_notional": float(delta * price_raw), "traded_value": float(gross),
                    "fee": float(fee), "slippage_cost": float(gross - delta * price_raw),
                    "target_value": float(target_value), "target_weight": float(target_weights.get(code, 0.0)),
                    "blocked": False, "block_reason": "",
                })

        position_value = 0.0
        for code, position in positions.items():
            price = hfq_close[i, code_position[code]]
            if np.isfinite(price):
                position_value += position["base_adj"] * price
        total_value = cash + position_value
        stale_codes_rows.append(set(positions) - active_target)
        equity_rows.append({
            "date": date, "cash": float(cash), "position_value": float(position_value),
            "total_value": float(total_value), "positions": int(len(positions)),
            "exposure": float(position_value / total_value) if total_value else 0.0,
        })
        last_total_value = float(total_value)
        if total_value > equity_peak:
            equity_peak = float(total_value)
        if drawdown_deleverage is not None and equity_peak > 0:
            entry_level, multiplier, recovery = drawdown_deleverage
            current_drawdown = total_value / equity_peak - 1.0
            if not deleverage_active and current_drawdown <= entry_level:
                deleverage_active = True
                deleverage_scale = float(multiplier)
                deleverage_events += 1
            elif deleverage_active and current_drawdown >= recovery:
                deleverage_active = False

        if date in rebalance_dates and i + 1 < len(dates):
            expected_row = expected[i]
            valid = np.isfinite(expected_row) & (expected_row > config.min_expected_return)
            if valid.any() and int(config.max_names) > 0:
                candidates = np.flatnonzero(valid)
                if len(candidates) > int(config.max_names):
                    expected_candidates = expected_row[candidates]
                    if select_score is not None:
                        score_candidates = np.nan_to_num(
                            select_score[i, candidates], nan=-np.inf, posinf=-np.inf
                        )
                        order = np.lexsort((-score_candidates, -expected_candidates))
                    else:
                        order = np.argsort(-expected_candidates, kind="stable")
                    candidates = candidates[order[: int(config.max_names)]]
                keep = np.zeros(len(expected_row), dtype=bool)
                keep[candidates] = True
                valid = keep
            if valid.any():
                excess = expected_row[valid] - config.min_expected_return
                weights = _cap_weights(excess, np.ones_like(excess), config.max_weight)
                selected = np.flatnonzero(valid)
                pending = {codes[j]: float(weights[k]) for k, j in enumerate(selected) if weights[k] > 0}
            else:
                pending = {}
            scale = 1.0
            if exposure_values is not None:
                scale *= float(exposure_values[i])
            if deleverage_active:
                scale *= float(deleverage_scale)
            if scale < 1.0:
                pending = {code: weight * scale for code, weight in pending.items()}
            elif scale <= 0.0:
                pending = {}
            active_target = set(pending)

    equity = pd.DataFrame(equity_rows)
    equity["date"] = pd.to_datetime(equity["date"])
    equity["return"] = equity["total_value"].pct_change().fillna(0.0)
    excess_rows = equity[equity["positions"] > int(config.max_names)]
    excess_days = int(len(excess_rows))
    excess_days_with_blocked_sell = int(sum(1 for day in excess_rows["date"] if day in blocked_sell_dates))
    stale_position_days = int(sum(1 for stale in stale_codes_rows if stale))
    unexplained_stale_positions = int(sum(len(stale - blocked_sell_codes) for stale in stale_codes_rows))
    trades = pd.DataFrame(trade_rows)
    metrics = annualized_metrics(equity["return"])
    total_fees = float(trades["fee"].sum()) if not trades.empty else 0.0
    traded_value_total = float(trades["traded_value"].sum()) if not trades.empty else 0.0
    average_equity = float(equity["total_value"].mean())
    blocked_total = int(sum(blocked.values()))
    non_lot = int((trades["shares"] % config.lot_size != 0).sum()) if not trades.empty else 0
    report = {
        "frequency": frequency,
        "horizon_sessions": FREQUENCY_HORIZON[frequency],
        "config": asdict(config),
        "metrics": metrics,
        "sessions": int(len(equity)),
        "rebalances": int(len([d for d in rebalance_dates if d <= dates[-2]])),
        "trades": int(len(trades)),
        "blocked_orders": blocked_total,
        "blocked_reasons": blocked,
        "total_fees": total_fees,
        "fee_drag_on_initial_capital": total_fees / float(config.initial_capital),
        "total_traded_value": traded_value_total,
        "turnover_multiple_on_initial_capital": traded_value_total / float(config.initial_capital),
        "turnover_multiple_on_average_equity": traded_value_total / average_equity if average_equity else None,
        "excess_position_days": excess_days,
        "excess_position_days_with_blocked_sell": excess_days_with_blocked_sell,
        "stale_position_days": stale_position_days,
        "unexplained_stale_positions": unexplained_stale_positions,
        "excess_positions_explained_by_blocked_sells": bool(unexplained_stale_positions == 0),
        "average_positions": float(equity["positions"].mean()),
        "max_positions": int(equity["positions"].max()),
        "final_positions": int(equity["positions"].iloc[-1]),
        "average_exposure": float(equity["exposure"].mean()),
        "average_cash_ratio": float((equity["cash"] / equity["total_value"]).mean()),
        "final_cash_ratio": float(equity["cash"].iloc[-1] / equity["total_value"].iloc[-1]) if equity["total_value"].iloc[-1] else None,
        "final_equity": float(equity["total_value"].iloc[-1]),
        "overlay": {
            "exposure_series": bool(exposure_values is not None),
            "drawdown_deleverage": list(drawdown_deleverage) if drawdown_deleverage else None,
            "drawdown_deleverage_events": int(deleverage_events),
            "no_trade_band": float(no_trade_band),
            "band_skipped_legs": int(band_skipped),
            "explicit_rebalance_calendar": bool(explicit_calendar),
        },
        "reconciliation": {
            "negative_cash_rows": int((equity["cash"] < -1e-6).sum()),
            "non_lot_trade_rows": non_lot,
            "trades_with_zero_shares": int((trades["shares"] <= 0).sum()) if not trades.empty else 0,
            "total_value_equals_cash_plus_positions": bool(
                np.allclose(equity["total_value"], equity["cash"] + equity["position_value"], atol=1e-6)
            ),
            "fees_nonnegative": bool((trades["fee"] >= 0).all()) if not trades.empty else True,
            "cash_plus_final_positions": float(
                equity["cash"].iloc[-1] + equity["position_value"].iloc[-1]
            ),
        },
    }
    return {"report": report, "equity": equity, "trades": trades}
