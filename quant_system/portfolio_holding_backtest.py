"""Non-overlapping portfolio holding backtest for A-share research.

Signals are formed at session t close, orders execute at t+1 open, and each
rebalance creates one non-overlapping holding interval. The module is designed
as the common benchmark for OHLCV, PIT, event, and ML signals.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldingConfig:
    holding_sessions: int = 5
    rebalance_sessions: int = 5
    quantile: float = 0.2
    initial_capital: float = 1_000_000.0
    cash_buffer: float = 0.05
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    max_adv_participation: float = 0.10
    lot_size: int = 100
    min_commission: float = 5.0


def _fee(notional: float, side: str, cfg: HoldingConfig) -> float:
    commission = max(cfg.min_commission, notional * cfg.commission_bps / 10000.0)
    transfer = notional * cfg.transfer_fee_bps / 10000.0
    stamp = notional * cfg.stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    return commission + transfer + stamp


def _require_columns(frame: pd.DataFrame) -> None:
    required = {"date", "code", "raw_open", "raw_close", "amount", "signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"panel_missing_columns:{','.join(missing)}")


def run_portfolio_backtest(panel: pd.DataFrame, config: HoldingConfig = HoldingConfig(), *, direction: int = 1) -> dict:
    """Run a non-overlapping long-only portfolio from a dated signal column."""
    _require_columns(panel)
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for col in ("raw_open", "raw_close", "amount", "signal"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ("suspended", "limit_up_locked", "limit_down_locked"):
        if col not in data:
            data[col] = False
        data[col] = data[col].fillna(False).astype(bool)
    if "adv_amount_20" not in data:
        data["adv_amount_20"] = pd.to_numeric(data["amount"], errors="coerce")
    else:
        data["adv_amount_20"] = pd.to_numeric(data["adv_amount_20"], errors="coerce")
    data = data.dropna(subset=["date", "code", "raw_open", "raw_close"]).sort_values(["date", "code"])
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    if len(dates) <= config.holding_sessions + 1:
        raise ValueError("insufficient_sessions_for_holding_backtest")
    by_date = {date: frame.set_index("code") for date, frame in data.groupby("date", sort=True)}
    cash = float(config.initial_capital)
    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    holding_rows: list[dict] = []
    previous_weights: dict[str, float] = {}
    incomplete_execution = False
    # A completed holding occupies entry..exit inclusive. Advance at least one
    # session beyond exit so two holding intervals never share a mark date.
    step = max(config.holding_sessions + 1, config.rebalance_sessions)
    rebalance_indices = range(0, len(dates) - config.holding_sessions - 1, step)

    for start_idx in rebalance_indices:
        signal_date = dates[start_idx]
        entry_idx = start_idx + 1
        exit_idx = entry_idx + config.holding_sessions
        entry_date, exit_date = dates[entry_idx], dates[exit_idx]
        signal_frame = by_date[signal_date].dropna(subset=["signal"])
        if signal_frame.empty:
            continue
        ranked = signal_frame.assign(_score=signal_frame["signal"] * direction).sort_values("_score", ascending=False)
        n = max(1, int(len(ranked) * config.quantile))
        target_codes = ranked.head(n).index.astype(str).tolist()
        entry = by_date[entry_date]
        eligible = entry.loc[entry.index.intersection(target_codes)].copy()
        eligible = eligible[(~eligible.suspended) & (~eligible.limit_up_locked) & eligible.raw_open.gt(0)]
        if eligible.empty:
            continue
        # Equal-weight desired portfolio; cap each order by ADV participation.
        investable = cash * (1.0 - config.cash_buffer)
        target_value = investable / len(eligible)
        positions: dict[str, dict[str, float]] = {}
        for code, row in eligible.iterrows():
            price = float(row.raw_open) * (1.0 + config.slippage_bps / 10000.0)
            adv_cap = float(row.adv_amount_20) * config.max_adv_participation
            desired_shares = int(target_value / price // config.lot_size * config.lot_size)
            cap_shares = int(max(0.0, adv_cap) / price // config.lot_size * config.lot_size)
            shares = min(desired_shares, cap_shares)
            if shares <= 0:
                continue
            notional = shares * price; fee = _fee(notional, "buy", config)
            if notional + fee > cash:
                shares = int(max(0.0, cash - config.min_commission) / price // config.lot_size * config.lot_size)
                notional = shares * price; fee = _fee(notional, "buy", config) if shares else 0.0
            if shares:
                cash -= notional + fee
                positions[code] = {"shares": float(shares), "entry_price": price, "entry_notional": notional}
                trade_rows.append({"signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "code": code, "side": "buy", "shares": shares, "price": price, "notional": notional, "fee": fee, "blocked": False})
        total_bought = sum(position["entry_notional"] for position in positions.values())
        if not total_bought:
            continue
        # Mark the one holding interval at daily close; no new portfolio is opened inside it.
        entry_value = cash + total_bought
        for mark_idx in range(entry_idx, exit_idx + 1):
            mark_date = dates[mark_idx]; mark = by_date[mark_date]
            position_value = 0.0
            for code, position in positions.items():
                if code in mark.index and np.isfinite(mark.at[code, "raw_close"]):
                    position_value += position["shares"] * float(mark.at[code, "raw_close"])
            total_value = cash + position_value
            equity_rows.append({"date": mark_date, "signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "cash": cash, "position_value": position_value, "total_value": total_value, "holding_codes": len(positions)})
        # Liquidate at exit close with sell costs; next interval starts after this point.
        exit_frame = by_date[exit_date]
        exit_value = 0.0; exit_fees = 0.0; blocked_position_value = 0.0; blocked_this_period = False
        for code, position in positions.items():
            shares = int(position["shares"])
            if code not in exit_frame.index or bool(exit_frame.at[code, "suspended"]) or bool(exit_frame.at[code, "limit_down_locked"]):
                close_mark = float(exit_frame.at[code, "raw_close"]) if code in exit_frame.index and np.isfinite(exit_frame.at[code, "raw_close"]) else 0.0
                blocked_position_value += shares * close_mark; blocked_this_period = True
                trade_rows.append({"signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "code": code, "side": "sell", "shares": shares, "price": np.nan, "notional": 0.0, "fee": 0.0, "blocked": True})
                holding_rows.append({"signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "code": code, "blocked_exit": True})
                continue
            price = float(exit_frame.at[code, "raw_close"]) * (1.0 - config.slippage_bps / 10000.0)
            sell_notional = shares * price; fee = _fee(sell_notional, "sell", config)
            cash += sell_notional - fee; exit_value += sell_notional; exit_fees += fee
            trade_rows.append({"signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "code": code, "side": "sell", "shares": shares, "price": price, "notional": sell_notional, "fee": fee, "blocked": False})
            holding_rows.append({"signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "code": code, "blocked_exit": False})
        previous_weights = {code: position["entry_notional"] for code, position in positions.items()}
        equity_rows.append({"date": exit_date, "signal_date": signal_date, "entry_date": entry_date, "exit_date": exit_date, "cash": cash, "position_value": blocked_position_value, "total_value": cash + blocked_position_value, "holding_codes": int(blocked_this_period)})
        if blocked_this_period:
            incomplete_execution = True
            break

    if not equity_rows:
        raise ValueError("no_executable_holding_period")
    equity = pd.DataFrame(equity_rows).drop_duplicates("date", keep="last").sort_values("date")
    trades = pd.DataFrame(trade_rows)
    holdings = pd.DataFrame(holding_rows)
    equity["return"] = equity["total_value"].pct_change().fillna(0.0)
    curve = equity["total_value"] / float(config.initial_capital)
    returns = equity["return"].iloc[1:]
    years = len(returns) / 252.0
    sufficient_history = len(returns) >= 252
    result = {"config": asdict(config), "sessions": int(len(equity)), "holding_periods": int(equity[["signal_date", "entry_date", "exit_date"]].drop_duplicates().shape[0]), "trades": int(len(trades)), "blocked_exits": int(trades.get("blocked", pd.Series(dtype=bool)).sum()), "execution_status": "incomplete_execution" if incomplete_execution else "complete", "total_return": float(curve.iloc[-1] - 1.0), "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if sufficient_history and years > 0 and curve.iloc[-1] > 0 else None, "sharpe": float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if sufficient_history and len(returns) > 1 and returns.std(ddof=1) > 0 else None, "max_drawdown": float((curve / curve.cummax() - 1.0).min()), "total_fees": float(trades["fee"].sum()) if not trades.empty else 0.0, "reconciliation": {"signal_to_executed_weight": "recorded_per_trade", "blocked_exit_policy": "position_marked_and_execution_blocked"}}
    return {"report": result, "equity": equity, "trades": trades, "holdings": holdings}


def run_continuous_portfolio_backtest(panel: pd.DataFrame, config: HoldingConfig = HoldingConfig(), *, direction: int = 1) -> dict:
    """Continuous target-weight rebalancing with next-open execution.

    This mirrors the Backtrader adapter contract: at each rebalance signal date
    (t close) compute an equal-weight target portfolio, then at t+1 open execute
    the delta against current holdings. Holdings persist between rebalances, and
    equity is marked at every close.
    """
    _require_columns(panel)
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for col in ("raw_open", "raw_close", "amount", "signal"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ("suspended", "limit_up_locked", "limit_down_locked"):
        if col not in data:
            data[col] = False
        data[col] = data[col].fillna(False).astype(bool)
    if "adv_amount_20" not in data:
        data["adv_amount_20"] = pd.to_numeric(data["amount"], errors="coerce")
    else:
        data["adv_amount_20"] = pd.to_numeric(data["adv_amount_20"], errors="coerce")
    data = data.dropna(subset=["date", "code", "raw_open", "raw_close"]).sort_values(["date", "code"])
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    if len(dates) <= config.rebalance_sessions + 1:
        raise ValueError("insufficient_sessions_for_continuous_backtest")
    by_date = {date: frame.set_index("code") for date, frame in data.groupby("date", sort=True)}
    cash = float(config.initial_capital)
    holdings: dict[str, float] = {}
    pending_target: dict[str, float] | None = None
    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    rejected = 0
    last_close_value = float(config.initial_capital)
    rebalance_dates = set(dates[::max(1, config.rebalance_sessions)])
    for date in dates:
        frame = by_date.get(date)
        if frame is None:
            continue
        # t+1 open execution of the target sealed at t close. The dollar target is
        # based on the portfolio value at the signal close (last_close_value), not
        # the execution open, matching the canonical Backtrader engine.
        if pending_target is not None:
            investable = last_close_value * (1.0 - config.cash_buffer)
            target_value = investable / max(1, len(pending_target))
            # Sell deltas first to free cash. target_value is the per-name target
            # notional (investable / len(target)), so it must NOT be multiplied by
            # the equal-weight fraction again (that double-divides and leaves 1/n of
            # capital invested). This aligns with the canonical Backtrader engine's
            # `desired = total * weight / price`.
            for code, shares in list(holdings.items()):
                weight = pending_target.get(code, 0.0)
                if code not in frame.index or not np.isfinite(frame.at[code, "raw_open"]) or frame.at[code, "raw_open"] <= 0:
                    desired = 0
                elif weight <= 0:
                    desired = 0
                else:
                    desired = int(target_value / max(float(frame.at[code, "raw_open"]), 1e-12) // config.lot_size * config.lot_size)
                delta = desired - int(shares)
                if delta >= 0:
                    continue
                blocked = bool(frame.at[code, "suspended"] or frame.at[code, "limit_down_locked"]) if code in frame.index else True
                if not blocked:
                    price = float(frame.at[code, "raw_open"]) * (1.0 - config.slippage_bps / 10000.0)
                    sell_shares = -delta; notional = sell_shares * price; fee = _fee(notional, "sell", config)
                    cash += notional - fee
                    holdings[code] = shares - sell_shares
                    trade_rows.append({"date": date, "code": code, "side": "sell", "shares": sell_shares, "price": price, "notional": notional, "fee": fee, "blocked": False})
                else:
                    rejected += 1
                    trade_rows.append({"date": date, "code": code, "side": "sell", "shares": -delta, "price": np.nan, "notional": 0.0, "fee": 0.0, "blocked": True})
            # Buy deltas within cash and ADV cap.
            for code, weight in pending_target.items():
                if code not in frame.index or not np.isfinite(frame.at[code, "raw_open"]) or frame.at[code, "raw_open"] <= 0:
                    continue
                blocked = bool(frame.at[code, "suspended"] or frame.at[code, "limit_up_locked"])
                if blocked:
                    rejected += 1
                    continue
                price = float(frame.at[code, "raw_open"]) * (1.0 + config.slippage_bps / 10000.0)
                desired = int(target_value / price // config.lot_size * config.lot_size)
                current = int(holdings.get(code, 0.0))
                delta = desired - current
                if delta <= 0:
                    continue
                adv_cap = float(frame.at[code, "adv_amount_20"]) * config.max_adv_participation
                cap_shares = int(max(0.0, adv_cap) / price // config.lot_size * config.lot_size)
                buy_shares = min(delta, cap_shares)
                notional = buy_shares * price; fee = _fee(notional, "buy", config)
                if notional + fee > cash:
                    buy_shares = int(max(0.0, cash - config.min_commission) / price // config.lot_size * config.lot_size)
                    notional = buy_shares * price; fee = _fee(notional, "buy", config) if buy_shares else 0.0
                if buy_shares:
                    cash -= notional + fee
                    holdings[code] = current + buy_shares
                    trade_rows.append({"date": date, "code": code, "side": "buy", "shares": buy_shares, "price": price, "notional": notional, "fee": fee, "blocked": False})
            pending_target = None
        position_value = 0.0
        for code, shares in holdings.items():
            if code in frame.index and np.isfinite(frame.at[code, "raw_close"]):
                position_value += shares * float(frame.at[code, "raw_close"])
        total_value = cash + position_value
        last_close_value = total_value
        equity_rows.append({"date": date, "cash": cash, "position_value": position_value, "total_value": total_value, "positions": len(holdings)})
        if date in rebalance_dates:
            signal_frame = frame.dropna(subset=["signal"])
            if signal_frame.empty:
                pending_target = None
                continue
            ranked = signal_frame.assign(_score=signal_frame["signal"] * direction).sort_values("_score", ascending=False)
            n = max(1, int(len(ranked) * config.quantile))
            pending_target = {str(code): 1.0 / n for code in ranked.head(n).index}
    equity = pd.DataFrame(equity_rows).sort_values("date")
    trades = pd.DataFrame(trade_rows)
    if equity.empty:
        raise ValueError("no_executable_continuous_backtest")
    equity["return"] = equity["total_value"].pct_change().fillna(0.0)
    curve = equity["total_value"] / float(config.initial_capital)
    returns = equity["return"].iloc[1:]
    years = len(returns) / 252.0
    sufficient_history = len(returns) >= 252
    result = {"config": asdict(config), "sessions": int(len(equity)), "trades": int(len(trades)), "rejected_orders": rejected, "execution_status": "complete", "total_return": float(curve.iloc[-1] - 1.0), "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if sufficient_history and years > 0 and curve.iloc[-1] > 0 else None, "sharpe": float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if sufficient_history and len(returns) > 1 and returns.std(ddof=1) > 0 else None, "max_drawdown": float((curve / curve.cummax() - 1.0).min()), "total_fees": float(trades["fee"].sum()) if not trades.empty else 0.0}
    return {"report": result, "equity": equity, "trades": trades}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True); parser.add_argument("--output", required=True); parser.add_argument("--signal", default="mom_20")
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.panel); frame["signal"] = frame[args.signal]
    result = run_portfolio_backtest(frame)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    result["equity"].to_parquet(output.with_suffix(".equity.parquet"), index=False); result["trades"].to_parquet(output.with_suffix(".trades.parquet"), index=False); result["holdings"].to_parquet(output.with_suffix(".holdings.parquet"), index=False); output.with_suffix(".json").write_text(json.dumps(result["report"], ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["report"], ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
