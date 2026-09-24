"""Signal-driven long-only portfolio backtest on genuine RAW trade prices.

The engine separates three price roles that were previously conflated:

* **signal price**: HFQ (dividend/split adjusted) closes, used to build returns
  and cross-sectional scores;
* **trade price**: genuine RAW open/close, the only price a real order can hit,
  so share counts and the 100-share board lot are computed in RAW space;
* **total-return marking**: a held position is marked at RAW close plus the
  cash value of the adjustment-factor step since the previous session, which
  reproduces ``shares * hfq_close / factor_basis`` while keeping the cash book
  in genuine yuan.

Portfolio construction is intentionally *not* equal weight.  At each rebalance
the cross-section is standardised, names with a positive score enter (capped at
``max_positions``), names whose edge disappears leave, and capital is allocated
in proportion to the surviving scores.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalPortfolioConfig:
    rebalance_sessions: int = 5
    max_positions: int = 50
    exit_positions: int = 100
    cash_buffer: float = 0.05
    initial_capital: float = 1_000_000.0
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    lot_size: int = 100
    min_commission: float = 5.0
    max_adv_participation: float = 0.10
    max_weight: float = 0.10
    min_positions: int = 1


REQUIRED_COLUMNS = {
    "date", "code", "raw_open", "hfq_close", "adjust_factor",
    "signal", "suspended", "limit_up_locked", "limit_down_locked",
}


def _fee(notional: float, side: str, cfg: SignalPortfolioConfig) -> float:
    commission = max(cfg.min_commission, notional * cfg.commission_bps / 10000.0)
    transfer = notional * cfg.transfer_fee_bps / 10000.0
    stamp = notional * cfg.stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    return commission + transfer + stamp


def _capped_weights(scores: pd.Series, max_weight: float) -> dict[str, float]:
    """Score-proportional weights, renormalised until the per-name cap holds."""
    positive = scores[scores > 0]
    if positive.empty:
        return {}
    weights = positive / positive.sum()
    for _ in range(50):
        over = weights[weights > max_weight + 1e-12]
        if over.empty:
            break
        free = weights[weights <= max_weight + 1e-12]
        overflow = float((over - max_weight).sum())
        weights[over.index] = max_weight
        if free.empty or free.sum() <= 0:
            break
        weights[free.index] = free / free.sum() * (1.0 - max_weight * len(over))
        weights = weights.clip(upper=max_weight)
    total = float(weights.sum())
    if total <= 0:
        return {}
    weights = weights / total
    return {str(code): float(value) for code, value in weights.items()}


def build_panel_matrices(
    panel: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Pivot only the requested date window to keep peak memory bounded."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    for column in ("raw_open", "raw_close", "hfq_close", "adjust_factor", "signal", "adv_amount_20"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    if "adv_amount_20" not in frame.columns:
        frame["adv_amount_20"] = np.nan
    for column in ("suspended", "limit_up_locked", "limit_down_locked"):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame = frame.sort_values(["date", "code"])
    dates = pd.DatetimeIndex(sorted(frame["date"].dropna().unique()))
    matrices: dict[str, pd.DataFrame] = {}
    for column in ("raw_open", "raw_close", "hfq_close", "adjust_factor", "signal", "adv_amount_20"):
        if column in frame.columns:
            matrices[column] = frame.pivot(index="date", columns="code", values=column).reindex(dates)
    for column in ("suspended", "limit_up_locked", "limit_down_locked"):
        block = frame.pivot(index="date", columns="code", values=column).reindex(dates)
        matrices[column] = block.fillna(True).astype(bool)
    return matrices


def _panel_matrices(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return build_panel_matrices(panel)


def run_signal_portfolio_backtest(
    panel: pd.DataFrame,
    config: SignalPortfolioConfig = SignalPortfolioConfig(),
    *,
    start: str | None = None,
    end: str | None = None,
    matrices: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError("panel_missing_columns:" + ",".join(missing))
    if matrices is None:
        matrices = build_panel_matrices(panel, start=start, end=end)
    raw_open = matrices["raw_open"]
    raw_close = matrices["raw_close"]
    hfq_close_matrix = matrices["hfq_close"]
    factor = matrices["adjust_factor"]
    signal = matrices["signal"]
    adv = matrices["adv_amount_20"]
    suspended = matrices["suspended"]
    limit_up = matrices["limit_up_locked"]
    limit_down = matrices["limit_down_locked"]

    dates = hfq_close_matrix.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) <= config.rebalance_sessions + 1:
        raise ValueError("insufficient_sessions_for_signal_backtest")

    step = max(1, int(config.rebalance_sessions))
    rebalance_positions = list(range(0, len(dates) - 1, step))
    rebalance_dates = {dates[position] for position in rebalance_positions}

    slip_buy = 1.0 + config.slippage_bps / 10000.0
    slip_sell = 1.0 - config.slippage_bps / 10000.0
    cash = float(config.initial_capital)
    # Holdings are HFQ units; current RAW shares = units * adjustment_factor.
    holdings: dict[str, float] = {}
    pending_target: dict[str, float] | None = None
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    rejected = 0
    last_close_value = float(config.initial_capital)

    for position, date in enumerate(dates):
        row_open = raw_open.loc[date]
        row_close = raw_close.loc[date]
        row_factor = factor.loc[date]

        if pending_target is not None and position > 0:
            investable = last_close_value
            desired_value = {code: weight * investable for code, weight in pending_target.items()}

            # Sell legs first so cash is available for the buy legs.
            for code in list(holdings):
                if holdings[code] <= 0:
                    continue
                target_value = desired_value.get(code, 0.0)
                price_ref = row_open.get(code, np.nan)
                if not np.isfinite(price_ref) or price_ref <= 0:
                    if target_value <= 0:
                        rejected += 1
                    continue
                desired_raw = int(target_value / price_ref // config.lot_size * config.lot_size)
                current_raw = float(holdings[code]) * float(row_factor.get(code, np.nan))
                if not np.isfinite(current_raw):
                    continue
                if desired_raw <= 0:
                    # A full exit may dispose of any residual odd lot.
                    shares = int(round(current_raw))
                else:
                    delta = desired_raw - int(current_raw // config.lot_size * config.lot_size)
                    if delta >= 0:
                        continue
                    shares = int(-delta // config.lot_size * config.lot_size)
                shares = min(shares, int(round(current_raw)))
                if shares <= 0:
                    if desired_raw <= 0:
                        holdings.pop(code, None)
                    continue
                blocked = bool(suspended.at[date, code] or limit_down.at[date, code])
                if blocked:
                    rejected += 1
                    trade_rows.append({"date": date, "code": code, "side": "sell", "shares": shares,
                                       "price": np.nan, "notional": 0.0, "fee": 0.0, "blocked": True})
                    continue
                price = float(price_ref) * slip_sell
                notional = shares * price
                fee = _fee(notional, "sell", config)
                cash += notional - fee
                factor_now = float(row_factor.get(code, np.nan))
                holdings[code] -= shares / factor_now
                if holdings[code] <= 1e-12 or holdings[code] * factor_now < 1.0:
                    holdings.pop(code, None)
                trade_rows.append({"date": date, "code": code, "side": "sell", "shares": shares,
                                   "price": price, "notional": notional, "fee": fee, "blocked": False})

            # Buy legs sized against the live cash balance.
            for code, target_value in desired_value.items():
                if target_value <= 0:
                    continue
                price_ref = row_open.get(code, np.nan)
                if not np.isfinite(price_ref) or price_ref <= 0:
                    rejected += 1
                    continue
                if bool(suspended.at[date, code] or limit_up.at[date, code]):
                    rejected += 1
                    continue
                price = float(price_ref) * slip_buy
                desired_raw = int(target_value / price // config.lot_size * config.lot_size)
                current_raw = float(holdings.get(code, 0.0)) * float(row_factor.get(code, np.nan))
                if not np.isfinite(current_raw):
                    continue
                delta = desired_raw - int(current_raw // config.lot_size * config.lot_size)
                if delta <= 0:
                    continue
                adv_now = adv.at[date, code] if code in adv.columns else np.nan
                if np.isfinite(adv_now) and adv_now > 0:
                    cap_shares = int(adv_now * config.max_adv_participation / price // config.lot_size * config.lot_size)
                    delta = min(delta, max(cap_shares, 0))
                if delta <= 0:
                    continue
                notional = delta * price
                fee = _fee(notional, "buy", config)
                if notional + fee > cash:
                    fee_rate = (config.commission_bps + config.transfer_fee_bps) / 10000.0
                    delta = int(max(0.0, cash / (price * (1.0 + fee_rate))) // config.lot_size * config.lot_size)
                    for _ in range(100):
                        if delta <= 0:
                            break
                        notional = delta * price
                        fee = _fee(notional, "buy", config)
                        if notional + fee <= cash:
                            break
                        delta -= config.lot_size
                    if delta <= 0:
                        rejected += 1
                        continue
                    notional = delta * price
                    fee = _fee(notional, "buy", config)
                cash -= notional + fee
                holdings[code] = holdings.get(code, 0.0) + delta / float(row_factor.get(code, 1.0))
                trade_rows.append({"date": date, "code": code, "side": "buy", "shares": delta,
                                   "price": price, "notional": notional, "fee": fee, "blocked": False})
            pending_target = None

        position_value = 0.0
        row_hfq_close = hfq_close_matrix.loc[date]
        for code, shares in list(holdings.items()):
            hfq_close = row_hfq_close.get(code, np.nan)
            if not np.isfinite(hfq_close):
                continue
            value = shares * float(hfq_close)
            if value < 1.0:
                # Remove rounding dust so it cannot inflate the position count.
                holdings.pop(code, None)
                continue
            position_value += value
        total_value = cash + position_value
        last_close_value = total_value
        equity_rows.append({"date": date, "cash": cash, "position_value": position_value,
                            "total_value": total_value, "positions": len(holdings)})

        if date in rebalance_dates:
            scores = signal.loc[date].dropna()
            treatable = ~suspended.loc[date].reindex(scores.index).fillna(True)
            scores = scores[treatable]
            if scores.empty:
                pending_target = None
                continue
            ranked = scores.sort_values(ascending=False)
            entry_names = [code for code in ranked.index[: config.max_positions] if ranked[code] > 0]
            keep_names = [code for code in ranked.index[: config.exit_positions] if ranked[code] > 0]
            keep_set = set(keep_names)
            target_names = [code for code in holdings if code in keep_set]
            for code in entry_names:
                if code not in target_names:
                    target_names.append(code)
            if len(target_names) < config.min_positions:
                pending_target = {}
                continue
            pending_target = _capped_weights(ranked[target_names], config.max_weight)

    equity = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    trades = pd.DataFrame(trade_rows)
    if equity.empty:
        raise ValueError("no_executable_signal_backtest")
    equity["return"] = equity["total_value"].pct_change().fillna(0.0)
    curve = equity["total_value"] / float(config.initial_capital)
    returns = equity["return"].iloc[1:]
    years = len(returns) / 252.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    report = {
        "config": asdict(config),
        "sessions": int(len(equity)),
        "rebalances": int(len(rebalance_dates)),
        "orders": int(len(trades)),
        "rejected_orders": int(rejected),
        "trades": int((~trades["blocked"]).sum()) if not trades.empty else 0,
        "execution_status": "complete",
        "total_return": float(curve.iloc[-1] - 1.0),
        "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and curve.iloc[-1] > 0 else None,
        "annual_volatility": float(std * math.sqrt(252.0)),
        "sharpe": float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else None,
        "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
        "total_fees": float(trades["fee"].sum()) if not trades.empty else 0.0,
        "total_turnover": float(trades.loc[~trades["blocked"], "notional"].sum()) if not trades.empty else 0.0,
        "average_positions": float(equity["positions"].mean()),
        "final_positions": int(equity["positions"].iloc[-1]),
        "cash_ratio_final": float(equity["cash"].iloc[-1] / equity["total_value"].iloc[-1])
        if equity["total_value"].iloc[-1] else None,
    }
    return {"report": report, "equity": equity, "trades": trades}
