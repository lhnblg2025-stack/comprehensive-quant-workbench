"""Bounded cross-engine target-weight reconciliation helper."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .backtest_protocol import UnifiedBacktestResult, compare_results
from .backtrader_engine import run_backtrader
from .portfolio_holding_backtest import HoldingConfig, run_continuous_portfolio_backtest


def bounded_crosscheck(panel: pd.DataFrame, signal: str, *, days: int = 90, symbols: int = 20, holding_sessions: int = 5, quantile: float = .2) -> dict:
    data = panel.copy(); data["date"] = pd.to_datetime(data["date"]); dates = pd.DatetimeIndex(sorted(data.date.unique()))
    data = data[data.date >= dates[max(0, len(dates) - days)]].copy(); keep = sorted(data.code.astype(str).unique())[:symbols]; data = data[data.code.astype(str).isin(keep)].copy(); data["signal"] = pd.to_numeric(data[signal], errors="coerce")
    window_dates = pd.DatetimeIndex(sorted(data.date.unique())); step = max(holding_sessions + 1, holding_sessions); targets = {}
    for day in window_dates[::step]:
        ranked = data.loc[data.date.eq(day)].dropna(subset=["signal"]).nlargest(max(1, int(len(keep) * quantile)), "signal")
        if len(ranked): targets[day] = {str(code): 1.0 / len(ranked) for code in ranked.code}
    cfg = HoldingConfig(holding_sessions=holding_sessions, rebalance_sessions=step, quantile=quantile, initial_capital=1_000_000, cash_buffer=0.0)
    primary_raw = run_continuous_portfolio_backtest(data, cfg)["report"]
    primary = UnifiedBacktestResult(engine="portfolio_holding", status=primary_raw["execution_status"], initial_capital=cfg.initial_capital, final_value=cfg.initial_capital * (1 + primary_raw["total_return"]), total_return=primary_raw["total_return"], annual_return=primary_raw["annual_return"], sharpe=primary_raw["sharpe"], max_drawdown=primary_raw["max_drawdown"], observations=primary_raw["sessions"], orders=primary_raw["trades"], trades=primary_raw["trades"], blocked_orders=primary_raw.get("rejected_orders", 0), total_fees=primary_raw["total_fees"])
    secondary, secondary_equity, secondary_trades = run_backtrader(
        data,
        targets,
        capital=cfg.initial_capital,
        commission_bps=cfg.commission_bps,
        stamp_duty_bps=cfg.stamp_duty_bps,
        transfer_fee_bps=cfg.transfer_fee_bps,
        slippage_bps=cfg.slippage_bps,
        max_adv_participation=cfg.max_adv_participation,
        trade_state=data[[c for c in ("code", "date", "suspended", "limit_up_locked", "limit_down_locked") if c in data.columns]].drop_duplicates(["code", "date"]),
    )
    secondary.metadata.update({"canonical": True, "target_dates": len(targets), "trade_rows": len(secondary_trades)})
    comparison = compare_results(primary, secondary, return_tolerance=.05, final_value_tolerance=.05)
    comparison["canonical_engine"] = secondary.engine
    return {"window": {"start": str(data.date.min().date()), "end": str(data.date.max().date()), "symbols": len(keep), "target_dates": len(targets)}, "primary": primary.to_dict(), "secondary": secondary.to_dict(), "comparison": comparison}
