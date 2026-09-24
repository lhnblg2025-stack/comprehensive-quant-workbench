"""Batch next-open order-level validation for frozen PIT strategy candidates.

This executes explicit target-weight rebalances on raw next-open prices, maintains
cash and board-lot positions, applies commission/slippage, and marks equity at raw
close. It is intentionally restricted to OOS candidate validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .factor_backtest_runner import _rebalance_date_set, _select_with_industry_cap
from .research_pipeline import load_config


def _targets(panel: pd.DataFrame, factor: str, direction: int, quantile: float, rebalance: str, max_industry_weight: float) -> dict[pd.Timestamp, dict[str, float]]:
    """Create t-close signals for execution at the next trading session open."""
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    selected_dates = _rebalance_date_set(dates, rebalance)
    next_session = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    result = {}
    for date, group in panel.groupby("date", sort=True):
        signal_date = pd.Timestamp(date)
        if signal_date not in selected_dates or signal_date not in next_session:
            continue
        ranked = group.dropna(subset=[factor, "raw_close", "raw_open"]).assign(_score=lambda x: x[factor] * direction).sort_values("_score", ascending=False)
        n = max(1, int(len(ranked) * quantile))
        selected = _select_with_industry_cap(ranked, n, max_industry_weight)
        result[next_session[signal_date]] = {code: 1.0 / len(selected) for code in selected}
    return result


def _execute(panel: pd.DataFrame, targets: dict[pd.Timestamp, dict[str, float]], *, capital: float, commission_bps: float, slippage_bps: float) -> dict:
    by_date = {date: group.set_index("code") for date, group in panel.groupby("date", sort=True)}
    dates = sorted(by_date)
    cash = float(capital); holdings: dict[str, int] = {}; orders = []; trades = []; curve = []
    commission = commission_bps / 10000.0; slippage = slippage_bps / 10000.0
    for date in dates:
        frame = by_date[date]
        if date in targets:
            open_prices = pd.to_numeric(frame["raw_open"], errors="coerce")
            close_prices = pd.to_numeric(frame["raw_close"], errors="coerce")
            marked = cash + sum(shares * float(close_prices.get(code, 0.0)) for code, shares in holdings.items())
            target = targets[date]
            desired = {}
            for code, weight in target.items():
                price = float(open_prices.get(code, float("nan")))
                if not math.isfinite(price) or price <= 0:
                    continue
                desired[code] = int((marked * weight / (price * (1 + commission + slippage))) // 100 * 100)
            for code in list(holdings):
                price = float(open_prices.get(code, float("nan")))
                goal = desired.get(code, 0)
                current = holdings[code]
                if not math.isfinite(price) or price <= 0 or current <= goal:
                    continue
                quantity = current - goal; fill = price * (1 - slippage); fee = quantity * fill * commission
                cash += quantity * fill - fee; holdings[code] = goal
                if not goal: holdings.pop(code)
                orders.append({"date": str(date.date()), "code": code, "side": "sell", "shares": quantity, "fill": fill, "fee": fee}); trades.append(1)
            for code, goal in desired.items():
                price = float(open_prices.get(code, float("nan"))); current = holdings.get(code, 0); quantity = max(0, goal - current)
                if not quantity or not math.isfinite(price) or price <= 0:
                    continue
                fill = price * (1 + slippage); fee = quantity * fill * commission; needed = quantity * fill + fee
                if needed > cash:
                    quantity = int((cash / (fill * (1 + commission))) // 100 * 100); needed = quantity * fill * (1 + commission)
                if quantity <= 0: continue
                cash -= needed; holdings[code] = current + quantity
                orders.append({"date": str(date.date()), "code": code, "side": "buy", "shares": quantity, "fill": fill, "fee": quantity * fill * commission}); trades.append(1)
        close_prices = pd.to_numeric(frame["raw_close"], errors="coerce")
        position_value = sum(shares * float(close_prices.get(code, 0.0)) for code, shares in holdings.items())
        total = cash + position_value
        curve.append({"date": date, "cash": cash, "position_value": position_value, "total_value": total, "positions": len(holdings)})
    equity = pd.DataFrame(curve)
    equity["drawdown"] = equity.total_value / equity.total_value.cummax() - 1.0
    return {"orders": orders, "trades": len(trades), "equity": equity, "total_return": float(equity.total_value.iloc[-1] / capital - 1), "max_drawdown": float(equity.drawdown.min()), "final_cash": float(cash)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--panel", required=True); parser.add_argument("--matrix", required=True); parser.add_argument("--output", required=True); parser.add_argument("--top", type=int, default=10); parser.add_argument("--oos-ratio", type=float, default=.30)
    args = parser.parse_args(); config = load_config(args.config); panel = pd.read_parquet(args.panel); panel["date"] = pd.to_datetime(panel["date"])
    dates = pd.DatetimeIndex(sorted(panel.date.unique())); start = dates[int(len(dates) * (1 - args.oos_ratio))]; panel = panel[panel.date >= start].copy()
    matrix = pd.read_csv(args.matrix); candidates = matrix[(matrix.status == "complete") & (matrix.cost_x2_oos_total_return > 0)].sort_values(["oos_excess_annual", "oos_sharpe"], ascending=False).head(args.top)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); execution = config["execution"]; directions = config["factors"].get("directions", {}); results=[]
    for row in candidates.itertuples(index=False):
        targets = _targets(panel, row.factor, int(directions.get(row.factor, 1)), float(row.quantile), row.rebalance, float(config["portfolio"]["max_industry_weight"]))
        outcome = _execute(panel, targets, capital=float(execution["initial_capital"]), commission_bps=float(execution["commission_bps"]), slippage_bps=float(execution["slippage_bps"]))
        equity_path = output / f"{row.case}_equity.csv"; outcome["equity"].to_csv(equity_path, index=False)
        order_path = output / f"{row.case}_orders.json"; order_path.write_text(json.dumps(outcome["orders"], ensure_ascii=True, indent=2), encoding="utf-8")
        result={"case":row.case,"factor":row.factor,"rebalance":row.rebalance,"quantile":float(row.quantile),"oos_start":str(start.date()),"orders":len(outcome["orders"]),"trades":outcome["trades"],"total_return":outcome["total_return"],"max_drawdown":outcome["max_drawdown"],"equity_observations":len(outcome["equity"]),"final_cash":outcome["final_cash"]}; results.append(result); print(json.dumps(result,ensure_ascii=True),flush=True)
    report={"schema":"pit_order_validation/v2","data_quality":config.get("experiment",{}).get("data_quality"),"selection":"top_net_oos_excess_with_positive_2x_cost","validated_cases":len(results),"oos_start":str(start.date()),"execution":"raw_next_open_board_lot_cash_accounting","results":results}; (output/"order_level_validation.json").write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding="utf-8")
    return 0
if __name__=='__main__': raise SystemExit(main())
