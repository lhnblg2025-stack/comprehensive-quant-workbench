from __future__ import annotations

from .config import PortfolioConfig, StrategyConfig


def market_allows_new_buy(market_state: dict, config: StrategyConfig) -> bool:
    return int(market_state.get("risk", 5)) <= config.max_risk_score_for_new_buy


def position_size(entry_price: float, stop_price: float, portfolio: PortfolioConfig) -> int:
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return 0
    risk_cash = portfolio.initial_cash * portfolio.risk_per_trade_pct
    max_cash = portfolio.initial_cash * portfolio.max_position_pct
    risk_per_share = entry_price - stop_price
    shares_by_risk = int(risk_cash / risk_per_share // 100 * 100)
    shares_by_cap = int(max_cash / entry_price // 100 * 100)
    return max(0, min(shares_by_risk, shares_by_cap))


def enrich_trade_plan(signal: dict, market_state: dict, strategy: StrategyConfig, portfolio: PortfolioConfig) -> dict:
    if signal.get("status") == "insufficient_data":
        return signal
    close = float(signal["close"])
    stop = round(close * (1 - strategy.stop_loss_pct), 3)
    take_profit = round(close * (1 + strategy.take_profit_pct), 3)
    trail = round(close * (1 - strategy.trail_stop_pct), 3)
    allowed = market_allows_new_buy(market_state, strategy)
    shares = position_size(close, stop, portfolio) if signal.get("action") == "BUY_WATCH" and allowed else 0
    out = dict(signal)
    out.update(
        {
            "market_risk": market_state.get("risk"),
            "market_risk_level": market_state.get("risk_level"),
            "new_buy_allowed": allowed,
            "stop_loss": stop,
            "take_profit_ref": take_profit,
            "trailing_stop_ref": trail,
            "suggested_shares": shares,
            "suggested_cash": round(shares * close, 2),
        }
    )
    if signal.get("action") == "BUY_WATCH" and not allowed:
        out["action"] = "MARKET_RISK_BLOCKED"
    return out

