"""Local A-share quant research system.

牧云天枢 V13.0.0 (2026-08-14): 盘中决策链五视角 + 盘后龙虎榜/低估池 + GTJA因子 + 融合链回传修复
"""

__version__ = "13.0.0"

__all__ = [
    "backtest", "data", "risk", "signals", "watchlist",
    "market_temperature", "north_flow", "education",
    "signal_backtest", "fundamental",
    "trade_db", "trading_journal",
    "risk_budget", "ml_signals", "close_review",
    "sector_rotation", "ashare_special", "macro_calendar",
    "rule_engine", "strategy_engine",
    "stock_pool", "earnings_calendar", "cross_section", "trade_executor",
    "simulation", "portfolio_risk", "combined_backtest", "advanced_strategies",
    "market_pulse", "chart_patterns", "risk_management_pro",
    # ── V5 深度模块 ──
    "v5_api",
    "sentiment_factory",
    "market_depth",
    "portfolio_diagnostics",
    "cross_market",
    "signal_tracker",
    "alert_engine",
    "performance_deep",
    "rebalance_advisor",
    "seasonality",
    # ── 工具链集成 ──
    "integrations",
    # ── V5.1 新增 ──
    "model_registry",
    "parquet_store",
    "closed_loop",
    "full_chain",
]
