"""Auditable catalog of the strategy families published in this release.

Only the three strategies selected for public release expose concrete signals:

* ``momentum_12_1`` — 12-1 month cross-sectional momentum (``trend`` family)
* ``rsi_reversal``  — 14-day RSI mean reversion (``reversal`` family)
* ``low_volatility``— 60-day low-volatility cross-section (``volatility`` family)

The remaining families are listed as *unreleased* so the catalog stays honest
about the workbench's architecture without publishing private signal designs,
parameters or research campaigns. See ``DECLASSIFICATION.md``.
"""
from __future__ import annotations

from typing import Any

PUBLIC_STRATEGIES: tuple[str, ...] = ("momentum_12_1", "rsi_reversal", "low_volatility")

STRATEGY_CATALOG: tuple[dict[str, Any], ...] = (
    {"id": "trend", "name": "趋势跟踪", "description": "跨截面动量与趋势排序。", "signals": ("momentum_12_1",)},
    {"id": "reversal", "name": "均值回复", "description": "超卖反转类截面信号。", "signals": ("rsi_reversal",)},
    {"id": "volatility", "name": "低波与波动率", "description": "低波动率截面排序。", "signals": ("low_volatility",)},
    {"id": "volume_flow", "name": "量价与流动性", "description": "量能、换手与冲击成本代理（本版未公开具体信号）。", "signals": ()},
    {"id": "cross_section", "name": "横截面多因子", "description": "统一 HFQ 信号、RAW 次日开盘执行、显式成本与 CSI300 可比口径（本版未公开具体信号）。", "signals": ()},
    {"id": "pit_value_quality", "name": "PIT 价值质量", "description": "公告日财务与历史行业口径；缺少交易状态、生命周期或容量时保持 data_blocked。", "signals": ()},
    {"id": "execution_audit", "name": "执行审计", "description": "阻断事件、整手、流动性、涨跌停和停牌原因归因。", "signals": ()},
    {"id": "event_flow", "name": "事件与资金", "description": "融资融券、龙虎榜、涨停结构和公告事件，需通过 PIT 门禁。", "signals": ()},
    {"id": "market_timing", "name": "市场择时", "description": "指数趋势、宽度、波动率和宏观状态过滤。", "signals": ()},
    {"id": "pairs", "name": "配对与套利", "description": "价差、协整和相对价值，需多标的数据面板。", "signals": ()},
    {"id": "ml", "name": "机器学习", "description": "滚动训练、分类/回归和 walk-forward challenger。", "signals": ()},
)

STRATEGY_DATA_METADATA: dict[str, dict[str, Any]] = {
    "trend": {
        "research_status": "READY_RESEARCH",
        "required_data_domains": ["stock_price_daily"],
        "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE",
        "known_limitations": ["frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"],
    },
    "reversal": {
        "research_status": "READY_RESEARCH",
        "required_data_domains": ["stock_price_daily"],
        "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE",
        "known_limitations": ["frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"],
    },
    "volatility": {
        "research_status": "READY_RESEARCH",
        "required_data_domains": ["stock_price_daily"],
        "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE",
        "known_limitations": ["frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"],
    },
    "volume_flow": {
        "research_status": "PARTIAL_RESEARCH",
        "required_data_domains": ["stock_price_daily", "liquidity_daily"],
        "formal_backtest_gate": "BLOCKED_SOURCE_AND_TRADE_STATE",
        "known_limitations": ["synthetic_proxy_fields", "source_unit_normalization", "frozen_universe"],
    },
    "cross_section": {
        "research_status": "PARTIAL_RESEARCH",
        "required_data_domains": ["stock_price_daily", "liquidity_daily"],
        "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE",
        "known_limitations": ["frozen_universe", "synthetic_proxy_fields", "no_pit_membership"],
    },
    "pit_value_quality": {
        "research_status": "BLOCKED",
        "required_data_domains": ["pit_financials", "pit_trade_state"],
        "formal_backtest_gate": "BLOCKED_PIT_ASOF",
        "known_limitations": ["announcement_asof_missing", "authoritative_trade_state_missing"],
    },
    "execution_audit": {
        "research_status": "BLOCKED",
        "required_data_domains": ["pit_trade_state", "order_ledger"],
        "formal_backtest_gate": "BLOCKED_AUTHORITATIVE_EXECUTION_STATE",
        "known_limitations": ["authoritative_trade_state_missing", "execution_reconciliation_missing"],
    },
    "market_timing": {
        "research_status": "READY_RESEARCH",
        "required_data_domains": ["official_index_daily"],
        "formal_backtest_gate": "BLOCKED_STOCK_UNIVERSE_AND_EXECUTION",
        "known_limitations": ["index_data_is_not_stock_trade_state", "frozen_stock_universe"],
    },
}


def catalog() -> list[dict[str, Any]]:
    return [
        dict(item, signals=list(item["signals"]), **STRATEGY_DATA_METADATA.get(item["id"], {}))
        for item in STRATEGY_CATALOG
    ]


def signal_catalog() -> list[dict[str, str]]:
    return [{"signal": signal, "family": item["id"], "family_name": item["name"], "description": item["description"]} for item in STRATEGY_CATALOG for signal in item["signals"]]


def find_signal(signal: str) -> dict[str, str] | None:
    return next((item for item in signal_catalog() if item["signal"] == signal), None)
