"""Auditable catalog of supported strategy families and panel signals."""
from __future__ import annotations

from typing import Any

STRATEGY_CATALOG: tuple[dict[str, Any], ...] = (
    {"id": "reversal", "name": "均值回复", "description": "公开 RSI 反转策略", "signals": ("rsi_rev_14",)},
    {"id": "volatility", "name": "低波与波动率", "description": "公开低波策略", "signals": ("low_vol_60",)},
    {"id": "momentum", "name": "动量", "description": "公开 12-1 月动量策略", "signals": ("mom_12_1",)},
)

STRATEGY_DATA_METADATA: dict[str, dict[str, Any]] = {
    "reversal": {"research_status": "READY_RESEARCH", "required_data_domains": ["stock_price_daily"], "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE", "known_limitations": ["frozen_universe", "research_only"]},
    "volatility": {"research_status": "READY_RESEARCH", "required_data_domains": ["stock_price_daily"], "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE", "known_limitations": ["frozen_universe", "research_only"]},
    "momentum": {"research_status": "READY_RESEARCH", "required_data_domains": ["stock_price_daily"], "formal_backtest_gate": "BLOCKED_SURVIVORSHIP_AND_TRADE_STATE", "known_limitations": ["frozen_universe", "research_only"]},
}


def catalog() -> list[dict[str, Any]]:
    return [dict(item, signals=list(item["signals"]), **STRATEGY_DATA_METADATA[item["id"]]) for item in STRATEGY_CATALOG]


def signal_catalog() -> list[dict[str, str]]:
    return [{"signal": signal, "family": item["id"], "family_name": item["name"], "description": item["description"]} for item in STRATEGY_CATALOG for signal in item["signals"]]


def find_signal(signal: str) -> dict[str, str] | None:
    return next((item for item in signal_catalog() if item["signal"] == signal), None)
