"""Public watchlist interface. No symbols or personal state are bundled."""
from __future__ import annotations
from typing import Any

def get_watchlist(refresh: bool = False) -> list[dict[str, Any]]:
    return []

def fetch_quotes(symbols: list[str] | None = None, force: bool = False) -> list[dict[str, Any]]:
    return []

def _cap_tier(value: float) -> str:
    return "unclassified"

def count_watchlist() -> dict[str, Any]:
    return {"total": 0, "total_market_cap_yi": 0, "tiers": {}}
