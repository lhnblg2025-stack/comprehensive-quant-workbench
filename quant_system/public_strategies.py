"""The three concrete strategies included in the public release.

The strategy engine, matrix backtester, data gates, and promotion framework
remain generic.  This module is the only concrete strategy surface shipped in
this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PublicStrategy:
    key: str
    family: str
    signal_column: str
    description: str

PUBLIC_STRATEGIES = (
    PublicStrategy("rsi_reversal", "reversal", "rsi_rev_14", "RSI(14) mean-reversion signal."),
    PublicStrategy("low_volatility", "volatility", "low_vol_60", "60-session low-volatility signal."),
    PublicStrategy("momentum_12_1", "momentum", "mom_12_1", "12-1 month momentum, skipping the most recent month."),
)
PUBLIC_STRATEGY_KEYS = tuple(item.key for item in PUBLIC_STRATEGIES)
PUBLIC_STRATEGY_BY_KEY = {item.key: item for item in PUBLIC_STRATEGIES}

def list_public_strategies() -> list[dict[str, str]]:
    return [item.__dict__.copy() for item in PUBLIC_STRATEGIES]

def select_signal(features: pd.DataFrame, strategy: str) -> pd.Series:
    """Return one approved signal column; reject all unpublished implementations."""
    spec = PUBLIC_STRATEGY_BY_KEY.get(strategy)
    if spec is None:
        raise ValueError(f"strategy_not_public:{strategy}")
    if spec.signal_column not in features.columns:
        raise ValueError(f"signal_missing:{spec.signal_column}")
    return features[spec.signal_column]
