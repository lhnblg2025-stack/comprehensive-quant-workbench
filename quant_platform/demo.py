"""Deterministic synthetic prices; no downloaded or account data."""
import numpy as np
import pandas as pd
from quant_system.backtest import run_backtest
from quant_system.config import StrategyConfig, PortfolioConfig

def prices():
    rng = np.random.default_rng(2026)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400)))
    return pd.DataFrame(dict(date=pd.bdate_range('2024-01-01', periods=400),
        open=close * 0.999, high=close * 1.01, low=close * 0.99,
        close=close, volume=np.full(400, 1000000)))

def run():
    return {'data_kind': 'synthetic', 'notice': '仅供研究，不构成投资建议',
            'result': run_backtest(prices(), StrategyConfig(), PortfolioConfig())}
