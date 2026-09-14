"""metrics_calculator.py — 绩效统一口径（P1-4：Sharpe 无风险利率统一）。

背景 (audit 回测层 P1-4)：此前同系统内 4 种 Sharpe 口径并存、不可比：
  - backtest.py:      excess = returns - 0.02/252，分母 excess.std()        → 扣 rf
  - backtest_pro.py:  ann_ret/ann_vol，不扣无风险利率                          → 不扣 rf
  - signal_backtest.py: rf = 0                                                  → rf=0
  - backtest_engine.py compute_sharpe: 扣 rf=0.02，ddof=1，年化 sqrt(252)      → 扣 rf

统一口径 = blacktest_engine.compute_sharpe 作为基准：扣年化 rf / trading_days，
分母用 ddof=1（样本标准差），年化 sqrt(trading_days)。所有引擎/metrics 复用此处。
"""
from __future__ import annotations

import math
from typing import List, Union

import numpy as np
import pandas as pd


class MetricsCalculator:
    """统一绩效指标计算器。

    以 backtest_engine.compute_sharpe 为基准实现；其余引擎/metrics 复用同一口径，
    消除跨引擎 Sharpe 数值不可比问题。
    """

    DEFAULT_RISK_FREE_RATE = 0.02
    DEFAULT_TRADING_DAYS = 252

    @staticmethod
    def sharpe(
        returns: Union[pd.Series, np.ndarray, List[float]],
        rf: float = DEFAULT_RISK_FREE_RATE,
        ddof: int = 1,
        trading_days: int = DEFAULT_TRADING_DAYS,
        annualize: bool = True,
    ) -> float:
        """统一 Sharpe 口径：扣无风险利率、样本标准差(ddof)、年化 sqrt(trading_days)。

        Args:
            returns: 日收益率序列。
            rf: 年化无风险利率（默认 0.02）。
            ddof: 标准差自由度（默认 1 = 样本标准差，与 backtest_engine 一致）。
            trading_days: 年化交易日数（默认 252）。
            annualize: 是否年化。

        Returns:
            年化 Sharpe，样本不足(<2)或 std≈0 时返回 0.0。
        """
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        daily_rf = rf / trading_days
        excess = arr - daily_rf
        mean_excess = float(np.mean(excess))
        std_excess = float(np.std(arr, ddof=ddof))
        if std_excess < 1e-10:
            return 0.0
        sharpe = mean_excess / std_excess
        if annualize:
            sharpe *= math.sqrt(trading_days)
        return float(sharpe)
