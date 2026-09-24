"""
Base factor utilities for QuantV6.
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 本模块为 QuantV6 因子基座(含 common_dates 唯一真源)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.math_utils import mad_winsorize, zscore


def safe_returns(close: pd.Series, limit: float = 0.5, flag: bool = False):
    """安全单日收益率（pct_change 的防御版）。

    - prev_close 为 0/NaN（退市/停牌/脏数据）→ 收益率 NaN，而不是 ±inf
    - 单日收益率 |ret| > limit（默认 0.5，即 ±50%）→ 判定异常并置 NaN
      （A 股涨跌停 ±10%/±20%，0.5 阈值不会误伤）
    - flag=True 时返回 (returns, anomaly) 元组：anomaly 为 bool Series，
      异常日（含 inf 源）为 True，便于外部单独加标记列
    D4收敛登记: 独特工具保留
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    prev = close.shift(1)
    ret = close / prev - 1
    invalid = prev.isna() | (prev <= 0) | close.isna()
    ret = ret.where(~invalid)  # 0/NaN 前收 → NaN，消除 ±inf
    anomaly = ret.abs() > limit
    ret = ret.where(~anomaly.fillna(False))
    if flag:
        return ret, anomaly.fillna(False)
    return ret


@dataclass
class FactorResult:
    """Cross-sectional factor values for one trade date."""

    name: str
    values: pd.Series
    date: str = ""
    direction: int = 1
    meta: dict = field(default_factory=dict)

    def standardized(self) -> pd.Series:
        """方向感知标准化（越大越看多）：去极值 → z-score → 乘 direction。

        P0-2：这是单因子场景的等价入口；zoo.compute_factor_frame 在原始值上
        乘 direction（z-score 线性等价，最终方向一致），二者任选其一即可，
        不得叠加（会双重翻转）。修复了原实现 inf→pd.NA 导致 astype(float) 崩溃的路径。
        """
        vals = pd.to_numeric(self.values, errors="coerce").astype(float)
        vals = vals.replace([float("inf"), float("-inf")], float("nan"))
        vals = vals.dropna()
        if vals.empty:
            return pd.Series(dtype=float)
        return zscore(mad_winsorize(vals)) * self.direction


@dataclass
class Factor:
    """Small callable factor wrapper."""

    name: str
    func: Callable[[dict[str, pd.DataFrame]], pd.Series]
    direction: int = 1
    description: str = ""
    active: bool = True
    base_direction: int | None = None  # 定义时基准方向（校准覆盖只改 direction，不改 base）

    def __post_init__(self) -> None:
        # 2026-08-14: 方向校准以 base_direction 为基线，否则覆盖叠加导致振荡
        if self.base_direction is None:
            self.base_direction = self.direction

    def compute(self, data: dict[str, pd.DataFrame], date: str = "") -> FactorResult:
        values = self.func(data)
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        return FactorResult(self.name, values, date=date, direction=self.direction)


def latest_close(data: dict[str, pd.DataFrame]) -> pd.Series:
    """D4收敛登记: 独特工具保留"""
    out = {}
    for sym, df in data.items():
        if df is not None and len(df) and "close" in df.columns:
            out[sym] = float(df["close"].astype(float).iloc[-1])
    return pd.Series(out, dtype=float)


def latest_return(data: dict[str, pd.DataFrame], window: int = 20,
                  limit: float = 0.5) -> pd.Series:
    """N 日窗口收益（close[-1]/close[-window-1]-1），安全版。

    - 基期 close 为 0/NaN（停牌/退市）→ 该股返回 NaN（旧行为返回 0.0，误导因子）
    - 窗口内任一日出现异常单日跳变（|ret| > limit，默认 ±50%）或无效价
      → 窗口收益判定不可信，该股返回 NaN，避免异常跳变污染动量/反转因子
    D4收敛登记: 独特工具保留
    """
    out = {}
    for sym, df in data.items():
        if df is None or len(df) <= window or "close" not in df.columns:
            continue
        close = df["close"].astype(float)
        base = close.iloc[-window - 1]
        if not np.isfinite(base) or base <= 0:
            continue  # 基期无效 → NaN 而非 0/±inf
        win = safe_returns(close.iloc[-window - 1:], limit=limit)
        if win.iloc[1:].isna().any():
            continue  # 窗口内有异常跳变/无效价 → 该窗口收益不可信
        out[sym] = float(close.iloc[-1] / base - 1)
    return pd.Series(out, dtype=float)


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    """D4收敛登记: 独特工具保留"""
    series = series.dropna()
    return float(series.iloc[-1]) if len(series) else default



def common_dates(panels: dict[str, pd.DataFrame]) -> pd.Index:
    """D4收敛登记: 已收敛至唯一真源(原composite.py/neutralize.py重复实现已收敛至此)"""
    if not panels:
        return pd.Index([])
    idx = None
    for p in panels.values():
        idx = p.index if idx is None else idx.intersection(p.index)
    return idx
