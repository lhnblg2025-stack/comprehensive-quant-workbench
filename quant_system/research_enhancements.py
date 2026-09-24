"""Research enhancements: exposure neutralization and signal/execution reconciliation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_factor(frame: pd.DataFrame, factor: str, *, industry_col: str | None = "industry", size_col: str | None = "market_cap") -> pd.Series:
    if factor not in frame:
        raise ValueError(f"factor not found: {factor}")
    y = pd.to_numeric(frame[factor], errors="coerce")
    columns = []
    if size_col and size_col in frame:
        size = np.log(pd.to_numeric(frame[size_col], errors="coerce").clip(lower=1.0))
        columns.append(size.rename("log_size"))
    if industry_col and industry_col in frame:
        columns.append(pd.get_dummies(frame[industry_col].astype(str), prefix="industry", drop_first=True, dtype=float))
    if not columns:
        return y - y.mean()
    design = pd.concat(columns, axis=1); valid = y.notna() & design.notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, name=f"{factor}_neutral")
    if valid.sum() <= design.shape[1] + 2:
        return result
    x = np.column_stack([np.ones(valid.sum()), design.loc[valid].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, y.loc[valid].to_numpy(dtype=float), rcond=None)
    result.loc[valid] = y.loc[valid].to_numpy(dtype=float) - x @ beta
    return result


def reconcile_signal_execution(signal_returns: pd.Series, execution_equity: pd.Series, *, initial_capital: float, explicit_costs: float = 0.0, tolerance: float = 0.03) -> dict:
    signal = pd.to_numeric(signal_returns, errors="coerce").dropna()
    equity = pd.to_numeric(execution_equity, errors="coerce").dropna()
    signal_return = float((1.0 + signal).prod() - 1.0) if len(signal) else 0.0
    execution_return = float(equity.iloc[-1] / initial_capital - 1.0) if len(equity) else 0.0
    total_gap = execution_return - signal_return
    cost_drag = explicit_costs / initial_capital
    unexplained_gap = total_gap + cost_drag
    return {"signal_return": signal_return, "execution_return": execution_return, "total_gap": total_gap, "explicit_cost_drag": -cost_drag, "unexplained_gap": unexplained_gap, "within_tolerance": abs(unexplained_gap) <= tolerance, "tolerance": tolerance}


def compare_benchmarks(strategy: pd.Series, benchmarks: dict[str, pd.Series], annualization: float = 252.0) -> dict:
    result = {}
    for name, benchmark in benchmarks.items():
        aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
        excess = aligned.strategy - aligned.benchmark
        std = excess.std(ddof=1)
        result[name] = {"observations": len(aligned), "excess_total": float((1 + excess).prod() - 1) if len(excess) else None, "information_ratio": float(excess.mean() / std * np.sqrt(annualization)) if len(excess) > 1 and std > 0 else None}
    return result
