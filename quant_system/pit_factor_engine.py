"""Point-in-time factor construction after an as-of join.

The functions deliberately require an ``available_date`` column.  Raw files
under ``data_warehouse/financial`` are snapshots without announcement dates;
they must first pass ``pit_fundamentals.attach_pit_fundamentals`` and cannot be
used directly by this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _rank(frame: pd.DataFrame, value: pd.Series, ascending: bool = True) -> pd.Series:
    return value.groupby(frame["date"], sort=False).rank(pct=True, ascending=ascending, method="average")


def build_pit_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Build cross-sectional factors; fail closed when PIT provenance is absent."""
    required = {"date", "code", "available_date"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"pit_factor_input_missing:{','.join(sorted(missing))}")
    out = frame.copy(); out["date"] = pd.to_datetime(out["date"]); out["available_date"] = pd.to_datetime(out["available_date"])
    if out[["date", "available_date"]].isna().any().any():
        raise ValueError("missing_pit_dates")
    if out.duplicated(["date", "code"]).any():
        raise ValueError("duplicate_pit_keys")
    if (out["available_date"] > out["date"]).any(): raise ValueError("future_fundamental_available_date")
    # Value: lower multiples are attractive. Missing/negative multiples remain unavailable.
    pe, pb, ps = _num(out, "pe_ttm"), _num(out, "pb"), _num(out, "ps")
    # ``_rank(..., ascending=False)`` makes the smallest multiple receive the
    # highest percentile, so cheaper names receive the more attractive score.
    out["factor_value"] = (_rank(out, pe.where(pe > 0), False) + _rank(out, pb.where(pb > 0), False) + _rank(out, ps.where(ps > 0), False)) / 3
    # Quality and growth: robust rank blends, with explicit source columns.
    roe = _num(out, "净资产收益率(%)"); cash = _num(out, "资产的经营现金流量回报率(%)"); growth = _num(out, "净利润增长率(%)")
    out["factor_quality"] = (_rank(out, roe) + _rank(out, cash)) / 2
    out["factor_growth"] = _rank(out, growth)
    debt = _num(out, "资产负债率(%)"); liquidity = _num(out, "流动比率")
    out["factor_safety"] = (_rank(out, debt, False) + _rank(out, liquidity)) / 2
    # Repeated as-of rows are not new releases. This is EPS release change,
    # not analyst revision or a surprise relative to market expectations.
    releases = out[["code", "available_date"]].copy()
    releases["_eps"] = _num(out, "加权每股收益(元)")
    if releases.groupby(["code", "available_date"])["_eps"].nunique().gt(1).any():
        raise ValueError("conflicting_eps_release")
    releases = releases.drop_duplicates(["code", "available_date"]).sort_values(["code", "available_date"])
    prior = releases.groupby("code")["_eps"].shift()
    releases["_revision"] = (releases["_eps"] - prior) / prior.abs().where(prior != 0)
    lookup = releases.set_index(["code", "available_date"])["_revision"]
    out["factor_earnings_revision"] = lookup.reindex(pd.MultiIndex.from_frame(out[["code", "available_date"]])).to_numpy()
    return out


def neutralize_by_group(frame: pd.DataFrame, factor: str, *, group_col: str = "industry", size_col: str = "total_mv") -> pd.Series:
    """Date-wise industry and log-size residual, without future information."""
    if factor not in frame: raise ValueError(f"missing_factor:{factor}")
    result = pd.Series(np.nan, index=frame.index, dtype=float, name=f"{factor}_neutral")
    y = pd.to_numeric(frame[factor], errors="coerce")
    size = np.log(_num(frame, size_col).clip(lower=1.0))
    for date, idx in frame.groupby("date", sort=False).groups.items():
        valid = y.loc[idx].notna() & size.loc[idx].notna()
        if group_col in frame: valid &= frame.loc[idx, group_col].notna()
        if valid.sum() < 5: continue
        g = pd.get_dummies(frame.loc[idx[valid], group_col].astype(str), dtype=float) if group_col in frame else pd.DataFrame(index=idx[valid])
        x = np.column_stack([np.ones(valid.sum()), size.loc[idx[valid]].to_numpy(), g.to_numpy()])
        beta, *_ = np.linalg.lstsq(x, y.loc[idx[valid]].to_numpy(), rcond=None)
        result.loc[idx[valid]] = y.loc[idx[valid]].to_numpy() - x @ beta
    return result


def market_regime_filter(index_returns: pd.Series, *, trend_window: int = 200, vol_window: int = 20) -> pd.Series:
    """Binary state gate: trend above long average and volatility not extreme."""
    r = pd.Series(index_returns, dtype=float); level = (1+r).cumprod(); ma = level.rolling(trend_window, min_periods=max(20, trend_window//4)).mean(); vol = r.rolling(vol_window, min_periods=max(5, vol_window//2)).std()
    threshold = vol.rolling(252, min_periods=30).quantile(.8)
    return ((level > ma) & ((threshold.isna()) | (vol <= threshold))).astype(float).rename("regime_trend_filter")
