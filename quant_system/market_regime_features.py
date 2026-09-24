"""Causal market-regime features derived from official SSE index daily bars."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("index_code", "index_name", "trade_date", "open", "high", "low", "close", "volume", "amount")


def load_index_daily(directory: str | Path) -> pd.DataFrame:
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no_index_parquet:{directory}")
    frames = [pd.read_parquet(path) for path in files]
    missing = sorted(set(REQUIRED_COLUMNS) - set(pd.concat(frames[:1], ignore_index=True).columns))
    if missing:
        raise ValueError(f"index_daily_missing_columns:{missing}")
    frame = pd.concat(frames, ignore_index=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["index_code"] = frame["index_code"].astype(str).str.zfill(6)
    frame = frame.dropna(subset=["trade_date", "index_code", "close"]).copy()
    if frame.duplicated(["index_code", "trade_date"]).any():
        raise ValueError("index_daily_duplicate_keys")
    frame = frame.sort_values(["index_code", "trade_date"])
    return frame.reset_index(drop=True)


def _style_regime(frame: pd.DataFrame) -> pd.Series:
    pivot = frame.pivot(index="trade_date", columns="index_code", values="return_20d")
    large = pivot.get("000300")
    small = pivot.get("000852")
    if large is None or small is None:
        return pd.Series("unavailable", index=frame["trade_date"])
    spread = small - large
    regime = pd.Series("neutral", index=spread.index, dtype="object")
    regime.loc[spread > 0.02] = "small_cap_lead"
    regime.loc[spread < -0.02] = "large_cap_lead"
    return frame["trade_date"].map(regime).fillna("unavailable")


def build_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using only each index's current and prior observations."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"index_daily_missing_columns:{missing}")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "index_code", "close"]).copy()
    data["index_code"] = data["index_code"].astype(str).str.zfill(6)
    data = data.sort_values(["index_code", "trade_date"]).drop_duplicates(["index_code", "trade_date"], keep="last")
    grouped = data.groupby("index_code", sort=False, group_keys=False)
    close = pd.to_numeric(data["close"], errors="coerce")
    daily_return = grouped["close"].pct_change()
    data["return_1d"] = daily_return
    for horizon in (5, 20, 60):
        data[f"return_{horizon}d"] = grouped["close"].pct_change(horizon)
    for window in (20, 60):
        data[f"volatility_{window}d"] = grouped["return_1d"].transform(lambda values: values.rolling(window, min_periods=window).std() * np.sqrt(252.0))
        data[f"ma_{window}d"] = grouped["close"].transform(lambda values: values.rolling(window, min_periods=window).mean())
    data["above_ma20"] = data["close"] > data["ma_20d"]
    data["above_ma60"] = data["close"] > data["ma_60d"]
    data["drawdown_60d"] = grouped["close"].transform(lambda values: values / values.rolling(60, min_periods=60).max() - 1.0)
    data["range_20d"] = grouped["high"].transform(lambda values: values.rolling(20, min_periods=20).max()) / grouped["low"].transform(
        lambda values: values.rolling(20, min_periods=20).min()
    ) - 1.0
    benchmark = data.loc[data["index_code"] == "000001", ["trade_date", "return_20d"]].rename(columns={"return_20d": "benchmark_return_20d"})
    data = data.merge(benchmark, on="trade_date", how="left")
    data["relative_strength_20d"] = data["return_20d"] - data["benchmark_return_20d"]
    data["style_regime"] = _style_regime(data)
    data["feature_asof"] = data["trade_date"]
    return data.sort_values(["trade_date", "index_code"]).reset_index(drop=True)


def build_from_directory(directory: str | Path) -> pd.DataFrame:
    return build_market_regime_features(load_index_daily(directory))


__all__ = ["build_from_directory", "build_market_regime_features", "load_index_daily"]
