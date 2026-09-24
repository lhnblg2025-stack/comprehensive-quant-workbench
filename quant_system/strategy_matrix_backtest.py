"""Price/volume strategy matrix with execution-valid A-share accounting.

The module separates three price notions:

* ``hfq_*`` -- total-return research prices used for signal construction and
  for marking a position through dividends and splits.
* ``raw_*`` -- genuine unadjusted prices used for lot sizing, order notional,
  cash accounting and the A-share trade-state gates.
* ``base_adj`` -- a per-position basis: ``mark_value = base_adj * hfq_close``.
  ``base_adj`` is set to ``raw_entry_notional / hfq_open`` when a position is
  opened, so the raw price scale drives share counts while the HFQ ratio drives
  the total-return economics.

At signal date ``t`` the engine estimates the expected return of each signal
decile from forward returns that were already fully realised before ``t`` (a
shift by the holding horizon), then executes at the ``t + 1`` open.  Capital is
allocated in proportion to the expected excess return of the eligible names; it
is *not* equal weighted.  Names whose expected edge disappears are sold.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


FREQUENCY_HORIZON: dict[str, int] = {"daily": 1, "weekly": 5, "monthly": 21}
HORIZONS: tuple[int, ...] = (1, 5, 21)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    family: str
    signal_column: str
    formula: str
    note: str = ""


@dataclass(frozen=True)
class MatrixConfig:
    initial_capital: float = 1_000_000.0
    cash_buffer: float = 0.02
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    max_adv_participation: float = 0.10
    lot_size: int = 100
    min_commission: float = 5.0
    max_weight: float = 0.05
    min_expected_return: float = 0.0
    max_names: int = 50
    calibration_min_periods: int = 252


# Fifteen deliberately distinct price/volume strategy kinds.  Every signal
# column is already oriented so that a larger value means "more attractive".
STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec("mom_5", "5日动量", "momentum", "mom_5", "close / close.shift(5) - 1"),
    StrategySpec("mom_20", "20日动量", "momentum", "mom_20", "close / close.shift(20) - 1"),
    StrategySpec("mom_60", "60日动量", "momentum", "mom_60", "close / close.shift(60) - 1"),
    StrategySpec("mom_120", "120日动量", "momentum", "mom_120", "close / close.shift(120) - 1"),
    StrategySpec("rev_5", "5日反转", "reversal", "rev_5", "-(close / close.shift(5) - 1)"),
    StrategySpec("ma_5_20", "5/20均线趋势", "trend", "ma_5_20", "MA5 / MA20 - 1"),
    StrategySpec("ma_20_60", "20/60均线趋势", "trend", "ma_20_60", "MA20 / MA60 - 1"),
    StrategySpec("breakout_20", "20日通道突破", "breakout", "breakout_20", "close / prior20_high - 1"),
    StrategySpec("dist_52w", "52周高点距离", "breakout", "dist_52w", "close / rolling252_high - 1"),
    StrategySpec("rsi_rev_14", "RSI(14)反转", "oscillator", "rsi_rev_14", "(50 - RSI14) / 50"),
    StrategySpec("boll_rev_20", "布林带反转", "oscillator", "boll_rev_20", "-(close - MA20) / (2 * std20)"),
    StrategySpec("low_vol_20", "20日低波动", "volatility", "low_vol_20", "-std(ret1, 20)"),
    StrategySpec("volume_surge_20", "20日放量", "volume", "volume_surge_20", "volume / MA20(volume) - 1"),
    StrategySpec("macd_hist", "MACD柱线", "trend", "macd_hist", "EMA12 - EMA26 - EMA9(EMA12 - EMA26)"),
    StrategySpec("lottery_rev_20", "20日极值反转", "lottery", "lottery_rev_20", "-max(ret1, 20)"),
)

STRATEGY_BY_KEY = {spec.key: spec for spec in STRATEGIES}


def strategy_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in STRATEGIES)


def _rolling_by_code(data: pd.DataFrame, column: str, window: int, min_periods: int, method: str) -> pd.Series:
    grouped = data.groupby("code", sort=False)[column].rolling(window, min_periods=min_periods)
    result = getattr(grouped, method)()
    return result.reset_index(level=0, drop=True).reindex(data.index)


def _ewm_by_code(data: pd.DataFrame, column: str, **kwargs: Any) -> pd.Series:
    grouped = data.groupby("code", sort=False)[column].ewm(**kwargs)
    result = grouped.mean()
    return result.reset_index(level=0, drop=True).reindex(data.index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def build_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute all strategy signals, execution gates and forward labels."""
    required = {
        "date", "code", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
        "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("panel_missing_columns:" + ",".join(missing))

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)
    numeric = [
        "hfq_open", "hfq_high", "hfq_low", "hfq_close",
        "raw_open", "raw_high", "raw_low", "raw_close",
        "volume", "amount",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    grouped = data.groupby("code", sort=False)
    hfq_close = data["hfq_close"]
    data["ret_1d"] = grouped["hfq_close"].pct_change()

    for window in (5, 20, 60, 120):
        data[f"mom_{window}"] = grouped["hfq_close"].pct_change(window)
    data["rev_5"] = -data["mom_5"]

    for window in (5, 20, 60):
        data[f"ma_{window}"] = _rolling_by_code(data, "hfq_close", window, window, "mean")
    data["ma_5_20"] = _safe_divide(data["ma_5"], data["ma_20"]) - 1.0
    data["ma_20_60"] = _safe_divide(data["ma_20"], data["ma_60"]) - 1.0

    prior_high_20 = _rolling_by_code(data, "hfq_high", 20, 20, "max").groupby(data["code"], sort=False).shift(1)
    data["breakout_20"] = _safe_divide(hfq_close, prior_high_20) - 1.0
    high_252 = _rolling_by_code(data, "hfq_close", 252, 120, "max")
    data["dist_52w"] = _safe_divide(hfq_close, high_252) - 1.0

    delta = grouped["hfq_close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _ewm_by_code(data.assign(_gain=gain), "_gain", alpha=1.0 / 14.0, min_periods=14, adjust=False)
    avg_loss = _ewm_by_code(data.assign(_loss=loss), "_loss", alpha=1.0 / 14.0, min_periods=14, adjust=False)
    rs = _safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss.gt(0.0), 100.0)
    rsi = rsi.where(avg_gain.gt(0.0), 0.0)
    data["rsi_14"] = rsi
    data["rsi_rev_14"] = (50.0 - rsi) / 50.0

    std_20 = _rolling_by_code(data, "hfq_close", 20, 20, "std")
    data["boll_rev_20"] = -_safe_divide(hfq_close - data["ma_20"], 2.0 * std_20)

    data["vol_20"] = _rolling_by_code(data, "ret_1d", 20, 20, "std")
    data["low_vol_20"] = -data["vol_20"]

    volume_ma_20 = _rolling_by_code(data, "volume", 20, 20, "mean")
    data["volume_surge_20"] = _safe_divide(data["volume"], volume_ma_20) - 1.0

    ema12 = _ewm_by_code(data, "hfq_close", span=12, min_periods=12, adjust=False)
    ema26 = _ewm_by_code(data, "hfq_close", span=26, min_periods=26, adjust=False)
    macd = ema12 - ema26
    macd_signal = _ewm_by_code(data.assign(_macd=macd), "_macd", span=9, min_periods=9, adjust=False)
    data["macd_hist"] = macd - macd_signal

    max_ret_20 = _rolling_by_code(data, "ret_1d", 20, 20, "max")
    data["lottery_rev_20"] = -max_ret_20

    # Execution gates.  ``tradable`` is deliberately conservative: a missing
    # quote, zero volume or an explicit suspended flag all block the order.
    data["raw_prev_close"] = grouped["raw_close"].shift(1)
    explicit_tradable = pd.Series(True, index=data.index)
    if "is_tradable" in data.columns:
        explicit_tradable &= data["is_tradable"].fillna(False).astype(bool)
    if "is_suspended" in data.columns:
        explicit_tradable &= ~data["is_suspended"].fillna(True).astype(bool)
    quote_ok = (
        data["raw_open"].gt(0.0)
        & data["hfq_open"].gt(0.0)
        & data["hfq_close"].gt(0.0)
        & data["volume"].gt(0.0)
    )
    data["tradable"] = (explicit_tradable & quote_ok).fillna(False)
    data["suspended"] = ~data["tradable"]
    # 9.5% band proxy for the main board; the run is research grade and the
    # production gate stays BLOCK until an authoritative PIT status feed lands.
    data["limit_up_locked"] = (data["raw_open"] >= data["raw_prev_close"] * 1.095).fillna(False)
    data["limit_down_locked"] = (data["raw_open"] <= data["raw_prev_close"] * 0.905).fillna(False)
    data["amount_ma_20"] = _rolling_by_code(data, "amount", 20, 20, "mean")
    for horizon in HORIZONS:
        data[f"fwd_ret_{horizon}"] = grouped["hfq_close"].shift(-horizon) / hfq_close - 1.0
        # Calendar date when this forward label is fully realised.  Calibration
        # is gated on this date, including when an individual name skips days.
        data[f"fwd_realized_{horizon}"] = grouped["date"].shift(-horizon)
    return data


def signal_percentile(features: pd.DataFrame, signal_column: str) -> pd.Series:
    """Per-date cross-sectional percentile rank (0 = weakest)."""
    return features.groupby("date", sort=False)[signal_column].rank(pct=True, method="average")


def add_signal_bucket(features: pd.DataFrame, signal_column: str) -> pd.Series:
    """Return a per-date decile bucket (0 = weakest, 9 = strongest)."""
    percentile = signal_percentile(features, signal_column)
    return np.floor(percentile.clip(lower=0.0, upper=0.999999) * 10.0).astype("float64")


def build_trailing_decile_table(
    features: pd.DataFrame,
    bucket: pd.Series,
    horizon: int,
    min_periods: int = 252,
    n_buckets: int = 10,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    """Trailing decile means using only labels realised by each decision date."""
    target_column = f"fwd_ret_{horizon}"
    realized_column = f"fwd_realized_{horizon}"
    if target_column not in features.columns or realized_column not in features.columns:
        raise ValueError(f"missing_target:{target_column}")
    work = pd.DataFrame({
        "date": pd.to_datetime(features["date"]).to_numpy(),
        "bucket": bucket.to_numpy(dtype=float),
        "target": pd.to_numeric(features[target_column], errors="coerce").to_numpy(dtype=float),
        "realized": pd.to_datetime(features[realized_column]).to_numpy(),
    })
    clean = work.dropna(subset=["target", "bucket", "realized"])
    clean = clean[clean["bucket"].between(0, n_buckets - 1)]
    if clean.empty:
        raise ValueError("empty_calibration_rows")
    daily = clean.groupby(["date", "bucket"], sort=False).agg(
        target_mean=("target", "mean"), realized=("realized", "max")
    ).reset_index()
    dates = pd.DatetimeIndex(sorted(work["date"].unique()))
    decision_dates = dates.values.astype("datetime64[ns]")
    table = np.full((len(dates), n_buckets), np.nan, dtype=float)
    groups_per_bucket: dict[str, int] = {}
    leakage_violations = 0
    for b in range(n_buckets):
        block = daily[daily["bucket"] == b].sort_values("realized")
        groups_per_bucket[str(b)] = int(len(block))
        if block.empty:
            continue
        by_realized = block.groupby("realized", sort=True)["target_mean"].mean()
        expanding = by_realized.expanding(min_periods=min_periods).mean()
        realized_dates = by_realized.index.values.astype("datetime64[ns]")
        values = expanding.to_numpy(dtype=float)
        pos = np.searchsorted(realized_dates, decision_dates, side="right") - 1
        valid = pos >= 0
        column = np.full(len(dates), np.nan, dtype=float)
        if valid.any():
            safe_pos = np.where(valid, pos, 0)
            leakage_violations += int(np.sum(valid & (realized_dates[safe_pos] > decision_dates)))
            column[valid] = values[pos[valid]]
        table[:, b] = column
    diagnostics = {
        "target_column": target_column,
        "realized_date_column": realized_column,
        "horizon_sessions": int(horizon),
        "min_periods": int(min_periods),
        "n_buckets": int(n_buckets),
        "calibration_groups_per_bucket": groups_per_bucket,
        "mapping_dates": int(np.isfinite(table).any(axis=1).sum()),
        "label_leakage_violations": int(leakage_violations),
    }
    return table, dates, diagnostics


def compute_trailing_decile_expected_return(
    features: pd.DataFrame,
    bucket: pd.Series,
    horizon: int,
    min_periods: int = 252,
    percentile: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Interpolate trailing decile means across continuous signal strength."""
    n_buckets = 10
    table, dates, diagnostics = build_trailing_decile_table(
        features, bucket, horizon, min_periods=min_periods, n_buckets=n_buckets
    )
    all_dates = dates.values.astype("datetime64[ns]")
    row_dates = pd.to_datetime(features["date"]).values.astype("datetime64[ns]")
    date_pos = np.searchsorted(all_dates, row_dates, side="left")
    clipped_pos = np.clip(date_pos, 0, len(all_dates) - 1)
    date_pos = np.where((date_pos < len(all_dates)) & (all_dates[clipped_pos] == row_dates), date_pos, -1)
    if percentile is not None:
        x = np.clip(percentile.to_numpy(dtype=float) * n_buckets - 0.5, 0.0, n_buckets - 1.0)
    else:
        x = bucket.to_numpy(dtype=float)
    bucket_values = bucket.to_numpy(dtype=float)
    row_ok = (date_pos >= 0) & np.isfinite(bucket_values) & np.isfinite(x)
    lower = np.floor(np.where(row_ok, x, 0.0)).astype(int)
    upper = np.minimum(lower + 1, n_buckets - 1)
    weight = np.where(row_ok, x - lower, 0.0)
    expected = np.full(len(features), np.nan, dtype=float)
    if row_ok.any():
        idx = np.flatnonzero(row_ok)
        lo = table[date_pos[idx], lower[idx]]
        hi = table[date_pos[idx], upper[idx]]
        expected[idx] = lo * (1.0 - weight[idx]) + hi * weight[idx]
    series = pd.Series(expected, index=features.index, name="expected_ret")
    diagnostics["expected_non_null_rows"] = int(np.isfinite(expected).sum())
    diagnostics["expected_percentile_interpolated"] = bool(percentile is not None)
    return series, diagnostics


def rank_ic_series(frame: pd.DataFrame, signal_column: str, target_column: str) -> pd.Series:
    """Vectorised per-date Spearman rank IC."""
    sample = frame[["date", signal_column, target_column]].dropna().copy()
    if sample.empty:
        return pd.Series(dtype=float)
    sample["_rx"] = sample.groupby("date", sort=False)[signal_column].rank(method="average")
    sample["_ry"] = sample.groupby("date", sort=False)[target_column].rank(method="average")
    sample["_rx2"] = sample["_rx"] ** 2
    sample["_ry2"] = sample["_ry"] ** 2
    sample["_rxy"] = sample["_rx"] * sample["_ry"]
    grouped = sample.groupby("date", sort=False)
    n = grouped.size().astype(float)
    sx = grouped["_rx"].sum()
    sy = grouped["_ry"].sum()
    sxx = grouped["_rx2"].sum()
    syy = grouped["_ry2"].sum()
    sxy = grouped["_rxy"].sum()
    numerator = n * sxy - sx * sy
    denominator = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2)).replace(0.0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).dropna()


def cross_section_stats(
    frame: pd.DataFrame,
    signal_column: str,
    horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Rank IC and quantile spreads for one strategy/horizon/window."""
    target = f"fwd_ret_{horizon}"
    sample = frame[["date", signal_column, target]].dropna().copy()
    sample = sample[sample["date"].between(start, end, inclusive="both")]
    if sample.empty:
        return {"observations": 0}
    ic = rank_ic_series(sample, signal_column, target)
    sample["_pct"] = sample.groupby("date", sort=False)[signal_column].rank(pct=True, method="average")
    sample["_q5"] = np.minimum(np.floor(sample["_pct"].clip(upper=0.999999) * 5.0), 4.0)
    sample["_d10"] = np.minimum(np.floor(sample["_pct"] * 10.0), 9.0)
    q5 = sample.groupby(["date", "_q5"], sort=False)[target].mean().unstack("_q5")
    d10 = sample.groupby(["date", "_d10"], sort=False)[target].mean().unstack("_d10")
    spread_q5 = (q5[4.0] - q5[0.0]).dropna() if {4.0, 0.0} <= set(q5.columns) else pd.Series(dtype=float)
    spread_d10 = (d10[9.0] - d10[0.0]).dropna() if {9.0, 0.0} <= set(d10.columns) else pd.Series(dtype=float)
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    return {
        "observations": int(len(sample)),
        "dates": int(sample["date"].nunique()),
        "rank_ic_mean": float(ic.mean()) if len(ic) else None,
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(ic.mean() / ic_std) if len(ic) and ic_std > 0 else None,
        "rank_ic_positive_ratio": float((ic > 0).mean()) if len(ic) else None,
        "top_bottom_decile_spread": float(spread_d10.mean()) if len(spread_d10) else None,
        "top_bottom_quintile_spread": float(spread_q5.mean()) if len(spread_q5) else None,
        "quintile_spread_hit_ratio": float((spread_q5 > 0).mean()) if len(spread_q5) else None,
    }


def _fee(notional: float, side: str, config: MatrixConfig) -> float:
    if notional <= 0:
        return 0.0
    commission = max(config.min_commission, notional * config.commission_bps / 10000.0)
    transfer = notional * config.transfer_fee_bps / 10000.0
    stamp = notional * config.stamp_duty_bps / 10000.0 if side == "sell" else 0.0
    return float(commission + transfer + stamp)


def _floor_lot(notional: float, price: float, lot_size: int) -> int:
    if not np.isfinite(notional) or not np.isfinite(price) or notional <= 0 or price <= 0:
        return 0
    return max(0, int(notional // (price * lot_size)) * lot_size)


def _cap_weights(excess: np.ndarray, valid: np.ndarray, max_weight: float) -> np.ndarray:
    """Proportional-to-excess weights with an iterative per-name cap."""
    raw = np.where(valid, np.maximum(excess, 0.0), 0.0)
    total = float(raw.sum())
    if total <= 0:
        return np.zeros_like(raw)
    weights = raw / total
    for _ in range(50):
        over = weights > max_weight + 1e-15
        if not over.any():
            break
        spare = float((weights[over] - max_weight).sum())
        weights[over] = max_weight
        under = (~over) & (weights > 0)
        if not under.any():
            break
        under_sum = float(weights[under].sum())
        if under_sum <= 0:
            break
        weights[under] += spare * weights[under] / under_sum
    return weights


def _rebalance_date_set(dates: Iterable[pd.Timestamp], frequency: str) -> set[pd.Timestamp]:
    index = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    if index.empty:
        return set()
    if frequency == "daily":
        return set(index)
    if frequency == "weekly":
        return set(pd.Series(index, index=index).groupby(index.to_period("W")).max())
    if frequency == "monthly":
        return set(pd.Series(index, index=index).groupby(index.to_period("M")).max())
    if frequency == "5d":
        return set(index[::5])
    raise ValueError("unknown_frequency:" + frequency)


def annualized_metrics(values: pd.Series, annualization: float = 252.0) -> dict[str, Any]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    if returns.empty:
        return {
            "observations": 0, "total_return": None, "annual_return": None,
            "annual_volatility": None, "sharpe": None, "max_drawdown": None,
        }
    curve = (1.0 + returns).cumprod()
    years = len(returns) / annualization
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    drawdown = curve / curve.cummax() - 1.0
    return {
        "observations": int(len(returns)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and curve.iloc[-1] > 0 else None,
        "annual_volatility": float(std * math.sqrt(annualization)),
        "sharpe": float(returns.mean() / std * math.sqrt(annualization)) if std > 0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def _pivot(frame: pd.DataFrame, column: str, dates: pd.DatetimeIndex, codes: list[str],
           ffill: bool = False, fill: Any = None, dtype: Any = float) -> np.ndarray:
    matrix = frame.pivot(index="date", columns="code", values=column).reindex(index=dates, columns=codes)
    if ffill:
        matrix = matrix.ffill()
    if fill is not None:
        matrix = matrix.fillna(fill)
    return matrix.to_numpy(dtype=dtype)


def simulate_portfolio(
    frame: pd.DataFrame,
    config: MatrixConfig,
    frequency: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run one strategy/frequency portfolio with full trade audit fields."""
    required = {
        "date", "code", "expected_ret", "raw_open", "hfq_open", "hfq_close",
        "amount_ma_20", "tradable", "limit_up_locked", "limit_down_locked",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("simulation_missing_columns:" + ",".join(missing))
    if frequency not in FREQUENCY_HORIZON:
        raise ValueError("unknown_frequency:" + frequency)

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    for column in ("expected_ret", "raw_open", "hfq_open", "hfq_close", "amount_ma_20"):
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    if start is not None:
        data = data[data["date"] >= pd.Timestamp(start)]
    if end is not None:
        data = data[data["date"] <= pd.Timestamp(end)]
    data = data.dropna(subset=["date", "code"]).sort_values(["date", "code"])
    if data.empty:
        raise ValueError("empty_simulation_window")
    for flag in ("tradable", "limit_up_locked", "limit_down_locked"):
        data[flag] = pd.to_numeric(data[flag], errors="coerce").fillna(0.0).astype(float)
    has_select_score = "select_score" in data.columns
    if has_select_score:
        data["select_score"] = pd.to_numeric(data["select_score"], errors="coerce").astype(float)

    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    codes = sorted(data["code"].unique())
    if len(dates) < 3:
        raise ValueError("insufficient_simulation_dates")
    code_position = {code: position for position, code in enumerate(codes)}

    expected = _pivot(data, "expected_ret", dates, codes)
    raw_open = _pivot(data, "raw_open", dates, codes)
    hfq_open = _pivot(data, "hfq_open", dates, codes)
    hfq_close = _pivot(data, "hfq_close", dates, codes, ffill=True)
    adv = _pivot(data, "amount_ma_20", dates, codes, fill=0.0)
    tradable = _pivot(data, "tradable", dates, codes, fill=False, dtype=bool)
    limit_up = _pivot(data, "limit_up_locked", dates, codes, fill=False, dtype=bool)
    limit_down = _pivot(data, "limit_down_locked", dates, codes, fill=False, dtype=bool)
    select_score = (
        _pivot(data, "select_score", dates, codes)
        if has_select_score
        else None
    )

    rebalance_dates = _rebalance_date_set(dates, frequency)
    slip_buy = 1.0 + config.slippage_bps / 10000.0
    slip_sell = 1.0 - config.slippage_bps / 10000.0
    cash = float(config.initial_capital)
    positions: dict[str, dict[str, float]] = {}
    pending: dict[str, float] | None = None
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    stale_codes_rows: list[set] = []
    blocked: dict[str, int] = {}
    blocked_sell_dates: set = set()
    blocked_sell_codes: set = set()
    active_target: set = set()
    cursor: dict[str, Any] = {"date": None}

    def note(reason: str, code: str | None = None) -> None:
        blocked[reason] = blocked.get(reason, 0) + 1
        day = cursor["date"]
        if code is not None and reason.startswith("sell"):
            # A holding that is no longer in the target book can only survive
            # if at least one exit attempt for it was blocked.
            blocked_sell_dates.add(day)
            blocked_sell_codes.add(code)

    for i, date in enumerate(dates):
        cursor["date"] = date
        if pending is not None:
            open_value = 0.0
            for code, position in positions.items():
                price = hfq_open[i, code_position[code]]
                if not np.isfinite(price):
                    price = hfq_close[i, code_position[code]]
                if np.isfinite(price):
                    open_value += position["base_adj"] * price
            investable = max(0.0, (cash + open_value) * (1.0 - config.cash_buffer))
            targets = {code: investable * weight for code, weight in pending.items()}
            target_weights = dict(pending)
            pending = None

            # Sell legs first so sale proceeds can fund the buy legs.
            for code in list(positions):
                j = code_position[code]
                current = int(positions[code]["shares"])
                price_raw = raw_open[i, j]
                target_value = targets.get(code, 0.0)
                if not np.isfinite(price_raw) or price_raw <= 0:
                    note("sell_no_quote", code)
                    continue
                desired = _floor_lot(target_value, price_raw, config.lot_size)
                if desired >= current:
                    continue
                if not bool(tradable[i, j]):
                    note("sell_not_tradable", code)
                    continue
                if bool(limit_down[i, j]):
                    note("sell_limit_down", code)
                    continue
                available_adv = float(adv[i, j])
                if not np.isfinite(available_adv) or available_adv <= 0:
                    note("sell_no_adv", code)
                    continue
                sell_shares = current - desired
                cap = int(available_adv * config.max_adv_participation / price_raw // config.lot_size) * config.lot_size
                sell_shares = int(min(sell_shares, max(cap, 0)))
                if sell_shares <= 0:
                    note("sell_adv_cap", code)
                    continue
                price_hfq = hfq_open[i, j]
                if not np.isfinite(price_hfq):
                    price_hfq = hfq_close[i, j]
                if not np.isfinite(price_hfq) or price_hfq <= 0:
                    note("sell_no_hfq", code)
                    continue
                fraction = sell_shares / current
                traded_value = positions[code]["base_adj"] * fraction * price_hfq
                raw_notional = sell_shares * price_raw
                fee = _fee(raw_notional, "sell", config)
                cash += traded_value * slip_sell - fee
                positions[code]["base_adj"] *= (1.0 - fraction)
                positions[code]["shares"] = current - sell_shares
                if positions[code]["shares"] <= 0 or positions[code]["base_adj"] <= 1e-12:
                    del positions[code]
                trade_rows.append({
                    "date": date, "code": code, "side": "sell", "shares": int(sell_shares),
                    "raw_price": float(price_raw), "hfq_price": float(price_hfq),
                    "raw_notional": float(raw_notional), "traded_value": float(traded_value),
                    "fee": float(fee), "slippage_cost": float(traded_value * (config.slippage_bps / 10000.0)),
                    "target_value": float(target_value), "target_weight": float(target_weights.get(code, 0.0)),
                    "blocked": False, "block_reason": "",
                })

            # Buy legs, largest target first, sized against live cash.
            for code, target_value in sorted(targets.items(), key=lambda item: item[1], reverse=True):
                if target_value <= 0:
                    continue
                j = code_position[code]
                price_raw = raw_open[i, j]
                if not np.isfinite(price_raw) or price_raw <= 0:
                    note("buy_no_quote")
                    continue
                if not bool(tradable[i, j]):
                    note("buy_not_tradable")
                    continue
                if bool(limit_up[i, j]):
                    note("buy_limit_up")
                    continue
                price_hfq = hfq_open[i, j]
                if not np.isfinite(price_hfq) or price_hfq <= 0:
                    note("buy_no_hfq")
                    continue
                price = float(price_raw) * slip_buy
                current = int(positions.get(code, {}).get("shares", 0))
                desired = _floor_lot(target_value, price, config.lot_size)
                delta = desired - current
                if delta <= 0:
                    continue
                available_adv = float(adv[i, j])
                if not np.isfinite(available_adv) or available_adv <= 0:
                    note("buy_no_adv")
                    continue
                cap = int(available_adv * config.max_adv_participation / price // config.lot_size) * config.lot_size
                if cap <= 0:
                    note("buy_adv_cap")
                    continue
                delta = int(min(delta, cap))
                while delta > 0:
                    gross = delta * price
                    if gross + _fee(gross, "buy", config) <= cash:
                        break
                    delta -= config.lot_size
                if delta <= 0:
                    note("buy_insufficient_cash")
                    continue
                gross = delta * price
                fee = _fee(gross, "buy", config)
                cash -= gross + fee
                position = positions.setdefault(code, {"shares": 0.0, "base_adj": 0.0})
                position["shares"] += delta
                # Basis uses the unslipped raw consideration: the slipped
                # price is what cash actually pays, so the slippage shows up
                # immediately as a loss instead of inflating the position.
                position["base_adj"] += (delta * float(price_raw)) / float(price_hfq)
                trade_rows.append({
                    "date": date, "code": code, "side": "buy", "shares": int(delta),
                    "raw_price": float(price_raw), "hfq_price": float(price_hfq),
                    "raw_notional": float(delta * price_raw), "traded_value": float(gross),
                    "fee": float(fee), "slippage_cost": float(gross - delta * price_raw),
                    "target_value": float(target_value), "target_weight": float(target_weights.get(code, 0.0)),
                    "blocked": False, "block_reason": "",
                })

        position_value = 0.0
        for code, position in positions.items():
            price = hfq_close[i, code_position[code]]
            if np.isfinite(price):
                position_value += position["base_adj"] * price
        total_value = cash + position_value
        stale_codes_rows.append(set(positions) - active_target)
        equity_rows.append({
            "date": date, "cash": float(cash), "position_value": float(position_value),
            "total_value": float(total_value), "positions": int(len(positions)),
            "exposure": float(position_value / total_value) if total_value else 0.0,
        })

        if date in rebalance_dates and i + 1 < len(dates):
            expected_row = expected[i]
            valid = np.isfinite(expected_row) & (expected_row > config.min_expected_return)
            if valid.any() and int(config.max_names) > 0:
                candidates = np.flatnonzero(valid)
                if len(candidates) > int(config.max_names):
                    expected_candidates = expected_row[candidates]
                    if select_score is not None:
                        score_candidates = np.nan_to_num(
                            select_score[i, candidates], nan=-np.inf, posinf=-np.inf
                        )
                        order = np.lexsort((-score_candidates, -expected_candidates))
                    else:
                        order = np.argsort(-expected_candidates, kind="stable")
                    candidates = candidates[order[: int(config.max_names)]]
                keep = np.zeros(len(expected_row), dtype=bool)
                keep[candidates] = True
                valid = keep
            if valid.any():
                excess = expected_row[valid] - config.min_expected_return
                weights = _cap_weights(excess, np.ones_like(excess), config.max_weight)
                selected = np.flatnonzero(valid)
                pending = {codes[j]: float(weights[k]) for k, j in enumerate(selected) if weights[k] > 0}
            else:
                pending = {}
            active_target = set(pending)

    equity = pd.DataFrame(equity_rows)
    equity["date"] = pd.to_datetime(equity["date"])
    equity["return"] = equity["total_value"].pct_change().fillna(0.0)
    excess_rows = equity[equity["positions"] > int(config.max_names)]
    excess_days = int(len(excess_rows))
    excess_days_with_blocked_sell = int(sum(1 for day in excess_rows["date"] if day in blocked_sell_dates))
    stale_position_days = int(sum(1 for stale in stale_codes_rows if stale))
    unexplained_stale_positions = int(sum(len(stale - blocked_sell_codes) for stale in stale_codes_rows))
    trades = pd.DataFrame(trade_rows)
    metrics = annualized_metrics(equity["return"])
    total_fees = float(trades["fee"].sum()) if not trades.empty else 0.0
    traded_value_total = float(trades["traded_value"].sum()) if not trades.empty else 0.0
    average_equity = float(equity["total_value"].mean())
    blocked_total = int(sum(blocked.values()))
    non_lot = int((trades["shares"] % config.lot_size != 0).sum()) if not trades.empty else 0
    report = {
        "frequency": frequency,
        "horizon_sessions": FREQUENCY_HORIZON[frequency],
        "config": asdict(config),
        "metrics": metrics,
        "sessions": int(len(equity)),
        "rebalances": int(len([d for d in rebalance_dates if d <= dates[-2]])),
        "trades": int(len(trades)),
        "blocked_orders": blocked_total,
        "blocked_reasons": blocked,
        "total_fees": total_fees,
        "fee_drag_on_initial_capital": total_fees / float(config.initial_capital),
        "total_traded_value": traded_value_total,
        "turnover_multiple_on_initial_capital": traded_value_total / float(config.initial_capital),
        "turnover_multiple_on_average_equity": traded_value_total / average_equity if average_equity else None,
        "excess_position_days": excess_days,
        "excess_position_days_with_blocked_sell": excess_days_with_blocked_sell,
        "stale_position_days": stale_position_days,
        "unexplained_stale_positions": unexplained_stale_positions,
        "excess_positions_explained_by_blocked_sells": bool(unexplained_stale_positions == 0),
        "average_positions": float(equity["positions"].mean()),
        "max_positions": int(equity["positions"].max()),
        "final_positions": int(equity["positions"].iloc[-1]),
        "average_exposure": float(equity["exposure"].mean()),
        "average_cash_ratio": float((equity["cash"] / equity["total_value"]).mean()),
        "final_cash_ratio": float(equity["cash"].iloc[-1] / equity["total_value"].iloc[-1]) if equity["total_value"].iloc[-1] else None,
        "final_equity": float(equity["total_value"].iloc[-1]),
        "reconciliation": {
            "negative_cash_rows": int((equity["cash"] < -1e-6).sum()),
            "non_lot_trade_rows": non_lot,
            "trades_with_zero_shares": int((trades["shares"] <= 0).sum()) if not trades.empty else 0,
            "total_value_equals_cash_plus_positions": bool(
                np.allclose(equity["total_value"], equity["cash"] + equity["position_value"], atol=1e-6)
            ),
            "fees_nonnegative": bool((trades["fee"] >= 0).all()) if not trades.empty else True,
            "cash_plus_final_positions": float(
                equity["cash"].iloc[-1] + equity["position_value"].iloc[-1]
            ),
        },
    }
    return {"report": report, "equity": equity, "trades": trades}


def benchmark_metrics(path: str | Any, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].between(start, end, inclusive="both")].sort_values("date")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    returns = close.pct_change().dropna()
    metrics = annualized_metrics(returns)
    return {
        "source": str(path),
        **metrics,
        "date_min": str(frame["date"].min().date()) if len(frame) else None,
        "date_max": str(frame["date"].max().date()) if len(frame) else None,
    }
