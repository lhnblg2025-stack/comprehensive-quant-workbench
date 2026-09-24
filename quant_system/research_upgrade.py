"""Auditable research protocol for chronological A-share experiments.

The legacy engines remain untouched.  New studies can import this module to
enforce point-in-time boundaries, purged walk-forward folds, factor metadata,
and a common risk/robustness report.  The holdout cannot be used by selection
helpers before :func:`freeze` has been called.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchBoundaries:
    development_end: str = "2024-12-31"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-12-31"
    holdout_start: str = "2026-01-01"
    holdout_end: str | None = None
    embargo_days: int = 5
    label_horizon_days: int = 5

    def __post_init__(self) -> None:
        d, vs, ve, hs = map(pd.Timestamp, (self.development_end, self.validation_start, self.validation_end, self.holdout_start))
        he = pd.Timestamp(self.holdout_end) if self.holdout_end else None
        if not (d < vs <= ve < hs):
            raise ValueError("research_boundaries_must_be_strictly_chronological")
        if he is not None and he < hs:
            raise ValueError("holdout_end_before_holdout_start")
        if self.embargo_days < 0 or self.label_horizon_days < 1:
            raise ValueError("invalid_embargo_or_label_horizon")


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    direction: int = 1
    source: str = "price"
    availability_lag_days: int = 1
    requires_pit: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("factor_direction_must_be_minus_one_or_one")
        if self.availability_lag_days < 0:
            raise ValueError("availability_lag_must_be_nonnegative")


FACTOR_CATALOG: tuple[FactorSpec, ...] = (
    FactorSpec("value_composite", "value", -1, "pit_valuation", 1, True, "PE/PB/PS percentile composite"),
    FactorSpec("quality_roe", "quality", 1, "pit_financial", 1, True, "ROE with winsorisation"),
    FactorSpec("quality_cashflow", "quality", 1, "pit_financial", 1, True, "operating cash flow / assets"),
    FactorSpec("earnings_revision", "growth", 1, "pit_financial", 1, True, "point-in-time earnings surprise/revision"),
    FactorSpec("financial_safety", "safety", -1, "pit_financial", 1, True, "leverage and distress composite"),
    FactorSpec("industry_relative_momentum", "industry", 1, "price", 1, False, "stock momentum minus industry momentum"),
    FactorSpec("regime_trend_filter", "market_state", 1, "index", 1, False, "market trend/volatility state gate"),
    FactorSpec("fund_flow_persistence", "flow", 1, "fund_flow", 1, True, "persistent northbound/institutional flow"),
    FactorSpec("event_quality", "event", 1, "announcements", 1, True, "buyback/dividend/earnings events"),
    FactorSpec("residual_quality_momentum", "residual", 1, "price", 1, False, "industry and beta neutral momentum"),
)


def split_frame(frame: pd.DataFrame, boundaries: ResearchBoundaries, *, date_col: str = "date", partition: str = "development", allow_holdout: bool = False) -> pd.DataFrame:
    """Return one partition.

    The holdout branch is intentionally locked here as well as in
    :func:`evaluate_holdout`; callers must go through the latter so a frozen
    manifest is always recorded in the experiment artifacts.
    """
    if date_col not in frame:
        raise ValueError(f"missing_date_column:{date_col}")
    d = frame.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().any():
        raise ValueError("invalid_dates")
    if partition == "development":
        mask = d[date_col] <= pd.Timestamp(boundaries.development_end)
    elif partition == "validation":
        mask = d[date_col].between(boundaries.validation_start, boundaries.validation_end)
    elif partition == "holdout":
        if not allow_holdout:
            raise PermissionError("holdout_requires_evaluate_holdout_manifest")
        mask = d[date_col] >= pd.Timestamp(boundaries.holdout_start)
        if boundaries.holdout_end:
            mask &= d[date_col] <= pd.Timestamp(boundaries.holdout_end)
    else:
        raise ValueError("partition_must_be_development_validation_or_holdout")
    return d.loc[mask].copy()


def select_candidates(frame: pd.DataFrame, boundaries: ResearchBoundaries, *, score_col: str, n: int = 10, date_col: str = "date") -> pd.DataFrame:
    """Select candidates using development data only."""
    if n < 1:
        raise ValueError("n_must_be_positive")
    dev = split_frame(frame, boundaries, date_col=date_col, partition="development")
    if score_col not in dev:
        raise ValueError(f"missing_score_column:{score_col}")
    return dev.sort_values(score_col, ascending=False).head(n).copy()


def evaluate_holdout(frame: pd.DataFrame, boundaries: ResearchBoundaries, *, frozen_manifest: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Read holdout rows only when a valid frozen manifest exists."""
    manifest = Path(frozen_manifest)
    if not manifest.is_file():
        raise PermissionError("holdout_locked_until_freeze_manifest_exists")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("frozen") is not True or payload.get("boundaries", {}).get("holdout_start") != boundaries.holdout_start:
        raise PermissionError("invalid_or_mismatched_freeze_manifest")
    return split_frame(frame, boundaries, date_col=date_col, partition="holdout", allow_holdout=True)


def purged_folds(dates: Sequence[Any], boundaries: ResearchBoundaries, *, train_days: int = 504, test_days: int = 126, step_days: int = 126) -> list[dict[str, Any]]:
    """Create non-overlapping folds and remove the label horizon + embargo."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    dev = idx[idx <= pd.Timestamp(boundaries.development_end)]
    out: list[dict[str, Any]] = []
    start = train_days
    while start + boundaries.label_horizon_days + boundaries.embargo_days + test_days <= len(dev):
        train = dev[start - train_days:start]
        test_start = start + boundaries.label_horizon_days + boundaries.embargo_days
        test = dev[test_start:test_start + test_days]
        out.append({"train_start": str(train[0].date()), "train_end": str(train[-1].date()), "test_start": str(test[0].date()), "test_end": str(test[-1].date()), "purge_days": boundaries.label_horizon_days, "embargo_days": boundaries.embargo_days})
        start += step_days
    return out


def audit_point_in_time(frame: pd.DataFrame, boundaries: ResearchBoundaries, *, date_col: str = "date", availability_col: str | None = "available_at", universe_col: str | None = None) -> dict[str, Any]:
    """Audit timestamps, duplicate keys, negative prices, and survivorship hints."""
    result: dict[str, Any] = {"rows": int(len(frame)), "status": "PASS", "errors": [], "warnings": []}
    if date_col not in frame:
        result["errors"].append(f"missing:{date_col}")
    else:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        result["invalid_dates"] = int(dates.isna().sum())
        if result["invalid_dates"]:
            result["errors"].append("invalid_dates")
    if {date_col, "code"}.issubset(frame.columns):
        result["duplicate_keys"] = int(frame.duplicated([date_col, "code"]).sum())
        if result["duplicate_keys"]:
            result["errors"].append("duplicate_keys")
    for col in ("close", "raw_close", "hfq_close"):
        if col in frame:
            result[f"non_positive_{col}"] = int((pd.to_numeric(frame[col], errors="coerce") <= 0).sum())
            if result[f"non_positive_{col}"]:
                result["errors"].append(f"non_positive:{col}")
    if availability_col and availability_col in frame and date_col in frame:
        avail = pd.to_datetime(frame[availability_col], errors="coerce")
        signal = pd.to_datetime(frame[date_col], errors="coerce")
        leaks = (avail > signal).sum()
        result["availability_after_signal"] = int(leaks)
        if leaks:
            result["errors"].append("future_availability")
    if universe_col and universe_col not in frame:
        result["warnings"].append("missing_historical_universe_membership")
    result["status"] = "BLOCK" if result["errors"] else ("WARN" if result["warnings"] else "PASS")
    return result


def _drawdown(r: pd.Series) -> tuple[float, int, int]:
    """Return max drawdown, longest underwater run, and trough-to-peak recovery.

    The first observation is included in NAV (so an initial loss cannot be
    silently dropped).  ``underwater`` counts observations below the previous
    high-water mark; ``recovery`` measures the longest time from a drawdown
    trough until that high-water mark is regained.
    """
    # Include the initial capital as a high-water mark.  Starting the peak at
    # the first post-return NAV would hide a loss on the first observation.
    nav = pd.concat([pd.Series([1.0]), (1.0 + r.astype(float)).cumprod().reset_index(drop=True)], ignore_index=True)
    if nav.empty:
        return 0.0, 0, 0
    high = nav.cummax()
    dd = nav / high - 1.0
    worst = float(dd.min())
    longest = 0
    recovery = 0
    underwater = 0
    trough_index: int | None = None
    for i, value in enumerate(dd.to_numpy()):
        if value < -1e-12:
            underwater += 1
            longest = max(longest, underwater)
            if trough_index is None or value < dd.iloc[trough_index]:
                trough_index = i
        else:
            if trough_index is not None:
                recovery = max(recovery, i - trough_index)
            underwater = 0
            trough_index = None
    # The initial NAV is never counted in either counter.
    return worst, longest, recovery


def performance_report(returns: Iterable[float] | pd.Series, benchmark: pd.Series | None = None, *, annualization: int = 252, risk_free: float = 0.0, costs: float = 0.0, dates: Iterable[Any] | pd.Series | None = None) -> dict[str, Any]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"observations": 0, "status": "INSUFFICIENT_DATA"}
    excess = r - risk_free / annualization
    vol = float(r.std(ddof=1) * np.sqrt(annualization)) if len(r) > 1 else np.nan
    downside_component = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside = float(np.sqrt(np.mean(downside_component ** 2)) * np.sqrt(annualization)) if len(r) else np.nan
    total = float((1 + r).prod() - 1)
    periods = float(len(r))
    if dates is not None:
        date_values = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
        if len(date_values) >= 2:
            span_days = max((date_values.iloc[-1] - date_values.iloc[0]).days, 1)
            periods = max(span_days / 365.25 * annualization, 1.0)
    ann = float((1 + total) ** (annualization / periods) - 1) if total > -1 else -1.0
    mdd, underwater, recovery = _drawdown(r)
    losses = r[r < 0]
    var_95 = float(r.quantile(0.05))
    tail = losses[losses <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95
    out = {
        "observations": int(len(r)), "total_return": total, "annual_return": ann,
        "volatility": vol,
        "sharpe": float(excess.mean() / excess.std(ddof=1) * np.sqrt(annualization)) if len(r) > 1 and excess.std(ddof=1) > 0 else np.nan,
        "sortino": float(excess.mean() * annualization / downside) if downside and downside > 0 else np.nan,
        "max_drawdown": mdd, "max_drawdown_duration_days": underwater,
        "max_drawdown_recovery_days": recovery,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else np.nan,
        "win_rate": float((r > 0).mean()), "positive_periods": int((r > 0).sum()),
        "negative_periods": int((r < 0).sum()),
        "gain_to_pain": float(r[r > 0].sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else np.nan,
        "var_95": var_95, "cvar_95": cvar_95,
        "skew": float(r.skew()), "cost_drag": float(costs),
    }
    if benchmark is not None:
        pair = pd.concat([r.rename("strategy"), pd.Series(benchmark, dtype=float).rename("benchmark")], axis=1).dropna()
        if len(pair) > 1:
            b = pair.benchmark; e = pair.strategy - b; te = e.std(ddof=1)
            active_growth = float((1.0 + pair.strategy).prod() / (1.0 + b).prod() - 1.0)
            beta = float(pair.strategy.cov(b) / b.var()) if b.var() > 0 else np.nan
            alpha = float((pair.strategy.mean() - beta * b.mean()) * annualization) if np.isfinite(beta) else np.nan
            out.update({
                "benchmark_total_return": float((1 + b).prod() - 1),
                "excess_total_return": active_growth,
                "excess_period_win_rate": float((e > 0).mean()),
                "tracking_error": float(te * np.sqrt(annualization)),
                "information_ratio": float(e.mean() / te * np.sqrt(annualization)) if te > 0 else np.nan,
                "beta": beta, "alpha": alpha,
                "correlation": float(pair.strategy.corr(b)),
            })
    return out


def robustness_report(returns: Iterable[float] | pd.Series, *, seed: int = 42, blocks: int = 500, block_length: int = 10) -> dict[str, Any]:
    """Moving-block bootstrap for Sharpe and total return confidence intervals."""
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    if len(r) < max(20, block_length * 2):
        return {"status": "INSUFFICIENT_DATA", "observations": int(len(r))}
    rng = np.random.default_rng(seed); starts = np.arange(len(r) - block_length + 1); sharpes=[]; totals=[]
    for _ in range(blocks):
        sample=[]
        while len(sample) < len(r):
            start = int(rng.choice(starts)); sample.extend(r[start:start + block_length])
        x=np.asarray(sample[:len(r)]); sharpes.append(x.mean()/x.std(ddof=1)*np.sqrt(252)); totals.append((1+x).prod()-1)
    return {"status": "PASS", "observations": int(len(r)), "bootstrap_blocks": blocks, "block_length": block_length, "sharpe_observed": float(r.mean()/r.std(ddof=1)*np.sqrt(252)), "sharpe_ci95": [float(np.quantile(sharpes,.025)), float(np.quantile(sharpes,.975))], "prob_sharpe_positive": float(np.mean(np.asarray(sharpes)>0)), "total_return_ci95": [float(np.quantile(totals,.025)), float(np.quantile(totals,.975))]}


def freeze(output: str | Path, *, boundaries: ResearchBoundaries, inputs: Iterable[str | Path], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write immutable experiment manifest; this is the gate before holdout access."""
    out = Path(output)
    previous_sha = None
    if out.is_file():
        try:
            previous_sha = json.loads(out.read_text(encoding="utf-8")).get("manifest_sha256")
        except (OSError, json.JSONDecodeError):
            previous_sha = None
    paths=[]
    for p in inputs:
        path=Path(p)
        if path.is_file(): paths.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload={"schema":"research_upgrade.v1", "frozen":True, "boundaries":asdict(boundaries), "inputs":paths, "config":dict(config or {})}
    if previous_sha:
        payload["supersedes_manifest_sha256"] = previous_sha
    payload["manifest_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload
