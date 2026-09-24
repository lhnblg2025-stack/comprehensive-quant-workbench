"""Point-in-time fundamentals, industry history, and low-frequency factors.

The input contract intentionally requires an actual announcement date.  A report-period
is not a public-information timestamp and must never be substituted for one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

FUNDAMENTAL_REQUIRED_COLUMNS = {"code", "report_period", "announcement_date"}
INDUSTRY_REQUIRED_COLUMNS = {"code", "industry", "effective_date"}


@dataclass(frozen=True)
class PITSummary:
    fundamentals_rows: int
    fundamentals_with_announcement_date: int
    industry_rows: int
    panel_rows_with_fundamentals: int
    panel_rows_with_industry: int


def _normalise_code(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _dates(values: pd.Series, field: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce").dt.normalize().astype("datetime64[ns]")
    if result.isna().any():
        raise ValueError(f"invalid_dates:{field}")
    return result


def validate_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical financial statements and normalize their PIT keys."""
    missing = FUNDAMENTAL_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"fundamentals_missing_columns:{','.join(sorted(missing))}")
    out = frame.copy()
    out["code"] = _normalise_code(out["code"])
    out["report_period"] = _dates(out["report_period"], "report_period")
    out["announcement_date"] = _dates(out["announcement_date"], "announcement_date")
    if (out["announcement_date"] < out["report_period"]).any():
        raise ValueError("announcement_before_report_period")
    if out.duplicated(["code", "report_period", "announcement_date"]).any():
        raise ValueError("duplicate_fundamental_release")
    return out.sort_values(["code", "announcement_date", "report_period"])


def validate_industry_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate time-varying industry classifications without using today's labels."""
    missing = INDUSTRY_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"industry_missing_columns:{','.join(sorted(missing))}")
    out = frame.copy()
    out["code"] = _normalise_code(out["code"])
    out["effective_date"] = _dates(out["effective_date"], "effective_date")
    if "end_date" not in out.columns:
        out["end_date"] = pd.NaT
    else:
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce").dt.normalize()
    if out["industry"].isna().any() or (out["industry"].astype(str).str.len() == 0).any():
        raise ValueError("invalid_industry")
    invalid = out["end_date"].notna() & (out["end_date"] < out["effective_date"])
    if invalid.any():
        raise ValueError("industry_end_before_effective")
    out = out.sort_values(["code", "effective_date", "end_date"])
    for _, group in out.groupby("code", sort=False):
        starts = group["effective_date"].to_numpy()
        ends = group["end_date"].to_numpy()
        for index in range(1, len(group)):
            if pd.notna(ends[index - 1]) and starts[index] <= ends[index - 1]:
                raise ValueError("overlapping_industry_history")
    return out


def _availability_dates(announcement_dates: pd.Series, calendar: Iterable[object], lag_sessions: int) -> pd.Series:
    if lag_sessions < 1:
        raise ValueError("fundamental_availability_lag_sessions must be at least one")
    sessions = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna().normalize().unique().sort_values()
    if not len(sessions):
        raise ValueError("empty_trading_calendar")
    positions = sessions.searchsorted(pd.DatetimeIndex(announcement_dates).normalize(), side="left") + lag_sessions
    available = pd.Series(pd.NaT, index=announcement_dates.index, dtype="datetime64[ns]")
    valid = positions < len(sessions)
    available.loc[valid] = sessions[positions[valid]]
    return available


def _asof_by_code(panel: pd.DataFrame, releases: pd.DataFrame, release_date: str) -> pd.DataFrame:
    """As-of join each security independently, avoiding global date leakage."""
    pieces = []
    release_columns = [column for column in releases.columns if column != "code"]
    for code, left in panel.groupby("code", sort=False):
        right = releases[releases["code"] == code]
        if right.empty:
            piece = left.copy()
            for column in release_columns:
                piece[column] = np.nan
        else:
            piece = pd.merge_asof(
                left.sort_values("date"),
                right.drop(columns=["code"]).sort_values(release_date),
                left_on="date",
                right_on=release_date,
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "code"])


def attach_pit_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame, calendar: Iterable[object], *, lag_sessions: int = 1) -> pd.DataFrame:
    """Attach only the latest financial release available at each signal date."""
    required = {"date", "code", "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel_missing_columns:{','.join(sorted(missing))}")
    releases = validate_fundamentals(fundamentals)
    releases = releases.copy()
    releases["available_date"] = _availability_dates(releases["announcement_date"], calendar, lag_sessions)
    releases = releases.dropna(subset=["available_date"])
    # Revisions published later replace a previous version only from their own release date.
    releases = releases.sort_values(["code", "available_date", "announcement_date", "report_period"]).drop_duplicates(
        ["code", "available_date", "report_period"], keep="last"
    )
    left = panel.copy()
    left["code"] = _normalise_code(left["code"])
    left["date"] = _dates(left["date"], "panel_date")
    return _asof_by_code(left, releases, "available_date")


def attach_pit_industry(panel: pd.DataFrame, industry_history: pd.DataFrame) -> pd.DataFrame:
    """Attach the classification effective on each signal date."""
    history = validate_industry_history(industry_history)
    left = panel.copy()
    left["code"] = _normalise_code(left["code"])
    left["date"] = _dates(left["date"], "panel_date")
    joined = _asof_by_code(left, history, "effective_date")
    valid = joined["end_date"].isna() | (joined["date"] <= joined["end_date"])
    joined.loc[~valid, "industry"] = np.nan
    return joined


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)


def _zscore_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
    def standardize(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if len(valid) < 3:
            return pd.Series(np.nan, index=group.index)
        std = valid.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - valid.mean()) / std
    return values.groupby(dates, group_keys=False).apply(standardize)


def add_fundamental_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Create transparent value and quality factors from as-of fundamentals.

    Canonical optional metrics are eps_ttm, book_value_per_share, operating_cashflow_per_share,
    roe, net_margin, cfo_to_net_income, and debt_to_assets.  Missing input fields result in
    missing factors rather than a proxy from report-period data.
    """
    out = panel.copy()
    price = _numeric(out, "close").where(lambda value: value > 0)
    out["value_earnings_yield"] = _numeric(out, "eps_ttm") / price
    out["value_book_to_price"] = _numeric(out, "book_value_per_share") / price
    out["value_ocf_to_price"] = _numeric(out, "operating_cashflow_per_share") / price
    out["quality_roe"] = _numeric(out, "roe")
    out["quality_net_margin"] = _numeric(out, "net_margin")
    out["quality_cash_conversion"] = _numeric(out, "cfo_to_net_income")
    out["quality_low_leverage"] = -_numeric(out, "debt_to_assets")
    out["quality_gross_margin"] = _numeric(out, "gross_margin")
    out["quality_interest_coverage"] = _numeric(out, "ebit_to_interest")
    out["quality_current_ratio"] = _numeric(out, "current_ratio")
    out["quality_cfo_to_assets"] = _numeric(out, "cfo_to_assets")
    out["growth_profit_yoy"] = _numeric(out, "yoy_ni")
    out["growth_eps_yoy"] = _numeric(out, "yoy_eps")
    out["growth_equity_yoy"] = _numeric(out, "yoy_equity")
    out["cashflow_to_revenue"] = _numeric(out, "cfo_to_revenue")
    value_components = ["value_earnings_yield", "value_book_to_price", "value_ocf_to_price"]
    # Stable core composite: optional coverage-sparse fields remain standalone signals.
    quality_components = ["quality_roe", "quality_net_margin", "quality_low_leverage"]
    out["value_composite"] = pd.concat([_zscore_by_date(out[column], out["date"]) for column in value_components], axis=1).mean(axis=1, skipna=False)
    out["quality_composite"] = pd.concat([_zscore_by_date(out[column], out["date"]) for column in quality_components], axis=1).mean(axis=1, skipna=False)
    out["value_quality_composite"] = pd.concat(
        [_zscore_by_date(out["value_composite"], out["date"]), _zscore_by_date(out["quality_composite"], out["date"])], axis=1
    ).mean(axis=1, skipna=False)
    return out


def industry_neutralize(panel: pd.DataFrame, factor_names: Iterable[str], *, min_industry_size: int = 3) -> pd.DataFrame:
    """Demean each factor within contemporaneous PIT industry groups.

    The residual factors carry an ``_industry_neutral`` suffix.  Small industries are
    deliberately excluded to prevent a single stock from becoming a zero-residual signal.
    """
    if "industry" not in panel:
        raise ValueError("industry_column_required_for_neutralization")
    out = panel.copy()
    valid_industry = out["industry"].notna()
    for factor in factor_names:
        if factor not in out:
            raise ValueError(f"factor_not_found:{factor}")
        counts = out.loc[valid_industry].groupby(["date", "industry"])[factor].transform("count")
        means = out.loc[valid_industry].groupby(["date", "industry"])[factor].transform("mean")
        result = pd.Series(np.nan, index=out.index, dtype=float)
        eligible = valid_industry.copy()
        eligible.loc[valid_industry] = counts >= min_industry_size
        result.loc[eligible] = pd.to_numeric(out.loc[eligible, factor], errors="coerce") - means.loc[eligible]
        out[f"{factor}_industry_neutral"] = result
    return out


def coverage_summary(panel: pd.DataFrame, factor_names: Iterable[str], *, min_stocks: int, min_coverage_ratio: float | None = None) -> dict[str, object]:
    """Report PIT coverage using an absolute floor and optional universe ratio."""
    if min_stocks < 2:
        raise ValueError("min_stocks must be at least two")
    if min_coverage_ratio is not None and not 0 < min_coverage_ratio <= 1:
        raise ValueError("min_coverage_ratio must be in (0, 1]")
    dates = pd.to_datetime(panel["date"])
    per_date = panel.assign(_date=dates).groupby("_date")["code"].nunique()
    factor_coverage = {}
    for factor in factor_names:
        if factor not in panel:
            factor_coverage[factor] = {"usable_dates": 0, "median_stocks": 0, "minimum_stocks": 0}
            continue
        counts = panel.dropna(subset=[factor]).assign(_date=dates).groupby("_date")["code"].nunique().reindex(per_date.index, fill_value=0)
        ratio = counts / per_date.replace(0, np.nan)
        eligible = counts >= min_stocks
        if min_coverage_ratio is not None:
            eligible &= ratio >= min_coverage_ratio
        factor_coverage[factor] = {
            "usable_dates": int(eligible.sum()),
            "median_stocks": int(counts.median()) if len(counts) else 0,
            "minimum_stocks": int(counts.min()) if len(counts) else 0,
            "median_coverage_ratio": float(ratio.median()) if len(ratio) else 0.0,
            "minimum_coverage_ratio": float(ratio.min()) if len(ratio) else 0.0,
        }
    return {
        "required_min_stocks": int(min_stocks),
        "required_min_coverage_ratio": min_coverage_ratio,
        "panel_dates": int(len(per_date)),
        "panel_median_stocks": int(per_date.median()) if len(per_date) else 0,
        "panel_minimum_stocks": int(per_date.min()) if len(per_date) else 0,
        "factors": factor_coverage,
        "status": "PASS" if all(item["usable_dates"] > 0 for item in factor_coverage.values()) else "BLOCK",
    }
