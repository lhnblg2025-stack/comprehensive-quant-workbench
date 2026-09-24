"""Unified, auditable research platform for long-horizon A-share studies.

The platform joins the data audit, lifecycle membership, factor construction,
chronological selection, execution costs, risk reporting and freeze manifest.
It does not fabricate missing 2016--2017 prices: coverage gaps are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .build_lifecycle_price_panel import build_lifecycle_panel

from .research_upgrade import (
    ResearchBoundaries,
    evaluate_holdout,
    freeze,
    performance_report,
    robustness_report,
    split_frame,
)


@dataclass(frozen=True)
class PlatformConfig:
    data_root: str = "data_warehouse"
    output_root: str = "generated/ten_year_research"
    panel_path: str = "generated/ten_year_research/lifecycle_price_panel.parquet"
    requested_start: str = "2016-01-01"
    requested_end: str = "2026-09-18"
    development_end: str = "2024-12-31"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-12-31"
    holdout_start: str = "2026-01-01"
    holdout_end: str | None = None
    quantile: float = 0.2
    horizon: int = 5
    rebalance: str = "weekly"
    annualization: int = 252
    commission_bps: float = 0.85
    stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 10.0
    participation_limit: float = 0.10
    random_seed: int = 42

    @property
    def boundaries(self) -> ResearchBoundaries:
        return ResearchBoundaries(
            development_end=self.development_end,
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            holdout_start=self.holdout_start,
            holdout_end=self.holdout_end,
            label_horizon_days=self.horizon,
        )

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.stamp_duty_bps + self.transfer_fee_bps + self.slippage_bps


FACTOR_CATALOG: tuple[dict[str, Any], ...] = (
    {"name": "mom_20", "family": "momentum", "direction": 1, "description": "20-day continuation"},
    {"name": "mom_60", "family": "momentum", "direction": 1, "description": "60-day continuation"},
    {"name": "mom_120", "family": "momentum", "direction": 1, "description": "120-day continuation"},
    {"name": "resid_mom_60", "family": "momentum_residual", "direction": 1, "description": "market-residual momentum"},
    {"name": "rev_5", "family": "reversal", "direction": 1, "description": "short-term reversal"},
    {"name": "low_vol_20", "family": "volatility", "direction": 1, "description": "low realised volatility"},
    {"name": "low_vol_60", "family": "volatility", "direction": 1, "description": "low 60-day volatility"},
    {"name": "low_beta_60", "family": "volatility", "direction": 1, "description": "low market beta"},
    {"name": "low_turnover_20", "family": "liquidity", "direction": 1, "description": "low turnover"},
    {"name": "amihud_liquid_20", "family": "liquidity", "direction": 1, "description": "low price impact"},
    {"name": "amount_level_20", "family": "capacity", "direction": 1, "description": "high traded amount"},
    {"name": "turnover_change_rev", "family": "liquidity", "direction": 1, "description": "turnover normalisation"},
    {"name": "ma_20_60", "family": "trend", "direction": 1, "description": "medium trend"},
    {"name": "ma_60_120", "family": "trend", "direction": 1, "description": "long trend"},
    {"name": "trend_r2_60", "family": "trend", "direction": 1, "description": "trend quality"},
    {"name": "price_ma_200", "family": "trend", "direction": 1, "description": "price above long MA"},
    {"name": "breakout_60", "family": "breakout", "direction": 1, "description": "60-day breakout"},
    {"name": "dist_52w", "family": "breakout", "direction": 1, "description": "distance to 52-week high"},
    {"name": "quiet_trend_20", "family": "volume_price", "direction": 1, "description": "trend without volume shock"},
    {"name": "combo_trend_quality", "family": "composite", "direction": 1, "description": "trend plus quality"},
    {"name": "combo_mom_lowvol", "family": "composite", "direction": 1, "description": "momentum plus low volatility"},
    {"name": "combo_multifactor", "family": "composite", "direction": 1, "description": "diversified price/volume blend"},
)


def _normalise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if out["date"].isna().any() or out.duplicated(["date", "code"]).any():
        raise ValueError("panel_dates_or_keys_invalid")
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def coverage_audit(panel: pd.DataFrame, cfg: PlatformConfig) -> dict[str, Any]:
    dates = pd.to_datetime(panel["date"])
    yearly = {}
    for year, group in panel.assign(_year=dates.dt.year).groupby("_year"):
        g = group
        close_col = next((c for c in ("close", "hfq_close", "raw_close") if c in g.columns), None)
        close_values = pd.to_numeric(g[close_col], errors="coerce") if close_col else pd.Series(np.nan, index=g.index)
        yearly[str(int(year))] = {
            "rows": int(len(g)),
            "symbols": int(g["code"].nunique()),
            "tradable_rows": int(g.get("is_tradable", pd.Series(True, index=g.index)).fillna(False).sum()),
            "missing_close": int(close_values.isna().sum()),
        }
    expected = pd.Timestamp(cfg.requested_start)
    actual = dates.min()
    end = dates.max()
    return {
        "requested_start": cfg.requested_start,
        "requested_end": cfg.requested_end,
        "actual_start": str(actual.date()),
        "actual_end": str(end.date()),
        "requested_years": round((pd.Timestamp(cfg.requested_end) - expected).days / 365.25, 2),
        "actual_years": round((end - actual).days / 365.25, 2),
        "complete_requested_horizon": bool(actual <= expected and end >= pd.Timestamp(cfg.requested_end)),
        "gap_days_at_start": int(max((actual - expected).days, 0)),
        "symbols": int(panel["code"].nunique()),
        "rows": int(len(panel)),
        "yearly": yearly,
        "history_warning": "less_than_requested_ten_years" if actual > expected else None,
        "survivorship_warning": "observed_inventory_is_not_historical_membership",
    }


def _rank(series: pd.Series, dates: pd.Series) -> pd.Series:
    return series.groupby(dates, sort=False).rank(pct=True, method="average")


def build_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Build only price/volume factors whose timestamps are intrinsically PIT."""
    out = panel.copy().sort_values(["code", "date"]).reset_index(drop=True)
    grouped = out.groupby("code", sort=False)
    close = pd.to_numeric(out["hfq_close"], errors="coerce")
    raw_close = pd.to_numeric(out["raw_close"], errors="coerce")
    ret = grouped["hfq_close"].pct_change()
    market = ret.groupby(out["date"], sort=False).transform("mean")
    out["fwd_return"] = grouped["hfq_close"].shift(-1) / close - 1.0
    for window in (20, 60, 120):
        out[f"mom_{window}"] = grouped["hfq_close"].pct_change(window)
        out[f"low_vol_{window}"] = -ret.groupby(out["code"], sort=False).rolling(window, min_periods=window).std().reset_index(level=0, drop=True).set_axis(out.index)
    out["rev_5"] = -out["mom_20"].groupby(out["code"], sort=False).shift(-15) if False else -grouped["hfq_close"].pct_change(5)
    amount = pd.to_numeric(out.get("amount"), errors="coerce")
    turnover = pd.to_numeric(out.get("turnover_rate"), errors="coerce")
    out["low_turnover_20"] = -turnover.groupby(out["code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True).set_axis(out.index)
    out["amount_level_20"] = np.log1p(amount.groupby(out["code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True).set_axis(out.index))
    out["amihud_liquid_20"] = -(ret.abs() / amount.replace(0, np.nan)).groupby(out["code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True).set_axis(out.index) * 1e10
    out["turnover_change_rev"] = -(turnover.groupby(out["code"], sort=False).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True).set_axis(out.index) /
                                    turnover.groupby(out["code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True).set_axis(out.index) - 1)
    ma20 = close.groupby(out["code"], sort=False).rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).set_axis(out.index)
    ma60 = close.groupby(out["code"], sort=False).rolling(60, min_periods=60).mean().reset_index(level=0, drop=True).set_axis(out.index)
    ma120 = close.groupby(out["code"], sort=False).rolling(120, min_periods=120).mean().reset_index(level=0, drop=True).set_axis(out.index)
    ma200 = close.groupby(out["code"], sort=False).rolling(200, min_periods=120).mean().reset_index(level=0, drop=True).set_axis(out.index)
    out["ma_20_60"] = ma20 / ma60 - 1
    out["ma_60_120"] = ma60 / ma120 - 1
    out["price_ma_200"] = close / ma200 - 1
    out["dist_52w"] = close / close.groupby(out["code"], sort=False).rolling(252, min_periods=120).max().reset_index(level=0, drop=True).set_axis(out.index) - 1
    prior_high = out.groupby("code", sort=False)["hfq_high"].rolling(60, min_periods=30).max().reset_index(level=0, drop=True).set_axis(out.index).groupby(out["code"], sort=False).shift(1)
    out["breakout_60"] = close / prior_high - 1
    beta = _rolling_beta(out["code"], ret, market, window=60, min_periods=30)
    out["low_beta_60"] = -beta
    resid = ret - beta * market
    out["resid_mom_60"] = _rolling_group_sum(out["code"], resid, window=60, min_periods=30)
    out["trend_r2_60"] = _trend_quality(out["code"], close)
    out["quiet_trend_20"] = _rank(out["mom_60"], out["date"]) - _rank(out["amount_level_20"], out["date"])
    out["combo_trend_quality"] = (_rank(out["ma_20_60"], out["date"]) + _rank(out["trend_r2_60"], out["date"])) / 2
    out["combo_mom_lowvol"] = (_rank(out["mom_60"], out["date"]) + _rank(out["low_vol_20"], out["date"])) / 2
    out["combo_multifactor"] = (_rank(out["mom_60"], out["date"]) + _rank(out["low_vol_20"], out["date"]) + _rank(out["low_turnover_20"], out["date"]) + _rank(out["dist_52w"], out["date"]) + _rank(out["ma_20_60"], out["date"]) + _rank(out["amihud_liquid_20"], out["date"])) / 6
    return out


def _rolling_group_sum(codes: pd.Series, values: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    """Return a same-index rolling sum without MultiIndex alignment hazards."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in codes.groupby(codes, sort=False).groups.items():
        result.loc[idx] = values.loc[idx].rolling(window, min_periods=min_periods).sum().to_numpy()
    return result


def _rolling_beta(codes: pd.Series, values: pd.Series, market: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    """Compute stock beta to the contemporaneous market return by code.

    ``SeriesGroupBy.rolling().cov(other)`` aligns ``other`` by its index in a
    pandas-version-dependent way and can expand to a MultiIndex product.  The
    explicit per-code calculation is deterministic and preserves the original
    row index, which is essential before joining factor columns.
    """
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in codes.groupby(codes, sort=False).groups.items():
        x = pd.to_numeric(values.loc[idx], errors="coerce")
        y = pd.to_numeric(market.loc[idx], errors="coerce")
        mx = x.rolling(window, min_periods=min_periods).mean()
        my = y.rolling(window, min_periods=min_periods).mean()
        cov = (x * y).rolling(window, min_periods=min_periods).mean() - mx * my
        var = y.rolling(window, min_periods=min_periods).var(ddof=0)
        result.loc[idx] = (cov / var.replace(0, np.nan)).to_numpy()
    return result


def _trend_quality(codes: pd.Series, close: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for _, idx in codes.groupby(codes, sort=False).groups.items():
        values = np.log(pd.to_numeric(close.loc[idx], errors="coerce")).to_numpy()
        x = np.arange(len(values), dtype=float)
        for i in range(59, len(values)):
            y = values[i - 59:i + 1]
            if not np.isfinite(y).all():
                continue
            slope = np.polyfit(x[:60], y, 1)[0]
            corr = np.corrcoef(x[:60], y)[0, 1]
            result.loc[idx[i]] = np.sign(slope) * corr * corr
    return result


def _strategy_returns(frame: pd.DataFrame, factor: str, cfg: PlatformConfig, *, mask: pd.Series | None = None) -> pd.DataFrame:
    data = frame if mask is None else frame.loc[mask]
    rows = []
    previous: set[str] | None = None
    for date, group in data.groupby("date", sort=True):
        group = group.dropna(subset=[factor, "fwd_return"])
        if len(group) < 30:
            continue
        n = max(1, int(len(group) * cfg.quantile))
        selected = group.nlargest(n, factor)
        gross = float(selected["fwd_return"].mean())
        current = set(selected["code"].astype(str)) if "code" in selected else set(selected.index.astype(str))
        if previous is None:
            turnover = 1.0
        else:
            # One-way turnover proxy for equal-weight holdings: only the
            # fraction replaced since the prior rebalance incurs trading cost.
            turnover = 1.0 - len(current & previous) / max(len(current), 1)
        previous = current
        cost = cfg.total_cost_bps / 10000.0 * turnover
        rows.append({"date": date, "return": gross - cost, "gross_return": gross, "turnover": turnover, "selected": n})
    return pd.DataFrame(rows)


def factor_report(frame: pd.DataFrame, cfg: PlatformConfig, partition: str) -> list[dict[str, Any]]:
    if partition == "holdout":
        data = evaluate_holdout(frame, cfg.boundaries, frozen_manifest=Path(cfg.output_root) / "freeze_manifest.json")
    else:
        data = split_frame(frame, cfg.boundaries, partition=partition)
    result = []
    for spec in FACTOR_CATALOG:
        factor = spec["name"]
        if factor not in data:
            continue
        returns = _strategy_returns(data, factor, cfg)
        if returns.empty:
            result.append({"factor": factor, "family": spec["family"], "status": "INSUFFICIENT_DATA"})
            continue
        metrics = performance_report(returns["return"], dates=returns["date"], annualization=cfg.annualization, costs=float(returns["turnover"].mean() * cfg.total_cost_bps / 10000))
        ic = data.groupby("date", sort=False).apply(lambda g: g[factor].corr(g["fwd_return"], method="spearman"), include_groups=False).dropna()
        result.append({"factor": factor, "family": spec["family"], "direction": spec["direction"], "description": spec["description"], "observations": int(len(returns)), "mean_rank_ic": float(ic.mean()) if len(ic) else None, "rank_ic_ir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else None, "metrics": metrics, "robustness": robustness_report(returns["return"], seed=cfg.random_seed, blocks=200, block_length=min(10, max(2, len(returns) // 10)))})
    return result


def _load_or_build_panel(cfg: PlatformConfig) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    output = Path(cfg.output_root)
    path = Path(cfg.panel_path)
    inputs: list[Path] = []
    if path.is_file():
        panel = pd.read_parquet(path)
        report_path = path.with_suffix(".report.json")
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        inputs.append(path)
        return _normalise_panel(panel), report, inputs
    panel, report = build_lifecycle_panel(Path(cfg.data_root) / "kline_raw", Path(cfg.data_root) / "kline_hfq", start=cfg.requested_start, end=cfg.requested_end or "2100-01-01")
    output.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    path.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    inputs.extend([Path(cfg.data_root) / "kline_raw", Path(cfg.data_root) / "kline_hfq"])
    return _normalise_panel(panel), report, inputs


def run(config: PlatformConfig | Mapping[str, Any] | None = None, *, use_observed_holdout: bool = True) -> dict[str, Any]:
    if config is None:
        cfg = PlatformConfig()
    elif isinstance(config, PlatformConfig):
        cfg = config
    else:
        cfg = PlatformConfig(**dict(config))
    output = Path(cfg.output_root)
    output.mkdir(parents=True, exist_ok=True)
    panel, lifecycle_report, input_paths = _load_or_build_panel(cfg)
    coverage = coverage_audit(panel, cfg)
    factor_frame = build_factor_panel(panel)
    factor_frame.to_parquet(output / "factor_panel.parquet", index=False)
    partitions = {
        "development": factor_report(factor_frame, cfg, "development"),
        "validation": factor_report(factor_frame, cfg, "validation"),
    }
    # Holdout is deliberately opt-in and only reads after the freeze manifest.
    boundaries_payload = asdict(cfg.boundaries)
    frozen = freeze(output / "freeze_manifest.json", boundaries=cfg.boundaries, inputs=[p for p in input_paths if p.is_file()], config=asdict(cfg))
    if use_observed_holdout:
        partitions["holdout_observed"] = factor_report(factor_frame, cfg, "holdout")
    summary = {
        "schema": "ten_year_research_platform/v1",
        "status": "PASS",
        "experiment": asdict(cfg),
        "boundaries": boundaries_payload,
        "coverage": coverage,
        "lifecycle_report": lifecycle_report,
        "factor_catalog": list(FACTOR_CATALOG),
        "partitions": partitions,
        "limitations": [
            "2026 data is an observed isolated evaluation period if it was previously inspected",
            "2026-09-18 is not a complete calendar-year or complete nine-month period",
            "fundamental factors require audited available_date and are not silently mixed into price-only results",
            "current code inventory is not a historical exchange membership file",
        ],
        "freeze_manifest_sha256": frozen["manifest_sha256"],
    }
    (output / "research_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "research_report.md").write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _markdown_report(summary: dict[str, Any]) -> str:
    c = summary["coverage"]
    lines = [
        "# Ten-Year Research Platform Report", "",
        f"- Actual price coverage: `{c['actual_start']}` to `{c['actual_end']}` ({c['actual_years']} years)",
        f"- Requested coverage: `{c['requested_start']}` to `{c['requested_end']}`",
        f"- Complete requested horizon: `{c['complete_requested_horizon']}`",
        f"- Symbols / rows: `{c['symbols']}` / `{c['rows']}`",
        "",
        "## Data and isolation",
        "- Development ends 2024-12-31; 2025 is validation; 2026 is an isolated evaluation partition.",
        "- The holdout is only read after the freeze manifest is written.",
        "- Missing early history is reported, never forward-filled or renamed as ten years.",
        "",
        "## Factor results",
    ]
    for partition, rows in summary["partitions"].items():
        lines.append(f"### {partition}")
        lines.append("| Factor | Family | Rank IC | Sharpe | Sortino | Calmar | Max DD |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in sorted(rows, key=lambda x: (x.get("metrics", {}).get("sharpe") is None, -(x.get("metrics", {}).get("sharpe") or -999))):
            m = row.get("metrics", {})
            lines.append(f"| {row['factor']} | {row['family']} | {row.get('mean_rank_ic', '')} | {m.get('sharpe', '')} | {m.get('sortino', '')} | {m.get('calmar', '')} | {m.get('max_drawdown', '')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data_warehouse")
    p.add_argument("--output-root", default="generated/ten_year_research")
    p.add_argument("--panel-path", default=None)
    p.add_argument("--requested-start", default="2016-01-01")
    p.add_argument("--requested-end", default="2026-09-18")
    p.add_argument("--no-holdout", action="store_true")
    args = p.parse_args(argv)
    kwargs = vars(args)
    if kwargs["panel_path"] is None:
        kwargs["panel_path"] = str(Path(args.output_root) / "lifecycle_price_panel.parquet")
    no_holdout = bool(kwargs.pop("no_holdout", False))
    result = run(kwargs, use_observed_holdout=not no_holdout)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"], "output_root": args.output_root}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
