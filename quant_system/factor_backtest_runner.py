"""Reproducible factor evaluation and signal-portfolio backtest runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from .research_protocol import DEFAULT_PROTOCOL

DEFAULT_FACTORS = ["mom20", "mom60", "mom120", "dist_52w", "amp20", "turnover", "vol_ratio20", "pe_pct252", "pb_pct252"]


@dataclass(frozen=True)
class Config:
    data_dir: str
    output_dir: str
    factors: tuple[str, ...]
    quantile: float = 0.2
    cost_bps: float = 15.0
    min_stocks: int = 30
    # 5:3 chronological train/test split (3/8 OOS).
    oos_ratio: float = DEFAULT_PROTOCOL.test_ratio
    benchmark: str = "equal_weight"
    annualization: float = 252.0
    robustness_quantiles: tuple[float, ...] = (0.1, 0.2, 0.3)
    factor_directions: tuple[tuple[str, int], ...] = ()
    rebalance: str = "daily"
    max_industry_weight: float = 0.0


def _validate_parameters(quantile, cost_bps, min_stocks, oos_ratio=None, annualization=252.0):
    if not 0 < quantile < 0.5: raise ValueError("quantile must be between 0 and 0.5")
    if cost_bps < 0: raise ValueError("cost_bps must be non-negative")
    if min_stocks < 2: raise ValueError("min_stocks must be at least 2")
    if oos_ratio is not None and not 0 < oos_ratio < 1: raise ValueError("oos_ratio must be between 0 and 1")
    if annualization <= 0: raise ValueError("annualization must be positive")
    if oos_ratio is not None and abs(oos_ratio - DEFAULT_PROTOCOL.test_ratio) > 1e-9:
        raise ValueError("research_protocol_requires_5_3_train_test_split")


def _read_snapshots(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.parquet"))
    if not files: raise FileNotFoundError(f"no parquet snapshots under {data_dir}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if not {"code", "date", "close"}.issubset(frame.columns): continue
        frame = frame.copy()
        if pd.api.types.is_datetime64_any_dtype(frame["date"]):
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        else:
            raw_dates = frame["date"].astype(str)
            compact = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
            frame["date"] = compact.fillna(pd.to_datetime(raw_dates, errors="coerce"))
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frames.append(frame)
    if not frames: raise ValueError("no usable snapshots with code/date/close")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date", "code", "close"])
    out = out[out["close"] > 0]
    return out.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")


def _metrics(returns: pd.Series, annualization=252.0) -> dict:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty: return {"observations": 0, "total_return": None, "annual_return": None, "volatility": None, "sharpe": None, "max_drawdown": None}
    curve = (1 + returns).cumprod(); years = len(returns) / annualization
    total = float(curve.iloc[-1] - 1); annual = float(curve.iloc[-1] ** (1 / years) - 1) if curve.iloc[-1] > 0 else None
    std = returns.std(ddof=1)
    vol = float(std * math.sqrt(annualization)) if len(returns) > 1 else None
    sharpe = float(returns.mean() / std * math.sqrt(annualization)) if len(returns) > 1 and std > 0 else None
    return {"observations": int(len(returns)), "total_return": total, "annual_return": annual, "volatility": vol, "sharpe": sharpe, "max_drawdown": float((curve / curve.cummax() - 1).min())}


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach a label whose entry timing matches the signal contract.

    With raw open prices, a signal observed at session ``t`` close enters at
    ``t+1`` open and is marked through ``t+2`` open. Close-only inputs retain
    the legacy next-close research label and are explicitly not execution-valid.
    """
    if "forward_return" in panel.columns:
        return panel
    data = panel.sort_values(["date", "code"]).copy(); dates = pd.DatetimeIndex(sorted(pd.to_datetime(data.date.unique())))
    if "raw_open" in data.columns:
        data["entry_date"] = data.date.map({dates[index]: dates[index + 1] for index in range(len(dates) - 1)})
        data["exit_date"] = data.date.map({dates[index]: dates[index + 2] for index in range(len(dates) - 2)})
        opens = data[["code", "date", "raw_open"]].rename(columns={"date": "entry_date", "raw_open": "entry_open"})
        data = data.merge(opens, on=["code", "entry_date"], how="left")
        exits = data[["code", "date", "raw_open"]].rename(columns={"date": "exit_date", "raw_open": "exit_open"})
        data = data.merge(exits, on=["code", "exit_date"], how="left")
        data["forward_return"] = pd.to_numeric(data["exit_open"], errors="coerce") / pd.to_numeric(data["entry_open"], errors="coerce") - 1.0
        data["return_basis"] = "raw_next_open_to_following_open"
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["next_date"] = data.date.map({dates[index]: dates[index + 1] for index in range(len(dates) - 1)})
    prices = data[["code", "date", "close"]].rename(columns={"date": "next_date", "close": "next_close"})
    prices["next_date"] = pd.to_datetime(prices["next_date"], errors="coerce")
    data = data.merge(prices, on=["code", "next_date"], how="left")
    data["forward_return"] = data.next_close / data.close - 1
    data["return_basis"] = "legacy_close_to_next_close"
    return data


def _rebalance_date_set(dates, rebalance="daily"):
    index = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    if rebalance == "weekly":
        return set(pd.Series(index, index=index).groupby(index.to_period("W")).max())
    if rebalance == "monthly":
        return set(pd.Series(index, index=index).groupby(index.to_period("M")).max())
    return set(index)


def _select_with_industry_cap(ranked: pd.DataFrame, n: int, max_industry_weight: float) -> set[str]:
    if max_industry_weight <= 0 or "industry" not in ranked.columns:
        return set(ranked.head(n).code)
    cap = max(1, int(math.floor(n * max_industry_weight + 1e-12)))
    selected: list[str] = []
    counts: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        industry = getattr(row, "industry", None)
        if pd.isna(industry):
            continue
        key = str(industry)
        if counts.get(key, 0) >= cap:
            continue
        selected.append(str(row.code))
        counts[key] = counts.get(key, 0) + 1
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError("industry_cap_infeasible_for_selected_cross_section")
    return set(selected)


def _factor_returns(panel, factor, quantile, cost_bps, min_stocks, direction=1, rebalance="daily", max_industry_weight=0.0):
    _validate_parameters(quantile, cost_bps, min_stocks)
    diagnostics = {"factor": factor, "dates": int(panel.date.nunique()), "usable_dates": 0, "missing_ratio": None, "mean_long_turnover": None, "mean_short_turnover": None, "ic_mean": None, "ic_ir": None, "ic_positive_pct": None}
    if factor not in panel.columns:
        diagnostics["error"] = "factor_not_found"; return pd.DataFrame(), diagnostics
    data = _prepare_panel(panel)
    diagnostics["missing_ratio"] = float(data[factor].isna().mean())
    rows = []; previous_long = set(); previous_short = set(); rebalance_dates = _rebalance_date_set(data.date.unique(), rebalance)
    grouped = data.groupby("date", sort=True)
    for date, group in grouped:
        g = group.dropna(subset=[factor, "close"])
        if len(g) < min_stocks: continue
        ranked = g.assign(_score=g[factor] * direction).sort_values("_score", ascending=False)
        n = max(1, int(len(ranked) * quantile))
        if pd.Timestamp(date) in rebalance_dates or not previous_long:
            long_set = _select_with_industry_cap(ranked, n, max_industry_weight)
            short_set = _select_with_industry_cap(ranked.iloc[::-1], n, max_industry_weight)
            lt = 1.0 if not previous_long else 1 - len(long_set & previous_long) / max(1, len(long_set))
            st = 1.0 if not previous_short else 1 - len(short_set & previous_short) / max(1, len(short_set))
            previous_long, previous_short = long_set, short_set
        else:
            lt = st = 0.0
        by_code = g.set_index("code")
        long_returns = by_code.reindex(list(previous_long))["forward_return"]
        short_returns = by_code.reindex(list(previous_short))["forward_return"]
        rows.append((date, long_returns.mean(), short_returns.mean(), g["forward_return"].mean(), lt, st, g[factor].corr(g.forward_return, method="spearman")))
    if not rows:
        diagnostics["error"] = "insufficient_cross_section"; return pd.DataFrame(), diagnostics
    result = pd.DataFrame(rows, columns=["date", "top_gross", "bottom_gross", "benchmark", "long_turnover", "short_turnover", "ic"]).set_index("date")
    bps = cost_bps / 10000.0
    result["long_cost"] = result.long_turnover * bps; result["short_cost"] = result.short_turnover * bps
    result["long_only"] = result.top_gross - result.long_cost
    result["long_short_gross"] = result.top_gross - result.bottom_gross
    result["long_short"] = result.long_short_gross - result.long_cost - result.short_cost
    diagnostics["usable_dates"] = len(result); diagnostics["mean_long_turnover"] = float(result.long_turnover.mean()); diagnostics["mean_short_turnover"] = float(result.short_turnover.mean()); diagnostics["mean_turnover"] = diagnostics["mean_long_turnover"]
    ic = result.ic.dropna(); diagnostics["ic_mean"] = float(ic.mean()) if len(ic) else None; diagnostics["ic_positive_pct"] = float((ic > 0).mean()) if len(ic) else None
    diagnostics["ic_ir"] = float(ic.mean() / ic.std(ddof=1) * math.sqrt(252)) if len(ic) > 1 and ic.std(ddof=1) > 0 else None
    return result, diagnostics


def _factor_returns_variants(panel, factor, quantiles, cost_bps, min_stocks, direction=1, rebalance="daily", max_industry_weight=0.0):
    """Calculate multiple quantile portfolios with one cross-sectional sort per date."""
    quantiles = tuple(float(q) for q in quantiles)
    for q in quantiles:
        _validate_parameters(q, cost_bps, min_stocks)
    data = _prepare_panel(panel)
    if factor not in data.columns:
        return {q: pd.DataFrame() for q in quantiles}
    rows = {q: [] for q in quantiles}; previous = {q: (set(), set()) for q in quantiles}; rebalance_dates = _rebalance_date_set(data.date.unique(), rebalance)
    for date, group in data.groupby("date", sort=True):
        g = group.dropna(subset=[factor, "close"])
        if len(g) < min_stocks:
            continue
        ranked = g.assign(_score=g[factor] * direction).sort_values("_score", ascending=False)
        benchmark = ranked["forward_return"].mean(); ic = g[factor].corr(g["forward_return"], method="spearman")
        by_code = g.set_index("code")
        for q in quantiles:
            n = max(1, int(len(ranked) * q)); prev_long, prev_short = previous[q]
            if pd.Timestamp(date) in rebalance_dates or not prev_long:
                long_set = _select_with_industry_cap(ranked, n, max_industry_weight)
                short_set = _select_with_industry_cap(ranked.iloc[::-1], n, max_industry_weight)
                long_turnover = 1.0 if not prev_long else 1 - len(long_set & prev_long) / max(1, len(long_set))
                short_turnover = 1.0 if not prev_short else 1 - len(short_set & prev_short) / max(1, len(short_set))
                previous[q] = (long_set, short_set)
            else:
                long_set, short_set = prev_long, prev_short; long_turnover = short_turnover = 0.0
            long_return = by_code.reindex(list(long_set))["forward_return"].mean(); short_return = by_code.reindex(list(short_set))["forward_return"].mean()
            rows[q].append((date, long_return, short_return, benchmark, long_turnover, short_turnover, ic))
    results = {}
    bps = cost_bps / 10000.0
    for q, values in rows.items():
        if not values:
            results[q] = pd.DataFrame(); continue
        result = pd.DataFrame(values, columns=["date", "top_gross", "bottom_gross", "benchmark", "long_turnover", "short_turnover", "ic"]).set_index("date")
        result["long_cost"] = result.long_turnover * bps; result["short_cost"] = result.short_turnover * bps
        result["long_only"] = result.top_gross - result.long_cost; result["long_short_gross"] = result.top_gross - result.bottom_gross
        result["long_short"] = result.long_short_gross - result.long_cost - result.short_cost
        results[q] = result
    return results


def _split_metrics(series, oos_ratio, annualization=252.0):
    _validate_parameters(.2, 0, 2, oos_ratio, annualization)
    if series.empty:
        empty = _metrics(series, annualization); return {"full": empty, "is": empty, "oos": empty, "oos_decay": None}
    cut = max(1, int(len(series) * (1 - oos_ratio))); ins, oos = series.iloc[:cut], series.iloc[cut:]
    im, om = _metrics(ins, annualization), _metrics(oos, annualization)
    decay = om["annual_return"] - im["annual_return"] if im["annual_return"] is not None and om["annual_return"] is not None else None
    return {"full": _metrics(series, annualization), "is": im, "oos": om, "oos_decay": decay}


def _relative_metrics(strategy, benchmark, oos_ratio, annualization=252.0):
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    result = _split_metrics(aligned.strategy - aligned.benchmark, oos_ratio, annualization)
    excess = aligned.strategy - aligned.benchmark; std = excess.std(ddof=1)
    result["aligned_observations"] = len(aligned); result["tracking_error"] = float(std * math.sqrt(annualization)) if len(excess) > 1 else None
    result["information_ratio"] = float(excess.mean() / std * math.sqrt(annualization)) if len(excess) > 1 and std > 0 else None
    return result


def run(config: Config) -> dict:
    _validate_parameters(config.quantile, config.cost_bps, config.min_stocks, config.oos_ratio, config.annualization)
    panel = _prepare_panel(_read_snapshots(Path(config.data_dir))); output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True); results = []
    directions = dict(config.factor_directions)
    for factor in config.factors:
        direction = int(directions.get(factor, 1))
        normal, diagnostics = _factor_returns(panel, factor, config.quantile, config.cost_bps, config.min_stocks, direction, config.rebalance, config.max_industry_weight)
        if normal.empty:
            results.append({"factor": factor, "diagnostics": diagnostics}); continue
        long_only, long_short, benchmark = normal.long_only, normal.long_short, normal.benchmark
        doubled, _ = _factor_returns(panel, factor, config.quantile, config.cost_bps * 2, config.min_stocks, direction, config.rebalance, config.max_industry_weight)
        cost_series = doubled["long_only"] if not doubled.empty else pd.Series(dtype=float)
        quantile_results = {}
        for q in config.robustness_quantiles:
            variant, _ = _factor_returns(panel, factor, q, config.cost_bps, config.min_stocks, direction, config.rebalance, config.max_industry_weight)
            quantile_results[str(q)] = _split_metrics(variant["long_only"] if not variant.empty else pd.Series(dtype=float), config.oos_ratio, config.annualization)
        robustness = {"cost_x2": _split_metrics(cost_series, config.oos_ratio, config.annualization), "quantiles": quantile_results}
        half = max(1, len(normal) // 2)
        cost_x2 = robustness["cost_x2"]
        results.append({"factor": factor, "diagnostics": diagnostics, "long_only": _split_metrics(long_only, config.oos_ratio, config.annualization), "long_short": _split_metrics(long_short, config.oos_ratio, config.annualization), "benchmark": _split_metrics(benchmark, config.oos_ratio, config.annualization), "vs_benchmark": _relative_metrics(long_only, benchmark, config.oos_ratio, config.annualization), "robustness": robustness, "cost_x2_total_return": cost_x2["full"]["total_return"], "first_half": _metrics(long_only.iloc[:half], config.annualization), "second_half": _metrics(long_only.iloc[half:], config.annualization)})
        normal.assign(factor=factor).to_csv(output / f"{factor}_daily_returns.csv")
    manifest = {"schema_version": 2, "config": asdict(config), "rows": len(panel), "date_min": str(panel.date.min().date()), "date_max": str(panel.date.max().date()), "columns": list(panel.columns), "results": results}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    (output / "factor_backtest_report.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, default=str), encoding="utf-8"); _write_markdown(manifest, output / "factor_backtest_report.md"); return manifest


def _fmt(v): return "NA" if v is None else f"{v:.4f}"

def _write_markdown(report, path):
    lines = ["# Factor Quant Backtest Report", "", f"- Data: `{report['date_min']} .. {report['date_max']}`", f"- Rows: {report['rows']}", f"- Manifest SHA256: `{report['manifest_sha256']}`", "", "| Factor | IC mean | IC IR | OOS annual | OOS Sharpe | OOS max DD | OOS excess annual | OOS IR | Cost x2 OOS | Turnover |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["results"]:
        if "long_only" not in item: lines.append(f"| {item['factor']} | NA | NA | NA | NA | NA | NA | NA | NA | NA |"); continue
        oos, rel = item["long_only"]["oos"], item["vs_benchmark"]["oos"]; c2 = item["robustness"]["cost_x2"]["oos"]
        lines.append(f"| {item['factor']} | {_fmt(item['diagnostics']['ic_mean'])} | {_fmt(item['diagnostics']['ic_ir'])} | {_fmt(oos['annual_return'])} | {_fmt(oos['sharpe'])} | {_fmt(oos['max_drawdown'])} | {_fmt(rel['annual_return'])} | {_fmt(item['vs_benchmark']['information_ratio'])} | {_fmt(c2['total_return'])} | {_fmt(item['diagnostics']['mean_long_turnover'])} |")
    lines += ["", "## Interpretation", "", "Signal portfolio research report. Returns use next available snapshot close, explicit long/short turnover costs, chronological IS/OOS split, quantile and cost stress tests. It is not an order-level broker simulation; review data version, survivorship, capacity and execution assumptions before deployment."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir", default="../data_warehouse/feature_store"); p.add_argument("--output-dir", default="generated/factor_backtest"); p.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS); p.add_argument("--quantile", type=float, default=.2); p.add_argument("--cost-bps", type=float, default=15.); p.add_argument("--min-stocks", type=int, default=30); p.add_argument("--oos-ratio", type=float, default=DEFAULT_PROTOCOL.test_ratio); p.add_argument("--annualization", type=float, default=252.); p.add_argument("--rebalance", choices=("daily", "weekly", "monthly"), default="daily")
    a = p.parse_args(argv); r = run(Config(a.data_dir, a.output_dir, tuple(a.factors), a.quantile, a.cost_bps, a.min_stocks, a.oos_ratio, annualization=a.annualization, rebalance=a.rebalance)); print(json.dumps({"output_dir": a.output_dir, "manifest_sha256": r["manifest_sha256"], "factors": len(r["results"])}, ensure_ascii=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
