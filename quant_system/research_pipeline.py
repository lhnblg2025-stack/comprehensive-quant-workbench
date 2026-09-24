"""Config-driven reproducible research pipeline."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .dual_price_quality import validate_dual_price_panel
from .execution_adapter import run_order_level_backtest
from .factor_backtest_runner import Config, _factor_returns, _factor_returns_variants, _metrics, _prepare_panel, _relative_metrics, _select_with_industry_cap, _split_metrics, run as run_factor_backtest
from .pit_fundamentals import add_fundamental_factors, attach_pit_fundamentals, attach_pit_industry, coverage_summary, industry_neutralize
from .research_contracts import QualityResult, make_forward_return, resolve_factor_specs, validate_panel, write_manifest
from .research_enhancements import reconcile_signal_execution
from .production_controls import freeze_research_release, verify_research_release, validate_corporate_actions, build_research_evidence


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("research_pipeline requires PyYAML") from exc
    with Path(path).open(encoding="utf-8") as fh:
        result = yaml.safe_load(fh)
    if not isinstance(result, dict):
        raise ValueError("research config must be a mapping")
    return result


def rolling_windows(n_dates: int, train: int, validation: int, oos: int, step: int, purge: int = 0, embargo: int = 0) -> list[dict[str, tuple[int, int]]]:
    if min(train, validation, oos, step) <= 0 or min(purge, embargo) < 0:
        raise ValueError("window sizes must be positive and purge/embargo non-negative")
    windows = []
    start = 0
    while start + train + validation + purge + embargo + oos <= n_dates:
        train_end = start + train
        validation_end = train_end + validation
        oos_start = validation_end + purge + embargo
        windows.append({"train": (start, train_end), "validation": (train_end, validation_end), "oos": (oos_start, oos_start + oos)})
        start += step
    return windows


def _load_optional_table(path_value: Any, required: set[str]) -> tuple[pd.DataFrame | None, str | None]:
    if not path_value:
        return None, None
    path = Path(str(path_value))
    if not path.is_file():
        return None, f"missing:{path}"
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        return None, f"missing_columns:{','.join(sorted(missing))}"
    return frame, None


def _apply_membership(panel: pd.DataFrame, membership: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if membership is None:
        return panel, {"status": "not_configured", "removed_rows": 0}
    data = membership.copy()
    data["code"] = data["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    data["start_date"] = pd.to_datetime(data["start_date"], errors="coerce")
    data["end_date"] = pd.to_datetime(data.get("end_date"), errors="coerce") if "end_date" in data else pd.NaT
    joined = panel.copy(); joined["date"] = pd.to_datetime(joined["date"]); joined["code"] = joined["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    joined = joined.merge(data[["code", "start_date", "end_date"]], on="code", how="left")
    valid = joined["start_date"].isna() | ((joined["date"] >= joined["start_date"]) & (joined["end_date"].isna() | (joined["date"] <= joined["end_date"])))
    filtered = joined.loc[valid, panel.columns].copy()
    return filtered, {"status": "applied", "removed_rows": int((~valid).sum()), "membership_rows": int(len(data))}


def _apply_corporate_actions(panel: pd.DataFrame, actions: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if actions is None:
        return panel, {"status": "not_configured", "applied_rows": 0}
    data = actions.copy()
    data["code"] = data["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    data["ex_date"] = pd.to_datetime(data["ex_date"], errors="coerce")
    ratio = pd.to_numeric(data.get("ratio", 1.0), errors="coerce").fillna(1.0) if "ratio" in data else pd.Series(1.0, index=data.index)
    data["_ratio"] = ratio.where(ratio > 0, 1.0)
    out = panel.copy(); out["date"] = pd.to_datetime(out["date"])
    applied = 0
    for row in data.dropna(subset=["code", "ex_date"])[["code", "ex_date", "action_type", "_ratio"]].itertuples(index=False, name=None):
        code, ex_date, action_type, action_ratio = row
        mask = (out["code"].astype(str) == str(code)) & (out["date"] < ex_date)
        if action_type in {"split", "bonus", "stock_split"}:
            out.loc[mask, "close"] = out.loc[mask, "close"] / action_ratio
            applied += int(mask.sum())
    return out, {"status": "applied", "applied_rows": applied, "action_rows": int(len(data))}


def _targets_from_factor(panel: pd.DataFrame, factor: str, dates: list[pd.Timestamp], quantile: float, direction: int = 1, rebalance: str = "daily", max_industry_weight: float = 0.0) -> dict[pd.Timestamp, dict[str, float]]:
    date_index = pd.DatetimeIndex(dates)
    if rebalance == "weekly":
        selected_dates = pd.Series(date_index, index=date_index).groupby(date_index.to_period("W")).max().tolist()
    elif rebalance == "monthly":
        selected_dates = pd.Series(date_index, index=date_index).groupby(date_index.to_period("M")).max().tolist()
    else:
        selected_dates = list(date_index)
    targets = {}
    for date in selected_dates:
        group = panel[pd.to_datetime(panel["date"]) == date].dropna(subset=[factor, "close"])
        if group.empty:
            continue
        n = max(1, int(len(group) * quantile)); ranked = group.assign(_score=group[factor] * direction).sort_values("_score", ascending=False)
        selected = _select_with_industry_cap(ranked, n, max_industry_weight)
        targets[date] = {str(code): 1.0 / len(selected) for code in selected}
    return targets


def _rolling_oos(panel: pd.DataFrame, dates: pd.DatetimeIndex, windows: list[dict[str, tuple[int, int]]], factors: tuple[str, ...], directions: dict[str, int], quantiles: tuple[float, ...], cost_bps: float, min_stocks: int, annualization: float, rebalance: str = "daily", max_industry_weight: float = 0.0) -> list[dict[str, Any]]:
    cached = {}
    for factor in factors:
        variants = _factor_returns_variants(panel, factor, quantiles, cost_bps, min_stocks, directions.get(factor, 1), rebalance, max_industry_weight)
        for quantile, frame in variants.items():
            cached[(factor, quantile)] = frame
    results = []
    for number, window in enumerate(windows, 1):
        train_end = dates[window["train"][1] - 1]
        validation_start, validation_end = dates[window["validation"][0]], dates[window["validation"][1] - 1]
        oos_start, oos_end = dates[window["oos"][0]], dates[window["oos"][1] - 1]
        selected: dict[str, float] = {}; factor_results: dict[str, Any] = {}
        for factor in factors:
            candidates = []
            for quantile in quantiles:
                variant = cached[(factor, quantile)]
                validation_returns = variant.loc[validation_start:validation_end, "long_only"] if not variant.empty else pd.Series(dtype=float)
                candidates.append((float(_metrics(validation_returns, annualization).get("sharpe") or -float("inf")), quantile))
            score, chosen = max(candidates, key=lambda item: item[0])
            selected[factor] = chosen
            chosen_frame = cached[(factor, chosen)]
            oos_returns = chosen_frame.loc[oos_start:oos_end, "long_only"] if not chosen_frame.empty else pd.Series(dtype=float)
            factor_results[factor] = {"selected_quantile": chosen, "validation_sharpe": None if score == -float("inf") else score, "oos": _metrics(oos_returns, annualization), "oos_observations": int(len(oos_returns))}
        results.append({"window_id": number, "dates": {"train_end": str(train_end.date()), "validation": [str(validation_start.date()), str(validation_end.date())], "oos": [str(oos_start.date()), str(oos_end.date())]}, "selected_parameters": selected, "factors": factor_results})
    return results


def run_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    data = config.get("data", {})
    panel_dir = Path(data["panel_dir"])
    panel_file = data.get("panel_file")
    files = [Path(panel_file)] if panel_file else sorted(panel_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet snapshots under {panel_dir}")
    frames = [pd.read_parquet(p) for p in files]
    panel = pd.concat(frames, ignore_index=True)
    calendar_path = data.get("calendar")
    calendar = None
    if calendar_path and Path(calendar_path).is_file():
        cal = pd.read_parquet(calendar_path) if Path(calendar_path).suffix == ".parquet" else pd.read_csv(calendar_path)
        calendar = pd.to_datetime(cal.iloc[:, 0], errors="coerce").dropna()
    membership, membership_error = _load_optional_table(data.get("historical_membership"), {"code", "start_date"})
    panel, membership_summary = _apply_membership(panel, membership)
    actions, actions_error = _load_optional_table(data.get("corporate_actions"), set())
    if actions is not None:
        actions = actions.rename(columns={"symbol": "code", "date": "ex_date"})
        missing_action_cols = {"code", "ex_date", "action_type"} - set(actions.columns)
        if missing_action_cols:
            actions_error = f"missing_columns:{','.join(sorted(missing_action_cols))}"; actions = None
    panel, action_summary = _apply_corporate_actions(panel, actions)
    # Freeze the exact inputs consumed by this run. Research is production-grade
    # only when it can be replayed byte-for-byte; this release never authorizes live trading.
    release_path = Path(config.get("outputs", {}).get("release_manifest", "")) if config.get("outputs", {}).get("release_manifest") else Path(config.get("outputs", {}).get("root", "generated/runs")) / config.get("experiment", {}).get("id", "research") / "research_release.json"
    release_inputs = [*files]
    for configured_input in (data.get("calendar"), data.get("corporate_actions"), config.get("pit", {}).get("fundamentals"), config.get("pit", {}).get("industry_history")):
        if configured_input and Path(str(configured_input)).is_file():
            release_inputs.append(Path(str(configured_input)))
    release = freeze_research_release(release_path, release_inputs, as_of=str(pd.to_datetime(panel["date"]).max().date()), provider=str(data.get("provider", "versioned-local-parquet")), limitations=["research_only", "live_oms_not_configured"])
    release_check = verify_research_release(release)
    if release_check["status"] != "PASS":
        raise ValueError("research_release_verification_blocked:" + ",".join(release_check["errors"]))
    pit = config.get("pit", {})
    pit_required = bool(pit.get("required", False))
    pit_inputs: list[Path] = []
    pit_summary: dict[str, Any] = {"status": "not_configured"}
    calendar_for_pit = calendar if calendar is not None else pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    fundamental_path = pit.get("fundamentals")
    industry_path = pit.get("industry_history")
    if pit_required and not fundamental_path:
        raise ValueError("pit_fundamentals_required")
    if pit_required and not industry_path:
        raise ValueError("pit_industry_history_required")
    if fundamental_path:
        fundamentals, fundamentals_error = _load_optional_table(fundamental_path, {"code", "report_period", "announcement_date"})
        if fundamentals is None:
            raise ValueError(f"pit_fundamentals_unavailable:{fundamentals_error}")
        panel = attach_pit_fundamentals(panel, fundamentals, calendar_for_pit, lag_sessions=int(pit.get("availability_lag_sessions", 1)))
        panel = add_fundamental_factors(panel)
        pit_inputs.append(Path(str(fundamental_path)))
        pit_summary["fundamentals"] = {"status": "applied", "releases": int(len(fundamentals)), "lag_sessions": int(pit.get("availability_lag_sessions", 1))}
    if industry_path:
        industries, industries_error = _load_optional_table(industry_path, {"code", "industry", "effective_date"})
        if industries is None:
            raise ValueError(f"pit_industry_unavailable:{industries_error}")
        panel = attach_pit_industry(panel, industries)
        pit_inputs.append(Path(str(industry_path)))
        pit_summary["industry"] = {"status": "applied", "history_rows": int(len(industries))}
    neutralization = str(config.get("factors", {}).get("neutralization", "none"))
    neutralize_names = tuple(config.get("factors", {}).get("industry_neutralize", ()))
    if neutralization == "industry":
        if not industry_path:
            raise ValueError("industry_neutralization_requires_pit_industry_history")
        if not neutralize_names:
            raise ValueError("industry_neutralization_requires_factor_list")
        panel = industry_neutralize(panel, neutralize_names, min_industry_size=int(pit.get("min_industry_size", 3)))
        pit_summary["industry_neutralization"] = {"status": "applied", "factors": list(neutralize_names), "min_industry_size": int(pit.get("min_industry_size", 3))}
    elif neutralization != "none":
        raise ValueError(f"unsupported_neutralization:{neutralization}")
    if pit_required:
        factor_names = tuple(config.get("factors", {}).get("names", ()))
        pit_summary["coverage"] = coverage_summary(panel, factor_names, min_stocks=int(data.get("min_stocks", 500)))
        if pit_summary["coverage"]["status"] != "PASS":
            raise ValueError("pit_factor_coverage_below_minimum")
        pit_summary["status"] = "PASS"
    quality = validate_panel(panel, calendar)
    dual_price_quality = None
    if {"hfq_close", "raw_close"}.issubset(panel.columns):
        dual_price_quality = validate_dual_price_panel(panel, min_symbols=int(data.get("min_stocks", 1)))
        if dual_price_quality["status"] == "BLOCK":
            raise ValueError(f"dual-price quality gate blocked: {dual_price_quality['errors']}")
    if quality.status == "BLOCK":
        raise ValueError(f"data quality gate blocked: {quality.errors}")
    label_summary = {}
    for horizon in config.get("labels", {}).get("horizons", [1]):
        labels = make_forward_return(panel, int(horizon), calendar)
        label_summary[str(horizon)] = {"rows": int(len(labels.frame)), "valid": int(labels.frame["forward_return"].notna().sum())}
    validation = config.get("validation", {})
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    windows = rolling_windows(len(dates), int(validation.get("train_days", 504)), int(validation.get("validation_days", 126)), int(validation.get("oos_days", 126)), int(validation.get("step_days", 63)), int(validation.get("purge_days", 0)), int(validation.get("embargo_days", 0)))
    portfolio = config.get("portfolio", {})
    execution = config.get("execution", {})
    output_root = Path(config.get("outputs", {}).get("root", "generated/runs")) / config.get("experiment", {}).get("id", "research")
    factor_specs = resolve_factor_specs(config)
    factors = tuple(spec.name for spec in factor_specs)
    directions = dict((spec.name, spec.direction) for spec in factor_specs)
    cost_bps = float(execution.get("commission_bps", 0)) + float(execution.get("stamp_duty_bps", 0)) + float(execution.get("transfer_fee_bps", 0)) + float(execution.get("slippage_bps", 0))
    output_root.mkdir(parents=True, exist_ok=True)
    filtered_panel_path = output_root / "filtered_panel.parquet"
    panel.to_parquet(filtered_panel_path, index=False)
    result = run_factor_backtest(Config(str(output_root), str(output_root / "factor"), factors, float(portfolio.get("default_quantile", .2)), cost_bps, int(config.get("data", {}).get("min_stocks", 30)), float(validation.get("oos_ratio", .3)), annualization=float(validation.get("annualization", 252.0)), factor_directions=tuple(directions.items()), rebalance=str(portfolio.get("rebalance", "daily")), max_industry_weight=float(portfolio.get("max_industry_weight", 0.0))))
    prepared_panel = _prepare_panel(panel)
    window_results = _rolling_oos(prepared_panel, dates, windows, factors, directions, tuple(float(q) for q in portfolio.get("quantiles", [portfolio.get("default_quantile", .2)])), cost_bps, int(config.get("data", {}).get("min_stocks", 30)), float(validation.get("annualization", 252.0)), str(portfolio.get("rebalance", "daily")), float(portfolio.get("max_industry_weight", 0.0)))
    actions, actions_error = _load_optional_table(data.get("corporate_actions"), {"code", "ex_date", "action_type"})
    history_summary = {"membership_rows": int(len(membership)) if membership is not None else 0, "membership_removed_rows": int(membership_summary.get("removed_rows", 0)), "corporate_action_rows": int(len(actions)) if actions is not None else 0, "corporate_action_applied_rows": int(action_summary.get("applied_rows", 0)), "membership_status": membership_summary.get("status", membership_error or "not_configured"), "corporate_actions_status": action_summary.get("status", actions_error or "not_configured")}
    corporate_action_evidence = {"status": "not_configured", "errors": []}
    if actions is not None:
        corporate_action_evidence = validate_corporate_actions(actions, release_as_of=release["as_of"])
        if corporate_action_evidence["status"] != "PASS" and bool(config.get("pit", {}).get("required", False)):
            raise ValueError("corporate_actions_evidence_blocked:" + ",".join(corporate_action_evidence["errors"]))
    benchmark_cfg = config.get("benchmark", {})
    benchmark_status = "unavailable"
    benchmark_series = None
    benchmark_summary = {"type": benchmark_cfg.get("type", "universe_equal_weight"), "matched": 0, "missing": 0}
    if benchmark_cfg.get("type") == "custom_return_series" and benchmark_cfg.get("source"):
        benchmark_path = Path(str(benchmark_cfg["source"]))
        if benchmark_path.is_file():
            benchmark = pd.read_parquet(benchmark_path) if benchmark_path.suffix == ".parquet" else pd.read_csv(benchmark_path)
            if {"date", "return"}.issubset(benchmark.columns):
                benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
                valid = benchmark.dropna(subset=["date", "return"])
                valid = valid.drop_duplicates("date").set_index("date")
                benchmark_series = pd.to_numeric(valid["return"], errors="coerce")
                matched = int(benchmark_series.reindex(dates).notna().sum())
                benchmark_summary.update({"matched": matched, "missing": int(len(dates) - matched), "source": str(benchmark_path)})
                if bool(benchmark_cfg.get("require_alignment", False)) and matched != len(dates):
                    raise ValueError(f"benchmark_alignment_required:missing={len(dates) - matched}")
                benchmark_status = "available" if matched else "unavailable"
    elif benchmark_cfg.get("type") == "universe_equal_weight":
        benchmark_status = "available"
        benchmark_summary["matched"] = int(len(dates))
    if benchmark_series is not None:
        for item in result["results"]:
            factor = item.get("factor")
            daily_path = output_root / "factor" / f"{factor}_daily_returns.csv"
            if not daily_path.exists() or "long_only" not in item:
                continue
            daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            aligned_benchmark = benchmark_series.reindex(daily.index)
            item["benchmark"] = _split_metrics(aligned_benchmark, float(validation.get("oos_ratio", .3)), float(validation.get("annualization", 252.0)))
            item["vs_benchmark"] = _relative_metrics(daily["long_only"], aligned_benchmark, float(validation.get("oos_ratio", .3)), float(validation.get("annualization", 252.0)))
    raw_columns = {"raw_open", "raw_high", "raw_low", "raw_close"}
    execution_result = {"status": "unavailable", "reason": "missing_raw_ohlc_or_no_factor"}
    if execution.get("mode") in {"research_then_order_level", "order_level"} and raw_columns.issubset(panel.columns) and factors:
        execution_factor = factors[0]
        all_execution_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
        if windows:
            execution_start, execution_end = windows[-1]["oos"]
            execution_dates = list(all_execution_dates[execution_start:execution_end])
            context_start = all_execution_dates[max(0, execution_start - 1)]
            context_end = all_execution_dates[min(len(all_execution_dates) - 1, execution_end)]
            execution_panel = panel[(pd.to_datetime(panel["date"]) >= context_start) & (pd.to_datetime(panel["date"]) <= context_end)].copy()
        else:
            execution_dates = list(all_execution_dates[:-1])
            execution_panel = panel.copy()
        execution_panel[["open", "high", "low", "close"]] = execution_panel[["raw_open", "raw_high", "raw_low", "raw_close"]].to_numpy()
        execution_quantile = float(window_results[-1]["selected_parameters"].get(execution_factor, portfolio.get("default_quantile", .2))) if window_results else float(portfolio.get("default_quantile", .2))
        targets = _targets_from_factor(panel, execution_factor, execution_dates, execution_quantile, directions.get(execution_factor, 1), str(portfolio.get("rebalance", "daily")), float(portfolio.get("max_industry_weight", 0.0)))
        investable = 1.0 - float(portfolio.get("cash_buffer", 0.0))
        targets = {date: {symbol: weight * investable for symbol, weight in weights.items()} for date, weights in targets.items()}
        if targets:
            try:
                action_frame = None
                if actions is not None:
                    action_frame = actions.rename(columns={"code": "symbol", "ex_date": "date"})
                order_result = run_order_level_backtest(execution_panel, targets, capital=float(execution.get("initial_capital", 1_000_000.0)), commission_rate=float(execution.get("commission_bps", 0)) / 10000.0, slippage_rate=float(execution.get("slippage_bps", 0)) / 10000.0, end_mode=str(execution.get("end_mode", "mark_to_market")), corporate_actions=action_frame)
                final_equity = order_result.equity_curve.iloc[-1] if not order_result.equity_curve.empty else {}
                signal_frame = _factor_returns_variants(prepared_panel, execution_factor, [execution_quantile], cost_bps, int(data.get("min_stocks", 30)), directions.get(execution_factor, 1), str(portfolio.get("rebalance", "daily")))[execution_quantile]
                signal_same_period = signal_frame.loc[execution_dates[0]:execution_dates[-1], "long_only"] if not signal_frame.empty else pd.Series(dtype=float)
                reconciliation = reconcile_signal_execution(signal_same_period, order_result.equity_curve["total_value"], initial_capital=float(execution.get("initial_capital", 1_000_000.0)))
                execution_result = {"status": "available", "factor": execution_factor, "selected_quantile": execution_quantile, "end_mode": str(execution.get("end_mode", "mark_to_market")), "orders": len(order_result.orders), "trades": len(order_result.trades), "equity_observations": len(order_result.equity_curve), "total_return": order_result.total_return, "max_drawdown": order_result.max_drawdown, "force_close_blocks": len(order_result.force_close_blocks), "final_cash": float(final_equity.get("cash", 0.0)), "final_position_value": float(final_equity.get("position_value", 0.0)), "corporate_actions_applied": len(getattr(order_result, "applied_corporate_actions", [])), "reconciliation": reconciliation}
            except Exception as exc:
                execution_result = {"status": "failed", "reason": str(exc), "factor": execution_factor}
    result["execution_result"] = execution_result
    result["benchmark_status"] = benchmark_status
    result["benchmark_summary"] = benchmark_summary
    result["history_summary"] = history_summary
    result["pit_summary"] = pit_summary
    result["data_release"] = {"release_id": release["release_id"], "status": release["status"], "evidence_sha256": release["evidence_sha256"], "manifest": str(release_path)}
    result["corporate_action_evidence"] = corporate_action_evidence
    result["research_evidence"] = build_research_evidence(release=release, quality={"status": quality.status, "errors": list(quality.errors), "warnings": list(quality.warnings)}, pit=pit_summary, corporate_actions=corporate_action_evidence, factors=factors)
    result["dual_price_quality"] = dual_price_quality
    result["factor_specs"] = [spec.__dict__ for spec in factor_specs]
    result["label_summary"] = label_summary
    result["rolling_windows"] = windows
    result["rolling_window_results"] = window_results
    result["rolling_oos_status"] = "validated" if len(windows) >= int(validation.get("min_oos_windows", 1)) else "insufficient_history"
    result["quality_gate"] = {**quality.__dict__, "status": quality.status if result["rolling_oos_status"] == "validated" else "WARN", "rolling_oos_status": result["rolling_oos_status"]}
    report_path = output_root / "research_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    artifacts = [output_root / "factor" / "factor_backtest_report.json", output_root / "factor" / "factor_backtest_report.md"]
    artifacts.extend((output_root / "factor").glob("*_daily_returns.csv"))
    manifest_quality = QualityResult("WARN" if result["rolling_oos_status"] != "validated" else quality.status, quality.checks, quality.errors, quality.warnings + (("insufficient_history",) if result["rolling_oos_status"] != "validated" else ()))
    manifest = write_manifest(output_root, config=config, inputs=[*files, *pit_inputs], quality=manifest_quality, seed=int(config.get("experiment", {}).get("random_seed", 42)), root=Path(__file__).resolve().parents[1], artifacts=artifacts)
    result["experiment_manifest_sha256"] = manifest["manifest_sha256"]
    report_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = run_config(args.config)
    print(json.dumps({"status": "ok", "manifest_sha256": result["experiment_manifest_sha256"], "windows": len(result["rolling_windows"])}, ensure_ascii=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
