#!/usr/bin/env python3
"""Run a reproducible 5-year train / 3-year out-of-sample price backtest.

The input is the strict ``long_price_panel/v2`` artifact built from genuine
unadjusted RAW shards plus cumulative HFQ factors.  The model is fitted only on
2018-2022 and evaluated only on 2023-2025; the 2026 tail is not used for either
model selection or the reported OOS metrics.

This is deliberately a price-only research backtest.  HFQ OHLC drives returns
and signal construction (so dividends/splits are represented in total-return
space), while genuine RAW bars drive suspension and limit-up/down gates.  It is
not an authoritative production order replay because the repository does not
yet contain a complete corporate-action ledger, ST state, or point-in-time
index membership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.backtest_service import data_gate
from quant_system.portfolio_holding_backtest import HoldingConfig, run_continuous_portfolio_backtest

FEATURES = (
    "mom_20",
    "mom_60",
    "mom_120",
    "vol_20",
    "amp_20",
    "volume_ratio_20",
    "turnover_20",
    "log_amount_20",
)
FEATURE_COLUMNS = tuple(f"{name}__rank" for name in FEATURES)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
REQUIRED_COLUMNS = {
    "date", "code", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount",
    "adjust_factor",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annualized_metrics(values: pd.Series) -> dict[str, Any]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    if returns.empty:
        return {
            "observations": 0,
            "total_return": None,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }
    curve = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    drawdown = curve / curve.cummax() - 1.0
    return {
        "observations": int(len(returns)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and curve.iloc[-1] > 0 else None,
        "annual_volatility": float(std * math.sqrt(252.0)),
        "sharpe": float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def rank_ic(frame: pd.DataFrame, score_column: str, target_column: str) -> pd.Series:
    sample = frame[["date", score_column, target_column]].dropna().copy()
    if sample.empty:
        return pd.Series(dtype=float)

    def _one(group: pd.DataFrame) -> float:
        if len(group) < 10:
            return np.nan
        scores = group[score_column].rank(method="average")
        targets = group[target_column].rank(method="average")
        if scores.std(ddof=0) == 0 or targets.std(ddof=0) == 0:
            return np.nan
        return float(scores.corr(targets))

    return sample.groupby("date", sort=True)[[score_column, target_column]].apply(_one).dropna()


def safe_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else None


def safe_std(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1)) if len(clean) > 1 else None


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)

    grouped = data.groupby("code", sort=False, group_keys=False)
    data["hfq_return_1d"] = grouped["hfq_close"].pct_change()
    for window in (20, 60, 120):
        data[f"mom_{window}"] = grouped["hfq_close"].pct_change(window)
    data["amplitude_1d"] = data["hfq_high"] / data["hfq_low"] - 1.0
    data["vol_20"] = grouped["hfq_return_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    data["amp_20"] = grouped["amplitude_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    data["volume_ma_20"] = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    data["amount_ma_20"] = grouped["amount"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    data["volume_ratio_20"] = data["volume"] / data["volume_ma_20"].replace(0.0, np.nan)
    data["log_amount_20"] = np.log1p(data["amount_ma_20"].clip(lower=0.0))
    if "turnover_rate" in data.columns:
        data["turnover_20"] = grouped["turnover_rate"].transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
    else:
        data["turnover_20"] = np.nan
    data["target_fwd5"] = grouped["hfq_close"].shift(-5) / data["hfq_close"] - 1.0
    data["target_date"] = grouped["date"].shift(-5)

    # Point-in-time cross-sectional ranks: every date uses only that date's
    # known cross-section at close.  The target is demeaned cross-sectionally
    # only while fitting, never when predicting.
    for name in FEATURES:
        data[f"{name}__rank"] = data.groupby("date")[name].rank(pct=True, method="average") - 0.5
    data["target_cs"] = data["target_fwd5"] - data.groupby("date")["target_fwd5"].transform("mean")

    # True RAW-derived trade gates.  Missing bars are treated as suspended in
    # the execution grid below; limit gates use the prior raw close.
    data["raw_prev_close"] = grouped["raw_close"].shift(1)
    data["limit_up_locked"] = data["raw_open"] >= data["raw_prev_close"] * 1.095
    data["limit_down_locked"] = data["raw_open"] <= data["raw_prev_close"] * 0.905
    data["suspended"] = data["volume"].le(0.0) | data["raw_open"].le(0.0)
    return data


def select_alpha(train: pd.DataFrame, max_folds: int = 4, block_days: int = 126,
                 embargo_days: int = 5) -> dict[str, Any]:
    dates = np.array(sorted(pd.to_datetime(train["date"]).unique()))
    folds: list[tuple[int, int]] = []
    for end in range(len(dates), block_days, -block_days):
        start = end - block_days
        if start < 252:
            break
        folds.append((start, end))
    folds = folds[:max_folds][::-1]
    if not folds:
        raise ValueError("insufficient_alpha_validation_folds")

    fold_results: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    fold_details: list[dict[str, Any]] = []
    for start, end in folds:
        validation_dates = set(dates[start:end])
        training_dates = set(dates[: max(0, start - embargo_days)])
        fit_rows = train[train["date"].isin(training_dates)]
        validation_rows = train[train["date"].isin(validation_dates)]
        if fit_rows.empty or validation_rows.empty:
            continue
        fold_row: dict[str, Any] = {
            "fit_start": str(min(training_dates).date()) if training_dates else None,
            "fit_end": str(max(training_dates).date()) if training_dates else None,
            "validation_start": str(min(validation_dates).date()) if validation_dates else None,
            "validation_end": str(max(validation_dates).date()) if validation_dates else None,
            "alpha_ic": {},
        }
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(fit_rows[list(FEATURE_COLUMNS)], fit_rows["target_cs"])
            scored = validation_rows[["date", "target_cs"]].copy()
            scored["_prediction"] = model.predict(validation_rows[list(FEATURE_COLUMNS)])
            ic = rank_ic(scored, "_prediction", "target_cs")
            value = safe_mean(ic)
            fold_results[alpha].append(float(value) if value is not None else np.nan)
            fold_row["alpha_ic"][str(alpha)] = value
        fold_details.append(fold_row)

    summary: dict[str, Any] = {}
    for alpha, values in fold_results.items():
        clean = [value for value in values if value is not None and math.isfinite(value)]
        summary[str(alpha)] = {
            "mean_validation_ic": float(np.mean(clean)) if clean else None,
            "folds": values,
        }
    valid = [(alpha, row["mean_validation_ic"]) for alpha, row in summary.items() if row["mean_validation_ic"] is not None]
    if not valid:
        raise ValueError("alpha_validation_failed")
    best_alpha_text = max(valid, key=lambda item: (item[1], -float(item[0])))[0]
    return {
        "grid": list(ALPHAS),
        "folds": fold_details,
        "summary": summary,
        "selected_alpha": float(best_alpha_text),
        "selection_metric": "mean_daily_cross_sectional_rank_ic",
        "embargo_sessions": embargo_days,
    }


def build_execution_grid(test: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(test["date"]).unique()))
    codes = sorted(test["code"].astype(str).str.zfill(6).unique())
    grid = pd.MultiIndex.from_product([calendar, codes], names=["date", "code"]).to_frame(index=False)
    columns = [
        "date", "code", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
        "volume", "amount", "amount_ma_20", "turnover_rate", "signal",
        "suspended", "limit_up_locked", "limit_down_locked",
    ]
    available = [column for column in columns if column in test.columns]
    grid = grid.merge(test[available], on=["date", "code"], how="left")
    grid["has_bar"] = grid["hfq_close"].notna()
    grid = grid.sort_values(["code", "date"]).reset_index(drop=True)
    for column in ("hfq_open", "hfq_high", "hfq_low", "hfq_close", "amount", "amount_ma_20", "turnover_rate"):
        if column in grid.columns:
            grid[column] = grid.groupby("code", sort=False)[column].ffill()
    grid["suspended"] = ((~grid["has_bar"]) | grid["suspended"].eq(True)).astype(bool)
    grid["limit_up_locked"] = grid["limit_up_locked"].eq(True)
    grid["limit_down_locked"] = grid["limit_down_locked"].eq(True)
    grid["volume"] = grid["volume"].fillna(0.0)
    grid["amount"] = grid["amount"].fillna(0.0)
    grid["amount_ma_20"] = grid["amount_ma_20"].fillna(0.0)
    grid["adv_amount_20"] = grid["amount_ma_20"]
    grid["raw_open"] = grid["hfq_open"]
    grid["raw_high"] = grid["hfq_high"]
    grid["raw_low"] = grid["hfq_low"]
    grid["raw_close"] = grid["hfq_close"]
    grid = grid.dropna(subset=["raw_open", "raw_close"]).sort_values(["date", "code"]).reset_index(drop=True)
    return grid


def benchmark_metrics(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].between(start, end, inclusive="both")].sort_values("date")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("benchmark_empty_in_test_period")
    returns = close.pct_change()
    return {
        "source": str(path.relative_to(ROOT)),
        "observations": int(len(close)),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        **annualized_metrics(returns),
    }


def quantile_spread(test: pd.DataFrame, quantiles: int = 5) -> dict[str, Any]:
    sample = test[["date", "signal", "target_fwd5"]].dropna().copy()
    if sample.empty:
        return {"observations": 0, "mean_top_minus_bottom": None, "daily_series": []}

    sample["_rank"] = sample.groupby("date")["signal"].rank(method="first")
    sample["_size"] = sample.groupby("date")["signal"].transform("size")
    sample["bucket"] = np.floor(
        (sample["_rank"] - 1.0) / (sample["_size"] / float(quantiles))
    ).clip(upper=float(quantiles - 1))
    sample.loc[sample["_size"] < quantiles, "bucket"] = np.nan
    grouped = sample.dropna(subset=["bucket"]).groupby(["date", "bucket"])["target_fwd5"].mean().unstack()
    if grouped.empty or grouped.shape[1] < 2:
        return {"observations": 0, "mean_top_minus_bottom": None, "daily_series": []}
    spread = grouped.iloc[:, -1] - grouped.iloc[:, 0]
    return {
        "observations": int(len(spread.dropna())),
        "mean_top_minus_bottom": float(spread.mean()),
        "median_top_minus_bottom": float(spread.median()),
        "daily_series": [
            {"date": str(index.date()), "spread": float(value)}
            for index, value in spread.dropna().items()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default="generated/long_price_panel/price_panel.parquet")
    parser.add_argument("--output", default="generated/formal_5y3y_backtest_20260922")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--rebalance-sessions", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--benchmark", default="data_warehouse/market/index_daily_沪深300.parquet")
    args = parser.parse_args()

    panel_path = (ROOT / args.panel).resolve() if not Path(args.panel).is_absolute() else Path(args.panel)
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    benchmark_path = (ROOT / args.benchmark).resolve() if not Path(args.benchmark).is_absolute() else Path(args.benchmark)
    if not panel_path.is_file():
        raise FileNotFoundError(f"panel_missing:{panel_path}")
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"benchmark_missing:{benchmark_path}")

    panel = pd.read_parquet(panel_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(panel.columns))
    if missing_columns:
        raise ValueError("panel_missing_columns:" + ",".join(missing_columns))
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    if panel["date"].isna().any() or panel["code"].nunique() < 800:
        raise ValueError("panel_quality_gate_failed")
    ratio = panel["hfq_close"] / panel["raw_close"]
    factor_gap = float((ratio - panel["adjust_factor"]).abs().max())
    if factor_gap > 1e-4:
        raise ValueError(f"panel_factor_consistency_failed:{factor_gap}")

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    if not (panel["date"].min() <= train_start + pd.Timedelta(days=5)):
        raise ValueError("panel_does_not_start_early_enough_for_train")
    if not (panel["date"].max() >= test_end):
        raise ValueError("panel_does_not_cover_full_test_period")

    features = build_features(panel)
    train_mask = (
        features["date"].between(train_start, train_end, inclusive="both")
        & features["target_date"].notna()
        & (features["target_date"] <= train_end)
    )
    test_mask = features["date"].between(test_start, test_end, inclusive="both")
    train = features.loc[train_mask, ["date", "code", "target_cs", "target_fwd5", *FEATURE_COLUMNS]].dropna().copy()
    test = features.loc[test_mask].copy()
    test_model_rows = test.dropna(subset=list(FEATURE_COLUMNS)).copy()
    if train.empty or test_model_rows.empty:
        raise ValueError("empty_train_or_test_feature_rows")
    if train["date"].max() >= test_start:
        raise ValueError("train_test_overlap")
    if features.loc[train_mask, "target_date"].max() > train_end:
        raise ValueError("train_target_leakage")

    alpha_report = select_alpha(train)
    model = Ridge(alpha=float(alpha_report["selected_alpha"]), fit_intercept=True)
    model.fit(train[list(FEATURE_COLUMNS)], train["target_cs"])
    train = train.assign(signal=model.predict(train[list(FEATURE_COLUMNS)]))
    test_model_rows = test_model_rows.assign(signal=model.predict(test_model_rows[list(FEATURE_COLUMNS)]))
    signal_frame = test_model_rows[["date", "code", "signal", "target_fwd5"]].copy()

    train_ic = rank_ic(train, "signal", "target_cs")
    test_ic = rank_ic(test_model_rows, "signal", "target_fwd5")
    test_spread = quantile_spread(test_model_rows)

    execution_grid = build_execution_grid(test_model_rows)
    config = HoldingConfig(
        holding_sessions=5,
        rebalance_sessions=int(args.rebalance_sessions),
        quantile=float(args.quantile),
        initial_capital=float(args.initial_capital),
        cash_buffer=0.05,
        commission_bps=0.85,
        stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=10.0,
        max_adv_participation=0.10,
        # A-share cash equities trade in 100-share lots.
        lot_size=100,
        min_commission=5.0,
    )
    production_gate = data_gate(panel, require_production=True)
    backtest = run_continuous_portfolio_backtest(execution_grid, config, direction=1)
    equity = backtest["equity"].copy()
    trades = backtest["trades"].copy()
    executed = trades[~trades["blocked"]].copy() if "blocked" in trades.columns else trades.copy()
    if not executed.empty:
        executed["_signed"] = np.where(executed["side"].eq("buy"), executed["shares"], -executed["shares"])
        active_positions_final = int(executed.groupby("code")["_signed"].sum().ne(0).sum())
    else:
        active_positions_final = 0
    metrics = annualized_metrics(equity["total_value"].pct_change())
    benchmark = benchmark_metrics(benchmark_path, test_start, test_end)
    excess_annual = (
        metrics["annual_return"] - benchmark["annual_return"]
        if metrics["annual_return"] is not None and benchmark["annual_return"] is not None
        else None
    )

    output.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output / "equity.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    signal_frame.to_parquet(output / "oos_signals.parquet", index=False)
    model_report = {
        "type": "ridge",
        "selected_alpha": float(alpha_report["selected_alpha"]),
        "features": list(FEATURES),
        "feature_columns": list(FEATURE_COLUMNS),
        "coefficients": {name: float(value) for name, value in zip(FEATURE_COLUMNS, model.coef_)},
        "intercept": float(model.intercept_),
        "alpha_selection": alpha_report,
        "train_rows": int(len(train)),
        "test_rows": int(len(test_model_rows)),
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test_model_rows["date"].nunique()),
    }
    (output / "model.json").write_text(json.dumps(model_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = {
        "schema": "formal-5y3y-backtest/v1",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "dataset": {
            "path": str(panel_path.relative_to(ROOT)),
            "sha256": sha256(panel_path),
            "symbols": int(panel["code"].nunique()),
            "rows": int(len(panel)),
            "date_min": str(panel["date"].min().date()),
            "date_max": str(panel["date"].max().date()),
            "factor_consistency_max_abs_gap": factor_gap,
        },
        "split": {
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_sessions": int(train["date"].nunique()),
            "test_sessions": int(test_model_rows["date"].nunique()),
            "embargo_sessions": 5,
            "test_2026_tail_excluded": True,
        },
        "model": {
            "selected_alpha": float(alpha_report["selected_alpha"]),
            "features": list(FEATURES),
            "selection_metric": "mean_daily_cross_sectional_rank_ic",
        },
        "signal_quality": {
            "train_rank_ic_mean": safe_mean(train_ic),
            "train_rank_ic_std": safe_std(train_ic),
            "train_rank_ic_ir": safe_mean(train_ic) / safe_std(train_ic) if safe_std(train_ic) not in (None, 0.0) else None,
            "test_rank_ic_mean": safe_mean(test_ic),
            "test_rank_ic_std": safe_std(test_ic),
            "test_rank_ic_ir": safe_mean(test_ic) / safe_std(test_ic) if safe_std(test_ic) not in (None, 0.0) else None,
            "test_positive_ic_ratio": float((test_ic > 0).mean()) if len(test_ic) else None,
            "test_observations": int(len(test_ic)),
            "test_quantile_spread": {
                "observations": test_spread["observations"],
                "mean_top_minus_bottom": test_spread["mean_top_minus_bottom"],
                "median_top_minus_bottom": test_spread["median_top_minus_bottom"],
            },
        },
        "backtest": {
            "engine": "quant_system.portfolio_holding_backtest.run_continuous_portfolio_backtest",
            "execution_mode": "research_total_return_proxy",
            "signal_price": "hfq",
            "mark_price": "hfq",
            "trade_state_source": "raw_ohlcv_proxy",
            "quantile": float(args.quantile),
            "rebalance_sessions": int(args.rebalance_sessions),
            "config": backtest["report"]["config"],
            "metrics": metrics,
            "engine_report": backtest["report"],
            "active_positions_final": active_positions_final,
            "benchmark": benchmark,
            "annual_excess_return": excess_annual,
            "production_gate": production_gate,
        },
        "limitations": [
            "survivorship_bias: the 800-symbol universe consists of stocks that survived through the full history",
            "price_only: no point-in-time fundamentals, industry classification, ST state, or index membership",
            "corporate_actions: HFQ total-return proxy is used; an authoritative cash-dividend/split ledger is not available",
            "limit_rules: 9.5% RAW open proxy is used for 10% limit boards; board-specific and ST rules are not authoritative",
            "execution_mode: research total-return simulation, not production order-level replay",
            "2026 tail is present in the panel but excluded from both train and test metrics",
        ],
        "artifacts": {
            "equity": "equity.csv",
            "trades": "trades.csv",
            "signals": "oos_signals.parquet",
            "model": "model.json",
            "report": "report.json",
            "report_markdown": "report.md",
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "# 5年训练 / 3年测试正式回测",
        "",
        f"- 数据集: `{report['dataset']['path']}` ({report['dataset']['symbols']} 只, {report['dataset']['rows']} 行)",
        f"- 训练: {report['split']['train_start']} → {report['split']['train_end']} ({report['split']['train_sessions']} 个交易日)",
        f"- 测试: {report['split']['test_start']} → {report['split']['test_end']} ({report['split']['test_sessions']} 个交易日)",
        f"- 模型: Ridge alpha={report['model']['selected_alpha']}，训练期内嵌时间序列选参",
        f"- 测试期 Rank IC: {fmt(report['signal_quality']['test_rank_ic_mean'])}，正 IC 日占比 {fmt(report['signal_quality']['test_positive_ic_ratio'])}",
        f"- 测试期总收益: {fmt(report['backtest']['metrics']['total_return'])}",
        f"- 测试期年化收益: {fmt(report['backtest']['metrics']['annual_return'])}",
        f"- 测试期夏普: {fmt(report['backtest']['metrics']['sharpe'])}",
        f"- 测试期最大回撤: {fmt(report['backtest']['metrics']['max_drawdown'])}",
        f"- CSI300 同期年化: {fmt(report['backtest']['benchmark']['annual_return'])}",
        f"- 年化超额: {fmt(report['backtest']['annual_excess_return'])}",
        f"- 生产门禁: {report['backtest']['production_gate']['status']} ({', '.join(report['backtest']['production_gate']['blockers'])})",
        "",
        "## 解释边界",
        "",
        "该结果使用真实 800 只未复权行情和累计 HFQ 因子重建的数据，训练和测试按时间严格分离。它是价格型研究回测，不构成生产级 PIT 回测：当前缺少完整公司行动台账、ST 状态、行业/指数 PIT 成员关系和更精确的涨跌停规则。",
        "",
        "## 产物",
        "",
        f"- `{output / 'equity.csv'}`",
        f"- `{output / 'trades.csv'}`",
        f"- `{output / 'oos_signals.parquet'}`",
        f"- `{output / 'model.json'}`",
        f"- `{output / 'report.json'}`",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "dataset_symbols": report["dataset"]["symbols"],
        "dataset_rows": report["dataset"]["rows"],
        "train_sessions": report["split"]["train_sessions"],
        "test_sessions": report["split"]["test_sessions"],
        "selected_alpha": report["model"]["selected_alpha"],
        "test_rank_ic_mean": report["signal_quality"]["test_rank_ic_mean"],
        "test_total_return": report["backtest"]["metrics"]["total_return"],
        "test_annual_return": report["backtest"]["metrics"]["annual_return"],
        "test_sharpe": report["backtest"]["metrics"]["sharpe"],
        "test_max_drawdown": report["backtest"]["metrics"]["max_drawdown"],
        "benchmark_annual_return": report["backtest"]["benchmark"]["annual_return"],
        "annual_excess_return": report["backtest"]["annual_excess_return"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
