#!/usr/bin/env python3
"""Final consistency check for the 15x3 corrected strategy matrix.

Recomputes every cross-artifact invariant from the written files instead of
trusting the values stored inside report.json / audit.json:

* file hashes and row counts declared in audit.json,
* report.json results vs strategy_matrix_results.csv,
* report.md tables vs the CSV,
* audit check flags vs the raw artifacts,
* equity curve final value and trade fees vs the CSV row,
* panel header claims (symbols, rows, date range, HFQ factor gap),
* training-label leakage diagnostics,
* headline = top test Sharpe in the CSV,
* position counts above ``max_names`` are explained by blocked sell orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "generated/strategy_matrix_15x3_corrected_20260922"
DEFAULT_PANEL = ROOT / "generated/long_price_panel/price_panel.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_close(left: pd.Series, right: pd.Series, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    args = parser.parse_args()

    out = Path(args.output).resolve()
    panel_path = Path(args.panel).resolve()
    report = json.loads((out / "report.json").read_text(encoding="utf8"))
    audit = json.loads((out / "audit.json").read_text(encoding="utf8"))
    markdown = (out / "report.md").read_text(encoding="utf8")
    results = pd.read_csv(out / "strategy_matrix_results.csv")
    equity = pd.read_parquet(out / "equity_curves.parquet")
    trades = pd.read_parquet(out / "trades.parquet")
    panel = pd.read_parquet(panel_path, columns=["date", "code", "hfq_close", "raw_close", "adjust_factor"])

    checks: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, value: bool, detail: object = None) -> None:
        checks[name] = {"pass": bool(value), "detail": detail}
        if not value:
            failures.append(name)

    # --- declared hashes and row counts -----------------------------------
    declared = {
        "results_csv": out / "strategy_matrix_results.csv",
        "equity_curves": out / "equity_curves.parquet",
        "trades": out / "trades.parquet",
    }
    loaded_rows = {"results_csv": len(results), "equity_curves": len(equity), "trades": len(trades)}
    for key, path in declared.items():
        record = audit["outputs"][key]
        check(f"hash_{key}", record["sha256"] == sha256(path), record["sha256"])
        check(f"rows_{key}", record["rows"] == loaded_rows[key], {"declared": record["rows"], "actual": loaded_rows[key]})

    # --- report.json vs CSV ------------------------------------------------
    json_results = pd.DataFrame(report["results"])
    check("report_json_row_count_matches_csv", len(json_results) == len(results), len(results))
    check("report_json_columns_match_csv", set(json_results.columns) == set(results.columns))
    numeric_cols = [c for c in results.columns if c in json_results.columns and results[c].dtype.kind in "fib"]
    for column in numeric_cols:
        check(f"report_json_numeric_{column}", numeric_close(results[column], json_results[column]))
    string_cols = [c for c in results.columns if c not in numeric_cols]
    for column in string_cols:
        check(f"report_json_string_{column}", bool((results[column].astype(str) == json_results[column].astype(str)).all()))

    # --- grid coverage -----------------------------------------------------
    expected_rows = 15 * 3 * 2
    check("grid_rows_15x3x2", len(results) == expected_rows, len(results))
    check("grid_strategy_count", results["strategy"].nunique() == 15, results["strategy"].nunique())
    check("grid_frequency_count", set(results["frequency"]) == {"daily", "weekly", "monthly"}, sorted(set(results["frequency"])))
    check("grid_windows", set(results["window"]) == {"train_2018_2022", "test_2023_2025"}, sorted(set(results["window"])))
    per_freq = results[results["window"] == "test_2023_2025"].groupby("frequency")["strategy"].nunique().to_dict()
    check("grid_test_15_per_frequency", all(v == 15 for v in per_freq.values()), per_freq)

    # --- headline ----------------------------------------------------------
    best = results[results["window"] == "test_2023_2025"].sort_values(["sharpe", "annual_return"], ascending=False).iloc[0]
    headline = report["headline_test"]
    check("headline_strategy_matches_csv_best", headline["strategy"] == best["strategy"], [headline["strategy"], best["strategy"]])
    check("headline_frequency_matches_csv_best", headline["frequency"] == best["frequency"], [headline["frequency"], best["frequency"]])
    check("headline_total_return_matches_csv", abs(float(headline["total_return"]) - float(best["total_return"])) < 1e-9)
    check("headline_sharpe_matches_csv", abs(float(headline["sharpe"]) - float(best["sharpe"])) < 1e-9)
    check("audit_headline_matches_report", audit["headline_test"] == report["headline_test"])

    # --- audit flags recomputed from artifacts ----------------------------
    check("audit_status_block", audit["status"] == "BLOCK", audit["status_reason"])
    check("all_execution_complete", bool(results["execution_status"].eq("complete").all()))
    check("no_negative_cash_rows", bool((results["negative_cash_rows"] == 0).all()))
    check("no_non_lot_trades", bool((results["non_lot_trade_rows"] == 0).all()))
    check("trade_shares_multiple_of_100", bool((trades["shares"] % 100 == 0).all()))
    check("all_nav_reconciled", bool(results["net_value_reconciled"].all()))
    check("fees_nonnegative", bool((trades["fee"] >= 0).all()))
    check("buy_slippage_1_001", bool(np.allclose(trades.loc[trades.side == "buy", "traded_value"] / trades.loc[trades.side == "buy", "raw_notional"], 1.001, rtol=1e-9)))

    # --- equity / trade read-back vs CSV ----------------------------------
    eq_last = equity.groupby(["strategy", "frequency", "window"]).agg(final_value=("total_value", "last"), sessions=("date", "count")).reset_index()
    merged = results.merge(eq_last, on=["strategy", "frequency", "window"], how="left")
    check("equity_final_value_matches_csv_all_runs", numeric_close(merged["final_equity"], merged["final_value"]))
    check("equity_session_count_matches_csv_all_runs", bool((merged["observations"] == merged["sessions"]).all()))
    fee = trades.groupby(["strategy", "frequency", "window"]).agg(fee=("fee", "sum"), n=("shares", "size"), traded=("traded_value", "sum")).reset_index()
    merged_fee = results.merge(fee, on=["strategy", "frequency", "window"], how="left")
    check("trade_fees_match_csv_all_runs", numeric_close(merged_fee["total_fees"], merged_fee["fee"]))
    check("trade_counts_match_csv_all_runs", bool((merged_fee["trades"] == merged_fee["n"]).all()))
    check("traded_value_matches_csv_all_runs", numeric_close(merged_fee["total_traded_value"], merged_fee["traded"], rtol=1e-7))

    # --- window boundaries -------------------------------------------------
    test_dates = pd.to_datetime(equity.loc[equity["window"] == "test_2023_2025", "date"])
    train_dates = pd.to_datetime(equity.loc[equity["window"] == "train_2018_2022", "date"])
    check("test_window_excludes_2026", bool(test_dates.max() <= pd.Timestamp("2025-12-31")), str(test_dates.max().date()))
    check("test_window_starts_2023", bool(test_dates.min() >= pd.Timestamp("2023-01-01")), str(test_dates.min().date()))
    check("train_window_ends_2022", bool(train_dates.max() <= pd.Timestamp("2022-12-31")), str(train_dates.max().date()))

    # --- label leakage diagnostics ----------------------------------------
    calibration = report["calibration"]
    check("calibration_entries_45", len(calibration) == 45, len(calibration))
    violations = {k: v["label_leakage_violations"] for k, v in calibration.items() if v["label_leakage_violations"]}
    check("no_label_leakage_in_any_calibration", not violations, violations or "all zero")

    # --- blocked-sell explanation for positions above max_names ------------
    check("excess_positions_explained_all_runs", bool(results["excess_positions_explained_by_blocked_sells"].all()), results[["excess_position_days", "unexplained_stale_positions"]].sum().to_dict())
    excess = results[results["excess_position_days"] > 0]
    check("excess_runs_have_blocked_sell_orders", bool((excess["blocked_sell_orders"] > 0).all()), int(len(excess)))

    # --- independent cash-flow replay of the headline trade ledger ----------
    strategy_key = report["headline_test"]["strategy"]
    frequency_key = report["headline_test"]["frequency"]
    window_key = report["headline_test"]["window"]
    run_trades = trades[(trades.strategy == strategy_key) & (trades.frequency == frequency_key) & (trades.window == window_key)].copy()
    run_equity = equity[(equity.strategy == strategy_key) & (equity.frequency == frequency_key) & (equity.window == window_key)].sort_values("date")
    replay_dates = list(pd.to_datetime(run_equity["date"]))
    mark_codes = sorted(run_trades["code"].unique())
    mark_panel = panel[panel["code"].isin(mark_codes)]
    mark_panel = mark_panel.assign(date=pd.to_datetime(mark_panel["date"]))
    mark_close = (
        mark_panel.pivot(index="date", columns="code", values="hfq_close")
        .reindex(replay_dates)
        .ffill()
    )
    run_trades = run_trades.assign(date=pd.to_datetime(run_trades["date"]))
    cash = 1_000_000.0
    book: dict[str, dict[str, float]] = {}
    replay_values: list[float] = []
    for day in replay_dates:
        day_trades = run_trades[run_trades["date"] == day]
        for _, row in day_trades[day_trades["side"] == "sell"].iterrows():
            position = book.get(row["code"])
            if position is None or position["shares"] <= 0:
                continue
            fraction = float(row["shares"]) / position["shares"]
            cash += float(row["traded_value"]) * (1.0 - 0.001) - float(row["fee"])
            position["base_adj"] *= 1.0 - fraction
            position["shares"] -= float(row["shares"])
            if position["shares"] <= 0:
                book.pop(row["code"], None)
        for _, row in day_trades[day_trades["side"] == "buy"].iterrows():
            cash -= float(row["traded_value"]) + float(row["fee"])
            position = book.setdefault(row["code"], {"shares": 0.0, "base_adj": 0.0})
            position["shares"] += float(row["shares"])
            position["base_adj"] += (float(row["shares"]) * float(row["raw_price"])) / float(row["hfq_price"])
        position_value = 0.0
        for code, position in book.items():
            if code in mark_close.columns:
                price = mark_close.loc[day, code]
                if pd.notna(price):
                    position_value += position["base_adj"] * float(price)
        replay_values.append(cash + position_value)
    replay_series = pd.Series(replay_values, index=run_equity["date"])
    difference = (replay_series - run_equity.set_index("date")["total_value"]).abs()
    check(
        "trade_ledger_replay_reproduces_equity_curve",
        bool(difference.max() < 1.0),
        {"max_abs_diff": float(difference.max()), "final_replay": float(replay_series.iloc[-1]), "final_reported": float(run_equity["total_value"].iloc[-1])},
    )

    # --- panel header ------------------------------------------------------
    dataset = report["dataset"]
    panel_date = pd.to_datetime(panel["date"])
    gap = float((panel["hfq_close"] / panel["raw_close"] - panel["adjust_factor"]).abs().max())
    check("panel_symbols_800", dataset["symbols"] == 800 and panel["code"].nunique() == 800, {"report": dataset["symbols"], "panel": int(panel["code"].nunique())})
    check("panel_rows_1_684_041", dataset["rows"] == 1684041 and len(panel) == 1684041, {"report": dataset["rows"], "panel": len(panel)})
    check("panel_date_min_2018_01_02", dataset["date_min"] == "2018-01-02" and str(panel_date.min().date()) == "2018-01-02", str(panel_date.min().date()))
    check("panel_date_max_2026_09_18", dataset["date_max"] == "2026-09-18" and str(panel_date.max().date()) == "2026-09-18", str(panel_date.max().date()))
    check("panel_hfq_factor_gap_le_1e_4", gap <= 1e-4, gap)

    # --- report.md tables vs CSV ------------------------------------------
    names = {spec["name"]: spec["key"] for spec in report["strategies"]}
    pattern = re.compile(
        r"^\| (.+?) \| (daily|weekly|monthly) \| (-?[\d.]+)% \| (-?[\d.]+)% \| (-?[\d.]+) \| (-?[\d.]+)% \| (-?[\d.]+)% \| ([\d.]+)% \| ([\d.]+) \| (-?[\d.]+) \|$",
        re.MULTILINE,
    )
    rows = pattern.findall(markdown)
    check("markdown_test_rows_45", len(rows) == 45, len(rows))
    mismatches = []
    for name, frequency, total, annual, sharpe, drawdown, excess, fee, avg_pos, ic in rows:
        if name not in names:
            mismatches.append(f"unknown strategy name {name}")
            continue
        row = results[(results.strategy == names[name]) & (results.frequency == frequency) & (results.window == "test_2023_2025")].iloc[0]
        for label, md_value, csv_value, tolerance in (
            ("total_return", float(total) / 100, row.total_return, 5e-5),
            ("annual_return", float(annual) / 100, row.annual_return, 5e-5),
            ("sharpe", float(sharpe), row.sharpe, 5e-4),
            ("max_drawdown", float(drawdown) / 100, row.max_drawdown, 5e-5),
            ("annual_excess", float(excess) / 100, row.annual_excess_return, 5e-5),
            ("fee_drag", float(fee) / 100, row.total_fees / 1e6, 5e-5),
            ("average_positions", float(avg_pos), row.average_positions, 0.05),
            ("rank_ic", float(ic), row.rank_ic_mean, 5e-4),
        ):
            if abs(md_value - csv_value) > tolerance:
                mismatches.append(f"{name}/{frequency}/{label}: md={md_value} csv={csv_value}")
    check("markdown_test_table_matches_csv", not mismatches, mismatches[:10] or "all 45 rows match")
    check("markdown_states_block_gate", "BLOCK" in markdown)
    check("markdown_retires_old_47_69", "47.69" in markdown)

    payload = {
        "schema": "strategy-matrix-15x3-consistency-check/v1",
        "output": str(out.relative_to(ROOT)),
        "audit_status": audit["status"],
        "status": "PASS" if not failures else "FAIL",
        "checks_passed": sum(1 for c in checks.values() if c["pass"]),
        "checks_total": len(checks),
        "failures": failures,
        "checks": checks,
        "headline_test": report["headline_test"],
        "panel": {
            "symbols": int(panel["code"].nunique()),
            "rows": int(len(panel)),
            "date_min": str(panel_date.min().date()),
            "date_max": str(panel_date.max().date()),
            "hfq_factor_gap": gap,
        },
    }
    (out / "consistency_check.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf8")

    lines = [
        "# 15×3 策略矩阵一致性核对",
        "",
        f"- 状态：{payload['status']}（{payload['checks_passed']}/{payload['checks_total']} 项通过）",
        f"- 生产门禁：{audit['status']} — {audit['status_reason']}",
        f"- 数据面板：{payload['panel']['symbols']} 只、{payload['panel']['rows']:,} 行、{payload['panel']['date_min']} 至 {payload['panel']['date_max']}；HFQ 因子一致性误差 {gap:.2e}",
        f"- 结果矩阵：{len(results)} 行（15 策略 × 日/周/月 × 训练/测试），测试期 {int((results.window=='test_2023_2025').sum())} 行",
        f"- 头部组合：{headline['strategy_name']} / {headline['frequency']}，总收益 {headline['total_return']*100:.2f}%，年化 {headline['annual_return']*100:.2f}%，Sharpe {headline['sharpe']:.3f}，最大回撤 {headline['max_drawdown']*100:.2f}%",
        "",
    ]
    if failures:
        lines += ["## 未通过项", ""] + [f"- {name}" for name in failures] + [""]
    else:
        lines += ["## 结论", "", "报告、审计摘要、结果 CSV、净值曲线与交易账逐项一致；训练标签无泄漏；超出 50 只的持仓全部来自被阻塞的卖出。", ""]
    (out / "consistency_check.md").write_text("\n".join(lines), encoding="utf8")

    print(json.dumps({"status": payload["status"], "passed": payload["checks_passed"], "total": payload["checks_total"], "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
