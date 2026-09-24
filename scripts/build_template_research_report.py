#!/usr/bin/env python3
"""Build a strict, auditable summary from completed template backtests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generated/strategy_lab/template_full_execution.json"
OUT_JSON = ROOT / "generated/strategy_lab/template_research_report.json"
OUT_CSV = ROOT / "generated/strategy_lab/template_research_summary.csv"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    templates = payload.get("templates", {})
    rows = []
    annual = {}
    required = {"daily_equity.csv", "order_events.csv", "blocked_events.csv", "annual_performance.csv"}
    for name, record in sorted(templates.items()):
        detail = ROOT / record["detail_dir"] if record.get("detail_dir") else ROOT / "generated/strategy_lab/template_full_execution_details" / name
        files = {path.name for path in detail.glob("*.csv")}
        annual_path = detail / "annual_performance.csv"
        annual_rows = pd.read_csv(annual_path).to_dict("records") if annual_path.is_file() else []
        annual[name] = annual_rows
        metrics = record.get("metrics", {})
        benchmark_total = None
        benchmark_annual = None
        excess_total = None
        excess_annual = None
        if annual_rows:
            benchmark_total = float(pd.Series([r.get("benchmark_total_return", 0.0) for r in annual_rows]).add(1.0).prod() - 1.0)
            years = float(record.get("run_scope", {}).get("sessions", 0)) / 252.0
            benchmark_annual = float((1.0 + benchmark_total) ** (1.0 / years) - 1.0) if years > 0 and benchmark_total > -1 else None
            strategy_total = metrics.get("total_return")
            strategy_annual = metrics.get("annual_return")
            if strategy_total is not None and benchmark_total > -1:
                excess_total = float((1.0 + float(strategy_total)) / (1.0 + benchmark_total) - 1.0)
            if strategy_annual is not None and benchmark_annual is not None:
                excess_annual = float((1.0 + float(strategy_annual)) / (1.0 + benchmark_annual) - 1.0)
        rows.append({
            "template": name,
            "status": record.get("status"),
            "total_return": metrics.get("total_return"),
            "annual_return": metrics.get("annual_return"),
            "benchmark_total_return": benchmark_total,
            "benchmark_annual_return": benchmark_annual,
            "excess_total_return": excess_total,
            "excess_annual_return": excess_annual,
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "orders": record.get("orders", metrics.get("orders", 0)),
            "filled_orders": record.get("filled_orders", metrics.get("trades", 0)),
            "blocked_orders": record.get("blocked_orders", metrics.get("blocked_orders", 0)),
            "rejected_orders": record.get("rejected_orders", metrics.get("rejected_orders", 0)),
            "blocked_reasons": json.dumps(record.get("order_reason_counts", {}), ensure_ascii=False, sort_keys=True),
            "strategy_parameters": json.dumps(record.get("strategy_parameters", {}), ensure_ascii=True, sort_keys=True),
            "detail_complete": required.issubset(files),
            "annual_rows": len(annual_rows),
        })
    frame = pd.DataFrame(rows)
    complete = len(rows) == 15 and bool(len(frame)) and bool((frame["status"].isin(["executed", "failed", "blocked"])).all()) and bool(frame["detail_complete"].all())
    report = {
        "schema": "strategy_lab_template_research_report/v1",
        "status": "complete_research_proxy" if complete else "incomplete_no_final_report",
        "source_artifact": str(SOURCE.relative_to(ROOT)),
        "summary": payload.get("summary", {}),
        "research_proxy": {
            "templates": rows,
            "annual_performance": annual,
        },
        "production_pit": {
            "status": payload.get("data_gate", {}).get("production_status", "DATA_BLOCKED"),
            "blockers": payload.get("data_gate", {}).get("production_blockers", payload.get("data_gate", {}).get("blockers", [])),
        },
        "validation": {
            "template_count": len(rows),
            "all_terminal": complete,
            "all_detail_files_present": bool(len(frame)) and bool(frame["detail_complete"].all()),
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(json.dumps({"json": str(OUT_JSON), "csv": str(OUT_CSV), "status": report["status"], "templates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
