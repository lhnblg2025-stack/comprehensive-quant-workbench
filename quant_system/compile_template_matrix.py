"""Compile comparable frontend template execution and data-gate evidence."""
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd


def compile_matrix(execution_path: str | Path, output: str | Path) -> dict:
    source = Path(execution_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    templates = payload.get("templates", {})
    rows = []
    annual_frames = []
    reasons = Counter()
    for name, record in sorted(templates.items()):
        metrics = record.get("metrics", {})
        reason_counts = {str(k): int(v) for k, v in record.get("order_reason_counts", {}).items()}
        reasons.update(reason_counts)
        annual_path = record.get("annual_path")
        if annual_path:
            path = Path(annual_path)
            if not path.is_absolute():
                path = source.parents[2] / path
            if path.is_file():
                annual = pd.read_csv(path)
                annual.insert(0, "strategy", name)
                annual_frames.append(annual)
        rows.append({
            "strategy": name,
            "status": record.get("status"),
            "classification": record.get("classification", "DATA_BLOCKED"),
            "annual_return": metrics.get("annual_return"),
            "total_return": metrics.get("total_return"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "orders": record.get("orders", 0),
            "filled_orders": record.get("filled_orders", metrics.get("trades", 0)),
            "blocked_orders": record.get("blocked_orders", metrics.get("blocked_orders", 0)),
            "rejected_orders": record.get("rejected_orders", metrics.get("rejected_orders", 0)),
            "order_reason_counts": reason_counts,
            "data_gate_status": payload.get("data_gate", {}).get("classification", "unknown"),
            "data_gate_blockers": payload.get("data_gate", {}).get("blockers", []),
        })
    annual = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(destination.with_name(destination.stem + "_annual.csv"), index=False)
    report = {
        "schema": "frontend_template_matrix/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source),
        "dataset": payload.get("scope", {}),
        "status": "DATA_BLOCKED" if payload.get("data_gate", {}).get("blockers") else "RESEARCH_READY",
        "template_count": len(rows),
        "executed_count": sum(row["status"] == "executed" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "blocked_order_total": int(sum(reasons.values())),
        "blocked_order_reasons": dict(reasons.most_common()),
        "templates": sorted(rows, key=lambda row: row["annual_return"] if row["annual_return"] is not None else float("-inf"), reverse=True),
        "annual_detail": str(destination.with_name(destination.stem + "_annual.csv")),
        "limitations": ["frontend_templates_are_executed_research_only", "authoritative_historical_trade_state_and_full_PIT_gates_required_for_admission"],
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# 前端策略统一回测矩阵", "", f"- 模板: {len(rows)}；执行成功: {report['executed_count']}；失败: {report['failed_count']}", f"- 样本: {payload.get('scope', {}).get('symbols')} 标的，{payload.get('scope', {}).get('start')} 至 {payload.get('scope', {}).get('end')}", f"- 数据状态: **{report['status']}**", "", "| 策略 | 年化 | 总收益 | Sharpe | 最大回撤 | 订单 | 成交 | 阻断 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["templates"]:
        fmt=lambda x:"NA" if x is None else f"{float(x):.2%}"
        lines.append(f"| {row['strategy']} | {fmt(row['annual_return'])} | {fmt(row['total_return'])} | {row['sharpe'] if row['sharpe'] is not None else 'NA'} | {fmt(row['max_drawdown'])} | {row['orders']} | {row['filled_orders']} | {row['blocked_orders']} |")
    lines += ["", "## 阻断原因", "", "| 原因 | 数量 |", "|---|---:|"] + [f"| {k} | {v} |" for k,v in reasons.most_common()] + ["", "逐年明细见配套 `_annual.csv`；所有模板因 PIT 交易状态门禁未通过，只能作为研究级执行证据。"]
    destination.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--execution", required=True); parser.add_argument("--output", required=True)
    args=parser.parse_args(); report=compile_matrix(args.execution,args.output); print(json.dumps({"output":args.output,"templates":report["template_count"],"blocked":report["blocked_order_total"]},ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
