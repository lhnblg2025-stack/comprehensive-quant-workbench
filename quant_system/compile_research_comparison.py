"""Compile the frozen-matrix, execution-block, and PIT readiness evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def compile_report(matrix_dir: str | Path, pit_audit: str | Path, source_catalog: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(matrix_dir)
    matrix = json.loads((root / "strategy_matrix.json").read_text(encoding="utf-8"))
    audit = json.loads(Path(pit_audit).read_text(encoding="utf-8"))
    catalog = json.loads(Path(source_catalog).read_text(encoding="utf-8"))
    blocks: Counter[str] = Counter()
    strategy_blocks: dict[str, dict[str, int]] = {}
    for case in matrix["cases"]:
        reasons = {str(key): int(value) for key, value in case.get("blocked_event_reasons", {}).items()}
        blocks.update(reasons)
        strategy_blocks[case["name"]] = reasons
    cases = sorted(matrix["cases"], key=lambda row: row.get("annual_return") if row.get("annual_return") is not None else float("-inf"), reverse=True)
    annual = pd.read_csv(root / "strategy_matrix_annual_performance.csv")
    annual_summary = (
        annual.groupby("strategy", as_index=False)
        .agg(years=("year", "nunique"), positive_excess_years=("annual_excess_return", lambda x: int((x > 0).sum())), mean_annual_excess=("annual_excess_return", "mean"))
        .sort_values("mean_annual_excess", ascending=False)
        .to_dict("records")
    )
    report: dict[str, Any] = {
        "schema": "research_comparison/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classification": "research_only",
        "frozen_matrix": {
            "bundle": matrix["bundle"],
            "bundle_manifest_sha256": matrix["bundle_manifest_sha256"],
            "signal_price": matrix["signal_price"],
            "execution_price": matrix["execution_price"],
            "costs": matrix["costs"],
            "benchmark": matrix["common_benchmark"],
            "strategy_count": len(cases),
            "strategies": [{key: value for key, value in case.items() if key != "annual_performance"} for case in cases],
            "annual_summary": annual_summary,
            "annual_performance_path": str(root / "strategy_matrix_annual_performance.csv"),
        },
        "blocked_events": {
            "total": int(sum(blocks.values())),
            "by_reason": dict(blocks.most_common()),
            "by_strategy": strategy_blocks,
            "event_file_pattern": str(root / "*/blocked_events.csv"),
        },
        "pit_local_audit": audit,
        "open_source_readiness": {
            "status": catalog["status"],
            "research_ready_domains": catalog["research_ready_domains"],
            "admission_blocking_domains": catalog["admission_blocking_domains"],
            "catalog_path": str(source_catalog),
        },
        "promotion_status": "DATA_BLOCKED",
        "promotion_reason": "Frozen 100-symbol HFQ/RAW result is not a 500-1000 symbol actual-PIT execution study; local actual-PIT panel is incomplete and source catalog retains lifecycle, ST, price-limit, and quarterly-release gaps.",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 策略全量回测与 PIT 审计可比报告",
        "",
        "- 分类: **research_only**；晋级状态: **DATA_BLOCKED**",
        f"- 冻结样本: `{matrix['bundle']}`",
        f"- 执行口径: {matrix['signal_price'].upper()} 信号 / {matrix['execution_price'].upper()} 次日开盘，显式佣金、印花税、过户费和 10bp 滑点。",
        f"- 基准: CSI300，区间年化 {matrix['common_benchmark']['annual_return']:.2%}。",
        "",
        "## 全量策略比较",
        "",
        "| 策略 | 年化收益 | 年化超额 | 夏普 | 最大回撤 | 交易 | 阻断 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        fmt = lambda value: "NA" if value is None else f"{value:.2%}"
        lines.append(f"| {case['name']} | {fmt(case['annual_return'])} | {fmt(case['excess_annual'])} | {case['sharpe']:.3f} | {case['max_drawdown']:.2%} | {case['trade_rows']} | {case['blocked_event_rows']} |")
    lines += [
        "",
        "## 阻断事件归因",
        "",
        "| 原因 | 事件数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {reason} | {count} |" for reason, count in blocks.most_common())
    lines += [
        "",
        "## PIT 数据与开源接入",
        "",
        f"- 本机实际 PIT 面板审计: **{audit['status']}**。",
        f"- 关键缺口: {', '.join(audit['errors'])}。",
        f"- 开源接入状态: **{catalog['status']}**；可用于研究的域: {', '.join(catalog['research_ready_domains'])}。",
        f"- 准入阻断域: {', '.join(catalog['admission_blocking_domains'])}。",
        "",
        "年度策略、CSI300 基准与年化超额的逐年明细见 `strategy_matrix_annual_performance.csv`；每条策略逐笔未成交/阻断事件见其目录内的 `blocked_events.csv`。",
    ]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--pit-audit", required=True)
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compile_report(args.matrix_dir, args.pit_audit, args.source_catalog, args.output)
    print(json.dumps({"output": args.output, "strategies": report["frozen_matrix"]["strategy_count"], "blocks": report["blocked_events"]["total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
