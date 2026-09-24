#!/usr/bin/env python3
"""Compare a paper equity curve against fixed market benchmarks.

Metrics are computed from the local data warehouse; missing benchmarks are
reported as missing instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))

BENCHMARKS = {
    "hs300": ("沪深300", "data_warehouse/index/sh000300.parquet"),
    "zz500": ("中证500", "data_warehouse/index/sh000905.parquet"),
}


def _returns(points: list[dict[str, Any]]) -> list[float]:
    values = [float(p["net_value"]) for p in sorted(points, key=lambda p: p["date"])]
    if len(values) < 2:
        return []
    return [values[i] / values[i - 1] - 1 for i in range(1, len(values))]


def annual_return(returns: list[float], periods_per_year: int = 250) -> float | None:
    if not returns:
        return None
    total = math.prod(1 + r for r in returns)
    years = len(returns) / periods_per_year
    return (total ** (1 / years) - 1) * 100 if years > 0 else None


def max_drawdown_pct(returns: list[float]) -> float | None:
    if not returns:
        return None
    peak = 0.0
    value = 1.0
    worst = 0.0
    for r in returns:
        value *= (1 + r)
        peak = max(peak, value)
        worst = min(worst, (value / peak - 1) * 100)
    return round(worst, 2)


def sharpe(returns: list[float], periods_per_year: int = 250) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    return round((mean / std) * math.sqrt(periods_per_year), 2) if std > 0 else None


def calmar(returns: list[float], periods_per_year: int = 250) -> float | None:
    ann = annual_return(returns, periods_per_year)
    mdd = max_drawdown_pct(returns)
    if ann is None or mdd is None or mdd == 0:
        return None
    return round(ann / abs(mdd), 2)


def strategy_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    returns = _returns(points)
    return {
        "points": len(points),
        "annual_return_pct": round(annual_return(returns), 2) if returns else None,
        "max_drawdown_pct": max_drawdown_pct(returns),
        "sharpe": sharpe(returns),
        "calmar": calmar(returns),
    }


def compare(strategy_points: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "benchmark-compare/v1",
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "strategy": strategy_metrics(strategy_points),
        "benchmarks": {},
    }
    for key, (label, rel) in BENCHMARKS.items():
        path = ROOT / rel
        if not path.is_file():
            result["benchmarks"][key] = {"label": label, "status": "missing"}
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            close = frame.get("close")
            if close is None:
                result["benchmarks"][key] = {"label": label, "status": "missing"}
                continue
            values = [float(v) for v in close.tail(len(strategy_points))]
            returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values))] if len(values) > 1 else []
            result["benchmarks"][key] = {
                "label": label,
                "status": "available",
                "annual_return_pct": round(annual_return(returns), 2) if returns else None,
                "max_drawdown_pct": max_drawdown_pct(returns),
                "sharpe": sharpe(returns),
            }
        except Exception as exc:  # noqa: BLE001
            result["benchmarks"][key] = {"label": label, "status": "error", "error": str(exc)[:120]}
    strategy_ann = result["strategy"].get("annual_return_pct")
    for key in result["benchmarks"]:
        bench = result["benchmarks"][key]
        if bench.get("status") == "available" and strategy_ann is not None and bench.get("annual_return_pct") is not None:
            bench["excess_vs_strategy_pct"] = round(strategy_ann - bench["annual_return_pct"], 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="compare strategy curve to benchmarks")
    parser.add_argument("curve", type=Path)
    args = parser.parse_args()
    data = json.loads(args.curve.read_text(encoding="utf-8"))
    result = compare(data.get("points", []))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
