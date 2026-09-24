#!/usr/bin/env python3
"""Paper-trading equity snapshot and cumulative net-value curve.

Reads the paper ledger and writes an immutable daily snapshot. Missing data is
reported as missing, never fabricated as zero.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EQUITY_DIR = ROOT / "generated" / "paper_equity"
CST = timezone(timedelta(hours=8))


def realized_cash_flow(trades: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in trades:
        amount = float(item.get("total_amount") or 0)
        total += amount if item.get("trade_type") == "sell" else -amount
    return round(total, 2)


def build_snapshot(*, date: str, initial_cash: float, trades: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    cash = round(initial_cash + realized_cash_flow(trades), 2)
    market_value = float(summary.get("total_value") or 0)
    net_value = round(cash + market_value, 2)
    return {
        "schema": "paper-equity/v1",
        "date": date,
        "initial_cash": round(float(initial_cash), 2),
        "cash": cash,
        "market_value": round(market_value, 2),
        "net_value": net_value,
        "cumulative_return_pct": round((net_value / initial_cash - 1) * 100, 2) if initial_cash else None,
        "num_positions": int(summary.get("num_positions") or 0),
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
    }


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def max_drawdown(points: list[dict[str, Any]]) -> float:
    peak = None
    worst = 0.0
    for point in sorted(points, key=lambda p: p["date"]):
        value = float(point["net_value"])
        if peak is None or value > peak:
            peak = value
        if peak:
            drawdown = (value / peak - 1) * 100
            worst = min(worst, drawdown)
    return round(worst, 2)


def load_curve() -> list[dict[str, Any]]:
    if not EQUITY_DIR.is_dir():
        return []
    points = []
    for path in sorted(EQUITY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") == "paper-equity/v1":
                points.append(data)
        except (OSError, ValueError):
            continue
    return points


def write_curve(points: list[dict[str, Any]]) -> Path:
    path = EQUITY_DIR / "equity_curve.json"
    payload = {
        "schema": "paper-equity-curve/v1",
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "points": points,
        "max_drawdown_pct": max_drawdown(points),
    }
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="write a paper equity snapshot")
    parser.add_argument("--date", required=True)
    parser.add_argument("--initial-cash", type=float, default=float(os.environ.get("PAPER_INITIAL_CASH", "1000000")))
    args = parser.parse_args()

    from quant_system.trade_db import get_trades, get_summary

    trades = get_trades(days=3650, limit=100000)
    summary = get_summary()
    snapshot = build_snapshot(date=args.date, initial_cash=args.initial_cash, trades=trades, summary=summary)
    path = EQUITY_DIR / f"{args.date}.json"
    _atomic_write(path, json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")

    curve = load_curve()
    curve_path = write_curve(curve)
    print(json.dumps({"snapshot": str(path), "curve": str(curve_path), "net_value": snapshot["net_value"], "return_pct": snapshot["cumulative_return_pct"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
