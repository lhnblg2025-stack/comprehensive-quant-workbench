#!/usr/bin/env python3
"""Reconcile paper orders with broker/manual fills and measure execution drift."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_ORDERS = {"order_id", "symbol", "side", "requested_shares", "suggested_price"}
REQUIRED_FILLS = {"order_id", "filled_shares", "filled_price"}


def reconcile(orders: pd.DataFrame, fills: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_orders = REQUIRED_ORDERS - set(orders.columns)
    missing_fills = REQUIRED_FILLS - set(fills.columns)
    if missing_orders or missing_fills:
        raise ValueError(f"missing columns orders={sorted(missing_orders)} fills={sorted(missing_fills)}")
    o = orders.copy(); f = fills.copy()
    o["order_id"] = o["order_id"].astype(str); f["order_id"] = f["order_id"].astype(str)
    for column in ("requested_shares", "suggested_price"):
        o[column] = pd.to_numeric(o[column], errors="coerce")
    for column in ("filled_shares", "filled_price"):
        f[column] = pd.to_numeric(f[column], errors="coerce")
    f = f.dropna(subset=["order_id", "filled_shares", "filled_price"])
    f = f[(f.filled_shares > 0) & (f.filled_price > 0)]
    grouped = f.assign(notional=f.filled_shares * f.filled_price).groupby("order_id", as_index=False).agg(
        filled_shares=("filled_shares", "sum"), filled_notional=("notional", "sum"), fill_events=("filled_shares", "size"))
    grouped["weighted_filled_price"] = grouped.filled_notional / grouped.filled_shares
    result = o.merge(grouped, on="order_id", how="left")
    result[["filled_shares", "filled_notional", "fill_events"]] = result[["filled_shares", "filled_notional", "fill_events"]].fillna(0)
    result["weighted_filled_price"] = result["weighted_filled_price"].where(result.filled_shares > 0)
    result["fill_rate"] = (result.filled_shares / result.requested_shares.replace(0, pd.NA)).fillna(0).clip(0, 1)
    result["unfilled_shares"] = (result.requested_shares - result.filled_shares).clip(lower=0)
    # Buy slippage: paid above suggestion is adverse. Sell: received below is adverse.
    direction = result.side.astype(str).str.lower().map({"buy": 1, "sell": -1}).fillna(1)
    result["slippage_bps"] = ((result.weighted_filled_price - result.suggested_price) / result.suggested_price * direction * 10000).where(result.filled_shares > 0)
    result["fill_status"] = result.apply(lambda row: "unfilled" if row.filled_shares <= 0 else ("partial" if row.fill_rate < 1 else "filled"), axis=1)
    numeric = result["slippage_bps"].dropna()
    summary = {
        "orders": int(len(result)), "filled_orders": int((result.fill_status == "filled").sum()),
        "partial_orders": int((result.fill_status == "partial").sum()), "unfilled_orders": int((result.fill_status == "unfilled").sum()),
        "requested_shares": float(result.requested_shares.sum()), "filled_shares": float(result.filled_shares.sum()),
        "fill_rate": float(result.filled_shares.sum() / result.requested_shares.sum()) if result.requested_shares.sum() else 0.0,
        "mean_adverse_slippage_bps": float(numeric.mean()) if not numeric.empty else None,
        "median_adverse_slippage_bps": float(numeric.median()) if not numeric.empty else None,
        "p95_adverse_slippage_bps": float(numeric.quantile(.95)) if not numeric.empty else None,
    }
    return result, summary


def run(orders_path: str | Path, fills_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    orders_file, fills_file, output = Path(orders_path), Path(fills_path), Path(output_dir)
    orders = pd.read_csv(orders_file); fills = pd.read_csv(fills_file)
    reconciled, summary = reconcile(orders, fills)
    output.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(output / "reconciled_orders.csv", index=False)
    payload = {"schema": "paper-execution-reconciliation/v1", "orders_sha256": hashlib.sha256(orders_file.read_bytes()).hexdigest(), "fills_sha256": hashlib.sha256(fills_file.read_bytes()).hexdigest(), "summary": summary}
    (output / "execution_reconciliation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", required=True); parser.add_argument("--fills", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(json.dumps(run(args.orders, args.fills, args.output), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
