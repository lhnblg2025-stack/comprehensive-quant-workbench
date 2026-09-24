#!/usr/bin/env python3
"""Import paper order intents and manual/broker fills into the native trade database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_system import trade_db


def import_orders(path: str | Path, release_id: str) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"symbol", "direction", "shares", "suggested_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"orders missing columns: {sorted(missing)}")
    rows = []
    for row in frame.to_dict("records"):
        result = trade_db.create_paper_order(
            symbol=str(row["symbol"]).zfill(6), direction=row["direction"], shares=row["shares"],
            suggested_price=row["suggested_price"], release_id=release_id,
            order_type=row.get("order_type") or "limit", notes=row.get("notes") or "",
        )
        if result.get("error"):
            raise ValueError(result["error"])
        rows.append(result)
    return rows


def import_fills(path: str | Path) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"order_id", "filled_shares", "filled_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fills missing columns: {sorted(missing)}")
    rows = []
    for row in frame.to_dict("records"):
        result = trade_db.record_paper_fill(
            int(row["order_id"]), filled_shares=row["filled_shares"], filled_price=row["filled_price"],
            commission=row.get("commission") or 0, stamp_tax=row.get("stamp_tax") or 0,
            filled_at=row.get("filled_at") or None,
        )
        if result.get("error"):
            raise ValueError(result["error"])
        rows.append(result)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True); parser.add_argument("--orders"); parser.add_argument("--fills")
    args = parser.parse_args()
    if not args.orders and not args.fills:
        parser.error("at least --orders or --fills is required")
    imported_orders = import_orders(args.orders, args.release_id) if args.orders else []
    imported_fills = import_fills(args.fills) if args.fills else []
    print(json.dumps({"release_id": args.release_id, "orders_imported": len(imported_orders), "fills_imported": len(imported_fills), "execution": trade_db.paper_execution_stats(release_id=args.release_id)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
