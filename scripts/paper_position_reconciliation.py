#!/usr/bin/env python3
"""Reconcile paper ledger positions against a manual/broker CSV snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

CST = timezone(timedelta(hours=8))


def reconcile(actual: pd.DataFrame, expected: list[dict]) -> tuple[pd.DataFrame, dict]:
    required = {"symbol", "shares"}
    missing = required - set(actual.columns)
    if missing:
        raise ValueError(f"actual positions missing columns: {sorted(missing)}")
    broker = actual.copy(); broker["symbol"] = broker.symbol.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    broker["shares"] = pd.to_numeric(broker.shares, errors="coerce").fillna(0).astype(int)
    broker = broker.groupby("symbol", as_index=False).agg(actual_shares=("shares", "sum"))
    system = pd.DataFrame(expected or [], columns=["symbol", "shares"])
    if not system.empty:
        system["symbol"] = system.symbol.astype(str).str.zfill(6); system["shares"] = pd.to_numeric(system.shares, errors="coerce").fillna(0).astype(int)
        system = system.groupby("symbol", as_index=False).agg(system_shares=("shares", "sum"))
    else:
        system = pd.DataFrame(columns=["symbol", "system_shares"])
    out = system.merge(broker, on="symbol", how="outer").fillna(0)
    out[["system_shares", "actual_shares"]] = out[["system_shares", "actual_shares"]].astype(int)
    out["share_difference"] = out.actual_shares - out.system_shares
    out["status"] = out.share_difference.map(lambda x: "matched" if x == 0 else ("missing_in_system" if x > 0 else "missing_in_actual"))
    summary = {"positions": len(out), "matched": int((out.status == "matched").sum()), "mismatched": int((out.status != "matched").sum()), "system_shares": int(out.system_shares.sum()), "actual_shares": int(out.actual_shares.sum())}
    return out.sort_values(["status", "symbol"]), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--actual", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from quant_system.trade_db import get_positions
    result, summary = reconcile(pd.read_csv(args.actual), get_positions())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "position_reconciliation.csv", index=False)
    payload = {"schema": "paper-position-reconciliation/v1", "generated_at": datetime.now(CST).isoformat(timespec="seconds"), "summary": summary}
    (output / "position_reconciliation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False)); return 0 if summary["mismatched"] == 0 else 2


if __name__ == "__main__": raise SystemExit(main())
