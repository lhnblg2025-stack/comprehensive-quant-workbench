#!/usr/bin/env python3
"""Export release-bound production target weights from the canonical StrategyEngine."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))


def run(day: str) -> dict:
    from quant_system.data_release import load_release
    from quant_system.strategy_engine import StrategyEngine, FactorStrategy

    release = load_release(ROOT, expected_day=day)
    strategy_report = ROOT / "generated" / "factor_backtest_mainboard_latest.json"
    validation = json.loads(strategy_report.read_text(encoding="utf-8")).get("validation", {}) if strategy_report.exists() else {}
    out = ROOT / "generated" / f"factor_strategy_targets_{day}.parquet"
    if validation.get("deployable") is not True:
        out.unlink(missing_ok=True)
        payload = {"schema": "production-targets/v1", "as_of": day, "release_id": release["release_id"],
                   "status": "WARN", "targets": 0, "path": str(out),
                   "reason": "strategy_not_deployable", "promotion_reasons": validation.get("promotion_reasons", [])}
        (ROOT / "generated" / f"factor_strategy_targets_{day}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    engine = StrategyEngine()
    # Production targets use only the OOS-promoted factor sleeve. Other
    # experimental/network strategies remain analysis evidence, not orders.
    engine.strategies = [FactorStrategy(weight=1.0)]
    result = engine.run(as_of=day, release_id=release["release_id"])
    rows = [{"date": day, "code": item.stock, "target_weight": item.weight,
             "confidence": item.confidence, "release_id": release["release_id"],
             "strategy": "FactorStrategy"} for item in result.get("allocations", [])]
    frame = pd.DataFrame(rows, columns=["date", "code", "target_weight", "confidence", "release_id", "strategy"])
    frame.to_parquet(out, index=False)
    status = "PASS" if not frame.empty else "WARN"
    payload = {"schema": "production-targets/v1", "as_of": day, "release_id": release["release_id"],
               "status": status, "targets": len(frame), "path": str(out),
               "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
               "strategy_meta": result.get("meta", {})}
    (ROOT / "generated" / f"factor_strategy_targets_{day}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--date", required=False)
    args = parser.parse_args()
    if args.date:
        day = args.date
    else:
        from build_data_release import _latest_completed_day
        day = _latest_completed_day().isoformat()
    print(json.dumps(run(day), ensure_ascii=False))
