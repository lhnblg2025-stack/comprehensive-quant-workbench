"""Run annual-PIT frozen candidates as a non-promotable coverage diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .factor_backtest_runner import Config, run


FROZEN = (
    ("value_book_to_price_industry_neutral", 1),
    ("quality_low_leverage_industry_neutral", 1),
    ("value_quality_composite_industry_neutral", 1),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-stocks", type=int, default=300)
    args = parser.parse_args(argv)
    panel = Path(args.panel); out = Path(args.output); inputs = out / "input"; inputs.mkdir(parents=True, exist_ok=True)
    linked = inputs / panel.name
    if not linked.exists():
        linked.symlink_to(panel.resolve())
    result = run(Config(data_dir=str(inputs), output_dir=str(out / "factor"), factors=tuple(name for name, _ in FROZEN), quantile=.20, cost_bps=15.0, min_stocks=args.min_stocks, oos_ratio=.30, factor_directions=FROZEN, rebalance="monthly"))
    report = {"schema": "annual_pit_frozen_diagnostic/v1", "status": "research_only", "min_stocks": args.min_stocks, "frozen_candidates": ["value_core", "quality_core", "balanced_core"], "blocked_for_promotion": ["500_stock_industry_coverage_not_met", "annual_only_pit", "historical_trade_state_unverified", "corporate_actions_missing", "rolling_oos_and_multiple_testing_pending"], "results": result}
    (out / "annual_pit_frozen_diagnostic.json").write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "factors": len(FROZEN), "output": str(out)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
