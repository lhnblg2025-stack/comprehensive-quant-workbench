"""Run the fixed 20-strategy price/volume discovery matrix.

Output is deliberately research-only: its inputs do not yet meet PIT,
corporate-action, and historical tradeability requirements for promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .factor_backtest_runner import Config, run


CASES: tuple[tuple[str, int], ...] = (
    ("mom_12_1", 1), ("mom_6_1", 1), ("mom_3_1", 1),
    ("trend_60", 1), ("trend_120", 1), ("trend_composite", 1),
    ("short_reversal_5", 1), ("short_reversal_20", 1),
    ("low_vol_60", 1), ("amihud_inverse", 1),
    ("volume_ratio_20", 1), ("turnover_proxy", 1),
    ("breakout_252", 1), ("range_compression", 1),
    ("volume_breakout", 1), ("price_volume_trend", 1),
    ("mom20", 1), ("mom60", 1), ("mom120", 1), ("dist_52w", 1),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    panel = Path(args.panel); output = Path(args.output); data_dir = output / "input"; data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "price_discovery_panel.parquet"
    if target.resolve() != panel.resolve():
        target.unlink(missing_ok=True)
        target.symlink_to(panel.resolve())
    result = run(Config(
        data_dir=str(data_dir), output_dir=str(output / "factor"), factors=tuple(name for name, _ in CASES),
        quantile=.20, cost_bps=15.0, min_stocks=300, oos_ratio=.30,
        annualization=252.0, factor_directions=CASES, rebalance="monthly",
    ))
    summary = {
        "schema": "price_discovery_20_strategy_matrix/v1", "status": "research_only",
        "strategy_count": len(CASES), "signal_price": "tencent_qfq", "execution_label": "tushare_raw_next_open_to_following_open",
        "blocked_for_promotion": ["pit_financials_missing", "historical_trade_state_unverified", "corporate_actions_missing", "adjusted_price_provenance_review"],
        "results": result,
    }
    (output / "price_discovery_matrix.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "strategies": len(CASES), "output": str(output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
