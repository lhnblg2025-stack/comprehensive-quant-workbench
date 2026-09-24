"""Run broad short/medium/long OHLCV factor discovery with fixed costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .factor_backtest_runner import Config, run
from .research_protocol import DEFAULT_PROTOCOL

FACTORS = (
    ("short_reversal_5", 1), ("short_reversal_20", 1), ("rsi_reversal_14", 1),
    ("gap_reversal", 1), ("range_compression", 1), ("volume_breakout", 1),
    ("volume_acceleration", 1), ("amount_acceleration", 1),
    ("mom_3_1", 1), ("mom_6_1", 1), ("trend_60", 1),
    ("ma_cross_20_60", 1), ("volatility_20", 1), ("low_vol_60", 1),
    ("amihud_inverse", 1), ("price_efficiency_20", 1), ("close_location_20", 1),
    ("mom_12_1", 1), ("mom120", 1), ("trend_120", 1),
    ("ma_cross_60_120", 1), ("volatility_120", 1), ("downside_vol_60", 1),
    ("breakout_252", 1), ("price_volume_trend", 1),
    ("obv_proxy_20", 1), ("money_flow_20", 1),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output); data_dir = output / "input"; data_dir.mkdir(parents=True, exist_ok=True)
    linked = data_dir / Path(args.panel).name
    if not linked.exists(): linked.symlink_to(Path(args.panel).resolve())
    result = run(Config(data_dir=str(data_dir), output_dir=str(output / "factor"), factors=tuple(name for name, _ in FACTORS), quantile=.20, cost_bps=15.0, min_stocks=DEFAULT_PROTOCOL.universe_size, oos_ratio=DEFAULT_PROTOCOL.test_ratio, factor_directions=FACTORS, rebalance="monthly"))
    report = {"schema": "broad_ohlcv_factor_matrix/v1", "status": "research_only", "factor_count": len(FACTORS), "horizons": {"short": list(FACTORS[:8]), "medium": list(FACTORS[8:18]), "long": list(FACTORS[18:])}, "blocked_for_promotion": ["historical_trade_state_unverified", "rolling_oos_multiple_testing_pending"], "results": result}
    (output / "broad_factor_matrix.json").write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "factor_count": len(FACTORS), "output": str(output)}, ensure_ascii=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
