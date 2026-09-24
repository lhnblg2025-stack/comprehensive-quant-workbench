"""Build an 800-stock price/volume discovery panel from dual-price source shards.

This is explicitly a discovery artifact. It does not claim PIT fundamentals,
corporate-action completeness, or historical tradeability validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    close = pd.to_numeric(data["hfq_close"], errors="coerce")
    high = pd.to_numeric(data["hfq_high"], errors="coerce")
    low = pd.to_numeric(data["hfq_low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    amount = pd.to_numeric(data["amount"], errors="coerce")
    returns = close.pct_change()
    data["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    data["mom_6_1"] = close.shift(21) / close.shift(126) - 1.0
    data["mom_3_1"] = close.shift(21) / close.shift(63) - 1.0
    data["trend_60"] = close / close.rolling(60, min_periods=40).mean() - 1.0
    data["trend_120"] = close / close.rolling(120, min_periods=80).mean() - 1.0
    data["trend_composite"] = data[["trend_60", "trend_120"]].mean(axis=1)
    data["short_reversal_5"] = -(close / close.shift(5) - 1.0)
    data["short_reversal_20"] = -(close / close.shift(20) - 1.0)
    data["low_vol_60"] = -returns.rolling(60, min_periods=40).std()
    data["amihud_inverse"] = -(returns.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    data["volume_ratio_20"] = volume / volume.rolling(20, min_periods=10).mean()
    data["turnover_proxy"] = amount / amount.rolling(20, min_periods=10).mean()
    data["breakout_252"] = close / high.rolling(252, min_periods=126).max()
    data["range_compression"] = -(high / low - 1.0).rolling(20, min_periods=10).mean()
    data["volume_breakout"] = data["breakout_252"] * data["volume_ratio_20"]
    data["price_volume_trend"] = returns.rolling(20, min_periods=10).sum() * data["volume_ratio_20"]
    # Independent OHLCV factors for short, medium, and long holding horizons.
    data["rsi_14"] = 100 - 100 / (1 + returns.clip(lower=0).rolling(14, min_periods=10).mean() / (-returns.clip(upper=0)).rolling(14, min_periods=10).mean())
    data["rsi_reversal_14"] = -data["rsi_14"]
    data["ma_cross_20_60"] = close.rolling(20, min_periods=15).mean() / close.rolling(60, min_periods=40).mean() - 1.0
    data["ma_cross_60_120"] = close.rolling(60, min_periods=40).mean() / close.rolling(120, min_periods=80).mean() - 1.0
    data["volatility_20"] = -returns.rolling(20, min_periods=15).std()
    data["volatility_120"] = -returns.rolling(120, min_periods=80).std()
    data["skew_60"] = returns.rolling(60, min_periods=40).skew()
    data["kurtosis_60"] = -returns.rolling(60, min_periods=40).kurt()
    data["gap_reversal"] = -(pd.to_numeric(data["hfq_open"], errors="coerce") / close.shift(1) - 1.0)
    data["close_location_20"] = (close - low.rolling(20, min_periods=10).min()) / (high.rolling(20, min_periods=10).max() - low.rolling(20, min_periods=10).min())
    data["price_efficiency_20"] = returns.rolling(20, min_periods=10).sum().abs() / returns.abs().rolling(20, min_periods=10).sum()
    data["obv_proxy_20"] = (np.sign(returns).fillna(0) * volume).rolling(20, min_periods=10).sum()
    data["money_flow_20"] = (close * volume).rolling(20, min_periods=10).sum() / amount.rolling(20, min_periods=10).sum()
    data["volume_acceleration"] = volume.rolling(5, min_periods=5).mean() / volume.rolling(60, min_periods=40).mean()
    data["amount_acceleration"] = amount.rolling(5, min_periods=5).mean() / amount.rolling(60, min_periods=40).mean()
    data["downside_vol_60"] = -returns.clip(upper=0).rolling(60, min_periods=40).std()
    data["adv_amount_20"] = amount.rolling(20, min_periods=10).mean()
    data["participation_limit"] = data["adv_amount_20"] * 0.10
    data["impact_bps_proxy"] = 100.0 * (amount / data["adv_amount_20"]).clip(lower=0, upper=1).pow(.5)
    # Signal aliases consumed by existing factor runner.
    data["mom20"] = close / close.shift(20) - 1.0
    data["mom60"] = close / close.shift(60) - 1.0
    data["mom120"] = close / close.shift(120) - 1.0
    data["dist_52w"] = close / high.rolling(252, min_periods=126).max() - 1.0
    data["amp20"] = (high / low - 1.0).rolling(20, min_periods=10).mean()
    data["vol_ratio20"] = data["volume_ratio_20"]
    data["turnover"] = data["turnover_proxy"]
    data["close"] = close
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    frames = []; missing_adjusted = []
    for path in sorted((root / "raw" / "prices").glob("*.parquet")):
        data = pd.read_parquet(path)
        if "hfq_close" not in data or not data["hfq_close"].notna().any():
            missing_adjusted.append(path.stem); continue
        data["date"] = pd.to_datetime(data["date"])
        data = data.dropna(subset=["hfq_open", "hfq_high", "hfq_low", "hfq_close"])
        frames.append(_features(data))
    if not frames:
        raise ValueError("no_adjusted_price_shards")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    symbols = int(panel.code.nunique())
    date_min = pd.Timestamp(panel.date.min())
    date_max = pd.Timestamp(panel.date.max())
    protocol_ok = symbols >= 800 and date_min <= pd.Timestamp("2018-01-01") and (date_max - date_min).days >= 365 * 8
    panel.to_parquet(output, index=False)
    coverage = panel.groupby(panel.date.dt.year).code.nunique().to_dict()
    report = {"schema": "price_discovery_panel/v2", "status": "research_only" if protocol_ok else "blocked_data_insufficient", "protocol": {"universe_size": 800, "years": 8, "train_years": 5, "test_years": 3, "train_test_ratio": "5:3"}, "symbols": symbols, "rows": int(len(panel)), "date_min": str(date_min.date()), "date_max": str(date_max.date()), "yearly_cross_section": {str(k): int(v) for k, v in coverage.items()}, "missing_adjusted_symbols": missing_adjusted, "signal_price": "hfq", "execution_price": "raw", "blocked_for_promotion": ["pit_financials_missing", "historical_trade_state_unverified", "corporate_actions_missing"] + ([] if protocol_ok else ["universe_or_date_coverage_below_800_8y"])}
    (output.parent / "price_discovery_panel_manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
