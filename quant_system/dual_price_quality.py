"""Quality checks for HFQ signal and RAW execution price panels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_dual_price_panel(panel: pd.DataFrame, min_symbols: int = 1, min_alignment: float = 0.99) -> dict:
    required = {"date", "code", "hfq_open", "hfq_high", "hfq_low", "hfq_close", "raw_open", "raw_high", "raw_low", "raw_close", "volume"}
    missing = sorted(required - set(panel.columns))
    checks = {"rows": int(len(panel)), "symbols": int(panel["code"].nunique()) if "code" in panel else 0, "missing_columns": missing}
    errors = []
    if missing:
        errors.append("missing_columns")
        return {"status": "BLOCK", "checks": checks, "errors": errors}
    prices = panel[["hfq_open", "hfq_high", "hfq_low", "hfq_close", "raw_open", "raw_high", "raw_low", "raw_close"]].apply(pd.to_numeric, errors="coerce")
    checks["invalid_prices"] = int((~np.isfinite(prices) | (prices <= 0)).sum().sum())
    checks["raw_ohlc_violations"] = int(((panel.raw_high < panel[["raw_open", "raw_close"]].max(axis=1)) | (panel.raw_low > panel[["raw_open", "raw_close"]].min(axis=1))).sum())
    ratio = pd.to_numeric(panel.hfq_close, errors="coerce") / pd.to_numeric(panel.raw_close, errors="coerce")
    checks["invalid_adjustment_ratio"] = int((~np.isfinite(ratio) | (ratio <= 0)).sum())
    checks["ratio_min"] = float(ratio.min()); checks["ratio_max"] = float(ratio.max())
    if checks["symbols"] < min_symbols: errors.append("insufficient_symbols")
    if checks["invalid_prices"]: errors.append("invalid_prices")
    if checks["raw_ohlc_violations"]: errors.append("raw_ohlc_violations")
    if checks["invalid_adjustment_ratio"] or ratio.notna().mean() < min_alignment: errors.append("adjustment_alignment")
    return {"status": "BLOCK" if errors else "PASS", "checks": checks, "errors": errors}
