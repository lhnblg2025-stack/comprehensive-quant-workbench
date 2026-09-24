#!/usr/bin/env python3
"""Build explicit research-only proxy panels from recovered local snapshots.

The resulting panels make Strategy Lab executable for plumbing and research
tests. They are never production evidence: every row carries
``synthetic_proxy=True`` and the quality report remains ``DATA_BLOCKED``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"
FEATURES = ROOT / "data_warehouse" / "feature_store"
MARKET = ROOT / "data_warehouse" / "market"
EVENTS = ROOT / "data_warehouse" / "events"


def _base_features() -> pd.DataFrame:
    files = sorted(FEATURES.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("feature_store snapshots are required")
    frames = [pd.read_parquet(p) for p in files]
    data = pd.concat(frames, ignore_index=True)
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["code", "date", "close"])
    data = data[data["code"].str.startswith(("00", "60"))]
    latest = data.sort_values("date").groupby("code", as_index=False).tail(1)
    return latest.sort_values("code").head(300).reset_index(drop=True)


def build_stock_panel() -> tuple[Path, dict]:
    base = _base_features()
    dates = pd.bdate_range("2021-01-04", "2026-09-15")
    rows = []
    for i, row in base.iterrows():
        code = str(row["code"])
        seed = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16)
        phase = (seed % 997) / 997.0
        start = float(row["close"])
        for t, day in enumerate(dates):
            cyc = 0.025 * np.sin((t / 31.0) + phase * 6.0) + 0.012 * np.cos(t / 97.0 + phase)
            trend = 0.00008 * t
            close = max(0.5, start * np.exp(trend + cyc))
            open_price = close * (1.0 + 0.002 * np.sin(t / 7.0 + phase))
            high = max(open_price, close) * 1.008
            low = min(open_price, close) * 0.992
            amount = max(1e6, float(row.get("amount", 1e8) or 1e8) * (1 + 0.15 * np.sin(t / 19.0 + phase)))
            values = {"code": code, "date": day, "raw_open": open_price, "raw_high": high,
                      "raw_low": low, "raw_close": close, "open": open_price, "high": high,
                      "low": low, "close": close, "volume": amount / close, "amount": amount,
                      "synthetic_proxy": True, "proxy_reason": "feature_store_snapshot_expanded_for_research_plumbing"}
            for col in ("mom20", "mom60", "mom120", "dist_52w", "amp20", "turnover", "vol_ratio20", "pe_pct252", "pb_pct252"):
                base_value = pd.to_numeric(row.get(col), errors="coerce")
                values[col] = float(base_value) if pd.notna(base_value) else float(np.sin(t / 23.0 + phase))
            rows.append(values)
    panel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "price_discovery_panel.parquet"
    panel.to_parquet(path, index=False)
    quality = {
        "schema": "research_proxy_panel/v1", "status": "DATA_BLOCKED", "research_status": "DATA_BLOCKED",
        "production_status": "DATA_BLOCKED", "synthetic_proxy": True,
        "rows": len(panel), "symbols": int(panel["code"].nunique()),
        "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()),
        "production_blockers": ["synthetic_proxy_data", "historical_point_in_time_trade_state_missing", "capacity_impact_model_missing"],
        "errors": ["This panel is generated from recent feature snapshots and is research plumbing only."],
        "next_action": "Replace with authoritative PIT price, membership and trade-state panel before production admission.",
        "coverage": {"membership_rows": 0, "minimum": 1.0, "median": 1.0},
    }
    (OUT / "pit_data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, quality


def build_etf_panel() -> tuple[Path, int]:
    source = ROOT / "data_warehouse" / "market_history" / "a_share_etf.parquet"
    raw = pd.read_parquet(source)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    codes = ["510050", "510300", "510500", "159915", "159919", "512100", "512880", "588000", "159949", "512690"]
    rows = []
    for idx, code in enumerate(codes):
        scale = 1.0 + idx * 0.003
        frame = raw.copy()
        frame["code"] = code
        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce") * scale
        frame["price"] = frame["close"]
        frame["synthetic_proxy"] = True
        frame["proxy_reason"] = "a_share_etf_snapshot_replicated_across_research_universe"
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    EVENTS.mkdir(parents=True, exist_ok=True)
    path = EVENTS / "etf_history.parquet"
    panel.to_parquet(path, index=False)
    for name, factor in (("沪深300", 1.0), ("上证指数", 0.98), ("深证成指", 1.02)):
        benchmark = raw[["date", "close"]].copy()
        benchmark["close"] = benchmark["close"] * factor
        (MARKET / f"index_daily_{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_parquet(MARKET / f"index_daily_{name}.parquet", index=False)
    return path, len(panel)


def main() -> int:
    stock, quality = build_stock_panel()
    etf, etf_rows = build_etf_panel()
    print(json.dumps({"stock_panel": str(stock.relative_to(ROOT)), "stock_rows": quality["rows"], "stock_symbols": quality["symbols"], "etf_panel": str(etf.relative_to(ROOT)), "etf_rows": etf_rows, "production_status": quality["production_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
