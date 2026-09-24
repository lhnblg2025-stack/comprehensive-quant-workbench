#!/usr/bin/env python3
"""Merge cloud PIT observations into explicit, provenance-preserving research tables."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"
CLOUD = BASE / "cloud_pit_tradability"
OUT = BASE / "pit_merged"


def _norm(frame: pd.DataFrame, code_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["code"] = out[code_col].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    return out


def merge_instrument(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    for filename, exchange, end_col in (("sh_delist.parquet", "SH", "暂停上市日期"), ("sz_delist.parquet", "SZ", "终止上市日期")):
        path = CLOUD / filename
        if not path.is_file():
            continue
        raw = pd.read_parquet(path)
        code_col = "公司代码" if "公司代码" in raw else "证券代码"
        date_col = "上市日期"
        item = _norm(raw, code_col)
        item["list_date"] = pd.to_datetime(item[date_col], errors="coerce")
        item["delist_date"] = pd.to_datetime(item[end_col], errors="coerce")
        item["exchange"] = exchange
        item["source"] = "tencent_cloud_exchange_delist_table"
        frames.append(item[["code", "list_date", "delist_date", "exchange", "source"]])
    delisted = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["code", "list_date", "delist_date", "exchange", "source"])
    prices = panel[["code", "date"]].copy()
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    observed = prices.groupby("code")["date"].agg(observed_start="min", observed_end="max").reset_index()
    out = observed.merge(delisted.drop_duplicates("code"), on="code", how="left")
    out["list_date"] = out["list_date"].fillna(out["observed_start"])
    out["membership_status"] = out["delist_date"].notna().map({True: "exchange_delist_known", False: "observed_interval_only"})
    out["source_as_of"] = out["source"].fillna("price_observation_interval_not_pit_membership")
    return out, {"rows": len(out), "exchange_delist_known": int(out["delist_date"].notna().sum()), "observed_interval_only": int((out["membership_status"] == "observed_interval_only").sum())}


def merge_trade_state(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base_state = BASE / "trade_state.parquet"
    state = pd.read_parquet(base_state) if base_state.is_file() else panel[["code", "date"]].copy()
    state["code"] = state["code"].astype(str).str.zfill(6)
    state["date"] = pd.to_datetime(state["date"], errors="coerce")
    state["state_source"] = state.get("source_as_of", "derived_calendar_board_rules_ohlcv")
    suspension_files = sorted(CLOUD.glob("suspend_*.parquet"))
    observations = []
    for path in suspension_files:
        frame = pd.read_parquet(path)
        if frame.empty or "代码" not in frame:
            continue
        item = pd.DataFrame({"code": frame["代码"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6), "date": pd.to_datetime(path.stem.replace("suspend_", ""), format="%Y%m%d"), "cloud_suspended": True, "suspension_reason": frame.get("停牌原因", pd.Series(index=frame.index, dtype=object)).astype(str)})
        observations.append(item)
    cloud = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(columns=["code", "date", "cloud_suspended", "suspension_reason"])
    out = state.merge(cloud.drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
    matched = out["cloud_suspended"].fillna(False).astype(bool)
    existing_suspended = out["suspended"].fillna(False).astype(bool) if "suspended" in out else pd.Series(False, index=out.index)
    out["suspended"] = matched | existing_suspended
    out["state_source"] = out["cloud_suspended"].notna().map({True: "tencent_cloud_akshare_stock_tfp_em_plus_derived", False: "derived_calendar_board_rules_ohlcv"})
    out["authority_status"] = out["cloud_suspended"].notna().map({True: "cloud_exchange_observation_partial", False: "unverified"})
    return out, {"rows": len(out), "cloud_suspension_matches": int(matched.sum()), "cloud_dates": int(len(suspension_files)), "authority": "partial_not_full_history"}


def main() -> None:
    panel_path = BASE / "price_discovery_panel.parquet"
    panel = pd.read_parquet(panel_path, columns=["code", "date"])
    instrument, instrument_summary = merge_instrument(panel)
    trade_state, state_summary = merge_trade_state(panel)
    OUT.mkdir(parents=True, exist_ok=True)
    instrument.to_parquet(OUT / "instrument_pit.parquet", index=False)
    trade_state.to_parquet(OUT / "trade_state_pit.parquet", index=False)
    manifest = {"schema": "pit_merge_cloud/v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "source": "tencent_cloud_plus_local_panel", "instrument": instrument_summary, "trade_state": state_summary, "status": "PARTIAL_RESEARCH_ONLY", "blockers": ["active_symbols_listing_dates_missing", "historical_suspension_coverage_partial", "historical_price_limits_missing", "historical_st_namechange_incomplete"], "files": ["instrument_pit.parquet", "trade_state_pit.parquet"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__": main()
