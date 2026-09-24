"""Build a frozen, resumable dataset bundle for factor and execution backtests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .historical_panel_builder import compute_price_factors
from .sources_all import fetch_daily_unified


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_corporate_actions(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "date", "action_type", "cash_per_share", "ratio", "source", "status"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    result = []
    for row in data.to_dict("records"):
        status = str(row.get("方案进度", ""))
        ex_date = pd.to_datetime(row.get("除权除息日"), errors="coerce")
        if pd.isna(ex_date) or "实施" not in status:
            continue
        cash_per_10 = pd.to_numeric(row.get("现金分红-现金分红比例"), errors="coerce")
        stock_per_10 = pd.to_numeric(row.get("送转股份-送转总比例"), errors="coerce")
        if pd.notna(cash_per_10) and float(cash_per_10) > 0:
            result.append({"symbol": symbol, "date": ex_date, "action_type": "dividend", "cash_per_share": float(cash_per_10) / 10.0, "ratio": 1.0, "source": "stock_fhps_detail_em", "status": status})
        if pd.notna(stock_per_10) and float(stock_per_10) > 0:
            result.append({"symbol": symbol, "date": ex_date, "action_type": "split", "cash_per_share": 0.0, "ratio": 1.0 + float(stock_per_10) / 10.0, "source": "stock_fhps_detail_em", "status": status})
    return pd.DataFrame(result, columns=columns)


def fetch_corporate_actions(symbol: str) -> pd.DataFrame:
    import akshare as ak
    return normalize_corporate_actions(symbol, ak.stock_fhps_detail_em(symbol=symbol))


def build_bundle(symbols: list[str], output_dir: str | Path, *, start: str, end: str, calendar_path: str | Path, benchmark_path: str | Path, listing_path: str | Path, delisted_path: str | Path, factor_registry: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir); cache = output / "cache"; cache.mkdir(parents=True, exist_ok=True)
    failures = []; price_frames = []; action_frames = []
    for symbol in sorted({str(item).zfill(6) for item in symbols}):
        price_cache = cache / f"{symbol}_prices.parquet"; action_cache = cache / f"{symbol}_actions.parquet"
        try:
            if price_cache.exists():
                merged = pd.read_parquet(price_cache)
            else:
                hfq = fetch_daily_unified(symbol, start=start.replace("-", ""), end=end.replace("-", ""), adjust="hfq")
                raw = fetch_daily_unified(symbol, start=start.replace("-", ""), end=end.replace("-", ""), adjust="")
                hfq["date"] = pd.to_datetime(hfq.date); raw["date"] = pd.to_datetime(raw.date)
                signal = hfq[["date", "open", "high", "low", "close"]].rename(columns={c: f"hfq_{c}" for c in ("open", "high", "low", "close")})
                execution = raw[["date", "open", "high", "low", "close", "volume", "amount"]].rename(columns={c: f"raw_{c}" for c in ("open", "high", "low", "close")})
                merged = signal.merge(execution, on="date", how="inner"); merged["code"] = symbol
                merged[["open", "high", "low", "close"]] = merged[["hfq_open", "hfq_high", "hfq_low", "hfq_close"]]
                merged = compute_price_factors(merged); merged.to_parquet(price_cache, index=False)
            if action_cache.exists():
                actions = pd.read_parquet(action_cache)
            else:
                actions = fetch_corporate_actions(symbol); actions.to_parquet(action_cache, index=False)
            price_frames.append(merged); action_frames.append(actions)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
    if not price_frames:
        raise ValueError("no price data collected")
    prices = pd.concat(price_frames, ignore_index=True).sort_values(["date", "code"])
    actions = pd.concat(action_frames, ignore_index=True).sort_values(["date", "symbol"]) if action_frames else pd.DataFrame()
    prices_path = output / "prices_hfq_raw.parquet"; actions_path = output / "corporate_actions.parquet"
    prices.to_parquet(prices_path, index=False); actions.to_parquet(actions_path, index=False)
    codes = set(prices.code.astype(str))
    listing = pd.read_parquet(listing_path); listing["code"] = listing.code.astype(str).str.zfill(6); listing = listing[listing.code.isin(codes)].rename(columns={"list_date": "listing_date"})
    delisted = pd.read_parquet(delisted_path); delisted["code"] = delisted.code.astype(str).str.zfill(6)
    # This bundle is a fixed research universe, not an index-membership history.
    # Availability starts at the first frozen price row and is still bounded by
    # the real listing/delisting dates. This prevents rows before membership
    # while keeping the PIT meaning explicit in the artifact.
    first_price = prices.groupby("code", as_index=False)["date"].min().rename(columns={"date": "first_price_date"})
    membership = listing.merge(first_price, on="code", how="right")
    membership = membership.merge(delisted[["code", "delist_date"]], on="code", how="left")
    membership["listing_date"] = pd.to_datetime(membership["listing_date"], errors="coerce")
    membership["first_price_date"] = pd.to_datetime(membership["first_price_date"], errors="coerce")
    membership["start_date"] = membership[["listing_date", "first_price_date"]].max(axis=1)
    membership = membership.rename(columns={"delist_date": "end_date"})
    membership["membership_type"] = "fixed_research_universe"
    membership["source"] = "bundle_first_price+listing_delisted"
    membership_path = output / "universe_membership.parquet"; membership.to_parquet(membership_path, index=False)
    calendar = pd.read_csv(calendar_path); calendar_path_out = output / "trade_calendar.parquet"; calendar.to_parquet(calendar_path_out, index=False)
    benchmark = pd.read_csv(benchmark_path); benchmark_path_out = output / "benchmark_csi300.parquet"; benchmark.to_parquet(benchmark_path_out, index=False)
    registry_path = output / "factor_registry.json"; registry_path.write_text(json.dumps(factor_registry, ensure_ascii=False, indent=2), encoding="utf-8")
    if config_path and Path(config_path).exists():
        (output / "backtest_config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    assets = []
    for path in (prices_path, actions_path, membership_path, calendar_path_out, benchmark_path_out, registry_path, output / "backtest_config.yaml"):
        if path.exists(): assets.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    manifest = {"schema_version": "1.0", "start": start, "end": end, "requested_symbols": len(set(symbols)), "collected_symbols": int(prices.code.nunique()), "price_rows": len(prices), "action_rows": len(actions), "membership_rows": len(membership), "failures": failures, "assets": assets}
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
