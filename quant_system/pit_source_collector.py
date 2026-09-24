"""Versioned source collector for A-share PIT universe and tradability inputs.

Qlib consumes a provider; it does not manufacture the provider's asset lifecycle,
ST history, suspension, or price-limit facts. This module probes and materializes
those facts only from configured licensed sources, records endpoint permissions,
and fails visibly when an authoritative source is unavailable.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"


def _tushare_client():
    from scripts.secret_loader import get_secret
    token = get_secret("TUSHARE_API_KEY")
    if not token:
        return None, "tushare_token_unavailable"
    try:
        import tushare as ts
        return ts.pro_api(token), None
    except Exception as exc:
        return None, f"tushare_client_error:{type(exc).__name__}:{str(exc)[:160]}"


def _probe_endpoint(name: str, fn) -> dict[str, Any]:
    try:
        frame = fn()
        return {"status": "available", "rows": int(len(frame)), "columns": list(frame.columns)}
    except Exception as exc:
        text = str(exc)
        if "没有接口" in text or "访问权限" in text:
            status = "permission_denied"
        elif "频率超限" in text or "频次" in text:
            status = "rate_limited"
        else:
            status = "error"
        return {"status": status, "error": f"{type(exc).__name__}:{text[:300]}"}


def probe_tushare_capabilities() -> dict[str, Any]:
    """Probe only tiny bounded endpoint requests; never persist secrets."""
    pro, error = _tushare_client()
    if pro is None:
        return {"provider": "tushare_pro", "configured": False, "error": error, "endpoints": {}}
    checks = {
        "trade_cal": lambda: pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240105", fields="exchange,cal_date,is_open"),
        "stock_basic": lambda: pro.stock_basic(exchange="", list_status="D", fields="ts_code,symbol,name,list_date,delist_date"),
        "namechange": lambda: pro.namechange(ts_code="600000.SH", start_date="20240101", end_date="20240131", fields="ts_code,name,start_date,end_date,ann_date,change_reason"),
        "suspend_d": lambda: pro.suspend_d(ts_code="600000.SH", start_date="20240101", end_date="20240131", fields="ts_code,trade_date,suspend_timing,suspend_type"),
        "stk_limit": lambda: pro.stk_limit(ts_code="600000.SH", start_date="20240101", end_date="20240105", fields="ts_code,trade_date,pre_close,up_limit,down_limit"),
    }
    return {"provider": "tushare_pro", "configured": True, "endpoints": {name: _probe_endpoint(name, fn) for name, fn in checks.items()}}


def materialize_local_security_lifecycle(output_dir: Path = PANEL_ROOT) -> dict[str, Any]:
    """Version existing security-master listing/delisting facts without claiming PIT membership."""
    master = output_dir / "security_master_source.parquet"
    if not master.is_file():
        return {"status": "missing", "error": "security_master_source_missing"}
    data = pd.read_parquet(master)
    required = {"ts_code", "symbol", "list_date", "delist_date"}
    missing = sorted(required - set(data.columns))
    if missing:
        return {"status": "invalid", "error": f"security_master_missing_columns:{','.join(missing)}"}
    out = data[["ts_code", "symbol", "list_date", "delist_date"]].copy()
    out["code"] = out["symbol"].astype(str).str.zfill(6)
    out["list_date"] = pd.to_datetime(out["list_date"], errors="coerce")
    out["delist_date"] = pd.to_datetime(out["delist_date"], errors="coerce")
    out["source_as_of"] = "security_master_snapshot_not_dated_membership"
    path = output_dir / "pit_security_lifecycle_snapshot.parquet"
    out.to_parquet(path, index=False)
    return {"status": "materialized_snapshot_only", "path": str(path), "rows": int(len(out)), "listed_dates": int(out["list_date"].notna().sum()), "delist_dates": int(out["delist_date"].notna().sum())}


def _security_codes(output_dir: Path) -> list[tuple[str, str]]:
    master = output_dir / "security_master_source.parquet"
    if not master.is_file():
        return []
    frame = pd.read_parquet(master, columns=["ts_code", "symbol"])
    return [(str(row.ts_code), str(row.symbol).zfill(6)) for row in frame.dropna().itertuples(index=False)]


def _collect_symbol_shards(pro, endpoint: str, codes: list[tuple[str, str]], output_dir: Path, *, start: str, end: str) -> dict[str, Any]:
    root = output_dir / "raw" / "pit_sources" / endpoint
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    stored = 0
    method = getattr(pro, endpoint)
    for ts_code, symbol in codes:
        path = root / f"{symbol}.parquet"
        if path.is_file():
            stored += 1
            continue
        try:
            frame = method(ts_code=ts_code, start_date=start, end_date=end)
            if frame is None:
                frame = pd.DataFrame()
            frame.to_parquet(path, index=False)
            stored += 1
        except Exception as exc:
            failures.append({"ts_code": ts_code, "error": f"{type(exc).__name__}:{str(exc)[:240]}"})
            # Permission/rate failures are provider-wide; stop instead of issuing
            # hundreds of identical rejected requests.
            text = str(exc)
            if "没有接口" in text or "访问权限" in text or "频率超限" in text:
                break
    return {"endpoint": endpoint, "requested": len(codes), "stored": stored, "failures": failures, "shard_root": str(root)}


def materialize_observed_calendar(output_dir: Path = PANEL_ROOT) -> dict[str, Any]:
    """Materialize sessions observed in the frozen price panel; not official calendar."""
    panel = output_dir / "price_discovery_panel.parquet"
    if not panel.is_file():
        return {"status": "missing", "error": "price_discovery_panel_missing"}
    dates = pd.to_datetime(pd.read_parquet(panel, columns=["date"])["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    frame = pd.DataFrame({"date": dates, "is_open": True, "source_as_of": "observed_price_panel_sessions_not_exchange_calendar"})
    path = output_dir / "observed_trade_calendar.parquet"
    frame.to_parquet(path, index=False)
    return {"status": "materialized_research_only", "path": str(path), "sessions": int(len(frame)), "date_min": str(frame["date"].min().date()), "date_max": str(frame["date"].max().date())}


def materialize_permitted_tushare_sources(output_dir: Path = PANEL_ROOT, *, start: str = "20160101", end: str = "20260831") -> dict[str, Any]:
    """Resumably collect licensed PIT inputs; never derive unavailable fields."""
    capabilities = probe_tushare_capabilities()
    pro, error = _tushare_client()
    if pro is None:
        return {"status": "unavailable", "error": error, "domains": {}}
    endpoints = capabilities.get("endpoints", {})
    domains: dict[str, Any] = {"observed_trade_calendar": materialize_observed_calendar(output_dir)}
    # Official calendar is small but still subject to endpoint rate limits.
    if endpoints.get("trade_cal", {}).get("status") == "available":
        try:
            calendar = pro.trade_cal(exchange="", start_date=start, end_date=end, fields="exchange,cal_date,is_open,pretrade_date")
            calendar_path = output_dir / "pit_trade_calendar.parquet"
            calendar.to_parquet(calendar_path, index=False)
            domains["trade_cal"] = {"status": "stored", "rows": int(len(calendar)), "path": str(calendar_path)}
        except Exception as exc:
            domains["trade_cal"] = {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
    codes = _security_codes(output_dir)
    for endpoint in ("namechange", "suspend_d", "stk_limit"):
        status = endpoints.get(endpoint, {}).get("status")
        if status != "available":
            domains[endpoint] = {"status": "skipped", "source_status": status or "unknown"}
            continue
        result = _collect_symbol_shards(pro, endpoint, codes, output_dir, start=start, end=end)
        result["status"] = "complete" if not result["failures"] and result["stored"] == result["requested"] else "partial"
        domains[endpoint] = result
    manifest = {"schema": "tushare_pit_materialization/v1", "start": start, "end": end, "domains": domains, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (output_dir / "tushare_pit_materialization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run(output_dir: Path = PANEL_ROOT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capabilities = probe_tushare_capabilities()
    lifecycle = materialize_local_security_lifecycle(output_dir)
    endpoints = capabilities.get("endpoints", {})
    required = {"stock_basic", "namechange", "suspend_d", "stk_limit"}
    unavailable = {key: endpoints.get(key, {}).get("status", "not_configured") for key in required if endpoints.get(key, {}).get("status") != "available"}
    report = {
        "schema": "pit_source_capabilities/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "capabilities": capabilities,
        "local_lifecycle": lifecycle,
        "required_for_authoritative_admission": sorted(required),
        "unavailable_or_limited": unavailable,
        "status": "READY" if not unavailable else "SOURCE_GAPS",
        "next_action": "grant_tushare_permissions_or_configure_licensed_exchange_vendor" if unavailable else "run_versioned_pit_materialization",
    }
    path = output_dir / "pit_source_capabilities.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
