"""Build field-level readiness catalog for the A-share research provider."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_fundamentals import coverage_summary

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2"
FACTORS = (
    "value_book_to_price_industry_neutral",
    "quality_low_leverage_industry_neutral",
    "value_quality_composite_industry_neutral",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build(root: Path = PANEL_ROOT) -> dict[str, Any]:
    price_manifest = _json(root / "price_collection_manifest.json")
    financial_report = _json(root / "annual_pit_financial_report.json")
    action_report = _json(root / "corporate_actions_report.json")
    capabilities = _json(root / "pit_source_capabilities.json")
    cloud_manifest = _json(root / "cloud_pit_tradability" / "manifest.json")
    pit_panel_path = root / "annual_pit_research_panel.parquet"
    coverage = {"status": "missing"}
    if pit_panel_path.is_file():
        panel = pd.read_parquet(pit_panel_path, columns=["date", "code", *FACTORS])
        coverage = coverage_summary(panel, FACTORS, min_stocks=30, min_coverage_ratio=0.80)
    observed_calendar = root / "observed_trade_calendar.parquet"
    domains = {
        "observed_trade_calendar": {
            "status": "READY_RESEARCH" if observed_calendar.is_file() else "MISSING",
            "source": "observed_price_panel_sessions_not_exchange_calendar",
            "sessions": int(len(pd.read_parquet(observed_calendar))) if observed_calendar.is_file() else 0,
            "limitations": ["not_official_exchange_calendar"],
        },
        "daily_price_liquidity": {
            "status": "READY_RESEARCH",
            "source": price_manifest.get("price_source"),
            "requested": price_manifest.get("requested"),
            "stored": price_manifest.get("stored"),
            "limitations": ["fixed_research_universe"],
        },
        "annual_pit_financials": {
            "status": "READY_RESEARCH" if financial_report.get("symbols") == 500 and financial_report.get("announcement_coverage") == financial_report.get("releases") else "INCOMPLETE",
            "source": "annual_fundamental_releases+actual_disclosure_dates",
            "symbols": financial_report.get("symbols"),
            "releases": financial_report.get("releases"),
            "coverage": coverage,
            "limitations": financial_report.get("limitations", []),
        },
        "corporate_actions": {
            "status": "PARTIAL_RESEARCH" if action_report.get("actions") else "MISSING",
            "actions": action_report.get("actions", 0),
            "symbols": action_report.get("action_symbols", 0),
            "failures": action_report.get("failures", []),
        },
        "security_lifecycle_membership": {
            "status": "BLOCKED_SOURCE",
            "reason": "stock_basic_rate_limited_and_snapshot_has_no_listing_dates",
            "required": ["list_date", "delist_date", "dated_membership"],
        },
        "historical_st_namechange": {
            "status": "BLOCKED_SOURCE",
            "endpoint": "tushare.namechange",
            "source_status": capabilities.get("unavailable_or_limited", {}).get("namechange", "unknown"),
        },
        "historical_suspension": {
            "status": "PARTIAL_RESEARCH" if cloud_manifest.get("remote", {}).get("stored", 0) else "BLOCKED_PERMISSION",
            "endpoint": "tencent_cloud_akshare_stock_tfp_em",
            "source_status": capabilities.get("unavailable_or_limited", {}).get("suspend_d", "unknown"),
            "cloud_observed_dates": cloud_manifest.get("remote", {}).get("stored", 0),
            "limitations": ["absence_is_not_tradable_true", "partial_date_coverage", "not_exchange_authoritative_full_state"],
        },
        "historical_price_limits": {
            "status": "BLOCKED_PERMISSION",
            "endpoint": "tushare.stk_limit",
            "source_status": capabilities.get("unavailable_or_limited", {}).get("stk_limit", "unknown"),
        },
        "quarterly_pit_financials": {
            "status": "MISSING",
            "reason": "quarterly_releases_not_collected",
        },
    }
    blockers = [name for name, info in domains.items() if str(info.get("status", "")).startswith(("BLOCKED", "MISSING"))]
    report = {
        "schema": "pit_source_catalog/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "domains": domains,
        "research_ready_domains": [name for name, info in domains.items() if info.get("status") in {"READY_RESEARCH", "PARTIAL_RESEARCH"}],
        "admission_blocking_domains": blockers,
        "status": "SOURCE_GAPS" if blockers else "READY",
    }
    path = root / "pit_source_catalog.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
