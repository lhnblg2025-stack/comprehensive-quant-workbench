"""Fail-closed data gates for strategy research and formal backtests.

The gate intentionally separates ``allowed_for_research`` from
``allowed_for_formal_backtest``.  A research result can be useful while still
being unsuitable for production claims because the current universe is a
frozen sample and authoritative point-in-time trade state is incomplete.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_warehouse/research_panels/ashare_tushare_500_10y_v2/price_discovery_panel.parquet"
DEFAULT_INDEX_DIR = ROOT / "data_warehouse/staging/public_sse_index_daily_20260923/normalized"
DEFAULT_INDEX_MANIFEST = ROOT / "data_warehouse/staging/public_sse_index_daily_20260923/manifest.json"
DEFAULT_AUDIT = ROOT / "data_warehouse/strategy_data_readiness_audit.json"


@dataclass(frozen=True)
class StrategyRequirement:
    family: str
    required_data_domains: tuple[str, ...]
    required_panel_fields: tuple[str, ...] = ()
    required_index_fields: tuple[str, ...] = ()
    research_policy: str = "allow"
    formal_policy: str = "block"
    known_limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyGate:
    family: str
    status: str
    allowed_for_research: bool
    allowed_for_formal_backtest: bool
    required_data_domains: tuple[str, ...]
    available_data: tuple[str, ...]
    missing_fields: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    data_fingerprints: dict[str, str]
    date_range: dict[str, str | None]
    universe: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIREMENTS: dict[str, StrategyRequirement] = {
    "trend": StrategyRequirement(
        "trend", ("stock_price_daily",), ("date", "code", "close", "high", "low"),
        research_policy="allow", formal_policy="block",
        known_limitations=("frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"),
    ),
    "reversal": StrategyRequirement(
        "reversal", ("stock_price_daily",), ("date", "code", "close", "high", "low"),
        research_policy="allow", formal_policy="block",
        known_limitations=("frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"),
    ),
    "volatility": StrategyRequirement(
        "volatility", ("stock_price_daily",), ("date", "code", "close", "high", "low"),
        research_policy="allow", formal_policy="block",
        known_limitations=("frozen_universe", "synthetic_proxy_fields", "no_authoritative_trade_state"),
    ),
    "volume_flow": StrategyRequirement(
        "volume_flow", ("stock_price_daily", "liquidity_daily"),
        ("date", "code", "close", "volume", "amount", "turnover"),
        research_policy="allow", formal_policy="block",
        known_limitations=("synthetic_proxy_fields", "source_unit_normalization", "frozen_universe"),
    ),
    "cross_section": StrategyRequirement(
        "cross_section", ("stock_price_daily", "liquidity_daily"),
        ("date", "code", "close", "amount", "mom20", "mom60", "mom120"),
        research_policy="allow", formal_policy="block",
        known_limitations=("frozen_universe", "synthetic_proxy_fields", "no_pit_membership"),
    ),
    "market_timing": StrategyRequirement(
        "market_timing", ("official_index_daily",), required_index_fields=("trade_date", "index_code", "close", "volume", "amount"),
        research_policy="allow", formal_policy="block",
        known_limitations=("index_data_is_not_stock_trade_state", "frozen_stock_universe"),
    ),
    "pit_value_quality": StrategyRequirement(
        "pit_value_quality", ("pit_financials", "pit_trade_state"),
        ("date", "code"), research_policy="block", formal_policy="block",
        known_limitations=("announcement_asof_missing", "authoritative_trade_state_missing"),
    ),
    "execution_audit": StrategyRequirement(
        "execution_audit", ("pit_trade_state", "order_ledger"),
        ("date", "code"), research_policy="block", formal_policy="block",
        known_limitations=("authoritative_trade_state_missing", "execution_reconciliation_missing"),
    ),
    "unbiased_market": StrategyRequirement(
        "unbiased_market", ("pit_membership", "security_master", "pit_trade_state"),
        ("date", "code"), research_policy="block", formal_policy="block",
        known_limitations=("point_in_time_membership_missing", "delisted_securities_missing"),
    ),
    "event_flow": StrategyRequirement(
        "event_flow", ("pit_events", "pit_trade_state"), ("date", "code"),
        research_policy="block", formal_policy="block",
        known_limitations=("point_in_time_event_asof_missing",),
    ),
    "pairs": StrategyRequirement(
        "pairs", ("multi_asset_price_daily",), ("date", "code", "close"),
        research_policy="block", formal_policy="block",
        known_limitations=("dedicated_multi_asset_panel_missing",),
    ),
    "ml": StrategyRequirement(
        "ml", ("stock_price_daily", "pit_trade_state", "pit_membership"),
        ("date", "code", "close"), research_policy="block", formal_policy="block",
        known_limitations=("point_in_time_training_universe_missing",),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _inspect_panel(path: Path, required: tuple[str, ...]) -> tuple[set[str], list[str], dict[str, Any], dict[str, str]]:
    if not path.is_file():
        return set(), ["stock_price_panel_missing"], {}, {}
    frame = pd.read_parquet(path)
    columns = set(frame.columns)
    aliases = {
        "close": ("close", "hfq_close", "raw_close"),
        "high": ("high", "hfq_high", "raw_high"),
        "low": ("low", "hfq_low", "raw_low"),
        "open": ("open", "hfq_open", "raw_open"),
        "turnover": ("turnover", "turnover_rate", "raw_turnover"),
        "volume": ("volume", "vol"),
        "amount": ("amount",),
        "date": ("date", "trade_date"),
        "code": ("code", "ts_code", "symbol"),
    }
    missing = [
        field for field in required
        if not any(alias in columns for alias in aliases.get(field, (field,)))
    ]
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in columns else pd.Series(dtype="datetime64[ns]")
    proxy_ratio = float(frame["synthetic_proxy"].fillna(False).mean()) if "synthetic_proxy" in columns and len(frame) else None
    universe = {
        "rows": int(len(frame)),
        "symbols": int(frame["code"].nunique()) if "code" in columns else 0,
        "synthetic_proxy_ratio": proxy_ratio,
        "frozen_universe_warning": True,
    }
    date_range = {
        "min": dates.min().date().isoformat() if dates.notna().any() else None,
        "max": dates.max().date().isoformat() if dates.notna().any() else None,
    }
    available = {
        field for field in required
        if any(alias in columns for alias in aliases.get(field, (field,)))
    }
    return available, missing, {**universe, **date_range}, {"stock_panel": _sha256(path)}


def _inspect_indexes(directory: Path, manifest_path: Path, required: tuple[str, ...]) -> tuple[list[str], list[str], dict[str, Any], dict[str, str]]:
    manifest = _load_manifest(manifest_path)
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    available: list[str] = []
    missing: list[str] = []
    if not files:
        return available, ["official_index_daily_missing"], {}, {}
    date_values: list[pd.Timestamp] = []
    codes: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path)
        missing.extend(field for field in required if field not in frame.columns)
        available.extend(field for field in required if field in frame.columns and field not in available)
        if "trade_date" in frame:
            dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
            if not dates.empty:
                date_values.extend(dates.tolist())
        if "index_code" in frame:
            codes.update(frame["index_code"].astype(str).unique())
    manifest_statuses = {str(item.get("status")) for item in manifest.get("datasets", [])}
    if manifest_statuses and manifest_statuses != {"PASS"}:
        missing.append("official_index_manifest_not_all_pass")
    fingerprint = _sha256(manifest_path) if manifest_path.is_file() else ""
    return (
        sorted(set(available)),
        sorted(set(missing)),
        {
            "files": len(files),
            "indexes": len(codes),
            "date_min": min(date_values).date().isoformat() if date_values else None,
            "date_max": max(date_values).date().isoformat() if date_values else None,
            "manifest_status": "PASS" if not manifest_statuses or manifest_statuses == {"PASS"} else "REVIEW",
        },
        {"index_manifest": fingerprint},
    )


def evaluate_strategy(
    family: str,
    *,
    panel_path: str | Path = DEFAULT_PANEL,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    index_manifest: str | Path = DEFAULT_INDEX_MANIFEST,
    audit_path: str | Path = DEFAULT_AUDIT,
) -> StrategyGate:
    """Evaluate one strategy family without running a strategy or mutating data."""
    if family not in REQUIREMENTS:
        raise KeyError(f"unknown_strategy_family:{family}")
    requirement = REQUIREMENTS[family]
    available: list[str] = []
    missing: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = list(requirement.known_limitations)
    fingerprints: dict[str, str] = {}
    date_range: dict[str, str | None] = {"min": None, "max": None}
    universe: dict[str, Any] = {}

    if requirement.required_panel_fields:
        panel_columns, panel_missing, panel_summary, panel_fingerprints = _inspect_panel(Path(panel_path), requirement.required_panel_fields)
        available.extend(sorted(panel_columns & set(requirement.required_panel_fields)))
        missing.extend(panel_missing)
        fingerprints.update(panel_fingerprints)
        date_range.update({"min": panel_summary.get("min"), "max": panel_summary.get("max")})
        universe.update({key: value for key, value in panel_summary.items() if key not in {"min", "max"}})
        if (panel_summary.get("synthetic_proxy_ratio") or 0.0) > 0:
            warnings.append("synthetic_proxy_fields_present")
        if not panel_summary:
            blockers.append("stock_price_panel_missing")

    if requirement.required_index_fields:
        index_available, index_missing, index_summary, index_fingerprints = _inspect_indexes(
            Path(index_dir), Path(index_manifest), requirement.required_index_fields
        )
        available.extend(index_available)
        missing.extend(index_missing)
        fingerprints.update(index_fingerprints)
        date_range.update({"min": index_summary.get("date_min"), "max": index_summary.get("date_max")})
        universe.update(index_summary)

    if missing:
        blockers.extend(f"missing_field:{field}" for field in sorted(set(missing)))
    if requirement.research_policy == "block":
        blockers.append("research_policy_blocked")
    allowed_research = not blockers
    allowed_formal = allowed_research and requirement.formal_policy == "allow"
    if not allowed_formal:
        warnings.append("formal_backtest_not_admitted")
    if blockers:
        status = "BLOCKED"
    elif family in {"volume_flow", "cross_section"}:
        status = "PARTIAL_RESEARCH"
    else:
        status = "READY_RESEARCH"
    return StrategyGate(
        family=family,
        status=status,
        allowed_for_research=allowed_research,
        allowed_for_formal_backtest=allowed_formal,
        required_data_domains=requirement.required_data_domains,
        available_data=tuple(sorted(set(available))),
        missing_fields=tuple(sorted(set(missing))),
        blockers=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
        data_fingerprints=fingerprints,
        date_range=date_range,
        universe=universe,
    )


def evaluate_families(families: list[str] | tuple[str, ...], **kwargs: Any) -> dict[str, dict[str, Any]]:
    return {family: evaluate_strategy(family, **kwargs).to_dict() for family in families}


def readiness_summary(gates: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        "ready_research": sum(value["status"] == "READY_RESEARCH" for value in gates.values()),
        "partial_research": sum(value["status"] == "PARTIAL_RESEARCH" for value in gates.values()),
        "blocked": sum(value["status"] == "BLOCKED" for value in gates.values()),
    }


def require_research_access(
    family: str,
    *,
    panel_path: str | Path = DEFAULT_PANEL,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    index_manifest: str | Path = DEFAULT_INDEX_MANIFEST,
    audit_path: str | Path = DEFAULT_AUDIT,
) -> StrategyGate:
    """Return a research gate or fail closed for a blocked strategy family."""
    gate = evaluate_strategy(
        family,
        panel_path=panel_path,
        index_dir=index_dir,
        index_manifest=index_manifest,
        audit_path=audit_path,
    )
    if not gate.allowed_for_research:
        raise RuntimeError(json.dumps({
            "strategy_family": family,
            "status": gate.status,
            "blockers": gate.blockers,
        }, ensure_ascii=True))
    return gate
