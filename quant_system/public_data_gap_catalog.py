"""Auditable catalog of public-data gaps used by the research workbench.

The catalog is intentionally a status document, not a claim that a web page is
downloadable.  A domain becomes ``AVAILABLE`` only when a local staging
artifact is present and its manifest says the artifact passed validation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_ROOT = ROOT / "data_warehouse" / "staging"
DEFAULT_OUTPUT_ROOT = DEFAULT_STAGING_ROOT / "public_data_gap_catalog_20260923"


@dataclass(frozen=True)
class PublicDataDomain:
    key: str
    title: str
    priority: str
    frequency: str
    required_fields: tuple[str, ...]
    strategy_uses: tuple[str, ...]
    official_sources: tuple[str, ...]
    candidate_paths: tuple[str, ...]
    known_limitations: tuple[str, ...] = ()


DOMAINS: tuple[PublicDataDomain, ...] = (
    PublicDataDomain(
        "official_index_daily", "官方指数日行情", "P1", "daily",
        ("index_code", "trade_date", "close", "volume", "amount"),
        ("market_timing", "trend", "volatility"),
        ("https://www.sse.com.cn/market/sseindex/",
         "https://www.szse.cn/market/trend/index.html",
         "https://www.csindex.com.cn/"),
        ("**/*index_daily*.parquet",),
        ("benchmark data does not provide stock membership or tradability",),
    ),
    PublicDataDomain(
        "exchange_calendar", "交易所官方交易日历", "P0", "daily",
        ("cal_date", "is_open"), ("all",),
        ("https://www.sse.com.cn/disclosure/dealinstruc/calendar/",
         "https://www.szse.cn/market/trend/index.html"),
        ("**/*trade_calendar*.parquet", "**/*trade_cal*.parquet"),
        ("observed price sessions are not an exchange calendar",),
    ),
    PublicDataDomain(
        "security_lifecycle", "上市退市与证券生命周期", "P0", "event",
        ("code", "list_date", "delist_date"), ("unbiased_market", "ml"),
        ("https://www.sse.com.cn/assortment/stock/list/share/",
         "https://www.szse.cn/market/product/stock/list/index.html"),
        ("**/*lifecycle*.parquet", "**/*security*master*.parquet"),
        ("snapshot security master is not dated membership",),
    ),
    PublicDataDomain(
        "suspension_resume", "历史停牌复牌", "P0", "event",
        ("code", "trade_date", "suspend_type"), ("unbiased_market", "execution_audit"),
        ("https://www.sse.com.cn/market/stockdata/overview/day/",
         "https://www.szse.cn/disclosure/listed/notice/index.html"),
        ("**/*suspend*.parquet", "**/*suspension*.parquet"),
        ("absence of a row is not proof of tradability",),
    ),
    PublicDataDomain(
        "risk_warning_history", "历史 ST 与风险警示", "P0", "event",
        ("code", "start_date", "end_date", "status"), ("unbiased_market", "event_flow"),
        ("https://www.sse.com.cn/assortment/stock/list/share/",
         "https://www.szse.cn/market/product/stock/list/index.html"),
        ("**/*namechange*.parquet", "**/*risk*warning*.parquet", "**/*st*.parquet"),
        ("current names cannot reconstruct historical ST state",),
    ),
    PublicDataDomain(
        "price_limits", "历史涨跌停价格", "P0", "daily",
        ("code", "trade_date", "up_limit", "down_limit"), ("execution_audit", "unbiased_market"),
        ("https://www.sse.com.cn/market/stockdata/overview/day/",
         "https://www.szse.cn/market/trend/index.html"),
        ("**/*stk_limit*.parquet", "**/*price*limit*.parquet"),
        ("limit rules differ by board and event state",),
    ),
    PublicDataDomain(
        "corporate_actions", "分红送转配股拆并股", "P0", "event",
        ("code", "ex_date", "action_type", "factor"), ("trend", "reversal", "execution_audit"),
        ("https://www.sse.com.cn/market/stockdata/overview/day/",
         "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/listedinfo/announcement"),
        ("**/*corporate*action*.parquet", "**/*dividend*.parquet", "**/*action*.parquet"),
        ("adjustment factors must reconcile to raw prices",),
    ),
    PublicDataDomain(
        "pit_index_membership", "指数历史成分与生效日", "P0", "event",
        ("index_code", "code", "effective_date"), ("market_timing", "unbiased_market"),
        ("https://www.csindex.com.cn/",
         "https://www.sse.com.cn/market/sseindex/"),
        ("**/*membership*.parquet", "**/*constituent*.parquet", "**/*roster*.csv"),
        ("a single snapshot is not point-in-time membership",),
    ),
    PublicDataDomain(
        "pit_financials", "公告可用时间与重述财务", "P0", "quarterly",
        ("code", "report_period", "announced_at"), ("pit_value_quality", "ml"),
        ("https://www.cninfo.com.cn/", "https://www.csrc.gov.cn/"),
        ("**/*pit*financial*.parquet", "**/*financial*release*.parquet"),
        ("report period is not announcement availability time",),
    ),
    PublicDataDomain(
        "margin_history", "融资融券历史", "P1", "daily",
        ("code", "trade_date", "margin_balance"), ("volume_flow", "event_flow"),
        ("https://www.sse.com.cn/market/others/margin/",
         "https://www.szse.cn/market/product/stock/margin/"),
        ("**/*margin*.parquet", "**/*financing*.parquet"),
    ),
    PublicDataDomain(
        "northbound_history", "北向资金与持股历史", "P1", "daily",
        ("code", "trade_date", "shares_held"), ("volume_flow", "cross_section"),
        ("https://www.hkex.com.hk/Mutual-Market/Stock-Connect"),
        ("**/*north*.parquet", "**/*connect*.parquet"),
        ("historical availability and disclosure timing must be recorded",),
    ),
    PublicDataDomain(
        "lhb_history", "龙虎榜历史", "P1", "daily",
        ("code", "trade_date", "reason"), ("event_flow", "volume_flow"),
        ("https://www.sse.com.cn/market/stockdata/overview/day/",
         "https://www.szse.cn/market/trend/index.html"),
        ("**/*lhb*.parquet", "**/*dragon*.parquet"),
    ),
    PublicDataDomain(
        "industry_history", "历史行业映射", "P1", "event",
        ("code", "industry", "effective_date"), ("cross_section", "ml"),
        ("https://www.csindex.com.cn/", "https://www.csrc.gov.cn/"),
        ("**/*industry*.parquet", "**/*sector*.parquet"),
        ("current industry labels cannot stand in for historical mappings",),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _matching_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(found)


def _status(files: list[Path], root: Path) -> tuple[str, dict[str, Any]]:
    if not files:
        return "MISSING", {"files": 0, "manifests": []}
    manifests = []
    passed = 0
    for file in files:
        manifest = next(
            (candidate for candidate in (file.parent / "manifest.json", file.parent.parent / "manifest.json", root / "manifest.json")
             if candidate.is_file()),
            None,
        )
        if manifest is None:
            continue
        record = _load_json(manifest)
        manifests.append(str(manifest.relative_to(root)))
        statuses = [
            str(item.get("status", "")).upper()
            for item in record.get("datasets", [])
            if isinstance(item, dict)
        ]
        statuses += [str(record.get("status", "")).upper()]
        if statuses and all(status in {"PASS", "AVAILABLE", "READY", "COMPLETE"} for status in statuses if status):
            passed += 1
    if passed:
        status = "AVAILABLE"
    else:
        status = "PARTIAL"
    return status, {
        "files": len(files),
        "manifests": manifests,
        "artifact_sha256": [_sha256(path) for path in files[:20]],
    }


def build_catalog(
    staging_root: str | Path = DEFAULT_STAGING_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    domains: Iterable[PublicDataDomain] = DOMAINS,
) -> dict[str, Any]:
    staging = Path(staging_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for domain in domains:
        files = _matching_files(staging, domain.candidate_paths) if staging.is_dir() else []
        status, evidence = _status(files, staging)
        if status == "AVAILABLE" and domain.known_limitations:
            status = "RESEARCH_ONLY"
        records.append({
            **asdict(domain),
            "required_fields": list(domain.required_fields),
            "strategy_uses": list(domain.strategy_uses),
            "official_sources": list(domain.official_sources),
            "candidate_paths": list(domain.candidate_paths),
            "known_limitations": list(domain.known_limitations),
            "status": status,
            "evidence": evidence,
        })
    counts = {status: sum(item["status"] == status for item in records) for status in
              ("AVAILABLE", "RESEARCH_ONLY", "PARTIAL", "MISSING", "BLOCKED_PERMISSION")}
    report = {
        "schema": "public_data_gap_catalog/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "staging_root": str(staging),
        "domains": records,
        "counts": counts,
        "admission_blockers": [
            item["key"] for item in records
            if item["status"] in {"MISSING", "PARTIAL", "BLOCKED_PERMISSION"}
            and item["priority"] == "P0"
        ],
        "policy": {
            "staging_only": True,
            "no_main_panel_mutation": True,
            "research_only_until_pit_complete": True,
        },
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Public Data Gap Catalog",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "| Domain | Priority | Status | Strategy uses |",
        "|---|---|---|---|",
    ]
    for item in records:
        lines.append(
            f"| {item['title']} (`{item['key']}`) | {item['priority']} | "
            f"`{item['status']}` | {', '.join(item['strategy_uses'])} |"
        )
    lines.extend(["", "## Admission blockers", ""])
    lines.extend(f"- `{key}`" for key in report["admission_blockers"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    build_catalog()
