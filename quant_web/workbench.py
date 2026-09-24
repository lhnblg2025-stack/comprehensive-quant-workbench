"""Compact projection of the canonical decision snapshot for the web UI.

The full decision snapshot is intentionally audit-friendly and can contain
thousands of candidate rows.  A browser landing page should not download or
render that payload.  This module keeps the canonical source intact and
projects only the decision-grade fields required for the first screen.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


WORKBENCH_CONTRACT = "quant-web.workbench.v2"


def _clean(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else default
    return value


def _rows(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value[:limit] if isinstance(row, Mapping)]


def _source_status(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    dates = snapshot.get("source_dates") or {}
    freshness = snapshot.get("source_freshness") or {}
    chain = snapshot.get("source_chain") or {}
    names = (
        ("fusion", "市场融合"),
        ("fund_forces", "资金合力"),
        ("concept_fund_flow", "概念资金"),
        ("industry_fund_flow", "行业资金"),
        ("research_flow", "研报提取"),
        ("stock_fund_flow", "个股资金"),
    )
    result = []
    snapshot_day = str(snapshot.get("as_of") or "")[:10]
    for key, label in names:
        as_of = dates.get(key)
        fresh = freshness.get(key)
        # 部分源（研报/个股资金）没有单独写 source_freshness，按实际数据日与
        # 快照日比较；同日即有效，不能因字段缺省误报陈旧。
        if fresh is True or (fresh is None and as_of and str(as_of)[:10] == snapshot_day):
            status = "available"
        elif as_of:
            status = "stale"
        else:
            status = "missing"
        result.append({
            "key": key,
            "label": label,
            "status": status,
            "as_of": as_of,
            "fresh": fresh is True,
            "source": chain.get(key),
        })
    return result


def project_snapshot(snapshot: Mapping[str, Any], *, requested_date: str | None = None) -> dict[str, Any]:
    """Project a full snapshot into a stable, small, UI-oriented response."""
    market = dict(snapshot.get("market") or {})
    zt = dict(market.get("zt") or {})
    counts = dict(snapshot.get("counts") or {})
    degraded = bool(snapshot.get("degraded"))
    source_status = _source_status(snapshot)
    available_sources = sum(row["status"] == "available" for row in source_status)
    stale_sources = sum(row["status"] == "stale" for row in source_status)
    missing_sources = sum(row["status"] == "missing" for row in source_status)
    raw_confidence = _clean(snapshot.get("confidence"))
    if raw_confidence is None:
        # 置信度只反映证据覆盖，不代表收益概率；缺失和陈旧会明确扣分。
        raw_confidence = max(0.1, min(0.95, (available_sources * 1.0 + stale_sources * 0.45) / max(1, len(source_status))))
    confidence_components = dict(snapshot.get("confidence_components") or {})
    confidence_components.setdefault("base", f"{available_sources}/{len(source_status)} 个核心来源可用")
    confidence_components.setdefault("degraded_sources", f"陈旧{stale_sources}个，缺失{missing_sources}个")
    confidence_components.setdefault("social_sentiment", "未作为决策主证据")
    candidates = _rows(snapshot.get("short_term_candidates"), 5)
    if not candidates:
        candidates = _rows(snapshot.get("observation_candidates"), 5)
    mainlines = _rows(snapshot.get("mainlines"), 6)
    return {
        "ok": bool(snapshot.get("ok", True)),
        "contract": WORKBENCH_CONTRACT,
        "source_contract": snapshot.get("schema_version") or "decision-snapshot.v1",
        "mode": snapshot.get("mode") or "after_close",
        "requested_date": requested_date or snapshot.get("requested_date"),
        "as_of": snapshot.get("as_of"),
        "generated_at": snapshot.get("generated_at"),
        "state": "degraded" if degraded else "ready",
        "state_label": "证据降级，仅作观察" if degraded else "证据链完整，可继续复核",
        "market": {
            "temperature": _clean(market.get("temperature")),
            "force_index": _clean(market.get("force_index")),
            "breadth": _clean(market.get("breadth")),
            "emotion_stage": market.get("emotion_stage") or "待确认",
            "risk_flags": [str(x) for x in (market.get("risk_flags") or [])[:6]],
            "limit_up": _clean(zt.get("zt_cnt")),
            "limit_down": _clean(zt.get("dt_cnt")),
            "broken_board": _clean(zt.get("zb_cnt")),
            "max_board": _clean(zt.get("max_board")),
        },
        "mainlines": mainlines,
        "candidates": candidates,
        "candidate_mode": "short_term" if snapshot.get("short_term_candidates") else "observation",
        "counts": {
            "scanned": counts.get("scanned", 0),
            "mainlines": counts.get("mainlines", 0),
            "execution_candidates": counts.get("execution_candidates", 0),
            "observation_candidates": counts.get("observation_candidates", 0),
            "risk_flags": counts.get("risk_flags", 0),
        },
        "intraday_coverage": dict(snapshot.get("intraday_coverage") or {}),
        "source_status": source_status,
        "source_dates": dict(snapshot.get("source_dates") or {}),
        "date_mismatches": dict(snapshot.get("date_mismatches") or {}),
        "decision_basis": dict(snapshot.get("decision_basis") or {}),
        "confidence": raw_confidence,
        "confidence_components": confidence_components,
        "empty_policy": {
            "candidates": "没有同日主线、行业资金和风险门控确认时显示观察态，不补零、不凑数",
            "missing_source": "显示缺失源、最后数据日和影响，不把缺失当成看空或看多",
        },
    }


def build_workbench(*, mode: str = "after_close", date: str | None = None) -> dict[str, Any]:
    """Build the compact projection from a persisted snapshot when possible.

    The landing page is a read view.  It must prefer an already sealed artifact
    and only compute a snapshot when no eligible artifact exists; otherwise a
    refresh of the browser would unexpectedly start a full-market scan.
    """
    from pathlib import Path
    import json
    import re
    import sys

    root = Path(__file__).resolve().parents[1]
    generated = root / "generated"
    pattern = f"decision_snapshot_{mode}_*.json"
    candidates = []
    for path in generated.glob(pattern):
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
        if match and (not date or match.group(1) <= date):
            candidates.append((match.group(1), path))
    if candidates:
        _, path = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(snapshot, dict):
                return project_snapshot(snapshot, requested_date=date)
        except (OSError, ValueError, TypeError):
            pass

    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from unified_decision_snapshot import build_snapshot

    snapshot = build_snapshot(mode=mode, date=date)
    return project_snapshot(snapshot, requested_date=date)
