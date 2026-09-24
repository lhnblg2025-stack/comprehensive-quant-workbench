#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared research-report contract and renderer.

The contract separates facts, mechanisms and actions.  It is deliberately
stdlib-only so every report generator can use it, even when optional market
data packages are unavailable.
"""
from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

REPORT_CONTRACT = "research_report.v1"
SCOPE = "analysis_decision_assist_only"
SCOPE_DETAIL = {
    "analysis": "用于市场、行业、商品、个股和策略研究分析",
    "execution": "仅提供分析与决策辅助，不连接券商，不下真实订单",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        return _json_safe(value.item())
    except Exception:
        return str(value)


def _list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _canonical_source_status(value: Any) -> str:
    """Map legacy source states to the research contract vocabulary."""
    state = str(value or "unknown").lower()
    return {
        "available": "ok", "success": "ok", "true": "ok",
        "missing": "failed", "error": "failed", "unavailable": "failed",
        "partial": "fallback", "provider_empty": "failed",
    }.get(state, state if state in {"ok", "failed", "stale", "fallback", "proxy"} else "failed")


def _source(item: Any, index: int) -> dict:
    if isinstance(item, str):
        item = {"name": item}
    item = dict(item or {})
    item.setdefault("id", f"src-{index}")
    item.setdefault("name", item.get("source") or "未命名来源")
    item.setdefault("kind", "unknown")
    item["status"] = _canonical_source_status(item.get("status"))
    item.setdefault("observed_at", item.get("as_of"))
    item.setdefault("retrieved_at", None)
    item.setdefault("locator", item.get("path") or item.get("url"))
    item.setdefault("note", "")
    return _json_safe(item)


def _evidence(item: Any, index: int) -> dict:
    if isinstance(item, str):
        return {"id": f"ev-{index}", "claim": item, "value": None, "unit": "text", "observed_at": None, "source_ref": None, "quality": "primary", "note": ""}
    item = dict(item or {})
    item.setdefault("id", f"ev-{index}")
    item.setdefault("claim", item.get("name") or "未命名事实")
    item.setdefault("value", None)
    item.setdefault("unit", "text")
    item.setdefault("observed_at", item.get("as_of"))
    item.setdefault("source_ref", None)
    item.setdefault("quality", "unknown")
    item.setdefault("note", "")
    return _json_safe(item)


def _chapter(item: Any, index: int) -> dict:
    if isinstance(item, str):
        item = {"title": f"章节 {index}", "conclusion": item}
    item = dict(item or {})
    item.setdefault("title", f"章节 {index}")
    item.setdefault("conclusion", "暂无结论")
    item.setdefault("evidence", [])
    item.setdefault("implication", "暂无资产或策略含义")
    item.setdefault("next_check", "暂无后续验证条件")
    return _json_safe(item)


def make_report_contract(
    report_type: str,
    title: str,
    as_of: Any,
    *,
    subject: dict[str, Any] | None = None,
    report_id: str | None = None,
    period_start: Any = None,
    period_end: Any = None,
    trading_days: int | None = None,
    summary: Any = None,
    evidence: Iterable[Any] | None = None,
    transmission: Iterable[Any] | None = None,
    risks: Iterable[Any] | None = None,
    actions: Iterable[Any] | None = None,
    data_gaps: Iterable[Any] | None = None,
    sources: Iterable[Any] | None = None,
    chapters: Iterable[Any] | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> dict:
    """Create the canonical report object; unknown data stays null/empty."""
    if isinstance(chapters, dict):
        chapter_values = [dict(value, title=key) if isinstance(value, dict) else {"title": key, "conclusion": value}
                          for key, value in chapters.items()]
    else:
        chapter_values = list(chapters or [])
    summary_value = summary if isinstance(summary, dict) else {"stance": "observe", "summary": str(summary or "暂无综合结论"), "horizon": "weeks", "confidence": None}
    summary_value.setdefault("stance", "observe")
    summary_value.setdefault("summary", "暂无综合结论")
    summary_value.setdefault("horizon", "weeks")
    summary_value.setdefault("confidence", None)
    source_values = [_source(item, i + 1) for i, item in enumerate(sources or [])]
    evidence_values = [_evidence(item, i + 1) for i, item in enumerate(evidence or [])]
    for item in evidence_values:
        if not item.get("observed_at") and as_of:
            item["observed_at"] = str(as_of)[:10]
    # Every relationship gets a stable id so the viewer and downstream agents can
    # reference it without parsing Markdown.
    transmission_values = []
    for i, item in enumerate(transmission or [], 1):
        value = dict(item or {})
        value.setdefault("id", f"tr-{i}")
        value.setdefault("evidence_refs", [])
        value.setdefault("confidence", None)
        transmission_values.append(_json_safe(value))
    risk_values = []
    for i, item in enumerate(risks or [], 1):
        value = dict(item or {})
        value.setdefault("id", f"risk-{i}")
        value.setdefault("evidence_refs", [])
        risk_values.append(_json_safe(value))
    action_values = []
    for i, item in enumerate(actions or [], 1):
        value = dict(item or {})
        value.setdefault("id", f"act-{i}")
        value.setdefault("evidence_refs", [])
        value.setdefault("condition", "需进一步验证")
        value.setdefault("invalidated_by", "证据失效或日期错位")
        action_values.append(_json_safe(value))
    # Convert chapter evidence labels into real evidence IDs. Legacy callers often
    # supplied prose or a summary object here; preserve it as an auditable fact.
    chapter_values = [_chapter(item, i + 1) for i, item in enumerate(chapter_values)]
    evidence_by_claim = {str(item.get("claim")): item.get("id") for item in evidence_values if item.get("claim")}
    for chapter in chapter_values:
        refs = chapter.get("evidence") if isinstance(chapter.get("evidence"), list) else [chapter.get("evidence")]
        normalized_refs = []
        for ref in refs:
            if isinstance(ref, str) and ref in {item.get("id") for item in evidence_values}:
                normalized_refs.append(ref)
                continue
            claim = str(ref)
            evidence_values.append({"id": f"ev-{len(evidence_values) + 1}", "claim": f"{chapter.get('title')}: {claim}", "value": ref, "unit": "text", "observed_at": str(as_of)[:10] if as_of else None, "source_ref": source_values[0].get("id") if source_values else "src-undisclosed", "quality": "primary" if source_values else "proxy", "note": "章节摘要转为可引用事实"})
            normalized_refs.append(evidence_values[-1]["id"])
        chapter["evidence"] = normalized_refs
    if not source_values and evidence_values:
        source_values = [{"id": "src-undisclosed", "name": "未声明来源", "kind": "unknown", "status": "failed", "observed_at": None, "retrieved_at": None, "locator": None, "note": "证据未声明可追溯来源"}]
        for item in evidence_values:
            item["source_ref"] = "src-undisclosed"
            item["quality"] = "proxy"
    gaps = []
    for item in data_gaps or []:
        if isinstance(item, str):
            item = {"field": item, "reason": "missing", "impact": "相关结论降级", "fallback": "无", "as_of": None}
        item = dict(item or {})
        item.setdefault("field", "未命名缺口")
        item.setdefault("reason", "missing")
        item.setdefault("impact", "相关结论降级")
        item.setdefault("fallback", "无")
        item.setdefault("as_of", None)
        gaps.append(_json_safe(item))
    status_states = [str(x.get("status")) for x in source_values]
    failed = sum(state in {"failed", "error"} for state in status_states)
    degraded = (not source_values) or failed > 0 or bool(gaps) or any(state in {"stale", "fallback", "proxy"} for state in status_states)
    state = "failed" if source_values and failed == len(source_values) else "degraded" if degraded else "ok"
    confidence = summary_value.get("confidence")
    if confidence is None:
        evidence_ratio = len(evidence_values) / max(len(evidence_values) + len(gaps), 1)
        source_ratio = sum(state == "ok" for state in status_states) / max(len(status_states), 1)
        confidence = round(max(0.0, min(1.0, evidence_ratio * source_ratio)), 3)
    subject_value = dict(subject or {"id": "market", "name": title, "kind": "market"})
    subject_value.setdefault("id", subject_value.get("name") or "market")
    subject_value.setdefault("name", title)
    subject_value.setdefault("kind", "market")
    stable_id = report_id or f"{report_type}:{subject_value['id']}:{str(as_of or 'unknown')[:10]}"
    return _json_safe({
        "contract": REPORT_CONTRACT,
        "report_id": stable_id,
        "report_type": report_type,
        "scope": SCOPE,
        "scope_detail": dict(SCOPE_DETAIL),
        "title": title,
        "subject": subject_value,
        "as_of": {
            "period_start": str(period_start)[:10] if period_start else None,
            "period_end": str(period_end)[:10] if period_end else None,
            "data_as_of": str(as_of)[:10] if as_of else None,
            "timezone": "Asia/Shanghai",
        },
        "period": {"start": str(period_start)[:10] if period_start else None, "end": str(period_end)[:10] if period_end else None, "trading_days": trading_days},
        "status": {"state": state, "confidence": confidence, "missing_required": [g["field"] for g in gaps], "source_summary": {"ok": status_states.count("ok"), "failed": failed, "stale": sum(s in {"stale", "fallback", "proxy"} for s in status_states)}},
        "conclusion": summary_value,
        "evidence": evidence_values,
        "transmission": transmission_values,
        "risks": risk_values,
        "actions": action_values,
        "data_gaps": gaps,
        "sources": source_values,
        "chapters": [_chapter(item, i + 1) for i, item in enumerate(chapter_values)],
        "quality": {"evidence_count": len(evidence_values), "source_count": len(source_values), "gap_count": len(gaps), "validated": False},
        "metadata": _json_safe(metadata or {}),
        "detail": _json_safe(detail or {}),
    })


def _parse_contract_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _validate_confidence(value: Any, label: str, problems: list[str]) -> None:
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{label}_confidence_not_numeric")
        return
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        problems.append(f"{label}_confidence_range")


def validate_report_contract(report: dict) -> list[str]:
    """Return deterministic problems; never throw for malformed report input."""
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["report_not_object"]
    required = ("contract", "report_id", "report_type", "scope", "scope_detail", "title", "subject", "as_of", "status", "conclusion", "evidence", "transmission", "risks", "actions", "data_gaps", "sources", "chapters")
    problems.extend(f"missing:{key}" for key in required if key not in report)
    if report.get("contract") != REPORT_CONTRACT:
        problems.append("contract_version")
    if report.get("scope") != SCOPE:
        problems.append("scope")
    if not report.get("report_id"):
        problems.append("report_id")
    if not isinstance(report.get("subject"), dict) or not report.get("subject", {}).get("id"):
        problems.append("subject")
    as_of = report.get("as_of")
    report_date = None
    if not isinstance(as_of, dict):
        problems.append("as_of_not_object")
    else:
        report_date = _parse_contract_date(as_of.get("data_as_of"))
        if not as_of.get("data_as_of") and (report.get("status") or {}).get("state") != "failed":
            problems.append("data_as_of")
    conclusion = report.get("conclusion")
    if not isinstance(conclusion, dict) or not conclusion.get("summary"):
        problems.append("conclusion")
    _validate_confidence((report.get("status") or {}).get("confidence"), "status", problems)
    _validate_confidence((conclusion or {}).get("confidence") if isinstance(conclusion, dict) else None, "conclusion", problems)
    list_buckets = ("evidence", "transmission", "risks", "actions", "data_gaps", "sources", "chapters")
    for bucket in list_buckets:
        if not isinstance(report.get(bucket), list):
            problems.append(f"{bucket}_not_list")
    source_items = report.get("sources") if isinstance(report.get("sources"), list) else []
    evidence_items = report.get("evidence") if isinstance(report.get("evidence"), list) else []
    source_ids = {item.get("id") for item in source_items if isinstance(item, dict) and item.get("id")}
    evidence_ids = {item.get("id") for item in evidence_items if isinstance(item, dict) and item.get("id")}
    allowed_source = {"ok", "failed", "stale", "fallback", "proxy"}
    for index, item in enumerate(source_items, 1):
        if not isinstance(item, dict):
            problems.append(f"source_{index}_object"); continue
        for key in ("id", "name", "kind", "status"):
            if not item.get(key):
                problems.append(f"source_{index}_{key}")
        if item.get("status") not in allowed_source:
            problems.append(f"source_{index}_status")
        observed = _parse_contract_date(item.get("observed_at"))
        if report_date and observed and observed > report_date:
            problems.append(f"source_{index}_future_observed_at")
    for index, item in enumerate(evidence_items, 1):
        if not isinstance(item, dict):
            problems.append(f"evidence_{index}_object"); continue
        for key in ("id", "claim", "observed_at", "source_ref", "quality"):
            if not item.get(key):
                problems.append(f"evidence_{index}_{key}")
        if item.get("source_ref") not in source_ids:
            problems.append(f"evidence_{index}_source_ref")
        observed = _parse_contract_date(item.get("observed_at"))
        if report_date and observed and observed > report_date:
            problems.append(f"evidence_{index}_future_observed_at")
    for index, item in enumerate(report.get("chapters") if isinstance(report.get("chapters"), list) else [], 1):
        if not isinstance(item, dict):
            problems.append(f"chapter_{index}_object"); continue
        for key in ("title", "conclusion", "implication", "next_check"):
            if not item.get(key):
                problems.append(f"chapter_{index}_{key}")
        if not isinstance(item.get("evidence"), list):
            problems.append(f"chapter_{index}_evidence")
        if isinstance(item.get("evidence"), list):
            for ref in item["evidence"]:
                if ref not in evidence_ids:
                    problems.append(f"chapter_{index}_evidence_ref")
    for bucket in ("transmission", "risks", "actions"):
        items = report.get(bucket) if isinstance(report.get(bucket), list) else []
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict):
                problems.append(f"{bucket}_{index}_object"); continue
            if not item.get("id"):
                problems.append(f"{bucket}_{index}_id")
            refs = item.get("evidence_refs")
            if refs is None or not isinstance(refs, list):
                problems.append(f"{bucket}_{index}_evidence_refs")
            else:
                for ref in refs:
                    if ref not in evidence_ids:
                        problems.append(f"{bucket}_{index}_evidence_ref")
            if bucket == "actions":
                for key in ("action", "condition", "invalidated_by"):
                    if not item.get(key):
                        problems.append(f"actions_{index}_{key}")
            _validate_confidence(item.get("confidence"), f"{bucket}_{index}", problems)
    return list(dict.fromkeys(problems))


def contract_to_markdown(report: dict) -> str:
    conclusion = report.get("conclusion") or {}
    period = report.get("as_of") or {}
    lines = [f"## 研究契约摘要：{report.get('title', '未命名报告')}", "", f"- 报告范围：{(report.get('scope_detail') or {}).get('analysis', '分析与决策辅助')}；不连接券商执行。", f"- 数据截至：{period.get('data_as_of') or '未知'}；区间：{period.get('period_start') or '-'} 至 {period.get('period_end') or '-'}。", f"- 结论：{conclusion.get('summary') or '暂无'}（置信度 {conclusion.get('confidence') if conclusion.get('confidence') is not None else '未评估'}）", "", "### 事实证据"]
    for item in report.get("evidence", []):
        value = item.get("value")
        value_text = "" if value is None else f"：{value}{item.get('unit') or ''}"
        lines.append(f"- {item.get('claim', '未命名事实')}{value_text}（{item.get('observed_at') or '日期缺失'}；{item.get('quality') or '未标注'}）")
    lines += ["", "### 传导与含义"]
    for item in report.get("transmission", []):
        lines.append(f"- {item.get('from', '?')} → {item.get('to', '?')}：{item.get('mechanism') or '机制未补充'}（{item.get('direction') or '未知'}）")
    lines += ["", "### 风险与行动"]
    for item in report.get("risks", []):
        lines.append(f"- 风险：{item.get('description') or '?'}；触发：{item.get('trigger') or '未定义'}；影响：{item.get('impact') or '未定义'}")
    for item in report.get("actions", []):
        lines.append(f"- 动作：{item.get('action') or 'observe'} {item.get('target') or ''}；条件：{item.get('condition') or '未定义'}；失效：{item.get('invalidated_by') or '未定义'}")
    lines += ["", "### 数据缺口"]
    lines.extend(f"- {item.get('field')}: {item.get('impact')}；处理：{item.get('fallback')}" for item in report.get("data_gaps", []))
    if not report.get("data_gaps"):
        lines.append("- 未发现已登记数据缺口。")
    lines += ["", "### 来源状态"]
    lines.extend(f"- {item.get('name')}: {item.get('status')}；数据日 {item.get('observed_at') or '未知'}；{item.get('note') or ''}" for item in report.get("sources", []))
    return "\n".join(lines) + "\n"


def validate_and_write_contract(path: str | Path, report: dict) -> tuple[Path, Path]:
    """Validate first, then write the JSON contract and Markdown projection."""
    problems = validate_report_contract(report)
    if problems:
        raise ValueError("invalid research report contract: " + ", ".join(problems[:12]))
    report = _json_safe(report)
    report.setdefault("quality", {})["validated"] = True
    report["quality"]["validation_errors"] = []
    target = Path(path)
    json_path = target.with_suffix(target.suffix + ".research.json")
    md_path = target.with_suffix(target.suffix + ".research.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(contract_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_contract_sidecar(path: str | Path, report: dict) -> tuple[Path, Path]:
    """Backward-compatible validated sidecar writer."""
    return validate_and_write_contract(path, report)


def enrich_legacy_report(report_type: str, title: str, as_of: Any, summary: str, sections: dict[str, Any], **kwargs) -> dict:
    """Adapt an existing report without forcing a lossy Markdown reparse."""
    chapters = []
    for name, content in sections.items():
        chapters.append({"title": name, "conclusion": content if isinstance(content, str) else str(content), "evidence": [name], "implication": "见原报告主体的资产与策略含义", "next_check": "按原报告观察清单复核"})
    return make_report_contract(report_type, title, as_of, summary=summary, chapters=chapters, **kwargs)


def write_legacy_sidecar(path: str | Path, report_type: str, title: str, as_of: Any,
                         summary: str, sections: dict[str, Any], **kwargs) -> tuple[Path, Path]:
    """Create the same auditable sidecar for an existing Markdown/HTML report."""
    report = enrich_legacy_report(report_type, title, as_of, summary, sections, **kwargs)
    return write_contract_sidecar(path, report)


def _first_date(value: Any) -> str | None:
    """Find an explicitly supplied observation date without using retrieval time."""
    if isinstance(value, dict):
        for key in ("data_as_of", "as_of", "observed_at", "source_date", "date", "trade_date"):
            candidate = value.get(key)
            if candidate and str(candidate)[:4].isdigit():
                return str(candidate)[:10]
        for nested in value.values():
            found = _first_date(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _first_date(nested)
            if found:
                return found
    elif value and str(value)[:4].isdigit() and len(str(value)[:10]) >= 10:
        return str(value)[:10]
    return None


def contract_from_daily_review(review: dict) -> dict:
    """Adapt a persisted daily review without collapsing its analytical depth."""
    data_as_of = str(review.get("date") or _first_date(review) or "")[:10] or None
    blocks = review.get("blocks") or {}
    sources: list[dict] = []
    evidence: list[dict] = []
    gaps: list[dict] = []
    refs_by_block: dict[str, list[str]] = {}

    def _lag_days(observed: Any) -> int | None:
        try:
            if not observed or not data_as_of:
                return None
            return (datetime.strptime(data_as_of, "%Y-%m-%d") - datetime.strptime(str(observed)[:10], "%Y-%m-%d")).days
        except (TypeError, ValueError):
            return None

    def _facts(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
        """Extract decision facts, not just top-level date/confidence metadata."""
        if depth > 20:
            return []
        out: list[tuple[str, Any]] = []
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if key in {"date", "generated_at", "ts", "confidence"}:
                    continue
                if isinstance(item, (str, int, float, bool)) and item is not None:
                    out.append((path, item))
                elif isinstance(item, dict):
                    out.extend(_facts(item, path, depth + 1))
                elif isinstance(item, list) and item:
                    out.append((path + ".count", len(item)))
                    named = []
                    for row in item:
                        if isinstance(row, dict):
                            label = row.get("name") or row.get("target") or row.get("concept") or row.get("symbol") or row.get("code")
                            if label:
                                named.append(str(label))
                        elif isinstance(row, (str, int, float)):
                            named.append(str(row))
                    if named:
                        out.append((path + ".top", "、".join(named)))
        elif isinstance(value, list):
            out.append((prefix + ".count" if prefix else "count", len(value)))
        return out

    for name, raw in blocks.items():
        block = raw if isinstance(raw, dict) else {}
        failed = bool(block.get("error"))
        source_id = f"src-block-{name}"
        observed = _first_date(block) or data_as_of
        lag = _lag_days(observed)
        stale = lag is not None and lag > 5
        source_state = "failed" if failed else "stale" if stale else "ok"
        sources.append({
            "id": source_id,
            "name": block.get("source") or name,
            "kind": "model" if name in {"engines", "engine_fusion", "prediction_loop", "ml_verdict"} else "warehouse",
            "status": source_state,
            "observed_at": observed,
            "retrieved_at": block.get("ts") or review.get("generated_at"),
            "locator": f"generated/review_{data_as_of}.json" if data_as_of else None,
            "note": block.get("error") or (f"较报告日滞后 {lag} 天" if stale else "复盘链结构化信号块"),
        })
        if failed or stale:
            gaps.append({
                "field": name,
                "reason": "source_failed" if failed else "stale",
                "impact": "该信号不进入强结论" if failed else f"来源较报告日滞后 {lag} 天，仅作历史参考",
                "fallback": "保留其他可用信号，不补零",
                "as_of": observed,
            })
            if failed:
                continue
        value = block.get("value")
        facts = _facts(value)
        refs_by_block[name] = []
        for key, item in facts:
            evidence_id = f"ev-{len(evidence) + 1}"
            refs_by_block[name].append(evidence_id)
            evidence.append({
                "id": evidence_id,
                "claim": f"{name}.{key}",
                "value": item,
                "unit": "text" if isinstance(item, str) else "value",
                "observed_at": observed,
                "source_ref": source_id,
                "quality": "historical" if stale else "primary",
                "note": f"来源滞后 {lag} 天" if stale else "",
            })
        if not facts and value not in (None, {}, []):
            evidence_id = f"ev-{len(evidence) + 1}"
            refs_by_block[name] = [evidence_id]
            evidence.append({"id": evidence_id, "claim": f"{name} 已生成结构化信号", "value": str(value), "unit": "text", "observed_at": observed, "source_ref": source_id, "quality": "historical" if stale else "primary"})

    health = review.get("health") or {}
    for item in health.get("degraded", []) or health.get("degraded_datasets", []) or []:
        gaps.append({"field": str(item), "reason": "stale", "impact": "日报结论降级", "fallback": "不对缺失数据作方向性替代", "as_of": data_as_of})

    decision = review.get("decision") or {}
    battle = review.get("battle_map") or {}
    stance = decision.get("stance") or decision.get("定调") or battle.get("action_card") or battle.get("recommended") or "observe"
    summary = decision.get("summary") or decision.get("reason") or decision.get("定调") or battle.get("recommended") or "每日复盘按市场、资金、因子、回测和风险门控形成观察结论。"

    chapter_specs = [
        ("市场状态与宽度", ["market_temperature", "intraday_bias"], "市场温度、情绪、宽度与盘中基调共同决定风险预算。", "复核下一交易时段宽度、温度和指数方向"),
        ("主线、龙头与产业链", ["strong_direction", "leader_sentiment", "lhb"], "主线结构、龙头梯队、席位与产业链需要同日交叉确认。", "复核主线扩散、龙头晋级和产业链联动"),
        ("候选池与机会", ["stock_picks", "opportunity"], "候选池按可执行与观察分层，不以排名替代交易门控。", "检查候选的触发、失效和流动性条件"),
        ("因子分析与样本外质量", ["factor_signal", "engine_fusion"], "因子方向、强弱和引擎共识必须结合样本期与来源日期。", "更新 IC/OOS、方向稳定性和因子权重"),
        ("回测与预测验证", ["backtest_check", "prediction_loop", "ml_verdict"], "回测成本、预测命中与校准决定结论只能研究还是可进入观察执行。", "验证新样本、成本、回撤和预测实际结果"),
        ("技术、海外与宏观", ["technical", "overseas", "macro_veto", "data_base"], "技术结构、海外风险和宏观否决共同限制方向与仓位。", "检查技术破位、海外冲击和宏观否决条件"),
        ("引擎融合与模型意见", ["engines", "model_verdict"], "引擎产出只有进入融合并有真实证据时才影响最终判断。", "复核引擎覆盖、共识和冲突"),
        ("风险门控与行动卡", ["freshness", "battle_map_card", "intraday_rows"], "数据新鲜度、作战卡和盘中记录共同约束执行。", "检查仓位、竞价锚点、止损和失效条件"),
    ]
    chapters = []
    for title, keys, implication, next_check in chapter_specs:
        refs = [ref for key in keys for ref in refs_by_block.get(key, [])]
        present = [key for key in keys if key in blocks]
        if not present:
            continue
        failed_names = [key for key in present if (blocks.get(key) or {}).get("error")]
        conclusion = f"已接入：{'、'.join(present)}"
        if failed_names:
            conclusion += f"；降级：{'、'.join(failed_names)}"
        chapters.append({"title": title, "conclusion": conclusion, "evidence": refs, "implication": implication, "next_check": next_check})

    all_refs = [item["id"] for item in evidence]
    risk_refs = [ref for key in ("freshness", "macro_veto", "battle_map_card") for ref in refs_by_block.get(key, [])] or all_refs
    risks = [{"description": str(item), "trigger": "相关数据继续恶化或未按时更新", "impact": "降低风险预算并暂停新增暴露", "severity": "medium", "evidence_refs": risk_refs} for item in (battle.get("degraded", []) or health.get("degraded", []))]
    for gap in gaps:
        risks.append({"description": f"{gap['field']}：{gap['impact']}", "trigger": gap["reason"], "impact": "相关章节降级为观察", "severity": "medium", "evidence_refs": risk_refs})
    if not risks:
        risks = [{"description": "复盘链仍需次日价格和资金确认", "trigger": "触发条件未满足", "impact": "仅保留观察，不将复盘结论当作执行信号", "severity": "low", "evidence_refs": all_refs}]

    position = battle.get("position_range") or (review.get("battle_map") or {}).get("position_range") or "按风险预算"
    actions = [{"action": "validate", "target": "次日市场状态、主线资金与候选触发", "condition": "宽度、趋势、主线或资金至少两类同向确认", "invalidated_by": "数据日期错位、来源失败、技术破位或风险门控触发", "horizon": "next_session", "risk_limit": str(position), "evidence_refs": risk_refs}]

    transmissions = [
        {"from": "市场温度/宽度/宏观", "to": "风险预算", "mechanism": "情绪、宽度和宏观否决决定仓位上限", "direction": "mixed", "evidence_refs": [ref for key in ("market_temperature", "intraday_bias", "macro_veto") for ref in refs_by_block.get(key, [])][:10], "confidence": None},
        {"from": "主线/龙头/资金", "to": "候选池", "mechanism": "主线结构、梯队和资金确认决定候选是否可比", "direction": "mixed", "evidence_refs": [ref for key in ("strong_direction", "leader_sentiment", "lhb", "stock_picks") for ref in refs_by_block.get(key, [])][:10], "confidence": None},
        {"from": "因子/回测/预测", "to": "结论置信度", "mechanism": "样本外质量、成本后回测和预测验证共同约束模型权重", "direction": "mixed", "evidence_refs": [ref for key in ("factor_signal", "backtest_check", "prediction_loop", "engine_fusion") for ref in refs_by_block.get(key, [])][:10], "confidence": None},
    ]
    transmissions = [item for item in transmissions if item["evidence_refs"]]

    return make_report_contract(
        "daily_review", f"每日综合复盘 {data_as_of or ''}".strip(), data_as_of,
        subject={"id": "ashare-market", "name": "中国市场", "kind": "market"},
        report_id=f"daily_review:ashare-market:{data_as_of or 'unknown'}",
        summary={"stance": str(stance), "summary": str(summary), "horizon": "intraday", "confidence": None},
        evidence=evidence, transmission=transmissions, risks=risks, actions=actions,
        data_gaps=gaps, sources=sources, chapters=chapters,
        detail={"review": _json_safe(review), "blocks": _json_safe(blocks), "battle_map": _json_safe(battle)},
        metadata={"run_id": review.get("run_id"), "legacy_fields": ["blocks", "battle_map", "pools", "decision"], "block_count": len(blocks), "mixed_source_dates": len({source.get("observed_at") for source in sources if source.get("observed_at")}) > 1},
    )


def contract_from_stock_overview(payload: dict, *, report_type: str = "stock") -> dict:
    """Adapt /api/stock_overview output without changing its legacy response."""
    symbol = str(payload.get("symbol") or payload.get("code") or "unknown").zfill(6)
    name = payload.get("name") or symbol
    source_status = payload.get("source_status") or {}
    sources = []
    for key, raw in source_status.items():
        item = dict(raw or {})
        sources.append({"id": f"src-{key}", "name": item.get("label") or key, "kind": "market", "status": item.get("status") or "missing", "observed_at": item.get("as_of") or payload.get("as_of"), "locator": item.get("source"), "note": item.get("message") or ""})
    declared = {item["id"] for item in sources}
    profile = payload.get("profile") or {}
    quote = profile.get("quote") or {}
    profile_has_quote = bool(quote)
    # 审计修复: 个股证据固定引用 src-profile/src-stock_flow；当调用方的
    # source_status 缺少对应键时必须补声明，否则 validate 报 evidence_source_ref
    # 且 sidecar 写入失败。声明使用 payload 真实状态，不伪造可用。
    _profile_status = "available" if profile_has_quote else "missing"
    if "src-profile" not in declared:
        sources.append({"id": "src-profile", "name": "个股画像", "kind": "market",
                        "status": _profile_status, "observed_at": payload.get("as_of"),
                        "locator": "data_warehouse/kline", "note": "由个股载荷声明"})
    for key, label, value in (("stock_flow", "个股主力资金", payload.get("stock_flow")), ("flow", "行业资金代理", payload.get("flow")), ("commodity", "行业商品/期货", payload.get("commodity")), ("research", "研报提取", payload.get("research")), ("news", "个股新闻", payload.get("stock_news"))):
        if value and f"src-{key}" not in declared:
            raw_status = value.get("status") if isinstance(value, dict) else None
            sources.append({"id": f"src-{key}", "name": label, "kind": "market", "status": raw_status or "ok", "observed_at": (value.get("as_of") if isinstance(value, dict) else None) or payload.get("as_of"), "locator": None, "note": "由个股深度载荷声明"})
    evidence = []
    for key, label, value, unit, observed, source in [("quote", "最新价格", quote.get("close"), "CNY", quote.get("date"), "profile"), ("quote_change", "最新涨跌幅", quote.get("pct_chg"), "pct", quote.get("date"), "profile"), ("profile_score", "画像总分", (profile.get("score") or {}).get("total"), "score", profile.get("data_as_of") or payload.get("as_of"), "profile")]:
        if value is not None:
            evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": label, "value": value, "unit": unit, "observed_at": observed, "source_ref": f"src-{source}", "quality": "primary"})
    flow = payload.get("stock_flow") or {}
    if flow.get("rows"):
        row = flow["rows"][-1]
        evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": "个股主力净流入", "value": row.get("main_net_yi"), "unit": "CNY_100m", "observed_at": row.get("date") or flow.get("as_of") or payload.get("as_of"), "source_ref": "src-stock_flow", "quality": "primary"})
    commodity = payload.get("commodity") or {}
    for item in (commodity.get("items") or [])[:6]:
        if item.get("return_20d_pct") is not None:
            evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": f"{item.get('name') or item.get('tag')}近20日变化", "value": item.get("return_20d_pct"), "unit": "pct", "observed_at": item.get("as_of") or payload.get("as_of"), "source_ref": "src-commodity", "quality": "primary"})
    gaps = []
    for key, raw in source_status.items():
        state = str((raw or {}).get("status") or "missing")
        if state not in {"available", "partial", "ok"}:
            gaps.append({"field": key, "reason": "stale" if state == "stale" else "missing", "impact": (raw or {}).get("message") or "该来源不进入强结论", "fallback": "保留为观察证据，不补零", "as_of": (raw or {}).get("as_of")})
    gate = payload.get("decision_gate") or {}
    summary = payload.get("decision_explanation") or {}
    stance = summary.get("stance") or gate.get("label") or "observe"
    refs = [item["id"] for item in evidence]
    subject_kind = "stock" if report_type == "stock" else "industry" if report_type == "industry" else "commodity"
    return make_report_contract(report_type, f"{name}{'个股' if subject_kind == 'stock' else '行业' if subject_kind == 'industry' else '商品'}研究报告", payload.get("as_of") or _first_date(payload), subject={"id": symbol if subject_kind == "stock" else name, "name": name, "kind": subject_kind}, report_id=f"{report_type}:{symbol}:{payload.get('as_of') or 'unknown'}", summary={"stance": str(stance), "summary": "; ".join(summary.get("supporting_factors") or []) or str(summary.get("method_note") or "仅作研究观察，等待条件确认"), "horizon": "days" if subject_kind == "stock" else "weeks", "confidence": None}, evidence=evidence, transmission=[{"from": "价格/资金/商品", "to": f"{name}盈利与估值", "mechanism": "价格趋势与资金证据先确认，商品仅作为行业暴露与成本传导，不直接等同买点", "direction": "mixed", "evidence_refs": refs[:6], "confidence": None}], risks=[{"description": x, "trigger": "来源陈旧、缺失或交易门控阻断", "impact": "仅保留观察", "severity": "high" if "风险" in str(x) else "medium", "evidence_refs": refs[:3]} for x in (gate.get("blockers") or gate.get("warnings") or ["研究证据不能替代交易门控"])[:6]], actions=[{"action": "observe" if not gate.get("trade_allowed") else "validate", "target": name, "condition": "价格/趋势与资金或基本面证据同日确认", "invalidated_by": "数据日期错位、来源失败或止损条件触发", "horizon": "next_review", "risk_limit": "按纸面交易风控执行", "evidence_refs": refs[:5]}], data_gaps=gaps, sources=sources, chapters=[{"title": "结论", "conclusion": str(stance), "evidence": refs[:5], "implication": "结论必须回溯到价格、趋势、资金或商品证据", "next_check": "复核最新数据日与门控状态"}, {"title": "证据与传导", "conclusion": "个股级资金与行业商品证据分层展示", "evidence": refs[:8], "implication": "商品变化不直接转化为股票买点", "next_check": "检查成本、需求和估值是否同步"}], metadata={"symbol": symbol, "legacy_data_contract": payload.get("data_contract")}, detail={"stock_flow": payload.get("stock_flow") or {}, "flow": payload.get("flow") or {}, "research": payload.get("research") or [], "stock_news": payload.get("stock_news") or {}, "score_hierarchy": payload.get("score_hierarchy") or {}, "ic_weight_explanation": payload.get("ic_weight_explanation") or {}, "commodity": payload.get("commodity") or {}, "decision": payload.get("decision") or {}, "market_context": payload.get("market_context") or {}, "evidence_summary": payload.get("evidence_summary") or {}})


def contract_from_industry_snapshot(name: str, snapshot: dict, *, as_of: Any = None) -> dict:
    """Build an industry report from the real weekly industry observation."""
    observed = as_of or snapshot.get("observed_date") or snapshot.get("as_of")
    source = {"id": "src-industry-weekly", "name": "申万一级指数历史行情", "kind": "warehouse", "status": "ok" if snapshot.get("weekly_return") is not None else "failed", "observed_at": observed, "locator": "data_warehouse/industry/sw_first_hist.parquet", "note": snapshot.get("note") or "行业表现与成交额，不等同主力资金"}
    evidence = []
    for claim, key, unit in (("行业周度收益", "weekly_return", "pct"), ("行业成交额", "amount_sum", "CNY"), ("行业观测天数", "observations", "count")):
        if snapshot.get(key) is not None:
            evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": f"{name}{claim}", "value": snapshot.get(key), "unit": unit, "observed_at": observed, "source_ref": "src-industry-weekly", "quality": "primary"})
    gaps = [] if evidence else [{"field": "industry_performance", "reason": "missing", "impact": "不能形成行业方向结论", "fallback": "仅保留观察", "as_of": observed}]
    refs = [f"ev-{i + 1}" for i in range(len(evidence))]
    return make_report_contract("industry", f"{name}行业研究报告", observed, subject={"id": str(snapshot.get("code") or name), "name": name, "kind": "industry"}, report_id=f"industry:{snapshot.get('code') or name}:{observed or 'unknown'}", summary={"stance": "observe", "summary": "行业表现与成交额用于轮动观察，需结合供需、盈利和资金确认。", "horizon": "weeks", "confidence": None}, evidence=evidence, transmission=[{"from": "行业表现/成交额", "to": f"{name}盈利与龙头", "mechanism": "行业表现先通过需求、成本和盈利预期传导，再评估龙头相对强弱", "direction": "mixed", "evidence_refs": refs, "confidence": None}], risks=[{"description": "行业行情不等同于主力资金或基本面改善", "trigger": "成交额上升但盈利/供需未确认", "impact": "只保留轮动观察，不形成单因子买点", "severity": "medium", "evidence_refs": refs[:2]}], actions=[{"action": "validate", "target": name, "condition": "行业表现、成交额与供需/盈利或资金至少两类同日确认", "invalidated_by": "日期错位、来源失败或成交额代理被误读", "horizon": "next_week", "evidence_refs": refs}], data_gaps=gaps, sources=[source], chapters=[{"title": "行业表现", "conclusion": "按真实行业指数历史数据观察轮动", "evidence": refs, "implication": "不把行业指数成交额称为主力资金", "next_check": "补充供需、盈利与资金证据"}], metadata={"classification": "申万一级指数", "data_contract": "market_weekly.v1"}, detail={"snapshot": _json_safe(snapshot)})


def contract_from_market_history(payload: dict, *, report_type: str = "commodity") -> dict:
    """Adapt /api/market_history output for commodity and industry readers."""
    asset = str(payload.get("asset") or payload.get("symbol") or "unknown")
    name = str(payload.get("name") or asset)
    observed = payload.get("as_of") or _first_date(payload)
    source_status = "ok" if payload.get("ok") and payload.get("rows") else str(payload.get("status") or "missing")
    source = {"id": "src-history", "name": payload.get("source") or "market history", "kind": "market", "status": source_status, "observed_at": observed, "locator": payload.get("source"), "note": payload.get("error") or "历史序列"}
    rows = payload.get("rows") or []
    latest = rows[-1] if rows else {}
    evidence = []
    for key, label, unit in (("close", "最新价格", "CNY"), ("ret", "最新日变化", "pct"), ("change_yi", "两融日变化", "CNY_100m")):
        if latest.get(key) is not None:
            evidence.append({"id": f"ev-{len(evidence) + 1}", "claim": label, "value": latest.get(key), "unit": unit, "observed_at": latest.get("date") or observed, "source_ref": "src-history", "quality": "primary"})
    gaps = []
    if not evidence:
        gaps.append({"field": "history", "reason": "source_failed", "impact": "没有价格或供需序列，不能形成方向性结论", "fallback": "仅展示来源失败状态", "as_of": observed})
    refs = [f"ev-{i + 1}" for i in range(len(evidence))]
    return make_report_contract(report_type, name + ("行业报告" if report_type == "industry" else "商品报告"), observed, subject={"id": asset, "name": name, "kind": report_type}, report_id=f"{report_type}:{asset}:{observed or 'unknown'}", summary={"stance": "observe", "summary": "价格或历史序列仅作为事实输入，需结合供需、资金和金融条件确认。", "horizon": "weeks", "confidence": None}, evidence=evidence, transmission=[{"from": "价格/库存/供需", "to": name, "mechanism": "商品价格先通过成本、需求和金融条件传导，再评估行业或标的影响", "direction": "mixed", "evidence_refs": refs, "confidence": None}], risks=[{"description": "历史序列或实时数据覆盖不足", "trigger": "来源失败、数据陈旧或日期错位", "impact": "不把单因子变化转成买点", "severity": "medium", "evidence_refs": refs[:1]}], actions=[{"action": "validate", "target": name, "condition": "价格与供需/库存或金融条件同日确认", "invalidated_by": "关键来源失败或证据日期错位", "horizon": "next_review", "evidence_refs": refs}], data_gaps=gaps, sources=[source], chapters=[{"title": "价格与数据状态", "conclusion": "历史序列按真实观测日展示", "evidence": refs, "implication": "单一价格因子不能直接推导股票买点", "next_check": "补充供需、库存和金融条件"}], metadata={"asset": asset, "data_contract": payload.get("data_contract")}, detail={"history": _json_safe(payload)})


if __name__ == "__main__":
    sample = make_report_contract("weekly", "自测报告", "2026-08-28", summary="观察", evidence=[{"claim": "成交量", "value": 100, "unit": "亿", "observed_at": "2026-08-28", "source_ref": "src-1"}], sources=[{"id": "src-1", "name": "本地市场数据", "status": "ok", "observed_at": "2026-08-28"}], chapters=[{"title": "核心", "conclusion": "观察", "evidence": ["成交量"], "implication": "等待确认", "next_check": "下周复核"}])
    assert not validate_report_contract(sample), validate_report_contract(sample)
    assert "事实证据" in contract_to_markdown(sample)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
