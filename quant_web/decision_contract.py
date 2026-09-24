"""Stock overview decision contract: freshness, evidence quality and trade gate."""
from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any


SCORE_MAX = 100.0
SCORE_MIN = 0.0

SOURCE_POLICY = {
    "profile": {"label": "个股画像", "limit_days": 1, "weight": 0.25},
    "decision": {"label": "决策快照", "limit_days": 1, "weight": 0.10},
    "stock_flow": {"label": "个股主力资金", "limit_days": 3, "weight": 0.20},
    "flow": {"label": "行业资金代理", "limit_days": 1, "weight": 0.10},
    "commodity": {"label": "行业商品/期货", "limit_days": 5, "weight": 0.18},
    "research": {"label": "研报提取", "limit_days": 90, "weight": 0.10},
    "news": {"label": "个股新闻", "limit_days": 7, "weight": 0.05},
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _age(as_of: Any, reference: date) -> int | None:
    parsed = _date(as_of)
    return max(0, (reference - parsed).days) if parsed else None


def normalize_source_status(source_status: dict[str, dict], reference_date: Any = None) -> dict[str, dict]:
    """Add a common status shape without changing the original source semantics."""
    reference = _date(reference_date) or datetime.now().date()
    normalized: dict[str, dict] = {}
    for key, policy in SOURCE_POLICY.items():
        raw = dict(source_status.get(key) or {})
        raw.setdefault("label", policy["label"])
        status = str(raw.get("status") or "missing")
        raw["status"] = status
        raw["freshness_limit_days"] = policy["limit_days"]
        raw["freshness_days"] = _age(raw.get("as_of"), reference)
        if raw.get("freshness_days") is not None and raw["freshness_days"] > policy["limit_days"]:
            if status == "available":
                raw["status"] = "stale"
            raw.setdefault("message", f"数据已陈旧 {raw['freshness_days']} 天，阈值 {policy['limit_days']} 天")
        raw.setdefault("sample_size", raw.get("count") or raw.get("rows") or 0)
        raw.setdefault("source", raw.get("source") or "未声明")
        raw["is_decision_eligible"] = raw["status"] in {"available", "partial"}
        normalized[key] = raw
    return normalized


def _bounded_score(value: Any) -> float | None:
    """Accept only the public score contract; invalid values are unavailable."""
    number = _finite(value)
    if number is None or number < SCORE_MIN or number > SCORE_MAX:
        return None
    return number


def _quality_score(status: dict, *, optional: bool = False) -> float | None:
    state = status.get("status")
    if optional and state in {"unmapped", "not_applicable"}:
        return None
    if state == "available":
        return 100.0
    if state == "partial":
        return 65.0
    if state == "stale":
        return 25.0
    if state == "mapping_missing":
        return 20.0
    if state in {"missing", "error", "unattributed", "unmapped", "provider_empty", "mapping_missing"}:
        return 0.0
    return 15.0


def build_evidence_quality(source_status: dict[str, dict]) -> dict:
    parts = []
    weighted = 0.0
    weight_sum = 0.0
    for key, policy in SOURCE_POLICY.items():
        status = source_status.get(key) or {}
        score = _quality_score(status, optional=key == "commodity")
        if score is None:
            parts.append({"key": key, "status": status.get("status"), "score": None, "weight": 0.0, "optional": True})
            continue
        weight = float(policy["weight"])
        weighted += score * weight
        weight_sum += weight
        parts.append({"key": key, "status": status.get("status"), "score": round(score, 1), "weight": weight, "optional": False})
    return {
        "score": round(weighted / weight_sum, 1) if weight_sum else None,
        "parts": parts,
        "method": "source_status_quality_weighted_by_freshness_policy",
    }


def build_decision_gate(profile: dict, decision: dict, source_status: dict[str, dict],
                        *, reference_date: Any = None) -> dict:
    statuses = normalize_source_status(source_status, reference_date)
    blockers: list[str] = []
    warnings: list[str] = []
    required: list[str] = []
    failed: list[str] = []
    stale_sources: list[str] = []

    profile_status = statuses["profile"]
    if not profile.get("quote") or profile_status["status"] in {"missing", "error", "stale"}:
        blockers.append(profile_status.get("message") or "行情画像不可用")
        required.append("quote")
        failed.append("quote")

    decision_status = statuses["decision"]
    if decision_status.get("freshness_days") is not None and decision_status["freshness_days"] > 1:
        stale_sources.append("decision")
        warnings.append(f"决策快照距行情 {decision_status['freshness_days']} 天")
    if decision_status["status"] in {"missing", "error", "stale"}:
        warnings.append(decision_status.get("message") or "当前股票不在最新候选快照中")
        failed.append("decision")

    for key in ("stock_flow", "flow"):
        status = statuses[key]
        if status["status"] == "stale":
            stale_sources.append(key)
            message = status.get("message") or f"{status['label']}陈旧"
            if key == "stock_flow":
                blockers.append(message)
            else:
                warnings.append(message)
            failed.append(key)
        elif status["status"] in {"missing", "error", "mapping_missing", "unattributed", "provider_empty"}:
            message = status.get("message") or f"{status['label']}不可用"
            if key == "stock_flow":
                blockers.append(message)
                required.append("stock_flow")
            else:
                warnings.append(message)
            failed.append(key)

    commodity = profile.get("commodity") or {}
    commodity_status = statuses["commodity"]
    if commodity_status["status"] not in {"unmapped", "not_applicable"}:
        if commodity_status["status"] == "stale":
            stale_sources.append("commodity")
            warnings.append(commodity_status.get("message") or "行业商品数据陈旧")
            failed.append("commodity")
        elif commodity_status["status"] in {"missing", "error"} or float(commodity.get("coverage") or 0) < 0.5:
            warnings.append(commodity_status.get("message") or "行业商品暴露覆盖不足")
            failed.append("commodity")

    for key in ("research", "news"):
        status = statuses[key]
        if status["status"] == "stale":
            stale_sources.append(key)
            warnings.append(status.get("message") or f"{status['label']}陈旧")
        elif status["status"] in {"missing", "error", "unattributed"}:
            warnings.append(status.get("message") or f"{status['label']}不可用")

    profile_score = profile.get("score") or {}
    if profile_score.get("total") is None:
        blockers.append("没有足够的有效画像维度形成总分")
        failed.append("profile_score")
    else:
        coverage = profile_score.get("coverage") or {}
        coverage_ratio = _finite(coverage.get("coverage_ratio"))
        if coverage_ratio is None and coverage.get("coverage_ratio") not in (None, ""):
            warnings.append("画像覆盖率字段不可解析，总分仅作观察")
            failed.append("profile_coverage")
        elif coverage_ratio is not None and coverage_ratio < 0.60:
            warnings.append(f"画像有效权重覆盖仅 {coverage_ratio:.0%}，总分仅作观察")
            failed.append("profile_coverage")

    explicit_allowed = decision.get("trade_allowed")
    decision_blockers = list(decision.get("short_term_blockers") or decision.get("blockers") or [])
    if explicit_allowed is False and decision_blockers:
        blockers.extend(str(item) for item in decision_blockers[:5])

    if blockers and any(x in failed for x in ("quote", "profile_score")):
        state = "insufficient_data"
        label = "数据不足"
    elif blockers:
        state = "blocked"
        label = "交易阻断"
    elif stale_sources:
        state = "stale"
        label = "证据陈旧"
    elif explicit_allowed is True and not failed:
        state = "tradable"
        label = "可进入候选"
    else:
        state = "review_only"
        label = "仅复核"

    primary = (blockers or warnings or ["数据链完整，可继续按策略门控复核"])[0]
    return {
        "status": state,
        "label": label,
        "primary_reason": primary,
        "trade_allowed": state == "tradable",
        "min_trade_level": "actionable" if state == "tradable" else "watch",
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "required_evidence": list(dict.fromkeys(required)),
        "failed_evidence": list(dict.fromkeys(failed)),
        "stale_sources": list(dict.fromkeys(stale_sources)),
        "source_status": statuses,
        "computed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def build_score_hierarchy(profile: dict, decision: dict, gate: dict) -> dict:
    """Build a bounded score using the evidence required by the active score.

    A missing source is not globally "critical". It becomes required only when
    the active decision layer says it is part of that comparison (currently the
    short-term layer requires same-day industry-flow evidence). This keeps
    profile-only research useful while preventing a mixed short-term ranking.
    """
    evidence = build_evidence_quality(gate.get("source_status") or {})
    failed = set(gate.get("failed_evidence") or [])
    gate_status = str(gate.get("status") or "review_only")
    source_status = gate.get("source_status") or {}
    requirements: list[str] = []
    profile_total = (profile.get("score") or {}).get("total")
    if profile_total is not None or profile.get("score"):
        requirements.extend(["quote", "profile_score"])
    if decision.get("short_term_score") is not None or decision.get("score_breakdown"):
        requirements.append("flow")
        # A score breakdown containing the structure factor proves that
        # mainline evidence is part of this comparison, even if no label match
        # survived serialization.
        if decision.get("mainline_match") or "概念/题材结构" in (decision.get("score_breakdown") or {}):
            requirements.append("mainline_structure")
    if _bounded_score(profile_total) is None and "profile_score" in requirements:
        failed.add("profile_score")
    if not profile.get("quote") and "quote" in requirements:
        failed.add("quote")
    if "mainline_structure" in requirements and not decision.get("mainline_match"):
        failed.add("mainline_structure")
    for key in ("flow",):
        state = str((source_status.get(key) or {}).get("status") or "missing")
        if state not in {"available", "partial"}:
            failed.add(key)
    comparison_failed = sorted(set(requirements) & failed)
    comparable = not comparison_failed and gate_status not in {"insufficient_data", "blocked"}
    candidates = [
        ("profile_score", "基础画像", (profile.get("score") or {}).get("total"), 0.45,
         "趋势/估值/财务/流动性/事件/商品规则画像"),
        ("short_term_score", "短线决策分", decision.get("short_term_score"), 0.25,
         "盘口、主线、资金和梯队加分制"),
        ("decision_score", "决策证据分", decision.get("decision_score"), 0.15,
         "候选证据综合分，和基础画像不是同一口径"),
        ("evidence_quality", "证据质量", evidence.get("score"), 0.15,
         evidence.get("method")),
    ]
    layers = []
    weighted = 0.0
    weight_sum = 0.0
    for key, label, value, weight, note in candidates:
        number = _bounded_score(value)
        used = comparable and number is not None
        reason = None if used else (
            f"当前评分所需证据缺失，结果不可比：{'、'.join(comparison_failed)}" if comparison_failed
            else "门控未通过，结果仅作观察" if not comparable
            else "输入分数不在 0-100 范围或不可解析"
        )
        layers.append({"key": key, "label": label, "score": round(number, 1) if number is not None else None,
                       "weight": weight if used else 0.0, "eligible": used,
                       "comparable": comparable, "reason": reason or note})
        if used:
            weighted += number * weight
            weight_sum += weight
    final_score = round(min(SCORE_MAX, max(SCORE_MIN, weighted / weight_sum)), 1) if weight_sum else None
    return {
        "final_score": final_score,
        "final_level": ("强" if final_score is not None and final_score >= 70 else
                         "偏强" if final_score is not None and final_score >= 60 else
                         "中性" if final_score is not None and final_score >= 45 else
                         "偏弱" if final_score is not None else "不可用"),
        "trade_bucket": gate_status,
        "comparable": comparable,
        "not_comparable_reasons": comparison_failed or ([] if comparable else [gate_status]),
        "required_for_comparison": requirements,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "layers": layers,
        "evidence_quality": evidence,
        "score_status": "comparable" if comparable else ("unavailable" if final_score is None else "not_comparable"),
        "method": "bounded_0_100_available_layers_only_when_active_comparison_evidence_is_valid",
        "missing_policy": "仅当某证据被当前评分层声明为比较所需时才阻断；非比较证据缺失只在证据状态中展示，不补零、不冒充通过。", 
    }
