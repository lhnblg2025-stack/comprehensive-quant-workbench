"""IC/OOS calibration evidence for the stock decision workspace.

The IC report is a calibration input, not a trading signal by itself. This module
keeps the business-rule prior and empirical factor evidence separate, applies
sample/coverage/OOS gates, and returns enough metadata for a human to audit the
result. Missing or stale reports fall back to the declared prior weights.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IC_DIR = ROOT / "generated" / "ic_report"
IC_CSV = IC_DIR / "FACTOR_IC_REPORT_VECTORIZED.csv"
OOS_JSON = IC_DIR / "IC_OOS_REPORT.json"

DEFAULT_DIMENSION_WEIGHTS = {
    "trend": 0.25,
    "valuation": 0.16,
    "financial": 0.20,
    "liquidity": 0.14,
    "events": 0.07,
    "commodity": 0.18,
}
CATEGORY_DIMENSION = {
    "momentum": "trend",
    "reversal": "trend",
    "value": "valuation",
    "quality": "financial",
    "growth": "financial",
    "liquidity": "liquidity",
    "volatility": "liquidity",
    "event": "events",
    "sentiment": "events",
    "flow": "events",
    "industry": "valuation",
    "macro": "events",
    "commodity": "commodity",
}


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _iso_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().date().isoformat()
    except OSError:
        return None


def _safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _normalise(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 0:
        return {key: 0.0 for key in values}
    return {key: max(0.0, float(value)) / total for key, value in values.items()}


def _factor_strength(row: dict) -> float | None:
    ic = _finite(row.get("ic_mean"))
    icir = _finite(row.get("icir"))
    coverage = _finite(row.get("coverage"))
    if ic is None or icir is None:
        return None
    coverage = 1.0 if coverage is None else max(0.0, min(1.0, coverage))
    # IC and ICIR are deliberately bounded before blending so one unstable
    # statistic cannot dominate the entire calibration layer.
    return (0.55 * min(abs(ic) / 0.10, 1.0) + 0.45 * min(abs(icir) / 2.0, 1.0)) * math.sqrt(coverage)


def build_ic_weight_explanation(min_observations: int = 120, min_coverage: float = 0.80,
                                base_weights: dict[str, float] | None = None) -> dict:
    """Build a strict, JSON-safe IC/OOS calibration report.

    ``base_weights`` is the business prior supplied by the caller. Keeping the
    prior explicit prevents a later model overlay from silently reverting to a
    second hard-coded weight table.
    """
    base = dict(base_weights or DEFAULT_DIMENSION_WEIGHTS)
    if not IC_CSV.exists():
        return {
            "status": "missing", "source": str(IC_CSV.relative_to(ROOT)), "as_of": None,
            "forward_return_days": None, "universe": None,
            "method": "fallback_business_prior", "min_observations": min_observations,
            "fallback": "IC报告不存在，使用业务规则先验权重；未将规则权重称为IC权重",
            "weights": [], "category_weights": {}, "dimension_weights": base,
            "included_factor_count": 0, "oos": {"status": "missing", "as_of": None, "flipped_count": None},
        }

    oos = _safe_json(OOS_JSON) if OOS_JSON.exists() else {}
    flipped = {str(value) for value in (oos.get("flipped_factors") or [])}
    stable = {str(item.get("factor")) for item in (oos.get("stable_top") or []) if isinstance(item, dict)}
    try:
        frame = pd.read_csv(IC_CSV)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error", "source": str(IC_CSV.relative_to(ROOT)), "as_of": _iso_mtime(IC_CSV),
            "forward_return_days": None, "universe": None,
            "method": "fallback_business_prior", "min_observations": min_observations,
            "fallback": f"IC报告读取失败: {str(exc)[:160]}", "weights": [],
            "category_weights": {}, "dimension_weights": base, "included_factor_count": 0,
            "oos": {"status": "available" if oos else "missing", "as_of": oos.get("generated_at"), "flipped_count": len(flipped)},
        }

    rows = []
    category_raw: dict[str, float] = {}
    for raw in frame.to_dict("records"):
        factor = str(raw.get("factor") or "").strip()
        category = str(raw.get("category") or "unknown").strip()
        observations = _finite(raw.get("n_days"))
        coverage = _finite(raw.get("coverage"))
        strength = _factor_strength(raw)
        reasons = []
        if not factor:
            reasons.append("factor_missing")
        if observations is None or observations < min_observations:
            reasons.append("insufficient_observations")
        if coverage is not None and coverage < min_coverage:
            reasons.append("low_coverage")
        if strength is None:
            reasons.append("ic_or_icir_missing")
        if factor in flipped and factor not in stable:
            reasons.append("oos_direction_flip")
        included = not reasons
        row = {
            "factor": factor, "category": category,
            "ic_mean": _finite(raw.get("ic_mean")), "icir": _finite(raw.get("icir")),
            "win_rate": _finite(raw.get("winrate")),
            "observations": int(observations) if observations is not None else None,
            "coverage": coverage, "grade": raw.get("grade"),
            "direction": int(_finite(raw.get("direction")) or 0),
            "raw_abs_ic_weight": round(abs(_finite(raw.get("ic_mean")) or 0.0), 6),
            "strength": round(strength, 6) if strength is not None else None,
            "final_weight": 0.0, "included": included,
            "exclude_reason": ";".join(reasons) if reasons else None,
        }
        rows.append(row)
        if included:
            category_raw[category] = category_raw.get(category, 0.0) + float(strength or 0.0)

    included_total = sum(float(row["strength"] or 0.0) for row in rows if row["included"])
    if included_total > 0:
        for row in rows:
            if row["included"]:
                row["final_weight"] = round(float(row["strength"] or 0.0) / included_total, 6)
    category_weights = _normalise(category_raw)
    empirical_dimension_raw = {key: 0.0 for key in base}
    for category, weight in category_weights.items():
        dimension = CATEGORY_DIMENSION.get(category)
        if dimension in empirical_dimension_raw:
            empirical_dimension_raw[dimension] += weight
    empirical_dimensions = _normalise(empirical_dimension_raw)
    # Only a bounded 20% overlay is allowed. The rule prior remains dominant,
    # and commodity keeps its prior when the IC report has no commodity factors.
    calibrated = {
        key: round(0.80 * base[key] + 0.20 * empirical_dimensions.get(key, 0.0), 6)
        for key in base
    }
    calibrated = _normalise(calibrated)
    return {
        "status": "available" if included_total > 0 else "fallback",
        "source": str(IC_CSV.relative_to(ROOT)), "as_of": _iso_mtime(IC_CSV),
        "forward_return_days": oos.get("params", {}).get("forward"),
        "universe": "warehouse IC snapshot; CSV has no embedded rolling as-of",
        "method": "bounded_20pct_ic_strength_overlay_with_oos_gate",
        "min_observations": min_observations, "min_coverage": min_coverage,
        "fallback": "有效因子不足时保留业务规则先验；OOS翻转因子不参与校准",
        "weights": rows, "category_weights": category_weights,
        "empirical_dimension_weights": empirical_dimensions,
        "dimension_weights": calibrated,
        "included_factor_count": sum(1 for row in rows if row["included"]),
        "oos": {
            "status": "available" if oos else "missing",
            "as_of": oos.get("generated_at"), "sign_keep_rate": oos.get("sign_keep_rate"),
            "flipped_count": len(flipped), "stable_top": sorted(stable),
        },
    }


def calibrated_dimension_weights(base: dict[str, float] | None = None) -> tuple[dict[str, float], dict]:
    """Return calibrated dimensions and the full audit object."""
    prior = dict(base or DEFAULT_DIMENSION_WEIGHTS)
    evidence = build_ic_weight_explanation(base_weights=prior)
    if evidence.get("status") in {"missing", "error"}:
        return prior, evidence
    weights = evidence.get("dimension_weights") or prior
    return {key: float(weights.get(key, prior.get(key, 0.0))) for key in prior}, evidence
