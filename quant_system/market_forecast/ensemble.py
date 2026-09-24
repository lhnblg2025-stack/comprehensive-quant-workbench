# 已融合：quant_v6.predict.ensemble 移植为自包含实现
"""Prediction ensemble layer."""
from __future__ import annotations

from dataclasses import dataclass, field

from quant_system.market_forecast.calibration import clip_probability


@dataclass
class EnsemblePrediction:
    p_up: float = 0.5
    expected_return: float = 0.0
    confidence: float = 0.0
    components: dict = field(default_factory=dict)


def _get(obj, names: list[str], default=0.5):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def ensemble_predictions(predictions: dict, weights: dict[str, float] | None = None) -> EnsemblePrediction:
    if not predictions:
        return EnsemblePrediction()
    weights = weights or {k: 1.0 for k in predictions}
    total = sum(max(v, 0.0) for v in weights.values()) or 1.0
    p = 0.0
    exp_ret = 0.0
    comps = {}
    used = 0
    for name, pred in predictions.items():
        w = max(weights.get(name, 0.0), 0.0) / total
        p_i = float(_get(pred, ["p_up", "p_up_1d", "continuation_prob"], 0.5))
        e_i = float(_get(pred, ["expected_return", "expected_return_1d", "expected_1d"], 0.0))
        p += w * p_i
        exp_ret += w * e_i
        comps[name] = {"weight": round(w, 3), "p_up": round(p_i, 3)}
        used += 1
    confidence = min(0.85, 0.25 + 0.1 * used + abs(p - 0.5))
    return EnsemblePrediction(clip_probability(p), round(exp_ret, 3), round(confidence, 3), comps)
