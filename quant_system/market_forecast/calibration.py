"""Probability calibration helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip_probability(p: float, lo: float = 0.05, hi: float = 0.95) -> float:
    try:
        return round(float(np.clip(p, lo, hi)), 4)
    except Exception:
        return 0.5


def bayes_shrink(p: float, n: int, prior: float = 0.5, strength: int = 80) -> float:
    """Shrink noisy probabilities toward a neutral prior."""
    n = max(int(n or 0), 0)
    return clip_probability((p * n + prior * strength) / (n + strength))


def calibrate_by_hit_rate(raw_p: float, history: pd.DataFrame | None = None,
                          bucket_width: float = 0.1) -> float:
    """
    Calibrate raw probability using historical bucket hit rate.
    history columns: p, y where y is 0/1 outcome.
    """
    if history is None or len(history) < 30 or not {"p", "y"}.issubset(history.columns):
        return clip_probability(raw_p)
    p = clip_probability(raw_p)
    lo = max(0.0, p - bucket_width / 2)
    hi = min(1.0, p + bucket_width / 2)
    bucket = history[(history["p"] >= lo) & (history["p"] <= hi)]
    if len(bucket) < 10:
        return bayes_shrink(p, len(history))
    hit = float(bucket["y"].astype(float).mean())
    return bayes_shrink(hit, len(bucket), prior=p, strength=30)


def brier_score(p: pd.Series, y: pd.Series) -> float:
    df = pd.concat([p, y], axis=1).dropna()
    if len(df) == 0:
        return 0.0
    return float(((df.iloc[:, 0].astype(float) - df.iloc[:, 1].astype(float)) ** 2).mean())
