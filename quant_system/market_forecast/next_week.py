# 已融合：quant_v6.predict.next_week 移植为自包含实现
"""Next week return prediction."""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from quant_system.market_forecast.history_similarity import build_feature_db, compute_current_features, find_similar_days


@dataclass
class NextWeekPrediction:
    p_up_5d: float = 0.5
    p_down_5d: float = 0.5
    expected_return_5d: float = 0.0
    confidence: float = 0.0
    method: str = "similarity"
    detail: dict = field(default_factory=dict)


def predict_next_week(index_df: pd.DataFrame, activity: dict | None = None, top_k: int = 20) -> NextWeekPrediction:
    if index_df is None or len(index_df) < 80:
        return NextWeekPrediction(method="prior")
    db = build_feature_db(index_df, min_rows=80)
    current = compute_current_features(index_df, activity)
    sim = find_similar_days(db, current, top_k=top_k) if current else None
    if sim is None or sim.n_matches == 0:
        ret5 = (index_df["close"].astype(float).shift(-5) / index_df["close"].astype(float) - 1) * 100
        hist = ret5.dropna().tail(500)
        if len(hist) == 0:
            return NextWeekPrediction(method="prior")
        p_down = float((hist < 0).mean())
        return NextWeekPrediction(round(1 - p_down, 3), round(p_down, 3), round(float(hist.mean()), 2), 0.25, "base_rate")
    p_down = sim.p_down_5d
    conf = min(0.75, 0.25 + sim.n_matches * 0.02)
    return NextWeekPrediction(
        p_up_5d=round(1 - p_down, 3),
        p_down_5d=p_down,
        expected_return_5d=sim.expected_return_5d,
        confidence=round(conf, 3),
        detail={"matches": sim.similar_days[:5]},
    )
