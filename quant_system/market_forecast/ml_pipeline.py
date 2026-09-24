# 已融合：quant_v6.predict.ml_pipeline 移植为自包含实现
"""Lightweight ML pipeline wrapper for QuantV6 predictions."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_system.market_forecast.next_day import NextDayModel, NextDayPrediction, predict_next_day


@dataclass
class MLPipelineResult:
    next_day: NextDayPrediction = field(default_factory=NextDayPrediction)
    trained: bool = False
    diagnostics: dict = field(default_factory=dict)


class MLPipeline:
    def __init__(self, min_train: int = 400):
        self.next_day_model = NextDayModel(min_train=min_train)

    def fit(self, index_df: pd.DataFrame) -> bool:
        return self.next_day_model.train(index_df)

    def predict(self, index_df: pd.DataFrame) -> MLPipelineResult:
        trained = self.next_day_model.is_trained
        pred = predict_next_day(index_df, model=self.next_day_model if trained else None)
        return MLPipelineResult(pred, trained, getattr(self.next_day_model, "model_detail", {}))


def run_ml_pipeline(index_df: pd.DataFrame, min_train: int = 400) -> MLPipelineResult:
    pipe = MLPipeline(min_train=min_train)
    pipe.fit(index_df)
    return pipe.predict(index_df)
