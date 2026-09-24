"""market_forecast — 指数级市场预测引擎（原 QuantV6 预测层，移植自包含）"""
from quant_system.market_forecast.breadth import BreadthSignal, breadth_probability, detect_breadth
from quant_system.market_forecast.calibration import bayes_shrink, brier_score, calibrate_by_hit_rate, clip_probability
from quant_system.market_forecast.ensemble import EnsemblePrediction, ensemble_predictions
from quant_system.market_forecast.history_similarity import HistorySimilarityEngine, SimilarDays, build_feature_db, compute_current_features, find_similar_days
from quant_system.market_forecast.ml_pipeline import MLPipeline, MLPipelineResult, run_ml_pipeline
from quant_system.market_forecast.next_day import NextDayModel, NextDayPrediction, build_features, predict_next_day
from quant_system.market_forecast.next_week import NextWeekPrediction, predict_next_week
from quant_system.market_forecast.regime import Regime, detect_regime, estimate_transition, is_bear, is_bull, next_state_odds
from quant_system.market_forecast.sector_rotation import SectorRotation, analyze_sector_rotation
from quant_system.market_forecast.sentiment_cycle import SentimentCycleTracker, SentimentStage, classify_stage, detect_sentiment, stage_next_day_stats
from quant_system.market_forecast.signal_center import SignalCenter
from quant_system.market_forecast.stock_rank import StockRankResult, apply_a_share_filters, rank_stocks
from quant_system.market_forecast.tail_risk import TailRisk, historical_base_rates, overheat_condition_stats, predict_tail_risk
from quant_system.market_forecast.trend import TrendPrediction, predict_trend
from quant_system.market_forecast.volatility_forecast import VolForecast, ewma_vol, garch11_approx, predict_volatility, realized_vol
from quant_system.market_forecast.prediction_report import MarketPrediction, aggregate, to_markdown
__all__ = [
    "SignalCenter", "MarketPrediction", "aggregate", "to_markdown",
    "Regime", "detect_regime", "estimate_transition", "next_state_odds", "is_bull", "is_bear",
    "NextDayModel", "NextDayPrediction", "build_features", "predict_next_day",
    "NextWeekPrediction", "predict_next_week",
    "SentimentCycleTracker", "SentimentStage", "classify_stage", "detect_sentiment", "stage_next_day_stats",
    "SimilarDays", "HistorySimilarityEngine", "build_feature_db", "compute_current_features", "find_similar_days",
    "TailRisk", "historical_base_rates", "overheat_condition_stats", "predict_tail_risk",
    "TrendPrediction", "predict_trend", "VolForecast", "realized_vol", "ewma_vol", "garch11_approx", "predict_volatility",
    "BreadthSignal", "detect_breadth", "breadth_probability", "EnsemblePrediction", "ensemble_predictions",
    "SectorRotation", "analyze_sector_rotation", "StockRankResult", "rank_stocks", "apply_a_share_filters",
    "MLPipeline", "MLPipelineResult", "run_ml_pipeline", "clip_probability", "bayes_shrink", "calibrate_by_hit_rate", "brier_score",
]  # P2-Q12-fix: 补回原预测层常用顶层导出，保持旧调用兼容
