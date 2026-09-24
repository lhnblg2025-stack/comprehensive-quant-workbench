"""
signal_center.py — QuantV6 信号中枢
串联：数据 → 预测层(市场) → 策略层(仓位/选股) → 输出综合 Signal。
这是整个系统的"心脏"，所有模块在此汇合。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.data.index_data import get_cyb_history
from quant_system.market_forecast._support.data.sentiment_raw import get_market_activity
from quant_system.market_forecast.history_similarity import HistorySimilarityEngine
from quant_system.market_forecast.next_day import NextDayModel, predict_next_day
from quant_system.market_forecast.prediction_report import MarketPrediction, aggregate, to_markdown
from quant_system.market_forecast.regime import detect_regime
from quant_system.market_forecast.sentiment_cycle import detect_sentiment, SentimentCycleTracker
from quant_system.market_forecast.tail_risk import predict_tail_risk
from quant_system.market_forecast.trend import predict_trend
from quant_system.market_forecast.volatility_forecast import predict_volatility

log = get_logger("qv6.signal")


@dataclass
class Signal:
    """最终信号（策略层消费）。"""
    score: float = 50.0
    bias: str = "neutral"
    position_coeff: float = 0.3     # 目标仓位系数 0~1
    market_prediction: MarketPrediction | None = None
    allocation: dict = field(default_factory=dict)
    picks: list = field(default_factory=list)
    hold: bool = False
    note: str = ""
    risk_flags: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    timestamp: str = ""


class SignalCenter:
    """信号中枢：一次调用产出完整市场信号。"""

    def __init__(self, index_symbol: str = "sz399006"):
        self.index_symbol = index_symbol
        self.similarity_engine = HistorySimilarityEngine(symbol=index_symbol)
        self.next_day_model = NextDayModel()
        self.sentiment_tracker = SentimentCycleTracker()
        self._trained = False

    def _ensure_model(self, index_df) -> None:
        if not self._trained:
            if self.next_day_model.train(index_df):
                self._trained = True
                log.info("次日预测模型训练完成")

    def generate(self, index_df=None, activity: dict | None = None) -> Signal:
        """
        生成综合信号（含市场预测 + 仓位建议）。
        index_df: 指数日线（None 则自动拉取创业板指历史）
        activity: 乐咕活跃度（None 则自动拉取）
        """
        from quant_system.market_forecast._support.common.time_utils import fmt

        if index_df is None or len(index_df) == 0:
            index_df = get_cyb_history(days=1500)
        if not activity:
            activity = get_market_activity()

        ts = fmt()

        # ── 预测层 ──
        regime = detect_regime(index_df)
        sentiment = detect_sentiment(activity, self.sentiment_tracker)
        similarity = self.similarity_engine.predict(index_df, activity)
        tail = predict_tail_risk(index_df, activity)
        trend = predict_trend(index_df)
        vol = predict_volatility(index_df)

        self._ensure_model(index_df)
        nd = predict_next_day(index_df, similarity, self.next_day_model)

        market_pred = aggregate(
            regime=regime, sentiment=sentiment, similarity=similarity,
            next_day=nd, tail_risk=tail, trend=trend, volatility=vol,
            timestamp=ts,
        )

        # ── 策略层：仓位系数 ──
        coeff, bias, note = self._position_coeff(market_pred)

        # ── 风险标记 ──
        risk_flags = list(market_pred.risk_flags)

        signal = Signal(
            score=round(market_pred.p_up_1d * 100, 1),
            bias=bias,
            position_coeff=coeff,
            market_prediction=market_pred,
            hold=coeff <= 0.05,
            note=note,
            risk_flags=risk_flags,
            components={
                "p_up_1d": market_pred.p_up_1d,
                "expected_1d": market_pred.expected_1d,
                "expected_5d": market_pred.expected_5d,
                "regime": market_pred.regime.regime_label if market_pred.regime else "unknown",
                "sentiment_stage": market_pred.sentiment.stage if market_pred.sentiment else "unknown",
                "p_drop_2pct": market_pred.tail_risk.p_drop_2pct if market_pred.tail_risk else 0.0,
                "confidence": market_pred.confidence,
            },
            timestamp=ts,
        )
        return signal

    def attach_news_sentiment(self, signal: Signal, symbols: list[str] | None = None,
                              timeout_limit: int = 20) -> Signal:
        """新闻情绪增强（可选）：全局情绪→仓位微调；个股情绪→picks 加分。

        - 数据源失败/超时自动跳过，绝不崩溃。
        - 全局 score > 0.05 仓位系数 +0.05；< -0.05 仓位系数 -0.05。
        - 个股 score > 0.2 时把该股加入 picks 备注。
        """
        from quant_system.market_forecast._support.data.news import probe as _news_probe
        import time

        t0 = time.time()
        try:
            symbols = symbols or ["600519", "000858", "002714", "601899"]
            res = _news_probe(symbols, limit_flash=20, limit_stock=5)
            gs = res.get("global_sentiment", {})
            gscore = float(gs.get("score", 0.0))
            stock_factors = res.get("stock_factors", {})

            note_extra = f"新闻情绪 {gscore:+.2f}（热度{gs.get('heat', '-')}）"
            signal.note = (signal.note + " | " + note_extra) if signal.note else note_extra

            # 全局情绪微调仓位（±0.05，限幅）
            if gscore > 0.05:
                signal.position_coeff = min(1.0, signal.position_coeff + 0.05)
            elif gscore < -0.05:
                signal.position_coeff = max(0.0, signal.position_coeff - 0.05)

            # 个股情绪加分项
            hot = [s for s, f in stock_factors.items()
                   if isinstance(f, dict) and float(f.get("score", 0)) > 0.2]
            if hot:
                signal.components["news_hot_stocks"] = hot

            signal.components["news_global_score"] = gscore
            signal.components["news_elapsed_s"] = round(time.time() - t0, 1)
        except Exception as exc:
            log.warning("新闻情绪增强失败（跳过）: %s", exc)
        return signal

    @staticmethod
    def _position_coeff(pred: MarketPrediction) -> tuple[float, str, str]:
        """预测 → 仓位系数。核心决策规则。"""

        # 1. 空仓条件（最硬）
        if pred.overall_bias == "bearish":
            return 0.0, "bearish", "看空：空仓避险"
        if pred.tail_risk is not None and pred.tail_risk.p_drop_2pct > 0.3:
            return 0.0, "bearish", f"尾部风险高(大跌概率{pred.tail_risk.p_drop_2pct:.0%})：空仓"

        # 2. 过热保护：大涨后+高潮 → 上限 40%
        overheat = False
        if pred.tail_risk is not None and pred.tail_risk.overheat_after_big_rise:
            overheat = True
        if pred.sentiment is not None and pred.sentiment.stage == "climax":
            overheat = True

        # 3. 正常映射
        if pred.overall_bias == "bullish":
            base = 0.8
            if overheat:
                base = min(base, 0.4)
            return round(base, 2), "bullish", "看多：持有/加仓" + ("（过热保护压仓）" if overheat else "")
        if pred.overall_bias == "neutral":
            base = 0.3
            if overheat:
                base = 0.2
            return round(base, 2), "neutral", "中性：轻仓观望" + ("（过热）" if overheat else "")
        return 0.0, "bearish", "看空：空仓"

    def to_markdown(self, signal: Signal) -> str:
        base = to_markdown(signal.market_prediction) if signal.market_prediction else ""
        lines = base.split("\n")
        lines.append("")
        lines.append(f"**仓位建议**: {signal.position_coeff:.0%} | 信号分 {signal.score:.0f}/100 [{signal.bias}]")
        if signal.note:
            lines.append(f"提示: {signal.note}")
        return "\n".join(lines)


def run_market_signal(index_df=None, activity: dict | None = None) -> Signal:
    """便捷入口：一次生成市场信号。"""
    return SignalCenter().generate(index_df, activity)
