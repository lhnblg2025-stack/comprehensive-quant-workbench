"""
prediction_report.py — QuantV6 预测汇总
聚合所有预测器输出，生成综合市场预测报告 + Markdown 渲染（飞书推送用）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast.history_similarity import SimilarDays
from quant_system.market_forecast.next_day import NextDayPrediction
from quant_system.market_forecast.regime import Regime
from quant_system.market_forecast.sentiment_cycle import SentimentStage
from quant_system.market_forecast.tail_risk import TailRisk
from quant_system.market_forecast.trend import TrendPrediction
from quant_system.market_forecast.volatility_forecast import VolForecast

log = get_logger("qv6.predreport")


@dataclass
class MarketPrediction:
    """综合市场预测（预测层最终输出）。"""
    p_up_1d: float = 0.5
    p_down_1d: float = 0.5
    expected_1d: float = 0.0
    expected_5d: float = 0.0
    regime: Regime | None = None
    sentiment: SentimentStage | None = None
    similarity: SimilarDays | None = None
    next_day: NextDayPrediction | None = None
    tail_risk: TailRisk | None = None
    trend: TrendPrediction | None = None
    volatility: VolForecast | None = None
    overall_bias: str = "neutral"     # bullish/bearish/neutral
    action_hint: str = "hold"         # 加仓/持有/减仓/空仓
    confidence: float = 0.3
    timestamp: str = ""
    risk_flags: list[str] = field(default_factory=list)


def _overall_bias(p_up: float, tail: TailRisk | None, sentiment: SentimentStage | None) -> str:
    """综合方向。

    修复（2026-08-06）：情绪高潮/退潮不再无条件判空。
    原实现：高潮/退潮 → 无条件 bearish → signal_center 仓位 0（空仓）。
    问题：A股强势期涨停常 ≥100（高潮），导致系统长期空仓，错过强势行情；
    且与 test_strategy 设计意图（过热→仓位≤40% 而非空仓）矛盾。
    新逻辑：
      - 高潮/退潮 + p_up 也低（<0.5）→ bearish（真正的顶部风险）
      - 高潮/退潮 + p_up 较高（≥0.5）→ neutral（过热降权，交给 signal_center 的
        过热保护：neutral+过热 → 20% 轻仓，而非空仓）
      - 尾部大跌概率硬风控保持：p_drop_2pct > 0.25 → bearish
    """
    # 尾部大跌概率硬风控（无条件）
    if tail is not None and tail.overheat_after_big_rise and p_up < 0.5:
        return "bearish"
    if tail is not None and tail.p_drop_2pct > 0.25 and p_up < 0.5:
        return "bearish"
    # 情绪高潮/退潮：仅当 p_up 也低时判空；否则降为 neutral（过热交给仓位保护）
    if sentiment is not None and sentiment.stage in ("climax", "ebb"):
        if p_up < 0.5:
            return "bearish"
        return "neutral"
    # 过热+大涨后（高危形态）：p_up 不低时也降为 neutral（不追高）
    if tail is not None and tail.overheat_after_big_rise:
        return "neutral"
    if p_up >= 0.55:
        return "bullish"
    if p_up <= 0.45:
        return "bearish"
    return "neutral"


def _action_hint(bias: str, p_up: float, confidence: float) -> str:
    if bias == "bullish" and confidence >= 0.4:
        return "加仓"
    if bias == "bullish":
        return "持有"
    if bias == "bearish" and confidence >= 0.4:
        return "空仓"
    if bias == "bearish":
        return "减仓"
    return "持有"


def aggregate(regime: Regime | None = None,
              sentiment: SentimentStage | None = None,
              similarity: SimilarDays | None = None,
              next_day: NextDayPrediction | None = None,
              tail_risk: TailRisk | None = None,
              trend: TrendPrediction | None = None,
              volatility: VolForecast | None = None,
              timestamp: str = "") -> MarketPrediction:
    """聚合所有预测器。

    修复 P0-1：改用归一化加权口径（权重 0.4/0.3/0.2/0.1 按实际激活源累加）：
      acc = 0.5 + Σ wᵢ·pᵢ          # 以 0.5 为基线的加权累加
      p   = 0.5 + (acc − 0.5·(1+Σw)) / Σw   # 归一化回概率空间
    等价于 Σ(wᵢ·pᵢ)/Σwᵢ（激活源的加权平均），单源 0.8 → 0.80，
    四源中性 → 0.50，不再系统性压向看空。
    修复 P1-2：next_day.method=="similarity" 时其 p_up 已 ≡ 1−similarity.p_down_1d
    （同一信号），且 method=="ml" 若已在 predict_next_day 内融合相似日
    （model_detail.fused_similarity=True），均不再重复计 similarity 的 0.3 权重。
    """
    pairs: list[tuple[float, float]] = []  # (权重, p_up)

    nd_uses_similarity = bool(
        next_day is not None and next_day.method in ("ml", "similarity")
        and (next_day.method == "similarity"
             or next_day.model_detail.get("fused_similarity", False))
    )
    if next_day is not None and next_day.method in ("ml", "similarity"):
        pairs.append((0.4, next_day.p_up))
    # 仅当 next_day 未消费相似日信号时才计入 similarity 权重，避免双计
    if (similarity is not None and similarity.n_matches >= 5
            and not nd_uses_similarity):
        pairs.append((0.3, 1 - similarity.p_down_1d))
    if trend is not None:
        pairs.append((0.2, trend.continuation_prob))
    if regime is not None:
        from quant_system.market_forecast.regime import next_state_odds
        odds = next_state_odds(regime)
        pairs.append((0.1, odds.get("up", 0.5)))

    if pairs:
        sum_w = sum(w for w, _ in pairs)
        acc = 0.5 + sum(w * p for w, p in pairs)
        p_up = 0.5 + (acc - 0.5 * (1 + sum_w)) / sum_w
    else:
        p_up = 0.5
    p_up = max(0.05, min(0.95, p_up))

    bias = _overall_bias(p_up, tail_risk, sentiment)

    # 置信度
    conf = 0.25
    if similarity is not None:
        conf += min(0.2, similarity.n_matches * 0.02)
    if next_day is not None:
        conf += 0.1 if next_day.method == "ml" else 0.05
    if tail_risk is not None and (tail_risk.heat_flag or tail_risk.overheat_after_big_rise):
        conf += 0.1
    conf = round(min(conf, 0.9), 2)

    risk_flags = []
    if tail_risk is not None and tail_risk.overheat_after_big_rise:
        risk_flags.append("过热+大涨后：警惕冲高回落")
    if tail_risk is not None and tail_risk.p_drop_2pct > 0.2:
        risk_flags.append(f"尾部大跌概率 {tail_risk.p_drop_2pct:.0%}")
    if sentiment is not None and sentiment.stage == "climax":
        risk_flags.append("情绪高潮：追高风险大")
    if trend is not None and trend.momentum_exhausted:
        risk_flags.append("动量衰竭：价格新高但加速度转负")

    expected_5d = similarity.expected_return_5d if similarity is not None else 0.0

    # P2-4：expected_1d 优先级——相似日统计（真实分布期望，n_matches>=5 时）
    # 优先于 next_day 的历史基准均值（prior/ML 路径的 expected 均为已实现均值，
    # 不含预测方向信息）。相似日无数据时才退回 next_day/0。
    if similarity is not None and similarity.n_matches >= 5:
        expected_1d = similarity.expected_return_1d
    elif next_day is not None:
        expected_1d = next_day.expected_return_1d
    else:
        expected_1d = 0.0

    return MarketPrediction(
        p_up_1d=round(p_up, 3),
        p_down_1d=round(1 - p_up, 3),
        expected_1d=round(expected_1d, 2),
        regime=regime, sentiment=sentiment, similarity=similarity,
        next_day=next_day, tail_risk=tail_risk, trend=trend, volatility=volatility,
        overall_bias=bias,
        action_hint=_action_hint(bias, p_up, conf),
        confidence=conf,
        timestamp=timestamp,
        risk_flags=risk_flags,
    )


def to_markdown(pred: MarketPrediction) -> str:
    """渲染为 Markdown 报告（飞书推送）。"""
    lines = ["📊 **牧云天枢 · 市场预测**"]
    lines.append(f"时间: {pred.timestamp} | 置信度 {pred.confidence:.0%}")
    lines.append("")

    emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(pred.overall_bias, "⚪")
    lines.append(f"**综合方向**: {emoji} {pred.overall_bias} | 建议: **{pred.action_hint}**")
    lines.append(f"次日上涨概率: {pred.p_up_1d:.0%} | 期望收益: {pred.expected_1d:+.2f}%")
    lines.append(f"次5日期望: {pred.expected_5d:+.2f}%")
    lines.append("")

    if pred.regime:
        r = pred.regime
        lines.append(f"**市场状态**: {r.trend}/{r.volatility} "
                     f"(vs300MA {r.price_vs_300ma:+.1f}%, 波动 {r.vol_annual:.0f}%)")
    if pred.sentiment:
        s = pred.sentiment
        lines.append(f"**情绪周期**: {s.stage_cn} (涨停{s.limit_up}/跌停{s.limit_down}, "
                     f"上涨比{s.rise_ratio:.0%})")
        stats = s.next_day_stats
        if stats:
            lines.append(f"  该阶段历史: 次日涨{stats.get('p_up', 0):.0%}/跌{stats.get('p_down', 0):.0%} "
                         f"均值{stats.get('mean_ret', 0):+.2f}%")
    if pred.tail_risk:
        t = pred.tail_risk
        flag = "⚠️" if t.overheat_after_big_rise or t.p_drop_2pct > 0.2 else ""
        lines.append(f"**尾部风险**{flag}: 次日大跌(<-2%)概率 {t.p_drop_2pct:.0%} | "
                     f"大涨概率 {t.p_rise_2pct:.0%}")
    if pred.similarity and pred.similarity.n_matches:
        sim = pred.similarity
        lines.append(f"**历史相似日** ({sim.n_matches}个): "
                     f"次日下跌概率 {sim.p_down_1d:.0%}, 均值 {sim.expected_return_1d:+.2f}%")
        for d in sim.similar_days[:3]:
            lines.append(f"  {d['date']} 相似{d['sim']:.2f} → 次日{d['next_1d']:+.2f}%")
    if pred.volatility:
        v = pred.volatility
        lines.append(f"**波动率**: 次日年化 {v.vol_1d:.0f}% (历史分位 {v.vol_percentile:.0%})")
    if pred.risk_flags:
        lines.append("")
        lines.append("🚨 **风险提示**:")
        for f in pred.risk_flags:
            lines.append(f"- {f}")
    return "\n".join(lines)
