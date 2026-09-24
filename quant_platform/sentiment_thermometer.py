"""
quant_platform.sentiment_thermometer — 情绪温度计（V3 融合层）

来源：quant_v6/strategy/sentiment_thermometer.py（V7.0 全市场仓位系数）
设计：纯函数式（无 quant_v6 依赖），输入市场级情绪指标 → 0-100 分 → 总仓位系数。

指标（可缺省，加权平均）:
  热股等权涨跌幅 / 全A等权涨跌幅 / 涨跌停比 / 炸板率 / 连板高度
  破净率 / 低价股占比 / 两融余额变化 / 成交额 / 北向净流入 / 新闻情感 / VIX映射
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# 指标权重（合计 1.0）
DEFAULT_WEIGHTS: dict[str, float] = {
    "hot_equal_ret": 0.15,      # 热股等权涨跌幅（赚钱效应）
    "all_equal_ret": 0.15,      # 全A等权涨跌幅
    "up_down_ratio": 0.10,      # 涨跌家数比
    "limit_up_ratio": 0.10,     # 涨停比例
    "zhaban_rate": 0.08,        # 炸板率（反向）
    "max_lianban": 0.07,        # 最高连板
    "margin_change": 0.08,      # 两融余额变化
    "amount_ratio": 0.07,       # 成交额 5 日均值比
    "north_flow": 0.08,         # 北向净流入
    "news_sentiment": 0.07,     # 新闻情感
    "pb_below_1": 0.05,         # 破净率（高=底）
}
# 指标方向: 1=越高越热, -1=越高越冷
DIRECTIONS: dict[str, int] = {
    "hot_equal_ret": 1, "all_equal_ret": 1, "up_down_ratio": 1,
    "limit_up_ratio": 1, "zhaban_rate": -1, "max_lianban": 1,
    "margin_change": 1, "amount_ratio": 1, "north_flow": 1,
    "news_sentiment": 1, "pb_below_1": 1,
}


def _z(v: float, hist_mean: float, hist_std: float) -> float:
    """历史 z-score → 0-100 概率分。"""
    if hist_std is None or hist_std <= 1e-12:
        return 50.0
    z = (v - hist_mean) / hist_std
    return float(100.0 / (1.0 + np.exp(-z)))


def _heuristic_score(name: str, v: float) -> float:
    """指标典型区间 → 0-100（粗略，历史分布就绪后可替换）。"""
    ranges: dict[str, tuple[float, float]] = {
        "hot_equal_ret": (-0.03, 0.03),     # -3% ~ +3%
        "all_equal_ret": (-0.02, 0.02),
        "up_down_ratio": (0.3, 3.0),
        "limit_up_ratio": (0.0, 0.05),
        "zhaban_rate": (0.0, 0.5),
        "max_lianban": (0, 8),
        "margin_change": (-0.02, 0.02),
        "amount_ratio": (0.6, 1.6),
        "north_flow": (-100, 100),          # 亿
        "news_sentiment": (-1.0, 1.0),
        "pb_below_1": (0.0, 0.3),
    }
    lo, hi = ranges.get(name, (0.0, 1.0))
    if hi == lo:
        return 50.0
    return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))


def compute_thermometer(metrics: dict[str, Any],
                        history: Optional[dict[str, tuple]] = None,
                        weights: Optional[dict[str, float]] = None) -> dict:
    """计算情绪温度计。

    Args:
        metrics: {指标名: 当前值}（可缺省，缺省指标不参与加权）
        history: {指标名: (均值, 标准差)} 历史分布（缺省用简单启发式）
        weights: 自定义权重（覆盖默认）

    Returns:
        {"score": 0-100, "position": 0-1, "bias": str, "parts": {...}}
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts: dict[str, float] = {}
    total_w = 0.0
    acc = 0.0
    for name, weight in w.items():
        if name not in metrics or metrics[name] is None or pd.isna(metrics[name]):
            continue
        v = float(metrics[name])
        if history and name in history:
            hm, hs = history[name]
            score = _z(v, hm, hs)
        else:
            score = _heuristic_score(name, v)
        score = score if DIRECTIONS.get(name, 1) > 0 else 100.0 - score
        parts[name] = round(score, 1)
        acc += score * weight
        total_w += weight
    if total_w <= 0:
        return {"score": 50.0, "position": 0.5, "bias": "neutral", "parts": {}}
    score = float(np.clip(acc / total_w, 0, 100))
    position = float(np.clip(score / 100.0, 0.2, 1.0))
    bias = "greed" if score >= 80 else ("bull" if score >= 60 else (
        "bear" if score < 40 else "neutral"))
    return {"score": round(score, 1), "position": position, "bias": bias,
            "parts": parts}


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    import json
    m = {"hot_equal_ret": 0.01, "all_equal_ret": 0.005, "up_down_ratio": 1.5,
         "limit_up_ratio": 0.02, "zhaban_rate": 0.3, "max_lianban": 5,
         "margin_change": 0.005, "amount_ratio": 1.1, "north_flow": 30}
    r = compute_thermometer(m)
    print(f"温度计: {r['score']} 分 | 仓位 {r['position']:.0%} | 情绪 {r['bias']}")
