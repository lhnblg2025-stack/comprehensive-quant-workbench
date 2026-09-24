"""
sentiment_factory — 多维情绪指标工厂 (V5)

不是单一"温度"，而是多个独立维度的情绪信号：
  - price_sentiment:   基于涨跌比/涨跌停/新高新低的价格情绪
  - volume_sentiment:  基于量能结构/主力资金流的量价情绪
  - funding_sentiment: 基于两融/北向的资金情绪
  - composite:         多维合成 + 历史分位校准

每个维度输出: score: -10~10, percentile: 0~100, direction: bullish/bearish/neutral
"""

from .price_sentiment import PriceSentiment
from .volume_sentiment import VolumeSentiment
from .funding_sentiment import FundingSentiment
from .composite import CompositeSentiment
from .sentiment_divergence import SentimentDivergence

__all__ = [
    "PriceSentiment", "VolumeSentiment", "FundingSentiment",
    "CompositeSentiment", "SentimentDivergence",
]
