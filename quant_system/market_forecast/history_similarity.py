"""
history_similarity.py — QuantV6 历史相似日检索 (k-NN)
特征向量：[当日涨跌幅, 前5日动量, 量比, RSI14, BIAS乖离, 涨停家数, 上涨家数比]
从近10年指数日线库检索 topK 相似日，输出次日/次5日收益分布与概率。
这是"预测驱动"核心模块：用历史说话，不拍脑袋。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.indicators import bias, rsi
from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.similarity")

# P2-2：核心特征只保留有可靠数据源的 5 个。
# 原 limit_up/rise_ratio 在无全市场历史时用指数涨跌幅伪造（np.where 近似），
# 且与真实 activity 尺度错配（db 里 50/10/30 vs current 真实家数/比例），
# 已删除——伪特征比无特征更危险（引入系统性偏差）。
CORE_FEATURES = ["pct", "mom5", "vol_ratio", "rsi14", "bias12"]
# 兼容旧接口：FEATURE_NAMES 指向核心特征集
FEATURE_NAMES = CORE_FEATURES


@dataclass
class SimilarDays:
    """相似日检索结果。"""
    similar_days: list[dict] = field(default_factory=list)  # [{date, sim, next_1d, next_5d}]
    p_down_1d: float = 0.5
    p_down_5d: float = 0.5
    expected_return_1d: float = 0.0
    expected_return_5d: float = 0.0
    median_return_1d: float = 0.0
    median_return_5d: float = 0.0
    n_matches: int = 0
    feature_vector: dict = field(default_factory=dict)



def build_feature_db(index_df: pd.DataFrame, min_rows: int = 600) -> pd.DataFrame:
    """
    从指数日线构建特征数据库。
    index_df: 索引=日期，列 close/high/low/volume（新浪 daily 格式需先 rename）。
    返回 DataFrame: 索引=日期，列=特征+next_1d+next_5d。
    """
    if index_df is None or len(index_df) < min_rows:
        return pd.DataFrame()

    df = index_df.copy()
    if "close" not in df.columns:
        return pd.DataFrame()
    close = df["close"].astype(float)

    df["pct"] = close.pct_change() * 100
    df["mom5"] = (close / close.shift(5) - 1) * 100
    vol_ma5 = df["volume"].shift(1).rolling(5).mean() if "volume" in df.columns else None
    df["vol_ratio"] = (df["volume"] / vol_ma5.replace(0, np.nan)) if vol_ma5 is not None else 1.0
    df["rsi14"] = rsi(close, 14)
    df["bias12"] = bias(close, 12)
    # P2-2：不再伪造 limit_up/rise_ratio（见 CORE_FEATURES 注释）

    df["next_1d"] = df["pct"].shift(-1)
    df["next_5d"] = (close.shift(-5) / close - 1) * 100
    df = df.dropna(subset=FEATURE_NAMES + ["next_1d"])
    return df.tail(3000)  # 近 ~12 年


def find_similar_days(feature_db: pd.DataFrame, current: dict, top_k: int = 10,
                      index_df: pd.DataFrame | None = None,
                      exclusion_days: int = 5) -> SimilarDays:
    """
    k-NN 相似日检索。
    current: 当日特征 {pct, mom5, vol_ratio, rsi14, bias12}
    exclusion_days: P2-1 排除最近 N 个交易日（近几日特征与今日几乎一致，
        会垄断最近邻导致检索退化）。默认排除最近 5 个交易日。
    """
    if feature_db is None or len(feature_db) == 0:
        return SimilarDays()

    db = feature_db.copy()

    # P2-1：排除最近 exclusion_days 个交易日
    if exclusion_days > 0:
        db = db.sort_index()
        if len(db) > exclusion_days:
            db = db.iloc[:-exclusion_days]
        else:
            return SimilarDays()

    vec = np.array([current.get(f, 0.0) for f in FEATURE_NAMES], dtype=float)

    # 标准化（用库内均值和标准差）
    mu = db[FEATURE_NAMES].mean()
    sd = db[FEATURE_NAMES].std().replace(0, 1.0)
    db_n = (db[FEATURE_NAMES] - mu) / sd
    vec_n = (vec - mu.values) / sd.values

    # 欧氏距离
    dist = ((db_n - vec_n) ** 2).sum(axis=1)
    db = db.assign(_dist=dist).sort_values("_dist")

    k = min(top_k, len(db))
    top = db.head(k)

    similar = []
    for date, r in top.iterrows():
        similar.append({
            "date": str(pd.Timestamp(date).date()),
            "sim": round(float(1 / (1 + r["_dist"])), 4),
            "next_1d": round(float(r["next_1d"]), 2),
            "next_5d": round(float(r["next_5d"]) if not pd.isna(r["next_5d"]) else 0.0, 2),
        })

    n1 = top["next_1d"].astype(float)
    n5 = top["next_5d"].astype(float).dropna()

    return SimilarDays(
        similar_days=similar,
        p_down_1d=round(float((n1 < 0).mean()), 3) if len(n1) else 0.5,
        p_down_5d=round(float((n5 < 0).mean()), 3) if len(n5) else 0.5,
        expected_return_1d=round(float(n1.mean()), 2) if len(n1) else 0.0,
        expected_return_5d=round(float(n5.mean()), 2) if len(n5) else 0.0,
        median_return_1d=round(float(n1.median()), 2) if len(n1) else 0.0,
        median_return_5d=round(float(n5.median()), 2) if len(n5) else 0.0,
        n_matches=k,
        feature_vector={f: round(float(current.get(f, 0)), 3) for f in FEATURE_NAMES},
    )


def compute_current_features(index_df: pd.DataFrame, activity: dict | None = None) -> dict:
    """从最新指数日线 + 活跃度计算当日特征向量。"""
    if index_df is None or len(index_df) < 30:
        return {}
    close = index_df["close"].astype(float)
    last = index_df.iloc[-1]
    prev = index_df.iloc[-2] if len(index_df) > 1 else last

    pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else 0.0
    mom5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else pct

    vol_ratio = 1.0
    if "volume" in index_df.columns:
        vol_ma5 = index_df["volume"].tail(6).head(5).mean()
        if vol_ma5 and vol_ma5 > 0:
            vol_ratio = float(index_df["volume"].iloc[-1] / vol_ma5)

    rsi14 = float(rsi(close, 14).iloc[-1])
    bias12 = float(bias(close, 12).iloc[-1])

    # P2-2：不再伪造 limit_up/rise_ratio——无真实全市场 activity 时缺省用中性值，
    # 但该字段已不在 CORE_FEATURES 中，不影响距离计算

    return {
        "pct": round(float(pct), 3),
        "mom5": round(float(mom5), 3),
        "vol_ratio": round(float(vol_ratio), 3),
        "rsi14": round(rsi14, 3),
        "bias12": round(bias12, 3),
    }


class HistorySimilarityEngine:
    """带缓存的历史相似日引擎。"""

    def __init__(self, symbol: str = "sz399006", max_db_rows: int = 3000):
        self.symbol = symbol
        self.max_db_rows = max_db_rows
        self._db: pd.DataFrame | None = None
        self._db_date: str = ""

    def _load_db(self, index_df: pd.DataFrame) -> pd.DataFrame:
        key = str(index_df.index[-1].date()) if len(index_df) else ""
        if self._db is None or key != self._db_date:
            self._db = build_feature_db(index_df, min_rows=200)
            self._db_date = key
        return self._db

    def predict(self, index_df: pd.DataFrame, activity: dict | None = None,
                top_k: int = 10) -> SimilarDays:
        db = self._load_db(index_df)
        current = compute_current_features(index_df, activity)
        if not current or db is None or len(db) == 0:
            return SimilarDays()
        return find_similar_days(db, current, top_k=top_k, index_df=index_df)
