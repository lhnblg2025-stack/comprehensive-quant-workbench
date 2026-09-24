# 已融合：quant_v6.predict.stock_rank 移植为自包含实现
"""Stock ranking utilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StockRankResult:
    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    top: list[str] = field(default_factory=list)
    bottom: list[str] = field(default_factory=list)
    score_col: str = "score"


def rank_stocks(factor_score: pd.Series | pd.DataFrame, blacklist: set[str] | None = None,
                top_n: int = 30) -> StockRankResult:
    blacklist = blacklist or set()
    if factor_score is None or len(factor_score) == 0:
        return StockRankResult()
    if isinstance(factor_score, pd.Series):
        df = factor_score.rename("score").to_frame()
    else:
        df = factor_score.copy()
        if "score" not in df.columns:
            numeric = df.select_dtypes("number")
            df["score"] = numeric.mean(axis=1) if len(numeric.columns) else 0.0
    df = df[~df.index.astype(str).isin(blacklist)]
    df = df.sort_values("score", ascending=False)
    return StockRankResult(
        ranked=df,
        top=[str(x) for x in df.head(top_n).index],
        bottom=[str(x) for x in df.tail(top_n).index],
    )


def apply_a_share_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    text = out.astype(str).agg(" ".join, axis=1)
    bad_label = text.str.contains(r"\*?ST|退市|北交所", case=False, regex=True)
    bj_label = text.str.contains(r"(^|\s)BJ(\s|$)", case=False, regex=True)  # P2-Q12-fix: BJ 仅匹配独立代码/标签，避免子串误伤
    mask = ~(bad_label | bj_label)
    return out[mask]
