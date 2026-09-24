"""
evaluation.py — 因子评估器（Factor Evaluator）
V4.1 feature

提供因子的 IC/ICIR/衰减/拥挤度/分层回收益等全维度评估。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

import logging
import numpy as np
import pandas as pd

# anaconda(py3.9/numpy2.x) 与 scipy1.9 二进制不兼容时的降级：
# 用 pandas rank 相关近似 Spearman（无 scipy 依赖）
try:
    from scipy import stats as _sp_stats
except Exception:  # noqa: BLE001
    _sp_stats = None


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman 秩相关：优先 scipy，不可用时用 pandas rank 相关降级。"""
    if _sp_stats is not None:
        try:
            return float(_sp_stats.spearmanr(a, b)[0])
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"[evaluation] 操作失败: {e}", exc_info=True)
    ra = a.rank()
    rb = b.rank()
    return float(ra.corr(rb))


class FactorEvaluator:
    """因子评估器"""

    @staticmethod
    def rank_ic(alpha: pd.Series, forward_return: pd.Series) -> float:
        """Rank IC（Spearman 相关系数）

        P2-Q5-fix (L447): 样本不足 30 记 NaN 而非 0.0——0.0 会把无有效样本的
        小截面误判为"IC 为零"并拉低均值；NaN 在均值/统计中自然被忽略。
        """
        common = alpha.dropna().index.intersection(forward_return.dropna().index)
        if len(common) < 30:
            return float("nan")
        return _spearman(alpha.loc[common], forward_return.loc[common])

    @staticmethod
    def pearson_ic(alpha: pd.Series, forward_return: pd.Series) -> float:
        """Pearson IC（样本不足 30 记 NaN，理由同 rank_ic）"""
        common = alpha.dropna().index.intersection(forward_return.dropna().index)
        if len(common) < 30:
            return float("nan")
        return alpha.loc[common].corr(forward_return.loc[common])

    @staticmethod
    def ic_series(alpha_df: pd.DataFrame, forward_return: pd.Series) -> pd.Series:
        """逐期 IC 序列

        P2-Q5-fix (L447): 校验并文档化输入格式。
        - 面板模式：alpha_df 与 forward_return 都必须为 (date, symbol) MultiIndex，
          否则 xs(d, level=0) 会 KeyError；本方法主动校验并给出清晰错误。
        - 单期模式：alpha_df 为普通 index 的 Series 样。
        - 某期在 forward_return 中无对应收益 → 该期记为 NaN（可见缺失，不硬拼）。
        """
        if isinstance(alpha_df.index, pd.MultiIndex):
            if not isinstance(forward_return.index, pd.MultiIndex):
                raise ValueError(
                    "ic_series 面板模式要求 forward_return 与 alpha_df 同为 "
                    "(date, symbol) MultiIndex 格式"
                )
            fwd_dates = set(forward_return.index.get_level_values(0))
            dates = alpha_df.index.get_level_values(0).unique()
            ics = {}
            for d in dates:
                if d not in fwd_dates:
                    ics[d] = float("nan")  # 该期无前向收益
                    continue
                a_slice = alpha_df.xs(d, level=0)
                r_slice = forward_return.xs(d, level=0)
                ics[d] = FactorEvaluator.rank_ic(a_slice, r_slice)
            return pd.Series(ics)
        # 单期
        return pd.Series([FactorEvaluator.rank_ic(alpha_df, forward_return)])

    @staticmethod
    def icir(ic_series: pd.Series, annual_factor: int = 252) -> float:
        """ICIR = mean(IC) / std(IC) * sqrt(annual_factor)"""
        if len(ic_series) < 5:
            return 0.0
        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=1)
        if ic_std < 1e-12:
            return 0.0
        return ic_mean / ic_std * np.sqrt(annual_factor)

    @staticmethod
    def quantile_returns(alpha: pd.Series, forward_return: pd.Series,
                         n_buckets: int = 10) -> pd.Series:
        """分层回收益"""
        common = alpha.dropna().index.intersection(forward_return.dropna().index)
        if len(common) < n_buckets * 5:
            return pd.Series(index=range(n_buckets), dtype=float)

        a = alpha.loc[common]
        r = forward_return.loc[common]
        labels = pd.qcut(a.rank(), n_buckets, labels=False, duplicates="drop")
        return r.groupby(labels).mean()

    @staticmethod
    def spread_return(alpha: pd.Series, forward_return: pd.Series,
                      n_buckets: int = 10) -> float:
        """多空收益差（Top组 - Bottom组）"""
        qr = FactorEvaluator.quantile_returns(alpha, forward_return, n_buckets)
        if len(qr) < n_buckets:
            return 0.0
        return qr.iloc[-1] - qr.iloc[0]

    @staticmethod
    def factor_decay(alpha_df: pd.DataFrame, forward_return: pd.Series,
                     max_lag: int = 20) -> pd.Series:
        """因子衰减曲线：计算滞后1~max_lag期的IC"""
        if not isinstance(alpha_df.index, pd.MultiIndex):
            raise ValueError("需要 MultiIndex (date, symbol) 格式")

        dates = sorted(alpha_df.index.get_level_values(0).unique())
        decay_ics = []
        for lag in range(1, max_lag + 1):
            lagged_return = forward_return.groupby(level=1).shift(-lag)
            # P1-Q5-fix: V5.4 对全部 (date, symbol) 混合算单一 Spearman——不同期收益
            #   混在一起，衰减曲线数值错误。改为逐 date 计算 rank_ic 后取均值
            #   （同 compute_ic 逻辑），再对无数据期做可见降级（均值默认忽略 NaN）。
            per_date = []
            for d in dates:
                a_slice = alpha_df.xs(d, level=0)
                r_slice = lagged_return.xs(d, level=0)
                per_date.append(FactorEvaluator.rank_ic(a_slice, r_slice))
            decay_ics.append(float(np.nanmean(per_date)) if per_date else 0.0)
        return pd.Series(decay_ics, index=range(1, max_lag + 1), name="decay_ic")

    @staticmethod
    def factor_turnover(alpha_df: pd.DataFrame, top_pct: float = 0.2) -> float:
        """因子换手率：Top 组换手占比（0~1）。

        P2-Q5-fix (M446): V5.4 返回截面排名绝对变化均值（0~N），不是换手率定义。
        改为相邻两期按因子值排名取前 top_pct 的股票，换手率 = 1 - 交集数/Top组数，
        与 factor_zoo.compute_factor_turnover 口径一致。
        """
        if not isinstance(alpha_df.index, pd.MultiIndex):
            raise ValueError("需要 MultiIndex (date, symbol) 格式")

        dates = sorted(alpha_df.index.get_level_values(0).unique())
        tos = []
        for i in range(1, len(dates)):
            prev = alpha_df.xs(dates[i - 1], level=0).dropna()
            curr = alpha_df.xs(dates[i], level=0).dropna()
            common = prev.index.intersection(curr.index)
            if len(common) == 0:
                continue
            n_top = max(1, int(len(common) * min(top_pct, 1.0)))
            prev_top = set(prev.loc[common].sort_values(ascending=False).head(n_top).index)
            curr_top = set(curr.loc[common].sort_values(ascending=False).head(n_top).index)
            turnover = 1.0 - len(prev_top & curr_top) / n_top
            tos.append(float(np.clip(turnover, 0.0, 1.0)))

        return float(np.mean(tos)) if tos else 0.0

    @staticmethod
    def factor_correlation_matrix(factors: pd.DataFrame) -> pd.DataFrame:
        """因子截面相关性矩阵"""
        return factors.corr()

    @staticmethod
    def factor_crowding(factors: pd.DataFrame, threshold: float = 0.7) -> dict:
        """因子拥挤度检测
        
        因子间平均相关性 > 0.7 视为拥挤
        """
        corr = factors.corr()
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        high_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        return {
            "avg_correlation": avg_corr,
            "crowding_level": "high" if avg_corr > 0.7 else "medium" if avg_corr > 0.4 else "low",
            "high_corr_pairs": high_pairs[:10],
            "n_high_pairs": len(high_pairs),
        }

    @staticmethod
    def full_report(alpha: pd.DataFrame, forward_return: pd.Series,
                    n_buckets: int = 10) -> dict:
        """因子全维度评估报告"""
        ic_series = FactorEvaluator.ic_series(alpha, forward_return)
        qr_series = FactorEvaluator.quantile_returns(
            alpha.iloc[-min(len(alpha), 1000):],
            forward_return.reindex(alpha.index).iloc[-min(len(alpha), 1000):],
            n_buckets,
        ) if len(alpha) > n_buckets else pd.Series(dtype=float)

        return {
            "rank_ic_mean": ic_series.mean(),
            "rank_ic_std": ic_series.std(ddof=1),
            "rank_icir": FactorEvaluator.icir(ic_series),
            "rank_ic_positive_pct": (ic_series > 0).mean(),
            "quantile_returns": qr_series.to_dict() if not qr_series.empty else {},
            "quantile_spread": FactorEvaluator.spread_return(
                alpha, forward_return, n_buckets),
            "n_periods": len(ic_series),
            "turnover": FactorEvaluator.factor_turnover(alpha),
        }
