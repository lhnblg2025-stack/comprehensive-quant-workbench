"""
combination.py — 因子合成器（Factor Combiner）
V4.1 feature

提供多因子合成方法：
1. 等权/加权 Rank 合成
2. Z-score 合成
3. IC 动态加权
4. 机器学习合成
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

logger = __import__('logging').getLogger(__name__)


def _zscore_col(x: pd.Series) -> pd.Series:
    """单列 z-score。P1-Q5-fix: V5.4 用 x.std().clip(lower=1e-12)——
    x.std() 是标量，无 .clip 方法（AttributeError），合成函数全挂。
    改为标量安全：std==0 或 NaN 时该列全 0。"""
    sd = x.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / sd


class FactorCombiner:
    """因子合成器"""

    def __init__(self):
        self._weights: Optional[np.ndarray] = None
        self._model = None

    @staticmethod
    def rank_combine(factors: pd.DataFrame, weights: Optional[list[float]] = None) -> pd.Series:
        """Rank 合成：先排名后加权"""
        ranks = factors.rank(pct=True)
        if weights is None:
            weights = [1.0 / factors.shape[1]] * factors.shape[1]
        w = np.array(weights) / sum(weights)
        combined = ranks.dot(w)
        return combined

    @staticmethod
    def zscore_combine(factors: pd.DataFrame, weights: Optional[list[float]] = None,
                       trim: float = 5.0) -> pd.Series:
        """Z-score 合成"""
        zs = factors.apply(_zscore_col)
        if trim > 0:
            zs = zs.clip(-trim, trim)
        if weights is None:
            weights = [1.0 / factors.shape[1]] * factors.shape[1]
        w = np.array(weights) / sum(weights)
        return zs.dot(w)

    @staticmethod
    def ic_weighted_combine(factors: pd.DataFrame, ic_history: pd.DataFrame) -> pd.Series:
        """IC 动态加权
        
        Parameters
        ----------
        factors : pd.DataFrame
            截面因子值（stocks x factors）
        ic_history : pd.DataFrame
            历史 IC 序列（periods x factors），最后一行作为最新权重
        
        Returns
        -------
        pd.Series
            合成因子
        """
        if len(ic_history) < 5:
            return FactorCombiner.rank_combine(factors)

        # ICIR 加权
        ic_mean = ic_history.mean()
        ic_std = ic_history.std(ddof=1).clip(lower=1e-12)
        icir = (ic_mean / ic_std).clip(lower=0)  # 负ICIR权重置0
        # P1-Q5-fix: ICIR 全≤0 时 icir.sum()==0 → 除零产生全 NaN 权重。
        #   回退等权（可见降级：负 ICIR 无信息，不强制分配权重）。
        total_icir = float(icir.sum())
        if total_icir > 0:
            weights = icir / total_icir
        else:
            weights = pd.Series(1.0 / len(icir), index=icir.index)

        # Z-score 后加权
        zs = factors.apply(_zscore_col)
        return zs.dot(weights.values)

    @staticmethod
    def decay_weighted_combine(factors: pd.DataFrame, ic_series: pd.Series | pd.DataFrame,
                               half_life: int = 20) -> pd.Series:
        """衰减加权合成

        按 IC 大小给近期 IC 更高权重。

        Parameters
        ----------
        ic_series : pd.DataFrame (periods x factors) 或 pd.Series（单因子 IC 时序）
            历史 IC 序列。DataFrame 时逐因子计算 Σ(IC_t × decay_t) 得到每因子权重；
            Series 时无跨因子区分度，退化为等权。
        """
        if len(ic_series) < 5:
            return FactorCombiner.rank_combine(factors)

        decay = np.array([0.5 ** (i / half_life) for i in range(len(ic_series))])
        decay = decay[::-1]  # 最近期权重最大
        decay /= decay.sum()

        # P1-Q5-fix: 逐因子列计算 Σ(IC_t×decay_t)。
        #   V5.4 对全部因子用同一个标量加权 → 每个因子权重相同 → 结果与等权
        #   zscore 合成完全一致，"IC 衰减加权"名不副实。
        if isinstance(ic_series, pd.DataFrame):
            # P1-Q5-fix: axis=0 逐行乘衰减（行=期数），不能 DataFrame*ndarray
            #   （会按列对齐报错/错配）。
            weighted_ic = ic_series.fillna(0.0).mul(decay, axis=0).sum()  # 每因子一个加权 IC
        else:
            weighted_ic = pd.Series(
                float((np.asarray(ic_series.values, dtype=float) * decay).sum()),
                index=factors.columns,
            )
        ic_weights = weighted_ic.clip(lower=0)
        if ic_weights.sum() <= 0:
            # P1-Q5-fix: 加权 IC 全≤0 时归一化除零 → 回退等权（可见降级）
            return FactorCombiner.rank_combine(factors)
        ic_weights = ic_weights / ic_weights.sum()

        return FactorCombiner.zscore_combine(factors, ic_weights.tolist())

    @staticmethod
    def ml_combine(factors: pd.DataFrame, forward_return: pd.Series,
                   method: str = "ridge") -> pd.Series:
        """机器学习合成

        用 Ridge 或 RandomForest 学习因子权重。

        P2-Q5-fix (M445): V5.4 用 model.fit(X, y) 后直接 predict(X)——训练集内
        预测（in-sample），若用于实盘选股构成前视/过拟合。改为 K 折交叉验证的
        样本外（out-of-fold）预测；ImportError 之外的异常（数据/数值问题）也
        可见回退到 Rank 合成，不再裸抛。
        """
        try:
            from sklearn.model_selection import cross_val_predict
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor

            X = factors.fillna(0).values
            y = forward_return.reindex(factors.index).fillna(0).values
            valid = np.isfinite(y)
            X, y = X[valid], y[valid]
            if len(X) < 10:
                logger.warning("ml_combine 有效样本 < 10，回退到 Rank 合成")
                return FactorCombiner.rank_combine(factors)

            if method == "ridge":
                model = Ridge(alpha=1.0)
            elif method == "rf":
                model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
            else:
                raise ValueError(f"未知方法: {method}")

            # 样本外（OOF）预测：每只股票的预测来自不含它的训练折，杜绝前视/过拟合
            n_splits = min(5, max(2, len(X) // 10))
            pred_oof = cross_val_predict(model, X, y, cv=n_splits, n_jobs=1)
            pred = pd.Series(np.nan, index=factors.index, name="ml_alpha")
            pred.iloc[np.where(valid)[0]] = pred_oof
            return pred
        except ImportError:
            logger.warning("sklearn 不可用，回退到 Rank 合成")
            return FactorCombiner.rank_combine(factors)
        except Exception as e:
            logger.warning(f"ml_combine 计算失败({e})，回退到 Rank 合成")
            return FactorCombiner.rank_combine(factors)

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._weights

    def multi_factor_alpha(self, methods: list[str], factors: pd.DataFrame,
                           forward_return: Optional[pd.Series] = None,
                           ic_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """多方法合成，返回每种的 Alpha"""
        results = {}
        for method in methods:
            if method == "rank":
                results[method] = self.rank_combine(factors)
            elif method == "zscore":
                results[method] = self.zscore_combine(factors)
            elif method == "ic_weighted" and ic_history is not None:
                results[method] = self.ic_weighted_combine(factors, ic_history)
            elif method == "decay" and ic_history is not None:
                # P1-Q5-fix: 传完整 ic_history（periods x factors），让 decay_weighted_combine
                #   逐因子计算 Σ(IC_t×decay_t)。V5.4 传 mean(axis=1) 压成单条 IC 时序 →
                #   所有因子权重相同 → 与等权合成无异。
                results[method] = self.decay_weighted_combine(factors, ic_history)
            elif method == "ml" and forward_return is not None:
                results[method] = self.ml_combine(factors, forward_return)
            else:
                results[method] = self.rank_combine(factors)
        return pd.DataFrame(results)
