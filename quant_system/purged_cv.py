"""
purged_cv.py — Purged / Combinatorial Purged Cross-Validation (验证层 · 模块 1/5)

目的
----
解决"IC 好看≠能赚钱"问题中的**时序泄漏**环节：
普通 KFold / TimeSeriesSplit 在金融数据上会把"标签窗口与训练样本重叠"的样本混进训练集，
导致回测/验证结果虚高。本模块实现 López de Prado 提出的：

1. PurgedCV        — 时序 K 折 + purge（剔除标签窗口与测试集重叠的训练样本）
                     + embargo（测试集结束后若干日训练样本禁用，防止信息泄漏）
2. CombinatorialPurgedCV — 组合式净化交叉验证（N 训练组 + k 测试组，
                     C(N+k, k) 种组合），可同时给出多组 OOS 分数分布与
                     PBO（回测过拟合概率）估计。

接口
----
    cv = PurgedCV(n_splits=5, embargo=5, purge=0)
    for train_idx, test_idx in cv.split(X, y, times):
        ...

    scores = purged_cv_score(X, y, times, estimator, n_splits=5,
                             embargo=5, purge=0, scoring=None)

    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo=5, purge=0)
    res = cpcv.evaluate(X, y, times, estimator, scoring=None)
    # res: {folds:[...], score_matrix, group_mean_sharpe, pbo, ...}

所有函数均不依赖其他模块，可独立 import / 独立运行。
"""
from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PurgedCV",
    "purged_cv_score",
    "CombinatorialPurgedCV",
    "expected_max_sharpe_of_trials",
]


# ──────────────────────────────────────────────────────────────────────────
# 工具：按时间排序的样本索引分组（连续时间块）
# ──────────────────────────────────────────────────────────────────────────
def _time_ordered_index(times: np.ndarray):
    """返回按时间升序排列的样本位置。"""
    times = np.asarray(times)
    return np.argsort(times, kind="stable")


def _contiguous_time_groups(times: np.ndarray, n_groups: int):
    """
    把按时间排序的样本切分成 n_groups 个**连续**时间块（块内样本数尽量均衡）。
    返回 list[(start_idx, end_idx)]，每个元素是排序后数组上的切片区间。
    """
    n = len(times)
    if n_groups > n:
        raise ValueError(f"n_groups({n_groups}) 超过样本数({n})")
    boundaries = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = []
    for i in range(n_groups):
        groups.append((boundaries[i], boundaries[i + 1]))
    return groups


# ──────────────────────────────────────────────────────────────────────────
# 1) PurgedCV — 净化时序交叉验证
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PurgedCV:
    """
    时序 K 折交叉验证，支持 purge（标签重叠剔除）与 embargo（禁运期）。

    参数
    ----
    n_splits : int       测试块数量（默认 5）
    embargo  : float     禁运期：测试块结束后 embargo 个时间单位内的样本禁止入训练集
    purge    : float     剔除期：测试块开始前 purge 个时间单位内的样本禁止入训练集
                         （一般设为标签预测跨度，如 5 日收益 → purge=5）
    times    : 数组      每个样本的时间戳（可排序即可，datetime/数值/字符串均可）

    用法
    ----
    >>> cv = PurgedCV(n_splits=5, embargo=5, purge=5)
    >>> for tr, te in cv.split(X, y, times):
    ...     model.fit(X[tr], y[tr]); pred = model.predict(X[te])
    """

    n_splits: int = 5
    embargo: float = 5
    purge: float = 0
    times: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits 至少为 2")
        if self.embargo < 0 or self.purge < 0:
            raise ValueError("embargo / purge 不能为负")

    def _check_times(self, times) -> np.ndarray:
        t = np.asarray(times) if times is not None else self.times
        if t is None or len(t) == 0:
            raise ValueError("需要提供 times（每个样本的时间戳）")
        return t

    @staticmethod
    def _shift_time(t, delta) -> object:
        """V11 审计修复（High）: 原实现 `t_min - self.purge` 在 times 为
        datetime/字符串时 TypeError（datetime - int 非法）。
        修正: datetime → timedelta 偏移；数值 → 直接相减；字符串 → 先转 datetime。"""
        if isinstance(t, (int, float, np.integer, np.floating)):
            return t - delta
        import pandas as pd
        try:
            ts = pd.Timestamp(t)
            if ts.tz is None:
                return ts - pd.Timedelta(days=delta)
            return ts - pd.Timedelta(days=delta)
        except Exception:
            return t  # 无法转换则原样返回（保持行为）

    def split(self, X=None, y=None, times=None, groups=None):
        """
        生成器：产出 (train_idx, test_idx, meta)。
        若 X/y 传了 numpy/pandas 对象，仅用于取样本数；实际划分只依赖 times。
        V11 审计修复: docstring 原写 2 元组，实际产出 3 元组（含 meta），已修正说明。
        """
        times = self._check_times(times)
        n = len(times)
        order = _time_ordered_index(times)
        sorted_times = times[order]
        # 用"样本序号"作为时间单位，保证块间连续性（对任意可排序时间都成立）
        pos = np.arange(n)
        groups_idx = _contiguous_time_groups(pos, self.n_splits)

        for g in range(self.n_splits):
            s, e = groups_idx[g]
            test_pos = set(range(int(s), int(e)))
            # 边界（时间单位=样本序号）
            t_min, t_max = sorted_times[s], sorted_times[e - 1]

            train_pos = []
            n_purged = n_embargoed = 0
            for i in range(n):
                if i in test_pos:
                    continue
                t = sorted_times[i]
                # purge：测试块开始前 purge 个时间单位（标签窗口重叠）
                if self.purge > 0 and t > self._shift_time(t_min, self.purge) and t <= t_min:
                    n_purged += 1
                    continue
                # 审计 2026-08-16：embargo 条件符号修正（原为 -self.embargo 恒不成立，
                # 与 CombinatorialPurgedCV 的 t_max + embargo 一致）
                if t > t_max and t <= self._shift_time(t_max, self.embargo):
                    n_embargoed += 1
                    continue
                train_pos.append(i)
            train_idx = order[train_pos]
            test_idx = order[list(test_pos)]
            yield train_idx, test_idx, {
                "fold": g,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_purged": n_purged,
                "n_embargoed": n_embargoed,
                "test_start": t_min,
                "test_end": t_max,
            }

    def get_n_splits(self, X=None, y=None, times=None, groups=None):
        return self.n_splits


def purged_cv_score(X, y, times, estimator, n_splits=5, embargo=5, purge=0,
                    scoring=None, fit_params=None):
    """
    便捷函数：跑 PurgedCV 并返回每折得分。

    参数
    ----
    scoring : callable(y_true, y_pred) -> float；默认用 estimator.score(X_test, y_test)
    """
    import numpy as _np

    fit_params = fit_params or {}
    cv = PurgedCV(n_splits=n_splits, embargo=embargo, purge=purge)
    X = _np.asarray(X)
    y = _np.asarray(y)
    results = []
    for train_idx, test_idx, meta in cv.split(X, y, times):
        estimator.fit(X[train_idx], y[train_idx], **fit_params)
        y_pred = estimator.predict(X[test_idx])
        if scoring is None:
            score = estimator.score(X[test_idx], y[test_idx])
        else:
            score = scoring(y[test_idx], y_pred)
        results.append({"score": score, **meta})
    return results


# ──────────────────────────────────────────────────────────────────────────
# 2) CombinatorialPurgedCV — 组合式净化交叉验证 + PBO
# ──────────────────────────────────────────────────────────────────────────
def _sharpe_from_returns(ret: np.ndarray, ann_factor: float = 252.0) -> float:
    ret = np.asarray(ret, dtype=float)
    if len(ret) < 2 or np.std(ret) == 0:
        return 0.0
    return float(np.mean(ret) / np.std(ret) * np.sqrt(ann_factor))


@dataclass
class CombinatorialPurgedCV:
    """
    组合式净化交叉验证（López de Prado, 2018）。

    把时间轴切为 (n_splits + n_test_splits) 个连续块，
    每次任选 n_test_splits 个块作测试、其余作训练（带 purge/embargo），
    共 C(n_splits+n_test_splits, n_test_splits) 种组合。

    相比普通 PurgedCV：
    - 每种组合的测试块不连续 → 覆盖不同市场状态，OOS 结果是一个分布而非单点
    - 可计算 PBO（Probability of Backtest Overfitting）：
      对每个组合取表现最好的测试块，看它在全组合排名中是否垫底。
    """

    n_splits: int = 6        # 训练块数量（每个组合中）
    n_test_splits: int = 2   # 每个组合中的测试块数量
    embargo: float = 5
    purge: float = 0

    def __post_init__(self):
        if self.n_splits < 2 or self.n_test_splits < 1:
            raise ValueError("n_splits>=2 且 n_test_splits>=1")
        if self.n_test_splits >= self.n_splits + self.n_test_splits:
            raise ValueError("测试块数量必须小于总块数")

    @property
    def total_groups(self) -> int:
        return self.n_splits + self.n_test_splits

    def _combinations(self):
        return list(itertools.combinations(range(self.total_groups), self.n_test_splits))

    def split(self, X=None, y=None, times=None, groups=None):
        """
        生成器：产出 dict{combo_id, test_groups, train_idx, test_idx, meta}。
        """
        times = np.asarray(times)
        n = len(times)
        order = _time_ordered_index(times)
        sorted_times = times[order]
        pos = np.arange(n)
        groups_idx = _contiguous_time_groups(pos, self.total_groups)

        for combo_id, test_gs in enumerate(self._combinations()):
            test_gs = set(test_gs)
            test_pos = set()
            for g in test_gs:
                s, e = groups_idx[g]
                test_pos.update(range(int(s), int(e)))
            # 测试块的时间范围（用于 purge/embargo）
            gs_sorted = sorted(test_gs)
            t_min = sorted_times[groups_idx[gs_sorted[0]][0]]
            t_max = sorted_times[groups_idx[gs_sorted[-1]][1] - 1]

            train_pos = []
            n_purged = n_embargoed = 0
            for i in range(n):
                if i in test_pos:
                    continue
                t = sorted_times[i]
                if self.purge > 0 and t > t_min - self.purge and t <= t_min:
                    n_purged += 1
                    continue
                if t > t_max and t <= t_max + self.embargo:
                    n_embargoed += 1
                    continue
                train_pos.append(i)
            yield {
                "combo_id": combo_id,
                "test_groups": sorted(test_gs),
                "train_idx": order[train_pos],
                "test_idx": order[list(test_pos)],
                "meta": {
                    "n_train": len(train_pos),
                    "n_test": len(test_pos),
                    "n_purged": n_purged,
                    "n_embargoed": n_embargoed,
                },
            }

    def evaluate(self, X, y, times, estimator, scoring=None, ann_factor=252.0,
                 verbose=False):
        """
        跑全部组合，返回：
        - folds       : 每个组合的 {combo_id, test_groups, score, sharpe(若 scoring 返回收益序列)}
        - score_matrix: shape (total_groups, n_combos) — 每组合内各测试块的分数
        - group_stats : 每个时间块的 OOS 分数均值/标准差
        - pbo         : 回测过拟合概率（0~1，越低越好）

        scoring : callable(y_true, y_pred) -> float，默认 estimator.score。
                  若 scoring 返回 (metrics_dict, returns_ndarray) 元组，则用其夏普计算 PBO。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        times = np.asarray(times)
        combos = self._combinations()
        # score_matrix[group, combo]：组合 combo 中测试块 group 的分数（未测为 nan）
        score_matrix = np.full((self.total_groups, len(combos)), np.nan)
        sharpe_matrix = np.full((self.total_groups, len(combos)), np.nan)
        folds = []
        for combo_id, test_gs in enumerate(combos):
            for split_info in self.split(X, y, times):
                if split_info["combo_id"] != combo_id:
                    continue
                tr, te = split_info["train_idx"], split_info["test_idx"]
                estimator.fit(X[tr], y[tr])
                y_pred = estimator.predict(X[te])
                if scoring is None:
                    score = float(estimator.score(X[te], y[te]))
                    rets = None
                else:
                    out = scoring(y[te], y_pred)
                    if isinstance(out, tuple):
                        score, rets = out[0], np.asarray(out[1], dtype=float)
                    else:
                        score, rets = float(out), None
                for g in split_info["test_groups"]:
                    score_matrix[g, combo_id] = score
                    if rets is not None and len(rets) > 1:
                        sharpe_matrix[g, combo_id] = _sharpe_from_returns(rets, ann_factor)
                    else:
                        sharpe_matrix[g, combo_id] = score
                folds.append({
                    "combo_id": combo_id,
                    "test_groups": split_info["test_groups"],
                    "score": score,
                    **split_info["meta"],
                })
                if verbose:
                    logger.info(f"  combo {combo_id} groups={split_info['test_groups']} "
                                f"score={score:.4f}")
        # PBO：对每个组合，取分数最高的测试块；统计其在整列排名中垫底(下半区)的比例
        pbo = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not np.all(np.isnan(score_matrix)):
                best_rows = np.nanargmax(score_matrix, axis=0)  # 每列最优块
                worst_rows = np.nanargmin(score_matrix, axis=0)
                # 每个块在全部组合中的平均排名（1=最好）
                rank_matrix = np.argsort(np.argsort(-score_matrix, axis=0), axis=0) + 1
                frac_ranks = rank_matrix / np.sum(~np.isnan(score_matrix), axis=0)
                best_ranks = frac_ranks[best_rows, np.arange(score_matrix.shape[1])]
                # 最优块却排在下半区的组合占比 = PBO
                pbo = float(np.mean(best_ranks > 0.5)) if len(best_ranks) else float("nan")
        group_stats = {
            "mean": np.nanmean(score_matrix, axis=1).tolist(),
            "std": np.nanstd(score_matrix, axis=1).tolist(),
            "n_combos": np.sum(~np.isnan(score_matrix), axis=1).tolist(),
        }
        return {
            "folds": folds,
            "score_matrix": score_matrix,
            "sharpe_matrix": sharpe_matrix,
            "group_stats": group_stats,
            "pbo": pbo,
            "n_combos": len(combos),
        }


def expected_max_sharpe_of_trials(n_trials: int, var_sr: float,
                                  skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    E[max SR] 近似（López de Prado 附录），用于 Deflated Sharpe 的基准夏普。

    参数
    ----
    n_trials : int    试验（因子/策略）次数 N
    var_sr   : float  夏普估计量的方差 V
    skew/kurt: float  收益偏度 / 峰度（影响 V 的修正）
    """
    from scipy.stats import norm

    if n_trials < 2:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    emc = (1.0 - gamma) * z1 + gamma * z2  # E[max z]
    return float(np.sqrt(var_sr) * emc)


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Deflated Sharpe Ratio（DSR）：
    考虑试验次数 N 与收益非正态（偏度/峰度）修正后的"真实"夏普显著性。

    公式（López de Prado, 2018）：
        V     = (1 - γ3·SR + (γ4-1)/4·SR²) / (n_obs - 1)
        SR*   = sqrt(V) · E[max z]（N 次试验下的期望最大夏普）
        DSR   = Φ( (SR - SR*) · sqrt(n_obs - 1) / sqrt(V) )

    返回 0~1 的概率。>0.95 才算显著；多因子海选后普遍要求 DSR 显著，
    否则"最好因子"很可能只是 data snooping 的产物。

    参数
    ----
    sharpe   : float  观测夏普（年化）
    n_obs    : int    观测期数（样本天数）
    n_trials : int    尝试过的试验次数（因子数/策略数）——越小越容易显著
    skew     : float  收益偏度（默认 0 正态）
    kurt     : float  收益峰度（默认 3 正态）
    """
    from scipy.stats import norm

    if n_obs < 3:
        return float("nan")
    # 夏普估计量的方差（含偏度/峰度修正）
    var_sr = (1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2) / (n_obs - 1.0)
    var_sr = max(var_sr, 1e-12)
    sr_star = expected_max_sharpe_of_trials(n_trials, var_sr, skew, kurt)
    dsr = norm.cdf((sharpe - sr_star) * np.sqrt(n_obs - 1.0) / np.sqrt(var_sr))
    return float(np.clip(dsr, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────
# 独立运行演示
# ──────────────────────────────────────────────────────────────────────────
def _demo():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    n = 1200
    times = np.arange(n)  # 模拟 1200 个交易日
    X = rng.normal(size=(n, 5))
    # 真实信号（前 5 个特征线性组合）+ 时间漂移，制造标签
    true_signal = X @ np.array([0.5, -0.3, 0.2, 0.1, 0.0]) + 0.2 * np.sin(times / 60)
    p = 1.0 / (1.0 + np.exp(-true_signal))
    y = (rng.random(n) < p).astype(int)

    logger.info("=" * 62)
    logger.info("PurgedCV 演示（n=1200, 5 折, purge=5, embargo=5）")
    logger.info("=" * 62)
    cv = PurgedCV(n_splits=5, embargo=5, purge=5)
    aucs = []
    for tr, te, meta in cv.split(X, y, times):
        m = LogisticRegression(max_iter=500)
        m.fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        aucs.append(auc)
        logger.info(f"  fold {meta['fold']}: train={meta['n_train']} test={meta['n_test']} "
              f"purged={meta['n_purged']} embargoed={meta['n_embargoed']} AUC={auc:.4f}")
    logger.info(f"  mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    logger.info()
    logger.info("=" * 62)
    logger.info("CombinatorialPurgedCV 演示（6 训练块 + 2 测试块 = 28 组合）")
    logger.info("=" * 62)

    def scoring_auc(y_true, y_pred_prob):
        return roc_auc_score(y_true, y_pred_prob)

    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo=5, purge=5)
    # 手动跑一遍得到每组合 AUC（这里用 predict_proba 的 scoring 包装）
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _Wrap(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.model = LogisticRegression(max_iter=500)

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

        def score(self, X, y):
            return roc_auc_score(y, self.model.predict_proba(X)[:, 1])

    est = _Wrap()
    res = cpcv.evaluate(X, y, times, est, scoring=None, verbose=True)
    logger.info(f"  组合数 = {res['n_combos']}, PBO = {res['pbo']:.3f} (<0.5 良好)")
    gm = np.nanmean(res["score_matrix"], axis=0)
    logger.info(f"  OOS AUC 分布: mean={np.mean(gm):.4f} std={np.std(gm):.4f} "
          f"min={np.min(gm):.4f} max={np.max(gm):.4f}")

    logger.info()
    logger.info("=" * 62)
    logger.info("Deflated Sharpe 演示")
    logger.info("=" * 62)
    rng2 = np.random.default_rng(7)
    # 100 次试验中最好的策略，真实夏普 0.6
    rets = rng2.normal(0.6 / np.sqrt(252), 0.01, size=252)
    from scipy.stats import kurtosis, skew
    sr = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
    dsr = deflated_sharpe(sr, n_obs=len(rets), n_trials=100,
                          skew=float(skew(rets)), kurt=float(kurtosis(rets, fisher=False)))
    logger.info(f"  best-of-100 Sharpe={sr:.3f}, DSR(100 trials)={dsr:.3f}")
    dsr1 = deflated_sharpe(sr, n_obs=len(rets), n_trials=1,
                           skew=float(skew(rets)), kurt=float(kurtosis(rets, fisher=False)))
    logger.info(f"  同一策略若只试 1 次: DSR={dsr1:.3f}  ← 试验次数对显著性影响巨大")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _demo()
