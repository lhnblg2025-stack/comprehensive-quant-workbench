"""
DEPRECATED (2026-08-07): 无生产引用，保留仅供参考。主用模块见 项目文档/量化交易系统/系统梳理报告.md

D6归档登记 (2026-08-11): 确认 DEPRECATED 状态——全库无生产引用（唯一引用方 = tests/
  integration_tests.py 测试），保留不删除。与 portfolio_optimizer.py 重叠的优化能力
  (BlackLitterman/HRP/Kelly/MeanCVaR/PortfolioOptimizerV2) 为同名异签名异实现保留；
  _historical_cvar_from_returns / _effective_n / _max_drawdown 与 risk_management_pro
  同名近名异口径保留；无真实重复（同名+同签名+同实现）可收敛。

portfolio_v2.py
================

V4.1 feature: portfolio_v2

高级组合构建引擎。

本模块把多种组合构建方法统一为类接口：

    optimize(returns, **kwargs) -> weights

其中 returns 统一约定为 T×N 的收益率矩阵，index 是日期，columns 是资产代码。
如果用户传入 numpy 数组，本模块会自动生成 Asset_000、Asset_001 等列名。
所有权重默认返回 pandas.Series，index 为资产代码，value 为目标权重。

实现模型：

1. BlackLitterman
   - Black-Litterman 模型，参考 Black & Litterman (1992)。
   - 先验收益使用 CAPM 逆向优化：Pi = delta * Sigma * w_mkt。
   - 主观观点通过 Bayesian 更新融合，输出后验收益和后验协方差。

2. HierarchicalRiskParity
   - HRP，参考 Lopez de Prado (2016)。
   - single linkage 层次聚类，准对角化，递归二分分配。
   - 不需要协方差矩阵求逆，对奇异协方差更稳定。

3. NestedClusteredOptimization
   - NCO，参考 Lopez de Prado (2019)。
   - 蒙特卡洛模拟，K-means 聚类，簇内优化，簇间优化。
   - 通过先分解再聚合降低估计误差对优化结果的放大。

4. KellyCriterion
   - 多资产 Kelly 近似：f* = Sigma^{-1} mu。
   - 支持 fractional Kelly，借贷限制，杠杆限制，做空开关。

5. MeanCVaROptimization
   - 均值-CVaR 优化。
   - 支持 Historical CVaR 的 Rockafellar-Uryasev 形式。
   - 支持 Gaussian CVaR 的闭式近似。
   - 支持满仓、行业限制、个股权重限制。

6. RegimeAwareAllocation
   - 根据当前 regime 或 regime 概率动态选择、插值权重。
   - 可对接 quant_system.market_analysis.regime。
   - 支持半衰期平滑，避免状态切换时权重跳变过大。

7. PortfolioOptimizerV2
   - 顶层整合入口。
   - optimize(method='black_litterman'|'hrp'|'nco'|'kelly'|'mean_cvar'|'regime', **kwargs)
   - 返回 weights、stats(vol, sharpe, max_dd)、metadata。

设计原则：

- 对研究环境友好：只依赖 numpy、pandas、scipy，sklearn 是可选依赖。
- 对实盘约束友好：默认 long-only，显式处理上下限、行业限制、杠杆限制。
- 对数值问题保守：协方差矩阵会做对称化、特征值下限修正和伪逆兜底。
- 对调试友好：每个类保留 metadata_、diagnostics_ 或后验属性，便于复盘。

注意：

- 本模块不直接做交易下单。
- 本模块不假设收益率频率，默认 periods_per_year=252 用于年化统计。
- 如果输入 returns 已经是月频或周频，请显式传入 periods_per_year。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm

# sklearn 是可选依赖。NCO 会优先使用 sklearn KMeans，缺失时使用本文件内置的轻量 KMeans。
try:  # pragma: no cover - 环境依赖分支。
    from sklearn.cluster import KMeans as _SklearnKMeans
except Exception:  # pragma: no cover - sklearn 不可用时走 numpy fallback。
    _SklearnKMeans = None  # type: ignore[assignment]


ArrayLike2D = Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]]
ArrayLike1D = Union[pd.Series, np.ndarray, Sequence[float], Mapping[str, float]]
WeightBounds = Union[tuple[float, float], Mapping[str, tuple[float, float]], None]
ObjectiveName = Literal["min_variance", "max_sharpe", "mean_variance"]
CVaRMethod = Literal["historical", "gaussian"]


__all__ = [
    "OptimizationError",
    "PortfolioStats",
    "OptimizationMetadata",
    "BlackLitterman",
    "HierarchicalRiskParity",
    "NestedClusteredOptimization",
    "KellyCriterion",
    "MeanCVaROptimization",
    "RegimeAwareAllocation",
    "PortfolioOptimizerV2",
    "demo_black_litterman",
    "demo_hrp",
    "demo_nco",
    "demo_kelly",
    "demo_mean_cvar",
    "demo_regime",
    "demo_portfolio_optimizer_v2",
]


# ══════════════════════════════════════════════════════════════════════════════
# 通用数据结构
# ══════════════════════════════════════════════════════════════════════════════


class OptimizationError(RuntimeError):
    """组合优化失败时抛出的异常。

    本模块尽量提供可用的 fallback，例如等权或逆方差权重。
    但如果约束本身不可行，例如 N 个资产上限之和小于满仓要求，则会抛出该异常。
    """


@dataclass
class PortfolioStats:
    """组合统计指标。

    Attributes:
        expected_return: 年化预期收益。
        vol: 年化波动率。
        sharpe: 年化夏普比。
        max_dd: 历史最大回撤。
        cvar_95: 95% CVaR，使用组合历史收益估计。
        effective_n: 有效持仓数，定义为 1 / sum(w_i^2)。
        turnover: 如果传入 previous_weights，则返回单边换手率。
    """

    expected_return: float = 0.0
    vol: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    cvar_95: float = 0.0
    effective_n: float = 0.0
    turnover: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """转换为普通字典，便于 JSON 序列化。"""
        return {
            "expected_return": float(self.expected_return),
            "vol": float(self.vol),
            "sharpe": float(self.sharpe),
            "max_dd": float(self.max_dd),
            "cvar_95": float(self.cvar_95),
            "effective_n": float(self.effective_n),
            "turnover": float(self.turnover),
        }


@dataclass
class OptimizationMetadata:
    """优化过程元数据。

    metadata 用于记录方法名、求解器状态、参数、输入维度和诊断信息。
    顶层 PortfolioOptimizerV2 会把各子模型的 metadata_ 合并进返回结果。
    """

    method: str
    success: bool = True
    message: str = ""
    n_assets: int = 0
    n_observations: int = 0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""
        return {
            "method": self.method,
            "success": bool(self.success),
            "message": self.message,
            "n_assets": int(self.n_assets),
            "n_observations": int(self.n_observations),
            "generated_at": self.generated_at,
            "details": self.details,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具函数
# ══════════════════════════════════════════════════════════════════════════════


def _asset_names(n_assets: int) -> list[str]:
    """为 numpy 输入生成稳定资产名。"""
    return [f"Asset_{i:03d}" for i in range(n_assets)]


def _as_returns_frame(returns: ArrayLike2D, assets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """把收益率输入统一转换为 DataFrame。

    中文注释说明：
    - 所有优化器内部都使用 DataFrame，避免资产顺序丢失。
    - 对缺失值采用保守处理：整行全空会删除，局部缺失填 0。
    - 如果某列全是 NaN 或无法转换为数值，则转换后会成为 0，这代表无收益信息。
    """
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim != 2:
            raise ValueError("returns 必须是二维矩阵，形状为 T×N。")
        if assets is None:
            assets = _asset_names(arr.shape[1])
        if len(assets) != arr.shape[1]:
            raise ValueError("assets 长度必须等于 returns 的列数。")
        frame = pd.DataFrame(arr, columns=list(assets))

    if frame.empty:
        raise ValueError("returns 不能为空。")

    frame = frame.copy()
    frame.columns = [str(col) for col in frame.columns]

    # P2-Q15-fix (L113): DataFrame 输入时 assets 参数也参与列选择/排序。
    # 旧实现仅对 numpy/array 输入生效，DataFrame 分支直接 returns.copy()，
    # assets 被静默忽略。现在按 assets 重排/筛选列，缺失列显式报错。
    if isinstance(returns, pd.DataFrame) and assets is not None:
        asset_list = [str(a) for a in assets]
        missing = [a for a in asset_list if a not in frame.columns]
        if missing:
            raise ValueError(f"assets 指定了不在 returns 中的列: {missing}")
        frame = frame[asset_list]

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Q15 修复：全 NaN 列 = 无收益信息（停牌/次新/数据缺失）。若直接填 0 会
    # 伪装成"零方差资产"，逆方差类方法（IVP/HRP/NCO）会把权重全部集中到它。
    # 这里发出警告，下游零方差保护会把该列权重置 0。
    all_nan_cols = [c for c in frame.columns if frame[c].isna().all()]
    if all_nan_cols:
        warnings.warn(
            f"[portfolio_v2] 检测到全 NaN 收益率列 {all_nan_cols}（无收益信息，"
            f"可能为停牌/次新/数据缺失），将按零方差处理并在权重中置 0。",
            RuntimeWarning,
        )

    # P2-Q15-fix (M108): 局部缺失（停牌/次新）填 0 会压低样本方差，
    # 逆方差类方法（IVP/HRP/NCO）因此系统性高估这类资产的权重。
    # 处理：1) 对局部缺失列发可见警告；2) 把每列缺失比例存入 frame.attrs，
    # 供 _annualized_mean_cov 按 1/(1-r) 做方差膨胀（按缺失比例惩罚）。
    partial_missing_cols = [c for c in frame.columns if frame[c].isna().any() and not frame[c].isna().all()]
    if partial_missing_cols:
        ratios = {c: float(frame[c].isna().mean()) for c in partial_missing_cols}
        warnings.warn(
            f"[portfolio_v2] 检测到局部缺失收益率列（可能为停牌/次新/数据缺失）："
            f"{ {c: round(r, 4) for c, r in ratios.items()} }。"
            f"缺失值将填 0，样本方差被低估；协方差估计已按缺失比例做方差膨胀（1/(1-r)）。",
            RuntimeWarning,
        )
        frame.attrs["missing_ratio"] = ratios

    frame = frame.fillna(0.0)

    if frame.shape[1] == 0:
        raise ValueError("returns 至少需要一个资产列。")
    if frame.shape[0] == 0:
        raise ValueError("returns 至少需要一行有效收益率。")

    return frame.astype(float)


def _series_from_any(values: ArrayLike1D, index: Sequence[str], name: str = "value") -> pd.Series:
    """把 dict、Series、array 统一转换为给定 index 的 Series。"""
    idx = pd.Index([str(x) for x in index])
    if isinstance(values, pd.Series):
        out = values.copy()
        out.index = [str(x) for x in out.index]
        return out.reindex(idx).astype(float).fillna(0.0).rename(name)
    if isinstance(values, Mapping):
        out = pd.Series({str(k): float(v) for k, v in values.items()}, dtype=float)
        return out.reindex(idx).astype(float).fillna(0.0).rename(name)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维向量。")
    if len(arr) != len(idx):
        raise ValueError(f"{name} 长度 {len(arr)} 与资产数 {len(idx)} 不一致。")
    return pd.Series(arr, index=idx, name=name, dtype=float)


def _annualized_mean_cov(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    shrinkage: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """估计年化均值和年化协方差。

    shrinkage 使用简单对角压缩：

        Sigma_shrunk = (1 - alpha) * Sigma_sample + alpha * diag(Sigma_sample)

    这不是完整 Ledoit-Wolf，但对研究环境足够稳健，可减少病态相关矩阵。
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage 必须在 [0, 1]。")

    mu = returns.mean(axis=0) * periods_per_year
    cov = returns.cov() * periods_per_year
    cov = cov.reindex(index=returns.columns, columns=returns.columns).fillna(0.0)

    # P2-Q15-fix (M108): 局部缺失列在 _as_returns_frame 被填 0，样本方差被压低到
    # 约 (1-r)*真方差（r=缺失比例）。这里按 1/(1-r) 对对角方差膨胀，使逆方差类
    # 方法（IVP/HRP/NCO）按缺失比例惩罚停牌/次新/数据缺失资产，避免系统性高估其权重。
    # 全缺失列（r=1）不在此膨胀，走零方差置 0 路径。
    missing_ratio = getattr(returns, "attrs", {}).get("missing_ratio", {})
    if missing_ratio:
        for col, r in missing_ratio.items():
            if col in cov.columns and 0.0 < r < 1.0:
                cov.loc[col, col] = cov.loc[col, col] * (1.0 / (1.0 - r))

    if shrinkage > 0.0 and cov.shape[0] > 1:
        diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
        cov = (1.0 - shrinkage) * cov + shrinkage * diag

    cov_values = _regularize_cov(cov.values)
    cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
    return mu.astype(float), cov.astype(float)


def _regularize_cov(cov: Union[np.ndarray, pd.DataFrame], epsilon: float = 1e-10) -> np.ndarray:
    """协方差矩阵数值修正。

    修正规则：
    1. NaN 和 inf 置为 0。
    2. 强制对称化。
    3. 特征值下限截断，避免伪负方差。

    这样做会略微改变极小特征值，但显著提升优化器稳定性。
    """
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("cov 必须是方阵。")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = 0.5 * (arr + arr.T)

    if arr.shape[0] == 1:
        return np.array([[max(float(arr[0, 0]), epsilon)]], dtype=float)

    try:
        eigvals, eigvecs = np.linalg.eigh(arr)
        eigvals = np.maximum(eigvals, epsilon)
        fixed = (eigvecs * eigvals) @ eigvecs.T
        fixed = 0.5 * (fixed + fixed.T)
        return fixed
    except np.linalg.LinAlgError:
        diag = np.diag(np.maximum(np.diag(arr), epsilon))
        return diag.astype(float)


def _safe_pinv(matrix: Union[np.ndarray, pd.DataFrame], rcond: float = 1e-10) -> np.ndarray:
    """安全伪逆。

    对称矩阵先做 regularize，再使用 pinv。
    对非对称矩阵直接 pinv。
    """
    arr = np.asarray(matrix, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        arr = _regularize_cov(arr, epsilon=rcond)
    return np.linalg.pinv(arr, rcond=rcond)


def _cov_to_corr(cov: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """协方差矩阵转换为相关系数矩阵。"""
    arr = _regularize_cov(cov)
    std = np.sqrt(np.maximum(np.diag(arr), 1e-16))
    outer = np.outer(std, std)
    corr = np.divide(arr, outer, out=np.zeros_like(arr), where=outer > 0)
    corr = np.clip(corr, -1.0, 1.0)
    corr[np.diag_indices_from(corr)] = 1.0
    return corr


def _corr_to_distance(corr: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """相关系数转换为 HRP 常用距离矩阵。

    Lopez de Prado (2016) 使用：

        d_ij = sqrt((1 - corr_ij) / 2)

    该距离把完全正相关映射为 0，把完全负相关映射为 1。
    """
    arr = np.asarray(corr, dtype=float)
    arr = np.clip(arr, -1.0, 1.0)
    dist = np.sqrt(np.maximum((1.0 - arr) / 2.0, 0.0))
    dist[np.diag_indices_from(dist)] = 0.0
    return dist


def _infer_bounds(
    assets: Sequence[str],
    bounds: WeightBounds = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成每个资产的上下限数组。"""
    names = [str(a) for a in assets]
    if bounds is None:
        lower = np.full(len(names), float(min_weight), dtype=float)
        upper = np.full(len(names), float(max_weight), dtype=float)
        return lower, upper

    if isinstance(bounds, tuple):
        lower = np.full(len(names), float(bounds[0]), dtype=float)
        upper = np.full(len(names), float(bounds[1]), dtype=float)
        return lower, upper

    lower = np.full(len(names), float(min_weight), dtype=float)
    upper = np.full(len(names), float(max_weight), dtype=float)
    for i, asset in enumerate(names):
        if asset in bounds:
            lo, hi = bounds[asset]
            lower[i] = float(lo)
            upper[i] = float(hi)
    return lower, upper


def _feasible_initial_weights(
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: Optional[float] = 1.0,
) -> np.ndarray:
    """在给定上下限内生成可行初始权重。"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower 和 upper 形状不一致。")
    if np.any(lower > upper):
        raise OptimizationError("权重下限不能大于上限。")

    n = len(lower)
    if n == 0:
        return np.array([], dtype=float)

    if target_sum is None:
        # Kelly 等允许现金存在的模型使用：先从 0 或 lower 出发。
        x0 = np.maximum(lower, 0.0)
        return np.minimum(x0, upper)

    target = float(target_sum)
    lower_sum = float(lower.sum())
    upper_sum = float(upper.sum())
    if lower_sum - 1e-12 > target:
        raise OptimizationError("权重下限之和超过目标满仓权重，约束不可行。")
    if upper_sum + 1e-12 < target:
        raise OptimizationError("权重上限之和小于目标满仓权重，约束不可行。")

    x0 = lower.copy()
    remaining = target - lower_sum
    capacity = upper - lower
    cap_sum = float(capacity.sum())
    if remaining <= 0:
        return x0
    if cap_sum <= 0:
        return x0
    x0 += capacity / cap_sum * remaining
    return np.minimum(np.maximum(x0, lower), upper)


def _series_weights(weights: np.ndarray, assets: Sequence[str], name: str = "weight") -> pd.Series:
    """把权重数组封装为 Series，并清理极小数值。"""
    # P2-Q15-fix (L115): 显式复制，避免原地修改调用方传入的 ndarray
    # （np.asarray 对同 dtype ndarray 不复制，arr[...]=0 会污染入参）。
    arr = np.array(weights, dtype=float, copy=True)
    arr[np.abs(arr) < 1e-12] = 0.0
    return pd.Series(arr, index=pd.Index([str(a) for a in assets]), name=name, dtype=float)


def _effective_n(weights: Union[pd.Series, np.ndarray]) -> float:
    """有效持仓数 Herfindahl inverse。

    D6收敛: 异名/近名异口径保留 —— 平方口径(Σw², 满仓多头), 与
    risk_management_pro._effective_number/_hhi(abs口径, 支持多空) 不同。
    """
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(np.square(w)))
    return 1.0 / denom if denom > 1e-16 else 0.0


def _max_drawdown(portfolio_returns: Union[pd.Series, np.ndarray]) -> float:
    """根据历史组合收益计算最大回撤。

    D6收敛: 异名/近名异口径保留 —— 负向输出, 与 risk_management_pro._max_drawdown_from_returns
    (损失正数)/portfolio_risk.compute_drawdown(净值百分数) 方向与量纲不同。
    """
    r = pd.Series(np.asarray(portfolio_returns, dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r.empty:
        return 0.0
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _historical_cvar_from_returns(
    returns: Union[pd.Series, np.ndarray],
    alpha: float = 0.95,
) -> float:
    """历史 CVaR。

    这里把 loss = -return，CVaR 是超过 VaR 阈值后的平均损失。
    返回值越大表示尾部损失越高。

    D6收敛: 异名异签名异口径保留 —— 与 risk_management_pro._historical_var_cvar 实现近同
    但输入(收益vs损失)/返回(cvar vs (var,cvar))/NaN处理(nan_to_num vs isfinite过滤)不同；
    与 portfolio_risk.compute_var(百分数报告)/portfolio_optimizer.cvar_optimize(尾部概率α) 亦异口径。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须在 (0, 1)。")
    r = np.asarray(returns, dtype=float)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    if r.size == 0:
        return 0.0
    losses = -r
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    if tail.size == 0:
        return max(var, 0.0)
    return float(np.mean(tail))


def _portfolio_stats_from_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    expected_returns: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    previous_weights: Optional[ArrayLike1D] = None,
) -> PortfolioStats:
    """从收益率矩阵和权重计算标准统计。"""
    aligned = weights.reindex(returns.columns).fillna(0.0).astype(float)
    port_ret = returns @ aligned

    if expected_returns is None or cov_matrix is None:
        mu, cov = _annualized_mean_cov(returns, periods_per_year=periods_per_year, shrinkage=0.0)
    else:
        mu = expected_returns.reindex(returns.columns).fillna(0.0).astype(float)
        cov = cov_matrix.reindex(index=returns.columns, columns=returns.columns).fillna(0.0).astype(float)

    w = aligned.values
    exp_ret = float(w @ mu.values)
    vol = float(math.sqrt(max(w @ _regularize_cov(cov.values) @ w, 0.0)))
    sharpe = (exp_ret - risk_free) / vol if vol > 1e-12 else 0.0
    max_dd = _max_drawdown(port_ret)
    cvar_95 = _historical_cvar_from_returns(port_ret, alpha=0.95)

    turnover = 0.0
    if previous_weights is not None:
        prev = _series_from_any(previous_weights, returns.columns, name="previous_weight")
        turnover = float(np.abs(aligned - prev).sum() / 2.0)

    return PortfolioStats(
        expected_return=exp_ret,
        vol=vol,
        sharpe=sharpe,
        max_dd=max_dd,
        cvar_95=cvar_95,
        effective_n=_effective_n(aligned.values),
        turnover=turnover,
    )


def _build_linear_constraints(
    assets: Sequence[str],
    full_investment: bool = True,
    target_sum: float = 1.0,
    sector_map: Optional[Mapping[str, str]] = None,
    sector_limits: Optional[Mapping[str, float]] = None,
    target_return: Optional[float] = None,
    expected_returns: Optional[np.ndarray] = None,
) -> list[Any]:
    """构建 scipy.optimize 支持的线性约束和字典约束。"""
    n = len(assets)
    constraints: list[Any] = []

    if full_investment:
        constraints.append(LinearConstraint(np.ones(n), target_sum, target_sum))

    # P2-Q15-fix (M110): 只传 sector_limits/industry_limits 而未传 sector_map 时，
    # 行业约束无法构建（需要资产→行业映射），旧实现被静默忽略。
    # 改为可见告警，避免调用方误以为行业限制已生效。
    if sector_limits and not sector_map:
        warnings.warn(
            "[portfolio_v2] 收到行业/板块上限（sector_limits/industry_limits）但未提供 "
            "sector_map（资产→行业映射），行业限制无法生效，已被忽略。请同时传入 sector_map。",
            RuntimeWarning,
        )

    if sector_map and sector_limits:
        asset_list = [str(a) for a in assets]
        for sector, limit in sector_limits.items():
            row = np.zeros(n, dtype=float)
            for i, asset in enumerate(asset_list):
                if sector_map.get(asset) == sector:
                    row[i] = 1.0
            if row.sum() > 0:
                constraints.append(LinearConstraint(row, -np.inf, float(limit)))

    if target_return is not None and expected_returns is not None:
        er = np.asarray(expected_returns, dtype=float)
        constraints.append(LinearConstraint(er, float(target_return), np.inf))

    return constraints


def _check_fallback_feasible(
    weights: np.ndarray,
    constraints: Sequence[Any],
    context: str = "优化器",
) -> None:
    """SLSQP 失败兜底前检查线性约束可行性。

    P2-Q15-fix (M107): 兜底权重（如 x0/可行初始权重）只满足个股上下限与满仓和，
    可能违反 sector_limits / target_return 等线性约束。若违反则抛
    OptimizationError，避免 BlackLitterman / MeanCVaR 等调用方静默产出违规组合。
    """
    for cons in constraints:
        if not isinstance(cons, LinearConstraint):
            continue  # 非线性约束（如做空 gross exposure）不在此检查。
        a = np.asarray(cons.A, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] != len(weights):
            continue  # MeanCVaR 增广变量约束（z/u）跳过。
        val = a @ weights
        if np.any(val < cons.lb - 1e-9) or np.any(val > cons.ub + 1e-9):
            raise OptimizationError(
                f"{context} 求解失败，兜底权重违反线性约束"
                f"（A·w={val.tolist()} vs lb={cons.lb}, ub={cons.ub}），无法返回合规组合。"
            )


def _solve_weight_optimizer(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    assets: Sequence[str],
    objective: ObjectiveName = "max_sharpe",
    risk_free: float = 0.0,
    risk_aversion: float = 2.5,
    bounds: WeightBounds = None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    full_investment: bool = True,
    sector_map: Optional[Mapping[str, str]] = None,
    sector_limits: Optional[Mapping[str, float]] = None,
    target_return: Optional[float] = None,
    max_iter: int = 1000,
) -> tuple[pd.Series, dict[str, Any]]:
    """通用均值方差权重求解器。

    支持三个目标：
    - min_variance: min w' Sigma w
    - max_sharpe: max (w'mu - rf) / sqrt(w'Sigmaw)
    - mean_variance: max w'mu - lambda/2 * w'Sigmaw

    数学出处：Markowitz (1952) 均值方差框架。
    """
    names = [str(a) for a in assets]
    mu = np.asarray(expected_returns, dtype=float)
    cov = _regularize_cov(cov_matrix)
    n = len(names)

    if n == 1:
        # P2-Q15-fix (L111): 单资产也必须尊重权重边界。
        # 旧实现直接返回 1.0，实测 max_weight=0.5 时仍输出 1.0。
        lower, upper = _infer_bounds(names, bounds=bounds, min_weight=min_weight, max_weight=max_weight)
        w_single = float(np.clip(1.0, lower[0], upper[0]))
        return _series_weights(np.array([w_single]), names), {"success": True, "message": "single asset"}

    lower, upper = _infer_bounds(names, bounds=bounds, min_weight=min_weight, max_weight=max_weight)
    x0 = _feasible_initial_weights(lower, upper, target_sum=1.0 if full_investment else None)
    scipy_bounds = Bounds(lower, upper)
    constraints = _build_linear_constraints(
        names,
        full_investment=full_investment,
        target_sum=1.0,
        sector_map=sector_map,
        sector_limits=sector_limits,
        target_return=target_return,
        expected_returns=mu,
    )

    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = math.sqrt(max(variance(w), 0.0))
        if vol <= 1e-12:
            return 1e6
        ret = float(w @ mu)
        return -((ret - risk_free) / vol)

    def neg_mean_variance(w: np.ndarray) -> float:
        ret = float(w @ mu)
        var = variance(w)
        return -(ret - 0.5 * risk_aversion * var)

    if objective == "min_variance":
        obj: Callable[[np.ndarray], float] = variance
    elif objective == "mean_variance":
        obj = neg_mean_variance
    elif objective == "max_sharpe":
        obj = neg_sharpe
    else:
        raise ValueError(f"未知 objective: {objective}")

    result = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": 1e-10, "disp": False},
    )

    if result.success and np.all(np.isfinite(result.x)):
        weights = result.x.astype(float)
        return _series_weights(weights, names), {
            "success": True,
            "message": str(result.message),
            "objective_value": float(result.fun),
            "n_iter": int(getattr(result, "nit", 0)),
        }

    # 兜底：如果求解器失败但约束是 long-only 满仓，则使用可行初始权重。
    fallback = x0.astype(float)

    # P2-Q15-fix (M107): 兜底前检查 sector_limits / target_return 等线性约束可行性。
    # 旧实现直接返回 x0，BlackLitterman 等调用方不检查 success=False，静默产出违规组合。
    _check_fallback_feasible(fallback, constraints, context="均值方差优化器")

    return _series_weights(fallback, names), {
        "success": False,
        "message": str(getattr(result, "message", "optimizer failed")),
        "objective_value": float(getattr(result, "fun", np.nan)),
        "fallback": "feasible_initial_weights",
    }


def _inverse_variance_weights(cov: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """逆方差组合权重。

    Q15 修复：零方差资产（diag < 1e-9，通常由全 NaN/停牌数据经
    ``_as_returns_frame`` 填 0 产生）直接赋 0 权重，避免 1/0 把
    权重全部集中到无数据标的上。
    """
    arr = _regularize_cov(cov)
    diag = np.diag(arr)
    zero_mask = diag < 1e-9
    if zero_mask.any():
        warnings.warn(
            "[portfolio_v2] 检测到零方差资产（可能为全 NaN/停牌数据），"
            "逆方差权重置 0。",
            RuntimeWarning,
        )
    inv_diag = np.where(zero_mask, 0.0, 1.0 / np.maximum(diag, 1e-9))
    total = inv_diag.sum()
    if total <= 1e-16:
        # 全部零方差：退化为等权，避免除零。
        n = len(diag)
        return np.full(n, 1.0 / n) if n > 0 else np.array([])
    return inv_diag / total


def _cluster_variance(cov: pd.DataFrame, cluster_assets: Sequence[str]) -> float:
    """计算一个簇的逆方差组合方差。"""
    sub_cov = cov.loc[list(cluster_assets), list(cluster_assets)]
    weights = _inverse_variance_weights(sub_cov.values)
    return float(weights @ sub_cov.values @ weights)


def _make_random_returns(
    n_obs: int = 504,
    n_assets: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """demo 使用的随机收益率数据。"""
    rng = np.random.default_rng(seed)
    base_corr = 0.25
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    vol = rng.uniform(0.12, 0.28, size=n_assets) / math.sqrt(252)
    cov = np.outer(vol, vol) * corr
    mean = rng.uniform(0.04, 0.16, size=n_assets) / 252
    data = rng.multivariate_normal(mean, cov, size=n_obs)
    idx = pd.bdate_range("2023-01-02", periods=n_obs)
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# 基类
# ══════════════════════════════════════════════════════════════════════════════


class BasePortfolioModel:
    """组合构建模型基类。

    子类统一暴露 optimize(returns, **kwargs) -> pd.Series。
    本基类只提供输入清洗和 metadata 辅助，不强制继承复杂框架。
    """

    method_name: str = "base"

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = int(periods_per_year)
        self.metadata_: OptimizationMetadata = OptimizationMetadata(method=self.method_name)
        self.diagnostics_: dict[str, Any] = {}

    def _prepare_returns(self, returns: ArrayLike2D, assets: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """清洗收益率输入。"""
        frame = _as_returns_frame(returns, assets=assets)
        return frame

    def optimize(self, returns: ArrayLike2D, **kwargs: Any) -> pd.Series:
        """子类必须实现的优化接口。"""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 1. Black-Litterman
# ══════════════════════════════════════════════════════════════════════════════


class BlackLitterman(BasePortfolioModel):
    """Black-Litterman 组合构建器。

    D6收敛: 同名异签名保留 —— 类接口(optimize(returns, views=...)), 与
    portfolio_optimizer.black_litterman 函数接口(views=[{tickers,bullish,confidence}]) 异实现。

    V4.1 feature: portfolio_v2

    核心公式，参考 Black & Litterman (1992)：

        Pi = delta * Sigma * w_mkt

    其中 Pi 是市场均衡超额收益，delta 是风险厌恶系数，Sigma 是协方差矩阵，
    w_mkt 是市场组合权重。

    主观观点的 Bayesian 更新：

        mu_bl = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
                * [(tau Sigma)^-1 Pi + P' Omega^-1 Q]

        Sigma_bl = Sigma + [(tau Sigma)^-1 + P' Omega^-1 P]^-1

    参数说明：
    - tau: 先验不确定性标量，常用 0.025 到 0.05。
    - P: 观点暴露矩阵，每行代表一个观点。
    - Q: 观点收益向量。
    - Omega: 观点误差协方差矩阵，confidence 越高，Omega 越小。

    用法示例：

    >>> returns = _make_random_returns()
    >>> views = [
    ...     {"type": "absolute", "assets": {"Asset_1": 1.0}, "view": 0.12, "confidence": 0.70},
    ...     {"type": "relative", "assets": {"Asset_2": 1.0, "Asset_3": -1.0}, "view": 0.03, "confidence": 0.60},
    ... ]
    >>> model = BlackLitterman(tau=0.05)
    >>> weights = model.optimize(returns, views=views)
    >>> model.posterior_returns_
    >>> model.posterior_covariance_
    """

    method_name = "black_litterman"

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        default_confidence: float = 0.5,
        periods_per_year: int = 252,
        shrinkage: float = 0.05,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau 应在 (0, 1]，常用区间为 0.025 到 0.05。")
        if risk_aversion <= 0:
            raise ValueError("risk_aversion 必须为正。")
        if not 0.0 < default_confidence < 1.0:
            raise ValueError("default_confidence 必须在 (0, 1)。")
        self.tau = float(tau)
        self.risk_aversion = float(risk_aversion)
        self.default_confidence = float(default_confidence)
        self.shrinkage = float(shrinkage)

        self.prior_returns_: pd.Series = pd.Series(dtype=float)
        self.posterior_returns_: pd.Series = pd.Series(dtype=float)
        self.posterior_covariance_: pd.DataFrame = pd.DataFrame(dtype=float)
        self.market_weights_: pd.Series = pd.Series(dtype=float)
        self.view_matrix_: pd.DataFrame = pd.DataFrame(dtype=float)
        self.view_returns_: pd.Series = pd.Series(dtype=float)
        self.omega_: pd.DataFrame = pd.DataFrame(dtype=float)

    def equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: Optional[ArrayLike1D] = None,
    ) -> pd.Series:
        """CAPM 逆向优化得到市场均衡收益 Pi。

        公式出处：Black-Litterman 模型的 reverse optimization。

            Pi = delta * Sigma * w_mkt

        这里的 Pi 是年化超额收益。如果用户希望加入无风险利率，可在外部加回。
        """
        assets = cov_matrix.index
        if market_weights is None:
            w_mkt = pd.Series(1.0 / len(assets), index=assets, dtype=float)
        else:
            w_mkt = _series_from_any(market_weights, assets, name="market_weight")
            total = float(w_mkt.sum())
            if abs(total) <= 1e-16:
                raise ValueError("market_weights 之和不能为 0。")
            w_mkt = w_mkt / total

        pi = self.risk_aversion * cov_matrix.values @ w_mkt.values
        self.market_weights_ = w_mkt.rename("market_weight")
        return pd.Series(pi, index=assets, name="prior_return", dtype=float)

    def _parse_view_assets(
        self,
        spec: Any,
        assets: Sequence[str],
        view_type: str = "absolute",
    ) -> np.ndarray:
        """把观点中的 assets 字段解析为 P 矩阵的一行。"""
        names = [str(a) for a in assets]
        pos = {asset: i for i, asset in enumerate(names)}
        row = np.zeros(len(names), dtype=float)

        if isinstance(spec, Mapping):
            for asset, exposure in spec.items():
                asset_str = str(asset)
                if asset_str not in pos:
                    raise ValueError(f"观点资产 {asset_str} 不在 returns 列中。")
                row[pos[asset_str]] = float(exposure)
            return row

        if isinstance(spec, str):
            if spec in pos:
                row[pos[spec]] = 1.0
                return row
            # P1-Q15-fix: 支持文档宣称的紧凑相对观点语法 {'A-B': {...}} / {'A>B': {...}}。
            # 原先这里直接 ValueError，导致 _parse_compact_view_key 成为死代码从未被调用；
            # 现在委托它解析含 -/ > 的键，解析失败会抛出清晰错误。
            return self._parse_compact_view_key(spec, assets)

        if isinstance(spec, Sequence):
            items = [str(x) for x in spec]
            for item in items:
                if item not in pos:
                    raise ValueError(f"观点资产 {item} 不在 returns 列中。")
            if view_type == "relative" and len(items) == 2:
                row[pos[items[0]]] = 1.0
                row[pos[items[1]]] = -1.0
            else:
                # 篮子 absolute view 默认等权暴露。
                for item in items:
                    row[pos[item]] = 1.0 / len(items)
            return row

        raise ValueError("观点 assets 字段必须是 str、list 或 dict。")

    def _parse_compact_view_key(self, key: str, assets: Sequence[str]) -> np.ndarray:
        """解析 {'A-B': {'view': 0.03}} 这种紧凑 relative view。"""
        names = [str(a) for a in assets]
        pos = {asset: i for i, asset in enumerate(names)}
        row = np.zeros(len(names), dtype=float)

        if ">" in key:
            left, right = [part.strip() for part in key.split(">", 1)]
            if left not in pos or right not in pos:
                raise ValueError(f"观点 {key} 包含未知资产。")
            row[pos[left]] = 1.0
            row[pos[right]] = -1.0
            return row

        if "-" in key and key not in pos:
            left, right = [part.strip() for part in key.split("-", 1)]
            if left in pos and right in pos:
                row[pos[left]] = 1.0
                row[pos[right]] = -1.0
                return row

        if key not in pos:
            raise ValueError(f"观点资产 {key} 不在 returns 列中。")
        row[pos[key]] = 1.0
        return row

    def _parse_views(
        self,
        views: Optional[Any],
        assets: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """把用户观点解析为 P、Q、confidence。

        支持输入形式：

        1. list[dict]
           {"type": "absolute", "assets": {"A": 1}, "view": 0.10, "confidence": 0.7}
           {"type": "relative", "assets": {"A": 1, "B": -1}, "view": 0.03}

        2. dict 紧凑形式
           {"A": 0.10, "A-B": {"view": 0.03, "confidence": 0.6}}

        3. dict 包含 views 键
           {"views": [...]}。
        """
        if views is None:
            return (
                np.zeros((0, len(assets)), dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [],
            )

        raw_views: list[Any]
        if isinstance(views, Mapping) and "views" in views:
            value = views.get("views")
            if not isinstance(value, Sequence):
                raise ValueError("views['views'] 必须是列表。")
            raw_views = list(value)
        elif isinstance(views, Mapping):
            raw_views = []
            for key, value in views.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("assets", key)
                    item.setdefault("label", key)
                else:
                    item = {"assets": key, "view": float(value), "label": key}
                raw_views.append(item)
        elif isinstance(views, Sequence):
            raw_views = list(views)
        else:
            raise ValueError("views 必须是 dict 或 list。")

        p_rows: list[np.ndarray] = []
        q_values: list[float] = []
        confidences: list[float] = []
        labels: list[str] = []

        for i, item in enumerate(raw_views):
            if not isinstance(item, Mapping):
                raise ValueError("每个观点必须是 dict。")
            view_type = str(item.get("type", item.get("kind", "absolute"))).lower()
            view_value = item.get("view", item.get("value", item.get("return", None)))
            if view_value is None:
                raise ValueError("观点缺少 view/value/return 字段。")

            confidence = float(item.get("confidence", self.default_confidence))
            confidence = float(np.clip(confidence, 1e-4, 0.9999))

            if "assets" in item:
                row = self._parse_view_assets(item["assets"], assets, view_type=view_type)
            elif "long" in item and "short" in item:
                row = np.zeros(len(assets), dtype=float)
                row += self._parse_view_assets(item["long"], assets, view_type="absolute")
                row -= self._parse_view_assets(item["short"], assets, view_type="absolute")
            else:
                raise ValueError("观点缺少 assets 或 long/short 字段。")

            if np.all(np.abs(row) < 1e-16):
                raise ValueError("观点暴露行不能全为 0。")

            # relative view 保持多空差值，absolute view 允许单资产或篮子。
            p_rows.append(row)
            q_values.append(float(view_value))
            confidences.append(confidence)
            labels.append(str(item.get("label", f"view_{i}")))

        return np.vstack(p_rows), np.asarray(q_values, dtype=float), np.asarray(confidences, dtype=float), labels

    def _build_omega(
        self,
        p_matrix: np.ndarray,
        tau_cov: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """根据 confidence 构建观点误差矩阵 Omega。

        Idzorek confidence 思路的简化版本：

            omega_i = ((1 - c_i) / c_i) * P_i tau Sigma P_i'

        c_i 越接近 1，观点误差越小；c_i 越接近 0，观点误差越大。
        """
        if p_matrix.shape[0] == 0:
            return np.zeros((0, 0), dtype=float)

        diag_values: list[float] = []
        for i in range(p_matrix.shape[0]):
            row = p_matrix[i : i + 1]
            view_var = float(row @ tau_cov @ row.T)
            view_var = max(view_var, 1e-12)
            c = float(np.clip(confidences[i], 1e-4, 0.9999))
            omega_i = ((1.0 - c) / c) * view_var
            diag_values.append(max(omega_i, 1e-12))
        return np.diag(diag_values)

    def compute_posterior(
        self,
        returns: ArrayLike2D,
        views: Optional[Any] = None,
        market_weights: Optional[ArrayLike1D] = None,
        assets: Optional[Sequence[str]] = None,
        expected_returns: Optional[ArrayLike1D] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """计算 Black-Litterman 后验收益与后验协方差。

        如果 expected_returns 未传入，则先用 CAPM 逆向优化得到 Pi。
        如果 cov_matrix 未传入，则从 returns 估计年化协方差。
        """
        frame = self._prepare_returns(returns, assets=assets)
        asset_index = pd.Index(frame.columns)

        if cov_matrix is None:
            _, cov = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        else:
            if isinstance(cov_matrix, pd.DataFrame):
                cov = cov_matrix.reindex(index=asset_index, columns=asset_index).fillna(0.0).astype(float)
            else:
                cov_arr = _regularize_cov(np.asarray(cov_matrix, dtype=float))
                cov = pd.DataFrame(cov_arr, index=asset_index, columns=asset_index)

        cov_values = _regularize_cov(cov.values)
        cov = pd.DataFrame(cov_values, index=asset_index, columns=asset_index)

        if expected_returns is None:
            pi = self.equilibrium_returns(cov, market_weights=market_weights)
        else:
            pi = _series_from_any(expected_returns, asset_index, name="prior_return")
            if market_weights is None:
                self.market_weights_ = pd.Series(1.0 / len(asset_index), index=asset_index, name="market_weight")
            else:
                self.market_weights_ = _series_from_any(market_weights, asset_index, name="market_weight")
                self.market_weights_ = self.market_weights_ / self.market_weights_.sum()

        p_matrix, q_vector, confidences, labels = self._parse_views(views, asset_index)
        tau_cov = self.tau * cov_values
        omega = self._build_omega(p_matrix, tau_cov, confidences)

        if p_matrix.shape[0] == 0:
            posterior_mu = pi.values.copy()
            posterior_cov = cov_values + tau_cov
        else:
            tau_cov_inv = _safe_pinv(tau_cov)
            omega_inv = _safe_pinv(omega)
            middle = _safe_pinv(tau_cov_inv + p_matrix.T @ omega_inv @ p_matrix)
            posterior_mu = middle @ (tau_cov_inv @ pi.values + p_matrix.T @ omega_inv @ q_vector)
            posterior_cov = cov_values + middle

        posterior_cov = _regularize_cov(posterior_cov)
        self.prior_returns_ = pi.rename("prior_return")
        self.posterior_returns_ = pd.Series(posterior_mu, index=asset_index, name="posterior_return")
        self.posterior_covariance_ = pd.DataFrame(posterior_cov, index=asset_index, columns=asset_index)
        self.view_matrix_ = pd.DataFrame(p_matrix, index=labels, columns=asset_index)
        self.view_returns_ = pd.Series(q_vector, index=labels, name="view_return")
        self.omega_ = pd.DataFrame(omega, index=labels, columns=labels)

        self.diagnostics_ = {
            "tau": self.tau,
            "risk_aversion": self.risk_aversion,
            "n_views": int(p_matrix.shape[0]),
            "view_labels": labels,
            "confidence": confidences.tolist(),
        }
        return self.posterior_returns_, self.posterior_covariance_

    def optimize(
        self,
        returns: ArrayLike2D,
        views: Optional[Any] = None,
        market_weights: Optional[ArrayLike1D] = None,
        expected_returns: Optional[ArrayLike1D] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        objective: ObjectiveName = "max_sharpe",
        risk_free: float = 0.0,
        bounds: WeightBounds = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_map: Optional[Mapping[str, str]] = None,
        sector_limits: Optional[Mapping[str, float]] = None,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """优化 Black-Litterman 组合权重。"""
        frame = self._prepare_returns(returns, assets=assets)
        posterior_mu, posterior_cov = self.compute_posterior(
            frame,
            views=views,
            market_weights=market_weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
        )
        weights, solver_info = _solve_weight_optimizer(
            posterior_mu.values,
            posterior_cov.values,
            posterior_mu.index,
            objective=objective,
            risk_free=risk_free,
            risk_aversion=self.risk_aversion,
            bounds=bounds,
            min_weight=min_weight,
            max_weight=max_weight,
            full_investment=True,
            sector_map=sector_map,
            sector_limits=sector_limits,
        )
        # P2-Q15-fix (M107): 调用方检查 solver success——失败但兜底满足约束时
        # 仍可见告警，避免静默使用次优/兜底权重。
        if not solver_info.get("success", False):
            warnings.warn(
                f"[BlackLitterman] SLSQP 优化失败，使用满足约束的兜底权重："
                f"{solver_info.get('message', '')}",
                RuntimeWarning,
            )
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=bool(solver_info.get("success", False)),
            message=str(solver_info.get("message", "")),
            n_assets=frame.shape[1],
            n_observations=frame.shape[0],
            details={**self.diagnostics_, "solver": solver_info},
        )
        return weights


# ══════════════════════════════════════════════════════════════════════════════
# 2. Hierarchical Risk Parity
# ══════════════════════════════════════════════════════════════════════════════


class HierarchicalRiskParity(BasePortfolioModel):
    """Hierarchical Risk Parity 组合构建器。

    V4.1 feature: portfolio_v2

    参考：Lopez de Prado (2016), Building Diversified Portfolios that Outperform Out of Sample。

    三步流程：
    1. Tree clustering: 使用相关距离 d_ij = sqrt((1 - corr_ij) / 2)，single linkage 聚类。
    2. Quasi-diagonalization: 根据聚类树调整资产顺序，让相似资产相邻。
    3. Recursive bisection: 递归二分，每次按两个子簇的方差反比分配资金。

    HRP 不需要求逆完整协方差矩阵，因此在资产数多、样本短、相关矩阵病态时更稳。

    用法示例：

    >>> returns = _make_random_returns()
    >>> hrp = HierarchicalRiskParity()
    >>> weights = hrp.optimize(returns)
    >>> hrp.sorted_assets_
    """

    method_name = "hrp"

    def __init__(
        self,
        linkage_method: str = "single",
        periods_per_year: int = 252,
        shrinkage: float = 0.05,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        self.linkage_method = linkage_method
        self.shrinkage = float(shrinkage)
        self.linkage_matrix_: np.ndarray = np.empty((0, 4), dtype=float)
        self.sorted_assets_: list[str] = []
        self.correlation_: pd.DataFrame = pd.DataFrame(dtype=float)
        self.covariance_: pd.DataFrame = pd.DataFrame(dtype=float)

    def _quasi_diagonal_order(self, linkage_matrix: np.ndarray, n_items: int) -> list[int]:
        """根据层次聚类树返回准对角化资产顺序。

        scipy linkage 中，叶子节点编号为 [0, n_items-1]，合并节点编号从 n_items 开始。
        递归展开最后一个根节点即可得到 HRP 的排序。
        """
        if n_items == 1:
            return [0]

        def expand(node_id: int) -> list[int]:
            if node_id < n_items:
                return [node_id]
            row = linkage_matrix[node_id - n_items]
            left = int(row[0])
            right = int(row[1])
            return expand(left) + expand(right)

        root_id = 2 * n_items - 2
        return expand(root_id)

    def _recursive_bisection(self, cov_sorted: pd.DataFrame) -> pd.Series:
        """递归二分分配。

        对任意左右两个子簇，先计算每个子簇内部逆方差组合的簇方差：

            cluster_var = w_ivp' Sigma_cluster w_ivp

        然后按风险反比分配：

            alpha_left = 1 - var_left / (var_left + var_right)
            weight_left *= alpha_left
            weight_right *= 1 - alpha_left

        方差越大的子簇获得越低权重。
        """
        assets = list(cov_sorted.index)
        weights = pd.Series(1.0, index=assets, dtype=float)
        clusters: list[list[str]] = [assets]

        while clusters:
            next_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                split = len(cluster) // 2
                left = cluster[:split]
                right = cluster[split:]
                if not left or not right:
                    continue

                var_left = _cluster_variance(cov_sorted, left)
                var_right = _cluster_variance(cov_sorted, right)
                denom = var_left + var_right
                if denom <= 1e-16:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))

                weights.loc[left] *= alpha
                weights.loc[right] *= 1.0 - alpha
                next_clusters.extend([left, right])
            clusters = next_clusters

        total = float(weights.sum())
        if total <= 1e-16:
            return pd.Series(1.0 / len(assets), index=assets, dtype=float)
        return weights / total

    def optimize(
        self,
        returns: ArrayLike2D,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """优化 HRP 权重。"""
        frame = self._prepare_returns(returns, assets=assets)
        asset_index = pd.Index(frame.columns)

        if cov_matrix is None:
            _, cov = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        else:
            if isinstance(cov_matrix, pd.DataFrame):
                cov = cov_matrix.reindex(index=asset_index, columns=asset_index).fillna(0.0).astype(float)
            else:
                cov = pd.DataFrame(_regularize_cov(cov_matrix), index=asset_index, columns=asset_index)

        # Q15 修复：零方差资产（全 NaN/停牌数据）不得参与聚类与分配。
        # 若让其进入 linkage/递归二分，单叶节点会拿到非零权重，
        # 造成 100% 集中（原实现实测 HRP weights={Asset_2:1.0}）。
        cov_values_full = _regularize_cov(cov.values)
        diag_full = np.diag(cov_values_full)
        zero_var_mask = diag_full < 1e-9
        active_assets = [str(a) for a, m in zip(asset_index, zero_var_mask) if not m]

        if zero_var_mask.any():
            warnings.warn(
                f"[HRP] 检测到零方差资产（可能为全 NaN/停牌数据）："
                f"{list(asset_index[zero_var_mask])}，权重置 0。",
                RuntimeWarning,
            )

        n_active = len(active_assets)
        if n_active == 0:
            # 全部零方差：退化为等权（无任何收益信息时没有更优选择）。
            weights = pd.Series(1.0 / len(asset_index), index=asset_index, name="weight")
            self.correlation_ = pd.DataFrame(_cov_to_corr(cov_values_full), index=asset_index, columns=asset_index)
            self.covariance_ = pd.DataFrame(cov_values_full, index=asset_index, columns=asset_index)
            self.sorted_assets_ = list(asset_index)
            self.metadata_ = OptimizationMetadata(
                method=self.method_name,
                success=True,
                message="HRP completed (all assets zero-variance → equal weight)",
                n_assets=frame.shape[1],
                n_observations=frame.shape[0],
            )
            return weights

        if n_active == 1:
            weights = pd.Series(0.0, index=asset_index, dtype=float)
            weights.loc[active_assets[0]] = 1.0
            weights = weights.rename("weight")
            self.correlation_ = pd.DataFrame(_cov_to_corr(cov_values_full), index=asset_index, columns=asset_index)
            self.covariance_ = pd.DataFrame(cov_values_full, index=asset_index, columns=asset_index)
            self.sorted_assets_ = active_assets
            self.metadata_ = OptimizationMetadata(
                method=self.method_name,
                success=True,
                message="HRP completed (single active asset)",
                n_assets=frame.shape[1],
                n_observations=frame.shape[0],
            )
            return weights

        cov_values = cov_values_full[np.ix_(~zero_var_mask, ~zero_var_mask)]
        corr = _cov_to_corr(cov_values)
        dist = _corr_to_distance(corr)

        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        order = self._quasi_diagonal_order(link, n_active)
        sorted_assets = [active_assets[i] for i in order]

        cov_sorted = pd.DataFrame(cov_values, index=active_assets, columns=active_assets).loc[sorted_assets, sorted_assets]
        weights_sorted = self._recursive_bisection(cov_sorted)
        weights = weights_sorted.reindex(asset_index).fillna(0.0).rename("weight")

        self.linkage_matrix_ = link
        self.sorted_assets_ = sorted_assets
        self.correlation_ = pd.DataFrame(_cov_to_corr(cov_values_full), index=asset_index, columns=asset_index)
        self.covariance_ = pd.DataFrame(cov_values_full, index=asset_index, columns=asset_index)
        self.diagnostics_ = {
            "linkage_method": self.linkage_method,
            "sorted_assets": sorted_assets,
        }
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=True,
            message="HRP completed",
            n_assets=frame.shape[1],
            n_observations=frame.shape[0],
            details=self.diagnostics_,
        )
        return weights


# ══════════════════════════════════════════════════════════════════════════════
# 3. Nested Clustered Optimization
# ══════════════════════════════════════════════════════════════════════════════


class NestedClusteredOptimization(BasePortfolioModel):
    """Nested Clustered Optimization 组合构建器。

    V4.1 feature: portfolio_v2

    参考：Lopez de Prado (2019), A Robust Estimator of the Efficient Frontier。

    NCO 的核心动机：
    - 直接对完整协方差矩阵做均值方差优化，会放大均值和协方差估计误差。
    - 先把高相关资产聚类，在簇内优化，再把每个簇当成一个组合做簇间优化。
    - 这样把一个高维优化问题拆成多个低维问题，降低估计误差的影响。

    本实现包含蒙特卡洛稳定步骤：
    - 先从样本收益估计均值和协方差。
    - 多次模拟收益矩阵或 bootstrap 重采样。
    - 每次执行 K-means 聚类、簇内优化、簇间优化。
    - 最终对多次权重取平均并归一化。

    用法示例：

    >>> returns = _make_random_returns(n_assets=10)
    >>> nco = NestedClusteredOptimization(n_clusters=3, n_monte_carlo=32)
    >>> weights = nco.optimize(returns, objective="min_variance")
    >>> nco.cluster_labels_
    """

    method_name = "nco"

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_monte_carlo: int = 64,
        sample_length: Optional[int] = None,
        objective: ObjectiveName = "min_variance",
        random_state: int = 42,
        periods_per_year: int = 252,
        shrinkage: float = 0.05,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        self.n_clusters = n_clusters
        self.n_monte_carlo = int(n_monte_carlo)
        self.sample_length = sample_length
        self.objective = objective
        self.random_state = int(random_state)
        self.shrinkage = float(shrinkage)

        self.cluster_labels_: pd.Series = pd.Series(dtype=int)
        self.cluster_weights_: pd.Series = pd.Series(dtype=float)
        self.intra_cluster_weights_: dict[int, pd.Series] = {}
        self.monte_carlo_weights_: pd.DataFrame = pd.DataFrame(dtype=float)

    def _choose_n_clusters(self, n_assets: int) -> int:
        """默认簇数量：sqrt(N)，并限制在 [2, N]。"""
        if n_assets <= 2:
            return n_assets
        if self.n_clusters is not None:
            return int(np.clip(self.n_clusters, 2, n_assets))
        return int(np.clip(round(math.sqrt(n_assets)), 2, n_assets))

    def _fallback_kmeans(
        self,
        features: np.ndarray,
        n_clusters: int,
        rng: np.random.Generator,
        max_iter: int = 100,
    ) -> np.ndarray:
        """sklearn 不可用时的轻量 K-means。

        该实现只用于资产聚类 fallback，不追求 sklearn 的所有边界特性。
        """
        n_samples = features.shape[0]
        if n_clusters >= n_samples:
            return np.arange(n_samples, dtype=int)

        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = features[init_idx].copy()
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    centers[k] = features[mask].mean(axis=0)
                else:
                    centers[k] = features[rng.integers(0, n_samples)]
        return labels.astype(int)

    def _cluster_assets(
        self,
        cov_matrix: pd.DataFrame,
        n_clusters: int,
        rng: np.random.Generator,
    ) -> pd.Series:
        """使用相关结构对资产做 K-means 聚类。"""
        assets = pd.Index(cov_matrix.index)
        n_assets = len(assets)
        if n_assets == 1:
            return pd.Series([0], index=assets, dtype=int)
        if n_clusters >= n_assets:
            return pd.Series(np.arange(n_assets), index=assets, dtype=int)

        corr = _cov_to_corr(cov_matrix.values)
        # 每个资产用它与其他资产的相关向量作为特征。
        # 这样高度相关资产在特征空间中更接近。
        features = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        if _SklearnKMeans is not None:
            model = _SklearnKMeans(n_clusters=n_clusters, n_init=10, random_state=int(rng.integers(0, 2**31 - 1)))
            labels = model.fit_predict(features).astype(int)
        else:
            labels = self._fallback_kmeans(features, n_clusters=n_clusters, rng=rng)

        # 防御：如果出现空簇或只有一个簇，退化为按资产顺序分簇。
        unique = np.unique(labels)
        if len(unique) < min(n_clusters, n_assets):
            labels = np.arange(n_assets) % n_clusters
        return pd.Series(labels, index=assets, dtype=int)

    def _optimize_subportfolio(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        objective: ObjectiveName,
        min_weight: float,
        max_weight: float,
    ) -> pd.Series:
        """对一个子组合执行簇内或簇间优化。

        P1-Q15-fix: 透传用户 min/max_weight（原调用方硬编码 [0, 1] 忽略用户边界）。
        子组合要求满仓（权重和=1），因此把边界夹取到可行区间：
          max_weight 至少放大到 1/n（等权满仓阈值），min_weight 至多压到 1/n，
        保证小簇不会因用户边界不可行而崩溃；最终 clip 阶段仍执行用户真实边界。
        """
        n = len(mu)
        if n == 1:
            # 单资产簇：簇内权重 1.0，最终 clip 阶段会按用户 max_weight 收紧。
            return pd.Series([1.0], index=mu.index, name="weight")
        lo = min(float(min_weight), 1.0 / n)
        hi = max(float(max_weight), 1.0 / n)
        try:
            weights, _ = _solve_weight_optimizer(
                mu.values,
                cov.values,
                mu.index,
                objective=objective,
                bounds=None,
                min_weight=lo,
                max_weight=hi,
                full_investment=True,
            )
        except OptimizationError as exc:
            # 极端不可行（如 min_weight 过大）兜底，退化为 [0, 1]，保持可见告警。
            warnings.warn(
                f"[NCO] 子组合边界不可行（min_weight={min_weight}, max_weight={max_weight}）：{exc}，"
                "退化到 [0, 1] 边界继续优化。",
                RuntimeWarning,
            )
            weights, _ = _solve_weight_optimizer(
                mu.values,
                cov.values,
                mu.index,
                objective=objective,
                bounds=None,
                min_weight=0.0,
                max_weight=1.0,
                full_investment=True,
            )
        return weights

    def _nco_once(
        self,
        returns: pd.DataFrame,
        objective: ObjectiveName,
        n_clusters: int,
        rng: np.random.Generator,
        min_weight: float,
        max_weight: float,
    ) -> tuple[pd.Series, pd.Series, dict[int, pd.Series], pd.Series]:
        """执行一次 NCO：聚类、簇内优化、簇间优化。"""
        mu, cov = _annualized_mean_cov(returns, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        # Q15 修复：零方差资产（全 NaN/停牌数据）剔除出聚类与优化流程，
        # 权重置 0，避免逆方差路径把 100% 权重集中到无数据标的。
        diag = np.diag(_regularize_cov(cov.values))
        zero_mask = diag < 1e-9
        if zero_mask.any():
            zero_cols = list(cov.columns[zero_mask])
            warnings.warn(
                f"[NCO] 检测到零方差资产（可能为全 NaN/停牌数据）：{zero_cols}，权重置 0。",
                RuntimeWarning,
            )
            returns = returns.drop(columns=zero_cols)
            if returns.shape[1] == 0:
                # 全部零方差：退化为等权。
                n = len(cov.columns)
                eq = pd.Series(1.0 / n, index=cov.columns, name="weight")
                labels = pd.Series(0, index=cov.columns, dtype=int)
                return eq, labels, {}, pd.Series(dtype=float)
            mu, cov = _annualized_mean_cov(returns, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        labels = self._cluster_assets(cov, n_clusters=n_clusters, rng=rng)

        intra_weights: dict[int, pd.Series] = {}
        cluster_return_frame = pd.DataFrame(index=returns.index)
        cluster_asset_lists: dict[int, list[str]] = {}

        for cluster_id in sorted(labels.unique()):
            assets = labels.index[labels == cluster_id].tolist()
            cluster_asset_lists[int(cluster_id)] = assets
            sub_mu = mu.loc[assets]
            sub_cov = cov.loc[assets, assets]
            # P1-Q15-fix: 簇内优化透传用户 min/max_weight（原来硬编码 [0, 1] 忽略用户边界）。
            # 小簇不可行由 _optimize_subportfolio 内部夹取到等权满仓阈值并兜底处理。
            sub_w = self._optimize_subportfolio(sub_mu, sub_cov, objective, min_weight=min_weight, max_weight=max_weight)
            intra_weights[int(cluster_id)] = sub_w
            cluster_return_frame[int(cluster_id)] = returns[assets] @ sub_w.reindex(assets).values

        cluster_mu, cluster_cov = _annualized_mean_cov(
            cluster_return_frame,
            periods_per_year=self.periods_per_year,
            shrinkage=min(self.shrinkage, 0.25),
        )
        # P1-Q15-fix: 簇间优化同样透传用户 min/max_weight，而非硬编码 [0, 1]。
        cluster_weights = self._optimize_subportfolio(
            cluster_mu,
            cluster_cov,
            objective=objective,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        final = pd.Series(0.0, index=returns.columns, name="weight", dtype=float)
        for cluster_id, sub_w in intra_weights.items():
            # 通用权重封装会把资产名标准化为字符串；簇编号可能因此从 int 变成 str。
            # 这里兼容两种索引，避免 NCO 回填簇权重时因为类型差异失败。
            if cluster_id in cluster_weights.index:
                cw = float(cluster_weights.loc[cluster_id])
            else:
                cw = float(cluster_weights.loc[str(cluster_id)])
            final.loc[sub_w.index] = cw * sub_w

        final = final / final.sum() if final.sum() > 1e-16 else pd.Series(1.0 / returns.shape[1], index=returns.columns)
        final = final.clip(lower=min_weight, upper=max_weight)
        # P1-Q15-fix: clip 后不再归一化。旧逻辑在 clip 后再次 /sum，会把被 max_weight
        # 压住的权重重新放大超限（实测 max_weight=0.3 时最差 0.8493）。
        # 现在 clip 后若 sum<1 保留现金；若 sum>1（min_weight>0 抬升导致）也保持边界内。
        _post_sum = float(final.sum())
        if abs(_post_sum - 1.0) > 1e-6:
            warnings.warn(
                f"[NCO] 权重 clip 到 [{min_weight}, {max_weight}] 后总和为 {_post_sum:.6f}"
                "（≠1），保留现金或保持约束内权重，min/max_weight 已满足。",
                RuntimeWarning,
            )
        return final.rename("weight"), labels, intra_weights, cluster_weights

    def _simulate_returns(
        self,
        base_returns: pd.DataFrame,
        rng: np.random.Generator,
        mode: Literal["gaussian", "bootstrap"] = "gaussian",
    ) -> pd.DataFrame:
        """蒙特卡洛生成一组收益率样本。"""
        sample_length = int(self.sample_length or len(base_returns))
        sample_length = max(sample_length, min(len(base_returns), 30))
        assets = base_returns.columns

        if mode == "bootstrap":
            idx = rng.integers(0, len(base_returns), size=sample_length)
            arr = base_returns.iloc[idx].values
            return pd.DataFrame(arr, columns=assets)

        mean = base_returns.mean().values
        cov = _regularize_cov(base_returns.cov().values)
        arr = rng.multivariate_normal(mean, cov, size=sample_length)
        return pd.DataFrame(arr, columns=assets)

    def optimize(
        self,
        returns: ArrayLike2D,
        objective: Optional[ObjectiveName] = None,
        n_clusters: Optional[int] = None,
        n_monte_carlo: Optional[int] = None,
        monte_carlo_mode: Literal["gaussian", "bootstrap"] = "gaussian",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """优化 NCO 权重。"""
        frame = self._prepare_returns(returns, assets=assets)
        obj = objective or self.objective
        full_columns = list(frame.columns)

        # Q15 修复：零方差资产（全 NaN/停牌数据）在**原始样本**上一次性剔除，
        # 防止蒙特卡洛模拟把零方差列重生成微小噪声而绕过簇内零方差保护，
        # 导致权重重新集中到无数据标的。
        _, cov0 = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        diag0 = np.diag(_regularize_cov(cov0.values))
        zero_cols = list(frame.columns[diag0 < 1e-9])
        if zero_cols:
            warnings.warn(
                f"[NCO] 检测到零方差资产（可能为全 NaN/停牌数据）：{zero_cols}，权重置 0。",
                RuntimeWarning,
            )
            frame = frame.drop(columns=zero_cols)
            if frame.shape[1] == 0:
                # 全部零方差：退化为等权。
                eq = pd.Series(1.0 / len(full_columns), index=full_columns, name="weight")
                self.cluster_labels_ = pd.Series(0, index=full_columns, dtype=int)
                self.cluster_weights_ = pd.Series(dtype=float)
                self.intra_cluster_weights_ = {}
                self.monte_carlo_weights_ = pd.DataFrame(eq).T
                self.diagnostics_ = {
                    "n_clusters": 0,
                    "n_monte_carlo_requested": 0,
                    "n_monte_carlo_successful": 0,
                    "objective": obj,
                    "monte_carlo_mode": monte_carlo_mode,
                    "sklearn_available": _SklearnKMeans is not None,
                    "note": "all assets zero-variance -> equal weight",
                }
                self.metadata_ = OptimizationMetadata(
                    method=self.method_name,
                    success=True,
                    message="NCO completed (all assets zero-variance -> equal weight)",
                    n_assets=len(full_columns),
                    n_observations=frame.shape[0],
                    details=self.diagnostics_,
                )
                return eq

        n_assets = frame.shape[1]
        k = int(n_clusters or self._choose_n_clusters(n_assets))
        mc = self.n_monte_carlo if n_monte_carlo is None else int(n_monte_carlo)
        mc = max(mc, 0)
        rng = np.random.default_rng(self.random_state)

        all_weights: list[pd.Series] = []
        last_labels = pd.Series(dtype=int)
        last_intra: dict[int, pd.Series] = {}
        last_cluster_weights = pd.Series(dtype=float)

        # 第一轮使用原始样本，保证结果与输入直接相关。
        w0, labels, intra, cluster_w = self._nco_once(
            frame,
            objective=obj,
            n_clusters=k,
            rng=rng,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        all_weights.append(w0)
        last_labels = labels
        last_intra = intra
        last_cluster_weights = cluster_w

        for _i in range(mc):
            try:
                sim = self._simulate_returns(frame, rng=rng, mode=monte_carlo_mode)
                wi, labels_i, intra_i, cluster_w_i = self._nco_once(
                    sim,
                    objective=obj,
                    n_clusters=k,
                    rng=rng,
                    min_weight=min_weight,
                    max_weight=max_weight,
                )
                # 零方差资产可能被 _nco_once 剔除，用 reindex 回填 0 而非直接改 index。
                wi = wi.reindex(frame.columns).fillna(0.0)
                all_weights.append(wi)
                last_labels = labels_i
                last_intra = intra_i
                last_cluster_weights = cluster_w_i
            except Exception as exc:  # pragma: no cover - 随机模拟异常兜底。
                warnings.warn(f"NCO Monte Carlo run failed: {exc}", RuntimeWarning)
                continue

        weights_frame = pd.concat(all_weights, axis=1).T.reindex(columns=frame.columns).fillna(0.0)
        avg_weights = weights_frame.mean(axis=0)
        avg_weights = avg_weights.clip(lower=min_weight, upper=max_weight)
        # P1-Q15-fix: 与 _nco_once 一致——clip 后不再归一化，避免 max_weight 被重新放大。
        # 理论上全部权重≈0 时用等权兜底，且同样约束到边界内。
        if avg_weights.sum() <= 1e-16:
            avg_weights = pd.Series(1.0 / n_assets, index=frame.columns).clip(lower=min_weight, upper=max_weight)

        # Q15 修复：把被剔除的零方差资产回填 0 权重，还原完整资产列。
        avg_weights = avg_weights.reindex(full_columns).fillna(0.0)

        self.cluster_labels_ = last_labels.reindex(frame.columns).fillna(-1).astype(int)
        self.cluster_weights_ = last_cluster_weights.rename("cluster_weight")
        self.intra_cluster_weights_ = last_intra
        self.monte_carlo_weights_ = weights_frame
        self.diagnostics_ = {
            "n_clusters": k,
            "n_monte_carlo_requested": mc,
            "n_monte_carlo_successful": len(all_weights) - 1,
            "objective": obj,
            "monte_carlo_mode": monte_carlo_mode,
            "sklearn_available": _SklearnKMeans is not None,
            "zero_variance_dropped": zero_cols,
        }
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=True,
            message="NCO completed",
            n_assets=len(full_columns),
            n_observations=frame.shape[0],
            details=self.diagnostics_,
        )
        return avg_weights.rename("weight")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Kelly Criterion
# ══════════════════════════════════════════════════════════════════════════════


class KellyCriterion(BasePortfolioModel):
    """多资产 Kelly 仓位优化器。

    D6收敛: 异名/近名异口径保留 —— 多资产 f*=Σ^{-1}μ + fractional/杠杆约束, 与
    portfolio_risk.compute_kelly_ratio/compute_kelly_from_trades(单资产胜率盈亏比) 异口径。

    V4.1 feature: portfolio_v2

    多资产 Kelly 在连续近似下最大化对数财富增长：

        max_w  mu' w - 1/2 * w' Sigma w

    一阶条件给出无约束解：

        f* = Sigma^{-1} mu

    其中 mu 是相对无风险利率的期望超额收益，Sigma 是同一频率下的协方差矩阵。
    本实现用 scipy.optimize.minimize 加入以下现实约束：

    - fractional Kelly: w = fraction * f*，常用 fraction=0.1 到 0.25。
    - long_only: 是否禁止做空。
    - leverage_limit: 总杠杆约束，long-only 下近似为 sum(w) <= leverage_limit。
    - borrowing_limit: 允许超过 100% 净资产的借款额度。
    - max_weight: 单资产上限。

    full_investment 语义（P2-Q15-fix M105）：
    默认 full_investment=False，输出为 fraction 倍的"满 Kelly"权重，保留现金
    （sum(w) = fraction * sum(full_kelly)，通常 < 1）。
    full_investment=True 时，fraction 只影响相对配比，最终权重会归一化到满仓
    （sum=1，且受 leverage_limit / 个股边界约束；边界阻止满仓时可见告警并保留现金）。

    用法示例：

    >>> returns = _make_random_returns()
    >>> kelly = KellyCriterion(fraction=0.2, leverage_limit=1.2)
    >>> weights = kelly.optimize(returns, long_only=True)
    >>> weights.sum()  # 小于等于 1.2，可能保留现金
    """

    method_name = "kelly"

    def __init__(
        self,
        fraction: float = 0.25,
        leverage_limit: float = 1.0,
        borrowing_limit: float = 0.0,
        borrowing_rate: float = 0.0,
        periods_per_year: int = 252,
        shrinkage: float = 0.05,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        if fraction <= 0:
            raise ValueError("fraction 必须为正。")
        if leverage_limit <= 0:
            raise ValueError("leverage_limit 必须为正。")
        self.fraction = float(fraction)
        self.leverage_limit = float(leverage_limit)
        self.borrowing_limit = float(borrowing_limit)
        self.borrowing_rate = float(borrowing_rate)
        self.shrinkage = float(shrinkage)
        self.full_kelly_weights_: pd.Series = pd.Series(dtype=float)
        self.growth_rate_: float = 0.0
        self.cash_weight_: float = 0.0

    def _kelly_objective(
        self,
        w: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        borrowing_rate: float,
    ) -> float:
        """负的 Kelly 增长率，供 minimize 使用。"""
        invested = float(np.sum(w))
        borrow_cost = max(invested - 1.0, 0.0) * borrowing_rate
        growth = float(mu @ w - 0.5 * w @ cov @ w - borrow_cost)
        return -growth

    def optimize(
        self,
        returns: ArrayLike2D,
        expected_returns: Optional[ArrayLike1D] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        risk_free: float = 0.0,
        fraction: Optional[float] = None,
        long_only: bool = True,
        allow_short: Optional[bool] = None,
        leverage_limit: Optional[float] = None,
        borrowing_limit: Optional[float] = None,
        borrowing_rate: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: Optional[float] = None,
        full_investment: bool = False,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """优化多资产 fractional Kelly 权重。"""
        frame = self._prepare_returns(returns, assets=assets)
        asset_index = pd.Index(frame.columns)

        if allow_short is not None:
            long_only = not allow_short
        frac = float(fraction if fraction is not None else self.fraction)
        lev_limit = float(leverage_limit if leverage_limit is not None else self.leverage_limit)
        borrow_limit = float(borrowing_limit if borrowing_limit is not None else self.borrowing_limit)
        borrow_rate = float(borrowing_rate if borrowing_rate is not None else self.borrowing_rate)

        if expected_returns is None or cov_matrix is None:
            mu, cov = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        else:
            mu = _series_from_any(expected_returns, asset_index, name="expected_return")
            if isinstance(cov_matrix, pd.DataFrame):
                cov = cov_matrix.reindex(index=asset_index, columns=asset_index).fillna(0.0).astype(float)
            else:
                cov = pd.DataFrame(_regularize_cov(cov_matrix), index=asset_index, columns=asset_index)

        excess = mu.values - float(risk_free)
        cov_values = _regularize_cov(cov.values)
        n = len(asset_index)

        if min_weight is None:
            min_weight = 0.0 if long_only else -max_weight
        lower = np.full(n, float(min_weight), dtype=float)
        upper = np.full(n, float(max_weight), dtype=float)

        # 借贷限制：long-only 下 sum(w) <= 1 + borrowing_limit。
        # leverage_limit 是更通用的总暴露限制，取二者较小者。
        max_net = min(lev_limit, 1.0 + max(borrow_limit, 0.0))
        constraints: list[Any] = []
        if full_investment:
            constraints.append(LinearConstraint(np.ones(n), 1.0, 1.0))
        else:
            constraints.append(LinearConstraint(np.ones(n), -np.inf, max_net))

        if not long_only:
            # 做空时用 gross exposure 约束。SLSQP 不支持 abs 线性约束，使用非线性约束。
            constraints.append({"type": "ineq", "fun": lambda w: lev_limit - float(np.sum(np.abs(w)))})

        x0 = _feasible_initial_weights(lower, upper, target_sum=1.0 if full_investment else None)
        if not full_investment and long_only:
            x0 = np.minimum(np.full(n, min(1.0 / n, max_weight), dtype=float), upper)

        result = minimize(
            lambda w: self._kelly_objective(w, excess, cov_values, borrow_rate),
            x0,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )

        if result.success and np.all(np.isfinite(result.x)):
            full_kelly = result.x.astype(float)
            success = True
            message = str(result.message)
        else:
            # 解析解兜底：f* = Sigma^{-1} mu，再裁剪到限制内。
            full_kelly = _safe_pinv(cov_values) @ excess
            full_kelly = np.clip(full_kelly, lower, upper)
            if long_only:
                full_kelly = np.maximum(full_kelly, 0.0)
            gross = float(np.sum(np.abs(full_kelly)))
            if gross > lev_limit and gross > 1e-16:
                full_kelly = full_kelly / gross * lev_limit
            success = False
            message = str(getattr(result, "message", "Kelly optimizer failed; used pseudo-inverse fallback"))

        fractional = frac * full_kelly
        if long_only:
            fractional = np.maximum(fractional, 0.0)
        gross_fractional = float(np.sum(np.abs(fractional)))
        if gross_fractional > lev_limit and gross_fractional > 1e-16:
            fractional = fractional / gross_fractional * lev_limit
        if long_only and fractional.sum() > max_net and fractional.sum() > 1e-16:
            fractional = fractional / fractional.sum() * max_net

        # P2-Q15-fix (M105): full_investment 的语义是"最终 fractional 权重满仓(sum=1)"。
        # 旧实现把 sum=1 约束加在"满 Kelly"上，随后 frac 缩放使输出权重和 = fraction
        # （实测 fraction=0.25 + full_investment=True → 权重和 0.25、现金 0.75）。
        # 现在对最终权重归一化到 min(1, max_net)：
        #   - 正常情况（max_net>=1）满仓 sum=1；
        #   - 若杠杆上限 <1 与满仓冲突，尊重杠杆并保留现金，且可见告警；
        #   - 归一化后重新 clip 到个股边界，若边界阻止满仓则可见告警。
        if full_investment:
            target_sum = min(1.0, max_net)
            if target_sum < 1.0:
                warnings.warn(
                    f"[Kelly] full_investment=True 与杠杆上限 max_net={max_net:.4f}<1 冲突，"
                    "按杠杆上限执行（保留现金）。",
                    RuntimeWarning,
                )
            s = float(fractional.sum())
            if s > 1e-16 and target_sum > 0:
                fractional = fractional / s * target_sum
            fractional = np.clip(fractional, lower, upper)
            post_sum = float(fractional.sum())
            if abs(post_sum - target_sum) > 1e-6:
                warnings.warn(
                    f"[Kelly] full_investment=True 时权重边界 [{float(min_weight)}, {float(max_weight)}] "
                    f"使最终权重和 = {post_sum:.6f}（≠{target_sum:.4f}），保留现金/边界内权重。",
                    RuntimeWarning,
                )

        weights = _series_weights(fractional, asset_index)
        self.full_kelly_weights_ = _series_weights(full_kelly, asset_index, name="full_kelly_weight")
        self.growth_rate_ = -float(self._kelly_objective(full_kelly, excess, cov_values, borrow_rate))
        self.cash_weight_ = float(1.0 - weights.sum())
        self.diagnostics_ = {
            "fraction": frac,
            "leverage_limit": lev_limit,
            "borrowing_limit": borrow_limit,
            "borrowing_rate": borrow_rate,
            "long_only": long_only,
            "full_investment": full_investment,
            "cash_weight": self.cash_weight_,
            "full_kelly_growth_rate": self.growth_rate_,
        }
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=success,
            message=message,
            n_assets=frame.shape[1],
            n_observations=frame.shape[0],
            details=self.diagnostics_,
        )
        return weights


# ══════════════════════════════════════════════════════════════════════════════
# 5. Mean-CVaR Optimization
# ══════════════════════════════════════════════════════════════════════════════


class MeanCVaROptimization(BasePortfolioModel):
    """均值-CVaR 组合优化器。

    V4.1 feature: portfolio_v2

    CVaR 定义：在置信水平 alpha 下，超过 VaR 的尾部平均损失。

        VaR_alpha(L)  = inf{l: P(L <= l) >= alpha}
        CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]

    Historical CVaR 使用 Rockafellar & Uryasev (2000) 形式：

        min_{w,z,u}  -mu'w + lambda * [z + 1/((1-alpha)T) * sum(u_t)]
        s.t.        u_t >= -r_t'w - z
                    u_t >= 0
                    sum(w) = 1

    Gaussian CVaR 对正态损失 L ~ N(mu_L, sigma_L) 的闭式：

        CVaR_alpha(L) = mu_L + sigma_L * phi(Phi^-1(alpha)) / (1-alpha)

    本类支持：
    - historical / gaussian 两种 CVaR。
    - 满仓约束。
    - 行业上限 sector_limits。
    - 个股上下限 asset_bounds 或 min_weight/max_weight。
    - 目标收益 target_return。

    用法示例：

    >>> returns = _make_random_returns()
    >>> sector_map = {c: "Tech" if i < 3 else "Value" for i, c in enumerate(returns.columns)}
    >>> opt = MeanCVaROptimization(alpha=0.95, risk_aversion=5.0)
    >>> weights = opt.optimize(returns, cvar_method="historical", sector_map=sector_map, sector_limits={"Tech": 0.6})
    >>> opt.cvar_
    """

    method_name = "mean_cvar"

    def __init__(
        self,
        alpha: float = 0.95,
        risk_aversion: float = 5.0,
        cvar_method: CVaRMethod = "historical",
        periods_per_year: int = 252,
        shrinkage: float = 0.05,
        max_history: Optional[int] = 1000,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha 必须在 (0, 1)。")
        if risk_aversion < 0:
            raise ValueError("risk_aversion 不能为负。")
        self.alpha = float(alpha)
        self.risk_aversion = float(risk_aversion)
        self.cvar_method = cvar_method
        self.shrinkage = float(shrinkage)
        self.max_history = max_history
        self.cvar_: float = 0.0
        self.var_: float = 0.0
        # P2-Q15-fix (L116): var_ 为周期口径、cvar_ 为年化口径（√ppy）。
        # var_annualized_ 提供与 cvar_ 同口径的年化 VaR，便于统计汇总对比。
        self.var_annualized_: float = 0.0
        self.objective_value_: float = 0.0

    def calculate_cvar(
        self,
        returns: ArrayLike2D,
        weights: ArrayLike1D,
        alpha: Optional[float] = None,
        method: Optional[CVaRMethod] = None,
        periods_per_year: Optional[int] = None,
    ) -> float:
        """计算给定权重的 CVaR。

        P2-Q15-fix (L114): 年化口径说明——historical/gaussian 均按 √periods_per_year
        年化（与 var_annualized_ 一致）。该口径仅在 iid/正态近似下严格成立，
        厚尾分布下有偏差；如需周期口径值请自行除以 √periods_per_year。
        """
        frame = self._prepare_returns(returns)
        w = _series_from_any(weights, frame.columns, name="weight")
        a = float(alpha if alpha is not None else self.alpha)
        m = method or self.cvar_method
        ppy = int(periods_per_year if periods_per_year is not None else self.periods_per_year)

        port = frame @ w
        if m == "historical":
            return _historical_cvar_from_returns(port, alpha=a) * math.sqrt(ppy)
        if m == "gaussian":
            mu = float(port.mean() * ppy)
            sigma = float(port.std(ddof=1) * math.sqrt(ppy))
            z = norm.ppf(a)
            return float(-mu + sigma * norm.pdf(z) / (1.0 - a))
        raise ValueError(f"未知 CVaR 方法: {m}")

    def _historical_optimize(
        self,
        frame: pd.DataFrame,
        mu: pd.Series,
        bounds: Bounds,
        weight_constraints: list[Any],
        alpha: float,
        risk_aversion: float,
        max_iter: int,
        full_investment: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Rockafellar-Uryasev historical CVaR 优化。"""
        returns_matrix = frame.values.astype(float)
        n_obs, n_assets = returns_matrix.shape
        lower_w = np.asarray(bounds.lb[:n_assets], dtype=float)
        upper_w = np.asarray(bounds.ub[:n_assets], dtype=float)
        # P1-Q15-fix: 允许现金场景（full_investment=False）target_sum 传 None，
        # 避免 max_weight 之和<1 时抛 OptimizationError。
        w0 = _feasible_initial_weights(lower_w, upper_w, target_sum=1.0 if full_investment else None)
        if not full_investment:
            # P1-Q15-fix: 从 0 出发会让 SLSQP 卡在全现金退化解（0 点是退化 KKT 点，
            # 实测 historical 求解器失败并兜底全 0）。改用等权夹取点作非退化初始猜测，
            # 与 Kelly 模型的处理一致；求解失败时兜底也不是全 0。
            w0 = np.maximum(np.minimum(np.full(n_assets, 1.0 / n_assets, dtype=float), upper_w), lower_w)
        initial_losses = -returns_matrix @ w0
        z0 = float(np.quantile(initial_losses, alpha))
        u0 = np.maximum(initial_losses - z0, 0.0)
        x0 = np.concatenate([w0, np.array([z0]), u0])

        # 变量 x = [w_1..w_N, z, u_1..u_T]
        lower = np.concatenate([lower_w, np.array([-np.inf]), np.zeros(n_obs)])
        upper = np.concatenate([upper_w, np.array([np.inf]), np.full(n_obs, np.inf)])
        var_bounds = Bounds(lower, upper)

        constraints: list[Any] = []
        # 原权重约束扩展到 x 变量。
        for cons in weight_constraints:
            if isinstance(cons, LinearConstraint):
                a = np.asarray(cons.A, dtype=float)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                pad = np.zeros((a.shape[0], 1 + n_obs), dtype=float)
                constraints.append(LinearConstraint(np.hstack([a, pad]), cons.lb, cons.ub))
            else:
                constraints.append(cons)

        # u_t >= -r_t'w - z，即 u_t + r_t'w + z >= 0。
        def tail_constraints(x: np.ndarray) -> np.ndarray:
            w = x[:n_assets]
            z = x[n_assets]
            u = x[n_assets + 1 :]
            return u + returns_matrix @ w + z

        constraints.append({"type": "ineq", "fun": tail_constraints})

        # P2-Q15-fix (L114): 目标函数统一周期口径。旧实现期望收益按 ×ppy 年化、
        # CVaR 按 √ppy 年化（cvar_scale=sqrt(ppy)），两项尺度不一致使有效
        # risk_aversion 被放大 √ppy 倍（语义失真）。现在期望收益与 CVaR 均用
        # 周期值（Rockafellar-Uryasev 原形式），risk_aversion 恢复其本义。
        def objective(x: np.ndarray) -> float:
            w = x[:n_assets]
            z = x[n_assets]
            u = x[n_assets + 1 :]
            expected_period = float(mu.values @ w) / self.periods_per_year
            cvar_period = z + np.mean(u) / (1.0 - alpha)
            return -expected_period + risk_aversion * cvar_period

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=var_bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
        )

        if result.success and np.all(np.isfinite(result.x[:n_assets])):
            return result.x[:n_assets].astype(float), {
                "success": True,
                "message": str(result.message),
                "objective_value": float(result.fun),
                "n_iter": int(getattr(result, "nit", 0)),
            }

        # P2-Q15-fix (M107): 兜底 w0 只满足个股边界，可能违反 sector/target 约束。
        _check_fallback_feasible(w0, weight_constraints, context="MeanCVaR historical")
        return w0, {
            "success": False,
            "message": str(getattr(result, "message", "historical CVaR optimizer failed")),
            "objective_value": float(getattr(result, "fun", np.nan)),
            "fallback": "feasible_initial_weights",
        }

    def _gaussian_optimize(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        bounds: Bounds,
        constraints: list[Any],
        alpha: float,
        risk_aversion: float,
        max_iter: int,
        full_investment: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Gaussian CVaR 优化。"""
        n_assets = len(mu)
        lower_w = np.asarray(bounds.lb, dtype=float)
        upper_w = np.asarray(bounds.ub, dtype=float)
        # P1-Q15-fix: 允许现金场景（full_investment=False）target_sum 传 None，
        # 避免 max_weight 之和<1 时抛 OptimizationError。
        x0 = _feasible_initial_weights(lower_w, upper_w, target_sum=1.0 if full_investment else None)
        if not full_investment:
            # P1-Q15-fix: 与 historical 一致，从 0 出发会卡退化点，改用等权夹取点。
            x0 = np.maximum(np.minimum(np.full(n_assets, 1.0 / n_assets, dtype=float), upper_w), lower_w)
        cov_values = _regularize_cov(cov.values)
        z = norm.ppf(alpha)
        tail_multiplier = norm.pdf(z) / (1.0 - alpha)

        def objective(w: np.ndarray) -> float:
            expected = float(mu.values @ w)
            vol = math.sqrt(max(float(w @ cov_values @ w), 0.0))
            cvar = -expected + vol * tail_multiplier
            return -expected + risk_aversion * cvar

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-10, "disp": False},
        )

        if result.success and np.all(np.isfinite(result.x)):
            return result.x.astype(float), {
                "success": True,
                "message": str(result.message),
                "objective_value": float(result.fun),
                "n_iter": int(getattr(result, "nit", 0)),
            }
        # P2-Q15-fix (M107): 兜底 x0 只满足个股边界，可能违反 sector/target 约束。
        _check_fallback_feasible(x0, constraints, context="MeanCVaR gaussian")
        return x0, {
            "success": False,
            "message": str(getattr(result, "message", "gaussian CVaR optimizer failed")),
            "objective_value": float(getattr(result, "fun", np.nan)),
            "fallback": "feasible_initial_weights",
        }

    def optimize(
        self,
        returns: ArrayLike2D,
        expected_returns: Optional[ArrayLike1D] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        cvar_method: Optional[CVaRMethod] = None,
        alpha: Optional[float] = None,
        risk_aversion: Optional[float] = None,
        full_investment: bool = True,
        asset_bounds: WeightBounds = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_map: Optional[Mapping[str, str]] = None,
        sector_limits: Optional[Mapping[str, float]] = None,
        industry_limits: Optional[Mapping[str, float]] = None,
        target_return: Optional[float] = None,
        max_iter: int = 1000,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """优化均值-CVaR 权重。"""
        frame = self._prepare_returns(returns, assets=assets)
        if self.max_history is not None and len(frame) > self.max_history:
            frame = frame.iloc[-int(self.max_history) :]

        asset_index = pd.Index(frame.columns)
        a = float(alpha if alpha is not None else self.alpha)
        method = cvar_method or self.cvar_method
        lam = float(risk_aversion if risk_aversion is not None else self.risk_aversion)

        if expected_returns is None:
            mu, cov_est = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)
        else:
            mu = _series_from_any(expected_returns, asset_index, name="expected_return")
            _, cov_est = _annualized_mean_cov(frame, periods_per_year=self.periods_per_year, shrinkage=self.shrinkage)

        if cov_matrix is None:
            cov = cov_est
        elif isinstance(cov_matrix, pd.DataFrame):
            cov = cov_matrix.reindex(index=asset_index, columns=asset_index).fillna(0.0).astype(float)
        else:
            cov = pd.DataFrame(_regularize_cov(cov_matrix), index=asset_index, columns=asset_index)

        limits = sector_limits or industry_limits
        lower, upper = _infer_bounds(asset_index, bounds=asset_bounds, min_weight=min_weight, max_weight=max_weight)
        bounds = Bounds(lower, upper)
        constraints = _build_linear_constraints(
            asset_index,
            full_investment=full_investment,
            target_sum=1.0,
            sector_map=sector_map,
            sector_limits=limits,
            target_return=target_return,
            expected_returns=mu.values,
        )

        if method == "historical":
            weights_arr, solver_info = self._historical_optimize(
                frame,
                mu,
                bounds,
                constraints,
                alpha=a,
                risk_aversion=lam,
                max_iter=max_iter,
                full_investment=full_investment,
            )
        elif method == "gaussian":
            weights_arr, solver_info = self._gaussian_optimize(
                mu,
                cov,
                bounds,
                constraints,
                alpha=a,
                risk_aversion=lam,
                max_iter=max_iter,
                full_investment=full_investment,
            )
        else:
            raise ValueError(f"未知 cvar_method: {method}")

        weights = _series_weights(weights_arr, asset_index)
        if full_investment and abs(weights.sum() - 1.0) > 1e-6:
            weights = (weights / weights.sum()).rename("weight") if weights.sum() > 1e-16 else weights

        portfolio_returns = frame @ weights.reindex(frame.columns).fillna(0.0)
        losses = -portfolio_returns.values
        # P2-Q15-fix (L116): var_ 为周期口径、cvar_ 为年化口径（√ppy，与
        # calculate_cvar 一致）。新增 var_annualized_（√ppy 口径）使 VaR 与 CVaR
        # 可在同一口径下对比，诊断中拆分字段名 var_period / var_annualized / cvar_annualized。
        self.var_ = float(np.quantile(losses, a))
        self.var_annualized_ = self.var_ * math.sqrt(self.periods_per_year)
        self.cvar_ = self.calculate_cvar(frame, weights, alpha=a, method=method)
        self.objective_value_ = float(solver_info.get("objective_value", np.nan))
        self.diagnostics_ = {
            "alpha": a,
            "risk_aversion": lam,
            "cvar_method": method,
            "full_investment": full_investment,
            "sector_limits": dict(limits or {}),
            "target_return": target_return,
            "var_period": self.var_,
            "var_annualized": self.var_annualized_,
            "cvar_annualized": self.cvar_,
            "solver": solver_info,
        }
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=bool(solver_info.get("success", False)),
            message=str(solver_info.get("message", "")),
            n_assets=frame.shape[1],
            n_observations=frame.shape[0],
            details=self.diagnostics_,
        )
        return weights


# ══════════════════════════════════════════════════════════════════════════════
# 6. Regime-Aware Allocation
# ══════════════════════════════════════════════════════════════════════════════


class RegimeAwareAllocation(BasePortfolioModel):
    """市场状态感知配置器。

    V4.1 feature: portfolio_v2

    该类不重新估计最优组合，而是在多组预先定义的 regime 权重之间选择或插值。
    典型用法是把 bull、bear、range、high_vol 下的目标组合先离线求好，实盘根据
    当前状态做平滑切换。

    注意：本实现为 long-only。regime 权重表中的负权重（多空组合）会被裁剪到 0
    并归一化，并发出 RuntimeWarning（P2-Q15-fix M109）。

    权重选择逻辑：

    1. 如果传入 regime_probabilities：
       target_w = sum_r prob_r * weight_map[r]

    2. 否则根据 current_regime 或 detector.predict() 选择：
       target_w = weight_map[current_regime]

    3. 如果传入 previous_weights 且 half_life > 0：
       decay = 0.5 ** (periods_elapsed / half_life)
       smooth_w = decay * previous_w + (1 - decay) * target_w

    用法示例：

    >>> returns = _make_random_returns()
    >>> regime_weights = {
    ...     "bull": {"Asset_1": 0.3, "Asset_2": 0.3, "Asset_3": 0.4},
    ...     "bear": {"Asset_1": 0.1, "Asset_2": 0.1, "Asset_3": 0.8},
    ... }
    >>> allocator = RegimeAwareAllocation(regime_weights, default_regime="bear", half_life=5)
    >>> weights = allocator.optimize(returns, current_regime="bull")
    """

    method_name = "regime"

    def __init__(
        self,
        regime_weights: Optional[Mapping[str, ArrayLike1D]] = None,
        default_regime: str = "range",
        half_life: float = 5.0,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(periods_per_year=periods_per_year)
        self.regime_weights = dict(regime_weights or {})
        self.default_regime = str(default_regime)
        self.half_life = float(half_life)
        self.current_regime_: str = self.default_regime
        self.target_weights_: pd.Series = pd.Series(dtype=float)
        self.smoothed_weights_: pd.Series = pd.Series(dtype=float)

    def _all_assets(
        self,
        returns: Optional[pd.DataFrame],
        regime_weights: Mapping[str, ArrayLike1D],
    ) -> pd.Index:
        """从 returns 和 regime 权重中推断完整资产列表。"""
        assets: list[str] = []
        if returns is not None:
            assets.extend([str(c) for c in returns.columns])
        for weights in regime_weights.values():
            if isinstance(weights, pd.Series):
                assets.extend([str(x) for x in weights.index])
            elif isinstance(weights, Mapping):
                assets.extend([str(x) for x in weights.keys()])
        if not assets:
            raise ValueError("无法推断资产列表，请传入 returns 或 regime_weights。")
        return pd.Index(sorted(dict.fromkeys(assets)))

    def _normalize_regime_map(
        self,
        regime_weights: Mapping[str, ArrayLike1D],
        assets: Sequence[str],
    ) -> dict[str, pd.Series]:
        """把 regime 权重表转换为统一 Series。"""
        out: dict[str, pd.Series] = {}
        for regime, weights in regime_weights.items():
            series = _series_from_any(weights, assets, name=str(regime))
            # P2-Q15-fix (M109): 本实现为 long-only。用户传入多空 regime 权重时，
            # 负权重此前被 clip(lower=0) 静默改写成纯多。现在发出可见告警。
            if float(series.min()) < -1e-12:
                warnings.warn(
                    f"[Regime] regime '{regime}' 权重含负值（多空组合），"
                    "当前实现为 long-only，负权重将被裁剪到 0 后归一化。",
                    RuntimeWarning,
                )
            if series.sum() > 1e-16:
                series = series.clip(lower=0.0)
                series = series / series.sum()
            out[str(regime)] = series.rename("weight")
        return out

    def detect_regime(
        self,
        returns: Optional[pd.DataFrame] = None,
        prices: Optional[pd.DataFrame] = None,
        detector: Optional[Any] = None,
        current_regime: Optional[str] = None,
        date: Any = None,
    ) -> str:
        """检测或读取当前 regime。

        对接 quant_system.market_analysis.regime：
        - 如果传入 detector，优先调用 detector.predict(date)。
        - 如果未传 detector 但传入 prices，则尝试使用 TrendRegime。
        - 如果都没有，则返回 current_regime 或 default_regime。
        """
        if current_regime is not None:
            return str(current_regime)

        if detector is not None:
            if hasattr(detector, "predict"):
                return str(detector.predict(date))
            if callable(detector):
                return str(detector(returns=returns, prices=prices, date=date))

        if prices is not None:
            try:
                # P1-Q22-fix: regime.py 只定义 TrendRegime(无 TrendRegimeDetector)，
                # 改用 TrendRegime.trend_regime()，避免 ImportError 静默回退默认 regime
                try:
                    from quant_system.market_analysis.regime import TrendRegime
                except Exception:
                    from market_analysis.regime import TrendRegime  # type: ignore
                model = TrendRegime()
                return str(model.trend_regime(prices))
            except Exception as exc:
                warnings.warn(f"Regime detector fallback to default because detection failed: {exc}", RuntimeWarning)

        return self.default_regime

    def _target_from_probabilities(
        self,
        normalized_map: Mapping[str, pd.Series],
        probabilities: Mapping[str, float],
        assets: Sequence[str],
    ) -> pd.Series:
        """按 regime 概率插值权重。"""
        target = pd.Series(0.0, index=pd.Index(assets), dtype=float)
        total_prob = 0.0
        for regime, prob in probabilities.items():
            p = max(float(prob), 0.0)
            if p <= 0:
                continue
            if regime in normalized_map:
                target += p * normalized_map[regime].reindex(assets).fillna(0.0)
                total_prob += p
        if total_prob <= 1e-16:
            fallback = normalized_map.get(self.default_regime) or next(iter(normalized_map.values()))
            return fallback.reindex(assets).fillna(0.0).rename("weight")
        target = target / total_prob
        return target.rename("weight")

    def _smooth_transition(
        self,
        target: pd.Series,
        previous_weights: Optional[ArrayLike1D],
        periods_elapsed: float,
        half_life: Optional[float] = None,
    ) -> pd.Series:
        """半衰期平滑权重切换。"""
        hl = self.half_life if half_life is None else float(half_life)
        if previous_weights is None or hl <= 0:
            return target.rename("weight")
        prev = _series_from_any(previous_weights, target.index, name="previous_weight")
        if prev.sum() > 1e-16:
            prev = prev.clip(lower=0.0) / prev.clip(lower=0.0).sum()
        decay = 0.5 ** (max(float(periods_elapsed), 0.0) / hl)
        smoothed = decay * prev + (1.0 - decay) * target
        if smoothed.sum() > 1e-16:
            smoothed = smoothed.clip(lower=0.0) / smoothed.clip(lower=0.0).sum()
        return smoothed.rename("weight")

    def optimize(
        self,
        returns: Optional[ArrayLike2D] = None,
        regime_weights: Optional[Mapping[str, ArrayLike1D]] = None,
        current_regime: Optional[str] = None,
        regime_probabilities: Optional[Mapping[str, float]] = None,
        previous_weights: Optional[ArrayLike1D] = None,
        periods_elapsed: float = 1.0,
        half_life: Optional[float] = None,
        prices: Optional[pd.DataFrame] = None,
        detector: Optional[Any] = None,
        date: Any = None,
        assets: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> pd.Series:
        """根据 regime 选择或插值权重。"""
        frame: Optional[pd.DataFrame] = None
        if returns is not None:
            frame = self._prepare_returns(returns, assets=assets)

        weight_map = dict(regime_weights or self.regime_weights)
        if not weight_map:
            if frame is None:
                raise ValueError("regime_weights 为空时必须传入 returns 以生成等权 fallback。")
            equal = pd.Series(1.0 / frame.shape[1], index=frame.columns, name="weight")
            self.current_regime_ = current_regime or self.default_regime
            self.target_weights_ = equal
            self.smoothed_weights_ = self._smooth_transition(equal, previous_weights, periods_elapsed, half_life)
            return self.smoothed_weights_

        asset_index = pd.Index([str(a) for a in assets]) if assets is not None else self._all_assets(frame, weight_map)
        normalized_map = self._normalize_regime_map(weight_map, asset_index)

        regime = self.detect_regime(
            returns=frame,
            prices=prices,
            detector=detector,
            current_regime=current_regime,
            date=date,
        )
        self.current_regime_ = regime

        if regime_probabilities:
            target = self._target_from_probabilities(normalized_map, regime_probabilities, asset_index)
        elif regime in normalized_map:
            target = normalized_map[regime].reindex(asset_index).fillna(0.0).rename("weight")
        elif self.default_regime in normalized_map:
            target = normalized_map[self.default_regime].reindex(asset_index).fillna(0.0).rename("weight")
        else:
            target = next(iter(normalized_map.values())).reindex(asset_index).fillna(0.0).rename("weight")

        if target.sum() > 1e-16:
            target = target.clip(lower=0.0) / target.clip(lower=0.0).sum()
        self.target_weights_ = target.rename("target_weight")
        smoothed = self._smooth_transition(target, previous_weights, periods_elapsed, half_life)
        self.smoothed_weights_ = smoothed

        self.diagnostics_ = {
            "current_regime": regime,
            "default_regime": self.default_regime,
            "used_probabilities": dict(regime_probabilities or {}),
            "half_life": self.half_life if half_life is None else half_life,
            "periods_elapsed": periods_elapsed,
        }
        self.metadata_ = OptimizationMetadata(
            method=self.method_name,
            success=True,
            message="Regime-aware allocation completed",
            n_assets=len(asset_index),
            n_observations=0 if frame is None else frame.shape[0],
            details=self.diagnostics_,
        )
        return smoothed.rename("weight")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PortfolioOptimizerV2 顶层整合
# ══════════════════════════════════════════════════════════════════════════════


class PortfolioOptimizerV2:
    """高级组合构建顶层入口。

    V4.1 feature: portfolio_v2

    统一入口：

        optimize(returns, method='black_litterman'|'hrp'|'nco'|'kelly'|'mean_cvar'|'regime', **kwargs) -> dict

    返回结构：

        {
            'weights': {'AssetA': 0.25, ...},
            'stats': {'vol': ..., 'sharpe': ..., 'max_dd': ...},
            'metadata': {...}
        }

    用法示例：

    >>> returns = _make_random_returns()
    >>> engine = PortfolioOptimizerV2()
    >>> result = engine.optimize(returns, method="hrp")
    >>> result["weights"]
    >>> result["stats"]["sharpe"]
    """

    SUPPORTED_METHODS = {
        "black_litterman",
        "bl",
        "hrp",
        "nco",
        "kelly",
        "mean_cvar",
        "cvar",
        "regime",
    }

    def __init__(
        self,
        periods_per_year: int = 252,
        risk_free: float = 0.0,
    ) -> None:
        self.periods_per_year = int(periods_per_year)
        self.risk_free = float(risk_free)
        self.last_result_: dict[str, Any] = {}
        self.last_model_: Optional[Any] = None

    def _make_model(self, method: str, kwargs: Mapping[str, Any]) -> tuple[str, Any]:
        """根据 method 创建对应模型实例。"""
        m = method.lower()
        if m == "bl":
            m = "black_litterman"
        if m == "cvar":
            m = "mean_cvar"
        if m not in self.SUPPORTED_METHODS:
            raise ValueError(f"未知优化方法: {method}")

        common = {"periods_per_year": int(kwargs.get("periods_per_year", self.periods_per_year))}

        if m == "black_litterman":
            model = BlackLitterman(
                tau=float(kwargs.get("tau", 0.05)),
                risk_aversion=float(kwargs.get("risk_aversion", 2.5)),
                default_confidence=float(kwargs.get("default_confidence", 0.5)),
                shrinkage=float(kwargs.get("shrinkage", 0.05)),
                **common,
            )
        elif m == "hrp":
            model = HierarchicalRiskParity(
                linkage_method=str(kwargs.get("linkage_method", "single")),
                shrinkage=float(kwargs.get("shrinkage", 0.05)),
                **common,
            )
        elif m == "nco":
            model = NestedClusteredOptimization(
                n_clusters=kwargs.get("n_clusters"),
                n_monte_carlo=int(kwargs.get("n_monte_carlo", 64)),
                sample_length=kwargs.get("sample_length"),
                objective=kwargs.get("objective", "min_variance"),
                random_state=int(kwargs.get("random_state", 42)),
                shrinkage=float(kwargs.get("shrinkage", 0.05)),
                **common,
            )
        elif m == "kelly":
            model = KellyCriterion(
                fraction=float(kwargs.get("fraction", 0.25)),
                leverage_limit=float(kwargs.get("leverage_limit", 1.0)),
                borrowing_limit=float(kwargs.get("borrowing_limit", 0.0)),
                borrowing_rate=float(kwargs.get("borrowing_rate", 0.0)),
                shrinkage=float(kwargs.get("shrinkage", 0.05)),
                **common,
            )
        elif m == "mean_cvar":
            model = MeanCVaROptimization(
                alpha=float(kwargs.get("alpha", 0.95)),
                risk_aversion=float(kwargs.get("risk_aversion", 5.0)),
                cvar_method=kwargs.get("cvar_method", "historical"),
                shrinkage=float(kwargs.get("shrinkage", 0.05)),
                max_history=kwargs.get("max_history", 1000),
                **common,
            )
        elif m == "regime":
            model = RegimeAwareAllocation(
                regime_weights=kwargs.get("regime_weights"),
                default_regime=str(kwargs.get("default_regime", "range")),
                half_life=float(kwargs.get("half_life", 5.0)),
                **common,
            )
        else:  # pragma: no cover - 前置校验已覆盖。
            raise ValueError(f"未知优化方法: {method}")
        return m, model

    def _metadata_payload(self, model: Any, method: str) -> dict[str, Any]:
        """收集模型 metadata，并附加模型特有输出。"""
        if hasattr(model, "metadata_"):
            meta = model.metadata_.to_dict()
        else:
            meta = OptimizationMetadata(method=method).to_dict()

        details = dict(meta.get("details", {}))
        if isinstance(model, BlackLitterman):
            details["posterior_returns"] = model.posterior_returns_.to_dict()
            details["posterior_covariance"] = model.posterior_covariance_.to_dict()
            details["prior_returns"] = model.prior_returns_.to_dict()
        elif isinstance(model, HierarchicalRiskParity):
            details["sorted_assets"] = model.sorted_assets_
        elif isinstance(model, NestedClusteredOptimization):
            details["cluster_labels"] = model.cluster_labels_.to_dict()
            details["cluster_weights"] = model.cluster_weights_.to_dict()
        elif isinstance(model, KellyCriterion):
            details["full_kelly_weights"] = model.full_kelly_weights_.to_dict()
            details["cash_weight"] = model.cash_weight_
        elif isinstance(model, MeanCVaROptimization):
            details["var_period"] = model.var_
            details["var_annualized"] = model.var_annualized_
            details["cvar_annualized"] = model.cvar_
        elif isinstance(model, RegimeAwareAllocation):
            details["current_regime"] = model.current_regime_
            details["target_weights"] = model.target_weights_.to_dict()

        meta["details"] = details
        return meta

    def optimize(
        self,
        returns: Optional[ArrayLike2D] = None,
        method: str = "hrp",
        previous_weights: Optional[ArrayLike1D] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """统一优化入口。"""
        normalized_method, model = self._make_model(method, kwargs)

        frame: Optional[pd.DataFrame]
        if returns is None:
            if normalized_method != "regime":
                raise ValueError("除 regime 方法外，returns 不能为空。")
            frame = None
        else:
            frame = _as_returns_frame(returns, assets=kwargs.get("assets"))

        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("risk_free", self.risk_free)
        if previous_weights is not None:
            call_kwargs.setdefault("previous_weights", previous_weights)

        if normalized_method == "regime":
            weights = model.optimize(frame, **call_kwargs)
        else:
            assert frame is not None
            weights = model.optimize(frame, **call_kwargs)

        weights = weights.astype(float).rename("weight")
        weights[np.abs(weights) < 1e-12] = 0.0

        if frame is not None:
            # 对 Kelly 允许现金，不强行归一；其他方法通常满仓。
            # P1-Q15-fix: 归一化不得破坏用户边界——显式 full_investment=False（允许现金）
            # 或 max_weight 使归一化后权重超限时，保留原权重不归一（保留现金）。
            _renorm = normalized_method != "kelly" and abs(weights.sum() - 1.0) > 1e-6 and weights.sum() > 1e-16
            if _renorm:
                _mw = call_kwargs.get("max_weight")
                if call_kwargs.get("full_investment") is False:
                    _renorm = False
                elif _mw is not None and float(_mw) < 1.0 and (weights / weights.sum()).max() > float(_mw) + 1e-9:
                    _renorm = False
            if _renorm:
                weights = weights / weights.sum()
            expected_returns = None
            cov_matrix = None
            if isinstance(model, BlackLitterman):
                expected_returns = model.posterior_returns_
                cov_matrix = model.posterior_covariance_
            stats = _portfolio_stats_from_returns(
                frame,
                weights,
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_free=self.risk_free,
                periods_per_year=self.periods_per_year,
                previous_weights=previous_weights,
            )
        else:
            stats = PortfolioStats(effective_n=_effective_n(weights.values))

        metadata = self._metadata_payload(model, normalized_method)
        metadata["stats_source"] = "historical_returns" if frame is not None else "weights_only"
        metadata["weight_sum"] = float(weights.sum())
        metadata["gross_exposure"] = float(np.abs(weights.values).sum())

        result = {
            "weights": {str(k): float(v) for k, v in weights.items()},
            "stats": stats.to_dict(),
            "metadata": metadata,
        }
        self.last_result_ = result
        self.last_model_ = model
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Demo / usage 示例
# ══════════════════════════════════════════════════════════════════════════════


def demo_black_litterman() -> pd.Series:
    """BlackLitterman 使用示例。"""
    returns = _make_random_returns(n_assets=5)
    views = [
        {"type": "absolute", "assets": {"Asset_1": 1.0}, "view": 0.12, "confidence": 0.70},
        {"type": "relative", "assets": {"Asset_2": 1.0, "Asset_3": -1.0}, "view": 0.03, "confidence": 0.60},
    ]
    model = BlackLitterman(tau=0.05, risk_aversion=2.5)
    return model.optimize(returns, views=views, max_weight=0.5)


def demo_hrp() -> pd.Series:
    """HierarchicalRiskParity 使用示例。"""
    returns = _make_random_returns(n_assets=8)
    model = HierarchicalRiskParity(linkage_method="single")
    return model.optimize(returns)


def demo_nco() -> pd.Series:
    """NestedClusteredOptimization 使用示例。"""
    returns = _make_random_returns(n_assets=10)
    model = NestedClusteredOptimization(n_clusters=3, n_monte_carlo=8, random_state=7)
    return model.optimize(returns, objective="min_variance")


def demo_kelly() -> pd.Series:
    """KellyCriterion 使用示例。"""
    returns = _make_random_returns(n_assets=6)
    model = KellyCriterion(fraction=0.2, leverage_limit=1.2, borrowing_limit=0.2)
    return model.optimize(returns, long_only=True, max_weight=0.4)


def demo_mean_cvar() -> pd.Series:
    """MeanCVaROptimization 使用示例。"""
    returns = _make_random_returns(n_assets=6)
    sector_map = {asset: "growth" if i < 3 else "defensive" for i, asset in enumerate(returns.columns)}
    model = MeanCVaROptimization(alpha=0.95, risk_aversion=4.0, cvar_method="gaussian")
    return model.optimize(
        returns,
        cvar_method="gaussian",
        sector_map=sector_map,
        sector_limits={"growth": 0.65, "defensive": 0.65},
        max_weight=0.45,
    )


def demo_regime() -> pd.Series:
    """RegimeAwareAllocation 使用示例。"""
    returns = _make_random_returns(n_assets=4)
    regime_weights = {
        "bull": {"Asset_1": 0.35, "Asset_2": 0.30, "Asset_3": 0.25, "Asset_4": 0.10},
        "range": {"Asset_1": 0.25, "Asset_2": 0.25, "Asset_3": 0.25, "Asset_4": 0.25},
        "bear": {"Asset_1": 0.10, "Asset_2": 0.15, "Asset_3": 0.25, "Asset_4": 0.50},
        "high_vol": {"Asset_1": 0.05, "Asset_2": 0.10, "Asset_3": 0.25, "Asset_4": 0.60},
    }
    previous = {"Asset_1": 0.25, "Asset_2": 0.25, "Asset_3": 0.25, "Asset_4": 0.25}
    model = RegimeAwareAllocation(regime_weights=regime_weights, default_regime="range", half_life=5)
    return model.optimize(returns, current_regime="bull", previous_weights=previous, periods_elapsed=2)


def demo_portfolio_optimizer_v2() -> dict[str, Any]:
    """PortfolioOptimizerV2 顶层入口使用示例。"""
    returns = _make_random_returns(n_assets=6)
    engine = PortfolioOptimizerV2(periods_per_year=252, risk_free=0.02)
    return engine.optimize(returns, method="hrp")


if __name__ == "__main__":
    # 手动运行本文件时，执行一组轻量 demo，方便快速验证接口。
    demos: dict[str, Callable[[], Any]] = {
        "black_litterman": demo_black_litterman,
        "hrp": demo_hrp,
        "nco": demo_nco,
        "kelly": demo_kelly,
        "mean_cvar": demo_mean_cvar,
        "regime": demo_regime,
        "engine": demo_portfolio_optimizer_v2,
    }
    for name, fn in demos.items():
        print(f"\n--- {name} ---")
        print(fn())
