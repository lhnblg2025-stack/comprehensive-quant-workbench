"""
portfolio.py — 投资组合优化引擎 V4

核心功能:
  1. 均值-方差优化 (Mean-Variance with Ledoit-Wolf shrinkage)
  2. 风险平价 (Risk Parity)
  3. Hierarchical Risk Parity (HRP)
  4. Black-Litterman 模型
  5. 凯利公式仓位计算
  6. 再平衡策略

所有函数接收 DataFrame (index=date, columns=symbols, values=returns) 统一接口。

P2-Q16-fix (L132): 【弃用提示】全库无外部调用方（仅自身 CLI）。V7 组合逻辑实际走
portfolio_optimizer.py（mean_variance_optimize / black_litterman /
equal_risk_contribution / cvar_optimize）。本模块保留作为历史/双轨实现，避免删除
既有功能，但新代码应统一使用 portfolio_optimizer.py，防止双轨漂移。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))


# ════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    """组合优化结果"""
    weights: dict[str, float]          # symbol -> weight
    expected_return: float             # 预期年化收益
    expected_vol: float                # 预期年化波动率
    sharpe_ratio: float                # 夏普比
    diversification_ratio: float       # 分散度 (加权平均波动率/组合波动率)
    method: str                        # 优化方法名
    top_holdings: list[tuple[str, float]] = field(default_factory=list)  # 前五大持仓
    n_positions: int = 0
    effective_n: float = 0.0           # 有效持仓数 (Herfindahl inverse)
    turnover: float = 0.0              # 单边换手率
    estimated_cost: float = 0.0        # 估算交易成本
    cvar: float = 0.0                  # 条件在险价值
    # P1-Q16-fix: 优化状态与告警（success / success_with_warning / infeasible）
    status: str = "success"
    warnings: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# 协方差估计
# ════════════════════════════════════════════════════════════════

def _cov_shrinkage(returns: pd.DataFrame, alpha: float = 0.15) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage 压缩协方差估计。

    Args:
        returns: T×N DataFrame
        alpha: 压缩强度 (0=样本协方差, 1=完全压缩)
    Returns:
        N×N 压缩协方差矩阵
    """
    T, N = returns.shape
    if N < 2:
        return returns.cov().values if not returns.empty else np.eye(1)

    # P2-Q16-fix (M118): 停牌/次新/零波动标的会带来常数列或全 NaN 列，使
    # returns.corr() 出现 NaN → prior 全 NaN → 返回含 NaN 协方差（HRP 输入后
    # linkage 直接 "condensed distance matrix must contain only finite values"）。
    # 清洗策略：样本协方差/相关系数的非有限元素置 0（该资产与其他资产不共变），
    # 保持矩阵维度不变（不删列，避免与调用方的 symbols 对齐断裂）；prior_std 对
    # 零方差列为 0 → prior 相应行列全 0，符合"零波动资产无共变"直觉。
    sample_cov = returns.cov().values
    if not np.all(np.isfinite(sample_cov)):
        sample_cov = np.nan_to_num(sample_cov, nan=0.0, posinf=0.0, neginf=0.0)
    sample_cov = (sample_cov + sample_cov.T) / 2.0  # 对称化（nan_to_num 后不保证对称）

    # Prior: 常数相关性 (所有变量对等)
    mean_var = np.trace(sample_cov) / N
    prior = np.diag(np.diag(sample_cov))  # 对角矩阵, 保留方差
    # 给非对角元素赋平均相关系数
    mean_corr = 0.0
    if N > 1:
        corr = returns.corr().values
        if not np.all(np.isfinite(corr)):
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        mask = ~np.eye(N, dtype=bool)
        mean_corr = corr[mask].mean()
    prior_corr = np.full((N, N), mean_corr)
    prior_corr[np.diag_indices(N)] = 1.0
    prior_std = np.sqrt(np.diag(sample_cov))
    prior = np.outer(prior_std, prior_std) * prior_corr

    # Shrinkage
    shrunk = (1 - alpha) * sample_cov + alpha * prior
    return shrunk


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """协方差→相关系数"""
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0] = 1e-10
    corr = cov / outer
    # P2-Q16-fix (M118 配套): 数值误差下相关系数可能略超 [-1,1]，导致 HRP 的
    # sqrt(2*(1-corr)) 出现 NaN/负值警告。clip 到有效域（零方差/高相关标的）。
    return np.clip(corr, -1.0, 1.0)


# ════════════════════════════════════════════════════════════════
# 均值-方差优化
# ════════════════════════════════════════════════════════════════

def _portfolio_stats(weights: np.ndarray, mean_ret: np.ndarray,
                     cov: np.ndarray) -> tuple[float, float]:
    """计算组合收益和方差"""
    ret = weights @ mean_ret
    var = weights @ cov @ weights
    return ret, var


def _normalize_current_weights(current_weights: np.ndarray | None, n_assets: int) -> np.ndarray:
    """将当前权重整理为长度为 N 的向量，缺省为等权。"""
    if n_assets <= 0:
        return np.array([])
    if current_weights is None:
        return np.ones(n_assets) / n_assets
    weights = np.asarray(current_weights, dtype=float)
    if weights.shape[0] != n_assets:
        raise ValueError(f"current_weights length {weights.shape[0]} != asset count {n_assets}")
    total = weights.sum()
    return weights / total if total > 0 else np.ones(n_assets) / n_assets


def _transaction_cost_metrics(
    current_weights: np.ndarray | None,
    target_weights: np.ndarray,
    cost_bps: float = 10,
) -> tuple[float, float]:
    """返回单边换手率与估算成本。"""
    current = _normalize_current_weights(current_weights, len(target_weights))
    turnover = float(np.sum(np.abs(target_weights - current)) / 2)
    cost = float(turnover * cost_bps / 10000)
    return turnover, cost


def _apply_transaction_cost(
    expected_ret: float,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_bps: float = 10,
) -> float:
    """扣除交易成本后实际收益 = 预期收益 - 换手率 * 成本(bps)。"""
    turnover = np.sum(np.abs(target_weights - current_weights)) / 2
    cost = turnover * cost_bps / 10000
    return float(expected_ret - cost)


def optimize_mean_variance(
    returns: pd.DataFrame,
    target_sharpe: bool = True,
    risk_free: float = 0.025,
    max_weight: float = 0.15,
    min_weight: float = 0.0,
    shrinkage_alpha: float = 0.15,
    current_weights: np.ndarray | None = None,
    cost_bps: float = 10,
    max_vol: float | None = None,
) -> PortfolioResult:
    """
    均值-方差优化。

    Args:
        returns: T×N 收益率DataFrame
        target_sharpe: True=最大化夏普比, False=最小化方差
        risk_free: 无风险利率
        max_weight: 单股票最大权重
        min_weight: 单股票最小权重
        shrinkage_alpha: 压缩强度

    Returns: PortfolioResult
    """
    symbols = returns.columns.tolist()
    N = len(symbols)
    if N == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0, method="mv_empty")

    # P1-Q16-fix: 求解前可行性预检。Σw=1 ∩ {min_weight≤w≤max_weight} 需
    # N*max_weight≥1 且 N*min_weight≤1。默认 max_weight=0.15 时 N<7 不可行，
    # 旧代码静默回退等权（0.3333 超限）。不可行时自动放宽 max_weight 到 1/N，
    # 并显式记录告警；真正不可行（min 约束本身矛盾）直接返回 infeasible。
    warnings: list[str] = []
    max_w = float(max_weight)
    if min_weight > max_weight:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="mv_infeasible", status="infeasible",
                               warnings=["min_weight > max_weight"])
    if N * min_weight > 1.0 + 1e-12:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="mv_infeasible", status="infeasible",
                               warnings=[f"min_weight={min_weight} 过高: N*min_weight>1 无可行解"])
    if N * max_w < 1.0 - 1e-12:
        max_w = 1.0 / N
        warnings.append(
            f"max_weight={max_weight} infeasible for N={N} (requires N*max_weight>=1); "
            f"relaxed to {max_w:.6f}"
        )

    mean_ret = returns.mean().values * 252  # V4.1 fix: 年化收益率 (日频×252)
    cov = _cov_shrinkage(returns, alpha=shrinkage_alpha) * 252  # V4.1 fix: 年化协方差 (与mean_ret一致，均年化)
    current = _normalize_current_weights(current_weights, N)

    # 约束: 权重在范围内, 满仓
    bounds = Bounds([min_weight] * N, [max_w] * N)
    constraints = [LinearConstraint(np.ones(N), 1.0, 1.0)]
    if max_vol is not None:
        constraints.append({"type": "ineq", "fun": lambda w: max_vol ** 2 - _portfolio_stats(w, mean_ret, cov)[1]})

    # 初始值: 等权（经预检 N*max_w≥1 且 N*min_weight≤1，等权点必在界内）
    x0 = np.array([1.0 / N] * N)

    if target_sharpe:
        # 最大化夏普比 = 最小化负夏普
        def neg_sharpe(w):
            ret, var = _portfolio_stats(w, mean_ret, cov)
            ret = _apply_transaction_cost(ret, current, w, cost_bps)
            vol = np.sqrt(var) if var > 0 else 1e-10
            return -(ret - risk_free) / vol

        result = minimize(neg_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})
        w_opt = result.x
    else:
        # 最小化方差
        def port_var(w):
            # P2-Q16-fix (L128): 删除死代码 adjusted_ret（min_vol 目标只依赖 var，
            # 交易成本通过 return 处的 estimated_cost 反映，此处 adjusted_ret 从未使用）
            _, var = _portfolio_stats(w, mean_ret, cov)
            return var

        result = minimize(port_var, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})
        w_opt = result.x

    # P1-Q16-fix: 解校验。SLSQP 失败或解超限时不得静默回退等权（旧版 result.success
    # 为 False 时直接取 x0=等权，可能违反 max_weight）。校验失败显式回退等权并标记。
    w_opt = np.asarray(w_opt, dtype=np.float64)
    feasible = (
        abs(w_opt.sum() - 1.0) <= 1e-3
        and np.all(w_opt >= min_weight - 1e-8)
        and np.all(w_opt <= max_w + 1e-6)
    )
    if not feasible:
        w_opt = np.full(N, 1.0 / N)  # 预检已保证等权在界内
        status = "infeasible"
        warnings.append("优化失败且解不可行，显式回退到等权可行点（非最优，不超上限）")
    elif not result.success:
        status = "converged_with_warning"
        warnings.append("SLSQP 未收敛，返回可行近似解")
    elif warnings:
        status = "success_with_warning"
    else:
        status = "success"

    ret_gross, var_opt = _portfolio_stats(w_opt, mean_ret, cov)
    turnover, estimated_cost = _transaction_cost_metrics(current, w_opt, cost_bps)
    ret_opt = ret_gross - estimated_cost
    vol_opt = np.sqrt(var_opt) if var_opt > 0 else 0
    sharpe = (ret_opt - risk_free) / vol_opt if vol_opt > 0 else 0

    # 分散度
    weighted_avg_vol = w_opt @ np.sqrt(np.diag(cov))
    div_ratio = weighted_avg_vol / vol_opt if vol_opt > 0 else 0

    weights = {s: round(w, 4) for s, w in zip(symbols, w_opt) if w > 0.001}
    top = sorted(weights.items(), key=lambda x: -x[1])[:5]

    # 有效持仓数 (Herfindahl 倒数)
    hhi = sum(w ** 2 for w in w_opt)
    eff_n = 1.0 / hhi if hhi > 0 else N

    return PortfolioResult(
        weights=weights,
        expected_return=round(ret_opt, 4),
        expected_vol=round(vol_opt, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=round(div_ratio, 3),
        method="max_sharpe" if target_sharpe else "min_vol",
        top_holdings=top,
        n_positions=len(weights),
        effective_n=round(eff_n, 1),
        turnover=round(turnover, 4),
        estimated_cost=round(estimated_cost, 6),
        status=status,
        warnings=warnings,
    )


# ════════════════════════════════════════════════════════════════
# 风险平价 (Risk Parity)
# ════════════════════════════════════════════════════════════════

def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """计算每个资产的风险贡献"""
    port_var = weights @ cov @ weights
    if port_var <= 0:
        return np.zeros_like(weights)
    # 边际风险贡献
    mrc = cov @ weights
    # 总风险贡献 = 权重 * 边际风险贡献 / 组合波动率
    port_vol = np.sqrt(port_var)
    rc = weights * mrc / port_vol if port_vol > 0 else np.zeros_like(weights)
    return rc


def optimize_risk_parity(
    returns: pd.DataFrame,
    max_weight: float = 0.25,
    shrinkage_alpha: float = 0.15,
    current_weights: np.ndarray | None = None,
    cost_bps: float = 10,
) -> PortfolioResult:
    """
    风险平价优化: 每个资产贡献相同的组合风险。

    使用 Spinu (2013) 公式: 最小化 sum_i (s_i * w_i - 1/N)^2
    其中 s_i = (w'Σw)^{-1/2} * (Σw)_i 是资产i的风险贡献份额
    """
    symbols = returns.columns.tolist()
    N = len(symbols)
    if N == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="rp_empty")

    # P1-Q16-fix: 求解前可行性预检。默认 max_weight=0.25 时 N<4 不可行，
    # 旧代码静默回退等权（N=3 → 0.3333 超限）。不可行时自动放宽到 1/N 并告警。
    warnings: list[str] = []
    max_w = float(max_weight)
    if max_w <= 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="rp_infeasible", status="infeasible",
                               warnings=["max_weight<=0"])
    if N * max_w < 1.0 - 1e-12:
        max_w = 1.0 / N
        warnings.append(
            f"max_weight={max_weight} infeasible for N={N} (requires N*max_weight>=1); "
            f"relaxed to {max_w:.6f}"
        )

    mean_ret = returns.mean().values * 252  # V4.1 fix: 年化收益率
    cov = _cov_shrinkage(returns, alpha=shrinkage_alpha) * 252  # V4.1 fix: 年化协方差(与mean_ret一致)
    current = _normalize_current_weights(current_weights, N)

    # 目标: 每个资产风险贡献相等
    target_rc = 1.0 / N

    def rc_penalty(w):
        rc = _risk_contribution(w, cov)
        total_rc = rc.sum()
        if total_rc <= 0:
            return 1e6
        rc_share = rc / total_rc
        return np.sum((rc_share - target_rc) ** 2)

    bounds = Bounds([0.0] * N, [max_w] * N)
    constraints = [LinearConstraint(np.ones(N), 1.0, 1.0)]
    x0 = np.array([1.0 / N] * N)  # 经预检等权点在界内

    result = minimize(rc_penalty, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-10})

    # P1-Q16-fix: 解校验。失败/超限时显式回退等权并标记，不静默输出超限权重。
    w_opt = np.asarray(result.x, dtype=np.float64)
    feasible = (
        abs(w_opt.sum() - 1.0) <= 1e-3
        and np.all(w_opt >= -1e-8)
        and np.all(w_opt <= max_w + 1e-6)
    )
    if not feasible:
        w_opt = np.full(N, 1.0 / N)
        status = "infeasible"
        warnings.append("优化失败且解不可行，显式回退到等权可行点（非最优，不超上限）")
    elif not result.success:
        status = "converged_with_warning"
        warnings.append("SLSQP 未收敛，返回可行近似解")
    elif warnings:
        status = "success_with_warning"
    else:
        status = "success"

    ret_gross, var_opt = _portfolio_stats(w_opt, mean_ret, cov)
    turnover, estimated_cost = _transaction_cost_metrics(current, w_opt, cost_bps)
    ret_opt = ret_gross - estimated_cost
    vol_opt = np.sqrt(var_opt) if var_opt > 0 else 0
    sharpe = (ret_opt - 0.025) / vol_opt if vol_opt > 0 else 0

    weighted_avg_vol = w_opt @ np.sqrt(np.diag(cov))
    div_ratio = weighted_avg_vol / vol_opt if vol_opt > 0 else 0

    weights = {s: round(w, 4) for s, w in zip(symbols, w_opt) if w > 0.001}
    top = sorted(weights.items(), key=lambda x: -x[1])[:5]
    hhi = sum(w ** 2 for w in w_opt)

    return PortfolioResult(
        weights=weights,
        expected_return=round(ret_opt, 4),
        expected_vol=round(vol_opt, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=round(div_ratio, 3),
        method="risk_parity",
        top_holdings=top,
        n_positions=len(weights),
        effective_n=round(1.0 / hhi if hhi > 0 else N, 1),
        turnover=round(turnover, 4),
        estimated_cost=round(estimated_cost, 6),
        status=status,
        warnings=warnings,
    )


# ════════════════════════════════════════════════════════════════
# HRP (Hierarchical Risk Parity)
# ════════════════════════════════════════════════════════════════

def _quasi_diag(link: np.ndarray) -> list[int]:
    """
    HRP 准对角化: 将树状聚类的叶子按聚类顺序排列。
    link: scipy linkage matrix
    Returns: 叶子索引列表 (按聚类顺序)
    """
    from scipy.cluster.hierarchy import ClusterNode, to_tree
    tree = to_tree(link, rd=False)
    order: list[int] = []

    def _traverse(node: ClusterNode):
        if node.is_leaf():
            order.append(node.id)
            return
        _traverse(node.get_left())
        _traverse(node.get_right())

    _traverse(tree)
    return order


def _get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """计算一个聚类的方差 (等权组合)"""
    if not items:
        return 0
    w = np.array([1.0 / len(items)] * len(items))
    sub_cov = cov[np.ix_(items, items)]
    return w @ sub_cov @ w


def optimize_hrp(
    returns: pd.DataFrame,
    linkage_method: str = 'ward',
    current_weights: np.ndarray | None = None,
    cost_bps: float = 10,
) -> PortfolioResult:
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016).

    三步骤:
      1. 层次聚类 (correlation-based)
      2. 准对角化
      3. 递归二分拆分方差
    """
    try:
        from scipy.cluster.hierarchy import linkage
    except ImportError:
        # Fallback to equal weight if scipy not available
        return _equal_weight_fallback(returns, "hrp_no_scipy")

    symbols = returns.columns.tolist()
    N = len(symbols)
    if N == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="hrp_empty")
    if N == 1:
        return PortfolioResult(weights={symbols[0]: 1.0}, expected_return=0,
                               expected_vol=0, sharpe_ratio=0,
                               diversification_ratio=1.0, method="hrp_single",
                               top_holdings=[(symbols[0], 1.0)], n_positions=1)

    cov = _cov_shrinkage(returns, alpha=0.15) * 252  # V4.1 fix: 年化协方差(与mean_ret一致)
    corr = _cov_to_corr(cov)

    # Step 1: 层次聚类
    dist = np.sqrt(2 * (1 - corr))
    dist = (dist + dist.T) / 2  # 对称
    np.fill_diagonal(dist, 0)
    link = linkage(dist[np.triu_indices(N, k=1)], method=linkage_method)

    # Step 2: 准对角化
    order = _quasi_diag(link)

    # Step 3: 递归二分分配
    weights = np.ones(N)
    clusters = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) == 1:
            continue
        # 分成两半
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]

        # 等权组合方差
        var_left = _get_cluster_var(cov, left)
        var_right = _get_cluster_var(cov, right)

        # 方差分配 (alpha_1 / (alpha_1 + alpha_2))
        alpha1 = 1 - var_left / (var_left + var_right + 1e-10)
        alpha2 = 1 - alpha1

        # 更新权重
        for i in left:
            weights[i] *= alpha1
        for i in right:
            weights[i] *= alpha2

        clusters.extend([left, right])

    w_opt = weights
    mean_ret = returns.mean().values * 252  # V4.1 fix: 年化收益率(cov已行406年化,一致)
    ret_gross, var_opt = _portfolio_stats(w_opt, mean_ret, cov)
    turnover, estimated_cost = _transaction_cost_metrics(current_weights, w_opt, cost_bps)
    ret_opt = ret_gross - estimated_cost
    vol_opt = np.sqrt(var_opt) if var_opt > 0 else 0
    sharpe = (ret_opt - 0.025) / vol_opt if vol_opt > 0 else 0

    weighted_avg_vol = w_opt @ np.sqrt(np.diag(cov))
    div_ratio = weighted_avg_vol / vol_opt if vol_opt > 0 else 0

    weights_dict = {s: round(w, 4) for s, w in zip(symbols, w_opt) if w > 0.001}
    top = sorted(weights_dict.items(), key=lambda x: -x[1])[:5]
    hhi = sum(w ** 2 for w in w_opt)

    return PortfolioResult(
        weights=weights_dict,
        expected_return=round(ret_opt, 4),
        expected_vol=round(vol_opt, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=round(div_ratio, 3),
        method="hrp",
        top_holdings=top,
        n_positions=len(weights_dict),
        effective_n=round(1.0 / hhi if hhi > 0 else N, 1),
        turnover=round(turnover, 4),
        estimated_cost=round(estimated_cost, 6),
    )


# ════════════════════════════════════════════════════════════════
# Black-Litterman 模型
# ════════════════════════════════════════════════════════════════

def optimize_black_litterman(
    returns: pd.DataFrame,
    views: dict[str, float],
    view_confidences: dict[str, float],
    market_cap_weights: Optional[dict[str, float]] = None,
    tau: float = 0.05,
    risk_free: float = 0.025,
    max_weight: float = 0.20,
) -> PortfolioResult:
    """
    Black-Litterman 模型: 市场均衡 + 主观观点 → 后验收益。

    Args:
        returns: T×N 收益率
        views: {symbol: expected_excess_return} 主观观点
        view_confidences: {symbol: 0-1 置信度}
        market_cap_weights: {symbol: 市值权重} 市场组合 (None=等权)
        tau: 观点不确定性标量 (越小越信任观点)
        risk_free: 无风险利率
        max_weight: 最大个股权重
    """
    symbols = returns.columns.tolist()
    N = len(symbols)
    if N == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="bl_empty")

    mean_ret = returns.mean().values * 252  # V4.1 fix: 年化收益率
    cov = _cov_shrinkage(returns, alpha=0.15) * 252  # V4.1 fix: 年化协方差(一致)

    # Step 1: 先验收益 (市场均衡)
    if market_cap_weights:
        mkt_w = np.array([market_cap_weights.get(s, 0) for s in symbols])
        mkt_w = mkt_w / mkt_w.sum() if mkt_w.sum() > 0 else np.ones(N) / N
    else:
        mkt_w = np.ones(N) / N

    # 先验: 市场隐含超额收益
    risk_aversion = 2.5  # 市场风险厌恶系数
    pi = risk_aversion * cov @ mkt_w  # 先验超额收益

    # Step 2: 观点矩阵
    view_symbols = [s for s in symbols if s in views]
    K = len(view_symbols)
    if K == 0:
        # 无观点: 直接返回市场组合
        return optimize_mean_variance(returns, target_sharpe=False,
                                      max_weight=max_weight)

    P = np.zeros((K, N))  # 观点连接矩阵
    Q = np.zeros(K)       # 观点收益向量
    Omega = np.zeros((K, K))  # 观点协方差

    for i, sym in enumerate(view_symbols):
        idx = symbols.index(sym)
        P[i, idx] = 1.0
        Q[i] = views[sym]
        # 观点不确定性: 置信度越低, 方差越大
        # P2-Q16-fix (M119): conf=0 时 (1-conf)/conf 除零。夹取到 [1e-6, 1]
        # 保证观点方差有限且为正（conf=1 表示完全确定，方差→0 的极限用下界兜底）。
        conf = min(max(float(view_confidences.get(sym, 0.5)), 1e-6), 1.0)
        Omega[i, i] = (1 - conf) / conf * np.diag(cov)[idx] * tau

    # Step 3: 后验收益
    # μ = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}π + P'Ω^{-1}Q]
    tau_cov_inv = np.linalg.inv(tau * cov + np.eye(N) * 1e-8)
    if K > 0:
        omega_inv = np.linalg.inv(Omega + np.eye(K) * 1e-8)
        # 后验协方差
        M_inv = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P + np.eye(N) * 1e-10)
        posterior_mean = M_inv @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
    else:
        posterior_mean = pi

    # Step 4: 优化后验收益
    # 最大化后验夏普比
    def neg_sharpe_bl(w):
        ret = w @ posterior_mean
        var = w @ cov @ w
        vol = np.sqrt(var) if var > 0 else 1e-10
        return -(ret - risk_free) / vol

    bounds = Bounds([0.0] * N, [max_weight] * N)
    constraints = [LinearConstraint(np.ones(N), 1.0, 1.0)]
    x0 = np.array([1.0 / N] * N)

    # V11 审计修复（Medium）: 原实现无可行性预检——N=3、max_weight=0.2 时
    # 等权 x0=0.333 超限（其余优化器都有 N*max_weight>=1 预检并自动放宽）。
    # 修正: 不可行时自动放宽 max_weight 到 1/N。
    eff_max = max_weight
    if N * max_weight < 1.0:
        eff_max = 1.0 / N + 1e-6
    bounds = Bounds([0.0] * N, [eff_max] * N)
    result = minimize(neg_sharpe_bl, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-10})
    w_opt = result.x if result.success else x0
    # 兜底: 若仍不可行（数值问题），降级为等权并 clip 到边界
    if w_opt.sum() <= 0 or np.any(w_opt < -1e-8):
        w_opt = np.full(N, 1.0 / N)
    w_opt = np.clip(w_opt, 0, eff_max)
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()

    ret_opt, var_opt = _portfolio_stats(w_opt, posterior_mean, cov)
    vol_opt = np.sqrt(var_opt) if var_opt > 0 else 0
    sharpe = (ret_opt - risk_free) / vol_opt if vol_opt > 0 else 0

    weights = {s: round(w, 4) for s, w in zip(symbols, w_opt) if w > 0.001}
    top = sorted(weights.items(), key=lambda x: -x[1])[:5]
    hhi = sum(w ** 2 for w in w_opt)

    return PortfolioResult(
        weights=weights,
        expected_return=round(ret_opt, 4),
        expected_vol=round(vol_opt, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=(w_opt @ np.sqrt(np.diag(cov))) / vol_opt if vol_opt > 0 else 0,
        method="black_litterman",
        top_holdings=top,
        n_positions=len(weights),
        effective_n=round(1.0 / hhi if hhi > 0 else N, 1),
    )


# ════════════════════════════════════════════════════════════════
# CVaR 优化
# ════════════════════════════════════════════════════════════════

def compute_cvar(weights: np.ndarray, returns: pd.DataFrame, alpha: float = 0.95) -> float:
    """给定权重计算历史场景组合 CVaR。

    P2-Q16-fix (M121): 明确 α 语义 —— 此处 alpha 是**置信度**（confidence level，
    0.95 = 95% CVaR，取最差 1-alpha=5% 场景的均值亏损）。与
    portfolio_optimizer.cvar_optimize 的 alpha=尾部概率 约定不同，调用方勿混用。
    """
    if returns.empty:
        return 0.0
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != returns.shape[1]:
        raise ValueError(f"weights length {w.shape[0]} != asset count {returns.shape[1]}")
    portfolio_returns = returns.fillna(0.0).values @ w
    losses = -portfolio_returns
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean() if len(tail) else var)


def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    current_weights: np.ndarray | None = None,
    cost_bps: float = 10,
    max_weight: float = 1.0,
    allow_short: bool = False,
) -> PortfolioResult:
    """使用 Rockafellar-Uryasev 公式最小化历史场景 CVaR。

    P2-Q16-fix (M121): 与 compute_cvar 一致，alpha 为**置信度**（0.95 = 95% CVaR），
    beta=1/((1-alpha)*T) 中的 (1-alpha) 即尾部概率。与 portfolio_optimizer.cvar_optimize
    的 alpha=尾部概率约定不同，调用方勿混用。
    """
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError as exc:
        raise ImportError("optimize_cvar requires scipy.optimize; please install scipy") from exc

    clean_returns = returns.dropna(how="all").fillna(0.0)
    symbols = clean_returns.columns.tolist()
    n_assets = len(symbols)
    n_scenarios = len(clean_returns)
    if n_assets == 0 or n_scenarios == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method="cvar_empty")

    current = _normalize_current_weights(current_weights, n_assets)
    scenarios = clean_returns.values
    mean_ret = clean_returns.mean().values * 252  # V4.1 fix: 年化收益率(与cov一致)
    cov = _cov_shrinkage(clean_returns, alpha=0.15) * 252  # V4.1 fix: 年化协方差(一致)
    beta = 1.0 / ((1.0 - alpha) * n_scenarios)

    def objective(x: np.ndarray) -> float:
        weights = x[:n_assets]
        eta = x[n_assets]
        u = x[n_assets + 1:]
        cvar_obj = eta + beta * np.sum(u)
        _, cost = _transaction_cost_metrics(current, weights, cost_bps)
        # P2-Q16-fix (L136): 原系数 1e-4 使成本惩罚数值上可忽略（10bps 全换手时
        # 1e-4*0.001=1e-7，相对日 CVaR(~0.02) 无任何约束力），"成本感知"名存实亡。
        # 权重提到 1.0：cost 为实际换手成本（小数），全换手约 10bps=0.001，占日
        # CVaR 的 ~5%，成为有意义的换手惩罚。'+'保持成本为正惩罚（若为'-'则鼓励换手）。
        return float(cvar_obj + 1.0 * cost)

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n_assets]) - 1.0},
    ]
    for i in range(n_scenarios):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i: x[n_assets + 1 + i] - (-scenarios[i] @ x[:n_assets] - x[n_assets]),
        })

    if allow_short:
        bounds = [(-max_weight, max_weight)] * n_assets + [(None, None)] + [(0.0, None)] * n_scenarios
    else:
        bounds = [(0.0, max_weight)] * n_assets + [(None, None)] + [(0.0, None)] * n_scenarios

    x0_weights = np.ones(n_assets) / n_assets
    losses0 = -(scenarios @ x0_weights)
    eta0 = float(np.quantile(losses0, alpha))
    u0 = np.maximum(losses0 - eta0, 0.0)
    x0 = np.concatenate([x0_weights, [eta0], u0])

    result = scipy_minimize(objective, x0, method="SLSQP", bounds=bounds,
                            constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    w_opt = result.x[:n_assets] if result.success else x0_weights
    w_opt = np.clip(w_opt, -max_weight if allow_short else 0.0, max_weight)
    w_opt = w_opt / w_opt.sum() if w_opt.sum() != 0 else x0_weights

    ret_gross, var_opt = _portfolio_stats(w_opt, mean_ret, cov)
    turnover, estimated_cost = _transaction_cost_metrics(current, w_opt, cost_bps)
    ret_opt = ret_gross - estimated_cost
    vol_opt = np.sqrt(var_opt) if var_opt > 0 else 0.0
    sharpe = (ret_opt - 0.025) / vol_opt if vol_opt > 0 else 0.0
    cvar_val = compute_cvar(w_opt, clean_returns, alpha)
    weighted_avg_vol = w_opt @ np.sqrt(np.diag(cov))
    div_ratio = weighted_avg_vol / vol_opt if vol_opt > 0 else 0.0

    weights_dict = {s: round(float(w), 4) for s, w in zip(symbols, w_opt) if abs(w) > 0.001}
    top = sorted(weights_dict.items(), key=lambda x: -abs(x[1]))[:5]
    hhi = sum(float(w) ** 2 for w in w_opt)
    return PortfolioResult(
        weights=weights_dict,
        expected_return=round(ret_opt, 4),
        expected_vol=round(vol_opt, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=round(div_ratio, 3),
        method="cvar",
        top_holdings=top,
        n_positions=len(weights_dict),
        effective_n=round(1.0 / hhi if hhi > 0 else n_assets, 1),
        turnover=round(turnover, 4),
        estimated_cost=round(estimated_cost, 6),
        cvar=round(cvar_val, 6),
    )


# ════════════════════════════════════════════════════════════════
# 凯利公式仓位计算
# ════════════════════════════════════════════════════════════════

@dataclass
class KellyResult:
    """凯利公式结果"""
    symbol: str
    win_rate: float          # 胜率
    avg_win: float           # 平均盈利率 (%)
    avg_loss: float          # 平均亏损率 (%)
    payoff_ratio: float      # 盈亏比
    full_kelly: float        # 全凯利仓位 (%)
    half_kelly: float        # 半凯利仓位 (%)
    n_samples: int           # 样本数


def compute_kelly(
    symbol: str,
    historical_trades: list[dict],
    max_kelly_pct: float = 0.25,
    half_kelly: bool = True,
) -> KellyResult:
    """
    凯利公式计算最优仓位。

    f* = (p * b - q) / b
    其中 p=胜率, q=1-p, b=盈亏比(平均盈利/平均亏损)

    建议用半凯利 (f*/2) 保守下注。

    Args:
        symbol: 股票代码
        historical_trades: 历史交易列表 [{"pnl_pct": ..., "outcome": "win"/"loss"}, ...]
        max_kelly_pct: 最大仓位上限
        half_kelly: 是否用半凯利
    """
    if not historical_trades:
        return KellyResult(
            symbol=symbol, win_rate=0, avg_win=0, avg_loss=0,
            payoff_ratio=0, full_kelly=0, half_kelly=0, n_samples=0,
        )

    # V11 审计修复（Medium）: t.get("pnl_pct",0) 在键存在但值为 None 时
    # None>0 抛 TypeError（外部数据如 simulation_state.json 常见）。
    # 修正: 用安全数值转换，None/NaN 视为 0。
    def _pnl(t):
        v = t.get("pnl_pct", 0)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    wins = [t for t in historical_trades if _pnl(t) > 0]
    losses = [t for t in historical_trades if _pnl(t) < 0]

    # P2-Q16-fix (L130): pnl_pct==0 的平局单不计入胜/负，也不计入分母。
    # 旧实现把 0 收益单计入 n_total 分母 → 胜率被系统性低估（平局越多偏差越大）。
    n_total = len(wins) + len(losses)
    win_rate = len(wins) / n_total if n_total > 0 else 0

    avg_win = sum(_pnl(t) for t in wins) / len(wins) / 100 if wins else 0
    avg_loss = abs(sum(_pnl(t) for t in losses) / len(losses)) / 100 if losses else 0.01

    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1

    # 全凯利
    if payoff_ratio > 0:
        full_kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
    else:
        full_kelly = 0

    full_kelly = max(0, min(full_kelly, max_kelly_pct))
    half = full_kelly / 2 if half_kelly else full_kelly

    return KellyResult(
        symbol=symbol, win_rate=round(win_rate, 3),
        avg_win=round(avg_win * 100, 2), avg_loss=round(avg_loss * 100, 2),
        payoff_ratio=round(payoff_ratio, 2),
        full_kelly=round(full_kelly, 4),
        half_kelly=round(half, 4),
        n_samples=n_total,
    )


# ════════════════════════════════════════════════════════════════
# 再平衡策略
# ════════════════════════════════════════════════════════════════

@dataclass
class RebalanceResult:
    """再平衡结果"""
    trades: list[dict]                   # 需要执行的交易
    turnover: float                       # 换手率 (%)
    current_weights: dict[str, float]     # 当前权重
    target_weights: dict[str, float]      # 目标权重
    deviation_pct: float                  # 最大偏离 (%)
    reason: str                           # 再平衡原因


def compute_rebalance(
    current_positions: list[dict],
    target_weights: dict[str, float],
    portfolio_value: float,
    threshold: float = 0.05,
    min_trade_value: float = 1000.0,
) -> RebalanceResult:
    """
    计算再平衡交易。

    Args:
        current_positions: 当前持仓 [{"symbol", "total_value", "current_price"}, ...]
        target_weights: {symbol: weight} 目标权重
        portfolio_value: 总资产值
        threshold: 偏离超过此比例才调仓 (5%)
        min_trade_value: 最小交易金额 (避免摩擦成本)

    Returns: RebalanceResult
    """
    current_weights = {}
    for p in current_positions:
        sym = p["symbol"]
        val = p.get("total_value", 0) or 0
        current_weights[sym] = val / portfolio_value if portfolio_value > 0 else 0

    # 计算偏离
    deviations = {}
    trades = []
    max_deviation = 0

    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    for sym in all_symbols:
        cw = current_weights.get(sym, 0)
        tw = target_weights.get(sym, 0)
        dev = tw - cw
        deviations[sym] = dev
        max_deviation = max(max_deviation, abs(dev))

        if abs(dev) > threshold:
            trade_value = dev * portfolio_value
            if abs(trade_value) >= min_trade_value:
                trades.append({
                    "symbol": sym,
                    "action": "buy" if dev > 0 else "sell",
                    "current_weight": round(cw, 4),
                    "target_weight": round(tw, 4),
                    "deviation": round(dev, 4),
                    "trade_value": round(trade_value, 2),
                })

    # P2-Q16-fix (L129): 换手率只统计实际执行的交易（超过阈值且达到 min_trade_value）。
    # 旧实现按全部偏离求和——包含未触发阈值/未达 min_trade_value 未执行的偏离，
    # 换手率系统性高估。
    turnover = sum(abs(t["deviation"]) for t in trades) / 2 * 100

    reason = "阈值触发" if max_deviation > threshold else "无需调整"

    return RebalanceResult(
        trades=sorted(trades, key=lambda x: -abs(x["deviation"])),
        turnover=round(turnover, 2),
        current_weights={k: round(v, 4) for k, v in current_weights.items()},
        target_weights={k: round(v, 4) for k, v in target_weights.items()},
        deviation_pct=round(max_deviation * 100, 2),
        reason=reason,
    )


# ════════════════════════════════════════════════════════════════
# 快捷接口
# ════════════════════════════════════════════════════════════════

def compute_portfolio(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    regime: str = "bull",
    **kwargs,
) -> PortfolioResult:
    """
    统一接口: 按方法名选择优化器。

    Args:
        returns: T×N 收益率DataFrame
        method: "max_sharpe" | "min_vol" | "risk_parity" | "hrp" | "black_litterman" | "cvar" | "regime"
        regime: 市场状态，method="regime" 或 method="auto" 时生效
        **kwargs: 传递给具体优化器的参数

    Returns: PortfolioResult
    """
    if returns.empty or returns.shape[1] == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0,
                               method=f"{method}_empty")

    if method in {"regime", "auto", "regime_aware"}:
        return regime_aware_portfolio(returns, regime=regime, current_weights=kwargs.pop("current_weights", None),
                                      cost_bps=kwargs.pop("cost_bps", 10))
    if method == "max_sharpe":
        return optimize_mean_variance(returns, target_sharpe=True, **kwargs)
    elif method == "min_vol":
        return optimize_mean_variance(returns, target_sharpe=False, **kwargs)
    elif method == "risk_parity":
        return optimize_risk_parity(returns, **kwargs)
    elif method == "hrp":
        return optimize_hrp(returns, **kwargs)
    elif method == "cvar":
        return optimize_cvar(returns, **kwargs)
    elif method == "black_litterman":
        views = kwargs.pop("views", {})
        view_confidences = kwargs.pop("view_confidences", {})
        return optimize_black_litterman(returns, views, view_confidences, **kwargs)
    elif method == "equal_weight" or method == "ew":
        return _equal_weight_fallback(returns, method)
    else:
        return _equal_weight_fallback(returns, method)


def regime_aware_portfolio(
    returns: pd.DataFrame,
    regime: str,
    current_weights: np.ndarray | None = None,
    cost_bps: float = 10,
) -> PortfolioResult:
    """根据市场状态自动选择组合优化方法，并应用交易成本。"""
    regime_key = str(regime).lower()
    if regime_key in {"bull", "牛市"}:
        return optimize_mean_variance(returns, target_sharpe=True, max_weight=1.0,
                                      current_weights=current_weights, cost_bps=cost_bps)
    if regime_key in {"bear", "熊市"}:
        return optimize_mean_variance(returns, target_sharpe=False, max_weight=0.25,
                                      max_vol=0.15, current_weights=current_weights,
                                      cost_bps=cost_bps)
    if regime_key in {"震荡", "sideways", "range"}:
        return optimize_risk_parity(returns, current_weights=current_weights, cost_bps=cost_bps)
    if regime_key in {"high_vol", "高波动", "volatile"}:
        return optimize_cvar(returns, current_weights=current_weights, cost_bps=cost_bps)
    return optimize_risk_parity(returns, current_weights=current_weights, cost_bps=cost_bps)


def _weights_dict_to_array(weights: dict[str, float], symbols: list[str]) -> np.ndarray:
    """按 symbols 顺序将权重字典转换为数组。"""
    arr = np.array([float(weights.get(symbol, 0.0)) for symbol in symbols])
    total = arr.sum()
    return arr / total if total > 0 else np.ones(len(symbols)) / len(symbols)


def _portfolio_performance(nav: pd.Series, returns: pd.Series) -> dict[str, float]:
    """计算净值序列对应的核心绩效指标。"""
    if nav.empty or returns.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    # P2-Q16-fix (L127): 用 return_series 累乘计算总收益。旧实现 nav[-1]/nav[0]-1
    # 漏掉首日收益（nav 首值即 1+r₀，相除后 r₀ 被抵消），与 sharpe/vol 用全序列
    # returns 的口径不一致。prod(1+returns)-1 覆盖全部交易日。
    total_return = float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)
    years = max(len(returns) / 252, 1 / 252)
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    annual_vol = float(returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    drawdown = nav / nav.cummax() - 1
    return {
        "annual_return": round(float(annual_return), 4),
        "annual_vol": round(annual_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(float(drawdown.min()), 4),
    }


def rebalance_backtest(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    rebalance_freq: str = "monthly",
    lookback: int = 252,
    cost_bps: float = 10,
    regime_series: pd.Series | None = None,
) -> dict:
    """多期再平衡回测：再平衡日前用历史窗口优化，持有期内权重随收益漂移。"""
    data = returns.sort_index().dropna(how="all").fillna(0.0)
    symbols = data.columns.tolist()
    n_assets = len(symbols)
    if data.empty or n_assets == 0 or len(data) <= lookback:
        return {"status": "insufficient_data", "weights": {}, "nav": pd.Series(dtype=float)}

    freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE"}
    rule = freq_map.get(rebalance_freq, rebalance_freq)
    scheduled = data.resample(rule).last().index
    rebalance_dates = [data.index[data.index.searchsorted(dt)] for dt in scheduled
                       if data.index.searchsorted(dt) < len(data) and data.index.searchsorted(dt) >= lookback]
    if not rebalance_dates:
        rebalance_dates = [data.index[lookback]]

    current_weights = np.ones(n_assets) / n_assets
    nav = 1.0
    nav_values: list[float] = []
    port_returns: list[float] = []
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    next_rebalance = set(rebalance_dates)

    for i, date in enumerate(data.index):
        if i >= lookback and date in next_rebalance:
            hist = data.iloc[i - lookback:i]
            regime = regime_series.loc[:date].iloc[-1] if regime_series is not None and not regime_series.loc[:date].empty else "bull"
            if regime_series is not None or method in {"regime", "auto", "regime_aware"}:
                result = regime_aware_portfolio(hist, str(regime), current_weights=current_weights, cost_bps=cost_bps)
            else:
                result = compute_portfolio(hist, method=method, current_weights=current_weights, cost_bps=cost_bps)
            target_weights = _weights_dict_to_array(result.weights, symbols)
            turnover, cost = _transaction_cost_metrics(current_weights, target_weights, cost_bps)
            nav *= (1 - cost)
            current_weights = target_weights
            weights_by_date[date] = {s: round(float(w), 4) for s, w in zip(symbols, current_weights)}
            turnovers[date] = round(turnover, 4)

        day_ret = float(data.loc[date].values @ current_weights)
        nav *= (1 + day_ret)
        port_returns.append(day_ret)
        nav_values.append(nav)
        drifted = current_weights * (1 + data.loc[date].values)
        current_weights = drifted / drifted.sum() if drifted.sum() > 0 else np.ones(n_assets) / n_assets

    nav_series = pd.Series(nav_values, index=data.index, name="nav")
    return_series = pd.Series(port_returns, index=data.index, name="returns")
    metrics = _portfolio_performance(nav_series, return_series)
    return {
        "status": "ok",
        "weights": weights_by_date,
        "nav": nav_series,
        "returns": return_series,
        "turnover": pd.Series(turnovers, name="turnover"),
        **metrics,
    }


def _equal_weight_fallback(returns: pd.DataFrame, method: str) -> PortfolioResult:
    """等权组合 (fallback)"""
    symbols = returns.columns.tolist()
    N = len(symbols)
    if N == 0:
        return PortfolioResult(weights={}, expected_return=0, expected_vol=0,
                               sharpe_ratio=0, diversification_ratio=0, method=method)

    w = np.ones(N) / N
    cov = _cov_shrinkage(returns, alpha=0.15) * 252  # V4.1 fix: 年化协方差
    mean_ret = returns.mean().values * 252  # V4.1 fix: 年化收益率(与cov一致)
    ret, var = _portfolio_stats(w, mean_ret, cov)
    vol = np.sqrt(var) if var > 0 else 0
    sharpe = (ret - 0.025) / vol if vol > 0 else 0

    weights = {s: 1.0 / N for s in symbols}
    return PortfolioResult(
        weights=weights,
        expected_return=round(ret, 4),
        expected_vol=round(vol, 4),
        sharpe_ratio=round(sharpe, 3),
        diversification_ratio=(w @ np.sqrt(np.diag(cov))) / vol if vol > 0 else 0,
        method=method,
        top_holdings=[(s, 1.0 / N) for s in symbols[:5]],
        n_positions=N,
        effective_n=float(N),
    )


def format_portfolio_result(result: PortfolioResult) -> str:
    """格式化为可读字符串"""
    lines = [
        f"📊 组合优化: {result.method}",
        f"  预期年化收益: {result.expected_return*100:.1f}%",
        f"  预期年化波动: {result.expected_vol*100:.1f}%",
        f"  夏普比: {result.sharpe_ratio:.2f}",
        f"  分散度: {result.diversification_ratio:.2f}x",
        f"  持仓数: {result.n_positions} (有效: {result.effective_n:.0f})",
        "",
        "  前5持仓:",
    ]
    for sym, w in result.top_holdings:
        lines.append(f"    {sym} {w*100:.1f}%")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="组合优化引擎")
    parser.add_argument("--method", default="risk_parity",
                        choices=["max_sharpe", "min_vol", "risk_parity", "hrp",
                                 "equal_weight"],
                        help="优化方法")
    parser.add_argument("--symbols", nargs="+",
                        default=["600519", "000858", "002714", "601899",
                                 "600036", "601166", "600900", "000333"],
                        help="股票代码列表")
    parser.add_argument("--days", type=int, default=252, help="回看天数")
    args = parser.parse_args()

    print(f"组合优化: {args.method}")
    print(f"标的: {args.symbols}")
    print()

    # 获取历史数据
    import akshare as ak
    all_returns = {}
    for sym in args.symbols:
        try:
            df = ak.stock_zh_a_hist(sym, "daily", adjust="qfq")
            if df is not None and len(df) > args.days:
                df = df.tail(args.days + 1)
                close = df["收盘"].values.astype(float)
                ret = np.diff(close) / close[:-1]
                all_returns[sym] = ret[-args.days:]
                print(f"  {sym}: {len(ret[-args.days:])}天")
        except Exception as e:
            print(f"  {sym}: ❌ {e}")

    if len(all_returns) < 2:
        print("至少需要2只股票")
        sys.exit(1)

    ret_df = pd.DataFrame(all_returns)
    result = compute_portfolio(ret_df, method=args.method)
    print()
    print(format_portfolio_result(result))
