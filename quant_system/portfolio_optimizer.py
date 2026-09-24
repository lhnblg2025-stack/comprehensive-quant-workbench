"""
V7 Portfolio Optimizer
======================
Portfolio optimization engine using scipy.optimize.
No cvxpy dependency.

Functions:
  - mean_variance_optimize           Max Sharpe / min volatility / target-return
  - black_litterman                  Simplify Black–Litterman with ML-signal views
  - equal_risk_contribution          ERC (risk-parity) portfolio
  - efficient_frontier               50-point frontier
  - portfolio_summary                Expected return, volatility, Sharpe, DR, max DD

D6收敛登记 (2026-08-11): 风控域收敛（保守策略）——本模块为组合优化生产真源
  （唯一调用方 asset_allocation.py）；均值方差/ERC/前沿面/汇总与
  portfolio_v2.py(DEPRECATED)、portfolio.py(DEPRECATED) 为同名异签名异实现 → 不迁移；
  cvar_optimize α=尾部概率 与 portfolio_v2.calculate_cvar(α=置信度+√ppy年化)/
  risk_management_pro._historical_var_cvar/portfolio_risk.compute_var 异口径 → 标注保留。
"""

from __future__ import annotations
import logging

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
# P2-Q16-fix (L133): 不再全局打补丁 np.asfarray / np.Inf / np.infinity（旧补丁会
# 影响进程内所有模块，属全局副作用）。本模块并未使用这三个符号，直接用标准
# np.asarray / np.inf 即可；若未来需要兼容旧式写法，应在使用处做局部别名。
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELTA: float = 2.5       # risk-aversion coefficient (BL)
TAU: float = 0.05        # prior uncertainty scalar (BL)
MAX_ITER = 2000
FRONTIER_POINTS = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_arrays(
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    """Align expected_returns and cov_matrix by common tickers."""
    er = pd.Series(expected_returns)
    common = er.index.intersection(cov_matrix.index).intersection(cov_matrix.columns)
    if len(common) == 0:
        raise ValueError("No common assets between expected_returns and cov_matrix.")
    er = er[common].values.astype(np.float64)
    cov = cov_matrix.loc[common, common].values.astype(np.float64)
    return common, er, cov


def _neg_sharpe(weights: np.ndarray, er: np.ndarray, cov: np.ndarray, rf: float) -> float:
    """Negative Sharpe ratio for minimisation (long-only weights assumed)."""
    port_ret = weights @ er
    port_vol = math.sqrt(weights @ cov @ weights)
    if port_vol < 1e-12:
        return 0.0
    return -(port_ret - rf) / port_vol


def _portfolio_metrics(
    weights: np.ndarray,
    er: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
) -> Dict[str, float]:
    """Compute standard portfolio metrics from weights."""
    w = np.asarray(weights, dtype=np.float64)
    port_ret = float(w @ er)
    port_var = float(w @ cov @ w)
    port_vol = math.sqrt(max(port_var, 0.0))
    sharpe = (port_ret - rf) / port_vol if port_vol > 1e-12 else 0.0

    # Diversification ratio = weighted avg vol / portfolio vol
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(w @ asset_vols)
    dr = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

    return {
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
        "diversification_ratio": dr,
        "variance": port_var,
    }


def _make_constraints(
    cov_matrix: pd.DataFrame,
    n: int,
    tickers: pd.Index,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
) -> List:
    """Build constraints list for scipy.optimize.minimize.

    Always includes:
      - sum(w) == 1
      - 0 <= w_i <= asset_ceiling
      - sector concentration <= sector_ceiling  (if sector_map provided)
    """
    # 1. Equality constraint: sum(w) == 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # 2. Sector-concentration inequality (if we have a sector map)
    if sector_map is not None:
        sectors = pd.Series(sector_map)
        # Ensure we only use tickers actually in the cov_matrix / tickers
        sector_of = sectors.reindex(tickers).fillna("_Other_")
        unique_sectors = sector_of.unique()
        for sec in unique_sectors:
            idx = np.where(sector_of.values == sec)[0]
            if len(idx) == 0:
                continue
            # sum_{i in sector} w_i <= sector_ceiling
            A = np.zeros(n)
            A[idx] = 1.0
            constraints.append(LinearConstraint(A, -np.inf, sector_ceiling))

    return constraints


def _make_bounds(n: int, asset_ceiling: float = 0.10) -> Bounds:
    """Long-only with individual asset cap."""
    return Bounds(np.zeros(n), np.full(n, asset_ceiling))


def _try_solve(
    obj_fun,
    x0: np.ndarray,
    bounds: Bounds,
    constraints: List,
    method: str = "SLSQP",
) -> Tuple[bool, np.ndarray, float]:
    """Wrapper around scipy.optimize.minimize with fallback."""
    for _ in range(3):  # retry with perturbations
        res = minimize(
            obj_fun,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": MAX_ITER, "ftol": 1e-10},
        )
        if res.success:
            return True, res.x, res.fun
        # perturb start
        x0 = np.clip(x0 + np.random.uniform(-0.02, 0.02, size=x0.shape), 0, 1)
        x0 /= x0.sum()
    return False, res.x, res.fun  # return final attempt anyway


# P1-Q16-fix: 求解前可行性预检。Σw=1 与 {0 ≤ w_i ≤ asset_ceiling} 的交集
# 非空当且仅当总容量 ≥ 1（等权点必须落在上限内，即 N*ceiling ≥ 1）。
# 不可行时自动放宽到最小可行上限并返回告警文案，禁止在求解失败后静默归一化
# 出超限权重（V5.4: N=5/ceiling=0.10 被归一化到全 0.20，status 仍 converged_with_warning）。
def _resolve_caps(
    n: int,
    tickers: pd.Index,
    asset_ceiling: float,
    sector_map: Optional[Dict[str, str]],
    sector_ceiling: float,
) -> Tuple[float, float, List[str]]:
    """Return (effective_asset_ceiling, effective_sector_ceiling, issues).

    If the requested caps make the feasible set empty, relax them to the
    smallest values that restore feasibility and record a human-readable issue
    so callers can surface it in the status.
    """
    issues: List[str] = []
    eff_asset = float(asset_ceiling)
    eff_sector = float(sector_ceiling)
    if n <= 0:
        return eff_asset, eff_sector, ["no assets to optimise (N=0)"]

    floor_asset = 1.0 / n
    if n * eff_asset < 1.0 - 1e-12:
        eff_asset = floor_asset
        issues.append(
            f"asset_ceiling={asset_ceiling} infeasible for N={n} "
            f"(requires N*ceiling>=1); relaxed to {eff_asset:.6f}"
        )

    if sector_map is not None:
        sectors = pd.Series(sector_map).reindex(tickers).fillna("_Other_")
        # Total weight capacity under current caps: each constrained sector can
        # hold at most min(m*asset_ceiling, sector_ceiling); "_Other_" is uncapped
        # by sector and holds m*asset_ceiling.
        capacity = 0.0
        for sec, group in sectors.groupby(sectors.values):
            m = len(group)
            if sec == "_Other_":
                capacity += m * eff_asset
            else:
                capacity += min(m * eff_asset, eff_sector)
        if capacity < 1.0 - 1e-12:
            eff_sector = 1.0  # make sector constraint non-binding -> feasible again
            issues.append(
                f"sector_ceiling={sector_ceiling} infeasible for N={n} "
                f"(total capacity {capacity:.3f}<1); relaxed to {eff_sector:.6f}"
            )
    return eff_asset, eff_sector, issues


# P1-Q16-fix: 候选解校验。仅当 Σw≈1、0≤w≤ceiling、(可选) w@er≈target 时
# 才接受该点为可行解；否则视为无效（不再静默当作最优点输出）。
def _verify_solution(
    w: np.ndarray,
    er: np.ndarray,
    target: Optional[float],
    asset_ceiling: float,
    sum_tol: float = 1e-3,
    ceiling_tol: float = 1e-6,
) -> bool:
    if w is None or not np.all(np.isfinite(w)):
        return False
    if abs(float(np.sum(w)) - 1.0) > sum_tol:
        return False
    if np.any(w < -ceiling_tol) or np.any(w > asset_ceiling + ceiling_tol):
        return False
    if target is not None:
        if abs(float(w @ er) - target) > max(1e-4, abs(target) * 1e-3):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Mean-Variance Optimization
# ---------------------------------------------------------------------------

def _equal_weight_x0(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def mean_variance_optimize(
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
    risk_target: Optional[float] = None,
    target_return: Optional[float] = None,
    objective: str = "max_sharpe",
    rf: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
) -> Dict[str, Any]:
    """Mean-Variance portfolio optimisation.

    Parameters
    ----------
    objective : str
        One of 'max_sharpe', 'min_vol', 'target_return'.

    Returns
    -------
    dict with keys: weights (Dict[str, float]), metrics (Dict), status (str).
    """
    tickers, er, cov = _to_arrays(expected_returns, cov_matrix)
    n = len(tickers)

    # P1-Q16-fix: 求解前可行性预检（asset/sector 上限不可行时自动放宽并显式告警）
    eff_asset, eff_sector, cap_issues = _resolve_caps(
        n, tickers, asset_ceiling, sector_map, sector_ceiling
    )
    bounds = _make_bounds(n, eff_asset)
    constraints = _make_constraints(cov_matrix, n, tickers, sector_map, eff_sector, eff_asset)

    x0 = _equal_weight_x0(n)
    target_ret: Optional[float] = None

    # --- dispatch by objective ---
    if objective == "min_vol":
        def obj(w):
            return float(w @ cov @ w)
        success, w_opt, _ = _try_solve(obj, x0, bounds, constraints)

    elif objective == "target_return":
        if target_return is None:
            return {"weights": {}, "metrics": {}, "status": "target_return required for objective=target_return",
                    "warnings": cap_issues}
        target_ret = float(target_return)
        # Add constraint: sum(w_i * er_i) = target_return
        constraints = list(constraints)
        constraints.append({"type": "eq", "fun": lambda w: w @ er - target_ret})

        def obj(w):
            return float(w @ cov @ w)
        success, w_opt, _ = _try_solve(obj, x0, bounds, constraints)

    else:  # max_sharpe (default)
        def obj(w):
            return _neg_sharpe(w, er, cov, rf)
        success, w_opt, _ = _try_solve(obj, x0, bounds, constraints)

    # P1-Q16-fix: 解校验。失败则从等权可行点重试一次；仍失败则显式降级到等权
    # 可行点并标记 infeasible——绝不静默归一化出超限权重（V5.4 的收敛后归一化删除）。
    if not _verify_solution(w_opt, er, target_ret, eff_asset):
        success2, w2, _ = _try_solve(obj, _equal_weight_x0(n), bounds, constraints)
        if _verify_solution(w2, er, target_ret, eff_asset):
            success, w_opt = True, w2
        else:
            success = False
            w_opt = _equal_weight_x0(n)  # 显式可行回退（等权点经预检在界内），status 标明不可行

    # 仅在数值容差内且归一化不违反有效上限时归一化（正常收敛下 sum≈1，不会触发）
    if abs(w_opt.sum() - 1.0) > 1e-6:
        s = w_opt.sum()
        if s > 1e-12:
            w_norm = w_opt / s
            if np.max(w_norm) <= eff_asset + 1e-6:
                w_opt = w_norm
            else:
                success = False

    weights = dict(zip(tickers, np.round(w_opt, 6)))
    metrics = _portfolio_metrics(w_opt, er, cov, rf)

    if not success:
        status = "infeasible"
        cap_issues.append("优化约束不可行，已显式降级到等权可行点（非最优，权重不超过有效上限）")
    elif cap_issues:
        status = "success_with_warning"
    else:
        status = "success"
    return {"weights": weights, "metrics": metrics, "status": status, "warnings": cap_issues}


# ---------------------------------------------------------------------------
# 2. Black-Litterman (simplified)
# ---------------------------------------------------------------------------

def black_litterman(
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
    market_cap_weights: Optional[Union[Dict[str, float], pd.Series]] = None,
    views: Optional[List[Dict[str, Any]]] = None,
    delta: float = DELTA,
    tau: float = TAU,
    rf: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
) -> Dict[str, Any]:
    """Simplified Black-Litterman model.

    D6收敛: 同名异签名保留 —— 函数接口(views=[{tickers,bullish,confidence}]), 与
    portfolio_v2.BlackLitterman 类接口(views=[{type,assets,view,confidence}]) 异实现。

    Parameters
    ----------
    market_cap_weights : dict/series, optional
        Prior market-cap weights. Defaults to equal weight if not provided.
    views : list of dict, optional
        Each view:
          - tickers: list of asset identifiers
          - bullish: bool (True = overweight, False = underweight)
          - confidence: float 0-1 (0 = no view, 1 = absolute)
        If None, posterior = prior.

    Returns
    -------
    dict with keys: prior_weights, posterior_weights, prior_returns,
                    posterior_returns, metrics, status.
    """
    tickers, er, cov = _to_arrays(expected_returns, cov_matrix)
    n = len(tickers)

    # ---- Prior: market-cap weights ----
    if market_cap_weights is not None:
        mc = pd.Series(market_cap_weights)
        mc = mc.reindex(tickers).fillna(0.0)
        w_prior = mc.values / max(mc.sum(), 1e-12)
    else:
        w_prior = _equal_weight_x0(n)

    # ---- Equilibrium returns (CAPM) ----
    #   Π = δ Σ w_mkt
    pi = delta * cov @ w_prior  # shape (n,)

    # ---- Views matrix P and Q ----
    if views:
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))

        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        for i, v in enumerate(views):
            idxs = [ticker_to_idx[t] for t in v["tickers"] if t in ticker_to_idx]
            if not idxs:
                continue
            n_assets = len(idxs)
            P[i, idxs] = 1.0 / n_assets  # equal-weight the view basket

            # View-implied return: scale by bullish/bearish
            avg_ret = np.mean(er[idxs])
            direction = 1.0 if v.get("bullish", True) else -1.0
            # Q[i] = direction * |avg_ret| * some confidence scaling
            # We anchor Q[i] relative to the equilibrium return of the basket
            basket_pi = P[i, :] @ pi
            # Q magnitude:  use the average absolute return of the basket
            Q[i] = basket_pi + direction * 0.5 * abs(basket_pi) * v.get("confidence", 0.5)

            # Uncertainty:  Omega[i,i] = (1 - confidence) * P[i] Σ P[i]^T * tau
            conf = v.get("confidence", 0.5)
            omega_ii = (1.0 - conf) * (P[i, :] @ (cov @ P[i, :])) * tau
            Omega[i, i] = max(omega_ii, 1e-10)

        # ---- Posterior (Black-Litterman) ----
        # μ = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1}  *  [(τΣ)^{-1} Π + P^T Ω^{-1} Q]
        # Σ_p = Σ + [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1}
        try:
            tau_cov_inv = np.linalg.inv(tau * cov)
        except np.linalg.LinAlgError:
            tau_cov_inv = np.linalg.pinv(tau * cov)

        # P2-Q16-fix (M120): 重复/高度相关观点会使 Omega（或其参与构成的后验信息
        # 矩阵）奇异，旧代码 np.linalg.inv(Omega) 无保护 → LinAlgError 崩溃
        # （tau_cov_inv 有 pinv 回退而 Omega 没有）。加 jitter + pinv 双保险。
        Omega_inv = np.linalg.pinv(Omega + np.eye(len(Omega)) * 1e-10)
        M = tau_cov_inv + P.T @ Omega_inv @ P
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M + np.eye(M.shape[0]) * 1e-10)
        mu_bl = M_inv @ (tau_cov_inv @ pi + P.T @ Omega_inv @ Q)
        cov_bl = cov + M_inv
    else:
        mu_bl = pi.copy()
        cov_bl = cov.copy()

    # ---- Posterior weights (MV optimisation on BL returns) ----
    post_er = pd.Series(mu_bl, index=tickers)
    post_cov = pd.DataFrame(cov_bl, index=tickers, columns=tickers)

    result = mean_variance_optimize(
        post_er, post_cov,
        objective="max_sharpe",
        rf=rf,
        sector_map=sector_map,
        sector_ceiling=sector_ceiling,
        asset_ceiling=asset_ceiling,
    )

    return {
        "prior_weights": dict(zip(tickers, np.round(w_prior, 6))),
        "posterior_weights": result["weights"],
        "prior_returns": dict(zip(tickers, np.round(pi, 6))),
        "posterior_returns": dict(zip(tickers, np.round(mu_bl, 6))),
        "metrics": result["metrics"],
        "status": result["status"],
    }


# ---------------------------------------------------------------------------
# 3. Equal Risk Contribution (ERC / Risk Parity)
# ---------------------------------------------------------------------------

def _erc_risk_contrib(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Return vector of risk contributions."""
    port_var = weights @ cov @ weights
    if port_var < 1e-12:
        return np.zeros_like(weights)
    # marginal contribution
    mrc = cov @ weights
    # total risk contribution
    trc = weights * mrc
    return trc / math.sqrt(port_var)


def _erc_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """Objective: scale-invariant sum of squared deviations of risk-contribution
    *shares* from 1/n.

    P1-Q16-fix: 原目标对「绝对值风险贡献」rc（Σrc=σ_p，含尺度项）求与 1/n 的
    平方差，非尺度不变——σ_p≠1 时目标会偏向抬高组合波动，ERC 权重不再是不变量，
    实测会收敛到角点解。改用 rc_i/σ_p（即 rc 份额，与 portfolio.py
    optimize_risk_parity 的 rc_share 形式一致），Σ(rc_share)=1 恒成立，
    目标与组合尺度无关。
    """
    rc = _erc_risk_contrib(weights, cov)
    total = float(rc.sum())
    n = len(rc)
    if total <= 1e-12:
        return 1e6  # zero-variance corner: push optimizer away
    target = 1.0 / n
    rc_share = rc / total  # rc_share_i = rc_i / σ_p, sums to 1
    return float(np.sum((rc_share - target) ** 2))


def equal_risk_contribution(
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
    rf: float = 0.0,
) -> Dict[str, Any]:
    """Equal Risk Contribution (Risk Parity) portfolio.

    Returns
    -------
    dict with keys: weights, metrics, status.
    """
    tickers, er, cov = _to_arrays(expected_returns, cov_matrix)
    n = len(tickers)

    # P1-Q16-fix: 求解前可行性预检（与 H01 同源：ERC 默认 asset_ceiling=0.10，
    # N<10 时 Σw=1 不可行，旧代码归一化后静默超限）
    eff_asset, eff_sector, cap_issues = _resolve_caps(
        n, tickers, asset_ceiling, sector_map, sector_ceiling
    )
    bounds = _make_bounds(n, eff_asset)
    base_constraints = _make_constraints(cov_matrix, n, tickers, sector_map, eff_sector, eff_asset)

    # ERC typically uses inverse-vol weights as a good starting point
    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
    x0 = inv_vol / inv_vol.sum()
    if np.max(x0) > eff_asset + 1e-9 or abs(x0.sum() - 1.0) > 1e-6:
        x0 = _equal_weight_x0(n)  # keep the start point inside the caps

    def obj(w):
        return _erc_objective(w, cov)

    success, w_opt, _ = _try_solve(obj, x0, bounds, base_constraints)

    # P1-Q16-fix: 解校验，失败则显式降级到等权可行点（不再静默归一化）
    if not _verify_solution(w_opt, er, None, eff_asset):
        success2, w2, _ = _try_solve(obj, _equal_weight_x0(n), bounds, base_constraints)
        if _verify_solution(w2, er, None, eff_asset):
            success, w_opt = True, w2
        else:
            success = False
            w_opt = _equal_weight_x0(n)

    if abs(w_opt.sum() - 1.0) > 1e-6:
        s = w_opt.sum()
        if s > 1e-12:
            w_norm = w_opt / s
            if np.max(w_norm) <= eff_asset + 1e-6:
                w_opt = w_norm
            else:
                success = False

    weights = dict(zip(tickers, np.round(w_opt, 6)))
    metrics = _portfolio_metrics(w_opt, er, cov, rf)
    metrics["risk_contributions"] = dict(
        zip(tickers, np.round(_erc_risk_contrib(w_opt, cov), 6))
    )

    if not success:
        status = "infeasible"
        cap_issues.append("ERC 优化约束不可行，已显式降级到等权可行点（非最优）")
    elif cap_issues:
        status = "success_with_warning"
    else:
        status = "success"
    return {"weights": weights, "metrics": metrics, "status": status, "warnings": cap_issues}


# ---------------------------------------------------------------------------
# 4. Efficient Frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
    points: int = FRONTIER_POINTS,
    rf: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
) -> Dict[str, Any]:
    """Compute the efficient frontier (min-vol frontier from min to max return).

    Returns
    -------
    dict:
      - frontier: list of dicts with return, volatility, sharpe, weights
      - max_sharpe_point: dict
      - min_vol_point: dict
      - status: str
    """
    tickers, er, cov = _to_arrays(expected_returns, cov_matrix)
    n = len(tickers)

    # P1-Q16-fix: 前沿可行性预检（与 H01 联动：N<1/ceiling 时前沿不再塌缩）
    eff_asset, eff_sector, cap_issues = _resolve_caps(
        n, tickers, asset_ceiling, sector_map, sector_ceiling
    )
    bounds = _make_bounds(n, eff_asset)
    base_constraints = _make_constraints(cov_matrix, n, tickers, sector_map, eff_sector, eff_asset)

    # -- Find min-vol portfolio (left-most point) --
    def min_var_obj(w):
        return float(w @ cov @ w)

    x0 = _equal_weight_x0(n)
    success_min, w_min, _ = _try_solve(min_var_obj, x0, bounds, list(base_constraints))
    if not _verify_solution(w_min, er, None, eff_asset):
        success_min, w_min = False, _equal_weight_x0(n)
    ret_min = float(w_min @ er)

    # -- Find max-return portfolio (top point) --
    # Single-asset max return within caps
    max_ret = er.max()
    # Upper bound as max return achievable (all on best single asset subject to ceiling)
    # But we want the actual max-return feasible portfolio
    def max_ret_obj(w):
        return -float(w @ er)
    success_max, w_max, _ = _try_solve(max_ret_obj, x0, bounds, list(base_constraints))
    if not _verify_solution(w_max, er, None, eff_asset):
        success_max, w_max = False, _equal_weight_x0(n)
    ret_max = float(w_max @ er)

    # Spread
    if ret_max <= ret_min + 1e-8:
        # degenerate case: only one point
        rets = np.array([ret_min])
    else:
        rets = np.linspace(ret_min, ret_max, points)

    # P1-Q16-fix: 每个目标收益点求解后校验 w@er≈target 与可行性；
    # 失败/无效点直接剔除，不再把末次失败的解当作有效前沿点输出。
    frontier: List[Dict] = []
    n_failed = 0
    for target in rets:
        cons = list(base_constraints)
        cons.append({"type": "eq", "fun": lambda w, t=target: w @ er - t})

        def obj(w):
            return float(w @ cov @ w)

        ok, w_opt, _ = _try_solve(obj, x0, bounds, cons)
        if not ok or not _verify_solution(w_opt, er, float(target), eff_asset):
            n_failed += 1
            continue
        m = _portfolio_metrics(w_opt, er, cov, rf)
        frontier.append({
            "return": m["expected_return"],
            "volatility": m["volatility"],
            "sharpe": m["sharpe_ratio"],
            "weights": dict(zip(tickers, np.round(w_opt, 6))),
        })

    # Max Sharpe point = tangent portfolio
    result_ms = mean_variance_optimize(
        expected_returns, cov_matrix,
        objective="max_sharpe",
        rf=rf,
        sector_map=sector_map,
        sector_ceiling=sector_ceiling,
        asset_ceiling=asset_ceiling,
    )
    if result_ms.get("status") in ("infeasible", "failed") or not result_ms.get("weights"):
        max_sharpe_point = {}
        cap_issues.append("max_sharpe 求解失败，max_sharpe_point 为空")
    else:
        max_sharpe_point = {
            "return": result_ms["metrics"]["expected_return"],
            "volatility": result_ms["metrics"]["volatility"],
            "sharpe": result_ms["metrics"]["sharpe_ratio"],
            "weights": result_ms["weights"],
        }

    # 全部目标点被剔除时，至少保留 min-vol 可行点，避免空前沿
    if not frontier and _verify_solution(w_min, er, None, eff_asset):
        m = _portfolio_metrics(w_min, er, cov, rf)
        frontier.append({
            "return": m["expected_return"],
            "volatility": m["volatility"],
            "sharpe": m["sharpe_ratio"],
            "weights": dict(zip(tickers, np.round(w_min, 6))),
        })

    min_vol_point = frontier[0] if frontier else {}
    if n_failed:
        cap_issues.append(f"{n_failed} 个前沿目标收益点不可行，已剔除")
    status = "success"
    if not frontier:
        status = "infeasible"
    elif n_failed or cap_issues:
        status = "success_with_warning"

    return {
        "frontier": frontier,
        "max_sharpe_point": max_sharpe_point,
        "min_vol_point": min_vol_point,
        "status": status,
        "warnings": cap_issues,
    }


# ---------------------------------------------------------------------------
# 5. CVaR (Conditional Value at Risk) Optimization
# ---------------------------------------------------------------------------

def cvar_optimize(
    expected_returns: Union[Dict[str, float], pd.Series],
    historical_returns: pd.DataFrame,
    alpha: float = 0.05,
    rf: float = 0.0,
    sector_map: Optional[Dict[str, str]] = None,
    sector_ceiling: float = 0.30,
    asset_ceiling: float = 0.10,
) -> Dict[str, Any]:
    """Minimize CVaR (Conditional Value at Risk) using historical scenarios.

    D6收敛: 异名/近名异口径保留 —— α=尾部概率(>0.5自动换算为1-α, RU形式), 与
    portfolio_v2.MeanCVaROptimization.calculate_cvar(α=置信度, √ppy年化)/
    risk_management_pro._historical_var_cvar(损失数组)/portfolio_risk.compute_var(百分数报告) 异口径。

    Rockafellar-Uryasev formulation: minimize VaR + (1/α)*E[max(-w'R - VaR, 0)]

    P2-Q16-fix (M121): 明确 α 语义。本函数 alpha 为**尾部概率**（default 0.05 =
    95% CVaR），与 portfolio.py compute_cvar / optimize_cvar 的 alpha=置信度 约定
    不同。为统一语义：若调用方按置信度约定传入 alpha>0.5（如 0.95），自动换算为
    尾部概率 1-alpha，两种约定得到同一结果，不再因混用产生相反结果。

    Parameters
    ----------
    historical_returns : pd.DataFrame
        T x N matrix of historical returns (scenarios)
    alpha : float
        Tail probability (default 0.05 = 95% CVaR); >0.5 视为置信度并换算为 1-alpha.

    Returns
    -------
    dict with keys: weights, metrics (incl CVaR), status
    """
    tickers = list(historical_returns.columns)
    T, N = historical_returns.shape
    er = np.array([expected_returns.get(t, 0.0) for t in tickers])
    R = historical_returns.values  # T x N
    # 统一 α 语义为尾部概率，夹取到 [0.01, 0.5] 保证目标有限且方向正确。
    if alpha > 0.5:
        alpha = 1.0 - alpha
    tail_alpha = float(min(max(alpha, 0.01), 0.5))
    confidence_pct = (1.0 - tail_alpha) * 100.0

    bounds = _make_bounds(N, asset_ceiling)
    constraints_list = _make_constraints(historical_returns, N, historical_returns.columns, sector_map, sector_ceiling, asset_ceiling)

    # Reformulate: minimize VaR + (1/α) * 1/T * sum(z_i)
    # s.t. z_i >= -w'R_i - VaR, z_i >= 0
    # We do this via scipy minimize by augmenting the variable space
    # Variables: [w_1..w_N, VaR, z_1..z_T]

    # Bounds: weights [0, ceiling], VaR unconstrained, z_i >= 0
    w_bounds = [(0, asset_ceiling) for _ in range(N)]
    var_bounds = [(None, None)]
    z_bounds = [(0, None) for _ in range(T)]
    all_bounds = w_bounds + var_bounds + z_bounds

    # Constraints
    cons_list = [
        {"type": "eq", "fun": lambda x: np.sum(x[:N]) - 1.0},  # sum(w)=1
    ]
    # Q16 修复：正确方向应为 z_i >= -w'R_i - VaR，即
    #   z_i + w'R_i + VaR >= 0  （scipy ineq: fun(x) >= 0）
    # 原实现返回 -w'R_i - VaR - z_i，等价于强制 z_i <= -w'R_i - VaR，
    # 约束方向写反 → 目标无下界 → SLSQP 输出数值垃圾（fun=-3e48）。
    # 与 portfolio.py optimize_cvar 的正确形式一致。
    for i in range(T):
        def _cvar_cons(x, idx=i):
            w = x[:N]
            var = x[N]
            z = x[N + 1 + idx]
            return float(z + w @ R[idx] + var)
        cons_list.append({"type": "ineq", "fun": _cvar_cons})

    def cvar_obj(x):
        var = x[N]
        z_sum = np.sum(x[N+1:])
        # P2-Q16-fix (M121): 用统一后的 tail_alpha（尾部概率）做 RU 缩放系数
        return float(var + (1.0 / tail_alpha) * (1.0 / T) * z_sum)

    try:
        from scipy.optimize import minimize as _min
        # 初始点：等权组合的 VaR / 尾部损失，保证 z_i 起点可行，加速收敛。
        # P2-Q16-fix (M121): 起始 VaR 取损失分布的 1-tail_alpha 分位（即置信度分位）
        losses0 = -(R @ (np.ones(N) / N))
        eta0 = float(np.quantile(losses0, 1.0 - tail_alpha))
        u0 = np.maximum(losses0 - eta0, 0.0)
        x0 = np.concatenate([np.full(N, 1.0 / N), [eta0], u0])
        res = _min(cvar_obj, x0, method="SLSQP", bounds=all_bounds, constraints=cons_list,
                   options={"maxiter": 500, "ftol": 1e-8})
        w_opt = res.x[:N]
        s = w_opt.sum()
        if s > 1e-12:
            w_opt = w_opt / s
        # P1-Q16 遗留修复: 归一化后校验 asset_ceiling/sector_ceiling 上限，
        # 违反时显式告警并记录 status，不再静默输出超限权重。
        _viol = [
            t for t, wgt in zip(tickers, w_opt)
            if wgt > asset_ceiling + 1e-8
        ]
        _sec_viol = []
        if sector_map:
            sec_w = {}
            for t, wgt in zip(tickers, w_opt):
                sec = sector_map.get(t, "其他")
                sec_w[sec] = sec_w.get(sec, 0.0) + wgt
            _sec_viol = [s for s, wgt in sec_w.items() if wgt > sector_ceiling + 1e-8]
        if _viol or _sec_viol:
            warnings = [
                f"cvar_optimize: 归一化后仍超限 asset_ceiling={asset_ceiling}: {_viol}",
                f"cvar_optimize: 归一化后仍超限 sector_ceiling={sector_ceiling}: {_sec_viol}",
            ]
            _warnings = list(warnings)
        else:
            _warnings = []
        weights = dict(zip(tickers, np.round(w_opt, 6)))

        # Compute portfolio CVaR（用统一后的 tail_alpha 取左尾）
        port_rets = R @ w_opt
        var_hist = np.percentile(port_rets, tail_alpha * 100)
        cvar_hist = -float(np.mean(port_rets[port_rets <= var_hist]))

        metrics = _portfolio_metrics(w_opt, er, np.cov(R.T), rf)
        metrics["cvar_95pct"] = round(float(cvar_hist), 6)
        metrics["VaR_95pct"] = round(float(-var_hist), 6)
        metrics["cvar_conf_pct"] = round(confidence_pct, 1)  # 实际置信度（标注语义）
        # P2-Q16-fix (M121): 对尾部测度做 sqrt(242) 年化缺乏理论依据（CVaR 不是
        # 方差量纲），显式标注为近似，生产环境请用滚动/块估计年化。
        metrics["cvar_ann"] = round(float(cvar_hist * math.sqrt(242)), 2)
        metrics["cvar_ann_note"] = ("sqrt(242) annualization is an approximation for "
                                    "tail measures; use rolling/block estimation in production")

        return {"weights": weights, "metrics": metrics, "status": "success" if res.success else "converged", "warnings": _warnings}
    except Exception as exc:
        # Fallback: equal weight（可见降级，status 带原因）
        w = np.full(N, 1.0 / N)
        port_rets = R @ w
        var_hist = np.percentile(port_rets, tail_alpha * 100)
        cvar_hist = -float(np.mean(port_rets[port_rets <= var_hist]))
        weights = dict(zip(tickers, np.round(w, 6)))
        metrics = _portfolio_metrics(w, er, np.cov(R.T), rf)
        metrics["cvar_95pct"] = round(float(cvar_hist), 6)
        metrics["VaR_95pct"] = round(float(-var_hist), 6)
        metrics["cvar_conf_pct"] = round(confidence_pct, 1)
        metrics["cvar_ann_note"] = ("sqrt(242) annualization is an approximation for "
                                    "tail measures; use rolling/block estimation in production")
        return {"weights": weights, "metrics": metrics, "status": f"fallback_equal_weight: {exc}"}


# ---------------------------------------------------------------------------
# 6. Portfolio Summary
# ---------------------------------------------------------------------------

def portfolio_summary(
    weights: Union[Dict[str, float], pd.Series],
    expected_returns: Union[Dict[str, float], pd.Series],
    cov_matrix: pd.DataFrame,
    historical_returns: Optional[pd.DataFrame] = None,
    rf: float = 0.0,
) -> Dict[str, Any]:
    """Compute a comprehensive summary of a given portfolio.

    D6收敛: 异名/近名异口径保留 —— weight_concentration 平方口径HHI/有效持仓数, 与
    risk_management_pro._hhi(abs口径)/portfolio_v2._effective_n 异口径；
    max_drawdown 负向 vs risk_management_pro._max_drawdown_from_returns 损失正数。

    Parameters
    ----------
    weights : dict / Series
        Portfolio weights (do not need to sum to 1; will be normalised).
    historical_returns : DataFrame, optional
        T x N DataFrame of historical returns. Required for max-drawdown
        estimation.

    Returns
    -------
    dict with keys: metrics (Dict), max_drawdown (float or None),
                    weight_concentration (Dict), status (str).
    """
    tickers, er, cov = _to_arrays(expected_returns, cov_matrix)
    w = pd.Series(weights).reindex(tickers).fillna(0.0).values.astype(np.float64)
    s = w.sum()
    if s > 1e-12:
        w = w / s
    else:
        w = _equal_weight_x0(len(tickers))

    metrics = _portfolio_metrics(w, er, cov, rf)

    # ---- Weight concentration ----
    w_sorted = np.sort(w)[::-1]
    cum_w = np.cumsum(w_sorted)
    top5_idx = min(5, len(w_sorted))
    concentration = {
        "num_assets": int(np.sum(w > 1e-6)),
        "top_1_weight": float(w_sorted[0]),
        "top_5_weight": float(cum_w[top5_idx - 1]) if top5_idx > 0 else 0.0,
        "herfindahl": float(np.sum(w ** 2)),
        "effective_n": float(1.0 / max(np.sum(w ** 2), 1e-12)),
    }

    # ---- Max drawdown from historical returns ----
    max_dd: Optional[float] = None
    if historical_returns is not None and not historical_returns.empty:
        try:
            # Align tickers
            cols = [t for t in tickers if t in historical_returns.columns]
            if cols:
                hist = historical_returns[cols].values.astype(np.float64)
                port_hist = hist @ w  # (T,)
                cum = np.cumprod(1.0 + port_hist)
                running_max = np.maximum.accumulate(cum)
                dd = (cum - running_max) / running_max
                max_dd = float(np.min(dd))
        except Exception as e:
            logging.getLogger(__name__).error(f"[portfolio_optimizer] 操作失败: {e}", exc_info=True)

    return {
        "metrics": metrics,
        "max_drawdown": max_dd,
        "weight_concentration": concentration,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    "mean_variance_optimize",
    "black_litterman",
    "equal_risk_contribution",
    "efficient_frontier",
    # P2-Q16-fix (L131): 补入 cvar_optimize——旧实现缺列导致 from * 导入缺失
    "cvar_optimize",
    "portfolio_summary",
]
