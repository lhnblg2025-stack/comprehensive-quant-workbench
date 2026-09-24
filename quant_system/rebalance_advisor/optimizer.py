"""
optimizer — 目标权重计算 (V5)

给定当前持仓 + 约束条件，输出优化的目标权重。

方法：
  1. 均值-方差优化（经典MVO）
  2. 风险平价（Risk Parity）
  3. 约束：单票上限/行业上限/换手率上限/因子中性

输入：holdings, constraints, market_views
输出：target_weights, expected_risk, expected_return
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """组合优化器——计算目标权重。"""

    DEFAULT_CONSTRAINTS = {
        "max_single_weight": 0.15,
        "min_single_weight": 0.01,
        "max_sector_weight": 0.40,
        "turnover_limit": 0.20,
        "target_beta": 1.0,
        "risk_free_rate": 0.025,
    }

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """风险平价权重（等风险贡献）。

        Args:
            cov_matrix: 协方差矩阵 (n x n)

        Returns:
            风险平价权重向量
        """
        n = cov_matrix.shape[0]
        w = np.ones(n) / n

        for _ in range(100):
            # 边际风险贡献
            portfolio_var = w @ cov_matrix @ w
            marginal_contrib = cov_matrix @ w / np.sqrt(portfolio_var)
            risk_contrib = w * marginal_contrib
            target_rc = np.mean(risk_contrib)

            # 梯度调整
            delta = target_rc - risk_contrib
            w_new = w + delta * 0.1
            w_new = np.maximum(w_new, 0)
            w_new = w_new / max(w_new.sum(), 1e-10)

            if np.max(np.abs(w_new - w)) < 1e-6:
                break
            w = w_new

        return w

    def apply_constraints(self, weights: np.ndarray,
                           codes: list[str],
                           constraints: dict[str, Any] | None = None,
                           current_weights: np.ndarray | None = None,
                           sectors: dict[str, str] | None = None,
                           betas: list[float] | None = None) -> np.ndarray:
        """应用约束到权重。

        Args:
            weights: 原始权重向量
            codes: 股票代码列表
            constraints: 约束字典
            current_weights: 当前权重（用于实施换手率上限），可选
            sectors: {code: 行业} 映射（用于实施行业上限），可选
            betas: 各股票 beta（用于实施目标 beta），可选

        Returns:
            约束后权重
        """
        if constraints is None:
            constraints = self.DEFAULT_CONSTRAINTS.copy()

        w = weights.copy()
        n = len(w)
        if n == 0:
            return w

        # P2-Q28-fix(M354): 原实现"先 clip 再一次性归一化"，n≤6 时归一化会把
        # 权重放大回超过单票上限（3 股等权 0.3333 > 0.15）。单票上限不可行
        # （n*max<1）时可见降级为等权 1/n 并记录日志，避免静默越限。
        max_single = constraints.get("max_single_weight", 0.15)
        min_single = constraints.get("min_single_weight", 0.01)
        if n * max_single < 1.0 - 1e-12:
            relaxed = 1.0 / n
            logger.warning(
                "单票上限 %.3f 在 n=%d 只股票下不可行 (n*max=%.3f<1)，"
                "已可见降级为等权 1/n=%.4f", max_single, n, n * max_single, relaxed
            )
            max_single = relaxed
        if min_single > 1.0 / n:
            min_single = 1.0 / n

        max_sector = constraints.get("max_sector_weight")
        turnover_limit = constraints.get("turnover_limit")
        target_beta = constraints.get("target_beta")

        sec_arr = None
        if max_sector and sectors:
            sec_arr = np.array([sectors.get(c, "") for c in codes])
            if not np.any(sec_arr != ""):
                sec_arr = None

        # P2-Q28-fix(M355): 迭代交替投影，同时满足 单票箱体/求和=1/行业上限/
        # 换手率上限/目标beta。原实现按顺序"缩放→Dykstra投影"，后续 beta 倾斜+
        # 投影会把行业与换手约束冲掉（实测可行场景行业权重 0.50 > 上限 0.40）。
        # 现逐约束交替投影直至收敛；收敛后仍有越限（约束组合不可行）时可见告警，
        # 而不是静默输出越限权重。
        for _ in range(200):
            w_prev = w.copy()
            w = self._project_box_and_simplex(w, min_single, max_single)
            if sec_arr is not None:
                w = self._enforce_sector_caps(w, sec_arr, max_sector, min_single, max_single)
                w = self._project_box_and_simplex(w, min_single, max_single)
            if turnover_limit is not None and current_weights is not None and len(current_weights) == n:
                w = self._enforce_turnover(w, current_weights, turnover_limit, min_single, max_single)
                w = self._project_box_and_simplex(w, min_single, max_single)
            if target_beta is not None and betas is not None and len(betas) == n:
                w = self._enforce_beta(w, betas, target_beta, min_single, max_single)
                w = self._project_box_and_simplex(w, min_single, max_single)
            if np.max(np.abs(w - w_prev)) < 1e-9:
                break

        # 断言约束后单票权重不超过上限
        assert w.max() <= max_single + 1e-9, \
            f"约束后单票权重 {w.max():.4f} > 上限 {max_single:.4f}"

        # P2-Q28-fix(M355): 收敛后校验——约束不可行/冲突时可见告警（降级必须可见）。
        # 容差 1e-6 用于吸收交替投影的浮点误差，避免 cap 边界处的误报。
        if sec_arr is not None:
            for s in np.unique(sec_arr):
                if not s:
                    continue
                sec_w = float(w[sec_arr == s].sum())
                if sec_w > max_sector + 1e-6:
                    logger.warning(
                        "行业 %s 权重 %.4f 超上限 %.4f（约束不可行或与单票上限冲突，"
                        "已优先保证单票上限）", s, sec_w, max_sector
                    )
        if turnover_limit is not None and current_weights is not None and len(current_weights) == n:
            actual_turn = float(np.sum(np.abs(w - current_weights)) / 2)
            if actual_turn > turnover_limit + 1e-6:
                logger.warning(
                    "换手率上限 %.3f 未完全满足，实际 %.3f（与单票上限冲突，单票上限优先）",
                    turnover_limit, actual_turn,
                )
        if target_beta is not None and betas is not None and len(betas) == n:
            actual_beta = float(np.dot(w, np.asarray(betas, dtype=float)))
            if abs(actual_beta - target_beta) > 0.05:
                logger.warning(
                    "目标beta %.2f 未完全满足，实际 %.2f（约束组合不可行）",
                    target_beta, actual_beta,
                )

        return w

    @staticmethod
    def _enforce_sector_caps(w: np.ndarray, sec_arr: np.ndarray, cap: float,
                             low: float, high: float) -> np.ndarray:
        """把权重投影到"每个行业权重 ≤ cap"的约束域。

        P2-Q28-fix(M355): 缩放超限行业至 cap，把释放出的权重按比例分配给未超限
        行业成员（避免后续 simplex 归一化把权重重新灌回超限行业）；行业上限整体
        不可行（Σcap<1）时可见降级（放宽 cap）并告警。外层循环随后做 box+simplex。
        """
        w = w.copy()
        secs = np.array([s for s in np.unique(sec_arr) if s])
        if len(secs) == 0:
            return w
        if cap * len(secs) < 1.0 - 1e-9:
            logger.warning(
                "行业上限 %.3f 对 %d 个行业不可行 (Σcap=%.3f<1)，已放宽至 1/n_sec=%.3f",
                cap, len(secs), cap * len(secs), 1.0 / len(secs),
            )
            cap = 1.0 / len(secs)
        for _ in range(200):
            w = np.clip(w, low, high)
            over = [s for s in secs if float(w[sec_arr == s].sum()) > cap + 1e-12]
            if not over:
                break
            freed = 0.0
            for s in over:
                mask = sec_arr == s
                sw = float(w[mask].sum())
                w[mask] = w[mask] * (cap / max(sw, 1e-12))
                freed += sw - cap
            under_mask = ~np.isin(sec_arr, over)
            if freed > 1e-12:
                base = float(w[under_mask].sum())
                if base > 1e-12:
                    add = np.zeros_like(w)
                    add[under_mask] = w[under_mask] / base * freed
                    w = np.clip(w + add, low, high)
                else:
                    logger.warning(
                        "行业约束不可行：无未超限行业可吸收权重 %.4f，停止分配", freed
                    )
                    break
        return w

    @staticmethod
    def _enforce_turnover(w: np.ndarray, current_weights: np.ndarray,
                          limit: float, low: float, high: float) -> np.ndarray:
        """限制单边换手率 ≤ limit：对调仓向量按比例压缩。

        P2-Q28-fix(M355): current 与 target 均为合法权重时凸组合保持和为 1，
        之后 clip 回单票箱体；clip 损失的归一化由外层 box+simplex 修复。
        """
        delta = w - current_weights
        one_way = float(np.sum(np.abs(delta)) / 2)
        if one_way <= limit + 1e-12:
            return np.clip(w, low, high)
        scale = limit / max(one_way, 1e-12)
        return np.clip(current_weights + delta * scale, low, high)

    @staticmethod
    def _enforce_beta(w: np.ndarray, betas, target_beta: float,
                      low: float, high: float) -> np.ndarray:
        """目标 beta 倾斜：使 dot(w, beta) 趋近 target_beta。"""
        beta_arr = np.asarray(betas, dtype=float)
        port_beta = float(np.dot(w, beta_arr))
        if abs(port_beta - target_beta) < 1e-9:
            return w
        mean_beta = float(beta_arr.mean())
        denom = float(np.sum((beta_arr - mean_beta) ** 2))
        if denom <= 1e-12:
            return w  # 所有 beta 相同，无法倾斜
        tilt = (target_beta - port_beta) / denom
        return np.clip(w + tilt * (beta_arr - mean_beta), low, high)

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """Euclidean 投影到概率单纯形 {x>=0, sum=1}（水填充法）。"""
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(v) + 1)
        cond = u * ind > cssv
        rho = int(np.nonzero(cond)[0][-1]) if np.any(cond) else 0
        theta = cssv[rho] / (rho + 1)
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _project_box_and_simplex(v: np.ndarray, low: float, high: float) -> np.ndarray:
        """Dykstra 交替投影：把 v 投影到 [low,high]^n ∩ {sum=1}。

        盒子与单纯形均为凸集；调用方已保证 low<=1/n<=high 使交集非空。
        该算法收敛到初始点 v 在交集上的欧氏投影。
        """
        x = v.copy()
        p = np.zeros_like(v)
        q = np.zeros_like(v)
        for _ in range(200):
            y = np.clip(x + p, low, high)
            p = x + p - y
            x = PortfolioOptimizer._project_simplex(y + q)
            q = y + q - x
            if np.max(np.abs(x - y)) < 1e-12:
                break
        return x

    def compute(self, holdings: list[dict[str, Any]],
                constraints: dict[str, Any] | None = None,
                cov_matrix: np.ndarray | None = None,
                returns: np.ndarray | None = None,
                sectors: dict[str, str] | None = None,
                betas: list[float] | None = None) -> dict[str, Any]:
        """计算目标权重。

        Args:
            holdings: 当前持仓 [{"code":..., "weight":..., "name":...}, ...]
            constraints: 约束
            cov_matrix: 真实协方差矩阵 (n x n)，可选；优先级高于 returns
            returns: 个股历史收益矩阵 (n_samples x n)，可选，用于推算协方差
            sectors: {code: 行业} 映射，可选（用于行业上限约束）
            betas: 各股票 beta，可选（用于目标 beta 约束）

        Returns:
            {
                "target_weights": [{"code":..., "name":..., "current_weight":...,
                                    "target_weight":..., "change":...}],
                "method": "risk_parity",
                "turnover": float,          # 单边换手率
                "constraints_applied": dict,
            }
        """
        if constraints is None:
            constraints = self.DEFAULT_CONSTRAINTS.copy()

        codes = [h["code"] for h in holdings]
        current_weights = np.array([h.get("weight", 0) for h in holdings])
        n = len(codes)

        if n == 0:
            return {"target_weights": [], "turnover": 0}

        # P2-Q28-fix(M355): 弃用随机协方差（原 np.random.seed(42) 生成的目标权重与
        # 真实收益数据零关联）。优先使用调用方提供的真实协方差/收益数据；
        # 两者都缺失时，可见降级为对角协方差（波动率取持仓自带字段或 0.20 默认），
        # 并记录日志，避免静默使用随机数。
        if cov_matrix is not None:
            cov = np.asarray(cov_matrix, dtype=float)
            if cov.shape != (n, n):
                logger.warning("cov_matrix 形状 %s 与持仓数 %d 不符，已忽略，回退对角协方差",
                               cov.shape, n)
                cov_matrix = None
        if cov_matrix is None and returns is not None:
            ret = np.asarray(returns, dtype=float)
            if ret.shape[1] == n and ret.shape[0] > 1:
                cov = np.cov(ret, rowvar=False)
                cov = cov + np.eye(n) * 1e-8  # 数值稳定性
            else:
                logger.warning("returns 形状 %s 与持仓数 %d 不符，已忽略", ret.shape, n)
        if cov_matrix is None and returns is None:
            vols = []
            for h in holdings:
                v = h.get("volatility") or h.get("risk") or h.get("annual_vol") or 0.20
                vols.append(float(v))
            cov = np.diag(np.asarray(vols, dtype=float) ** 2)
            logger.warning(
                "compute() 未提供真实协方差/收益数据，使用对角协方差 "
                "(vol=%s) —— 结果仅等权近似，需接入真实风险模型",
                [round(v, 3) for v in vols]
            )

        # 风险平价
        target = self.risk_parity(cov)
        target = self.apply_constraints(
            target, codes, constraints,
            current_weights=current_weights, sectors=sectors, betas=betas,
        )

        # 计算换手率
        turnover = float(np.sum(np.abs(target - current_weights)) / 2)

        # 构建结果
        target_list = []
        for i, h in enumerate(holdings):
            target_list.append({
                "code": codes[i],
                "name": h.get("name", ""),
                "current_weight": round(float(current_weights[i]), 4),
                "target_weight": round(float(target[i]), 4),
                "change": round(float(target[i] - current_weights[i]), 4),
            })

        return {
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone(__import__("datetime").timedelta(hours=8))
            ).isoformat(),
            "target_weights": target_list,
            "method": "risk_parity",
            "turnover": round(turnover, 4),
            "constraints_applied": constraints,
        }


def main() -> None:
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20},
        {"code": "000858", "name": "五粮液", "weight": 0.12},
        {"code": "300750", "name": "宁德时代", "weight": 0.10},
        {"code": "601318", "name": "中国平安", "weight": 0.08},
        {"code": "000333", "name": "美的集团", "weight": 0.06},
        {"code": "600036", "name": "招商银行", "weight": 0.05},
        {"code": "002415", "name": "海康威视", "weight": 0.04},
        {"code": "000002", "name": "万科A", "weight": 0.04},
        {"code": "002304", "name": "洋河股份", "weight": 0.03},
        {"code": "601166", "name": "兴业银行", "weight": 0.03},
    ]

    opt = PortfolioOptimizer()
    result = opt.compute(holdings)

    print("═" * 60)
    print("  组合优化结果 — 风险平价")
    print("═" * 60)
    print(f"\n  预估换手率: {result['turnover']:.1%}")
    print(f"  方法: {result['method']}")
    print()
    print(f"  {'代码':<8} {'名称':<10} {'当前':>8} {'目标':>8} {'变动':>8}")
    print(f"  {'-'*40}")
    for tw in result["target_weights"]:
        change = tw["change"]
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {tw['code']:<8} {tw['name']:<10} "
              f"{tw['current_weight']:>7.1%} {tw['target_weight']:>7.1%} "
              f"{arrow} {abs(change):>.1%}")


if __name__ == "__main__":
    main()
