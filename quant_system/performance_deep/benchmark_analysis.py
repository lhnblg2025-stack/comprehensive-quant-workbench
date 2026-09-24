"""
benchmark_analysis — 基准比较分析 (V5)

核心问题：组合相对基准的表现，是"真本事"还是"高Beta裸奔"？

核心指标：
  - Tracking Error（追踪误差）: std(R_pf - R_bm)，年化
  - Information Ratio（信息比率）: mean(active_return) / tracking_error，年化
  - Rolling Sharpe（滚动夏普比率）: 滚动窗口下组合的风险调整收益
  - Alpha/Beta: 对 R_pf = alpha + beta * R_bm + epsilon 做 OLS 回归

对标：GIPS 归因报告中的标准基准比较指标
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

TRADING_DAYS_PER_YEAR = 252


class BenchmarkAnalysis:
    """基准比较分析引擎。

    计算组合相对基准的追踪误差、信息比率、滚动夏普比率与 OLS 回归
    alpha/beta。

    核心假设：
      输入为等频（如日频）收益率序列，按年化 252 个交易日处理。
    """

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> None:
        """初始化。

        Args:
            risk_free_rate: 年化无风险利率，用于计算夏普比率
            periods_per_year: 年化周期数（日频默认252）
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def _tracking_error(self, active_returns: np.ndarray) -> float:
        """计算年化追踪误差。"""
        if len(active_returns) < 2:
            return 0.0
        return float(np.std(active_returns, ddof=1) * np.sqrt(self.periods_per_year))

    def _information_ratio(self, active_returns: np.ndarray) -> float:
        """计算年化信息比率。"""
        if len(active_returns) < 2:
            return 0.0
        te = self._tracking_error(active_returns)
        if te < 1e-12:
            return 0.0
        annualized_active = float(np.mean(active_returns) * self.periods_per_year)
        return annualized_active / te

    def _rolling_sharpe(self, returns: np.ndarray, window: int = 20) -> list[float]:
        """计算滚动窗口年化夏普比率序列。

        Args:
            returns: 组合收益率序列
            window: 滚动窗口长度

        Returns:
            滚动夏普比率列表，长度为 len(returns) - window + 1
        """
        n = len(returns)
        if n < window:
            return []
        daily_rf = self.risk_free_rate / self.periods_per_year
        sharpes = []
        for end in range(window, n + 1):
            win = returns[end - window:end]
            excess = win - daily_rf
            std = np.std(excess, ddof=1)
            if std < 1e-12:
                sharpes.append(0.0)
            else:
                sharpe = float(np.mean(excess) / std * np.sqrt(self.periods_per_year))
                sharpes.append(round(sharpe, 4))
        return sharpes

    @staticmethod
    def _ols_alpha_beta(pf_returns: np.ndarray, bm_returns: np.ndarray) -> dict[str, float]:
        """用 OLS 回归求 alpha/beta: R_pf = alpha + beta * R_bm + epsilon。

        Args:
            pf_returns: 组合收益率序列
            bm_returns: 基准收益率序列

        Returns:
            dict: alpha（单期，未年化）, beta, r_squared
        """
        n = min(len(pf_returns), len(bm_returns))
        if n < 3:
            return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

        y = pf_returns[:n]
        x = bm_returns[:n]

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.sum((x - x_mean) ** 2)

        if var_x < 1e-12:
            return {"alpha": float(y_mean), "beta": 0.0, "r_squared": 0.0}

        beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
        alpha = float(y_mean - beta * x_mean)

        y_pred = alpha + beta * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        return {
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "r_squared": round(max(0.0, min(1.0, r_squared)), 4),
        }

    def analyze(
        self,
        pf_returns: list[float] | np.ndarray,
        bm_returns: list[float] | np.ndarray,
        rolling_window: int = 20,
    ) -> dict[str, Any]:
        """执行基准比较分析。

        Args:
            pf_returns: 组合收益率序列（日频）
            bm_returns: 基准收益率序列（日频，与组合对齐）
            rolling_window: 滚动夏普比率的窗口长度

        Returns:
            dict:
                - tracking_error: 年化追踪误差
                - information_ratio: 年化信息比率
                - rolling_sharpe: 滚动夏普比率列表（组合）
                - alpha_beta_from_regression: {alpha, beta, r_squared}
                - annualized_alpha: 年化 alpha (alpha * periods_per_year)
                - active_return_mean: 平均主动收益（单期）
        """
        try:
            pf = np.asarray(pf_returns, dtype=float)
            bm = np.asarray(bm_returns, dtype=float)
            n = min(len(pf), len(bm))
            if n < 3:
                return {
                    "error": "样本数量不足(至少需要3期)",
                    "tracking_error": 0.0,
                    "information_ratio": 0.0,
                    "rolling_sharpe": [],
                    "alpha_beta_from_regression": {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0},
                }

            pf = pf[:n]
            bm = bm[:n]
            active = pf - bm

            tracking_error = round(self._tracking_error(active), 4)
            information_ratio = round(self._information_ratio(active), 4)
            rolling_sharpe = self._rolling_sharpe(pf, window=rolling_window)
            reg = self._ols_alpha_beta(pf, bm)
            annualized_alpha = round(reg["alpha"] * self.periods_per_year, 4)

            return {
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "rolling_sharpe": rolling_sharpe,
                "alpha_beta_from_regression": reg,
                "annualized_alpha": annualized_alpha,
                "active_return_mean": round(float(np.mean(active)), 6),
                "n_periods": n,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"基准比较分析失败: {exc}",
                "tracking_error": 0.0,
                "information_ratio": 0.0,
                "rolling_sharpe": [],
                "alpha_beta_from_regression": {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0},
            }


def main() -> None:
    """使用模拟数据自测 BenchmarkAnalysis。"""
    try:
        rng = np.random.default_rng(3)
        n = 120

        bm_returns = rng.normal(0.0003, 0.012, n)
        # 组合 = 0.9倍基准的beta暴露 + 正alpha + 噪音
        pf_returns = 0.0002 + 0.9 * bm_returns + rng.normal(0, 0.006, n)

        engine = BenchmarkAnalysis(risk_free_rate=0.02)
        result = engine.analyze(pf_returns.tolist(), bm_returns.tolist(), rolling_window=20)

        print("=== 基准比较分析结果 ===")
        print(f"追踪误差: {result['tracking_error']}")
        print(f"信息比率: {result['information_ratio']}")
        print(f"Alpha/Beta回归: {result['alpha_beta_from_regression']}")
        print(f"年化Alpha: {result['annualized_alpha']}")
        print(f"滚动夏普(前5): {result['rolling_sharpe'][:5]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 测试运行失败: {exc}")


if __name__ == "__main__":
    main()
