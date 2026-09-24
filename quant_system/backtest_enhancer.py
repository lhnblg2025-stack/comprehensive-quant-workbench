"""
DEPRECATED (2026-08-07): 无生产引用，保留仅供参考。主用模块见 项目文档/量化交易系统/系统梳理报告.md

D2收敛登记 (2026-08-11): 独立能力保留——存活偏差/前视偏差/子周期/MC 模拟/
参数敏感性等稳健性检验与归因 backtest_engine 未覆盖；integration_tests.py
仍引用 BacktestRobustness，文件与公开 API 均保留，不转发、不强迁。
"""

"""
backtest_enhancer.py — 回测稳健性与归因检验
V4.1 feature: 深度回测

提供回测结果的多维度稳健性验证：
1. 存活偏差检测
2. 前视偏差检测
3. 子周期稳定性检验
4. Monte Carlo 模拟
5. 参数敏感性分析
6. 回测绩效归因
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestRobustness:
    """回测稳健性检验套件

    D2收敛登记: 独立能力保留——稳健性检验 backtest_engine 未覆盖。
    """

    @staticmethod
    def check_survivorship_bias(
        backtest_symbols: list[str], all_symbols: list[str]
    ) -> dict:
        """检测存活偏差

        Parameters
        ----------
        backtest_symbols : list[str]
            回测中使用的股票代码列表
        all_symbols : list[str]
            同期市场上所有股票代码（含已退市）

        Returns
        -------
        dict
            - missing: 缺失的股票（已退市）
            - missing_ratio: 缺失比例
            - bias_risk: 'high' / 'medium' / 'low'
        """
        missing = sorted(set(all_symbols) - set(backtest_symbols))
        ratio = len(missing) / max(len(all_symbols), 1)

        if ratio > 0.3:
            risk = "high"
        elif ratio > 0.1:
            risk = "medium"
        else:
            risk = "low"

        return {
            "missing": missing[:20],  # 仅显示前20只
            "missing_count": len(missing),
            "missing_ratio": ratio,
            "bias_risk": risk,
        }

    @staticmethod
    def check_lookahead_bias(prices: pd.DataFrame, signals: pd.DataFrame) -> dict:
        """前视偏差检测

        正确的检测口径：核对"信号计算所用数据的时间戳 ≤ 信号日期"——
        即信号日期必须落在价格数据可用范围内（信号发出时该价格已实现）。
        原实现把"信号与次日收益的每一对观测"都记为 issue（最多100条/列），
        而信号与未来收益相关恰是策略预测力的正常体现 → 必然误报，连全零
        信号策略也会被判 issues_found=100, severity=high。
        """
        if not isinstance(signals.index, pd.DatetimeIndex):
            return {"error": "需要 DatetimeIndex"}

        issues = []
        if not isinstance(prices.index, pd.DatetimeIndex):
            return {"error": "prices 需要 DatetimeIndex", "issues_found": 0, "severity": "unknown"}

        price_dates = prices.index.sort_values()
        if price_dates.empty:
            return {"issues_found": 0, "sample_issues": [], "severity": "low",
                    "method": "时序核对（prices 为空）"}

        # P1-Q14-fix(H09): 仅做时序核对，不再用"信号-未来收益相关性"判前视。
        for col in signals.columns:
            if col not in prices.columns:
                continue
            sig_dates = signals[col].dropna().index
            for d in sig_dates:
                # 1) 信号日期必须在价格数据中（信号由当日收盘数据生成）
                if d not in prices.index:
                    issues.append({"date": str(d), "symbol": col,
                                   "problem": "信号日期不在价格数据中（数据未对齐）"})
                    continue
                # 2) 信号日期不能晚于价格数据末端（无未来数据可用）
                if d > price_dates[-1]:
                    issues.append({"date": str(d), "symbol": col,
                                   "problem": "信号日期晚于价格数据末端"})
                    continue
                # 3) 信号日期在价格数据范围内即视为已实现（d ≤ 最新价格日期）

        severity = "high" if len(issues) > 0 else "low"
        return {
            "issues_found": len(issues),
            "sample_issues": issues[:5],
            "severity": severity,
            "method": "时序核对：信号日期必须存在于价格数据且不晚于数据末端",
        }

    @staticmethod
    def subperiod_test(returns: pd.Series, n_periods: int = 4) -> pd.DataFrame:
        """子周期稳定性检验

        将回测期等分为 n_periods 个子周期，分别计算绩效指标。
        """
        n = len(returns)
        chunk = n // n_periods
        results = []
        for i in range(n_periods):
            start = i * chunk
            end = start + chunk if i < n_periods - 1 else n
            sub_ret = returns.iloc[start:end]
            sharpe = sub_ret.mean() / max(sub_ret.std(), 1e-12) * np.sqrt(252)
            results.append(
                {
                    "period": i + 1,
                    "start": returns.index[start],
                    "end": returns.index[min(end, n) - 1],
                    "returns": sub_ret.sum(),
                    "volatility": sub_ret.std() * np.sqrt(252),
                    "sharpe": sharpe,
                    "max_drawdown": BacktestRobustness._max_dd(sub_ret),
                }
            )
        return pd.DataFrame(results)

    @staticmethod
    def monte_carlo_simulation(
        returns: pd.Series, n_simulations: int = 1000, n_periods: int = 252
    ) -> pd.DataFrame:
        """Monte Carlo 模拟

        对原收益序列做 Bootstrap 重采样，评估绩效分布。
        """
        np.random.seed(42)
        simulated = np.random.choice(
            returns.values,
            size=(n_simulations, min(n_periods, len(returns))),
            replace=True,
        )
        sim_sharpes = (
            simulated.mean(axis=1)
            / np.maximum(simulated.std(axis=1), 1e-12)
            * np.sqrt(252)
        )

        return pd.DataFrame(
            {
                "sim_sharpe": sim_sharpes,
                "pct_5": np.percentile(sim_sharpes, 5),
                "pct_25": np.percentile(sim_sharpes, 25),
                "pct_50": np.percentile(sim_sharpes, 50),
                "pct_75": np.percentile(sim_sharpes, 75),
                "pct_95": np.percentile(sim_sharpes, 95),
                "prob_positive_sharpe": (sim_sharpes > 0).mean(),
            }
        )

    @staticmethod
    def parameter_sensitivity(test_fn: Callable, param_grid: dict) -> pd.DataFrame:
        """参数敏感性分析

        Parameters
        ----------
        test_fn : callable
            接收 **params 返回绩效指标的测试函数
        param_grid : dict
            参数网格，如 {'window': [10, 20, 30], 'threshold': [0.5, 0.8]}
        """
        from itertools import product

        keys = list(param_grid.keys())
        results = []
        for values in product(*param_grid.values()):
            params = dict(zip(keys, values))
            metrics = test_fn(**params)
            metrics.update(params)
            results.append(metrics)
        return pd.DataFrame(results)

    @staticmethod
    def _max_dd(returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return dd.min()


class BacktestAttribution:
    """回测归因分析

    D2收敛登记: 独立能力保留——收益/因子归因 backtest_engine 未覆盖。
    """

    # P2-Q14-fix(L094): 删除重复的 @staticmethod 装饰器（原双重装饰在
    # Python<3.10 下调用会抛 TypeError）。
    @staticmethod
    def return_attribution(
        weights: pd.DataFrame, stock_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """收益归因：选股收益 + 行业配置收益

        Parameters
        ----------
        weights : pd.DataFrame
            (date x symbol) 持仓权重
        stock_returns : pd.DataFrame
            (date x symbol) 个股收益率

        Returns
        -------
        pd.DataFrame
            (date, ) 列: total_return, stock_selection, allocation
        """
        common_symbols = weights.columns.intersection(stock_returns.columns)
        if len(common_symbols) == 0:
            return pd.DataFrame()

        w = weights[common_symbols].fillna(0)
        r = stock_returns[common_symbols].fillna(0)

        # 组合收益 = Σ wi * ri
        portfolio_ret = (w * r).sum(axis=1)

        # 简单等权基准收益
        benchmark_ret = r.mean(axis=1)

        # 选股收益 = 权重不变下的超额收益
        avg_w = w.mean(axis=0)
        stock_selection = ((w.div(w.sum(axis=1), axis=0) - avg_w) * r).sum(axis=1)

        return pd.DataFrame({
            "total_return": portfolio_ret,
            "stock_selection": stock_selection,
            "alloc_effect": portfolio_ret - stock_selection,
            "excess_return": portfolio_ret - benchmark_ret,
        })

    @staticmethod
    def factor_attribution(
        weights: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """归因到因子（行业、风格）

        Parameters
        ----------
        weights : pd.DataFrame
            (date x symbol) 持仓权重
        factor_exposures : pd.DataFrame
            (symbol x factor) 因子暴露
        factor_returns : pd.DataFrame
            (date x factor) 因子收益

        Returns
        -------
        pd.DataFrame
            (date, ) 列: 每期组合因子贡献 + total
        """
        # P1-Q14-fix(H07): 原实现完全忽略 weights，直接返回 factor_returns 原值并加
        # total=sum(因子收益)，并非组合因子归因，误导下游。现按文档实现：
        #   组合因子暴露(date, factor) = Σ_symbol 权重(symbol) × 暴露(symbol, factor)
        #   因子贡献(date, factor)     = 组合暴露 × 因子收益
        # 注：specific_return 需个股实际收益，本函数签名未提供，故不输出。
        common_symbols = weights.columns.intersection(factor_exposures.index)
        common_factors = factor_returns.columns.intersection(factor_exposures.columns)
        if len(common_symbols) == 0 or len(common_factors) == 0:
            return pd.DataFrame()

        dates = weights.index.intersection(factor_returns.index)
        if len(dates) == 0:
            return pd.DataFrame()

        w = weights.loc[dates, common_symbols].fillna(0)
        exposure = factor_exposures.loc[common_symbols, common_factors].fillna(0)

        # 组合因子暴露 = 权重 × 暴露  (date x factor)
        port_exposure = w @ exposure
        # 因子贡献 = 暴露 × 因子收益
        contributions = port_exposure.multiply(factor_returns.loc[dates, common_factors])

        result = contributions.copy()
        result["total"] = contributions.sum(axis=1)
        return result

    @staticmethod
    def summary_stats(returns: pd.Series) -> dict:
        """综合绩效指标"""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        # 计算各项指标
        total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        ann_ret = (1 + total_ret) ** (252 / max(len(returns), 1)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-12)

        # Calmar
        max_dd = dd.min()
        calmar = ann_ret / max(abs(max_dd), 1e-12)

        # 胜率
        win_rate = (returns > 0).mean()

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_loss_ratio = avg_win / max(avg_loss, 1e-12)

        return {
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            # P2-Q14-fix(M095): 原 n_trades 实为收益观测数而非交易笔数，
            # 指标名误导。summary_stats 只接收 returns 无法获得真实交易计数，
            # 故改名为 n_observations。
            "n_observations": len(returns),
        }

