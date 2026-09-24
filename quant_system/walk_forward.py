#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# V4.1 feature
"""
walk_forward.py — 滚动 Walk-Forward 验证框架

提供 WalkForwardValidator 类，用于：
1. 时序滚动分窗：训练集 → 预测→ 测试集（OOS），克服单一时段过拟合
2. 基于 sklearn ParameterGrid 的全参数网格搜索
3. 结合 overfitting_tests.py DSR 的稳定性检验
4. 输出 OOS 绩效汇总与参数稳定性指标

依赖
----
- numpy, pandas, scipy, sklearn
- quant_system.overfitting_tests (DSR)

参考文献
--------
- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies.
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

logger = __import__('logging').getLogger(__name__)

try:
    from quant_system.overfitting_tests import deflated_sharpe_ratio
except ImportError:
    # __main__ 直接运行时回退到相对导入
    from overfitting_tests import deflated_sharpe_ratio

# 默认窗口参数
_DEFAULT_TRAIN_WINDOW = 504   # ≈ 2 年交易日
_DEFAULT_TEST_WINDOW = 126    # ≈ 6 个月
_DEFAULT_STEP = 63            # 约 3 个月滚动一次

# 审计 2026-08-16：OOS 收益扣交易成本（此前零成本会系统性高估高换手参数）。
# 成本率 = 佣金 + 印花税 + 过户费 + 滑点（A股卖出端印花税，统一按单边近似）。
_COST_RATE = 0.0015  # ≈ 单边 15bp（佣金+滑点为主；印花税另计于卖出）


# ---------------------------------------------------------------------------
# 公共辅助函数
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: pd.Series, annual_factor: int = 244) -> float:
    """计算年化夏普比率（假设无风险利率为 0）。"""
    if len(returns) < 2:
        return 0.0
    ann_vol = returns.std() * np.sqrt(annual_factor)
    if ann_vol < 1e-12:
        return 0.0
    return returns.mean() * annual_factor / ann_vol


def _compute_max_dd(equity: pd.Series) -> float:
    """计算最大回撤（% 正数形式，如 -0.15 → 15.0）。"""
    if len(equity) < 2:
        return 0.0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(-dd.min() * 100)


def _compute_calmar(returns: pd.Series, annual_factor: int = 244) -> float:
    """Calmar 比率 = 年化收益 / 最大回撤。"""
    dd = _compute_max_dd((1 + returns).cumprod())
    if dd < 1e-6:
        return 0.0
    ann_ret = returns.mean() * annual_factor
    return ann_ret / (dd / 100.0)


def _compute_profit_factor(returns: pd.Series) -> float:
    """盈利因子 = 总盈利 / |总亏损|。"""
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses >= 0 or abs(losses) < 1e-12:
        return float("inf") if wins > 0 else 1.0
    return wins / abs(losses)


def _count_completed_trades(positions: pd.Series) -> int:
    """P2-Q17-fix(L159): 按仓位状态机统计完整开平仓对, 替代 diff().abs()>0 计数。

    0→x 为开仓(不计数), x→0 为平仓(+1 完整交易), 1→-1 翻转 = 旧仓平仓(+1)+新仓开仓。
    原实现把"仓位数值变化"当成交: 0→1 计1笔、1→-1 翻转只计1次(实为2笔),
    shift 后首日 0→x 也被误计入。
    """
    vals = positions.fillna(0).values
    n = 0
    prev = 0.0
    for v in vals:
        a = abs(v) > 1e-12
        p = abs(prev) > 1e-12
        if p and not a:
            n += 1
        elif p and a and np.sign(prev) != np.sign(v):
            n += 1
        prev = v
    return n


# ---------------------------------------------------------------------------
# WalkForwardValidator 核心类
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    滚动 Walk-Forward 验证器。

    将时间序列划分为连续的训练-测试窗口组，在每个组内对训练集进行
    参数网格搜索，选最优参数后在测试集（OOS）上评估，最后汇总全部
    OOS 绩效并进行参数稳定性检查。

    Parameters
    ----------
    train_window : int, default=504
        每个训练窗口的样本数（如交易日数），默认 ≈ 2 年。
    test_window : int, default=126
        每个测试窗口（OOS）的样本数，默认 ≈ 6 个月。
    step : int, default=63
        窗口滚动步长，默认约 3 个月。

    Notes
    -----
    若 ``step < test_window``，相邻窗口的测试期会重叠，聚合 OOS 收益时
    会对重叠日去重（重叠日取均值），避免重复计息；建议使用
    ``step >= test_window`` 的非重叠切分。
    """

    def __init__(
        self,
        train_window: int = _DEFAULT_TRAIN_WINDOW,
        test_window: int = _DEFAULT_TEST_WINDOW,
        step: int = _DEFAULT_STEP,
        embargo: int = 20,
    ) -> None:
        if train_window < 20:
            raise ValueError(f"train_window ({train_window}) 过小，至少需要 20")
        if test_window < 5:
            raise ValueError(f"test_window ({test_window}) 过小，至少需要 5")
        if step < 1:
            raise ValueError(f"step ({step}) 必须 > 0")
        if embargo < 0:
            raise ValueError(f"embargo ({embargo}) 必须 >= 0")
        if step < test_window:
            warnings.warn(
                f"[WFV] step ({step}) < test_window ({test_window})："
                f"相邻窗口测试期将重叠 {test_window - step} 个交易日，"
                f"聚合时会对重叠日收益取均值去重（Q17 修复）；"
                f"建议使用 step >= test_window 的非重叠切分。",
                RuntimeWarning,
            )

        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.embargo = embargo   # P2-Q17-fix(M157): 训练/测试间隔, 降低参数选择偏倚传导

    # ------------------------------------------------------------------
    # run() — 执行完整 Walk-Forward 验证
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        strategy_fn: Callable[..., pd.Series],
        param_grid: Dict[str, List[Any]],
        score_func: Callable[[pd.Series], float] | None = None,
        annual_factor: int = 244,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 验证。

        按时间顺序滑动窗口，对每个训练窗口用 ParameterGrid 搜索参数，
        选最优策略后测试样本外（OOS）绩效，最后汇总全时段 OOS 结果。

        Parameters
        ----------
        prices : pd.Series
            价格序列，index 为 DatetimeIndex，value 为收盘价。
        strategy_fn : Callable
            策略函数签名：strategy_fn(prices, **params) → pd.Series
            返回与 prices 等长的仓位信号 Series（-1 ~ 1 之间）。
        param_grid : Dict[str, List[Any]]
            参数网格，格式同 sklearn.model_selection.ParameterGrid。
            例如：{"ma_fast": [5, 10, 20], "ma_slow": [30, 60, 120]}。
        score_func : Callable, optional
            训练集上的打分函数，输入收益 Series（非仓位），输出越大越好。
            默认使用年化夏普比率。
        annual_factor : int, default=244
            年化因子（A 股通常 244 个交易日）。
        verbose : bool, default=True
            是否打印每个窗口的进度信息。

        Returns
        -------
        result : dict，包含以下字段：
            - "windows" : list[dict] — 每个窗口的详细结果
            - "oos_returns" : pd.Series — 全部 OOS 收益串联
            - "oos_equity" : pd.Series — 全部 OOS 净值曲线
            - "num_windows" : int — 窗口总数
            - "aggregate_sharpe" : float — 串联 OOS 年化夏普
            - "aggregate_calmar" : float — 串联 OOS Calmar 比率
            - "aggregate_max_dd" : float — 串联 OOS 最大回撤（%）
            - "aggregate_profit_factor" : float — 串联 OOS 盈利因子
            - "total_return_pct" : float — 串联 OOS 总收益率（%）
            - "best_params_per_window" : list[dict] — 每个窗口选的最优参数
            - "param_grid_size" : int — 参数网格总组合数
            - "config" : dict — 验证器配置
        """
        if not isinstance(prices, pd.Series):
            raise TypeError("prices 须为 pd.Series")
        if len(prices) < self.train_window + self.test_window:
            raise ValueError(
                f"数据长度 ({len(prices)}) 不足以支撑至少一个"
                f"完整窗口 (train={self.train_window}+test={self.test_window})"
            )

        if score_func is None:
            score_func = lambda rets: _compute_sharpe(rets, annual_factor)

        grid = list(ParameterGrid(param_grid))
        total_params = len(grid)
        if total_params == 0:
            raise ValueError("param_grid 为空，至少需一个参数组合")

        if verbose:
            print(f"[WFV] 参数网格大小: {total_params}")

        # ── 生成窗口索引 ─────────────────────────────────────────────
        n = len(prices)
        train_win, test_win, step = self.train_window, self.test_window, self.step
        embargo = self.embargo   # P2-Q17-fix(M157)

        windows: List[Dict[str, Any]] = []
        best_params_per_window: List[Dict[str, Any]] = []
        all_oos_rets: List[pd.Series] = []
        skipped_windows: List[Dict[str, Any]] = []   # P2-Q17-fix(M158)
        num_skipped = 0
        start = 0

        while start + train_win + embargo + test_win <= n:
            train_end = start + train_win
            test_start = train_end + embargo   # P2-Q17-fix(M157): 训练/测试间 embargo 间隔
            test_end = test_start + test_win

            train_idx = slice(start, train_end)
            test_idx = slice(test_start, test_end)

            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]

            if verbose:
                t0 = prices.index[start]
                t1 = prices.index[train_end - 1]
                tt0 = prices.index[test_start]
                tt1 = prices.index[test_end - 1]
                print(
                    f"  [WFV] 窗口 {len(windows)+1}: "
                    f"训练 {t0.date()}~{t1.date()} | "
                    f"测试 {tt0.date()}~{tt1.date()} (embargo {embargo})"
                )

            # ── 训练：在训练集上搜索最优参数 ─────────────────────
            best_score = -np.inf
            best_params = grid[0]
            best_train_positions = None

            for params in grid:
                try:
                    positions = strategy_fn(train_prices, **params)
                except Exception as exc:
                    if verbose:
                        print(f"    ⚠ 参数 {params} 执行错误: {exc}")
                    continue

                if not isinstance(positions, pd.Series):
                    continue
                # 仓位滞后一期，避免前视偏差
                positions = positions.shift(1).fillna(0.0).reindex(train_prices.index)
                train_rets = positions * train_prices.pct_change().fillna(0.0)
                # 审计 2026-08-16：按换手扣成本
                turnover = positions.diff().abs().fillna(positions.abs())
                train_rets = train_rets - turnover * _COST_RATE
                # P2-Q17-fix(M157): 参数选择时剔除训练窗尾部(embargo)样本,
                # 降低训练评分与紧邻测试期之间的相关性(选择偏倚传导)
                if embargo > 0 and len(train_rets) > embargo:
                    score = score_func(train_rets.iloc[:-embargo])
                else:
                    score = score_func(train_rets)

                if score > best_score:
                    best_score = score
                    best_params = params

            if verbose:
                print(f"    → 最优参数: {best_params}, 训练评分: {best_score:.4f}")

            # ── OOS 测试 ──────────────────────────────────────────
            try:
                oos_positions = strategy_fn(test_prices, **best_params)
            except Exception as exc:
                if verbose:
                    print(f"    ⚠ OOS 执行错误: {exc}")
                # P2-Q17-fix(M158): 记录跳过窗口, 不再从聚合结果中静默缺失
                num_skipped += 1
                skipped_windows.append({
                    "train_start": prices.index[start],
                    "train_end": prices.index[train_end - 1],
                    "test_start": prices.index[test_start],
                    "test_end": prices.index[test_end - 1],
                    "reason": f"OOS执行异常: {exc}",
                })
                start += step
                continue

            if not isinstance(oos_positions, pd.Series):
                # P2-Q17-fix(M158)
                num_skipped += 1
                skipped_windows.append({
                    "train_start": prices.index[start],
                    "train_end": prices.index[train_end - 1],
                    "test_start": prices.index[test_start],
                    "test_end": prices.index[test_end - 1],
                    "reason": "strategy_fn 返回非 pd.Series",
                })
                start += step
                continue

            # 仓位滞后一期，确保无前视
            oos_positions = (
                oos_positions.shift(1).fillna(0.0).reindex(test_prices.index)
            )
            oos_rets = oos_positions * test_prices.pct_change().fillna(0.0)
            # 审计 2026-08-16：OOS 按换手扣交易成本，修正零成本高估
            oos_turnover = oos_positions.diff().abs().fillna(oos_positions.abs())
            oos_rets = oos_rets - oos_turnover * _COST_RATE
            oos_equity = (1 + oos_rets).cumprod()

            window_sharpe = _compute_sharpe(oos_rets, annual_factor)
            window_max_dd = _compute_max_dd(oos_equity)
            window_trades = _count_completed_trades(oos_positions)  # P2-Q17-fix(L159)

            windows.append({
                "train_start": prices.index[start],
                "train_end": prices.index[train_end - 1],
                "test_start": prices.index[test_start],
                "test_end": prices.index[test_end - 1],
                "embargo": embargo,
                "best_params": best_params,
                "best_train_score": best_score,
                "oos_sharpe": window_sharpe,
                "oos_max_dd_pct": window_max_dd,
                "oos_total_return_pct": float(
                    (oos_equity.iloc[-1] - 1) * 100
                ),
                "oos_num_trades": int(window_trades),
                "oos_returns": oos_rets,
                "oos_equity": oos_equity,
            })

            best_params_per_window.append(dict(best_params))
            all_oos_rets.append(oos_rets)

            start += step

        # ── 无有效窗口 ───────────────────────────────────────────
        if not windows:
            result_no_windows: Dict[str, Any] = {
                "windows": [],
                "oos_returns": pd.Series(dtype=float),
                "oos_equity": pd.Series(dtype=float),
                "num_windows": 0,
                "num_skipped_windows": num_skipped,      # P2-Q17-fix(M158)
                "skipped_windows": skipped_windows,      # P2-Q17-fix(M158)
                "coverage_pct": 0.0,                     # P2-Q17-fix(M158)
                "aggregate_sharpe": 0.0,
                "aggregate_calmar": 0.0,
                "aggregate_max_dd": 0.0,
                "aggregate_profit_factor": 1.0,
                "total_return_pct": 0.0,
                "best_params_per_window": [],
                "param_grid_size": total_params,
                "config": self._config(),
            }
            if num_skipped:
                # P2-Q17-fix(M158): 全窗口被跳过时也显式标注
                result_no_windows["warning"] = (
                    f"全部 {num_skipped} 个窗口均被跳过(strategy_fn 执行异常/返回非Series), "
                    f"无有效 OOS 结果"
                )
            return result_no_windows

        # ── 串联全部 OOS 收益 ───────────────────────────────────
        # Q17 修复：当 step < test_window 时，相邻窗口测试期重叠，直接
        # concat 会产生重复日期索引，cumprod 对重叠日重复计息，导致
        # 聚合 total_return/sharpe/回撤全部失真。这里按日期去重
        # （重叠日取均值），保证聚合口径正确。
        oos_returns = pd.concat(all_oos_rets).sort_index()
        if oos_returns.index.has_duplicates:
            dup_count = int(oos_returns.index.duplicated().sum())
            warnings.warn(
                f"[WFV] 检测到 {dup_count} 个重叠日期（step < test_window），"
                f"聚合时对重叠日收益取均值去重。",
                RuntimeWarning,
            )
            oos_returns = oos_returns.groupby(level=0).mean().sort_index()
        oos_equity = (1 + oos_returns).cumprod()

        result: Dict[str, Any] = {
            "windows": windows,
            "oos_returns": oos_returns,
            "oos_equity": oos_equity,
            "num_windows": len(windows),
            "num_skipped_windows": num_skipped,          # P2-Q17-fix(M158)
            "skipped_windows": skipped_windows,          # P2-Q17-fix(M158)
            "coverage_pct": round(len(windows) / (len(windows) + num_skipped), 4)
            if (len(windows) + num_skipped) > 0 else 0.0,  # P2-Q17-fix(M158)
            "aggregate_sharpe": _compute_sharpe(oos_returns, annual_factor),
            "aggregate_calmar": _compute_calmar(oos_returns, annual_factor),
            "aggregate_max_dd": _compute_max_dd(oos_equity),
            "aggregate_profit_factor": _compute_profit_factor(oos_returns),
            "total_return_pct": float((oos_equity.iloc[-1] - 1) * 100),
            "best_params_per_window": best_params_per_window,
            "param_grid_size": total_params,
            "config": self._config(),
        }

        # P2-Q17-fix(M158): 存在跳过窗口时在结果中显式标注, 避免聚合指标被当成完整覆盖
        if num_skipped:
            result["warning"] = (
                f"存在 {num_skipped} 个跳过窗口(strategy_fn 执行异常/返回非Series), "
                f"OOS 覆盖度 {result['coverage_pct']:.1%}"
            )

        if verbose:
            print(f"\n[WFV] 完成: {len(windows)} 个窗口 (跳过 {num_skipped})")
            if num_skipped:
                print(f"      ⚠ 存在跳过窗口, OOS 覆盖度 {result['coverage_pct']:.1%}")
            print(f"      OOS 年化夏普:  {result['aggregate_sharpe']:.3f}")
            print(f"      OOS 最大回撤:  {result['aggregate_max_dd']:.2f}%")
            print(f"      OOS Calmar:    {result['aggregate_calmar']:.3f}")
            print(f"      OOS 总收益率:  {result['total_return_pct']:.2f}%")

        return result

    # ------------------------------------------------------------------
    # stability_check() — 参数稳定性检验
    # ------------------------------------------------------------------

    def stability_check(
        self,
        oos_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        分析 Walk-Forward 结果的参数稳定性。

        稳定性检查包括：
        1. **参数一致性**: 各窗口选出的最优参数是否一致（频率分布）。
        2. **参数波动率**: 连续窗口间参数变化的剧烈程度。
        3. **IS/OOS 相关性**: 各窗口训练分与测试分之间的 Spearman 秩相关，
           反映参数选择是否稳定预测未来（而非噪声拟合）。
        4. **Deflated Sharpe Ratio (DSR)**: 将每个窗口看作一次独立试验，
           计算串联 OOS 的 DSR，校正多重比较偏误。
        5. **Rank 信息系数 (Rank IC)**: 训练排名 → OOS 排名的跨期一致性。

        Parameters
        ----------
        oos_results : dict
            WalkForwardValidator.run() 的返回结果。

        Returns
        -------
        dict，包含以下字段：
            - "param_choice_freq" : dict — 每个参数组合被选中的次数 / 总窗口数
            - "param_volatility" : float — 窗口间参数变化的分段（0~1），1=每次变化
            - "is_oos_spearman_r" : float — 训练分与 OOS 分的 Spearman 秩相关系数
            - "is_oos_spearman_p" : float — 对应 p 值
            - "rank_ic_mean" : float — 秩 IC 均值（可选：若无分窗口全参数结果则为 NaN）
            - "oos_dsr" : float — 串联 OOS 收益的 Deflated Sharpe Ratio
            - "oos_dsr_pvalue" : float — DSR 对应 p 值
            - "oos_dsr_num_trials" : int — DSR 使用的试验次数
            - "best_param_consistency" : float — 最常见参数组合的出现比例
            - "selected_param_stats" : dict — 选中最优参数组合的统计摘要
        """
        windows = oos_results.get("windows", [])
        if not windows:
            return {"error": "无有效窗口，无法进行稳定性检查"}

        num_windows = len(windows)
        param_grid_size = oos_results.get("param_grid_size", 1)

        # ── 1. 参数选择频率 ──────────────────────────────────────
        param_keys = list(windows[0]["best_params"].keys())
        param_counts: Dict[str, int] = {}
        for w in windows:
            # 将参数组合序列化为字符串作为 key
            comb = tuple(sorted(w["best_params"].items()))
            comb_str = str(dict(comb))
            param_counts[comb_str] = param_counts.get(comb_str, 0) + 1

        # 计算出现比例
        param_choice_freq = {
            k: round(v / num_windows, 4) for k, v in param_counts.items()
        }
        # 频率从高到低排序
        param_choice_freq = dict(
            sorted(param_choice_freq.items(), key=lambda x: -x[1])
        )

        # ── 2. 参数变化率（volatility） ──────────────────────────
        # 连续窗口间参数是否变化
        changes = 0
        for i in range(1, num_windows):
            if windows[i]["best_params"] != windows[i - 1]["best_params"]:
                changes += 1
        param_volatility = changes / max(num_windows - 1, 1)

        # ── 3. IS/OOS Spearman 秩相关 ────────────────────────────
        train_scores = [w["best_train_score"] for w in windows]
        oos_sharpes = [w["oos_sharpe"] for w in windows]

        if len(set(train_scores)) > 1 and len(set(oos_sharpes)) > 1:
            # P2-Q17-fix(M160): scipy 缺失时返回 NaN 而非 NameError 崩溃
            try:
                from scipy.stats import spearmanr
                sp_r, sp_p = spearmanr(train_scores, oos_sharpes)
            except ImportError:
                logger.warning("Q17-M160: scipy 不可用, IS/OOS Spearman 秩相关返回 NaN")
                sp_r, sp_p = float("nan"), float("nan")
        else:
            sp_r, sp_p = 0.0, 1.0

        # ── 4. Deflated Sharpe Ratio（DSR） ──────────────────────
        oos_rets = oos_results.get("oos_returns")
        if isinstance(oos_rets, pd.Series) and len(oos_rets) > 5:
            sharpe_val = oos_results.get("aggregate_sharpe", _compute_sharpe(oos_rets))
            n_obs = len(oos_rets)
            skew = float(oos_rets.skew())
            kurt = float(oos_rets.kurtosis()) + 3  # scipy → 传统峰度

            dsr_val, dsr_p = deflated_sharpe_ratio(
                sharpe=sharpe_val,
                num_trials=param_grid_size,
                num_observations=n_obs,
                skew=skew,
                kurtosis=kurt,
                target_sharpe=0.0,
            )
        else:
            dsr_val, dsr_p = 0.0, 1.0

        # ── 5. Rank IC（秩信息系数）—— 窗口内参数排名一致性   ──
        # 注意：标准 Walk-Forward 仅保留最优参数，要完整计算 Rank IC
        # 需要在每个窗口保留全部参数的 OOS 评分；此处返回 NaN 并提示。
        rank_ic_mean = float("nan")

        # ── 6. 最常见参数组合的一致性比例 ───────────────────────
        best_consistency = max(param_counts.values()) / num_windows if param_counts else 0.0

        # ── 7. 被选参数统计摘要（数值型参数） ──────────────────
        selected_params_df = pd.DataFrame(
            [w["best_params"] for w in windows]
        )
        selected_param_stats = {}
        # 对每列数值型参数计算统计量
        for col in selected_params_df.columns:
            col_data = selected_params_df[col]
            if pd.api.types.is_numeric_dtype(col_data):
                selected_param_stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "unique": int(col_data.nunique()),
                    "mode": float(col_data.mode().iloc[0])
                    if not col_data.mode().empty else None,
                }
            else:
                # 分类参数：仅统计唯一值数量与最常见值
                selected_param_stats[col] = {
                    "unique": int(col_data.nunique()),
                    "mode": str(col_data.mode().iloc[0])
                    if not col_data.mode().empty else None,
                }

        return {
            "param_choice_freq": param_choice_freq,
            "param_volatility": param_volatility,
            "is_oos_spearman_r": sp_r,
            "is_oos_spearman_p": sp_p,
            "rank_ic_mean": rank_ic_mean,
            "oos_dsr": dsr_val,
            "oos_dsr_pvalue": dsr_p,
            "oos_dsr_num_trials": param_grid_size,
            "best_param_consistency": best_consistency,
            "selected_param_stats": selected_param_stats,
        }

    # ------------------------------------------------------------------
    # 配置序列化
    # ------------------------------------------------------------------

    def _config(self) -> Dict[str, int]:
        """返回验证器当前的窗口配置。"""
        return {
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step": self.step,
            "embargo": self.embargo,
        }

    def __repr__(self) -> str:
        return (
            f"WalkForwardValidator(train={self.train_window}, "
            f"test={self.test_window}, step={self.step}, "
            f"embargo={self.embargo})"
        )


# ---------------------------------------------------------------------------
# 使用示例（main / doctest 级演示）
# ---------------------------------------------------------------------------

def _demo_strategy(prices: pd.Series, *, ma_fast: int, ma_slow: int) -> pd.Series:
    """
    示例策略：双均线交叉。
    快均线 > 慢均线 → 多头 (+1)，否则空仓 (0)。
    """
    if len(prices) < max(ma_fast, ma_slow) + 1:
        return pd.Series(0.0, index=prices.index)

    fast = prices.rolling(ma_fast).mean()
    slow = prices.rolling(ma_slow).mean()
    signal = (fast > slow).astype(float)
    return signal


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from overfitting_tests import deflated_sharpe_ratio

    # ── 生成模拟价格并演示 ─────────────────────────────────────
    np.random.seed(42)
    n = 3000
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    price = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    prices = pd.Series(price, index=dates, name="close")

    wfv = WalkForwardValidator(train_window=504, test_window=126, step=63)

    param_grid = {
        "ma_fast": [5, 10, 20, 30],
        "ma_slow": [60, 120, 200],
    }

    result = wfv.run(prices, _demo_strategy, param_grid, verbose=True)

    if result["num_windows"] > 0:
        stability = wfv.stability_check(result)
        print(f"\n参数变化率:          {stability['param_volatility']:.2%}")
        print(f"IS/OOS Spearman r:   {stability['is_oos_spearman_r']:.4f}")
        print(f"DSR:                 {stability['oos_dsr']:.3f} "
              f"(p={stability['oos_dsr_pvalue']:.4f})")
        print(f"参数一致性:          {stability['best_param_consistency']:.2%}")
        print(f"最常见参数:          {list(stability['param_choice_freq'].keys())[0]}")
