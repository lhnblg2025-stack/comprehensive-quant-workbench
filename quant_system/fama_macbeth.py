"""
fama_macbeth.py — V8⁺ Fama-MacBeth 截面回归因子收益率估计

经典两步法:
  Step 1: 每天做截面回归  R_{i,t+1} = α_t + Σβ_{f,t}·Factor_{i,f,t} + ε_{i,t}
  Step 2: 对 β_{f,t} 做时间序列统计 (t-stat, mean, std, IR)

输出:
  - 每个因子的平均收益率、t-stat（判断因子是否显著）
  - 因子收益率协方差矩阵（可喂给 portfolio optimizer）
  - 因子自相关性、累计因子收益率曲线
"""

from __future__ import annotations
import logging

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_system.factor_model import get_model
from quant_system.data_store import get_store

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANNUAL_TRADING_DAYS = 242
# P2-Q6-fix (Q6-L461): 与注释"默认传12只以上"统一为 12；8 只股票做截面回归样本过小
MIN_CROSS_SECTION = 12


class FamaMacBeth:
    """Fama-MacBeth 截面回归因子收益率估计器"""

    def __init__(self):
        self._factor_model = get_model()
        self._store = get_store()

    # ── Step 1: 日度截面回归 ──────────────────────────────

    def cross_section_regression(
        self,
        date_str: str,
        symbols: list[str] | None = None,
        forward_days: int = 1,
        fwd_data: dict[str, tuple[list[str], list[float]]] | None = None,
    ) -> dict[str, Any]:
        """Run one day of cross-sectional regression.

        fwd_data: optional pre-loaded {symbol: (date_list, close_list)} to speed up multi-date runs.

        Returns:
          - factor_returns: {factor_name: coefficient}
          - alpha: intercept
          - t_stats: {factor_name: t-stat}
          - r_squared: float
          - n: int (number of stocks in regression)
        """
        # 1. 因子暴露
        factor_df = self._factor_model.compute_factors(date_str, symbols)
        if factor_df.empty or len(factor_df) < MIN_CROSS_SECTION:
            return {"n": 0, "status": "insufficient_cross_section"}

        # 2. 未来收益率 (forward_days)
        fwd_returns = []
        valid_symbols = []
        for sym in factor_df.index:
            try:
                if fwd_data and sym in fwd_data:
                    # P2-Q6-fix (Q6-M460): fwd_data 现在携带 volume 序列用于停牌剔除
                    dates, close_vals = fwd_data[sym][0], fwd_data[sym][1]
                    vol_vals = fwd_data[sym][2] if len(fwd_data[sym]) > 2 else None
                else:
                    # P2-Q6-fix (Q6-M460): fill_gaps=False 保留停牌日原始行（volume=0），
                    #   停牌日收盘价不再被前向填充成"未来收益=0"的假数据
                    df = self._store.get(sym, days=forward_days + 10, fill_gaps=False)
                    if df.empty:
                        continue
                    dates = [str(d)[:10] for d in df["date"].values]
                    close_vals = df["close"].values.tolist()
                    vol_vals = df["volume"].values.tolist() if "volume" in df.columns else None
                if date_str not in dates:
                    continue
                idx = dates.index(date_str)
                if idx + forward_days >= len(dates):
                    continue
                # P2-Q6-fix (Q6-M460): 停牌（volume=0）样本剔除 —— 停牌日 close 为
                #   最后成交价，forward 收益为 0 且暴露陈旧，污染截面回归
                base_close = float(close_vals[idx])
                if not math.isfinite(base_close) or base_close <= 0:
                    continue
                if vol_vals is not None and idx < len(vol_vals) and float(vol_vals[idx]) <= 0:
                    continue
                fwd_close = float(close_vals[idx + forward_days])
                if vol_vals is not None and idx + forward_days < len(vol_vals) and float(vol_vals[idx + forward_days]) <= 0:
                    continue
                ret_fwd = (fwd_close - base_close) / base_close
                if not math.isfinite(ret_fwd):
                    continue
                fwd_returns.append(float(ret_fwd))
                valid_symbols.append(sym)
            except Exception as e:
                logging.getLogger(__name__).error(f"[fama_macbeth] 操作失败: {e}", exc_info=True)
                continue

        if len(valid_symbols) < MIN_CROSS_SECTION:
            return {"n": len(valid_symbols), "status": "too_few_forward_returns"}

        # 3. 对齐
        X_df = factor_df.loc[valid_symbols]
        y = np.array(fwd_returns)

        # Remove NaN rows
        valid_mask = ~np.isnan(y)
        X_df = X_df.loc[valid_mask]
        y = y[valid_mask]

        if len(y) < MIN_CROSS_SECTION:
            return {"n": len(y), "status": "too_few_valid"}

        # 4. 截面回归（OLS；因子数 ≥ 股票数时退化到轻罚 Ridge）
        from sklearn.linear_model import LinearRegression

        X = X_df.values.astype(np.float64)
        feature_names = list(X_df.columns)

        # Standardize X
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        X_std[X_std < 1e-12] = 1.0
        X = (X - X_mean) / X_std
        X = np.where(np.isfinite(X), X, 0.0)

        # P2-Q6-fix (Q6-M457): 共线性剔除 —— 标准化后 |ρ|>0.999 的因子列只保留首个
        # （典型触发：ln_cap 与 ln_float_cap 完全同值），避免 X.T@X 奇异导致整个
        # 日期循环被 np.linalg.inv 的 LinAlgError 中断。
        if X.shape[1] > 1:
            keep_idx: list[int] = []
            for i in range(X.shape[1]):
                is_dup = False
                for j in keep_idx:
                    with np.errstate(all="ignore"):  # 常量列 corrcoef 会除零，仅告警
                        c = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    if not np.isfinite(c) or abs(c) <= 0.999:
                        continue
                    is_dup = True
                    break
                if not is_dup:
                    keep_idx.append(i)
            if len(keep_idx) < X.shape[1]:
                X = X[:, keep_idx]
                feature_names = [feature_names[i] for i in keep_idx]

        n_stocks, n_factors = X.shape
        # P2-Q6-fix (Q6-M462): 原实现按方差逐日筛选因子（n_factors>=n_stocks 时仅留
        #   max(n_stocks//2,3) 个），因子集逐日漂移且信息损失大。改为：因子数≥股票数时
        #   用轻罚 Ridge（α=1e-3）保留全部因子做全截面回归；否则 OLS。因子集在 run()
        #   层由并集固定、缺失期置 NaN 剔除（见 Q6-M456）。
        use_ridge = n_factors >= n_stocks
        if use_ridge:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1e-3, fit_intercept=True)
        else:
            model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        # 计算 t-stat（协方差用 pinv，避免奇异矩阵 LinAlgError）
        n, p = X.shape
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / max(n - p - 1, 1)
        try:
            if use_ridge:
                # Ridge 协方差 (X'X + αI)^{-1}
                gram = X.T @ X + 1e-3 * np.eye(p)
            else:
                gram = X.T @ X
            var_beta = mse * np.linalg.pinv(gram).diagonal() if n > p + 1 else np.ones(p) * 1e6
        except np.linalg.LinAlgError:
            var_beta = np.ones(p) * 1e6
        se_beta = np.sqrt(np.maximum(var_beta, 1e-12))
        t_stats = np.where(se_beta > 1e-12, model.coef_ / np.maximum(se_beta, 1e-12), 0.0)

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-12)

        return {
            "date": date_str,
            "n": int(n),
            "alpha": round(float(model.intercept_), 6),
            "factor_returns": {name: round(float(coef), 6) for name, coef in zip(feature_names, model.coef_)},
            "t_stats": {name: round(float(t), 4) for name, t in zip(feature_names, t_stats)},
            "r_squared": round(float(r_squared), 4),
            "status": "ok",
        }

    # ── Step 2: 时间序列统计 ──────────────────────────────

    @staticmethod
    def _newey_west_se(betas: np.ndarray, max_lags: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Newey-West (1987) HAC standard errors for a (T, K) beta matrix.

        P2-Q6-fix (Q6-M456): 逐列计算并剔除该列 NaN（因子缺失期），避免 NaN 泄漏进
        mean/自协方差；缺失列返回 NaN（由调用方按缺数据处理）。
        """
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 2:
            raise ValueError("betas must be a 2D array with shape (T, K)")
        T, K = betas.shape
        if T == 0 or K == 0:
            return np.array([]), np.array([])
        if max_lags is None:
            max_lags = int(4 * (T / 100) ** (2 / 9))
        max_lags = max(0, min(int(max_lags), T - 1))

        se = np.full(K, np.nan, dtype=np.float64)
        t_stats = np.full(K, np.nan, dtype=np.float64)
        for k in range(K):
            col = betas[:, k]
            col = col[~np.isnan(col)]
            n = len(col)
            if n < 2:
                continue
            mean_beta = float(np.mean(col))
            centered = col - mean_beta
            omega = float(np.sum(centered ** 2))
            ml = min(max_lags, n - 1)
            for lag in range(1, ml + 1):
                weight = 1 - lag / (ml + 1)
                gamma = float(np.sum(centered[lag:] * centered[:-lag]))
                omega += 2 * weight * gamma
            variance = omega / (n * n)
            se[k] = float(np.sqrt(max(variance, 1e-12)))
            t_stats[k] = mean_beta / se[k] if se[k] > 0 else np.nan
        return se, t_stats

    def run(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2026-04-01",
        end_date: str | None = None,
        forward_days: int = 1,
        step_days: int = 5,
        min_step_days: int = 1,
        min_sections: int = 8,
    ) -> dict[str, Any]:
        """Run Fama-MacBeth over a date range.

        Returns full results + time-series statistics for each factor.

        P2-Q6-fix (Q6-M454): ``daily_results`` 返回**全量**逐期结果（与各统计量同
        一样本口径）；``n_dates_skipped`` 报告因异常被跳过的日期数。
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start = datetime.strptime(start_date[:10], "%Y-%m-%d")
        end = datetime.strptime(end_date[:10], "%Y-%m-%d")

        requested_step_days = max(int(step_days), 1)
        min_step_days = max(int(min_step_days), 1)
        target_sections = max(int(min_sections), 5)

        # Pre-load forward return data for all symbols to avoid repeated DB queries
        all_symbols = list(symbols) if symbols else []
        # P2-Q6-fix (Q6-M460): fwd_data 扩展携带 volume 序列；fill_gaps=False 保留
        #   停牌日原始行（volume=0），供截面回归前剔除停牌样本
        fwd_data: dict[str, tuple[list[str], list[float], list[float]]] = {}
        for sym in all_symbols:
            try:
                df = self._store.get(
                    sym,
                    days=(end - start).days + forward_days + 10,
                    fill_gaps=False,
                )
                if df is not None and not df.empty:
                    fwd_data[sym] = (
                        [str(d)[:10] for d in df["date"].values],
                        df["close"].values.tolist(),
                        df["volume"].values.tolist() if "volume" in df.columns else [],
                    )
            except Exception as e:
                logging.getLogger(__name__).error(f"[fama_macbeth] 操作失败: {e}", exc_info=True)
                continue

        daily_results: list[dict] = []
        dates: list[str] = []
        n_skipped = 0
        actual_step_days = requested_step_days
        step = requested_step_days
        while True:
            trial_results: list[dict] = []
            trial_dates: list[str] = []
            probe = start
            while probe <= end:
                d = probe.strftime("%Y-%m-%d")
                try:
                    result = self.cross_section_regression(d, symbols, forward_days, fwd_data=fwd_data)
                except Exception:
                    # P2-Q6-fix (Q6-M457): 单日截面回归异常不再中断整个日期循环，
                    #   跳过该日期，并在返回中可见地报告跳过次数
                    result = None
                    n_skipped += 1
                if result is not None and result.get("status") == "ok":
                    trial_results.append(result)
                    trial_dates.append(d)
                probe += timedelta(days=step)

            daily_results = trial_results
            dates = trial_dates
            actual_step_days = step
            if len(daily_results) >= target_sections or step <= min_step_days:
                break
            step = max(min_step_days, step // 2)

        if not daily_results:
            return {"status": "no_results", "n_dates": 0}

        # 2. 时间序列统计
        # P2-Q6-fix (Q6-M456): 因子集取所有截面并集（逐日 compute_factors 会因因子
        #   剔除阈值/幸存股票变化导致因子集漂移）；某期缺失的因子置 NaN 而非 0.0，
        #   统计时按因子逐列剔除，避免 0 稀释均值/波动/协方差产生虚假零收益。
        all_factors = sorted(set().union(*(dr["factor_returns"].keys() for dr in daily_results)))
        ts_data: dict[str, list[float]] = {f: [] for f in all_factors}
        ts_alphas: list[float] = []
        ts_r2: list[float] = []

        for dr in daily_results:
            fr = dr["factor_returns"]
            for f in all_factors:
                ts_data[f].append(fr.get(f, np.nan))
            ts_alphas.append(dr.get("alpha", 0.0))
            ts_r2.append(dr.get("r_squared", 0.0))

        # 3. 因子统计
        factor_results: dict[str, dict[str, Any]] = {}
        for f in all_factors:
            vals = np.asarray(ts_data[f], dtype=np.float64)
            # P2-Q6-fix (Q6-M456): 剔除该因子缺失期（NaN），统计基于有效样本
            vals = vals[~np.isnan(vals)]
            n_obs = int(len(vals))
            if n_obs < 2:
                factor_results[f] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "t_stat": 0.0,
                    "sharpe_annualized": 0.0,
                    # P1-Q6-fix: 保留旧字段名作为兼容别名
                    "sharpe_monthly": 0.0,
                    "autocorr": 0.0,
                    "cum_return_pct": 0.0,
                    "n_obs": n_obs,
                    "significant_5pct": False,
                }
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1))
            t_stat = mu / (sigma / math.sqrt(n_obs)) if sigma > 1e-12 else 0.0
            # P1-Q6-fix: 因子收益按 step_days 个交易日间隔采样，年化需用实际间隔
            # sqrt(ANNUAL_TRADING_DAYS / actual_step_days)（原实现少除以 step_days，
            # 年化 Sharpe/IR 被高估 sqrt(step_days)≈2.24~3.16 倍）。
            # V11 审计修复（Medium）: 原年化只考虑 step_days（采样间隔），
            # 忽略 forward_days——每期收益实际跨度 step_days×forward_days 个交易日，
            # 年化 Sharpe/IR 被高估 sqrt(forward_days) 倍。
            # 修正: 年化因子用 (step_days × forward_days) 总跨度。
            period_span = max(actual_step_days * forward_days, 1)
            ann_factor = math.sqrt(ANNUAL_TRADING_DAYS / period_span)
            sharp_annualized = mu / sigma * ann_factor if sigma > 1e-12 else 0.0
            # 自相关（P2-Q6-fix (Q6-L458): 序列恒定时 corrcoef 返回 NaN，置 0 防泄漏）
            autocorr = 0.0
            if len(vals) > 2:
                with np.errstate(all="ignore"):
                    autocorr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
                if not np.isfinite(autocorr):
                    autocorr = 0.0
            # 累计收益率（防御极端负收益导致 cumprod 变号）
            cum_return = float(np.cumprod(np.maximum(1 + vals, 1e-8))[-1] - 1) * 100

            factor_results[f] = {
                "mean": round(float(mu), 6),
                "std": round(float(sigma), 6),
                "t_stat": round(float(t_stat), 4),
                "sharpe_annualized": round(float(sharp_annualized), 4),
                # P1-Q6-fix: 保留旧字段名作为兼容别名，值已是修正后的年化值
                "sharpe_monthly": round(float(sharp_annualized), 4),
                "autocorr": round(float(autocorr), 4),
                "cum_return_pct": round(float(cum_return), 2),
                "n_obs": n_obs,
                # P2-Q6-fix (Q6-L459): significant_5pct 口径为普通 t 检验（|t|>1.96），
                #   另提供 significant_5pct_nw（Newey-West HAC t 口径，见下）
                "significant_5pct": abs(t_stat) > 1.96,
            }

        # 3b. FDR 多重检验校正 (Benjamini-Hochberg)
        fdr_q = 0.10  # 10% false discovery rate
        from scipy.stats import t as _t_dist
        fdr_pvals = []
        for f_name in all_factors:
            fr = factor_results[f_name]
            t_val = abs(fr["t_stat"])
            # P2-Q6-fix (Q6-M456): p 值自由度按该因子有效样本数（n_obs-1）而非
            #   len(daily_results)-1，与剔除 NaN 后的 t_stat 口径一致
            df_pval = max(fr["n_obs"] - 1, 1)
            p_val = 2 * (1 - _t_dist.cdf(t_val, df_pval)) if fr["n_obs"] > 1 else 1.0
            fr["p_value"] = round(float(p_val), 6)
            fdr_pvals.append(p_val)
        # BH procedure
        sorted_idx = np.argsort(fdr_pvals)
        m = len(fdr_pvals)
        significant_fdr = [False] * m
        max_sig_idx = -1
        for i, idx in enumerate(sorted_idx):
            threshold = ((i + 1) / m) * fdr_q
            if fdr_pvals[idx] <= threshold:
                max_sig_idx = i
        if max_sig_idx >= 0:
            for i in range(max_sig_idx + 1):
                significant_fdr[sorted_idx[i]] = True
        for idx, f_name in enumerate(all_factors):
            factor_results[f_name]["significant_fdr"] = significant_fdr[idx]
            factor_results[f_name]["fdr_q"] = fdr_q

        beta_matrix = np.array([[dr["factor_returns"].get(f, np.nan) for f in all_factors] for dr in daily_results], dtype=np.float64)
        ordinary_se = np.array([
            factor_results[f]["std"] / math.sqrt(max(factor_results[f]["n_obs"], 1)) for f in all_factors
        ], dtype=np.float64)
        ordinary_t = np.array([factor_results[f]["t_stat"] for f in all_factors], dtype=np.float64)
        ordinary_p = np.array([factor_results[f]["p_value"] for f in all_factors], dtype=np.float64)
        max_lags_used = max(0, min(int(4 * (len(daily_results) / 100) ** (2 / 9)), len(daily_results) - 1))
        try:
            # P2-Q6-fix (Q6-M456): _newey_west_se 已改为逐列 NaN 感知（缺失期剔除）
            newey_west_se, newey_west_t = self._newey_west_se(beta_matrix, max_lags=max_lags_used)
            newey_west_pval = np.array([
                2 * (1 - _t_dist.cdf(abs(float(t_val)), max(factor_results[f]["n_obs"] - 1, 1)))
                if (not np.isnan(t_val) and factor_results[f]["n_obs"] > 1) else 1.0
                for f, t_val in zip(all_factors, newey_west_t)
            ], dtype=np.float64)
        except Exception:
            newey_west_se = ordinary_se.copy()
            newey_west_t = ordinary_t.copy()
            newey_west_pval = ordinary_p.copy()

        # P2-Q6-fix (Q6-L459): 用 Newey-West HAC t 值补充显著性口径
        #   （与 ordinary t 的 significant_5pct 并存，两口径均已文档化）
        for i, f in enumerate(all_factors):
            nw_t = float(newey_west_t[i])
            factor_results[f]["significant_5pct_nw"] = (not np.isnan(nw_t)) and abs(nw_t) > 1.96

        # 4. 因子收益率协方差
        factor_returns_df = pd.DataFrame(ts_data)
        factor_cov = factor_returns_df.cov()
        # P2-Q6-fix (Q6-M456): 无共同观测的因子对（如只在偶数日出现的 value 与只在
        #   奇数日出现的 growth）在样本期内从不同日共存，协方差无定义 → pandas cov
        #   返回 NaN。按独立性假设填 0，并在返回中可见地报告未定义协方差的对数。
        n_undefined_cov = int(np.isnan(factor_cov.values).sum() // 2)
        factor_cov = factor_cov.fillna(0.0)

        # 5. 因子协方差矩阵 + Newey-West 长期协方差 (HAC) 调整
        # P1-Q6-fix: 原实现是伪 NW（lag_cov 死计算、对全体元素加同一标量、结果从未返回），
        # 改为标准 HAC 估计:  F_nw = F + Σ_k w_k (Γ_k + Γ_kᵀ)，w_k = 1 - k/(L+1)，
        # Γ_k 为滞后 k 的自协方差矩阵，结果在返回值中输出。
        n_dates = len(daily_results)
        L = min(5, n_dates - 2) if n_dates > 2 else 0
        factor_cov_nw = factor_cov.values.tolist()
        if L > 0:
            # P2-Q6-fix (Q6-M456): 手工 HAC 的矩阵乘法在含 NaN 列时会把 NaN 传播进
            #   factor_cov_nw（因子缺失期置 NaN 后）。改为逐因子对用「两者都非 NaN」的
            #   日期子集计算（与 pandas cov 的 pairwise-complete 口径一致）。
            data_v = factor_returns_df.values.astype(np.float64)
            nw_cov = np.full((len(all_factors), len(all_factors)), np.nan, dtype=np.float64)
            for i in range(len(all_factors)):
                for j in range(i, len(all_factors)):
                    a, b = data_v[:, i], data_v[:, j]
                    valid = ~(np.isnan(a) | np.isnan(b))
                    if int(valid.sum()) < 2:
                        continue
                    ai, bj = a[valid], b[valid]
                    mi, mj = float(ai.mean()), float(bj.mean())
                    nw = float(np.mean((ai - mi) * (bj - mj)))  # lag-0 协方差
                    # V11 审计修复（Medium）: 原实现只算一个方向 gk 后强对称化——
                    # HAC 交叉自协方差 Γ_k ≠ Γ_kᵀ 时非对称项被错误对称化。
                    # 修正: 两个方向 (ai滞后bj / bj滞后ai) 的交叉协方差取平均。
                    for k in range(1, L + 1):
                        if len(ai) <= k:
                            break  # 有效样本不足 k+1，后续滞后阶同样不足
                        w = 1 - k / (L + 1)
                        gk_ab = float(np.mean((ai[k:] - mi) * (bj[:-k] - mj)))
                        gk_ba = float(np.mean((bj[k:] - mj) * (ai[:-k] - mi)))
                        nw += w * (gk_ab + gk_ba)
                    nw_cov[i, j] = nw_cov[j, i] = nw
            # 从未有有效样本的条目回退到普通协方差
            nw_cov = np.where(np.isnan(nw_cov), factor_cov.values.astype(np.float64), nw_cov)
            factor_cov_nw = nw_cov.tolist()

        return {
            "status": "ok",
            "n_dates": len(daily_results),
            "step_days": actual_step_days,
            "requested_step_days": requested_step_days,
            "date_range": f"{dates[0]} → {dates[-1]}",
            "forward_days": forward_days,
            "factor_results": factor_results,
            "se": ordinary_se,
            "t_stats": ordinary_t,
            "p_values": ordinary_p,
            "newey_west_se": newey_west_se,
            "newey_west_t": newey_west_t,
            "newey_west_pval": newey_west_pval,
            "max_lags_used": max_lags_used,
            "factor_cov": factor_cov.values.tolist() if not factor_cov.empty else [],
            "factor_cov_nw": factor_cov_nw,
            # P2-Q6-fix (Q6-M456): 样本期内无共同观测而被置 0 的协方差对数（可见降级）
            "n_undefined_cov_pairs": n_undefined_cov,
            "factor_names": all_factors,
            "alpha_mean": round(float(np.mean(ts_alphas)), 6),
            "r_squared_mean": round(float(np.mean(ts_r2)), 4),
            # P2-Q6-fix (Q6-M454): 返回完整 daily_results，与 factor_results/
            #   se/newey_west_*/factor_cov 全样本口径一致（原截断到最近 60 期导致
            #   下游逐期数据与统计量样本不一致）
            "daily_results": daily_results,
            # P2-Q6-fix (Q6-M457): 可见地报告因异常被跳过的日期数
            "n_dates_skipped": n_skipped,
        }

    # ── 因子收益率曲线 ──────────────────────────────────

    def factor_cumulative_returns(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2026-04-01",
        forward_days: int = 1,
    ) -> pd.DataFrame:
        """Get cumulative factor returns for charting."""
        result = self.run(symbols, start_date=start_date, forward_days=forward_days)
        if result.get("status") != "ok":
            return pd.DataFrame()

        daily = result.get("daily_results", [])
        if not daily:
            return pd.DataFrame()

        step = max(int(result.get("step_days", 1)), 1)
        dates = [d["date"] for d in daily]
        # P2-Q6-fix (Q6-M456): 因子取所有截面并集（与 run() 统计口径一致），
        #   缺失期置 0（该因子当期为 0 收益，不改变累计曲线）
        all_factors = sorted(set().union(*(d["factor_returns"].keys() for d in daily)))
        data = {f: [] for f in all_factors}
        for d in daily:
            for f in all_factors:
                data[f].append(d["factor_returns"].get(f, 0.0))

        # P2-Q6-fix (Q6-M455): 用全量 daily_results（原截断 60 期只画尾部）；每期因子
        #   收益 r 视为覆盖 step 个交易日的区间收益，折算为等量日收益 r_d=(1+r)^(1/step)-1
        #   后累乘，避免简单 cumsum 忽略截面采样间隔。
        r = pd.DataFrame(data, index=dates)
        base = np.maximum(1.0 + r.values, 1e-8)  # 防御极端截面回归系数
        r_daily = np.power(base, 1.0 / step) - 1.0
        cum_df = pd.DataFrame(
            np.cumprod(1.0 + r_daily, axis=0) - 1.0,
            index=dates,
            columns=r.columns,
        )
        return cum_df


# ── Singleton ──

_fm: FamaMacBeth | None = None


def get_fama_macbeth() -> FamaMacBeth:
    global _fm
    if _fm is None:
        _fm = FamaMacBeth()
    return _fm


# ── CLI ──

if __name__ == "__main__":
    fm = get_fama_macbeth()
    symbols = ["600519", "000858", "002714", "601899", "002594", "300750",
               "600036", "601318", "000333", "600276", "000568", "002415",
               "000001", "601166", "600900", "600887", "601398", "601939",
               "601288", "601988"]

    r = fm.run(symbols=symbols, start_date="2026-05-01", step_days=10)
    if r.get("status") == "ok":
        print(f"Fama-MacBeth: {r['n_dates']} 个截面, 日期范围: {r['date_range']}")
        print(f"Alpha均值: {r['alpha_mean']}, R²均值: {r['r_squared_mean']}")
        print(f"\n{'因子':>25} {'收益率':>10} {'t-stat':>8} {'IR年':>8} {'自相关':>8} {'累计%':>10} {'显著?':>6}")
        print("-" * 80)
        for name, fr in sorted(r["factor_results"].items(), key=lambda x: abs(x[1]["t_stat"]), reverse=True):
            sig = "✅" if fr["significant_5pct"] else " "
            print(f"{name:>25} {fr['mean']:>+10.6f} {fr['t_stat']:>+8.2f} {fr['sharpe_annualized']:>8.4f} {fr['autocorr']:>+8.4f} {fr['cum_return_pct']:>+10.2f} {sig:>6}")
