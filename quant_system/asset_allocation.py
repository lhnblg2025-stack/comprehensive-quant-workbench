"""
asset_allocation.py — V8 资产配置引擎

整合因子模型 + ML 信号 + 组合优化器，生成端到端资产配置建议。

功能:
  1. 因子驱动的资产配置: 基于因子暴露 + 风险模型构建 MV/ERC 组合
  2. ML 信号增强: 叠加 ML 预测作为 Black-Litterman 主观观点
  3. 行业/风格中性配置: 支持行业中性化和多约束优化
  4. 配置报告: 权重分布 / 风险贡献 / 压力测试
"""

from __future__ import annotations

import logging
import math
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

# ── 子模块 ──
from quant_system.factor_model import get_model as get_factor_model
from quant_system.data_store import get_store


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RISK_FREE = 0.02       # 2% 无风险利率（年化）
DEFAULT_SECTOR_CEILING = 0.30  # 单个行业上限
DEFAULT_ASSET_CEILING = 0.15   # 单个标的权重上限
TRADING_DAYS = 242             # A 股年化交易日数（本模块收益均为日频）
ASSET_CLASS_LABELS = {
    "large_cap": "大盘价值",
    "growth": "成长",
    "defensive": "防御",
    "cyclical": "周期",
    "commodity": "商品",
}


# ── 核心引擎 ────────────────────────────────────────

class AssetAllocator:
    """端到端资产配置引擎"""

    def __init__(self, risk_free: float = DEFAULT_RISK_FREE):
        self.risk_free = risk_free
        self._factor_model = get_factor_model()
        # P2-Q16-fix (L135): 行业数据源后台加载状态（类级去重 + 缓存）
        self._sector_loading = False
        self._sector_map_cache: dict[str, str] = {}

    def _fallback_historical_covariance(self, symbols: list[str], reason: str) -> pd.DataFrame:
        logger.warning("factor_covariance fallback to historical covariance: %s", reason)
        return self._historical_covariance(symbols)

    # ── 预期收益估计 ──────────────────────────────────

    def expected_returns_from_ml(
        self,
        symbols: list[str],
        ml_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Combine factor z-score and optional ML scores into expected returns.

        If ml_scores provided:  returns = 0.7 * factor_z_score + 0.3 * ml_score
        Otherwise:              returns = factor_z_score * 0.01 (scaled to daily)
        """
        # Get factor exposure
        today = datetime.now().strftime("%Y-%m-%d")
        factor_df = self._factor_model.compute_factors(today, symbols=symbols[:100])
        if factor_df.empty:
            # Fallback: equal expected returns
            return {s: DEFAULT_RISK_FREE / 242 for s in symbols}

        # Composite z-score: average of momentum + quality (strongest predictors)
        composite: dict[str, float] = {}
        for sym in symbols:
            if sym not in factor_df.index:
                composite[sym] = 0.0
                continue
            row = factor_df.loc[sym]
            mom_scores = [row.get(f, 0) for f in
                          ["mom_1m", "mom_3m", "mom_6m", "excess_ret_20d"]]
            qual_scores = [row.get(f, 0) for f in
                           ["roe", "roa"]]
            valid_mom = [s for s in mom_scores if not np.isnan(s) and abs(s) < 10]
            valid_qual = [s for s in qual_scores if not np.isnan(s) and abs(s) < 10]
            z = (np.mean(valid_mom) if valid_mom else 0) * 0.6 + \
                (np.mean(valid_qual) if valid_qual else 0) * 0.4
            composite[sym] = float(z)

        # Scale to daily return estimates (rough mapping: z=1 => ~0.3% daily excess)
        daily_ret = {s: v * 0.003 + DEFAULT_RISK_FREE / 242 for s, v in composite.items()}

        # Blend with ML signals if available
        if ml_scores:
            for sym in symbols:
                ml = ml_scores.get(sym, 0)
                daily_ret[sym] = daily_ret.get(sym, 0) * 0.7 + float(ml) * 0.3 * 0.003

        return daily_ret

    # ── 协方差矩阵估计 ────────────────────────────────

    def _recent_trading_dates(self, symbols: list[str], n_days: int = 60) -> list[str]:
        """从行情库取最近 n_days 个交易日（YYYY-MM-DD，升序）。"""
        store = get_store()
        today = datetime.now().strftime("%Y-%m-%d")
        for sym in symbols:
            try:
                df = store.get(sym, days=n_days + 20)
                if df is None or df.empty or "date" not in df.columns:
                    continue
                dates = sorted({str(x)[:10] for x in df["date"].tolist()})
                dates = [d for d in dates if d <= today]
                if len(dates) >= 10:
                    return dates[-n_days:]
            except Exception as e:
                logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                continue
        return []

    def _returns_on_dates(self, symbols: list[str], dates: list[str], min_symbols: int = 5) -> pd.DataFrame:
        """构建 index=dates、columns=symbols 的日收益率矩阵（pct_chg，百分数）。

        P2-Q16-fix (M126): min_symbols 可调。默认 5（因子协方差截面需要足够宽度），
        _historical_covariance 等对小组合的对齐路径传入 2。
        """
        store = get_store()
        date_set = set(dates)
        cols: dict[str, pd.Series] = {}
        for sym in symbols:
            try:
                df = store.get(sym, days=len(dates) + 40)
                if df is None or df.empty or "pct_chg" not in df.columns:
                    continue
                s = pd.Series(df["pct_chg"].values, index=df["date"].astype(str).str[:10]).astype(float)
                s = s[~s.index.duplicated(keep="last")]
                s = s[s.index.isin(date_set)]
                if len(s) >= 10:
                    cols[sym] = s
            except Exception as e:
                logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                continue
        if len(cols) < min_symbols:
            raise ValueError(f"insufficient return history for factor covariance: {len(cols)} symbols")
        return pd.DataFrame(cols).reindex(dates).fillna(0.0)

    def _estimate_factor_cov(self, symbols: list[str], n_days: int = 40, lags: int = 5) -> pd.DataFrame:
        """P1-Q16-fix: 因子协方差由历史因子收益的时序估计（Newey-West 在时间维）。

        V5.4 的 risk_model._newey_west_cov() 把 compute_factors 的单日截面 z-score
        （一行一股）当作时间序列计算——得到的是「截面协方差」而非「因子收益协方差」，
        Newey-West 的时间维实为股票数，F·C·F' 无量纲且与 S（百分数²）混杂。
        这里按 Barra 标准做法：对每个历史交易日 d 做截面回归 r_t ~ F_t（含截距），
        回归系数即当日因子收益；因子协方差 = 因子收益时序的 Newey-West HAC
        （T=交易日数）。截面暴露仅用于 F 载荷矩阵，不再被误当作时间序列。
        失败（历史不足/退化）时由调用方降级到历史协方差并可见告警。
        """
        dates = self._recent_trading_dates(symbols, n_days=n_days)
        if len(dates) < 15:
            raise ValueError(f"insufficient trading dates for factor covariance: {len(dates)} days")
        ret_df = self._returns_on_dates(symbols, dates)
        syms = ret_df.columns.tolist()

        factor_returns: list[np.ndarray] = []
        factor_names: list[str] | None = None
        for d in dates:
            fdf = self._factor_model.compute_factors(d, symbols=syms)
            if fdf is None or fdf.empty or len(fdf) < 5:
                continue
            if factor_names is None:
                factor_names = list(fdf.columns)
            if not factor_names:
                continue
            F = fdf.reindex(index=syms).reindex(columns=factor_names).fillna(0.0)
            r = ret_df.loc[d, syms].values.astype(np.float64)
            A = np.column_stack([np.ones(len(syms)), F.values.astype(np.float64)])
            try:
                coef, *_ = np.linalg.lstsq(A, r, rcond=None)
            except Exception as e:
                logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                continue
            factor_returns.append(np.asarray(coef[1:], dtype=np.float64))

        if len(factor_returns) < 15:
            raise ValueError(f"insufficient factor-return observations: {len(factor_returns)}")
        FR = np.vstack(factor_returns)          # T x k
        T, k = FR.shape
        dm = FR - FR.mean(axis=0)
        if k == 1:
            C = np.atleast_2d(float((dm * dm).sum() / (T - 1)))
        else:
            C = (dm.T @ dm) / (T - 1)
            for lag in range(1, lags + 1):
                if T <= lag:
                    continue
                w = 1.0 - lag / (lags + 1)
                g = (dm[lag:].T @ dm[:-lag]) / (T - 1)
                C = C + w * (g + g.T)
            # 半正定投影（Newey-West 相加项可能破坏 PSD）
            evals, evecs = np.linalg.eigh((C + C.T) / 2.0)
            evals = np.maximum(evals, 1e-8)
            C = evecs @ np.diag(evals) @ evecs.T
        C_df = pd.DataFrame(C, index=factor_names, columns=factor_names)
        if not np.all(np.isfinite(C_df.values)) or float(np.max(np.diag(C_df))) <= 1e-12:
            raise ValueError("factor covariance degenerate after PSD projection")
        return C_df

    def factor_covariance(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Build covariance matrix from factor model."""
        today = datetime.now().strftime("%Y-%m-%d")
        rm = self._factor_model.risk_model(today, symbols=symbols[:100])
        if rm.get("status") == "insufficient_data":
            return self._fallback_historical_covariance(symbols, "risk_model insufficient_data")

        # Build full covariance: factor_exposure @ factor_cov @ factor_exposure^T + specific_risk
        exposure = rm.get("factor_exposure", None)
        factor_cov = rm.get("factor_cov", None)
        specific = rm.get("specific_risk", None)

        if exposure is None or factor_cov is None or specific is None:
            return self._fallback_historical_covariance(symbols, "risk_model missing exposure/covariance/specific_risk")

        try:
            # P1-Q16-fix: C 不再直接采用 risk_model 的 factor_cov（其由单日截面
            # z-score 套 Newey-West 得到，非因子收益协方差，单位与 S 混杂）。
            # 改为从历史因子收益时序估计因子协方差；factor_cov 仅保留存在性校验。
            C = self._estimate_factor_cov(symbols).values.astype(np.float64)  # (k, k)

            # Build F matrix from exposure dict
            if isinstance(exposure, dict):
                syms = sorted([s for s in symbols if s in exposure])
                if len(syms) < 2:
                    return self._fallback_historical_covariance(symbols, "fewer than 2 assets with factor exposure")
                factor_keys = list(next(iter(exposure.values())).keys())
                F = np.array([[exposure[s].get(k, 0.0) for k in factor_keys] for s in syms], dtype=np.float64)
            else:
                syms = symbols
                F = np.array(exposure, dtype=np.float64)

            n, k = F.shape
            if k != C.shape[0]:
                return self._fallback_historical_covariance(symbols, "factor exposure shape does not match factor covariance")

            # Specific risk diagonal
            if isinstance(specific, (list, np.ndarray)):
                S_vals = np.array(specific[:n] if len(specific) >= n else [specific[0]]*n, dtype=np.float64)
            else:
                S_vals = np.full(n, float(specific))
            S = np.diag(S_vals ** 2)  # vol -> variance

            # Full covariance: F @ C @ F^T + S
            full_cov = F @ C @ F.T + S

            # Apply shrinkage to stabilize: blend with identity * mean(diag)
            diag_mean = float(np.mean(np.diag(full_cov)))
            shrinkage = 0.15  # 15% weight on identity
            full_cov = (1 - shrinkage) * full_cov + shrinkage * np.eye(n) * diag_mean
            diag = np.diag(full_cov)
            if (
                not np.all(np.isfinite(full_cov))
                or np.nanmax(np.abs(full_cov)) <= 1e-12
                or (len(diag) > 1 and np.allclose(diag, diag[0], rtol=1e-5, atol=1e-10))
            ):
                return self._fallback_historical_covariance(
                    syms,
                    "factor covariance is degenerate: non-finite, near-zero, or equal diagonal variances",
                )
            return pd.DataFrame(full_cov, index=syms, columns=syms)
        except Exception as exc:
            return self._fallback_historical_covariance(symbols, f"factor covariance build failed: {exc}")

    def _historical_covariance(
        self,
        symbols: list[str],
        days: int = 60,
    ) -> pd.DataFrame:
        """Quick historical covariance from DataStore.

        P2-Q16-fix (M126): 优先按日期对齐成面板再估计协方差。旧实现对各股独立取
        末 days 行再 column_stack——停牌/缺失交易日错位，联合分布失真（与 compute_var
        同源，程度较轻）。对齐路径数据不足时可见降级回退旧逻辑。
        """
        store = get_store()
        cols: list[str] = []
        arr: np.ndarray | None = None
        try:
            dates = self._recent_trading_dates(symbols, n_days=days)
            # P2-Q16-fix (M126): 对齐面板对小组合也适用（min_symbols=2），
            # 不再因为 <5 只标的整体回退到未对齐路径。
            ret_df = self._returns_on_dates(symbols, dates, min_symbols=2)
            ret_df = ret_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if len(ret_df.columns) >= 2 and len(ret_df) >= 5:
                arr = ret_df.values.T  # N x T（行=股票，列=交易日）
                cols = ret_df.columns.tolist()
        except Exception as exc:
            logger.warning("_historical_covariance aligned panel failed (%s), fallback to per-symbol tail", exc)

        if arr is None:
            # 兜底：逐股末 days 行（日期可能错位，仅作退化路径）
            rets: dict[str, np.ndarray] = {}
            for sym in symbols:
                try:
                    df = store.get(sym, days=days + 5)
                    if df is None or df.empty or "pct_chg" not in df.columns:
                        continue
                    r = df["pct_chg"].values[-days:]
                    if len(r) >= 2 and np.all(np.isfinite(r)):
                        rets[sym] = r
                except Exception as e:
                    logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                    continue
            if len(rets) < 2:
                return pd.DataFrame(1.0, index=symbols[:3], columns=symbols[:3])
            cols = list(rets.keys())
            arr = np.column_stack(list(rets.values())).T  # N x T（与对齐路径同方向）

        # np.cov 约定行=变量（股票），列=观测（交易日）→ 输出 N x N
        cov = np.cov(arr)
        if not np.all(np.isfinite(cov)):
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        # Shrinkage to stabilize
        n = cov.shape[0]
        diag_mean = float(np.mean(np.diag(cov)))
        shrinkage = 0.20
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(n) * diag_mean
        return pd.DataFrame(cov, index=cols, columns=cols)

    # ── 行业分类 ──────────────────────────────────────

    def get_sector_map(self, symbols: list[str]) -> dict[str, str]:
        """Get basic sector classification for symbols (CSI classification)."""
        # P2-Q16-fix (L135): 优先接 sector_rotation.get_stock_industry_map() 行业数据源
        # （覆盖全 A 股）。旧静态表仅覆盖 ~35 只，其余全归"其他"，行业上限约束形同虚设。
        # 数据源冷启动为全市场网络拉取（可能数十秒），不能阻塞 allocate 热路径：
        # 用类级去重 + 1s 超时的后台线程加载，超时/失败可见降级回退静态表。
        # CSI 行业一级分类 (粗略，仅离线回退)
        sector_of = {
            "600519": "消费", "000858": "消费", "000568": "消费",
            "002714": "消费", "600887": "消费", "600809": "消费",
            "601899": "周期", "600585": "周期", "601088": "周期",
            "600028": "周期", "601857": "周期", "600019": "周期",
            "601225": "周期", "600438": "周期",
            "300750": "制造", "002594": "制造", "000651": "制造",
            "600031": "制造", "002475": "制造", "300274": "制造",
            "600036": "金融", "601318": "金融", "601398": "金融",
            "601939": "金融", "601288": "金融", "601988": "金融",
            "601166": "金融", "600030": "金融", "601211": "金融",
            "600276": "医药", "000538": "医药", "300760": "医药",
            "600900": "公用", "601668": "建筑", "000002": "地产",
            "300059": "金融", "300124": "制造",
            "000333": "制造", "000725": "制造", "002415": "制造",
            "002304": "消费", "600690": "制造",
            "600309": "周期",
        }

        # 快速路径：会话内已有行业映射
        if self._sector_map_cache:
            return {s: self._sector_map_cache.get(s, sector_of.get(s, "其他")) for s in symbols}
        # 快速路径：sector_rotation 模块缓存已热（本进程其他调用已拉取过）
        try:
            import quant_system.sector_rotation as _sr
            item = getattr(_sr, "_CACHE", {}).get("stock_sw_map")
            if item and isinstance(item, dict) and item.get("data"):
                cached = item["data"]
                self._sector_map_cache = {str(k): str(v) for k, v in cached.items()}
                return {s: self._sector_map_cache.get(s, sector_of.get(s, "其他")) for s in symbols}
        except Exception as e:
            logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)

        # 慢路径：后台加载行业数据源（类级去重，不重复拉全市场），首次最多等 1s
        if not self._sector_loading:
            self._sector_loading = True

            def _load():
                try:
                    from quant_system.sector_rotation import get_stock_industry_map
                    ind = get_stock_industry_map()
                    if ind:
                        self._sector_map_cache = {str(k): str(v) for k, v in ind.items()}
                except Exception as e:
                    logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                finally:
                    self._sector_loading = False

            threading.Thread(target=_load, daemon=True).start()
            t_wait = _time.time()
            while self._sector_loading and (_time.time() - t_wait) < 1.0:
                _time.sleep(0.05)
            if self._sector_loading:
                logger.warning(
                    "get_sector_map: industry source cold-load >1s，本次回退静态表（后台继续加载，后续调用生效）"
                )

        if self._sector_map_cache:
            return {s: self._sector_map_cache.get(s, sector_of.get(s, "其他")) for s in symbols}
        return {s: sector_of.get(s, "其他") for s in symbols}

    # ── 配置 ──────────────────────────────────────────

    def allocate(
        self,
        symbols: list[str],
        objective: str = "max_sharpe",
        ml_scores: dict[str, float] | None = None,
        risk_target: float | None = None,
        sector_ceiling: float = DEFAULT_SECTOR_CEILING,
        asset_ceiling: float = DEFAULT_ASSET_CEILING,
    ) -> dict[str, Any]:
        """Run full asset allocation pipeline.

        Returns dict with:
          - weights: {symbol: weight}
          - metrics: expected_return, volatility, sharpe
          - risk_breakdown: risk contribution per symbol
          - sector_exposure: {sector: weight}
          - concentration: top1/top5/HHI
        """
        t0 = _time.time()
        result: dict[str, Any] = {
            "n_symbols": len(symbols),
            "objective": objective,
            "status": "running",
        }

        # 1. Expected returns
        er_dict = self.expected_returns_from_ml(symbols, ml_scores)
        er_series = pd.Series({s: er_dict.get(s, DEFAULT_RISK_FREE / 242) for s in symbols})

        # 2. Covariance
        cov = self.factor_covariance(symbols)
        common = list(er_series.index.intersection(cov.index).intersection(cov.columns))
        if len(common) < 4:
            result["status"] = "insufficient_common_assets"
            result["elapsed"] = round(_time.time() - t0, 2)
            return result

        er_series = er_series[common]
        cov = cov.loc[common, common]

        # 3. Sector map
        sector_map = self.get_sector_map(common)

        # 4. Run optimizer
        from quant_system.portfolio_optimizer import (
            mean_variance_optimize,
            equal_risk_contribution,
            
        )

        if objective == "erc":
            # P1-Q16 遗留修复: equal_risk_contribution(cov, ...) 把协方差传给了
            # expected_returns 位置参数且未传 cov_matrix → TypeError。改为正确传参。
            opt_result = equal_risk_contribution(
                er_series,
                cov,
                sector_map=sector_map,
                sector_ceiling=sector_ceiling,
                asset_ceiling=asset_ceiling,
            )
        elif objective == "min_vol":
            opt_result = mean_variance_optimize(
                er_series,
                cov,
                objective="min_vol",
                sector_map=sector_map,
                sector_ceiling=sector_ceiling,
                asset_ceiling=asset_ceiling,
            )
        elif objective == "target_return":
            # P2-Q16-fix (M125): objective=target_return 但 risk_target 缺失时，
            # 旧代码静默落入 else 的 max_sharpe 分支（还带着日频 rf），用户的目标
            # 被悄悄替换。改为显式报错返回（可见失败，不静默降级）。
            if risk_target is None:
                result["status"] = "invalid_params"
                result["message"] = "objective=target_return 必须提供 risk_target（目标年化收益）"
                result["elapsed"] = round(_time.time() - t0, 2)
                return result
            opt_result = mean_variance_optimize(
                er_series,
                cov,
                target_return=risk_target,
                objective=objective,
                sector_map=sector_map,
                sector_ceiling=sector_ceiling,
                asset_ceiling=asset_ceiling,
            )
        else:  # max_sharpe (default)
            # Q16 修复：expected_returns_from_ml 产出的是**日频**收益（~0.003），
            # 而 self.risk_free 是年化利率（0.02）。若直接把年化 rf 传入，
            # 超额收益恒为负 → 最小化 -(ret-rf)/vol 等价于最大化波动率
            # （max_sharpe 方向反转）。这里把 rf 换算成日频（rf/242）。
            rf_daily = self.risk_free / TRADING_DAYS
            opt_result = mean_variance_optimize(
                er_series,
                cov,
                objective="max_sharpe",
                rf=rf_daily,
                sector_map=sector_map,
                sector_ceiling=sector_ceiling,
                asset_ceiling=asset_ceiling,
            )
            # Detect degenerate solution: all weights near ceiling -> switch to min_vol
            w_vals = list(opt_result.get("weights", {}).values())
            if w_vals:
                unique_w = len(set(round(v, 4) for v in w_vals))
                near_ceiling = sum(1 for v in w_vals if v >= asset_ceiling * 0.95)
                if near_ceiling >= len(w_vals) * 0.6 or unique_w <= 2:
                    minvol_result = mean_variance_optimize(
                        er_series,
                        cov,
                        objective="min_vol",
                        sector_map=sector_map,
                        sector_ceiling=sector_ceiling,
                        asset_ceiling=asset_ceiling,
                    )
                    if minvol_result.get("weights"):
                        opt_result = minvol_result
                        opt_result["_fallback_from"] = "max_sharpe_degenerate_to_min_vol"

        # 5. Extract weights
        weights = opt_result.get("weights", {})
        if not weights:
            # Fallback: equal weight
            weights = {s: 1.0 / len(common) for s in common[:50]}
        elif isinstance(weights, dict):
            pass
        elif isinstance(weights, (list, np.ndarray)):
            weights = {common[i]: float(w) for i, w in enumerate(weights) if i < len(common)}

        # 6. Metrics
        metrics = opt_result.get("metrics", {})
        concentration = opt_result.get("weight_concentration", {})

        # 7. Risk breakdown (marginal risk contribution)
        risk_breakdown = self._risk_contribution(weights, cov)

        # 8. Sector exposure
        sector_exposure: dict[str, float] = {}
        for sym, w in weights.items():
            sec = sector_map.get(sym, "其他")
            sector_exposure[sec] = sector_exposure.get(sec, 0) + w

        # P1-Q16-fix: 透传优化器的可行性告警（如 asset_ceiling 被放宽），不再静默吞掉。
        # 真正的不可行（infeasible）升级为顶层 status；success_with_warning 保持顶层
        # success（向后兼容 allocation_report 等调用方），但告警明细通过 warnings 显式透出。
        opt_status = opt_result.get("status", "success")
        opt_warnings = list(opt_result.get("warnings", []))
        if opt_result.get("_fallback_from"):
            opt_warnings.append(opt_result["_fallback_from"])

        result_status = "infeasible" if opt_status == "infeasible" else "success"
        result.update({
            "weights": {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
            "metrics": metrics,
            "risk_breakdown": risk_breakdown,
            "sector_exposure": sector_exposure,
            "concentration": concentration,
            "status": result_status,
            "elapsed": round(_time.time() - t0, 2),
        })
        if opt_warnings:
            result["warnings"] = opt_warnings
        return result

    # ── 风险分解 ──────────────────────────────────────

    @staticmethod
    def _risk_contribution(
        weights: dict[str, float],
        cov: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute marginal and component risk contribution."""
        try:
            syms = list(weights.keys())
            w = np.array([weights[s] for s in syms])
            # Align covariance
            common = [s for s in syms if s in cov.index and s in cov.columns]
            if len(common) < 2:
                return {}
            w = np.array([weights[s] for s in common])
            C = cov.loc[common, common].values
            port_var = w @ C @ w
            if port_var < 1e-12:
                return {}
            port_vol = math.sqrt(port_var)
            # Marginal risk contribution
            mrc = C @ w / port_vol
            # Component risk contribution
            crc = w * mrc / port_vol
            return {s: round(float(crc[i]), 4) for i, s in enumerate(common)}
        except Exception:
            return {}

    # ── 压力测试 ──────────────────────────────────────

    def _estimate_stock_betas(
        self,
        symbols: list[str],
        lookback: int = 120,
    ) -> dict[str, float]:
        """估计个股相对沪深300(000300)的 beta（压力测试因子载荷）。

        P2-Q16-fix (M124 配套): 压力测试按个股 beta 加权冲击，而非把同一 factor
        乘到全部权重和。数据不足的标的回退 beta=1.0（可见，计入结果 note）。
        """
        betas: dict[str, float] = {}
        store = get_store()
        try:
            mkt_df = store.get("000300", days=lookback + 10)
            if mkt_df is None or mkt_df.empty or "pct_chg" not in mkt_df.columns:
                return betas
            mkt_s = pd.Series(
                mkt_df["pct_chg"].astype(float).values,
                index=mkt_df["date"].astype(str).str[:10],
            )
            mkt_s = mkt_s[~mkt_s.index.duplicated(keep="last")]
            for sym in symbols:
                try:
                    df = store.get(sym, days=lookback + 10)
                    if df is None or df.empty or "pct_chg" not in df.columns:
                        continue
                    s = pd.Series(
                        df["pct_chg"].astype(float).values,
                        index=df["date"].astype(str).str[:10],
                    )
                    s = s[~s.index.duplicated(keep="last")]
                    common = s.index.intersection(mkt_s.index)
                    if len(common) < 20:
                        continue
                    r = s[common].values
                    m = mkt_s[common].values
                    var_m = float(np.var(m))
                    if var_m <= 1e-12:
                        continue
                    betas[sym] = float(np.cov(r, m)[0, 1]) / var_m
                except Exception as e:
                    logger.error(f"[asset_allocation] 操作失败: {e}", exc_info=True)
                    continue
        except Exception:
            return betas
        return betas

    def _historical_cvar(
        self,
        weights: dict[str, float],
        confidence: float = 0.95,
    ) -> float | None:
        """组合历史日度 CVaR(confidence)，返回小数收益率（正数=损失）。

        用于 vol_mult 场景的尾部重估：波动放大 N 倍时，历史尾部损失近似线性放大。
        数据不足返回 None（调用方跳过该字段）。
        """
        store = get_store()
        try:
            dates = self._recent_trading_dates(list(weights.keys()), n_days=60)
            if len(dates) < 20:
                return None
            ret_df = self._returns_on_dates(list(weights.keys()), dates)
            if ret_df.empty or ret_df.shape[1] < 2:
                return None
            w = np.array([weights.get(s, 0.0) for s in ret_df.columns])
            port = ret_df.fillna(0.0).values @ w  # 百分数收益
            losses = -port  # 正数=亏损
            var = np.quantile(losses, confidence)
            tail = losses[losses >= var]
            return float(tail.mean() if len(tail) else var) / 100.0
        except Exception:
            return None

    def stress_test(
        self,
        weights: dict[str, float],
        scenarios: list[dict[str, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Simple stress test scenarios.

        P2-Q16-fix (M124): 冲击按个股 beta 加权（组合 beta=Σwᵢβᵢ），不再把同一
        factor 乘到全部权重和≈factor（忽略个股 beta/行业差异）；vol_mult 参与尾部
        重估（线性放大历史日 CVaR），不再仅展示。
        """
        if scenarios is None:
            scenarios = [
                {"name": "大盘跌-5%", "factor": -0.05},
                {"name": "大盘跌-10%", "factor": -0.10},
                {"name": "大盘涨+5%", "factor": 0.05},
                {"name": "波动骤升", "factor": -0.03, "vol_mult": 2.0},
            ]

        betas = self._estimate_stock_betas(list(weights.keys()))
        missing_beta = [s for s in weights if s not in betas]
        # 组合 beta = Σ wᵢ βᵢ（缺 beta 的标的按 1.0 计，可见 note）
        port_beta = sum(w * betas.get(sym, 1.0) for sym, w in weights.items())

        results = []
        for sc in scenarios:
            factor = sc.get("factor", 0)
            vol_mult = sc.get("vol_mult", 1.0)
            weighted_ret = port_beta * factor
            item: dict[str, Any] = {
                "scenario": sc["name"],
                "impact_pct": round(weighted_ret * 100, 2),
                "vol_multiplier": vol_mult,
                "portfolio_beta": round(port_beta, 3),
            }
            # vol_mult 用于重估尾部：波动放大 N 倍 → 历史日 CVaR 近似线性放大
            if vol_mult != 1.0:
                hist_cvar = self._historical_cvar(weights)
                if hist_cvar is not None:
                    item["tail_impact_pct"] = round(hist_cvar * vol_mult * 100, 2)
                    item["tail_impact_note"] = "vol_mult 线性放大历史日 CVaR(95)（近似）"
            if missing_beta:
                item["beta_note"] = f"{len(missing_beta)} 只标的无 beta 数据按 1.0 计"
            results.append(item)
        return results

    # ── 配置建议报告 ──────────────────────────────────

    def allocation_report(
        self,
        symbols: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a complete allocation report."""
        result = self.allocate(symbols, **kwargs)
        if result.get("status") != "success":
            return result

        # Add stress test
        weights = result.get("weights", {})
        result["stress_test"] = self.stress_test(weights)
        return result


# ── 单例 ─────────────────────────────────────────────

_allocator: AssetAllocator | None = None


def get_allocator() -> AssetAllocator:
    global _allocator
    if _allocator is None:
        _allocator = AssetAllocator()
    return _allocator


# ── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    symbols = ["600519", "000858", "002714", "601899", "002594",
               "300750", "600036", "601318", "000333", "600276",
               "000568", "002415", "000001", "601166", "600900"]
    alloc = get_allocator()
    result = alloc.allocation_report(symbols, objective="max_sharpe")
    if result.get("status") == "success":
        print(f"✅ 资产配置完成 ({result['elapsed']}s)")
        print(f"\n权重 Top 5:")
        for sym, w in list(result["weights"].items())[:5]:
            print(f"  {sym}: {w:.2%}")
        print(f"\n指标: {result['metrics']}")
        print(f"行业分布: {result['sector_exposure']}")
        print(f"集中度: {result['concentration']}")
    else:
        print(f"⚠️ {result.get('status')}")
