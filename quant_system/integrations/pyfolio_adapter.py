"""
pyfolio_adapter — 组合绩效归因适配层 (V5)

将 pyfolio-reloaded 集成到绩效分析流程：
  - 完整绩效面板（Sharpe/回撤/Calmar等）
  - 因子归因（Alpha/Beta/因子暴露）
  - 回撤深度分析（起止日期/恢复时间）
  - 滚动指标
  - 基准对比（跟踪误差/信息比）

降级策略: pyfolio不可用时手算，保证功能不中断。
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*zipline.assets.*")

import numpy as np
import pandas as pd

# P2-Q26-fix: numpy 2.0 移除了 np.NINF/np.PINF，pyfolio 0.9.9 的 sortino_ratio /
# downside_risk 在函数体内引用它们 → 每轮必抛 AttributeError 并静默降级手算，
# docstring 承诺的 calmar_ratio/sortino_ratio/downside_risk/stability 永不产出。
# 在此补齐兼容别名（pyfolio-reloaded 已修复，无需升级依赖）。
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

try:
    _HAS_PYFOLIO = True
except ImportError:
    _HAS_PYFOLIO = False


class PyfolioAnalyzer:
    """组合绩效分析（封装pyfolio）。"""

    def __init__(self) -> None:
        self.available = _HAS_PYFOLIO

    def tear_sheet(self, returns: pd.Series,
                   benchmark: pd.Series | None = None) -> dict:
        """组合绩效完整分析。

        Args:
            returns: 组合日收益率序列
            benchmark: 基准日收益率序列（可选）

        Returns:
            {sharpe, max_drawdown, calmar_ratio, annual_return, annual_vol,
             downside_risk, sortino_ratio, stability, var_95, cvar_95}
        """
        if returns.empty:
            return {"error": "empty returns"}

        ret = returns.dropna().values

        try:
            if self.available:
                import pyfolio.timeseries as ts

                sharpe = ts.sharpe_ratio(returns)
                max_dd = ts.max_drawdown(returns)
                calmar = ts.calmar_ratio(returns)
                ann_ret = ts.annual_return(returns)
                ann_vol = ts.annual_volatility(returns)
                sortino = ts.sortino_ratio(returns)
                downside = ts.downside_risk(returns)
                stability = ts.stability_of_timeseries(returns)

                # VaR/CVaR
                var_95 = float(np.percentile(ret, 5))
                cvar_95 = float(ret[ret <= var_95].mean()) if np.any(ret <= var_95) else var_95

                return {
                    "sharpe": round(float(sharpe), 2) if not np.isnan(sharpe) else 0,
                    "max_drawdown": round(float(max_dd), 4) if not np.isnan(max_dd) else 0,
                    "calmar_ratio": round(float(calmar), 2) if not np.isnan(calmar) else 0,
                    "annual_return": round(float(ann_ret), 4) if not np.isnan(ann_ret) else 0,
                    "annual_volatility": round(float(ann_vol), 4) if not np.isnan(ann_vol) else 0,
                    "sortino_ratio": round(float(sortino), 2) if not np.isnan(sortino) else 0,
                    "downside_risk": round(float(downside), 4) if not np.isnan(downside) else 0,
                    "stability": round(float(stability), 4) if not np.isnan(stability) else 0,
                    "var_95": round(var_95, 4),
                    "cvar_95": round(cvar_95, 4),
                    "count": int(len(ret)),
                }
        except Exception as e:
            # P2-Q26-fix: pyfolio 降级不再静默——记录日志并在结果中带 note 可见
            import logging
            logging.getLogger(__name__).warning(f"pyfolio tear_sheet degraded to manual calc: {e}")
            result = self._manual_tear_sheet(ret)
            result["note"] = f"pyfolio degraded to manual calc: {e}"
            return result

        # 降级: 手算（pyfolio 未安装）
        result = self._manual_tear_sheet(ret)
        result.setdefault("note", "pyfolio not available, manual calc")
        return result

    def _manual_tear_sheet(self, ret: np.ndarray) -> dict:
        """降级手算绩效指标。"""
        n = len(ret)
        if n < 2:
            return {}
        ann_ret = float(np.mean(ret)) * 252
        ann_vol = float(np.std(ret, ddof=1)) * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-8)
        cum = np.cumprod(1 + ret)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(np.min(dd))
        var_95 = float(np.percentile(ret, 5))
        cvar_95 = float(ret[ret <= var_95].mean()) if np.any(ret <= var_95) else var_95
        win_rate = float(np.mean(ret > 0))

        return {
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "annual_return": round(ann_ret, 4),
            "annual_volatility": round(ann_vol, 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "win_rate": round(win_rate, 4),
            "note": "pyfolio not available, manual calc",
        }

    def factor_decomposition(self, returns: pd.Series,
                              factor_returns: pd.DataFrame) -> dict:
        """因子归因: OLS求Alpha/Beta/因子暴露。"""
        ret = returns.dropna()
        factor = factor_returns.dropna()

        common_idx = ret.index.intersection(factor.index)
        ret = ret.loc[common_idx]
        factor = factor.loc[common_idx]

        if len(ret) < 20:
            return {"error": "insufficient data"}

        try:
            import statsmodels.api as sm
            X = sm.add_constant(factor.values)
            y = ret.values
            model = sm.OLS(y, X).fit()

            exposures = {}
            for i, col in enumerate(factor.columns):
                exposures[str(col)] = round(float(model.params[i + 1]), 4)

            return {
                "alpha": round(float(model.params[0]) * 252, 4),  # 年化Alpha
                "beta": round(float(exposures.get(list(factor.columns)[0], 0)
                                     if len(factor.columns) > 0 else 0), 4),
                "r_squared": round(float(model.rsquared), 4),
                "factor_exposures": exposures,
                "n_observations": int(model.nobs),
            }
        except Exception as e:
            return {"error": str(e)}

    def drawdown_analysis(self, returns: pd.Series) -> dict:
        """回撤深度分析。"""
        ret = returns.dropna()
        cum = (1 + ret).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak

        max_dd_idx = dd.idxmin()
        max_dd_val = dd.min()

        # 找起止
        peak_idx = cum[:max_dd_idx].idxmax() if not dd[:max_dd_idx].empty else ret.index[0]
        dd_start = peak_idx
        dd_end = max_dd_idx

        # 恢复时间
        recovery = None
        post_dd = cum[dd_end:]
        recovery_mask = post_dd >= cum.loc[peak_idx]
        if recovery_mask.any():
            recovery = post_dd[recovery_mask].index[0]

        return {
            "max_drawdown": round(float(max_dd_val), 4),
            "peak_date": str(dd_start.date()),
            "valley_date": str(dd_end.date()),
            "recovery_date": str(recovery.date()) if recovery is not None else None,
            "drawdown_duration_days": (dd_end - dd_start).days,
            "recovery_days": (recovery - dd_end).days if recovery is not None else None,
        }

    def rolling_stats(self, returns: pd.Series, window: int = 252) -> dict:
        """滚动指标计算。"""
        ret = returns.dropna()
        if len(ret) < window:
            window = len(ret) // 2
        if window < 20:
            return {"error": "insufficient data"}

        rolling_vol = ret.rolling(window).std() * np.sqrt(252)
        rolling_ret = ret.rolling(window).mean() * 252
        rolling_sharpe = rolling_ret / (rolling_vol + 1e-8)

        # P2-Q26-fix: 删除未使用的滚动最大回撤死代码（原 cum/rolling_cum 计算后从不使用）

        return {
            "rolling_sharpe_last": round(float(rolling_sharpe.iloc[-1]), 2),
            "rolling_vol_last": round(float(rolling_vol.iloc[-1]), 4),
            "rolling_sharpe_mean": round(float(rolling_sharpe.mean()), 2),
            "rolling_sharpe_std": round(float(rolling_sharpe.std()), 2),
        }

    def benchmark_comparison(self, returns: pd.Series,
                              benchmark: pd.Series) -> dict:
        """基准对比分析。"""
        ret = returns.dropna()
        bm = benchmark.dropna()
        common = ret.index.intersection(bm.index)
        ret = ret.loc[common]
        bm = bm.loc[common]

        if len(ret) < 20:
            return {"error": "insufficient data"}

        active = ret - bm
        tracking_error = float(active.std() * np.sqrt(252))
        active_return = float(active.mean() * 252)
        information_ratio = active_return / max(tracking_error, 1e-8)

        # Up/Down capture
        up_mask = bm > 0
        down_mask = bm < 0
        up_capture = float(ret[up_mask].mean() / bm[up_mask].mean()) if up_mask.any() else 0
        down_capture = float(ret[down_mask].mean() / bm[down_mask].mean()) if down_mask.any() else 0

        batting_avg = float((active > 0).mean())

        return {
            "tracking_error": round(tracking_error, 4),
            "information_ratio": round(information_ratio, 2),
            "active_return": round(active_return, 4),
            "batting_average": round(batting_avg, 4),
            "up_capture": round(up_capture, 4),
            "down_capture": round(down_capture, 4),
        }


def main() -> None:
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=500, freq="B")
    returns = pd.Series(np.random.randn(500) * 0.015 + 0.0005, index=dates)
    bm = pd.Series(np.random.randn(500) * 0.012 + 0.0003, index=dates)
    factor_ret = pd.DataFrame({
        "market": np.random.randn(500) * 0.01,
        "size": np.random.randn(500) * 0.005,
        "value": np.random.randn(500) * 0.005,
    }, index=dates)

    pa = PyfolioAnalyzer()
    print(f"Pyfolio available: {pa.available}")

    tear = pa.tear_sheet(returns)
    print(f"\nSharpe: {tear.get('sharpe', 'N/A')}")
    print(f"Max DD: {tear.get('max_drawdown', 'N/A')}")

    dd = pa.drawdown_analysis(returns)
    print(f"\nDrawdown peak: {dd.get('peak_date')}")
    print(f"Drawdown valley: {dd.get('valley_date')}")

    bm_comp = pa.benchmark_comparison(returns, bm)
    print(f"\nTracking error: {bm_comp.get('tracking_error', 'N/A')}")
    print(f"Info ratio: {bm_comp.get('information_ratio', 'N/A')}")


if __name__ == "__main__":
    main()
