"""
alphalens_adapter — 因子IC分析适配层 (V5)

将 alphalens-reloaded 集成到因子分析流程：
  - factor_zoo.py 因子 → IC/IR → 分层收益 → 因子排名
  - factor_model.py 的IC计算改用alphalens

降级策略: alphalens不可用时返回空数据，不阻塞启动。
"""

from __future__ import annotations
import logging

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="alphalens")
warnings.filterwarnings("ignore", message=".*distutils.*", category=DeprecationWarning)

import numpy as np
import pandas as pd

try:
    _HAS_ALPHALENS = True
except ImportError:
    _HAS_ALPHALENS = False


class AlphalensFactorAnalysis:
    """因子IC分析（封装alphalens）。"""

    def __init__(self) -> None:
        self.available = _HAS_ALPHALENS

    def _check(self) -> None:
        if not self.available:
            raise ImportError("alphalens not installed (pip install alphalens-reloaded)")

    def compute_ic(self, factor_data: pd.DataFrame, prices: pd.DataFrame,
                   periods: tuple[int, ...] = (1, 5, 10, 20)) -> dict:
        """因子IC分析。

        Args:
            factor_data: 因子值, index=date, columns=codes
            prices: 价格数据, index=date, columns=codes
            periods: 预测周期(交易日)

        Returns:
            {
                "ic_mean": float, "ic_ir": float, "ic_std": float,
                "ic_win_rate": float, "ic_series": dict,
                "period_breakdown": [{period, ic_mean, ic_ir, ic_std}],
            }
        """
        if not self.available:
            return {"available": False, "error": "alphalens not installed"}

        try:
            import alphalens as al

            # 展平为 alphalens 格式: multi-index (date, asset), factor
            factor_flat = factor_data.stack()
            factor_flat = factor_flat.rename("factor").to_frame()
            factor_flat.index.names = ["date", "asset"]

            # 价格格式: multi-index (date, asset), price
            price_flat = prices.stack()
            price_flat = price_flat.rename("price").to_frame()
            price_flat.index.names = ["date", "asset"]

            # 合并
            merged = factor_flat.join(price_flat, how="inner")

            # 计算前向收益
            for p in periods:
                merged[f"ret_{p}d"] = merged.groupby("asset")["price"].transform(
                    lambda x: x.shift(-p) / x - 1
                )

            result = {"period_breakdown": []}
            for p in periods:
                ret_col = f"ret_{p}d"
                valid = merged.dropna(subset=["factor", ret_col])
                if len(valid) < 10:
                    continue

                ic = valid.groupby("date").apply(
                    lambda g: g["factor"].corr(g[ret_col])
                ).dropna()

                if len(ic) > 0:
                    ic_mean = float(ic.mean())
                    ic_std = float(ic.std()) if len(ic) > 1 else 0
                    ic_ir = ic_mean / max(ic_std, 1e-8)
                    ic_win = float((ic > 0).mean())

                    result["period_breakdown"].append({
                        "period": p,
                        "ic_mean": round(ic_mean, 4),
                        "ic_ir": round(ic_ir, 2),
                        "ic_std": round(ic_std, 4),
                        "ic_win_rate": round(ic_win, 4),
                        "ic_count": int(len(ic)),
                    })

            if result["period_breakdown"]:
                p1 = result["period_breakdown"][0]
                result["ic_mean"] = p1["ic_mean"]
                result["ic_ir"] = p1["ic_ir"]
                result["ic_std"] = p1["ic_std"]
                result["ic_win_rate"] = p1["ic_win_rate"]

            # 尝试使用 alphalens 官方函数
            try:
                factor_data_al = al.utils.get_clean_factor_and_forward_returns(
                    factor_flat, prices, periods=list(periods),
                    quantiles=5, bins=None
                )
                al_ic = al.performance.factor_information_coefficient(factor_data_al)
                if al_ic is not None and not al_ic.empty:
                    result["alphalens_ic"] = {
                        str(k): {
                            "mean": round(float(v["mean"]), 4) if "mean" in v else None
                        }
                        for k, v in al_ic.items()
                    }
            except Exception as e:
                logging.getLogger(__name__).error(f"[alphalens_adapter] 操作失败: {e}", exc_info=True)

            return result

        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_quantile_returns(self, factor_data: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 quantiles: int = 5,
                                 periods: tuple[int, ...] = (1, 5)) -> dict:
        """分层收益分析。

        Returns:
            {
                "quantile_returns": {period: {q: return}},
                "long_short_return": float,          # 首个周期的多空收益(最高层-最低层)
                "long_short_sharpe": float,          # 首个周期的多空夏普(日频年化)
                "long_short_by_period": {period: return},
                "sharpe_by_period": {period: sharpe},
            }
        """
        if not self.available:
            return {"available": False, "error": "alphalens not installed"}

        try:
            import alphalens as al
            import numpy as np

            factor_flat = factor_data.stack()
            factor_flat = factor_flat.rename("factor").to_frame()
            factor_flat.index.names = ["date", "asset"]

            factor_data_al = al.utils.get_clean_factor_and_forward_returns(
                factor_flat, prices, periods=list(periods),
                quantiles=quantiles, bins=None
            )

            # P1-Q26-fix: alphalens 0.4.6 无 mean_return_by_quantile_and_period，
            # 使用现有 API mean_return_by_quantile（返回 (mean_ret, std_err)）。
            mean_ret, _ = al.performance.mean_return_by_quantile(factor_data_al)
            if mean_ret is None or mean_ret.empty:
                return {"available": False, "error": "mean_return_by_quantile returned empty"}

            # 各分层/各周期平均收益
            q_returns = {}
            for period in mean_ret.columns:
                q_returns[str(period)] = {
                    str(int(q)): round(float(v), 4)
                    for q, v in mean_ret[period].items()
                }

            # 多空收益 = 最高分层 - 最低分层；夏普用日度多空收益序列年化
            ls_daily = al.performance.factor_returns(factor_data_al, demeaned=True)
            ls_ret: dict[str, float] = {}
            ls_sharpe: dict[str, float] = {}
            top_q, bottom_q = mean_ret.index.max(), mean_ret.index.min()
            for period in mean_ret.columns:
                ls_ret[str(period)] = round(
                    float(mean_ret.loc[top_q, period] - mean_ret.loc[bottom_q, period]), 4
                )
                if ls_daily is not None and period in ls_daily.columns:
                    s = ls_daily[period].dropna()
                    if not s.empty:
                        std = float(s.std())
                        ls_sharpe[str(period)] = round(
                            float(s.mean()) / std * np.sqrt(252), 3
                        ) if std > 0 else 0.0

            if not ls_ret:
                return {"available": False, "error": "no period available for quantile returns"}

            primary = next(iter(ls_ret))
            # P1-Q26-fix: 删除硬编码 0.0；异常不再静默吞掉（返回 error 可见）
            return {
                "quantile_returns": q_returns,
                "long_short_return": ls_ret[primary],
                "long_short_sharpe": ls_sharpe.get(primary, 0.0),
                "long_short_by_period": ls_ret,
                "sharpe_by_period": ls_sharpe,
                "available": True,
            }

        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_factor_turnover(self, factor_data: pd.DataFrame) -> dict:
        """因子换手率（排名变化）。

        P2-Q26-fix: 原实现实际输出"排名自相关"（rank autocorr）却命名为换手率，
        语义误导；且单资产输入 autocorr=NaN → mean=NaN。
        现输出真实换手率（相邻两期排名发生变化的资产占比），并把排名自相关
        作为独立指标 rank_autocorr_mean 输出。
        """
        if not self.available:
            return {"available": False}
        try:
            factor_flat = factor_data.stack()
            factor_flat = factor_flat.rename("factor").to_frame()
            factor_flat.index.names = ["date", "asset"]

            # 横截面排名（每期对全部资产排名）
            rank_df = factor_flat["factor"].unstack()  # index=date, columns=asset
            if rank_df.shape[1] < 2:
                # 单资产：无法计算排名变化；autocorr=NaN → 显式返回0并注明
                return {
                    "available": True,
                    "mean_turnover": 0.0,
                    "turnover_1m": 0.0,
                    "rank_autocorr_mean": 0.0,
                    "note": "single asset: turnover=0, autocorr=NaN→0",
                }

            ranks = rank_df.rank(axis=1)
            # 真实换手率：相邻两期均有效的资产中排名发生变化的占比（取均值）
            turnover_series: list[float] = []
            for i in range(1, len(ranks)):
                prev_r, cur_r = ranks.iloc[i - 1], ranks.iloc[i]
                common = prev_r.notna() & cur_r.notna()
                if int(common.sum()) >= 2:
                    turnover_series.append(float((prev_r[common] != cur_r[common]).mean()))
            mean_turnover = float(np.mean(turnover_series)) if turnover_series else 0.0
            last_20 = turnover_series[-20:]
            turnover_1m = float(np.mean(last_20)) if last_20 else mean_turnover

            # 排名自相关（每只资产的时间序列 rank autocorr，取均值）
            autocorr = ranks.apply(lambda x: x.autocorr(), axis=0).dropna()
            ac_mean = float(autocorr.mean()) if len(autocorr) > 0 and not np.isnan(autocorr.mean()) else 0.0

            return {
                "available": True,
                "mean_turnover": round(mean_turnover, 4),
                "turnover_1m": round(turnover_1m, 4),
                "rank_autocorr_mean": round(ac_mean, 4),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_all(self, factor_data: pd.DataFrame,
                    prices: pd.DataFrame) -> dict:
        """一站式因子分析。"""
        return {
            "ic": self.compute_ic(factor_data, prices),
            "quantile": self.compute_quantile_returns(factor_data, prices),
            "turnover": self.compute_factor_turnover(factor_data),
        }


def main() -> None:
    """模拟数据测试"""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    codes = [f"{i:06d}" for i in range(100, 120)]

    factor_data = pd.DataFrame(
        np.random.randn(len(dates), len(codes)),
        index=dates, columns=codes
    )
    prices = 100 + np.cumsum(np.random.randn(len(dates), len(codes)) * 0.5, axis=0)
    prices = pd.DataFrame(prices, index=dates, columns=codes)

    afa = AlphalensFactorAnalysis()
    print(f"alphalens available: {afa.available}")

    ic = afa.compute_ic(factor_data, prices)
    print("\nIC Analysis:")
    for pb in ic.get("period_breakdown", []):
        print(f"  Period {pb['period']}d: IC={pb['ic_mean']:.4f}, IR={pb['ic_ir']:.2f}, Win={pb['ic_win_rate']:.0%}")


if __name__ == "__main__":
    main()
