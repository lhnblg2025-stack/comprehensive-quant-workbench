"""
factor_timing — 因子择时能力评估 (V5)

核心问题：组合经理是否具备"因子择时"能力？
即：在因子即将上涨前提高暴露，在因子即将下跌前降低暴露？

方法：
  IC = corr(Δexposure_t, factor_return_{t+1})
  IR = mean(IC 序列) / std(IC 序列)

  IC > 0 且显著 → 择时方向正确（提前加/减仓有效）
  IR 越高 → 择时能力越稳定可靠

对标：Grinold & Kahn《主动投资组合管理》因子择时 IC/IR 框架
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


class FactorTiming:
    """因子择时能力评估引擎。

    评估组合经理调整因子暴露的时机是否领先于因子收益的变化。

    核心假设：
      如果在暴露提升后，该因子未来收益确实为正（或降低暴露后为负），
      说明经理具备真实的择时能力，而非随机噪音。
    """

    def __init__(self, min_periods: int = 6) -> None:
        """初始化。

        Args:
            min_periods: 计算 IC/IR 所需的最少观测期数
        """
        self.min_periods = max(3, min_periods)

    @staticmethod
    def _rolling_ic(delta_exposure: np.ndarray, future_returns: np.ndarray) -> float:
        """计算单个因子的择时 IC（暴露变化与未来因子收益的相关系数）。

        Args:
            delta_exposure: Δexposure 序列 (t期暴露 - t-1期暴露)
            future_returns: 对应的未来1期因子收益序列

        Returns:
            Pearson 相关系数；数据不足或方差为0时返回 0.0
        """
        n = min(len(delta_exposure), len(future_returns))
        if n < 3:
            return 0.0
        x = delta_exposure[:n]
        y = future_returns[:n]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    def evaluate(
        self,
        exposure_history: dict[str, list[float]],
        factor_returns: dict[str, list[float]],
    ) -> dict[str, Any]:
        """评估各因子的择时能力。

        Args:
            exposure_history: {因子名: [暴露值序列]}，按时间顺序排列
            factor_returns: {因子名: [因子收益率序列]}，与 exposure_history 对齐。
               严格领先对齐要求返回序列至少比暴露序列长 2 期（ret[t] 为第 t 期收益，
               暴露变化从 t-1 到 t 生效，收益在 t+1 兑现）；数据不足时自动截断。

        Returns:
            dict:
                - factor_timing_ics: 各因子单期 IC（滚动窗口平均后的单值示例，
                  这里用整体样本 IC 近似）
                - timing_ir: 各因子 IR = mean(rolling IC)/std(rolling IC)
                - overall_timing_ability: 各因子 IR 的平均值
                - best_timed_factor: IR 最高的因子
                - interpretation: 文字解读
        """
        try:
            factors = sorted(set(exposure_history) & set(factor_returns))
            if not factors:
                return {
                    "error": "无共同因子数据",
                    "factor_timing_ics": {},
                    "timing_ir": {},
                    "overall_timing_ability": 0.0,
                    "best_timed_factor": None,
                    "interpretation": "数据不足，无法评估因子择时能力",
                }

            factor_ics: dict[str, float] = {}
            factor_irs: dict[str, float] = {}

            for f in factors:
                exp = np.asarray(exposure_history[f], dtype=float)
                ret = np.asarray(factor_returns[f], dtype=float)
                if len(exp) < 2 or len(ret) < 2:
                    factor_ics[f] = 0.0
                    factor_irs[f] = 0.0
                    continue

                delta_exp = np.diff(exp)  # delta_exp[i] = exp[i+1]-exp[i]（变化发生于 i→i+1）
                # P2-Q24-fix (M278): 严格领先对齐——暴露变化 delta_exp[i] 应对齐其后的
                # 下一期收益 ret[i+2]（原实现用 ret[i+1]，变化与收益同期，择时能力被高估）。
                # 语义：暴露从 t-1 到 t 变化，对应收益在 t+1 兑现。
                future_ret = ret[2:len(delta_exp) + 2]

                n = min(len(delta_exp), len(future_ret))
                if n < self.min_periods:
                    factor_ics[f] = self._rolling_ic(delta_exp[:n], future_ret[:n])
                    factor_irs[f] = 0.0
                    continue

                delta_exp = delta_exp[:n]
                future_ret = future_ret[:n]

                # 滚动窗口计算 IC 序列，用于估计 IR
                window = max(self.min_periods, n // 4)
                ic_series = []
                for start in range(0, n - window + 1):
                    ic = self._rolling_ic(
                        delta_exp[start:start + window],
                        future_ret[start:start + window],
                    )
                    ic_series.append(ic)

                overall_ic = self._rolling_ic(delta_exp, future_ret)
                factor_ics[f] = round(overall_ic, 4)

                if len(ic_series) >= 2 and np.std(ic_series) > 1e-8:
                    ir = float(np.mean(ic_series) / np.std(ic_series))
                else:
                    ir = 0.0
                factor_irs[f] = round(ir, 4)

            overall_ability = round(float(np.mean(list(factor_irs.values()))), 4) if factor_irs else 0.0
            best_factor = max(factor_irs, key=lambda k: factor_irs[k]) if factor_irs else None

            interpretation = self._build_interpretation(factor_irs, best_factor)

            return {
                "factor_timing_ics": factor_ics,
                "timing_ir": factor_irs,
                "overall_timing_ability": overall_ability,
                "best_timed_factor": best_factor,
                "interpretation": interpretation,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"因子择时评估失败: {exc}",
                "factor_timing_ics": {},
                "timing_ir": {},
                "overall_timing_ability": 0.0,
                "best_timed_factor": None,
                "interpretation": "评估过程出错",
            }

    @staticmethod
    def _build_interpretation(factor_irs: dict[str, float], best_factor: str | None) -> str:
        """生成中文解读文本。"""
        if not factor_irs:
            return "数据不足，无法评估因子择时能力"

        parts = []
        if best_factor is not None:
            best_ir = factor_irs[best_factor]
            if best_ir > 0.5:
                parts.append(f"{best_factor}因子择时能力突出(IR={best_ir:.2f})")
            elif best_ir > 0:
                parts.append(f"{best_factor}因子择时能力尚可(IR={best_ir:.2f})")
            else:
                parts.append("各因子择时能力普遍偏弱")

        negative = [f for f, ir in factor_irs.items() if ir < -0.1 and f != best_factor]
        if negative:
            parts.append(f"{negative[0]}择时为负,建议减少主动调仓频率")

        return ",".join(parts) if parts else "因子择时能力中性，无明显方向性偏差"


def main() -> None:
    """使用模拟数据自测 FactorTiming。"""
    try:
        rng = np.random.default_rng(7)
        n = 60

        # 构造一个"价值因子择时能力较强"的模拟场景：
        # 暴露变化领先于未来收益（正相关），动量因子择时为负
        value_future_ret = rng.normal(0.001, 0.02, n)
        value_delta_exp = 0.6 * np.roll(value_future_ret, -1) + rng.normal(0, 0.02, n)

        momentum_future_ret = rng.normal(0.0, 0.02, n)
        momentum_delta_exp = -0.4 * np.roll(momentum_future_ret, -1) + rng.normal(0, 0.02, n)

        size_future_ret = rng.normal(0.0, 0.02, n)
        size_delta_exp = rng.normal(0, 0.02, n)

        def to_exposure_series(delta: np.ndarray) -> list[float]:
            return list(np.cumsum(delta))

        exposure_history = {
            "value": to_exposure_series(value_delta_exp),
            "momentum": to_exposure_series(momentum_delta_exp),
            "size": to_exposure_series(size_delta_exp),
        }
        factor_returns = {
            "value": value_future_ret.tolist(),
            "momentum": momentum_future_ret.tolist(),
            "size": size_future_ret.tolist(),
        }

        engine = FactorTiming()
        result = engine.evaluate(exposure_history, factor_returns)

        print("=== 因子择时评估结果 ===")
        print(f"IC: {result['factor_timing_ics']}")
        print(f"IR: {result['timing_ir']}")
        print(f"整体择时能力: {result['overall_timing_ability']}")
        print(f"最佳择时因子: {result['best_timed_factor']}")
        print(f"解读: {result['interpretation']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 测试运行失败: {exc}")


if __name__ == "__main__":
    main()
