"""
attribution_stability — 归因稳定性分析 (V5)

核心问题：滚动归因得到的配置效应/选择效应，是持续稳定的能力，
还是几个窗口的运气爆发？

方法：
  对滚动归因结果序列计算 allocation_effect / selection_effect 的标准差，
  标准差越低说明该效应越稳定（可复制），越高说明波动巨大（可能是噪音或运气）。

  stability_score = 1 / (1 + std)   # 值域 (0, 1]，越接近1越稳定

对标：主动管理基本法则中的"IC 稳定性"评估思路，应用于归因效应序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


class AttributionStability:
    """归因稳定性分析引擎。

    输入滚动窗口 Brinson 归因结果列表，输出配置/选择效应的稳定性评分与趋势判断。
    """

    def __init__(self, trend_threshold: float = 1e-4) -> None:
        """初始化。

        Args:
            trend_threshold: 判断趋势方向的斜率阈值，低于此绝对值视为"平稳"
        """
        self.trend_threshold = trend_threshold

    @staticmethod
    def _linear_trend_slope(series: np.ndarray) -> float:
        """用一元线性回归估计序列的时间趋势斜率。

        Args:
            series: 效应值时间序列

        Returns:
            斜率（每期变化量）；数据不足时返回 0.0
        """
        n = len(series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = series.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-12:
            return 0.0
        slope = float(np.sum((x - x_mean) * (series - y_mean)) / denom)
        return slope

    def _classify_trend(self, slope: float) -> str:
        """根据斜率判断趋势方向的中文描述。"""
        if abs(slope) < self.trend_threshold:
            return "平稳"
        return "上升" if slope > 0 else "下降"

    def analyze(self, rolling_results: list[dict[str, Any]]) -> dict[str, Any]:
        """分析滚动归因结果的稳定性。

        Args:
            rolling_results: RollingAttribution.compute() 的输出，
                每个元素需含 'allocation_effect' 和 'selection_effect' 字段；
                优先用 'total_active_return'（总主动收益）做趋势分析，
                缺省时回退 alloc+select+interaction

        Returns:
            dict:
                - allocation_std: 配置效应标准差
                - selection_std: 选择效应标准差
                - allocation_mean: 配置效应均值
                - selection_mean: 选择效应均值
                - stability_score: 综合稳定性评分 (0,1]
                - allocation_stability: 配置效应单独的稳定性评分
                - selection_stability: 选择效应单独的稳定性评分
                - trend: 总体主动收益的趋势方向（上升/下降/平稳）
                - interpretation: 文字解读
        """
        try:
            valid = [
                r for r in rolling_results
                if isinstance(r, dict) and "allocation_effect" in r and "error" not in r
            ]
            if len(valid) < 2:
                return {
                    "error": "有效滚动窗口数据不足(至少需要2个)",
                    "allocation_std": 0.0,
                    "selection_std": 0.0,
                    "stability_score": 0.0,
                    "trend": "数据不足",
                }

            alloc_series = np.array([r["allocation_effect"] for r in valid], dtype=float)
            sel_series = np.array([r["selection_effect"] for r in valid], dtype=float)
            # P2-Q24-fix (L284): 回退时含 interaction_effect，与总主动收益定义
            # (alloc + select + interact) 一致，避免漏掉交互效应导致趋势判断偏移
            active_series = np.array(
                [r.get(
                    "total_active_return",
                    r["allocation_effect"] + r["selection_effect"] + r.get("interaction_effect", 0.0),
                ) for r in valid],
                dtype=float,
            )

            alloc_std = float(np.std(alloc_series))
            sel_std = float(np.std(sel_series))
            alloc_mean = float(np.mean(alloc_series))
            sel_mean = float(np.mean(sel_series))

            allocation_stability = 1.0 / (1.0 + alloc_std)
            selection_stability = 1.0 / (1.0 + sel_std)
            combined_std = float(np.std(alloc_series + sel_series))
            stability_score = 1.0 / (1.0 + combined_std)

            trend_slope = self._linear_trend_slope(active_series)
            trend = self._classify_trend(trend_slope)

            interpretation = self._build_interpretation(
                alloc_mean, sel_mean, allocation_stability, selection_stability, trend
            )

            return {
                "allocation_std": round(alloc_std, 4),
                "selection_std": round(sel_std, 4),
                "allocation_mean": round(alloc_mean, 4),
                "selection_mean": round(sel_mean, 4),
                "stability_score": round(stability_score, 4),
                "allocation_stability": round(allocation_stability, 4),
                "selection_stability": round(selection_stability, 4),
                "trend": trend,
                "trend_slope": round(trend_slope, 6),
                "n_windows": len(valid),
                "interpretation": interpretation,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"归因稳定性分析失败: {exc}",
                "allocation_std": 0.0,
                "selection_std": 0.0,
                "stability_score": 0.0,
                "trend": "未知",
            }

    @staticmethod
    def _build_interpretation(
        alloc_mean: float,
        sel_mean: float,
        alloc_stab: float,
        sel_stab: float,
        trend: str,
    ) -> str:
        """生成中文解读文本。"""
        parts = []
        dominant = "配置效应" if abs(alloc_mean) > abs(sel_mean) else "选择效应"
        parts.append(f"主动收益主要来自{dominant}")

        if alloc_stab > 0.7:
            parts.append("行业配置能力较为稳定")
        elif alloc_stab < 0.3:
            parts.append("行业配置效应波动较大,可能存在运气成分")

        if sel_stab > 0.7:
            parts.append("个股选择能力较为稳定")
        elif sel_stab < 0.3:
            parts.append("个股选择效应波动较大,持续性有待观察")

        parts.append(f"主动收益趋势{trend}")

        return ",".join(parts)


def main() -> None:
    """使用模拟数据自测 AttributionStability。"""
    try:
        rng = np.random.default_rng(21)
        n_windows = 20

        # 场景：配置效应稳定为正，选择效应波动大且带轻微上升趋势
        rolling_results = []
        for i in range(n_windows):
            alloc = 0.01 + rng.normal(0, 0.002)
            sel = 0.005 * (i / n_windows) + rng.normal(0, 0.015)
            rolling_results.append({
                "allocation_effect": alloc,
                "selection_effect": sel,
                "total_active_return": alloc + sel,
            })

        engine = AttributionStability()
        result = engine.analyze(rolling_results)

        print("=== 归因稳定性分析结果 ===")
        print(f"配置效应std: {result['allocation_std']}, 选择效应std: {result['selection_std']}")
        print(f"稳定性评分: {result['stability_score']}")
        print(f"趋势: {result['trend']}")
        print(f"解读: {result['interpretation']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] 测试运行失败: {exc}")


if __name__ == "__main__":
    main()
