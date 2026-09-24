"""
signal_correlation — 信号相关性分析 (V5)

核心问题：不同信号类型是否在传递重复信息？如果两个信号高度相关，
说明它们可能捕捉的是同一种市场现象，组合使用时应避免重复加权。

方法：
  1. 将每种信号类型转换为按日期对齐的方向序列 (bullish=+1, bearish=-1, neutral=0)
  2. 用最近共同交易日窗口计算各信号类型两两之间的 Pearson 相关系数
  3. 相关系数 > 0.7 (或 < -0.7) 的信号对标记为"冗余对"
"""

from __future__ import annotations
import logging

from typing import Any, Optional

import pandas as pd

from .signal_store import SignalStore

REDUNDANCY_THRESHOLD = 0.7
MIN_OVERLAP_DAYS = 5


class SignalCorrelation:
    """信号相关性分析引擎。

    Attributes:
        store: 信号存储实例
    """

    def __init__(self, store: Optional[SignalStore] = None) -> None:
        self.store = store if store is not None else SignalStore()

    @staticmethod
    def _direction_to_value(direction: str, strength: float = 1.0) -> float:
        """将信号方向转换为数值 (bullish=+strength, bearish=-strength, neutral=0)。"""
        if direction == "bullish":
            return strength if strength else 1.0
        if direction == "bearish":
            return -(strength if strength else 1.0)
        return 0.0

    def _build_daily_series(self) -> pd.DataFrame:
        """构建各信号类型按日期对齐的方向强度序列 (日期 x 信号类型)。

        同一天同一类型出现多条信号时取均值。
        """
        records = self.store.load()
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            ts = r.get("timestamp", "")
            try:
                date = pd.Timestamp(ts).tz_localize(None).normalize()
            except Exception as e:
                logging.getLogger(__name__).error(f"[signal_correlation] 操作失败: {e}", exc_info=True)
                continue
            value = self._direction_to_value(r.get("direction", "neutral"), float(r.get("strength", 1.0) or 1.0))
            rows.append({"date": date, "signal_type": r.get("signal_type", "unknown"), "value": value})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        pivot = df.groupby(["date", "signal_type"])["value"].mean().unstack("signal_type")
        return pivot

    def compute_correlation(self) -> dict[str, Any]:
        """计算各信号类型的两两相关性矩阵，并识别冗余信号对。

        Returns:
            {
                "correlation_matrix": {type1: {type2: corr, ...}, ...},
                "redundant_pairs": [{"sig1":..., "sig2":..., "corr":..., "recommendation":...}, ...],
            }
        """
        try:
            pivot = self._build_daily_series()
            if pivot.empty or pivot.shape[1] < 2:
                return {
                    "correlation_matrix": {},
                    "redundant_pairs": [],
                    "warning": "信号类型不足2种或无有效数据，无法计算相关性",
                }

            # P2-Q11-fix(M047): 无信号日不能填 0 当作中性信号；两两相关性只在
            # 两类信号均有触发的交集日上计算，并暴露重叠样本数，避免稀疏信号被扭曲。
            types = list(pivot.columns)
            corr_values: dict[tuple[str, str], float | None] = {}
            overlap_counts: dict[str, dict[str, int]] = {t: {} for t in types}
            for t1 in types:
                for t2 in types:
                    if t1 == t2:
                        continue
                    pair = pivot[[t1, t2]].dropna()
                    overlap_counts[t1][t2] = int(len(pair))
                    if len(pair) < MIN_OVERLAP_DAYS:
                        corr_values[(t1, t2)] = None
                        continue
                    if pair[t1].std() <= 1e-12 or pair[t2].std() <= 1e-12:
                        corr_values[(t1, t2)] = None
                        continue
                    corr_values[(t1, t2)] = float(pair[t1].corr(pair[t2], method="pearson"))

            correlation_matrix: dict[str, dict[str, float | None]] = {}
            for t1 in types:
                correlation_matrix[t1] = {}
                for t2 in types:
                    if t1 == t2:
                        continue
                    val = corr_values.get((t1, t2))
                    correlation_matrix[t1][t2] = round(val, 3) if val is not None and pd.notna(val) else None

            redundant_pairs: list[dict[str, Any]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for t1 in types:
                for t2 in types:
                    if t1 == t2:
                        continue
                    pair_key = tuple(sorted([t1, t2]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    corr_val = corr_values.get((t1, t2))
                    if corr_val is None or pd.isna(corr_val):
                        continue
                    if abs(corr_val) >= REDUNDANCY_THRESHOLD:
                        recommendation = "可合并" if corr_val > 0 else "方向相反，考虑作为对冲组合"
                        redundant_pairs.append({
                            "sig1": t1,
                            "sig2": t2,
                            "corr": round(float(corr_val), 3),
                            "overlap_days": overlap_counts[t1][t2],
                            "recommendation": recommendation,
                        })

            redundant_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

            return {
                "correlation_matrix": correlation_matrix,
                "redundant_pairs": redundant_pairs,
                "overlap_counts": overlap_counts,
                "min_overlap_days": MIN_OVERLAP_DAYS,
            }
        except Exception as exc:
            return {"error": str(exc), "correlation_matrix": {}, "redundant_pairs": []}


def main() -> None:
    """示例：写入几类模拟信号并计算相关性。"""
    from datetime import datetime, timedelta, timezone

    CST = timezone(timedelta(hours=8))
    store = SignalStore()
    now = datetime.now(CST)

    # breadth_thrust 和 volume_divergence 方向高度一致（模拟冗余信号）
    for i, offset in enumerate([100, 90, 80, 70, 60, 50, 40, 30]):
        direction = "bullish" if i % 2 == 0 else "bearish"
        ts = (now - timedelta(days=offset)).isoformat()
        store.save({
            "signal_type": "breadth_thrust", "direction": direction, "strength": 0.8,
            "target": "沪深300", "value": 0.6, "threshold": 0.5,
            "source_module": "market_depth.breadth_thrust", "metadata": {},
            "timestamp": ts, "expiry": ts,
        })
        store.save({
            "signal_type": "volume_divergence", "direction": direction, "strength": 0.7,
            "target": "沪深300", "value": 0.5, "threshold": 0.4,
            "source_module": "market_depth.volume_divergence", "metadata": {},
            "timestamp": ts, "expiry": ts,
        })

    sc = SignalCorrelation(store)
    result = sc.compute_correlation()

    print("═" * 55)
    print("  信号相关性矩阵")
    print("═" * 55)
    for t1, row in result.get("correlation_matrix", {}).items():
        print(f"  {t1}: {row}")

    print("\n  冗余信号对:")
    for p in result.get("redundant_pairs", []):
        print(f"    {p['sig1']} <-> {p['sig2']}: corr={p['corr']:+.3f} ({p['recommendation']})")


if __name__ == "__main__":
    main()
