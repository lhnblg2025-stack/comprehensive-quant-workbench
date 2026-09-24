"""
signal_decay — 信号衰减分析 (V5)

核心问题：信号发出后，其预测效力能维持多久？该在第几天平仓？

流程：
  1. 对某一信号类型的所有历史信号，计算信号后 1~60 日的平均累计（方向调整后）收益
  2. 用指数衰减模型拟合收益曲线，求半衰期
  3. 根据累计收益曲线的峰值位置，给出最佳持有期建议
"""

from __future__ import annotations
import logging

from datetime import timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from .signal_store import SignalStore

CST = timezone(timedelta(hours=8))

DECAY_DAYS = list(range(1, 61))
# 报告用的关键节点（避免输出60个点过于冗长，同时不遗漏细节可全量返回）
REPORT_DAYS = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60]


class SignalDecay:
    """信号衰减分析引擎。

    Attributes:
        store: 信号存储实例
    """

    def __init__(self, store: Optional[SignalStore] = None) -> None:
        self.store = store if store is not None else SignalStore()
        self._index_cache: Optional[pd.DataFrame] = None

    def _fetch_index_daily(self, symbol: str = "sh000300") -> pd.DataFrame:
        """获取沪深300指数日线数据。"""
        if self._index_cache is not None:
            return self._index_cache
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or df.empty:
                self._index_cache = pd.DataFrame()
                return self._index_cache
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            df["close"] = df["close"].astype(float)
            self._index_cache = df
            return df
        except Exception:
            self._index_cache = pd.DataFrame()
            return self._index_cache

    def _cumulative_return(self, signal_date: pd.Timestamp, days: int, index_df: pd.DataFrame) -> Optional[float]:
        """信号日后第 N 日的累计收益（相对信号发生日的收盘价）。"""
        if index_df.empty:
            return None
        try:
            # Q11-fix: normalize 后定位当日 bar；盘中信号（<15:00）t0 用
            # 下一交易日（side="right"），收盘后信号（>=15:00）用当日（side="left"）。
            day = signal_date.normalize()
            if signal_date.hour >= 15:
                t0_pos = index_df.index.searchsorted(day, side="left")
            else:
                t0_pos = index_df.index.searchsorted(day, side="right")
            if t0_pos >= len(index_df):
                return None
            t1_pos = t0_pos + days
            if t1_pos >= len(index_df):
                return None
            base = index_df["close"].iloc[t0_pos]
            fwd = index_df["close"].iloc[t1_pos]
            if base <= 0:
                return None
            return float(fwd / base - 1)
        except Exception:
            return None

    def _fit_exponential_decay(self, days: list[int], returns: list[float]) -> tuple[float, int]:
        """拟合方向调整后收益序列的效力半衰期。

        P2-Q11-fix(M046): 不再用 |收益| + clip 把负收益伪装成正衰减；峰值后若
        序列转负，先截取同号衰减段拟合，负收益段视为信号反转而非原效力衰减。
        该半衰期口径为方向调整后累计收益的经验衰减代理，不等同严格 IC 半衰期。

        Returns:
            (half_life, optimal_holding_period)
        """
        try:
            arr = np.array(returns, dtype=float)
            days_arr = np.array(days, dtype=float)
            if len(arr) < 3:
                return 0.0, days[-1] if days else 0

            peak_idx = int(np.argmax(arr))
            optimal_holding_period = int(days_arr[peak_idx])

            if arr[peak_idx] <= 1e-9:
                # P2-Q11-fix(M046): 方向调整后收益从未形成正效力峰值，返回最早可用
                # 持有期并把半衰期标为 0，避免用负收益绝对值拟合出误导性半衰期。
                return 0.0, int(days_arr[0])

            # 用峰值后的数据拟合衰减（若峰值在最后，无衰减段，半衰期设为总窗口的2倍，视为未衰减）
            decay_days = days_arr[peak_idx:]
            decay_returns = arr[peak_idx:]

            # P2-Q11-fix(M046): 只保留峰值后仍为正的同号衰减段；首次跌至 0/负数
            # 说明原方向效力已耗尽或反转，不参与指数衰减拟合。
            positive_mask = decay_returns > 1e-9
            if not positive_mask.all():
                first_non_positive = int(np.argmax(~positive_mask))
                decay_days = decay_days[:first_non_positive]
                decay_returns = decay_returns[:first_non_positive]

            if len(decay_days) < 3 or decay_returns[0] <= 1e-9:
                return float(days_arr[-1] * 2), optimal_holding_period

            # log(R/R0) = -(t-t0)/tau  =>  线性回归求 tau
            rel_t = decay_days - decay_days[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = decay_returns / decay_returns[0]
                log_ratio = np.log(ratio)

            valid = np.isfinite(log_ratio) & (rel_t > 0)
            if valid.sum() < 2:
                return float(days_arr[-1] * 2), optimal_holding_period

            # 最小二乘拟合 log_ratio = -rel_t / tau
            slope = np.polyfit(rel_t[valid], log_ratio[valid], 1)[0]
            if slope >= -1e-9:
                # 衰减不明显（甚至继续增强），半衰期视为超出观测窗口
                half_life = float(days_arr[-1] * 2)
            else:
                tau = -1.0 / slope
                half_life = float(tau * np.log(2))
                half_life = max(0.0, min(half_life, days_arr[-1] * 3))

            return round(half_life, 1), optimal_holding_period
        except Exception:
            return 0.0, days[-1] if days else 0

    def analyze_decay(self, signal_type: str) -> dict[str, Any]:
        """分析指定信号类型的衰减模式。

        Args:
            signal_type: 信号类型

        Returns:
            {
                "signal_type": str,
                "cumulative_returns": [{"day": int, "return": float}, ...],
                "half_life": float,
                "optimal_holding_period": int,
            }
        """
        try:
            signals = self.store.load(signal_type=signal_type)
            index_df = self._fetch_index_daily()

            if not signals:
                return {
                    "signal_type": signal_type,
                    "cumulative_returns": [],
                    "half_life": 0.0,
                    "optimal_holding_period": 0,
                    "warning": "无历史信号数据",
                }

            day_returns: dict[int, list[float]] = {d: [] for d in DECAY_DAYS}

            for sig in signals:
                ts = sig.get("timestamp", "")
                try:
                    sig_date = pd.Timestamp(ts).tz_localize(None)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[signal_decay] 操作失败: {e}", exc_info=True)
                    continue

                direction_mult = -1.0 if sig.get("direction") == "bearish" else 1.0

                for d in DECAY_DAYS:
                    ret = self._cumulative_return(sig_date, d, index_df)
                    if ret is not None:
                        day_returns[d].append(ret * direction_mult)

            avg_by_day: dict[int, float] = {}
            for d in DECAY_DAYS:
                vals = day_returns[d]
                avg_by_day[d] = float(np.mean(vals)) if vals else np.nan

            valid_days = [d for d in DECAY_DAYS if not np.isnan(avg_by_day[d])]
            if not valid_days:
                return {
                    "signal_type": signal_type,
                    "cumulative_returns": [],
                    "half_life": 0.0,
                    "optimal_holding_period": 0,
                    "warning": "数据不足，无法计算衰减",
                }

            half_life, optimal_period = self._fit_exponential_decay(
                valid_days, [avg_by_day[d] for d in valid_days]
            )

            report_days = [d for d in REPORT_DAYS if d in avg_by_day and not np.isnan(avg_by_day[d])]
            cumulative_returns = [
                {"day": d, "return": round(avg_by_day[d], 4)} for d in report_days
            ]

            return {
                "signal_type": signal_type,
                "cumulative_returns": cumulative_returns,
                "half_life": half_life,
                "optimal_holding_period": optimal_period,
                "sample_size": len(signals),
            }
        except Exception as exc:
            return {"signal_type": signal_type, "error": str(exc)}


def main() -> None:
    """示例：写入模拟信号并分析其衰减模式。"""
    from datetime import datetime

    store = SignalStore()
    now = datetime.now(CST)

    for offset in [300, 250, 200, 150, 120, 90, 70]:
        store.save({
            "signal_type": "volume_divergence",
            "direction": "bullish",
            "strength": 0.7,
            "target": "沪深300",
            "value": 0.5,
            "threshold": 0.4,
            "source_module": "market_depth.volume_divergence",
            "metadata": {},
            "timestamp": (now - timedelta(days=offset)).isoformat(),
            "expiry": (now - timedelta(days=offset - 5)).isoformat(),
        })

    sd = SignalDecay(store)
    result = sd.analyze_decay("volume_divergence")

    print("═" * 55)
    print("  信号衰减分析: volume_divergence")
    print("═" * 55)
    print(f"  半衰期: {result.get('half_life')} 天")
    print(f"  最佳持有期: {result.get('optimal_holding_period')} 天")
    print("  累计收益曲线:")
    for pt in result.get("cumulative_returns", []):
        print(f"    第{pt['day']:>3}日: {pt['return']:+.4f}")


if __name__ == "__main__":
    main()
