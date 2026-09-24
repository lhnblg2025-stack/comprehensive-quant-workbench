"""
signal_performance — 信号表现评估 (V5)

核心问题：历史信号触发后，市场实际走势如何？信号是否真的有效？

流程：
  1. 从 SignalStore 读取历史信号
  2. 获取沪深300指数日线，计算信号触发后 1/5/20/60 日的市场收益率
  3. 按信号类型分组统计：平均前瞻收益、胜率、Sharpe
  4. 结合市场状态（牛/熊/震荡）找出信号表现最好的市场环境
"""

from __future__ import annotations
import logging

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from .signal_store import SignalStore

CST = timezone(timedelta(hours=8))

FORWARD_WINDOWS = (1, 5, 20, 60)


class SignalPerformance:
    """信号历史表现评估引擎。

    Attributes:
        store: 信号存储实例
    """

    def __init__(self, store: Optional[SignalStore] = None) -> None:
        self.store = store if store is not None else SignalStore()
        self._index_cache: Optional[pd.DataFrame] = None

    # ────────────────────────────────────────────────────────────
    #  数据获取
    # ────────────────────────────────────────────────────────────

    def _fetch_index_daily(self, symbol: str = "sh000300") -> pd.DataFrame:
        """获取沪深300指数日线数据 (date, close 排序)。

        Returns:
            DataFrame，index 为日期 (Timestamp)，列包含 close。
            获取失败时返回空 DataFrame。
        """
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

    def _forward_return(self, signal_date: pd.Timestamp, days: int, index_df: pd.DataFrame) -> Optional[float]:
        """计算某个信号日之后 N 个交易日的市场收益率。

        Args:
            signal_date: 信号发生日期时间（含时点）
            days: 前瞻交易日数
            index_df: 指数日线 DataFrame (index 为日期, 含 close 列)

        Returns:
            收益率 (float)，数据不足时返回 None。

        Q11-fix: 原实现用 side="left" 取信号日当日收盘为 t0——盘中信号会把
        信号日剩余时段（信号时点后到收盘）的收益计入前瞻收益（前视）。
        现按信号时点区分：盘中（<15:00）t0 用下一交易日（side="right"），
        收盘后信号（>=15:00）当日收盘已知，可保留当日（side="left"）。
        """
        if index_df.empty:
            return None
        try:
            # Q11-fix: 先 normalize 去掉时间部分，保证 searchsorted 落在当日 bar；
            # 盘中信号（<15:00）t0 用下一交易日（side="right"），
            # 收盘后信号（>=15:00）当日收盘已知，可保留当日（side="left"）。
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

    @staticmethod
    def _is_expired(sig: dict[str, Any]) -> bool:
        """判断信号是否已过期（expiry < 当前时间）。

        P2-Q11-fix(M048): 分析口径与存储语义对齐——expiry 为空/解析失败视为
        永不过期；仅 expiry 明确早于当前时间的信号判定为过期。
        """
        expiry_raw = sig.get("expiry", "")
        if not expiry_raw:
            return False
        try:
            expiry_dt = pd.Timestamp(expiry_raw).tz_localize(None)
            return expiry_dt < datetime.now(CST)
        except Exception:
            return False

    def _market_regime_at(self, signal_date: pd.Timestamp, index_df: pd.DataFrame) -> str:
        """判断信号发生时的市场状态（简单版：MA250 相对位置）。

        Returns:
            "牛市" / "熊市" / "震荡市" / "未知"
        """
        if index_df.empty:
            return "未知"
        try:
            # Q11-fix: 盘中信号（<15:00）时信号日当日收盘尚不可知，市场状态
            # 必须用信号时点之前最后一个交易日（side="left"-1）；
            # 收盘后信号（>=15:00）可用当日收盘（side="right"-1）。
            if signal_date.hour >= 15:
                idx_pos = index_df.index.searchsorted(signal_date, side="right") - 1
            else:
                idx_pos = index_df.index.searchsorted(signal_date, side="left") - 1
            if idx_pos < 250:
                return "未知"
            window = index_df["close"].iloc[max(0, idx_pos - 249):idx_pos + 1]
            if len(window) < 100:
                return "未知"
            ma250 = window.mean()
            current = index_df["close"].iloc[idx_pos]
            deviation = current / ma250 - 1
            if deviation > 0.05:
                return "牛市"
            elif deviation < -0.05:
                return "熊市"
            return "震荡市"
        except Exception:
            return "未知"

    # ────────────────────────────────────────────────────────────
    #  核心评估
    # ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        signal_type: Optional[str] = None,
        include_expired: bool = False,
    ) -> dict[str, Any]:
        """评估历史信号的表现。

        Args:
            signal_type: 指定信号类型，None 表示评估全部类型（并按类型分组统计Sharpe）。
            include_expired: 是否计入已过期信号（expiry < 当前时间），默认 False。

        P2-Q11-fix(M048): 统计口径与存储语义对齐——默认剔除已过期信号
        （"信号失效"不再只依赖 delete_expired() 手动清理），返回中带
        include_expired / expired_filtered 字段供调用方感知口径。

        Returns:
            {
                "total_signals": int,
                "avg_forward_return": {"1d":..., "5d":..., "20d":..., "60d":...},
                "win_rate": {"1d":..., "5d":..., "20d":..., "60d":...},
                "sharpe_by_type": {type: sharpe, ...},
                "best_conditions": {"market_regime":..., "avg_return_in_regime":...},
                "detailed_signals": [{"id":..., "date":..., "type":..., "forward_20d":...}, ...],
            }
        """
        try:
            signals = self.store.load(signal_type=signal_type)
            total_loaded = len(signals)
            expired_filtered = 0
            if not include_expired:
                kept = [s for s in signals if not self._is_expired(s)]
                expired_filtered = total_loaded - len(kept)
                signals = kept
            index_df = self._fetch_index_daily()

            if not signals:
                return {
                    "total_signals": 0,
                    "avg_forward_return": {f"{d}d": None for d in FORWARD_WINDOWS},
                    "win_rate": {f"{d}d": None for d in FORWARD_WINDOWS},
                    "sharpe_by_type": {},
                    "best_conditions": None,
                    "detailed_signals": [],
                    "include_expired": include_expired,
                    "expired_filtered": expired_filtered,
                    "warning": "无历史信号数据",
                }

            forward_returns: dict[int, list[float]] = {d: [] for d in FORWARD_WINDOWS}
            returns_by_type: dict[str, list[float]] = {}
            regime_returns: dict[str, list[float]] = {}
            detailed: list[dict[str, Any]] = []

            for sig in signals:
                ts = sig.get("timestamp", "")
                try:
                    sig_date = pd.Timestamp(ts).tz_localize(None)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[signal_performance] 操作失败: {e}", exc_info=True)
                    continue

                direction_mult = 1.0
                if sig.get("direction") == "bearish":
                    direction_mult = -1.0  # 空头信号：市场下跌=胜

                row: dict[str, Any] = {
                    "id": sig.get("signal_id", ""),
                    "date": ts[:10] if ts else "",
                    "type": sig.get("signal_type", ""),
                    "direction": sig.get("direction", ""),
                }

                fwd20 = None
                for d in FORWARD_WINDOWS:
                    ret = self._forward_return(sig_date, d, index_df)
                    if ret is not None:
                        adj_ret = ret * direction_mult
                        forward_returns[d].append(adj_ret)
                        row[f"forward_{d}d"] = round(adj_ret, 4)
                        if d == 20:
                            fwd20 = adj_ret
                    else:
                        row[f"forward_{d}d"] = None

                stype = sig.get("signal_type", "unknown")
                if fwd20 is not None:
                    returns_by_type.setdefault(stype, []).append(fwd20)
                    regime = self._market_regime_at(sig_date, index_df)
                    regime_returns.setdefault(regime, []).append(fwd20)
                    row["market_regime"] = regime

                detailed.append(row)

            avg_forward_return: dict[str, Any] = {}
            win_rate: dict[str, Any] = {}
            for d in FORWARD_WINDOWS:
                vals = forward_returns[d]
                key = f"{d}d"
                if vals:
                    avg_forward_return[key] = round(float(np.mean(vals)), 4)
                    win_rate[key] = round(float(np.mean([1 if v > 0 else 0 for v in vals])), 4)
                else:
                    # P2-Q11-fix(M045): 某窗口无有效前瞻收益时用 None 而非 0.0，
                    # 避免调用方误读为"信号收益为 0"。
                    avg_forward_return[key] = None
                    win_rate[key] = None

            sharpe_by_type: dict[str, float] = {}
            for stype, rets in returns_by_type.items():
                arr = np.array(rets)
                if len(arr) > 1 and arr.std() > 1e-9:
                    sharpe_by_type[stype] = round(float(arr.mean() / arr.std() * np.sqrt(252 / 20)), 3)
                else:
                    sharpe_by_type[stype] = 0.0

            best_regime = "未知"
            best_avg = -np.inf
            for regime, rets in regime_returns.items():
                if regime == "未知" or not rets:
                    continue
                avg = float(np.mean(rets))
                if avg > best_avg:
                    best_avg = avg
                    best_regime = regime

            return {
                "total_signals": len(signals),
                "avg_forward_return": avg_forward_return,
                "win_rate": win_rate,
                "sharpe_by_type": sharpe_by_type,
                "best_conditions": {
                    "market_regime": best_regime,
                    # P2-Q11-fix(M045): 无有效 regime 数据时 avg_return 用 None，
                    # 与 0.0 占位语义区分。
                    "avg_return_in_regime": round(best_avg, 4) if best_avg != -np.inf else None,
                },
                "detailed_signals": detailed,
                "include_expired": include_expired,
                "expired_filtered": expired_filtered,
            }
        except Exception as exc:
            return {"error": str(exc), "total_signals": 0}


def main() -> None:
    """示例：写入模拟信号并评估其历史表现。"""
    from .signal_store import SignalStore

    store = SignalStore()
    now = datetime.now(CST)

    # 写入若干模拟历史信号（用于演示评估流程；实际使用中信号由各分析模块 save() 产生）
    for i, offset in enumerate([300, 250, 200, 150, 100, 80, 60, 40]):
        store.save({
            "signal_type": "breadth_thrust",
            "direction": "bullish" if i % 2 == 0 else "bearish",
            "strength": 0.7,
            "target": "沪深300",
            "value": 0.6,
            "threshold": 0.5,
            "source_module": "market_depth.breadth_thrust",
            "metadata": {},
            "timestamp": (now - timedelta(days=offset)).isoformat(),
            "expiry": (now - timedelta(days=offset - 5)).isoformat(),
        })

    sp = SignalPerformance(store)
    # P2-Q11-fix(M048): 示例信号 expiry 均为过去（演示用），显式 include_expired=True
    # 以保留演示效果；实际调用默认 False（剔除过期信号）。
    result = sp.evaluate(signal_type="breadth_thrust", include_expired=True)

    print("═" * 55)
    print("  信号表现评估: breadth_thrust")
    print("═" * 55)
    print(f"  信号总数: {result.get('total_signals')}")
    print(f"  平均前瞻收益: {result.get('avg_forward_return')}")
    print(f"  胜率: {result.get('win_rate')}")
    print(f"  Sharpe: {result.get('sharpe_by_type')}")
    print(f"  最佳市场条件: {result.get('best_conditions')}")


if __name__ == "__main__":
    main()
