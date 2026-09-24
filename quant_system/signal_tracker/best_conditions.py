"""
best_conditions — 最佳市场条件分析 (V5)

核心问题：某类信号在什么市场环境下最有效？（震荡/牛/熊 × 低波/中波/高波）

市场状态判断（用沪深300指数）：
  - 牛市: 收盘价显著高于250日均线 (偏离 > +5%)
  - 熊市: 收盘价显著低于250日均线 (偏离 < -5%)
  - 震荡市: 其余情况

波动率状态判断（20日年化波动率的历史分位）：
  - 低波: < 33% 分位
  - 中波: 33% ~ 67% 分位
  - 高波: > 67% 分位
"""

from __future__ import annotations
import logging

from datetime import timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from .signal_store import SignalStore

CST = timezone(timedelta(hours=8))


class BestConditions:
    """信号最佳市场条件分析引擎。

    Attributes:
        store: 信号存储实例
    """

    def __init__(self, store: Optional[SignalStore] = None) -> None:
        self.store = store if store is not None else SignalStore()
        self._index_cache: Optional[pd.DataFrame] = None

    def _fetch_index_daily(self, symbol: str = "sh000300") -> pd.DataFrame:
        """获取沪深300指数日线数据，附加250日均线、20日波动率及其历史分位。"""
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
            df["ma250"] = df["close"].rolling(250, min_periods=100).mean()
            df["ret"] = df["close"].pct_change()
            df["vol20"] = df["ret"].rolling(20, min_periods=10).std() * np.sqrt(252)
            # Q11-fix: 全样本 vol20.rank(pct=True) 含信号日之后的数据（前视）。
            # 分位计算改到 _regime_and_vol_at 内用截至信号日的数据切片重算。
            self._index_cache = df
            return df
        except Exception:
            self._index_cache = pd.DataFrame()
            return self._index_cache

    def _regime_and_vol_at(self, signal_date: pd.Timestamp, index_df: pd.DataFrame) -> tuple[str, str]:
        """判断信号发生日的市场状态与波动率状态。"""
        if index_df.empty:
            return "未知", "未知"
        try:
            # Q11-fix: normalize 后定位当日 bar；盘中信号（<15:00）市场状态
            # 用信号时点之前最后一个交易日（side="left"-1）；
            # 收盘后信号（>=15:00）可用当日收盘（side="right"-1）。
            day = signal_date.normalize()
            if signal_date.hour >= 15:
                pos = index_df.index.searchsorted(day, side="right") - 1
            else:
                pos = index_df.index.searchsorted(day, side="left") - 1
            if pos < 0 or pos >= len(index_df):
                return "未知", "未知"
            row = index_df.iloc[pos]

            ma250 = row.get("ma250", np.nan)
            close = row.get("close", np.nan)
            if np.isnan(ma250) or ma250 <= 0:
                regime = "未知"
            else:
                dev = close / ma250 - 1
                if dev > 0.05:
                    regime = "牛市"
                elif dev < -0.05:
                    regime = "熊市"
                else:
                    regime = "震荡市"

            # Q11-fix: 分位只在截至信号日的数据切片内重算（去掉全样本前视）
            vol_pctile = np.nan
            vol_hist = index_df["vol20"].iloc[: pos + 1].dropna()
            if len(vol_hist) >= 10:
                vol_pctile = float((vol_hist <= vol_hist.iloc[-1]).mean())
            if np.isnan(vol_pctile):
                vol_regime = "未知"
            elif vol_pctile < 0.33:
                vol_regime = "低波"
            elif vol_pctile > 0.67:
                vol_regime = "高波"
            else:
                vol_regime = "中波"

            return regime, vol_regime
        except Exception:
            return "未知", "未知"

    def _forward_return(self, signal_date: pd.Timestamp, days: int, index_df: pd.DataFrame) -> Optional[float]:
        """信号后 N 日的市场收益率。"""
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

    def analyze(self, signal_type: str, forward_days: int = 20) -> dict[str, Any]:
        """分析指定信号类型在不同市场条件下的胜率。

        Args:
            signal_type: 信号类型
            forward_days: 用于计算胜负的前瞻天数，默认20日

        Returns:
            {
                "signal_type": str,
                "conditions": [{"regime": str, "win_rate": float, "count": int}, ...],
                "volatility_regimes": [{"vol": str, "win_rate": float, "count": int}, ...],
                "best_regime_for_signal": str,
            }
        """
        try:
            signals = self.store.load(signal_type=signal_type)
            index_df = self._fetch_index_daily()

            if not signals:
                return {
                    "signal_type": signal_type,
                    "conditions": [],
                    "volatility_regimes": [],
                    "best_regime_for_signal": "未知",
                    "warning": "无历史信号数据",
                }

            regime_hits: dict[str, list[bool]] = {}
            vol_hits: dict[str, list[bool]] = {}
            combo_hits: dict[str, list[bool]] = {}

            # P2-Q11-fix(M048): 分析口径与存储语义对齐——默认排除已过期信号。
            # expiry 为空/解析失败视为永不过期；仅 expiry 明确早于当前时间才判定过期。
            now_naive = pd.Timestamp.now().tz_localize(None)
            for sig in signals:
                exp_raw = sig.get("expiry", "")
                if exp_raw:
                    try:
                        exp_dt = pd.Timestamp(exp_raw).tz_localize(None)
                        if exp_dt < now_naive:
                            continue
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[best_conditions] 操作失败: {e}", exc_info=True)

                ts = sig.get("timestamp", "")
                try:
                    sig_date = pd.Timestamp(ts).tz_localize(None)
                except Exception as e:
                    logging.getLogger(__name__).error(f"[best_conditions] 操作失败: {e}", exc_info=True)
                    continue

                direction_mult = -1.0 if sig.get("direction") == "bearish" else 1.0
                ret = self._forward_return(sig_date, forward_days, index_df)
                if ret is None:
                    continue
                win = (ret * direction_mult) > 0

                regime, vol_regime = self._regime_and_vol_at(sig_date, index_df)
                if regime != "未知":
                    regime_hits.setdefault(regime, []).append(win)
                if vol_regime != "未知":
                    vol_hits.setdefault(vol_regime, []).append(win)
                if regime != "未知" and vol_regime != "未知":
                    combo_key = f"{regime} + {vol_regime}"
                    combo_hits.setdefault(combo_key, []).append(win)

            conditions = [
                {
                    "regime": regime,
                    "win_rate": round(float(np.mean(hits)), 4),
                    "count": len(hits),
                }
                for regime, hits in regime_hits.items()
            ]
            conditions.sort(key=lambda x: x["win_rate"], reverse=True)

            volatility_regimes = [
                {
                    "vol": vol_regime,
                    "win_rate": round(float(np.mean(hits)), 4),
                    "count": len(hits),
                }
                for vol_regime, hits in vol_hits.items()
            ]
            volatility_regimes.sort(key=lambda x: x["win_rate"], reverse=True)

            best_combo = "未知"
            best_win_rate = -1.0
            for combo, hits in combo_hits.items():
                if len(hits) < 3:
                    continue  # 样本太少不纳入最佳组合评选
                wr = float(np.mean(hits))
                if wr > best_win_rate:
                    best_win_rate = wr
                    best_combo = combo

            return {
                "signal_type": signal_type,
                "conditions": conditions,
                "volatility_regimes": volatility_regimes,
                "best_regime_for_signal": best_combo,
                "best_combo_win_rate": round(best_win_rate, 4) if best_win_rate >= 0 else None,
            }
        except Exception as exc:
            return {"signal_type": signal_type, "error": str(exc)}


def main() -> None:
    """示例：写入模拟信号并分析其最佳市场条件。"""
    from datetime import datetime

    store = SignalStore()
    now = datetime.now(CST)

    for offset in [400, 350, 300, 260, 200, 150, 100, 60, 30]:
        store.save({
            "signal_type": "funding_sentiment",
            "direction": "bullish",
            "strength": 0.6,
            "target": "沪深300",
            "value": 0.5,
            "threshold": 0.4,
            "source_module": "sentiment_factory.funding_sentiment",
            "metadata": {},
            "timestamp": (now - timedelta(days=offset)).isoformat(),
            "expiry": (now - timedelta(days=offset - 5)).isoformat(),
        })

    bc = BestConditions(store)
    result = bc.analyze("funding_sentiment")

    print("═" * 55)
    print("  最佳市场条件分析: funding_sentiment")
    print("═" * 55)
    print("  市场状态分组:")
    for c in result.get("conditions", []):
        print(f"    {c['regime']:<6} 胜率{c['win_rate']:.1%}  样本{c['count']}")
    print("  波动率分组:")
    for v in result.get("volatility_regimes", []):
        print(f"    {v['vol']:<6} 胜率{v['win_rate']:.1%}  样本{v['count']}")
    print(f"  最佳组合: {result.get('best_regime_for_signal')}")


if __name__ == "__main__":
    main()
