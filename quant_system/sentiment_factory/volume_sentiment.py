"""
volume_sentiment — 基于成交量的情绪维度 (V5)

不是看放量/缩量本身，看量价结构：
- 放量上涨 vs 缩量上涨（缩量上涨 = 一致性高但后继乏力）
- 放量下跌 vs 缩量下跌（放量下跌 = 恐慌出逃）
- 成交量历史分位（当前量在1年/3年中的位置）
- 成交量与均量线的背离

核心问题：
- 放量不一定好（放量下跌是坏的，放量上涨是好的）
- 缩量不一定坏（缩量下跌是跌不动了）
- 关键在"量 + 价 + 方向"的组合判断
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent


class VolumeSentiment:
    """基于成交量的情绪维度。"""

    def __init__(self, cache_ttl: int = 300) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算当前量价情绪。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "score": 0,
            "direction": "neutral",
            "percentile": 50.0,
            "sub_signals": [],
            "metrics": {},
        }

        try:
            import akshare as ak

            # ── 获取沪深300量价数据 ──
            df = ak.stock_zh_index_daily(symbol="sh000300")
            if df is None or len(df) < 60:
                result["sub_signals"].append("⚠️ 量价数据不足(<60日)，按中性兜底")
                return result

            volume = df["volume"].values.astype(float)
            close = df["close"].values.astype(float)
            date = pd.to_datetime(df["date"])

            # ── 1. 当前成交量在历史中的分位 ──
            current_vol = volume[-1]
            vol_percentile = sum(volume < current_vol) / len(volume) * 100
            result["metrics"]["vol_percentile_1y"] = round(vol_percentile, 1)

            # ── 2. 成交量 vs 20日均量 ──
            vol_ma20 = np.mean(volume[-20:])
            vol_ma60 = np.mean(volume[-60:])
            vol_ratio_20 = current_vol / max(vol_ma20, 1)
            vol_ratio_60 = current_vol / max(vol_ma60, 1)
            result["metrics"]["vol_ratio_20"] = round(vol_ratio_20, 2)
            result["metrics"]["vol_ratio_60"] = round(vol_ratio_60, 2)

            # ── 3. 量价组合判断 ──
            ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
            ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

            # 5日量价分析
            vol_5d_avg = np.mean(volume[-5:])
            vol_20d_avg = np.mean(volume[-20:])
            vol_5d_vs_20d = vol_5d_avg / max(vol_20d_avg, 1)

            result["metrics"]["ret_5d"] = round(ret_5d, 2)
            result["metrics"]["ret_20d"] = round(ret_20d, 2)
            result["metrics"]["vol_5d_vs_20d"] = round(vol_5d_vs_20d, 2)

            # ── 4. 评分规则 ──
            score = 0.0

            # 放量上涨 = 积极
            if ret_5d > 2 and vol_5d_vs_20d > 1.2:
                score += 3
                result["sub_signals"].append(
                    f"放量上涨(5日{ret_5d:+.1f}%, 量比{vol_5d_vs_20d:.1f}x)"
                )
            # 缩量上涨 = 一致性高但后继乏力
            elif ret_5d > 2 and vol_5d_vs_20d < 0.8:
                score += 1
                result["sub_signals"].append(
                    f"缩量上涨(5日{ret_5d:+.1f}%, 量比{vol_5d_vs_20d:.1f}x) — 可能涨不动"
                )
            # 放量下跌 = 恐慌
            elif ret_5d < -2 and vol_5d_vs_20d > 1.3:
                score -= 3
                result["sub_signals"].append(
                    f"放量下跌(5日{ret_5d:+.1f}%, 量比{vol_5d_vs_20d:.1f}x) — 恐慌出逃"
                )
            # 缩量下跌 = 跌不动
            elif ret_5d < -2 and vol_5d_vs_20d < 0.7:
                score += 1
                result["sub_signals"].append(
                    f"缩量下跌(5日{ret_5d:+.1f}%, 量比{vol_5d_vs_20d:.1f}x) — 跌速放缓"
                )
            # 放量震荡 = 方向不明
            elif abs(ret_5d) < 1 and vol_5d_vs_20d > 1.3:
                score -= 1
                result["sub_signals"].append(
                    f"放量横盘(量比{vol_5d_vs_20d:.1f}x) — 多空分歧加大"
                )
            # 缩量横盘 = 观望
            elif abs(ret_5d) < 1 and vol_5d_vs_20d < 0.7:
                score += 0
                result["sub_signals"].append(
                    f"缩量横盘(量比{vol_5d_vs_20d:.1f}x) — 市场观望"
                )

            # P2-Q11-fix(L051): 极端成交量是叠加修正项，可与前面的量价结构分支
            # 同时计分，用于刻画情绪过热/冰点。
            # 量比极端值修正
            if vol_ratio_20 > 2.0:
                # 天量（不论方向）意味着情绪极端
                score += 1 if ret_5d > 0 else -1
                result["sub_signals"].append(
                    f"天量(当日量/20日均量{vol_ratio_20:.1f}x) — 情绪极端"
                )

            result["score"] = round(max(-10, min(10, score)), 1)

            # 方向
            if result["score"] >= 2:
                result["direction"] = "bullish"
            elif result["score"] <= -2:
                result["direction"] = "bearish"
            else:
                result["direction"] = "neutral"

            # 历史分位(基于量比的历史模拟)
            result["percentile"] = round(
                max(0, min(100, 50 + result["score"] * 6.8)), 1
            )

        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    vs = VolumeSentiment()
    result = vs.compute()
    print("═" * 50)
    print(f"  量价情绪评分: {result.get('score', 'N/A'):>+5}  |  "
          f"方向: {result.get('direction', 'N/A'):>8}  |  "
          f"分位: {result.get('percentile', 'N/A')}")
    print("═" * 50)
    for s in result.get("sub_signals", []):
        print(f"  • {s}")
    metrics = result.get("metrics", {})
    if metrics:
        print()
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
