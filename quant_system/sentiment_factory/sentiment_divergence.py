"""
sentiment_divergence — 情绪 vs 价格背离检测 (V5)

核心价值：当市场情绪和价格走势不一致时，往往意味着反转。
  1. 情绪上升 + 价格下跌 = 底部背离（买入信号）
  2. 情绪下降 + 价格上涨 = 顶部背离（卖出信号）
  3. 情绪与价格同步 = 趋势确认

不简单比较"情绪分数 vs 涨跌幅"，
而是比较"情绪的多维形态 vs 价格的客观走势"。
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scipy.stats import percentileofscore  # Q11-fix: 原文件未导入 stats，L77 NameError 导致整个背离检测失效

CST = timezone(timedelta(hours=8))

ROOT = Path(__file__).resolve().parent.parent


class SentimentDivergence:
    """情绪与价格背离检测。

    检测三种背离：
      1. 常规背离 — 情绪 vs 指数的走势分歧
      2. 极端背离 — 情绪分位 vs 价格分位的巨大差异
      3. 加速背离 — 情绪变化方向与价格变化方向持续相反
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """检测当前的情绪-价格背离。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "has_divergence": False,
            "divergences": [],
            "alerts": [],
            "score": 0,
        }

        try:
            # ── 1. 获取情绪分数 ──
            from quant_system.sentiment_factory.composite import CompositeSentiment
            cs = CompositeSentiment()
            sentiment = cs.compute()
            sentiment_score = sentiment.get("composite_score", 0)
            sentiment_pct = sentiment.get("composite_percentile", 50)

            result["sentiment_score"] = sentiment_score
            result["sentiment_percentile"] = sentiment_pct

            # ── 2. 获取指数走势 ──
            import akshare as ak
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            if idx is None or len(idx) < 60:
                return result

            close = idx["close"].values.astype(float)
            ret_20d = (close[-1] / close[-21] - 1) * 100 if len(idx) > 20 else 0
            ret_60d = (close[-1] / close[-61] - 1) * 100 if len(idx) > 60 else 0

            # 指数历史分位
            # Q11-fix: 原代码 stats.percentileofscore(...) 引用未导入的 stats（NameError），
            # 顶部已改 from scipy.stats import percentileofscore，此处直接用导入名。
            price_pct = percentileofscore(close, close[-1])
            result["index_return_20d"] = round(ret_20d, 2)
            result["index_return_60d"] = round(ret_60d, 2)
            result["index_percentile"] = round(float(price_pct), 1)

            # ── 3. 背离检测 ──

            # 常规背离: 情绪分数 vs 20日收益
            if sentiment_score >= 3 and ret_20d < -3:
                result["has_divergence"] = True
                d = {
                    "type": "底部背离",
                    "detail": f"情绪乐观({sentiment_score:+.0f})但指数下跌({ret_20d:+.1f}%)",
                    "implication": "情绪领先价格, 可能即将反弹",
                    "severity": "bullish"
                }
                result["divergences"].append(d)
                result["alerts"].append(f"🟢 底部背离: 情绪{sentiment_score:+.0f} vs 指数{ret_20d:+.1f}%")
                result["score"] += 3

            elif sentiment_score <= -3 and ret_20d > 3:
                result["has_divergence"] = True
                d = {
                    "type": "顶部背离",
                    "detail": f"情绪悲观({sentiment_score:+.0f})但指数上涨({ret_20d:+.1f}%)",
                    "implication": "情绪不认可这个上涨, 警惕回调",
                    "severity": "bearish"
                }
                result["divergences"].append(d)
                result["alerts"].append(f"🔴 顶部背离: 情绪{sentiment_score:+.0f} vs 指数{ret_20d:+.1f}%")
                result["score"] -= 3

            # 极端背离: 情绪分位 vs 价格分位差距 > 40
            pct_diff = sentiment_pct - result["index_percentile"]
            if abs(pct_diff) > 40:
                result["has_divergence"] = True
                d = {
                    "type": "极端分位背离",
                    "detail": (f"情绪分位{sentiment_pct:.0f}% vs "
                               f"指数分位{result['index_percentile']:.0f}%"),
                    "implication": "情绪和价格分期巨大, 市场观点分裂",
                    "severity": "warning"
                }
                result["divergences"].append(d)

            # 加速背离: 趋势方向分歧
            if sentiment_score > 0 and ret_60d < 0:
                result["alerts"].append(
                    f"⚠️ 趋势背离: 60日指数下跌{ret_60d:+.1f}%但情绪偏多"
                )
            elif sentiment_score < 0 and ret_60d > 0:
                result["alerts"].append(
                    f"⚠️ 趋势背离: 60日指数上涨{ret_60d:+.1f}%但情绪偏空"
                )

            result["score"] = round(max(-10, min(10, result["score"])), 1)

        except ImportError as e:
            # W2.5 修复: scipy 缺失不得静默产出"无明显背离"假绿
            result["error"] = f"scipy 缺失，背离计算不可用: {e}"
            result["unavailable"] = True
        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    sd = SentimentDivergence()
    r = sd.compute()
    print("═" * 55)
    print(f"  情绪-价格背离检测")
    print("═" * 55)
    if r.get("has_divergence"):
        print(f"  ⚡ 检测到 {len(r['divergences'])} 处背离")
        for d in r["divergences"]:
            marker = "🟢" if d.get("severity") == "bullish" else "🔴" if d.get("severity") == "bearish" else "⚠️"
            print(f"  {marker} {d['type']}")
            print(f"    {d['detail']}")
            print(f"    → {d['implication']}")
    else:
        print("  ✅ 无明显背离")
    print(f"\n  情绪分位: {r.get('sentiment_percentile', '?'):>5}")
    print(f"  指数分位: {r.get('index_percentile', '?'):>5}")
    print(f"  20日收益: {r.get('index_return_20d', '?'):>+}")
    for a in r.get("alerts", []):
        print(f"  {a}")


if __name__ == "__main__":
    main()
