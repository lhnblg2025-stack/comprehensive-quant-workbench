"""
composite — 多维情绪合成指数 (V5)

将 price_sentiment + volume_sentiment + funding_sentiment
合成为单一复合情绪指数，并检测各维度之间的背离。

核心价值：
  1. 单一维度可能误判（放量可能是好事也可能是坏事）
  2. 多个维度同时指向同一方向 = 高置信度
  3. 维度之间的背离 = 预警信号
  4. 综合指数 vs 大盘的背离 = 重要反转信号
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any


from .price_sentiment import PriceSentiment
from .volume_sentiment import VolumeSentiment
from .funding_sentiment import FundingSentiment

CST = timezone(timedelta(hours=8))


class CompositeSentiment:
    """多维情绪合成指数。"""

    def __init__(self) -> None:
        self.price = PriceSentiment()
        self.volume = VolumeSentiment()
        self.funding = FundingSentiment()
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl: int = 300

    def compute(self) -> dict[str, Any]:
        """计算复合情绪指数。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        # 获取各维度
        p = self.price.compute()
        v = self.volume.compute()
        f = self.funding.compute()

        ps = p.get("score", 0)
        vs = v.get("score", 0)
        fs = f.get("score", 0)

        # ── 维度背离检测 ──
        divergences = []
        scores = [ps, vs, fs]
        directions = [1 if s >= 1 else (-1 if s <= -1 else 0) for s in scores]
        n_bullish = sum(1 for d in directions if d == 1)
        n_bearish = sum(1 for d in directions if d == -1)

        # 检测两两背离
        labels = ["价格", "量价", "资金"]
        dim_scores = [ps, vs, fs]
        for i in range(3):
            for j in range(i + 1, 3):
                if directions[i] * directions[j] < 0:  # 方向相反
                    # P2-Q11-fix(L056): 先抽出分数变量再格式化，避免嵌套三元表达式
                    # 与格式说明符绑定不清。
                    left_score = dim_scores[i]
                    right_score = dim_scores[j]
                    divergences.append(f"{labels[i]}({left_score:+.0f}) "
                                       f"vs {labels[j]}({right_score:+.0f}) — 背离")

        # ── 综合评分（加权） ──
        # 价格情绪权重最高（最直接），量价其次，资金最低（有时滞）
        weights = {"price": 0.45, "volume": 0.35, "funding": 0.20}
        composite = ps * weights["price"] + vs * weights["volume"] + fs * weights["funding"]
        composite = round(max(-10, min(10, composite)), 1)

        # 置信度（基于一致性）
        if n_bullish >= 2 or n_bearish >= 2:
            confidence = "high"
            if n_bullish == 3 or n_bearish == 3:
                confidence = "very_high"
        elif len(divergences) > 0:
            confidence = "low"
        else:
            confidence = "medium"

        # ── 维度一致性 ──
        consistency = "一致" if n_bullish >= 2 or n_bearish >= 2 else (
            "分歧" if len(divergences) > 0 else "中性"
        )

        # ── 构建结果 ──
        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "composite_score": composite,
            "direction": ("bullish" if composite >= 2 else
                          "bearish" if composite <= -2 else "neutral"),
            "confidence": confidence,
            "consistency": consistency,
            "dimensions": {
                "price": {"score": ps, "direction": p.get("direction", "neutral"),
                          "percentile": p.get("percentile", 50)},
                "volume": {"score": vs, "direction": v.get("direction", "neutral"),
                           "percentile": v.get("percentile", 50)},
                "funding": {"score": fs, "direction": f.get("direction", "neutral"),
                            "percentile": f.get("percentile", 50)},
            },
            "divergences": divergences,
            "weights": weights,
            "signals": p.get("sub_signals", []) + v.get("sub_signals", []) + f.get("sub_signals", []),
            "alerts": [],
        }

        # 综合预警
        if divergences:
            result["alerts"].append(f"维度分歧: {len(divergences)}处背离")

        if p.get("divergence"):
            result["alerts"].append("价格指数背离")

        if composite >= 7:
            result["alerts"].append(f"综合情绪极度乐观({composite:+.0f}) — 警惕过热回调")
        elif composite <= -7:
            result["alerts"].append(f"综合情绪极度悲观({composite:+.0f}) — 关注反弹机会")

        # 分位
        result["composite_percentile"] = round(
            max(0, min(100, 50 + composite * 6.8)), 1
        )

        self.cache = result
        self.last_fetch = now
        return result

    def get_alerts(self) -> list[str]:
        """获取所有预警。"""
        data = self.compute()
        return data.get("alerts", [])


def main() -> None:
    cs = CompositeSentiment()
    result = cs.compute()

    print("═" * 60)
    print(f"  🌡️  合成情绪指数")
    print("═" * 60)

    c = result["composite_score"]
    dir_map = {"bullish": "📈 看多", "bearish": "📉 看空", "neutral": "➡️ 中性"}
    conf_map = {"high": "高", "very_high": "很高", "medium": "中", "low": "低"}

    print(f"  综合评分: {c:>+5}  |  方向: {dir_map.get(result['direction'], '?')}  "
          f"|  分位: {result['composite_percentile']:.0f}%")
    print(f"  置信度: {conf_map.get(result['confidence'], '?')}  "
          f"|  一致性: {result.get('consistency', '?')}")
    print()

    for name, dim in result["dimensions"].items():
        d = dim["direction"]
        arrow = "📈" if d == "bullish" else "📉" if d == "bearish" else "➡️"
        print(f"  {arrow} {name:<6}  {dim['score']:>+5}  "
              f"({dim.get('percentile', 50):.0f}分位)")

    print()
    for s in result["signals"]:
        print(f"  • {s}")

    if result["divergences"]:
        print()
        print(f"  ⚠️ 维度背离 (n={len(result['divergences'])})")
        for d in result["divergences"]:
            print(f"    ✗ {d}")

    for a in result["alerts"]:
        print(f"  🔔 {a}")


if __name__ == "__main__":
    main()
