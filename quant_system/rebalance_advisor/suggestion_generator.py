"""
suggestion_generator — 调仓建议生成 (V5)

基于目标权重 vs 当前权重的差异，生成人类可理解的调仓建议。

每个建议附带：
  - action: 买入/卖出/持有
  - reason: 为什么这么做
  - priority: high/medium/low
  - confidence: 0~1
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

CST = timezone(timedelta(hours=8))


class SuggestionGenerator:
    """调仓建议生成器。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}

    def generate(self, current_holdings: list[dict[str, Any]],
                 target_weights: list[dict[str, Any]],
                 constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        """生成调仓建议。

        Args:
            current_holdings: 当前持仓 [{"code", "weight", "name"}, ...]
            target_weights: 目标权重 [{"code", "target_weight", ...}, ...]
            constraints: 约束

        Returns:
            {
                "suggestions": [
                    {"action": "卖出", "code": "...", "name": "...",
                     "from_weight": 0.2, "to_weight": 0.15,
                     "amount_pct": 0.05, "reason": "...",
                     "priority": "high", "confidence": 0.8},
                ],
                "summary": {
                    "total_buy_pct": float,
                    "total_sell_pct": float,
                    "net_turnover": float,
                    "suggestion_count": int,
                }
            }
        """
        if constraints is None:
            constraints = {
                "max_single_weight": 0.15,
                "turnover_limit": 0.20,
            }

        # 构建 lookup
        target_map = {t["code"]: t for t in target_weights}
        current_map = {h["code"]: h for h in current_holdings}

        suggestions = []
        total_buy = 0.0
        total_sell = 0.0

        for code, target in target_map.items():
            current_w = target.get("current_weight", 0)
            target_w = target.get("target_weight", 0)
            diff = target_w - current_w

            if abs(diff) < 0.005:
                continue  # <0.5%差异不调

            name = target.get("name", current_map.get(code, {}).get("name", ""))
            max_single = constraints.get("max_single_weight", 0.15)

            # 确定动作
            if diff > 0:
                action = "买入"
                total_buy += diff
                # 理由
                reasons = []
                if current_w > max_single:
                    reasons.append("减仓到单票上限以下")
                if target_w - current_w > 0.02:
                    reasons.append("权重低于目标")

                if code not in current_map:
                    reasons.append("新加入组合")
                    action = "新买入"

                priority = "high" if abs(diff) > 0.03 else (
                    "medium" if abs(diff) > 0.01 else "low"
                )
                confidence = min(1.0, 0.5 + abs(diff) * 5)

            else:
                action = "卖出"
                total_sell -= diff
                reasons = []
                if target_w < constraints.get("min_single_weight", 0.01):
                    reasons.append("权重低于阈值，考虑清仓")
                elif current_w > max_single:
                    reasons.append(f"单票超限({current_w:.0%}>{max_single:.0%})")
                else:
                    reasons.append("减少暴露")

                priority = "high" if abs(diff) > 0.03 else (
                    "medium" if abs(diff) > 0.01 else "low"
                )
                confidence = min(1.0, 0.5 + abs(diff) * 5)

            if not reasons:
                reasons.append("权重调整")

            suggestions.append({
                "action": action,
                "code": code,
                "name": name,
                "from_weight": round(current_w, 4),
                "to_weight": round(target_w, 4),
                "amount_pct": round(abs(diff), 4),
                "reason": "; ".join(reasons),
                "priority": priority,
                "confidence": round(confidence, 2),
            })

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (priority_order.get(s["priority"], 9),
                                        -s["amount_pct"]))

        return {
            "timestamp": datetime.now(CST).isoformat(),
            "suggestions": suggestions,
            "summary": {
                "total_buy_pct": round(total_buy, 4),
                "total_sell_pct": round(total_sell, 4),
                "net_turnover": round(max(total_buy, total_sell), 4),
                "suggestion_count": len(suggestions),
                "high_priority_count": sum(1 for s in suggestions if s["priority"] == "high"),
            },
        }


def main() -> None:
    sg = SuggestionGenerator()

    current = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20},
        {"code": "000858", "name": "五粮液", "weight": 0.12},
    ]

    target = [
        {"code": "600519", "name": "贵州茅台", "current_weight": 0.20, "target_weight": 0.15},
        {"code": "000858", "name": "五粮液", "current_weight": 0.12, "target_weight": 0.10},
        {"code": "300750", "name": "宁德时代", "current_weight": 0.0, "target_weight": 0.08},
    ]

    result = sg.generate(current, target)

    print("═" * 60)
    print("  调仓建议")
    print("═" * 60)
    s = result["summary"]
    print(f"\n  买入: {s['total_buy_pct']:.1%}  |  卖出: {s['total_sell_pct']:.1%}  "
          f"|  换手: {s['net_turnover']:.1%}")
    print(f"  建议数: {s['suggestion_count']} (高优先级: {s['high_priority_count']})")
    print()
    for sug in result["suggestions"]:
        pri = {"high": "🔥", "medium": "⚡", "low": "💡"}
        print(f"  {pri.get(sug['priority'], '•')} {sug['action']} {sug['name']} "
              f"({sug['from_weight']:.0%} → {sug['to_weight']:.0%})")
        print(f"    {sug['reason']}")


if __name__ == "__main__":
    main()
