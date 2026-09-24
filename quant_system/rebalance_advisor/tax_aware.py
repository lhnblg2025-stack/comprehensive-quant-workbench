"""
tax_aware — 税收优化调仓 (V5)

A股主要交易成本：
  1. 印花税: 万5 (仅卖出, 2023年8月下调后)
  2. 红利税: 持股<1月20%, 1月-1年10%, >1年0%
  3. 佣金: 万0.85 (最低0.1元)  # D3/D组收敛: 异费率口径保留——本模块不实现佣金，
     docstring 沿用历史 0.1 元说明，不以 execution_broker 5 元口径为准

优化逻辑：
  - 卖出时尽量选持有超过1年的（红利税优惠）
  - 如果需要减仓某个行业，优先减高成本个股
  - 避免为了微调而触发短期卖出（印花税成本高）
"""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any

from quant_system import execution_broker as _execution_broker

CST = timezone(timedelta(hours=8))


class TaxAwareRebalancer:
    """税收优化调仓。"""

    def __init__(self) -> None:
        # D3/D组收敛: 印花税费率引用 execution_broker 唯一真源（同数值同方向）。
        self.stamp_duty_rate = _execution_broker.STAMP_TAX_RATE
        self.dividend_tax_short = 0.20   # <1月
        self.dividend_tax_medium = 0.10  # 1月-1年
        self.dividend_tax_long = 0.00    # >1年

    def optimize_sell_order(self, holdings: list[dict[str, Any]],
                            sell_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """优化卖出顺序：优先卖出税收成本最低的股票。

        Args:
            holdings: 当前持仓（含成本信息）
            sell_targets: 需要减仓的股票及金额

        Returns:
            排序后的卖出建议（税收成本从低到高）
        """
        enhanced = []
        for target in sell_targets:
            code = target.get("code", "")
            hold_info = next((h for h in holdings if h.get("code") == code), None)
            if hold_info:
                # 估算持有期
                cost_basis = hold_info.get("cost_price", 0)
                current_price = hold_info.get("current_price", cost_basis)
                unrealized_pnl = (current_price / max(cost_basis, 1) - 1) if cost_basis > 0 else 0
                hold_days = hold_info.get("hold_days", 0)

                # 税收成本评分（越低越好卖）
                if hold_days > 365:
                    tax_score = 0.0      # 长期持有, 无红利税
                elif hold_days > 30:
                    tax_score = 0.001    # 中期, 少量红利税
                else:
                    tax_score = 0.002    # 短期

                # 加印花税（全部卖出都要交）
                tax_score += self.stamp_duty_rate

                # P2-Q28-fix(M357): 移除亏损卖出 `tax_score *= 0.9` 随意系数。
                # A股无资本利得税，亏损卖出既不抵税、印花税也照收，原 0.9 系数
                # 无依据且注释自相矛盾（先承认无税，又假设"心理成本低"）。
                # 税收评分仅反映真实可量化的税负：印花税 + 红利税（按持有期）。

                enhanced.append({
                    **target,
                    "tax_score": round(tax_score, 6),
                    "hold_days": hold_days,
                    "unrealized_pnl": round(unrealized_pnl, 4),
                })
            else:
                enhanced.append({**target, "tax_score": self.stamp_duty_rate,
                                 "hold_days": 0, "unrealized_pnl": 0})

        # 税收成本从低到高排序
        enhanced.sort(key=lambda x: (x["tax_score"], -x.get("amount_pct", 0)))
        return enhanced

    def suggest(self, suggestions: list[dict[str, Any]],
                 holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对调仓建议进行税收优化。

        Args:
            suggestions: 原始调仓建议
            holdings: 当前持仓信息

        Returns:
            优化后的调仓建议（可能调整了卖出顺序、合并了小单）
        """
        if not suggestions:
            return suggestions

        # 提取卖出建议
        sell_sugs = [s for s in suggestions if "卖出" in s.get("action", "")]
        buy_sugs = [s for s in suggestions if "买入" in s.get("action", "")]

        # 优化卖出顺序
        if sell_sugs:
            sell_targets = []
            for s in sell_sugs:
                hold_info = next(
                    (h for h in holdings if h.get("code") == s.get("code")),
                    None
                )
                sell_targets.append({
                    "code": s["code"],
                    "name": s.get("name", ""),
                    "amount_pct": s.get("amount_pct", 0),
                    "from_weight": s.get("from_weight", 0),
                    "to_weight": s.get("to_weight", 0),
                    "cost_price": hold_info.get("cost_price", 0) if hold_info else 0,
                    "current_price": hold_info.get("current_price", 0) if hold_info else 0,
                    "hold_days": hold_info.get("hold_days", 0) if hold_info else 0,
                })

            optimized_sells = self.optimize_sell_order(holdings, sell_targets)

            # 合并回建议列表
            sell_map = {s["code"]: s for s in optimized_sells}
            result = []
            for s in suggestions:
                if "卖出" in s.get("action", "") and s["code"] in sell_map:
                    os = sell_map[s["code"]]
                    s["tax_optimized"] = True
                    s["tax_score"] = os.get("tax_score", 0)
                    s["hold_days"] = os.get("hold_days", 0)
                result.append(s)
        else:
            result = list(suggestions)

        return result

    def estimate_tax_saved(self, original: list[dict],
                           optimized: list[dict]) -> dict:
        """估算税收优化节省的成本。

        P2-Q28-fix(M357): 原为硬编码桩函数（恒返回 0.1bps）。现按 amount_pct 加权
        平均费率差计算：对每条卖出建议，费率取 tax_score（若已由 optimize_sell_order
        标注）否则回退为印花税率。返回相对调仓金额的加权平均费率差（bps）。

        A股无资本利得税，节税仅来自卖出顺序优化（优先卖长期持仓以降低红利税），
        因此节省通常很小。
        """
        def _weighted_avg_rate(items: list[dict]) -> float:
            total_amount = sum(abs(s.get("amount_pct", 0)) for s in items)
            if total_amount <= 1e-12:
                return 0.0
            cost = 0.0
            for s in items:
                amt = abs(s.get("amount_pct", 0))
                if amt <= 0:
                    continue
                ts = s.get("tax_score")
                if ts is None:
                    ts = self.stamp_duty_rate  # 未标注的卖出建议按印花税回退
                cost += float(ts) * amt
            return cost / total_amount

        try:
            orig_avg = _weighted_avg_rate(original or [])
            opt_avg = _weighted_avg_rate(optimized or [])
            saved_bps = round(max(orig_avg - opt_avg, 0.0) * 10000, 4)
        except Exception:
            saved_bps = 0.0
        return {
            "tax_saved_bps": saved_bps,
            "note": "A股无资本利得税，节税主要来自卖出顺序（红利税持有期）优化；"
                    "此处为基于 amount_pct 加权的平均费率差(bps)",
        }


def main() -> None:
    tar = TaxAwareRebalancer()

    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20,
         "cost_price": 1500, "current_price": 1800, "hold_days": 400},
        {"code": "000858", "name": "五粮液", "weight": 0.12,
         "cost_price": 120, "current_price": 130, "hold_days": 20},
    ]

    sell_targets = [
        {"code": "000858", "name": "五粮液", "amount_pct": 0.03},
        {"code": "600519", "name": "贵州茅台", "amount_pct": 0.05},
    ]

    optimized = tar.optimize_sell_order(holdings, sell_targets)

    print("═" * 55)
    print("  税收优化卖出顺序")
    print("═" * 55)
    print(f"  {'股票':<12} {'税率':>8} {'持有期':>8} {'盈亏':>8}")
    for s in optimized:
        pnl_str = f"{s['unrealized_pnl']:+.0%}" if s.get("unrealized_pnl") else "N/A"
        print(f"  {s['name']:<12} {s['tax_score']:>8.4f} "
              f"{s['hold_days']:>4d}天 {pnl_str}")


if __name__ == "__main__":
    main()
