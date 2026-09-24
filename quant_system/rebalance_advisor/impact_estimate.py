"""
impact_estimate — 调仓冲击成本估计 (V5)

给定调仓方案，估算：
  1. 交易成本（佣金+印花税+过户费）
  2. 市场冲击（大单买入/卖出对价格的短期影响）
  3. 总成本（bps）

逻辑：
   - 佣金: 万0.85, 最低5元/笔 (A股规则, 无免五)
   - 印花税: 万5 (仅卖出)
   - 过户费: 万0.1 (双边, 与 trading_system/execution_broker 口径一致)
   - 市场冲击: ADV% 为基础的平方根模型

单位约定：
   - suggestions[].amount_pct 为权重小数（如 0.05 表示组合 5%）
   - 金额换算: amount_yuan = amount_pct * 组合总市值（由 holdings 汇总）
   - 冲击模型输入统一为亿元
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_system import execution_broker as _execution_broker


class ImpactEstimator:
    """冲击成本估算器。"""

    def __init__(self) -> None:
        # D3/D组收敛: 费率唯一真源为 execution_broker 常量（同数值同方向），此处改为引用。
        # P1-Q28-fix: 佣金率 10 倍错误修正。原 0.00085=万8.5 与全系统标准
        #   （config.py:13、trading_system.py:94 均 0.000085=万0.85）不一致，
        #   导致换手 13% 场景佣金高估 10 倍（1.1bps → 0.11bps）。
        self.commission_rate = _execution_broker.COMMISSION_RATE      # 万0.85
        self.stamp_tax_rate = _execution_broker.STAMP_TAX_RATE        # 万5 (仅卖出)
        # P2-Q28-fix(M356/M365): 最低佣金 0.1→5.0（A股规则，无免五），与
        # config.PortfolioConfig.min_commission=5.0 对齐；新增过户费 万0.1 双边，
        # 与 trading_system.DEFAULT_TRANSFER_FEE_RATE 口径一致。
        self.min_commission = _execution_broker.MIN_COMMISSION        # A股最低佣金 5元/笔
        self.transfer_fee_rate = _execution_broker.TRANSFER_FEE_RATE  # 过户费: 万0.1, 双边收取
        self.adv_estimation = 100_000_000  # 默认日均成交额(元)

    def estimate_impact(self, amount_yi: float, adv_yi: float | None = None) -> float:
        """估算市场冲击成本 (bps).

        使用平方根冲击模型：
          impact_bps = 0.1 * sqrt(amount / adv) * 10000

        Args:
            amount_yi: 交易金额（亿元）
            adv_yi: 日均成交额（亿元），None则用默认

        Returns:
            冲击成本 (bps)
        """
        adv = adv_yi or self.adv_estimation / 1e8
        if adv <= 0 or amount_yi <= 0:
            return 0.0
        participation = amount_yi / adv
        # 平方根冲击模型
        impact_bps = 0.1 * np.sqrt(participation) * 100
        return round(min(impact_bps, 50.0), 2)  # 最大50bps

    def compute(self, holdings: list[dict[str, Any]],
                suggestions: list[dict[str, Any]]) -> dict[str, Any]:
        """估算调仓总成本。

        Args:
            holdings: 当前持仓（含市值）
            suggestions: 调仓建议列表

        Returns:
            {
                "total_cost_bps": float,       # 总成本(bps)
                "commission_bps": float,
                "stamp_tax_bps": float,
                "market_impact_bps": float,
                "estimated_cost_yi": float,     # 估计成本(亿元)
                "impact_stocks": [...]          # 冲击较大的个股
            }
        """
        # P2-Q28-fix(M356): 从 holdings 汇总组合总市值（原硬编码 1 亿元且
        # holdings 从未使用）。无市值字段时可见降级回退 1 亿元并在 assumptions 记录。
        total_value = 0.0
        adv_map: dict[str, float] = {}
        for h in holdings:
            val = h.get("total_value") or h.get("market_value") or h.get("value") or 0.0
            total_value += float(val)
            code = h.get("code", "")
            adv = h.get("adv") or h.get("avg_daily_amount") or h.get("avg_amount") or 0.0
            adv_map[code] = float(adv)

        fallback_value = total_value <= 0
        if fallback_value:
            total_value = 100_000_000.0  # 1 亿元（可见降级）

        turnover_sell = 0.0   # 卖出金额(元)
        turnover_buy = 0.0    # 买入金额(元)
        commission_yi = 0.0   # 亿元
        stamp_yi = 0.0        # 亿元
        transfer_yi = 0.0     # 亿元
        impact_stocks = []

        for sug in suggestions:
            amount_pct = sug.get("amount_pct", 0)   # 权重小数
            amount_yuan = amount_pct * total_value  # P2-Q28-fix: 权重→金额(元)
            action = sug.get("action", "")

            if "卖出" in action or "卖" in action:
                turnover_sell += amount_yuan
            if "买入" in action or "买" in action:
                turnover_buy += amount_yuan

            # P2-Q28-fix(M356): 每笔佣金应用最低 5 元（原 min_commission 定义后从未应用）
            if amount_yuan > 0:
                commission_yi += max(amount_yuan * self.commission_rate,
                                     self.min_commission) / 1e8
                if "卖出" in action or "卖" in action:
                    stamp_yi += amount_yuan * self.stamp_tax_rate / 1e8
                # P2-Q28-fix(M365): 过户费万0.1 双边收取
                transfer_yi += amount_yuan * self.transfer_fee_rate / 1e8

            # 个股冲击估算（单位统一为亿元；adv 从 holdings 取，缺失用默认）
            code = sug.get("code", "")
            adv_yuan = adv_map.get(code, 0.0) or self.adv_estimation
            adv_yi = adv_yuan / 1e8
            amount_yi = amount_yuan / 1e8
            impact_bps = self.estimate_impact(amount_yi, adv_yi)
            if impact_bps > 10:
                impact_stocks.append({
                    "code": code,
                    "name": sug.get("name", ""),
                    "turnover_pct": amount_pct,
                    "estimated_impact_bps": impact_bps,
                })

        # 总成本（bps 相对组合总市值）
        total_value_yi = total_value / 1e8
        commission_bps = commission_yi / total_value_yi * 10000 if total_value_yi > 0 else 0
        stamp_bps = stamp_yi / total_value_yi * 10000
        transfer_bps = transfer_yi / total_value_yi * 10000
        impact_bps_total = sum(s["estimated_impact_bps"] for s in impact_stocks)
        impact_yi = impact_bps_total / 10000 * total_value_yi

        return {
            "total_cost_bps": round(commission_bps + stamp_bps + transfer_bps
                                    + impact_bps_total, 2),
            "commission_bps": round(commission_bps, 2),
            "stamp_tax_bps": round(stamp_bps, 2),
            "transfer_fee_bps": round(transfer_bps, 2),
            "market_impact_bps": round(impact_bps_total, 2),
            "estimated_cost_yi": round(commission_yi + stamp_yi + transfer_yi
                                       + impact_yi, 6),
            "assumptions": {
                "commission_rate": self.commission_rate,
                "min_commission": self.min_commission,
                "stamp_tax_rate": self.stamp_tax_rate,
                "transfer_fee_rate": self.transfer_fee_rate,
                "total_portfolio_yi": round(total_value_yi, 6),
                "total_value_from_holdings": not fallback_value,
            },
            "impact_stocks": sorted(impact_stocks,
                                    key=lambda x: x["estimated_impact_bps"],
                                    reverse=True)[:5],
        }


def main() -> None:
    ie = ImpactEstimator()
    suggestions = [
        {"action": "卖出", "code": "600519", "name": "贵州茅台",
         "amount_pct": 0.05},
        {"action": "买入", "code": "300750", "name": "宁德时代",
         "amount_pct": 0.08},
    ]
    holdings = [{"code": "600519", "name": "贵州茅台", "weight": 0.20}]

    result = ie.compute(holdings, suggestions)
    print("═" * 55)
    print("  调仓成本估算")
    print("═" * 55)
    print(f"  总成本:    {result['total_cost_bps']:.1f} bps")
    # P1-Q28-fix: 与佣金率同步，用两位小数打印万0.85（原 .1f 会把 0.85 显示成万0.9/万8.5 自相矛盾）
    print(f"  佣金:      {result['commission_bps']:.2f} bps (万{ie.commission_rate*10000:.2f})")
    print(f"  印花税:    {result['stamp_tax_bps']:.1f} bps (万{ie.stamp_tax_rate*10000:.0f},仅卖)")
    print(f"  市场冲击:  {result['market_impact_bps']:.1f} bps")
    print(f"  估算成本:  {result['estimated_cost_yi']:.4f} 亿")
    print()
    if result.get("impact_stocks"):
        print("  ⚠️ 冲击较大个股:")
        for s in result["impact_stocks"]:
            print(f"    {s['name']}: 冲击{s['estimated_impact_bps']:.0f}bps")


if __name__ == "__main__":
    main()
