"""
report — 综合调仓报告 (V5)

整合 optimizer + suggestion_generator + impact_estimate + tax_aware
输出人类可读的 Markdown 调仓报告。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .optimizer import PortfolioOptimizer
from .suggestion_generator import SuggestionGenerator
from .impact_estimate import ImpactEstimator
from .tax_aware import TaxAwareRebalancer

CST = timezone(timedelta(hours=8))


class RebalanceReport:
    """综合调仓报告生成。"""

    def __init__(self) -> None:
        self.optimizer = PortfolioOptimizer()
        self.suggester = SuggestionGenerator()
        self.impact = ImpactEstimator()
        self.tax = TaxAwareRebalancer()

    def generate_markdown(self, holdings: list[dict[str, Any]],
                          constraints: dict[str, Any] | None = None) -> str:
        """生成 Markdown 调仓报告。

        Args:
            holdings: 当前持仓
            constraints: 约束

        Returns:
            markdown 报告字符串
        """
        now = datetime.now(CST).strftime("%Y-%m-%d %H:%M")

        # 获取优化结果
        opt_result = self.optimizer.compute(holdings, constraints)
        target_weights = opt_result.get("target_weights", [])

        # 生成建议
        sug_result = self.suggester.generate(holdings, target_weights, constraints)

        # 成本估算
        cost_result = self.impact.compute(holdings, sug_result.get("suggestions", []))

        # 税收优化
        tax_optimized = self.tax.suggest(sug_result.get("suggestions", []), holdings)

        lines = []
        lines.append(f"# 组合再平衡报告")
        lines.append(f"**生成时间**: {now}")
        lines.append(f"**持仓数量**: {len(holdings)} 只")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append("## 摘要")
        lines.append(f"")
        s = sug_result.get("summary", {})
        lines.append(f"- 建议数: {s.get('suggestion_count', 0)} 条")
        lines.append(f"- 高优先级: {s.get('high_priority_count', 0)} 条")
        lines.append(f"- 买入: {s.get('total_buy_pct', 0):.1%} | 卖出: {s.get('total_sell_pct', 0):.1%}")
        lines.append(f"- 换手率: {opt_result.get('turnover', 0):.1%}")
        lines.append(f"- 总成本: {cost_result.get('total_cost_bps', 0):.1f} bps")
        lines.append(f"")
        lines.append("## 调仓建议明细")
        lines.append(f"")
        lines.append("| 优先级 | 动作 | 股票 | 当前权重 | 目标权重 | 变动 | 理由 |")
        lines.append("|--------|------|------|----------|----------|------|------|")

        priority_emoji = {"high": "🔥", "medium": "⚡", "low": "💡"}

        for sug in tax_optimized:
            pri = priority_emoji.get(sug.get("priority", "low"), "•")
            action = sug.get("action", "")
            change = sug.get("amount_pct", 0)
            ch_str = f"+{change:.1%}" if change > 0 else f"-{change:.1%}"
            lines.append(
                f"| {pri} | {action} | {sug.get('name', '')} "
                f"| {sug.get('from_weight', 0):.1%} "
                f"| {sug.get('to_weight', 0):.1%} "
                f"| {ch_str} | {sug.get('reason', '')} |"
            )

        lines.append(f"")
        lines.append("## 成本分析")
        lines.append(f"")
        lines.append(f"- 佣金: {cost_result.get('commission_bps', 0):.1f} bps")
        lines.append(f"- 印花税: {cost_result.get('stamp_tax_bps', 0):.1f} bps (仅卖出)")
        lines.append(f"- 市场冲击: {cost_result.get('market_impact_bps', 0):.1f} bps")
        lines.append(f"- 合计: {cost_result.get('total_cost_bps', 0):.1f} bps")
        lines.append(f"")

        if cost_result.get("impact_stocks"):
            lines.append("### ⚠️ 高冲击个股")
            for s in cost_result["impact_stocks"]:
                lines.append(f"- {s.get('name', '')}: 预估冲击 {s.get('estimated_impact_bps', 0):.0f} bps")
            lines.append(f"")

        lines.append("## 约束条件")
        if constraints:
            lines.append(f"")
            for k, v in constraints.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.0%}")
                else:
                    lines.append(f"- {k}: {v}")
        else:
            lines.append("*默认约束*")
        lines.append(f"")

        lines.append("---")
        lines.append("*本报告由 V5 RebalanceAdvisor 自动生成，仅供参考*")

        return "\n".join(lines)

    def generate_simple(self, holdings: list[dict[str, Any]],
                        constraints: dict[str, Any] | None = None) -> dict:
        """生成结构化报告（非 Markdown）。"""
        opt_result = self.optimizer.compute(holdings, constraints)
        sug_result = self.suggester.generate(holdings,
                                              opt_result.get("target_weights", []),
                                              constraints)
        return {
            "timestamp": datetime.now(CST).isoformat(),
            "optimization": opt_result,
            "suggestions": sug_result,
        }


def main() -> None:
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20},
        {"code": "000858", "name": "五粮液", "weight": 0.12},
        {"code": "300750", "name": "宁德时代", "weight": 0.10},
        {"code": "601318", "name": "中国平安", "weight": 0.08},
    ]

    rr = RebalanceReport()
    report = rr.generate_markdown(holdings)
    print(report)


if __name__ == "__main__":
    main()
