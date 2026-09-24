"""
report — 持仓综合诊断报告 (V5)

整合 exposure / attribution / concentration / drawdown_analyzer 四个模块的
输出，生成一份人类可读的 Markdown 诊断报告，回答：
  我的持仓因子暴露如何？收益从哪来？是否过度集中？最大回撤谁背锅？

对标：券商/机构的组合诊断报告（一页纸摘要 + 分节详情）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .attribution import PnLAttribution
from .concentration import ConcentrationAnalysis
from .drawdown_analyzer import DrawdownAnalyzer
from .exposure import FactorExposure

CST = timezone(timedelta(hours=8))

ROOT = Path(__file__).resolve().parent.parent


class DiagnosticReport:
    """持仓综合诊断报告生成器。

    整合四个诊断子模块的输出，生成一份带结论性摘要的 Markdown 报告：
      1. 因子暴露 (FactorExposure) — 隐藏的因子敞口
      2. PnL 拆解 (PnLAttribution) — 收益来源
      3. 集中度 (ConcentrationAnalysis) — 分散化程度
      4. 回撤归因 (DrawdownAnalyzer) — 最大回撤的锅由谁背

    典型用法::

        dr = DiagnosticReport()
        markdown_text = dr.generate(holdings)

    Attributes:
        cache: 上一次生成结果缓存
        last_fetch: 上次生成的时间戳
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl
        self.exposure_engine = FactorExposure()
        self.attribution_engine = PnLAttribution()
        self.concentration_engine = ConcentrationAnalysis()
        self.drawdown_engine = DrawdownAnalyzer()

    def _section_exposure(self, holdings: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """生成因子暴露小节文本。"""
        try:
            result = self.exposure_engine.compute(holdings)
        except Exception as e:
            result = {"error": str(e)}

        lines = ["## 1. 因子暴露\n"]
        if result.get("error"):
            lines.append(f"⚠️ 计算失败: {result['error']}\n")
            return "\n".join(lines), result

        exposures = result.get("exposures", {})
        lines.append("| 因子 | 暴露值 |")
        lines.append("|---|---|")
        for f, v in sorted(exposures.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"| {f} | {v:+.3f} |")

        hidden_bets = result.get("hidden_bets", [])
        if hidden_bets:
            lines.append("\n**⚠️ 非故意因子偏离:**\n")
            for bet in hidden_bets:
                lines.append(f"- {bet['factor']}: {bet['exposure']:+.2f} ({bet['direction']})")

        return "\n".join(lines), result

    def _section_attribution(
        self, holdings: list[dict[str, Any]], start_date: str, end_date: str
    ) -> tuple[str, dict[str, Any]]:
        """生成 PnL 拆解小节文本。"""
        try:
            result = self.attribution_engine.compute(holdings, start_date, end_date)
        except Exception as e:
            result = {"error": str(e)}

        lines = ["## 2. 收益拆解 (PnL Attribution)\n"]
        if result.get("error"):
            lines.append(f"⚠️ 计算失败: {result['error']}\n")
            return "\n".join(lines), result

        lines.append(f"- 区间: {result['period']['start']} ~ {result['period']['end']}")
        lines.append(f"- 组合总收益: **{result['total_return']:+.2f}%**")
        lines.append(f"- 基准收益(沪深300): {result['benchmark_return']:+.2f}%")
        lines.append(f"- 超额收益: **{result['active_return']:+.2f}%**")

        attr = result.get("attribution", {})
        lines.append("\n| 拆解项 | 贡献 |")
        lines.append("|---|---|")
        lines.append(f"| Beta贡献 | {attr.get('beta_return', 0):+.2f}% |")
        lines.append(f"| Alpha贡献 | {attr.get('alpha_return', 0):+.2f}% |")
        lines.append(f"| 行业配置效应 | {attr.get('sector_allocation', 0):+.2f}% |")
        lines.append(f"| 个股选择效应 | {attr.get('stock_selection', 0):+.2f}% |")

        top = result.get("top_contributors", [])[:5]
        bottom = result.get("bottom_contributors", [])[:5]
        if top:
            lines.append("\n**正向贡献 Top 5:**\n")
            for c in top:
                lines.append(f"- {c['name']}: {c['contribution']:+.3f}% (权重{c['weight']:.1%})")
        if bottom:
            lines.append("\n**负向贡献 Top 5:**\n")
            for c in bottom:
                lines.append(f"- {c['name']}: {c['contribution']:+.3f}% (权重{c['weight']:.1%})")

        return "\n".join(lines), result

    def _section_concentration(self, holdings: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """生成集中度小节文本。"""
        try:
            result = self.concentration_engine.compute(holdings)
        except Exception as e:
            result = {"error": str(e)}

        lines = ["## 3. 集中度分析\n"]
        if result.get("error"):
            lines.append(f"⚠️ 计算失败: {result['error']}\n")
            return "\n".join(lines), result

        lines.append(f"- HHI: **{result['hhi']:.4f}**")
        lines.append(f"- Top1权重: {result['top1_weight']:.1%}")
        lines.append(f"- Top5权重: {result['top5_weight']:.1%}")
        lines.append(f"- Top10权重: {result['top10_weight']:.1%}")

        sc = result.get("sector_concentration", {})
        lines.append(
            f"- 行业HHI: {sc.get('hhi', 0):.4f}  "
            f"最大行业: {sc.get('top_sector', {}).get('name', '-')}"
            f" ({sc.get('top_sector', {}).get('weight', 0):.1%})"
        )

        style = result.get("style_concentration", {})
        lines.append(
            f"- 风格分布: 大盘{style.get('large_cap', 0):.1%} / "
            f"中盘{style.get('mid_cap', 0):.1%} / 小盘{style.get('small_cap', 0):.1%}"
        )

        warnings = result.get("risk_warnings", [])
        if warnings:
            lines.append("\n**⚠️ 风险预警:**\n")
            for w in warnings:
                lines.append(f"- {w}")
        else:
            lines.append("\n✓ 未触发集中度预警")

        return "\n".join(lines), result

    def _section_drawdown(
        self, holdings: list[dict[str, Any]], analysis_window: int
    ) -> tuple[str, dict[str, Any]]:
        """生成回撤归因小节文本。"""
        try:
            result = self.drawdown_engine.compute(holdings, analysis_window=analysis_window)
        except Exception as e:
            result = {"error": str(e)}

        lines = ["## 4. 回撤归因\n"]
        if result.get("error"):
            lines.append(f"⚠️ 计算失败: {result['error']}\n")
            return "\n".join(lines), result

        p = result.get("max_drawdown_period", {})
        lines.append(f"- 最大回撤: **{result['max_drawdown']:.2f}%** ({p.get('start')} ~ {p.get('end')})")
        lines.append(f"- 当前回撤: {result['current_drawdown']:.2f}%")

        contributors = result.get("drawdown_contributors", [])[:5]
        if contributors:
            lines.append("\n**回撤贡献 Top 5:**\n")
            for c in contributors:
                lines.append(
                    f"- {c['name']}: 贡献{c['contribution_to_drawdown']:.1%}  "
                    f"个股回撤{c['stock_drawdown']:+.2f}%  权重{c['weight']:.1%}"
                )

        rec = result.get("recovery_analysis", {})
        if rec.get("avg_recovery_days") is not None:
            lines.append(
                f"\n- 历史平均恢复天数: {rec['avg_recovery_days']}天  "
                f"最差: {rec['worst_recovery_days']}天"
            )

        rm = result.get("risk_metrics", {})
        lines.append(
            f"\n- 95% VaR: {rm.get('var_95', 0):.2f}%  "
            f"95% CVaR: {rm.get('cvar_95', 0):.2f}%  "
            f"下行波动率: {rm.get('downside_volatility', 0):.2f}%"
        )

        return "\n".join(lines), result

    def _summary(
        self,
        exposure_result: dict[str, Any],
        attribution_result: dict[str, Any],
        concentration_result: dict[str, Any],
        drawdown_result: dict[str, Any],
    ) -> str:
        """基于四个子模块结果生成一句话结论摘要。"""
        points: list[str] = []

        if not attribution_result.get("error"):
            active = attribution_result.get("active_return", 0)
            if active > 0:
                points.append(f"跑赢基准{active:.2f}%")
            elif active < 0:
                points.append(f"跑输基准{abs(active):.2f}%")

        if not concentration_result.get("error"):
            warnings = concentration_result.get("risk_warnings", [])
            if warnings:
                points.append(f"存在{len(warnings)}项集中度风险")
            else:
                points.append("集中度健康")

        if not exposure_result.get("error"):
            bets = exposure_result.get("hidden_bets", [])
            if bets:
                points.append(f"{len(bets)}个非故意因子偏离")

        if not drawdown_result.get("error"):
            points.append(f"最大回撤{drawdown_result.get('max_drawdown', 0):.2f}%")

        if not points:
            return "数据不足，无法生成结论摘要。"
        return "，".join(points) + "。"

    # ────────────────────────── 主入口 ──────────────────────────

    def generate(
        self,
        holdings: list[dict[str, Any]],
        start_date: str | None = None,
        end_date: str | None = None,
        analysis_window: int = 250,
    ) -> str:
        """生成完整的 Markdown 诊断报告。

        Args:
            holdings: 持仓列表 [{"code": "600519", "weight": 0.15,
                      "name": "贵州茅台"}, ...]
            start_date: PnL 归因起始日期，默认取 end_date 前180天
            end_date: PnL 归因结束日期，默认为今天
            analysis_window: 回撤分析回看窗口（交易日），默认250

        Returns:
            str: Markdown 格式的诊断报告文本。任何子模块失败不会中断整体
                 报告生成，会在对应小节标注错误信息。
        """
        try:
            if not holdings:
                return "# 持仓诊断报告\n\n⚠️ 无持仓数据，无法生成报告。"

            now = datetime.now(CST)
            if end_date is None:
                end_date = now.strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (now - timedelta(days=180)).strftime("%Y-%m-%d")

            exposure_text, exposure_result = self._section_exposure(holdings)
            attribution_text, attribution_result = self._section_attribution(
                holdings, start_date, end_date
            )
            concentration_text, concentration_result = self._section_concentration(holdings)
            drawdown_text, drawdown_result = self._section_drawdown(holdings, analysis_window)

            summary = self._summary(
                exposure_result, attribution_result, concentration_result, drawdown_result
            )

            total_weight = sum(float(h.get("weight", 0) or 0) for h in holdings)
            report_lines = [
                "# 持仓综合诊断报告",
                "",
                f"生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (CST)",
                f"持仓数量: {len(holdings)}只  权重合计: {total_weight:.2%}",
                "",
                f"**结论摘要**: {summary}",
                "",
                "---",
                "",
                exposure_text,
                "",
                "---",
                "",
                attribution_text,
                "",
                "---",
                "",
                concentration_text,
                "",
                "---",
                "",
                drawdown_text,
                "",
            ]
            return "\n".join(report_lines)
        except Exception as e:
            return f"# 持仓综合诊断报告\n\n⚠️ 报告生成失败: {e}"


def main() -> None:
    """示例：生成模拟持仓的综合诊断报告。"""
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.15, "cost_price": 1800.0},
        {"code": "000858", "name": "五粮液", "weight": 0.10, "cost_price": 150.0},
        {"code": "300750", "name": "宁德时代", "weight": 0.08, "cost_price": 200.0},
        {"code": "601318", "name": "中国平安", "weight": 0.07, "cost_price": 45.0},
        {"code": "000333", "name": "美的集团", "weight": 0.06, "cost_price": 55.0},
        {"code": "002415", "name": "海康威视", "weight": 0.05, "cost_price": 30.0},
        {"code": "600036", "name": "招商银行", "weight": 0.05, "cost_price": 35.0},
        {"code": "000002", "name": "万科A", "weight": 0.04, "cost_price": 10.0},
        {"code": "601166", "name": "兴业银行", "weight": 0.04, "cost_price": 18.0},
        {"code": "002304", "name": "洋河股份", "weight": 0.04, "cost_price": 100.0},
    ]

    dr = DiagnosticReport()
    report = dr.generate(holdings)
    print(report)


if __name__ == "__main__":
    main()
