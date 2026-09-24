"""
fundamental_analysis.py — 基本面分析与估值建模
V4.1 feature

提供：DCF估值、可比估值、财务健康评分、财报分析报告

D5收敛登记 (2026-08-11): 基本面域收敛（保守策略）——与 fundamental.py /
financial_data.py 的数值转换重复已由 D1(utils.to_float) 收敛；本模块为纯分析层
（无抓取），DCF/健康评分/报告为独立能力保留；估值倍数与增速/毛利率的量纲约定
与抓取层同名异口径，保留并显式标注。
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from quant_system.utils import to_float as _to_float_impl

logger = __import__('logging').getLogger(__name__)

# ── 基础常量 ──
RISK_FREE_RATE = 0.025  # 无风险利率
ERP = 0.06  # 股权风险溢价
DEFAULT_GROWTH = 0.05  # 默认永续增长率


def _to_float(value) -> Optional[float]:
    """D1 收敛: 转发 quant_system.utils.to_float（default=None + finite 过滤保留原语义）。"""
    if isinstance(value, bool):
        return None
    return _to_float_impl(value, default=None, finite=True)


def _normalize_growth(value) -> Optional[float]:
    """增速归一化：|v|>1.5 视为百分数(v/100)，否则视为小数。

    D5收敛: 同名异口径保留 —— 本函数输出小数(0.30=30%)，供评分/报告使用；
    fundamental.py(revenue_yoy/profit_yoy) 与 financial_data.py(sales_growth_yoy/
    profit_growth_yoy/earnings_growth_qoq) 均保持原始百分数，不强迁。
    """
    out = _to_float(value)
    if out is None:
        return None
    if abs(out) > 1.5:
        return out / 100.0
    return out


def _first_growth_value(*dicts: dict, keys: tuple[str, ...]) -> Optional[float]:
    for data in dicts:
        for key in keys:
            if key in data:
                value = _normalize_growth(data.get(key))
                if value is not None:
                    return value
    return None


def _positive_ratio(numerator: float, denominator) -> Optional[float]:
    denominator = _to_float(denominator)
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _compute_accruals_ratio(income: dict, balance: dict) -> Optional[float]:
    """P2-Q3-fix(M388): 应计比率 = (净利润 - 经营现金流) / 平均总资产。

    字段缺失时返回 None（表示「未知」，调用方按缺失处理，不硬编码兜底）。
    """
    net_profit = _to_float(income.get("net_profit"))
    ocf = _to_float(income.get("operating_cash_flow", income.get("ocf")))
    ta_cur = _to_float(balance.get("total_assets"))
    ta_prev = _to_float(balance.get("total_assets_prev", balance.get("prev_total_assets")))
    if net_profit is None or ocf is None or ta_cur is None or ta_cur <= 0:
        return None
    avg_ta = (ta_cur + (ta_prev if ta_prev is not None and ta_prev > 0 else ta_cur)) / 2.0
    if avg_ta <= 0:
        return None
    return (net_profit - ocf) / avg_ta


def _format_multiple(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}"


# ══════════════════════════════════════
# 1. DCF 估值模型
# ══════════════════════════════════════

class DCFValuation:
    """DCF 估值模型 (两阶段/三阶段)

    D5收敛登记: 独立能力保留（两阶段/三阶段DCF + WACC/CAPM + 敏感性分析，无等价实现）。
    """

    def __init__(self, wacc: float = 0.10, terminal_growth: float = 0.03,
                 projection_years: int = 10):
        """
        Parameters
        ----------
        wacc : float
            加权平均资本成本
        terminal_growth : float
            永续增长率
        projection_years : int
            预测期数
        """
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.projection_years = projection_years

    def two_stage_dcf(self, fcf_current: float, growth_rates: list[float],
                      shares_outstanding: float = 1.0, net_debt: float = 0.0) -> dict:
        """两阶段DCF：高速增长期 + 永续增长期

        Parameters
        ----------
        fcf_current : float
            当前自由现金流
        growth_rates : list[float]
            预测期各年增长率 (长度=projection_years)
        shares_outstanding : float
            总股本
        net_debt : float
            净债务

        Returns
        -------
        dict : {equity_value, per_share, inputs, phase_values}
        """
        # V4.1 fix: terminal_growth must be < WACC or terminal value diverges
        if self.terminal_growth >= self.wacc:
            raise ValueError(
                f"terminal_growth ({self.terminal_growth}) 必须 < WACC ({self.wacc}), "
                f"否则终值发散/为负"
            )

        # V4.1 fix: truncate growth_rates if longer than projection_years
        if len(growth_rates) > self.projection_years:
            import logging
            logging.getLogger(__name__).warning(
                f"growth_rates长度({len(growth_rates)}) > projection_years({self.projection_years}), 截断"
            )
            growth_rates = growth_rates[:self.projection_years]

        if len(growth_rates) < self.projection_years:
            # P2-Q3-fix(L392): 预测期补位用模块级 DEFAULT_GROWTH 配置，不再硬编码 0.05
            growth_rates = growth_rates + [DEFAULT_GROWTH] * (self.projection_years - len(growth_rates))

        fcf_values = []
        pv_factors = []
        fcf = fcf_current

        for i in range(self.projection_years):
            fcf *= (1 + growth_rates[i])  # V4.1 fix: growth_rates always padded to projection_years, else branch was dead code
            fcf_values.append(fcf)
            pv_factors.append(1 / (1 + self.wacc) ** (i + 1))

        # 预测期FCF现值
        pv_fcf = sum(f * p for f, p in zip(fcf_values, pv_factors))

        # 终值（Gordon Growth Model）
        terminal_fcf = fcf_values[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth)
        pv_terminal = terminal_value / (1 + self.wacc) ** self.projection_years

        # 企业价值 → 股权价值
        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value - net_debt
        per_share = equity_value / max(shares_outstanding, 1)

        return {
            "enterprise_value": round(enterprise_value, 2),
            "equity_value": round(equity_value, 2),
            "per_share": round(per_share, 2),
            "pv_fcf": round(pv_fcf, 2),
            "pv_terminal": round(pv_terminal, 2),
            "terminal_pct": round(pv_terminal / max(enterprise_value, 1e-8) * 100, 1),
            "inputs": {
                "wacc": self.wacc,
                "terminal_growth": self.terminal_growth,
                "projection_years": self.projection_years,
            },
            "phase_values": {
                "fcf_forecast": [round(f, 2) for f in fcf_values],
                "discount_factors": [round(p, 4) for p in pv_factors],
            },
        }

    def three_stage_dcf(self, fcf_current: float, high_growth_rates: list[float],
                         transition_growth_rates: list[float],
                         shares_outstanding: float = 1.0,
                         net_debt: float = 0.0) -> dict:
        """三阶段DCF：高速增长 + 过渡 + 永续

        P2-Q3-fix(L392): 注明简化 —— 当前实现为两阶段拼接（过渡期与高速增长期
        使用同一折现率 WACC），未引入独立过渡期折现率。
        """
        all_rates = high_growth_rates + transition_growth_rates
        return self.two_stage_dcf(fcf_current, all_rates, shares_outstanding, net_debt)

    @staticmethod
    def estimate_wacc(equity: float, debt: float, cost_equity: float,
                      cost_debt: float, tax_rate: float = 0.25) -> float:
        """估算WACC: (E/V) * Re + (D/V) * Rd * (1-T)"""
        total = equity + debt
        we = equity / max(total, 1e-8)
        wd = debt / max(total, 1e-8)
        return we * cost_equity + wd * cost_debt * (1 - tax_rate)

    @staticmethod
    def capm_cost_equity(beta: float, risk_free: float = RISK_FREE_RATE,
                         erp: float = ERP) -> float:
        """CAPM: Re = Rf + Beta * ERP"""
        return risk_free + beta * erp

    def sensitivity_analysis(self, fcf_current: float, growth_rates: list,
                              shares: float, net_debt: float,
                              wacc_range: list = None,
                              tg_range: list = None) -> pd.DataFrame:
        """敏感性分析：WACC × 终值增长率 对每股价值的影响

        P2-Q3-fix(M390): wacc≤tg 的组合会使终值发散，two_stage_dcf 抛 ValueError
        导致敏感性分析崩溃。这里直接跳过 wacc≤tg 的组合（单元格置 None）。
        """
        if wacc_range is None:
            wacc_range = np.arange(self.wacc - 0.02, self.wacc + 0.03, 0.01)
        if tg_range is None:
            tg_range = np.arange(0.01, 0.06, 0.01)

        results = []
        for wacc in wacc_range:
            row = {}
            for tg in tg_range:
                if wacc <= tg:
                    row[f"g={tg:.0%}"] = None  # 终值发散，跳过
                    continue
                # V4.1 fix: use temporary params dict instead of modifying self (thread-safe)
                temp_model = DCFValuation(wacc=wacc, terminal_growth=tg,
                                          projection_years=self.projection_years)
                val = temp_model.two_stage_dcf(fcf_current, growth_rates, shares, net_debt)
                row[f"g={tg:.0%}"] = val["per_share"]
            results.append({"wacc": wacc, **row})
        return pd.DataFrame(results).set_index("wacc")


# ══════════════════════════════════════
# 2. 可比估值分析
# ══════════════════════════════════════

class ComparableValuation:
    """可比估值分析 (PE/PB/PS/EV/EBITDA)

    D5收敛登记: 独立能力保留（可比公司对比/目标价，无等价实现）。
    """

    @staticmethod
    def compute_multiples(financials: dict, price: float,
                           shares: float) -> dict:
        """计算估值倍数

        D5收敛: 同名异口径保留 —— PE/PB/PS = 市值/年报科目(调用方 financials dict)，
        与 fundamental.py 东财行情字段 PE/PB、financial_data.py 收盘价/每股TTM 口径不同。
        """
        net_profit = financials.get("net_profit", 0) or 0
        revenue = financials.get("revenue", 0) or 0
        book_value = financials.get("equity", 0) or 0
        total_assets = financials.get("total_assets", 0) or 0
        market_cap = price * max(shares, 1)

        return {
            "market_cap": market_cap,
            "pe": _positive_ratio(market_cap, net_profit),
            "pb": _positive_ratio(market_cap, book_value),
            "ps": _positive_ratio(market_cap, revenue),
            "market_cap_to_assets": _positive_ratio(market_cap, total_assets),
        }

    @staticmethod
    def peer_comparison(target_multiples: dict,
                         peer_multiples: list[dict]) -> pd.DataFrame:
        """同业对比：目标 vs 同业中位数/均值"""
        if not peer_multiples:
            return pd.DataFrame()

        peer_df = pd.DataFrame(peer_multiples)
        stats = []
        for metric in ["pe", "pb", "ps"]:
            if metric not in target_multiples or metric not in peer_df.columns:
                continue
            target_value = _to_float(target_multiples[metric])
            if target_value is None or target_value <= 0:
                continue
            peer_vals = pd.to_numeric(peer_df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            peer_vals = peer_vals[peer_vals > 0]
            if len(peer_vals) == 0:
                continue
            stats.append({
                "metric": metric,
                "target": target_value,
                "peer_mean": peer_vals.mean(),
                "peer_median": peer_vals.median(),
                "peer_std": peer_vals.std(),
                "peer_min": peer_vals.min(),
                "peer_max": peer_vals.max(),
                "percentile": (peer_vals < target_value).mean(),
                "premium": (target_value - peer_vals.mean()) / max(peer_vals.mean(), 1e-8),
            })
        return pd.DataFrame(stats)

    @staticmethod
    def target_price_by_pe(target_eps: float, peer_pe: float) -> float:
        """基于可比PE的目标价"""
        return target_eps * peer_pe

    @staticmethod
    def target_price_by_pb(target_bvps: float, peer_pb: float) -> float:
        """基于可比PB的目标价"""
        return target_bvps * peer_pb


# ══════════════════════════════════════
# 3. 财务健康评分
# ══════════════════════════════════════

class FinancialHealthScore:
    """财务健康评分系统 (0-100)"""

    def __init__(self):
        self.scores = {}
        self.max_scores = {}

    def score_profitability(self, roe: float, roa: float,
                            gross_margin: float, net_margin: float) -> dict:
        """盈利能力评分 (满分65, 占总分约29.5%)

        P2-Q3-fix(L393): 权重/量纲约定文档化 ——
        量纲: roe/roa 为百分数(如 15 = 15%), gross_margin/net_margin 为小数
        (如 0.4 = 40%)。调用方传错量纲会得到错误评分。

        D5收敛: 同名异口径保留 —— gross_margin/net_margin 小数口径(0.4=40%)，
        与 fundamental.py / financial_data.py 的百分数字段(如 40=40%)不同；
        消费方接入抓取层数据时必须换算，D5 不强迁。
        """
        score = 0
        details = {}

        # ROE > 15% 优秀, > 8% 良好, > 0% 及格
        if roe > 15:
            score += 30
            details["roe"] = "优秀"
        elif roe > 8:
            score += 20
            details["roe"] = "良好"
        elif roe > 0:
            score += 10
            details["roe"] = "及格"
        else:
            details["roe"] = "亏损"

        # 毛利率 > 40%
        if gross_margin > 0.4:
            score += 20
        elif gross_margin > 0.2:
            score += 12
        else:
            score += 5

        # 净利率 > 15%
        if net_margin > 0.15:
            score += 15
        elif net_margin > 0.05:
            score += 8
        elif net_margin > 0:
            score += 3
        else:
            score += 0

        self.scores["profitability"] = score
        self.max_scores["profitability"] = 65
        return {"score": score, "max_score": 65, "pct": score / 65 * 100, "details": details}

    def score_solvency(self, current_ratio: float, debt_to_equity: float) -> dict:
        """偿债能力评分 (满分55, 占总分25%)

        P2-Q3-fix(L393): 量纲约定 —— current_ratio 为倍数(如 2.0=2倍),
        debt_to_equity 为 D/E 比率(如 0.5=50%)。
        """
        score = 0
        if current_ratio > 2.0:
            score += 30
        elif current_ratio > 1.5:
            score += 20
        elif current_ratio > 1.0:
            score += 10
        else:
            score += 5

        if debt_to_equity < 0.5:
            score += 25
        elif debt_to_equity < 1.0:
            score += 18
        elif debt_to_equity < 2.0:
            score += 10
        else:
            score += 5

        self.scores["solvency"] = score
        self.max_scores["solvency"] = 55
        return {"score": score, "max_score": 55, "pct": score / 55 * 100}

    def score_growth(self, revenue_growth: Optional[float], profit_growth: Optional[float]) -> dict:
        """成长能力评分 (满分50, 占总分约22.7%)

        P2-Q3-fix(L393): 量纲约定 —— 增速为小数(如 0.30 = 30%)。
        """
        if revenue_growth is None and profit_growth is None:
            self.scores.pop("growth", None)
            self.max_scores.pop("growth", None)
            return {"score": None, "max_score": 50, "pct": None, "skipped": True}

        revenue_growth = revenue_growth or 0.0
        profit_growth = profit_growth or 0.0
        score = 0

        if revenue_growth > 0.30:
            score += 25
        elif revenue_growth > 0.15:
            score += 18
        elif revenue_growth > 0:
            score += 10
        else:
            score += 0

        if profit_growth > 0.30:
            score += 25
        elif profit_growth > 0.15:
            score += 18
        elif profit_growth > 0:
            score += 10
        else:
            score += 0

        self.scores["growth"] = score
        self.max_scores["growth"] = 50
        return {"score": score, "max_score": 50, "pct": score / 50 * 100}

    def score_quality(self, operating_cf_margin: Optional[float],
                       accruals_ratio: Optional[float]) -> dict:
        """盈利质量评分 (满分50, 占总分约22.7%)

        P2-Q3-fix(M388): 输入缺失时按可用子项计分、缺项不计满分，不再用
        硬编码 0.1 / 0.05 兜底掩盖缺失（否则盈利质量分恒定且失真）。
        量纲约定: operating_cf_margin 为小数(如 0.2 = 20%)。
        """
        score = 0
        max_score = 0

        if operating_cf_margin is not None:
            max_score += 30
            if operating_cf_margin > 0.2:
                score += 30
            elif operating_cf_margin > 0.1:
                score += 20
            elif operating_cf_margin > 0:
                score += 10
            else:
                score += 0

        # 应计比率越低越好
        if accruals_ratio is not None:
            max_score += 20
            if accruals_ratio < 0:
                score += 20
            elif accruals_ratio < 0.1:
                score += 15
            elif accruals_ratio < 0.3:
                score += 8
            else:
                score += 3

        if max_score == 0:
            self.scores.pop("quality", None)
            self.max_scores.pop("quality", None)
            return {"score": None, "max_score": 0, "pct": None, "skipped": True}

        self.scores["quality"] = score
        self.max_scores["quality"] = max_score
        return {"score": score, "max_score": max_score, "pct": score / max_score * 100}

    def total_score(self) -> float:
        """总评分"""
        raw = sum(self.scores.values())
        max_raw = sum(self.max_scores.values())
        if max_raw <= 0:
            return 0.0
        return round(raw / max_raw * 100, 1)

    def rating(self) -> str:
        """评级"""
        s = self.total_score()
        if s >= 80:
            return "AAA"
        elif s >= 65:
            return "AA"
        elif s >= 50:
            return "A"
        elif s >= 35:
            return "BBB"
        else:
            return "BB"


# ══════════════════════════════════════
# 4. 财报分析报告
# ══════════════════════════════════════

class FinancialReportGenerator:
    """财报分析报告生成器

    D5收敛登记: 独立能力保留（整合 DCF/可比/健康评分的报告层，无等价实现）。
    """

    def __init__(self):
        self.dcf = DCFValuation()
        self.comparable = ComparableValuation()
        self.health = FinancialHealthScore()

    def generate(self, symbol: str, financials: dict, price: float,
                 shares: float) -> str:
        """生成完整基本面分析报告"""
        self.health.scores.clear()
        self.health.max_scores.clear()
        ratios = financials.get("ratios", {})
        income = financials.get("income", {})
        balance = financials.get("balance", {})

        revenue_growth = _first_growth_value(
            financials, ratios, income,
            keys=("revenue_growth", "revenue_yoy", "sales_growth"),
        )
        profit_growth = _first_growth_value(
            financials, ratios, income,
            keys=("profit_growth", "profit_yoy", "net_profit_growth", "earnings_growth"),
        )

        # 财务健康评分
        self.health.score_profitability(
            ratios.get("roe", 0), ratios.get("roa", 0),
            ratios.get("gross_margin", 0), ratios.get("net_margin", 0))
        self.health.score_solvency(
            ratios.get("current_ratio", 3), ratios.get("debt_to_equity", 0.3))
        self.health.score_growth(revenue_growth, profit_growth)
        # P2-Q3-fix(M388): 不再硬编码应计比率 0.05 / 兜底 0.1；取真实应计比率与经营现金流
        self.health.score_quality(
            _to_float(ratios.get("operating_cf_margin")),
            _compute_accruals_ratio(income, balance))

        # DCF 估值
        fcf = income.get("net_profit", 0) * 0.7  # 简化FCF近似
        # P2-Q3-fix(M389): 净债务 = 有息负债(短借+长借+应付债券) - 货币资金，
        # 不再把全部负债当有息负债（此前系统性低估股权价值）。
        int_debt = (
            balance.get("short_term_borrowing", 0)
            + balance.get("long_term_borrowing", 0)
            + balance.get("bonds_payable", 0)
        )
        net_debt = int_debt - balance.get("cash", 0)
        dcf_val = self.dcf.two_stage_dcf(
            fcf, [0.15, 0.12, 0.10, 0.08, 0.06],
            shares, net_debt)

        # 可比估值倍数
        multiples = self.comparable.compute_multiples(financials, price, shares)

        lines = [
            "=" * 55,
            f"基本面分析报告: {symbol}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"当前股价: {price:.2f}",
            "=" * 55,
            "",
            "【财务健康评分】",
            f"  总评分: {self.health.total_score()}/100 (评级: {self.health.rating()})",
            f"  盈利能力: {self.health.scores.get('profitability', 0):.0f}/65",
            f"  偿债能力: {self.health.scores.get('solvency', 0):.0f}/55",
            f"  成长: {self.health.scores['growth']:.0f}/{self.health.max_scores.get('growth', 50)}" if "growth" in self.health.scores else "  成长: 跳过（缺少营收/利润增速）",
            f"  盈利质量: {self.health.scores.get('quality', 0):.0f}/{self.health.max_scores.get('quality', 0)}" if "quality" in self.health.scores else "  盈利质量: 跳过（缺少现金流数据）",
            "",
            "【DCF 估值】",
            f"  每股价值: {dcf_val['per_share']:.2f}",
            f"  当前股价: {price:.2f}",
            ("  隐含空间: " + (f"{(dcf_val['per_share'] / price - 1) * 100:+.1f}%" if price > 0 else "N/A (price≤0)")),
            f"  终值占比: {dcf_val['terminal_pct']:.1f}%",
            "",
            "【估值倍数】",
            f"  PE: {_format_multiple(multiples['pe'])}",
            f"  PB: {_format_multiple(multiples['pb'])}",
            f"  PS: {_format_multiple(multiples['ps'])}",
            "",
            "【财务指标】",
            f"  ROE: {ratios.get('roe', 0):.1f}%",
            f"  ROA: {ratios.get('roa', 0):.1f}%",
            f"  毛利率: {ratios.get('gross_margin', 0)*100:.1f}%",
            f"  净利率: {ratios.get('net_margin', 0)*100:.1f}%",
            # P2-Q3-fix(M387): debt_to_equity 是产权比率(D/E)，此前被误标为「资产负债率」
            f"  产权比率(D/E): {ratios.get('debt_to_equity', 0)*100:.1f}%",
        ]

        # DuPont 分解
        dupont = ratios.get("dupont_roe", 0)
        if dupont:
            lines.extend([
                "",
                "【DuPont 分解】",
                f"  ROE (DuPont): {dupont:.1f}%",
                f"  净利率: {ratios.get('net_margin', 0)*100:.1f}%",
                f"  资产周转率: {ratios.get('dupont_asset_turnover', 0):.2f}",
                f"  权益乘数: {ratios.get('dupont_equity_multiplier', 0):.2f}",
            ])

        lines.append("")
        lines.append("=" * 55)
        return "\n".join(lines)


__all__ = [
    "DCFValuation", "ComparableValuation",
    "FinancialHealthScore", "FinancialReportGenerator",
]
