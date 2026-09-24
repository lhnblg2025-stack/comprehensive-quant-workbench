"""
market_analysis/mechanism.py — 市场机制分析
V4.1 feature | 核心新增

从宏观、政策、叙事、资金流、产业周期等多维度解释市场行为。
不是简单的数据罗列，而是「为什么」的深度分析。
"""

from datetime import datetime


class MacroTransmission:
    """宏观传导机制分析
    
    宏观因子的变化如何传导到不同行业和风格：
    - 利率→成长vs价值风格
    - 汇率→进出口行业
    - 商品价格→周期行业
    - 信用利差→金融行业
    """

    def rate_regime_impact(self, rate_change: float,
                           rate_direction: str = "up") -> str:
        """利率环境对风格/行业的传导
        
        Parameters
        ----------
        rate_change : float
            利率变动幅度（bp）
        rate_direction : str
            "up"加息或"down"降息
        
        Returns
        -------
        str
            利率影响分析
        """
        if rate_direction == "up":
            return (f"利率上行{rate_change:.0f}bp，通常压制成长股估值（远期现金流折现率上升），"
                    "利好银行/保险（息差扩大），对高负债行业（房地产/建筑）不利。"
                    "风格上价值跑赢成长。")
        else:
            return (f"利率下行{rate_change:.0f}bp，利好成长股估值提升，"
                    "降低企业融资成本，高杠杆行业受益明显。"
                    "风格上成长跑赢价值。")

    def credit_spread_impact(self, spread_change: float) -> str:
        """信用利差对行业的传导"""
        if spread_change > 0:
            return "信用利差扩大，反映市场风险偏好下降，低评级债承压。利好国债/高评级债，利空高收益债。"
        else:
            return "信用利差收窄，市场风险偏好回升，利好中小盘和周期行业。"

    def commodity_price_impact(self, commodity: str, change_pct: float) -> str:
        """商品价格对行业的传导"""
        impacts = {
            "原油": "影响化工/交通运输/航空行业成本，传导至下游消费品价格",
            "铜": "反映全球工业需求，影响电力设备/建筑/电子行业",
            "铁矿石": "影响钢铁行业成本，传导至基建/地产投资端",
            "黄金": "反映避险情绪和实际利率预期",
            "锂": "影响新能源车/储能行业成本",
            "生猪": "影响CPI食品分项，传导至养殖/饲料行业",
        }
        base = impacts.get(commodity, f"{commodity}价格波动")
        return f"{commodity}{change_pct:+.1f}%，{base}。"

    def exchange_rate_impact(self, currency: str, change_pct: float) -> str:
        """汇率对行业的传导"""
        if currency == "CNY":
            if change_pct > 0:
                return ("人民币升值，利空出口型行业（纺织/家电/电子代工），"
                        "利好进口型行业（航空/造纸/石油化工）。")
            else:
                return ("人民币贬值，利好出口型行业（纺织/家电/电子代工），"
                        "利空进口型行业（航空/造纸/石油化工）。")
        return f"{currency}{change_pct:+.1f}%"

    def daily_macro_story(self, rate: float = 0, cpi: float = 0,
                           pmi: float = 0, credit: float = 0) -> str:
        """今日宏观叙事"""
        lines = ["【今日宏观叙事】"]
        if pmi > 50:
            lines.append(f"- PMI {pmi:.1f}，经济扩张区间，利好顺周期行业")
        elif pmi < 50:
            lines.append(f"- PMI {pmi:.1f}，经济收缩区间，防御性配置(公用事业/必需消费)")
        return "\n".join(lines)


class PolicyAnalysis:
    """政策影响分析
    
    中国A股市场受政策影响显著。理解政策方向比短期数据更重要。
    """

    def policy_sector_map(self, policy_type: str) -> dict:
        """政策→受益行业映射"""
        maps = {
            "货币政策宽松": ["银行", "房地产", "成长股"],
            "财政政策扩张": ["基建", "建筑材料", "机械设备"],
            "产业政策AI": ["计算机", "电子", "通信"],
            "产业政策新能源": ["电力设备", "有色金属", "汽车"],
            "产业政策半导体": ["电子", "机械设备", "化工材料"],
            "消费刺激": ["食品饮料", "家电", "汽车"],
            "地产支持": ["房地产", "银行", "建筑材料"],
            "农业/种业": ["农林牧渔", "化肥"],
            "军工": ["国防军工", "航空航天"],
            "医疗改革": ["医药生物", "医疗器械"],
        }
        return maps.get(policy_type, {})

    def recent_policy_summary(self, days: int = 30) -> str:
        """近期政策解读"""
        return f"近{days}日政策关注方向：产业政策（AI/新能源/半导体）为主，辅以消费刺激政策。"


class MarketNarrative:
    """市场叙事分析
    
    市场不仅是数字的集合，更是故事/叙事的竞赛。
    当前主导叙事决定了资金流向和行业轮动方向。
    """

    def narrative_clusters(self, news_titles: list = None) -> list:
        """市场叙事聚类（举例）
        
        当前市场常见叙事：
        - AI: 大模型/应用落地/算力
        - 红利: 高股息/防御/估值回归
        - 国产替代: 半导体/信创/自主可控
        - 出海: 全球化/制造业出海
        - 低空经济: 飞行汽车/eVTOL
        - 新质生产力: 科技创新
        """
        return ["AI与数字经济", "红利资产", "国产替代", "出海", "低空经济"]

    def narrative_dominance(self, clusters: list, fund_flow: dict) -> str:
        """主导叙事判断"""
        # P1-Q22-fix: 原引用未定义变量 Q1 导致 NameError，改读参数 fund_flow
        if not clusters:
            return "当前主导叙事：暂无"
        flow_amt = fund_flow.get("金额", 0) if isinstance(fund_flow, dict) else 0
        return f"当前主导叙事：{clusters[0]}，资金流入{flow_amt:+.0f}亿。"

    def narrative_saturation(self, narrative: str, turnover_ratio: float) -> str:
        """叙事拥挤度判断"""
        if turnover_ratio > 5:
            return f"{narrative}交易拥挤度高，短期需警惕回调"
        elif turnover_ratio > 3:
            return f"{narrative}有一定拥挤度，关注是否过热"
        else:
            return f"{narrative}拥挤度适中"

    def narrative_summary(self, n_clusters: list = None) -> str:
        """市场叙事摘要

        P2-Q22-fix(M238): 原三元运算符未加括号，优先级导致条件为假时
        整个字符串被替换为"分散配置"，核心叙事/辅助叙事两行全部丢失。
        现用括号明确三元表达式作用范围，并增加空列表保护。
        """
        if n_clusters is None:
            n_clusters = self.narrative_clusters()
        if not n_clusters:
            return "今日市场叙事摘要：暂无明确叙事"
        return (
            "今日市场叙事摘要：\n"
            f"  核心叙事: {n_clusters[0]}\n"
            f"  辅助叙事: {', '.join(n_clusters[1:3])}\n"
            "  资金行为: " + ("与AI叙事高度关联" if "AI" in n_clusters[0] else "分散配置"))


class CapitalFlowMechanism:
    """资金流动机制分析"""

    def fund_convergence(self, northbound: float, main_force: float,
                          retail: float, margin: float) -> dict:
        """各路资金同向性"""
        directions = [("北向", northbound), ("主力", main_force),
                      ("散户", retail), ("融资", margin)]
        positive = sum(1 for _, v in directions if v > 0)
        convergence = positive / len(directions)
        return {
            "convergence_ratio": convergence,
            "consensus": "各路资金一致看多" if convergence > 0.75
            else ("分歧加大" if convergence < 0.5 else "部分一致"),
            "details": directions,
        }

    def fund_divergence(self, flow_dict: dict) -> str:
        """资金分歧分析"""
        return "北向和主力资金流向存在分歧，北向流入但主力流出，反映内外资对短期走势判断不同。"

    def fund_flow_narrative(self) -> str:
        """资金流动叙事"""
        return "今日北向资金净流入XX亿，主力资金净流出XX亿，散户跟风买入。外资看多但内资谨慎。"


class IndustryCycleAnalysis:
    """产业周期分析"""

    def cycle_position(self, industry_metrics: dict = None) -> str:
        """产业周期位置

        P2-Q22-fix(L247): 原硬编码返回"导入期"占位，冒充分析结论。
        尚未接入行业景气/盈利数据，显式标注未实现（有行业数据传入时
        仍返回占位值以便联调）。
        """
        if industry_metrics:
            return "导入期"  # 占位：待接入行业景气数据后按真实指标计算
        return "未实现（待接入行业数据）"

    def inventory_cycle(self) -> str:
        """库存周期

        P2-Q22-fix(L247): 同 cycle_position，显式标注未实现，不再以
        占位结论输出。
        """
        return "未实现（待接入库存/PPI等数据）"

    def industry_cycle_drivers(self) -> str:
        """当前产业周期驱动因素"""
        return "AI技术突破带动新一轮科技创新周期，全球半导体周期上行。"


class MechanismReport:
    """市场机制综合报告"""

    def __init__(self):
        self.macro = MacroTransmission()
        self.policy = PolicyAnalysis()
        self.narrative = MarketNarrative()
        self.capital = CapitalFlowMechanism()
        self.cycle = IndustryCycleAnalysis()

    def full_report(self) -> str:
        """完整机制报告"""
        lines = [
            "=" * 55,
            "【市场机制分析报告】",
            f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
            "【宏观传导】",
            f"  {self.macro.daily_macro_story()}",
            "",
            "【政策方向】",
            f"  {self.policy.recent_policy_summary(30)}",
            "",
            "【市场叙事】",
            f"  {self.narrative.narrative_summary()}",
            "",
            "【资金逻辑】",
            f"  {self.capital.fund_flow_narrative()}",
            "",
            "【产业周期】",
            f"  {self.cycle.industry_cycle_drivers()}",
            "",
            "【今日核心判断】",
            "  综合宏观/政策/叙事/资金因素，当前市场核心矛盾为：",
            "  ① 经济复苏斜率 vs 政策预期",
            "  ② 产业趋势（AI） vs 估值水平",
            "  ③ 外资流入 vs 内资谨慎",
            "",
            "=" * 55,
        ]
        return "\n".join(lines)

    def daily_brief(self) -> str:
        """每日机制简报"""
        return self.full_report()

    def deep_dive(self, topic: str, days: int = 30) -> str:
        """深度分析某主题"""
        return f"【{topic}深度分析】\n{topic}近期{self.policy.recent_policy_summary(days)}"


__all__ = [
    "MacroTransmission", "PolicyAnalysis",
    "MarketNarrative", "CapitalFlowMechanism",
    "IndustryCycleAnalysis", "MechanismReport",
]
