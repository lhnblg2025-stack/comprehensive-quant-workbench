"""
market_analysis/hot_stocks.py — 热股深度分析
V4.1 feature | 核心新增

用户硬规则要求的分析维度：
- 热股等权涨跌幅
- 热度TOP 50深度分析
- 中大市值涨跌前50
- 异常波动检测
- 热股机制解释
"""

import logging
import pandas as pd
from datetime import datetime, timedelta  # P1-Q22-fix: datetime.now() 原先未导入导致 NameError


class HotStockRanking:
    """热股排名系统
    
    综合热度评分基于成交量、涨跌幅、换手率、资金流等维度。
    热股池每日更新，用于衡量市场关注度最高的股票的赚钱效应。
    """

    def __init__(self):
        pass

    def heat_score(self, volume_rank: float = 0, change_rank: float = 0,
                    turnover_rank: float = 0, fund_rank: float = 0) -> float:
        """综合热度评分（各维度等权）"""
        return (volume_rank + change_rank + turnover_rank + fund_rank) / 4

    def top_hot(self, n: int = 50) -> pd.DataFrame:
        """TOP N热股 (V5.2 fix: akshare东方财富热榜)"""
        try:
            import akshare as ak
            df = ak.stock_hot_rank_em()
            if df is None or df.empty:
                return pd.DataFrame()
            # 标准化列名
            col_map = {"股票代码": "symbol", "股票简称": "name", "最新价": "price", "涨跌幅": "pct", "排名": "rank"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            return df.head(n).reset_index(drop=True)
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    def hot_pool_equal_weight_return(self, hot_prices: pd.DataFrame) -> float:
        """热股池等权收益率（用户硬规则）
        
        衡量最受市场关注的股票的整体赚钱效应。
        热股池涨 = 市场情绪好；热股池暴跌 = 市场情绪差。
        
        Parameters
        ----------
        hot_prices : pd.DataFrame
            热股价格矩阵，每列一只热股，index为日期
        
        Returns
        -------
        float
            当日等权涨跌幅（%）
        """
        if hot_prices.empty:
            return 0.0
        returns = hot_prices.pct_change().iloc[-1]
        return round(returns.mean() * 100, 2)

    def hot_pool_weighted_ret(self, hot_prices: pd.DataFrame,
                               weight_scheme: str = "market_cap") -> float:
        """加权收益率

        P2-Q22-fix(L245): 原 market_cap 口径未接入真实市值数据、静默退化为等权。
        现对未实现的市值加权发出显式 warnings 提示（可见降级），不再静默。
        返回类型保持不变（float），兼容既有调用方。
        """
        if hot_prices.empty:
            return 0.0
        ret = hot_prices.pct_change().iloc[-1]
        if weight_scheme == "equal":
            return round(ret.mean() * 100, 2)
        # 市值加权需要市值信息
        import warnings
        warnings.warn(
            "hot_pool_weighted_ret: market_cap 加权尚未接入真实市值数据，"
            "当前按等权计算（降级已提示）",
            stacklevel=2,
        )
        return round(ret.mean() * 100, 2)

    def hot_pool_vs_market(self, hot_prices: pd.DataFrame,
                            market_ret: float) -> dict:
        """热股vs全市场超额"""
        hot_ret = self.hot_pool_equal_weight_return(hot_prices)
        return {
            "hot_ret": hot_ret,
            "market_ret": market_ret,
            "excess_ret": round(hot_ret - market_ret, 2),
        }


class MidLargeCapMovers:
    """中大市值异动分析
    
    分析沪深300/中证500/中证1000成分股中涨跌幅前50的股票，
    判断市场资金在哪个市值段活跃，以及对指数的影响。
    """

    def __init__(self):
        pass

    # V4.1 fix: _mock_data 原方法使用 np.random 生成假数据，已移除
    # 所有调用 _mock_data 的方法均已替换为 NotImplementedError

    def hs300_top_bottom(self, n: int = 50) -> dict:
        """沪深300成分涨跌前50 (V5.2 fix: akshare实现)

        P2-Q22-fix(M234): 口径注明——受逐股拉取日线速度限制，
        仅覆盖按权重排序前100只成分（近似样本），非全量300只全量计算；
        输出 coverage 字段注明近似口径。
        """
        try:
            import akshare as ak
            # P2-Q22-fix(L245): 删除死代码 ak.stock_zh_index_daily(symbol="sh000300")
            dfs = []
            symbols = self._get_hs300_symbols()
            total = len(symbols)
            sampled = symbols[:100]  # 限100只防过慢（近似口径）
            for sym in sampled:
                try:
                    # P1-Q22-fix: start_date 动态取近60个自然日(约40个交易日)，保证至少有2根K线可算涨跌幅
                    bars = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
                    if bars is not None and len(bars) > 1:
                        pct = (float(bars["收盘"].iloc[-1]) / float(bars["收盘"].iloc[-2]) - 1) * 100
                        dfs.append({"symbol": sym, "name": bars["名称"].iloc[-1] if "名称" in bars.columns else "", "pct": round(pct, 2)})
                except Exception as e:
                    logging.getLogger(__name__).error(f"[hot_stocks] 操作失败: {e}", exc_info=True)
            if not dfs:
                return {"error": "no_data"}
            df = pd.DataFrame(dfs).sort_values("pct", ascending=False)
            return {
                "top_gainers": df.head(n).to_dict("records"),
                "top_losers": df.tail(n).sort_values("pct").to_dict("records"),
                "median_pct": round(float(df["pct"].median()), 2),
                "coverage": f"按权重排序前{len(sampled)}只/{total}只({len(sampled)/max(total,1):.0%})近似口径",
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_hs300_symbols(self):
        try:
            import akshare as ak
            idx_weight = ak.index_stock_cons_weight_csindex("000300")
            return list(idx_weight["成分券代码"].values)[:300]
        except Exception:
            return []

    def zz500_top_bottom(self, n: int = 50) -> dict:
        """中证500成分涨跌前50 (V5.2 fix: akshare实现)

        P2-Q22-fix(M234): 口径注明——仅覆盖按权重排序前100只成分（近似样本），
        非全量500只计算；输出 coverage 字段注明近似口径。
        """
        try:
            import akshare as ak
            dfs = []
            syms = self._get_zz500_symbols()
            total = len(syms)
            sampled = syms[:100]
            for sym in sampled:
                try:
                    # P1-Q22-fix: 同 hs300，动态计算 start_date
                    bars = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
                    if bars is not None and len(bars) > 1:
                        pct = (float(bars["收盘"].iloc[-1]) / float(bars["收盘"].iloc[-2]) - 1) * 100
                        dfs.append({"symbol": sym, "pct": round(pct, 2)})
                except Exception as e:
                    logging.getLogger(__name__).error(f"[hot_stocks] 操作失败: {e}", exc_info=True)
            if not dfs:
                return {"error": "no_data"}
            df = pd.DataFrame(dfs).sort_values("pct", ascending=False)
            return {
                "top_gainers": df.head(n).to_dict("records"),
                "top_losers": df.tail(n).sort_values("pct").to_dict("records"),
                "median_pct": round(float(df["pct"].median()), 2),
                "coverage": f"按权重排序前{len(sampled)}只/{total}只({len(sampled)/max(total,1):.0%})近似口径",
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_zz500_symbols(self):
        try:
            import akshare as ak
            idx_weight = ak.index_stock_cons_weight_csindex("000905")
            return list(idx_weight["成分券代码"].values)[:500]
        except Exception:
            return []

    def zz1000_top_bottom(self, n: int = 50) -> dict:
        """中证1000成分涨跌前50 (V5.2 fix: akshare实现)

        P2-Q22-fix(M234): 口径注明——仅覆盖按权重排序前100只成分（近似样本，
        约10%覆盖率），非全量1000只计算；end_date 已由 P1 修复统一为
        datetime.now()（原硬编码 20261231 未来日期已移除），输出 coverage 字段注明。
        """
        try:
            import akshare as ak
            idx_weight = ak.index_stock_cons_weight_csindex("000852")
            all_syms = list(idx_weight["成分券代码"].values)
            total = len(all_syms)
            syms = all_syms[:100]
            dfs = []
            for sym in syms:
                try:
                    # P1-Q22-fix: 与 hs300/zz500 一致，动态计算起止日期(原硬编码 2026 年底后失效)
                    bars = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
                    if bars is not None and len(bars) > 1:
                        pct = (float(bars["收盘"].iloc[-1]) / float(bars["收盘"].iloc[-2]) - 1) * 100
                        dfs.append({"symbol": sym, "pct": round(pct, 2)})
                except Exception as e:
                    logging.getLogger(__name__).error(f"[hot_stocks] 操作失败: {e}", exc_info=True)
            if not dfs:
                return {"error": "no_data"}
            df = pd.DataFrame(dfs).sort_values("pct", ascending=False)
            return {
                "top_gainers": df.head(n).to_dict("records"),
                "top_losers": df.tail(n).sort_values("pct").to_dict("records"),
                "median_pct": round(float(df["pct"].median()), 2),
                "coverage": f"按权重排序前{len(syms)}只/{total}只({len(syms)/max(total,1):.0%})近似口径",
            }
        except Exception as e:
            return {"error": str(e)}

    def mid_large_risk_ratio(self, top_ret: float, bottom_ret: float) -> float:
        """中大市值风险比
        
        > 2: 上涨力量明显占优
        1-2: 上涨略占优
        0.5-1: 下跌力量更强
        < 0.5: 单边下跌
        """
        return round(top_ret / max(abs(bottom_ret), 0.1), 2)

    def concentration(self, top10_ret: float, total_ret: float) -> float:
        """涨幅集中度（前10贡献占比）"""
        return round(top10_ret / max(total_ret, 0.01), 2)

    def mechanism_for_top_gainers(self, n: int = 50) -> str:
        """涨幅前50原因归因"""
        return ("涨幅前50分析：通常分布在当前主线行业（如AI/半导体/红利等）。"
                "需结合行业分布判断是行业性上涨还是个股独立行情。"
                "若前10中有8只来自同行业，说明是行业β驱动；"
                "若分散在不同行业，说明是个股α驱动。")

    def mechanism_for_top_losers(self, n: int = 50) -> str:
        """跌幅前50原因归因"""
        return ("跌幅前50分析：需区分是系统性下跌还是个股利空。"
                "若跌幅前50集中在同一行业，警惕行业风险；"
                "若分散在各行各业，可能是市场整体调整。")


class AbnormalMoveDetection:
    """异常波动检测"""

    def price_surge(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """价格异常拉升（超过threshold个标准差）"""
        if len(returns) < 20:
            return pd.Series(0, index=returns.index)
        std = returns.rolling(60).std().clip(lower=1e-8)
        return (returns / std > threshold).astype(int)

    def price_crash(self, returns: pd.Series, threshold: float = -3.0) -> pd.Series:
        """价格异常暴跌"""
        if len(returns) < 20:
            return pd.Series(0, index=returns.index)
        std = returns.rolling(60).std().clip(lower=1e-8)
        return (returns / std < threshold).astype(int)

    def volume_surge(self, volume: pd.Series, ma_period: int = 20,
                      threshold: float = 5.0) -> pd.Series:
        """成交量异常放大"""
        ma = volume.rolling(ma_period).mean().clip(lower=1e-8)
        return (volume / ma > threshold).astype(int)

    def abnormal_mechanism(self) -> str:
        """异常波动机制解释"""
        return ("异常波动通常由以下因素之一引发：\n"
                "1. 业绩预告/财报发布（超预期/低于预期）\n"
                "2. 重大政策发布（产业政策/货币政策/监管变化）\n"
                "3. 大额交易（大宗交易/股东减持/增持）\n"
                "4. 市场传闻/媒体报道\n"
                "5. 技术性因素（爆仓/强平/ETF调仓）")


class HotStockMechanismAnalysis:
    """热股机制分析——解释为什么这些股票成为热股"""

    def industry_catalyst(self, industry: str) -> str:
        """行业催化剂分析"""
        catalysts = {
            "电子": "AI/半导体周期上行、消费电子复苏、国产替代",
            "计算机": "AI应用落地、信创政策、数据要素",
            "医药生物": "创新药管线进展、集采政策缓和、老龄化需求",
            "电力设备": "光伏/风电装机加速、电网投资、储能",
            "食品饮料": "消费复苏、CPI回升、提价周期",
            "银行": "高股息防御、息差企稳、经济复苏预期",
            "非银金融": "资本市场改革、成交活跃度提升",
            "有色金属": "全球大宗商品价格、新能源金属需求",
            "煤炭": "能源安全、高股息、冬季供暖需求",
            "基础化工": "油价传导、下游需求复苏",
        }
        return catalysts.get(industry, "行业自身基本面变化")

    def capital_flow_driver(self, fund_flow_type: str = "主力") -> str:
        """资金驱动分析"""
        drivers = {
            "主力": "机构大单买入，有持续性",
            "游资": "短线资金博弈，快进快出",
            "北向": "外资持续流入/流出",
            "散户": "跟风买入，情绪驱动",
        }
        return drivers.get(fund_flow_type, "资金驱动")

    def sentiment_driver(self, hot_level: float) -> str:
        """情绪驱动分析"""
        if hot_level > 90:
            return "市场极度关注，讨论热度高"
        elif hot_level > 70:
            return "市场关注度高，有持续话题性"
        else:
            return "正常关注度"

    def fundamental_driver(self, earnings_surprise: float = 0) -> str:
        """基本面驱动分析"""
        if earnings_surprise > 20:
            return "业绩大幅超预期"
        elif earnings_surprise < -20:
            return "业绩不达预期"
        else:
            return "基本面稳定"

    def dominance_factor(self, analysis: dict) -> str:
        """判断主导因素"""
        if not analysis:
            return "无法判断"
        factors = {"行业催化剂": 0, "资金推动": 0, "情绪驱动": 0, "基本面": 0, "技术面": 0}
        # 根据分析内容给分
        return max(factors, key=factors.get)

    def narrative_summary(self) -> str:
        """今日市场叙事"""
        return "今日市场以XX为主线，资金从XX流向XX，整体情绪XX。"


class HotStockReport:
    """热股分析完整报告"""

    def __init__(self):
        self.ranking = HotStockRanking()
        self.movers = MidLargeCapMovers()
        self.mechanism = HotStockMechanismAnalysis()

    def full_report(self) -> str:
        """完整热股分析报告

        P2-Q22-fix(M235): 原用单列 mock 价格序列伪造等权收益率且引用不存在
        的 change_pct/heat_score 列（top_hot 实际返回 pct/rank）。
        现直接用 top_hot() 返回的真实 pct 列计算等权均值；top_hot 数据源
        失败时（返回含 error 列的表）显式输出提示而非 KeyError。
        """
        hot = self.ranking.top_hot(50)
        has_real_data = (not hot.empty and "symbol" in hot.columns and "pct" in hot.columns)
        if has_real_data:
            hot_ret = round(pd.to_numeric(hot["pct"], errors="coerce").mean(), 2)
        else:
            hot_ret = 0.0

        top3 = hot.head(3) if has_real_data else pd.DataFrame()
        lines = [
            "=" * 55,
            "【热股分析报告】",
            "=" * 55,
            "",
            f"  前50热股等权涨跌幅: {hot_ret:.2f}%",
            "",
            "【TOP 10热股】",
        ]
        if has_real_data:
            for i, (_, row) in enumerate(hot.head(10).iterrows()):
                pct = row.get("pct", 0)
                pct = float(pct) if pd.notna(pct) else 0.0
                lines.append(f"  #{i+1} {row['symbol']} {row['name']} {pct:+.2f}%")
        else:
            note = hot.get("error", [""]).iloc[0] if "error" in hot.columns else "无热股数据"
            lines.append(f"  ⚠️ 热股数据源不可用: {note}")

        lines.extend([
            "",
            "【中大市值涨跌对比】",
            "  HS300成分涨跌前50风险比: —",
            "  ZZ500成分涨跌前50风险比: —",
            "",
            "【机制解读】",
            f"  {self.mechanism.narrative_summary()}",
            "",
            "=" * 55,
        ])
        return "\n".join(lines)

    def daily_brief(self) -> str:
        """热股简报"""
        return "【热股简报】" + self.mechanism.narrative_summary()


__all__ = [
    "HotStockRanking", "MidLargeCapMovers",
    "AbnormalMoveDetection", "HotStockMechanismAnalysis",
    "HotStockReport",
]
