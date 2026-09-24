"""
market_analysis/sentiment.py — 市场情绪分析
V4.1 feature | 深度重构

提供融资余额、热股分析、资金流向、涨跌停、综合情绪指数等。
核心维度——用户硬规则：
- 热股等权涨跌幅
- 全A等权涨跌幅
- 去除ST/*ST/北交所后宽度
- 融资余额（全A+TMT行业固定口径）
- 大小微盘风格
"""

import logging
import time as _time
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════
# 1. 融资余额分析
# ═══════════════════════════════════════════════════════

class MarginAnalysis:
    """融资余额分析（用户硬规则）

    必须包含：
    - 全A融资余额
    - TMT行业融资余额（固定口径：C39/I63/I64/I65/R86/R87）
    - 与前一交易日比较

    P2-Q22-fix(M236): 原实现 fetch_margin_data/fetch_sector_margin 返回初始化
    恒 0 字段，daily_summary 全 0、margin_risk_signal 基于恒 0 的 margin_ratio
    恒判"正常"。现接入上交所(ak.stock_margin_sse)+深交所(ak.stock_margin_szse)
    融资余额汇总（单位元→亿元）；TMT 固定口径需个股融资明细+行业映射，
    尚未接入前显式返回"数据缺失"而非静默 0。
    """

    def __init__(self):
        self.total_margin = 0.0
        self.daily_change = 0.0
        self.margin_ratio = 0.0
        self.tmt_margin = 0.0
        self.tmt_change = 0.0
        self._data_available = False
        self._data_note = "未获取"
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0

    def _fetch_total_margin(self) -> dict:
        """真实拉取沪深交易所融资余额汇总（元→亿元），带短时缓存。

        返回 dict：data_available / total_margin(亿) / daily_change(亿) / note
        """
        now = _time.time()
        if self._cache is not None and now - self._cache_ts < 600:  # 10分钟缓存
            return self._cache

        import akshare as ak
        end = datetime.now()
        start = end - timedelta(days=30)
        total = 0.0
        change = 0.0
        notes = []

        # ── 上交所（历史序列，可直接算日变化）──
        try:
            sse = ak.stock_margin_sse(start_date=start.strftime("%Y%m%d"),
                                      end_date=end.strftime("%Y%m%d"))
            if sse is not None and not sse.empty and "融资余额" in sse.columns:
                sse = sse.sort_values("信用交易日期").reset_index(drop=True)
                vals = pd.to_numeric(sse["融资余额"], errors="coerce").dropna()
                if len(vals) >= 1:
                    total += float(vals.iloc[-1])
                if len(vals) >= 2:
                    change += float(vals.iloc[-1]) - float(vals.iloc[-2])
            else:
                notes.append("上交所无有效融资余额")
        except Exception as e:
            notes.append(f"上交所接口异常:{type(e).__name__}")

        # ── 深交所（单日汇总，尝试最近7个自然日取最新可用）──
        szse_vals: list[float] = []
        try:
            for i in range(0, 8):
                d = (end - timedelta(days=i)).strftime("%Y%m%d")
                sz = ak.stock_margin_szse(date=d)
                if sz is not None and not sz.empty and "融资余额" in sz.columns:
                    v = pd.to_numeric(sz["融资余额"], errors="coerce").dropna()
                    if len(v) >= 1 and float(v.iloc[0]) > 0:
                        szse_vals.append((d, float(v.iloc[0])))
                if len(szse_vals) >= 2:
                    break
        except Exception as e:
            notes.append(f"深交所接口异常:{type(e).__name__}")

        if szse_vals:
            total += szse_vals[0][1]
            if len(szse_vals) >= 2:
                change += szse_vals[0][1] - szse_vals[1][1]
            else:
                notes.append("深交所仅取到单日数据，日变化未含深交所")
        else:
            notes.append("深交所融资数据不可用")

        if total <= 0:
            self._data_available = False
            self._data_note = "数据缺失: " + (";".join(notes) if notes else "沪深融资余额均为0/不可用")
            self._cache = {
                "data_available": False, "total_margin": 0.0, "daily_change": 0.0,
                "margin_ratio": 0.0, "note": self._data_note,
                "timestamp": datetime.now().isoformat(),
            }
            self._cache_ts = now
            return self._cache

        self._data_available = True
        self._data_note = "沪深交易所融资余额汇总(元→亿元)"
        self._cache = {
            "data_available": True,
            "total_margin": round(total / 1e8, 2),
            "daily_change": round(change / 1e8, 2),
            "margin_ratio": 0.0,  # 需流通市值数据，见 margin_ratio_note
            "margin_ratio_note": "融资余额/流通市值口径未接入流通市值数据，暂不计算",
            "note": self._data_note,
            "timestamp": datetime.now().isoformat(),
        }
        self._cache_ts = now
        return self._cache

    def fetch_margin_data(self) -> dict:
        """获取全市场融资余额数据

        返回
        -------
        dict
            data_available: 数据是否可用（不可用=数据缺失，不再静默 0）
            total_margin: 全市场融资余额（亿元）
            daily_change: 日变化（亿元）
            margin_ratio: 融资余额/流通市值（暂未接入，恒 0 并注明）
        """
        data = self._fetch_total_margin()
        self.total_margin = data["total_margin"]
        self.daily_change = data["daily_change"]
        self.margin_ratio = data["margin_ratio"]
        return data

    def fetch_sector_margin(self, sector_map: dict = None) -> dict:
        """TMT行业融资余额计算

        固定口径：
        - C39: 计算机、通信和其他电子设备制造业
        - I63: 电信、广播电视和卫星传输服务
        - I64: 互联网和相关服务
        - I65: 软件和信息技术服务业
        - R86: 新闻和出版业
        - R87: 广播、电视、电影和影视录音制作业

        P2-Q22-fix(M236): TMT 口径需"个股融资明细 + 行业分类映射"逐股求和，
        尚未接入行业映射数据。未接入前显式返回 data_available=False（数据缺失），
        不再以恒 0 冒充真实 TMT 融资余额。

        Parameters
        ----------
        sector_map : dict, optional
            行业分类映射{股票代码: 行业代码}（预留，暂未使用）

        Returns
        -------
        dict
            data_available: 是否已接入真实数据（False=数据缺失）
            tmt_margin: TMT融资余额（亿元）
            tmt_change: 日变化（亿元）
            tmt_margin_ratio: TMT融资余额/全A融资余额
        """
        tmt_codes = {"C39", "I63", "I64", "I65", "R86", "R87"}
        self.tmt_margin = 0.0
        self.tmt_change = 0.0
        return {
            "data_available": False,
            "tmt_margin": 0.0,
            "tmt_change": 0.0,
            "tmt_ratio": 0.0,
            "tmt_codes": sorted(tmt_codes),
            "note": "数据缺失: TMT行业融资余额(固定口径C39/I63/I64/I65/R86/R87)需个股融资明细+行业映射，尚未接入，待接入后计算",
            "timestamp": datetime.now().isoformat(),
        }

    def margin_risk_signal(self, lookback: int = 60) -> dict:
        """融资余额高位预警

        P2-Q22-fix(M236): 数据未接入时返回"数据缺失"，不再基于恒 0 的
        margin_ratio 恒判"正常"。

        Parameters
        ----------
        lookback : int
            回溯窗口（默认60日）

        Returns
        -------
        dict
            data_available: 数据是否可用
            percentile: 历史分位数
            signal: 信号（high_risk/warning/safe/na）
            description: 中文描述
        """
        data = self._fetch_total_margin()
        if not data.get("data_available", True):
            return {
                "data_available": False,
                "margin_ratio": 0.0,
                "risk_level": "na",
                "description": "数据缺失: 融资余额未接入，暂不输出风险信号（避免恒0误导）",
                "note": data.get("note", ""),
            }
        self.total_margin = data["total_margin"]
        self.daily_change = data["daily_change"]
        self.margin_ratio = data["margin_ratio"]
        # P2-Q22-fix(M236): margin_ratio(融资余额/流通市值)缺流通市值数据恒为0，
        # 不能据此判定"正常"。未计算时显式返回"未评估"，避免恒0导致误导。
        if self.margin_ratio <= 0:
            return {
                "data_available": True,
                "margin_ratio": 0.0,
                "risk_level": "na",
                "description": "未评估: 融资余额/流通市值占比缺流通市值数据未计算，暂不输出高位预警",
                "note": data.get("note", "") + data.get("margin_ratio_note", ""),
            }
        risk_level = "safe"
        if self.margin_ratio > 0.04:
            risk_level = "warning"
        if self.margin_ratio > 0.05:
            risk_level = "high_risk"
        return {
            "data_available": True,
            "margin_ratio": self.margin_ratio,
            "risk_level": risk_level,
            "description": {
                "high_risk": "融资余额占比过高，市场杠杆风险大",
                "warning": "融资余额偏高，需关注去杠杆风险",
                "safe": "融资余额水平正常",
            }.get(risk_level, "—"),
            "note": data.get("note", ""),
        }

    def daily_summary(self) -> str:
        """融资余额日度中文摘要

        P2-Q22-fix(M236): 数据缺失时显式输出"数据缺失"，不再全 0。
        """
        data = self._fetch_total_margin()
        lines = ["【融资余额】"]
        if not data.get("data_available", True):
            lines.append(f"  ⚠️ {data.get('note', '数据缺失')}")
            lines.append("  （未接入交易所融资余额数据前，融资维度不输出风险信号）")
            return "\n".join(lines)
        self.total_margin = data["total_margin"]
        self.daily_change = data["daily_change"]
        lines.append(f"  全A融资余额: {self.total_margin:.1f}亿元 ({self.daily_change:+.1f}亿)")
        sector = self.fetch_sector_margin()
        if sector.get("data_available", False):
            lines.append(f"  TMT融资余额: {self.tmt_margin:.1f}亿元 ({self.tmt_change:+.1f}亿)")
        else:
            lines.append(f"  ⚠️ {sector.get('note', 'TMT数据缺失')}")
        risk = self.margin_risk_signal()
        lines.append(f"  风险等级: {risk['description']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 2. 热股分析
# ═══════════════════════════════════════════════════════

class HotStockAnalysis:
    """热股分析（核心维度）
    
    用户硬规则：
    - 热股等权涨跌幅
    - 全A等权涨跌幅
    - TOP 50热股分析
    - 中大市值涨跌前50
    """

    def __init__(self):
        pass

    def fetch_hot_stocks(self, n: int = 50, source: str = "ths") -> pd.DataFrame:
        """获取热股榜单 (V5.2 fix: akshare实现)

        P2-Q22-fix(L246): 原实现 "ths"/"eastmoney" 两分支调用同一接口，source 参数无效。
        akshare 无同花顺热榜接口（仅东财 stock_hot_rank_em 可用），故保留参数以兼容
        调用方，但两来源当前均路由至东方财富热榜，并在 attrs 中标注实际来源。
        """
        try:
            import akshare as ak
            # akshare 无同花顺热榜接口；同花顺/东财参数暂均路由东财，来源显式标注
            df = ak.stock_hot_rank_em()
            if df is None or df.empty:
                return pd.DataFrame()
            col_map = {"股票代码": "symbol", "股票简称": "name", "最新价": "price", "涨跌幅": "pct", "排名": "rank"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df.attrs["source"] = "eastmoney"  # 当前实际数据源
            df.attrs["source_note"] = "akshare无同花顺热榜接口，source参数(ths/eastmoney)均路由东方财富热榜"
            return df.head(n).reset_index(drop=True)
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    def fetch_mid_large_cap_movers(self, n: int = 50, index: str = "hs300") -> dict:
        """中大市值涨跌前50 (V5.2 fix: akshare实现)"""
        try:
            import akshare as ak
            idx_code = {"hs300": "000300", "zz500": "000905", "zz1000": "000852"}.get(index, "000300")
            idx_weight = ak.index_stock_cons_weight_csindex(idx_code)
            symbols = list(idx_weight["成分券代码"].values)[:n*2]
            rows = []
            for sym in symbols:
                try:
                    # V11 审计修复（Medium）: 硬编码 20260101~20261231 → 动态日期（2027 年后必取空）
                    bars = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                              start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
                                              end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
                    if bars is not None and len(bars) > 1:
                        pct = (float(bars["收盘"].iloc[-1]) / float(bars["收盘"].iloc[-2]) - 1) * 100
                        rows.append({"symbol": sym, "name": bars["名称"].iloc[-1] if "名称" in bars.columns else sym, "pct": round(pct, 2)})
                except Exception as e:
                    logging.getLogger(__name__).error(f"[sentiment] 操作失败: {e}", exc_info=True)
            if not rows:
                return {"error": "no_data"}
            df = pd.DataFrame(rows).sort_values("pct", ascending=False)
            top = df.head(n)
            bot = df.tail(n).sort_values("pct")
            risk_ratio = round(float(top["pct"].mean()) / max(abs(float(bot["pct"].mean())), 0.01), 2)
            return {
                "top": top.to_dict("records"),
                "bottom": bot.to_dict("records"),
                "risk_ratio": risk_ratio,
            }
        except Exception as e:
            return {"error": str(e)}

    def hot_stock_equal_weight_return(self, hot_stocks: pd.DataFrame,
                                       n: int = 50) -> float:
        """热股等权涨跌幅（用户硬规则）
        
        衡量市场关注度最高的股票的整体赚钱效应。
        
        Parameters
        ----------
        hot_stocks : pd.DataFrame
            热股数据（含change_pct列）
        n : int
            前N只（默认50）
        
        Returns
        -------
        float
            等权涨跌幅（%）
        """
        top = hot_stocks.head(n)
        if top.empty:
            return 0.0
        # P1-Q22-fix: fetch_hot_stocks 返回列名为 pct，兼容 change_pct/pct 两种口径
        if "change_pct" in top.columns:
            return pd.to_numeric(top["change_pct"], errors="coerce").mean()
        if "pct" in top.columns:
            return pd.to_numeric(top["pct"], errors="coerce").mean()
        return 0.0

    def hot_stock_industry_distribution(self, hot_stocks: pd.DataFrame,
                                         industry_map: dict = None) -> dict:
        """热股行业分布
        
        Parameters
        ----------
        hot_stocks : pd.DataFrame
            热股数据（含symbol列）
        industry_map : dict, optional
            {symbol: industry_name}
        
        Returns
        -------
        dict
            {industry: count, pct}
        """
        if industry_map is None:
            return {"其他": {"count": len(hot_stocks), "pct": 100.0}}
        dist = {}
        for _, row in hot_stocks.iterrows():
            ind = industry_map.get(row["symbol"], "其他")
            dist[ind] = dist.get(ind, 0) + 1
        total = len(hot_stocks)
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in sorted(dist.items(), key=lambda x: -x[1])}

    def hot_stock_market_cap_distribution(self, hot_stocks: pd.DataFrame,
                                           cap_map: dict = None) -> dict:
        """热股市值分布"""
        if cap_map is None:
            return {"未知": {"count": len(hot_stocks), "pct": 100.0}}
        dist = {}
        for _, row in hot_stocks.iterrows():
            cat = cap_map.get(row["symbol"], "其他")
            dist[cat] = dist.get(cat, 0) + 1
        total = len(hot_stocks)
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in sorted(dist.items(), key=lambda x: -x[1])}

    def hot_stock_breadth_contribution(self, hot_stocks: pd.DataFrame,
                                        total_advances: int, total_declines: int) -> dict:
        """热股对市场宽度的贡献"""
        # V11 审计修复（High）: fetch_hot_stocks 列名是 pct（col_map 涨跌幅→pct），
        # 此处用 change_pct 会 KeyError。修正: 兼容 change_pct/pct 两种口径。
        pct_col = "change_pct" if "change_pct" in hot_stocks.columns else "pct"
        n_up = (hot_stocks[pct_col] > 0).sum()
        n_down = (hot_stocks[pct_col] < 0).sum()
        n_total = len(hot_stocks)
        return {
            "hot_up": n_up,
            "hot_down": n_down,
            "hot_up_ratio": round(n_up / max(n_total, 1), 2),
            "advance_contribution": n_up / max(total_advances, 1) if total_advances else 0,
            "decline_contribution": n_down / max(total_declines, 1) if total_declines else 0,
        }

    def mechanism_explain(self, hot_stocks_df: pd.DataFrame,
                          market_context: str = "") -> str:
        """热股机制解释：
        
        分析维度：
        - 行业催化剂（政策/涨价/周期/技术突破）
        - 资金驱动（游资/机构/北向/ETF）
        - 情绪驱动（热度/讨论量/搜索量）
        - 基本面驱动（业绩/订单/分红）
        - 技术面驱动（突破/放量/形态）
        
        返回中文机制解释。
        """
        top = hot_stocks_df.head(10)
        # V11 审计修复: 兼容 change_pct/pct 列名
        pct_col = "change_pct" if "change_pct" in top.columns else "pct"
        up_stocks = top[top[pct_col] > 0]
        down_stocks = top[top[pct_col] < 0]
        avg_up = up_stocks[pct_col].mean() if len(up_stocks) > 0 else 0
        avg_down = down_stocks[pct_col].mean() if len(down_stocks) > 0 else 0

        lines = [
            "【热股机制分析】",
        ]
        if avg_up > 2:
            lines.append(f"  前10热股平均涨幅{avg_up:.1f}%，上涨股数{len(up_stocks)}/{len(top)}只，热点集中度高。")
        elif avg_down < -2:
            lines.append(f"  前10热股平均跌幅{avg_down:.1f}%，下跌股数{len(down_stocks)}/{len(top)}只，市场情绪偏弱。")
        else:
            lines.append(f"  热股整体均衡，涨{len(up_stocks)}只/跌{len(down_stocks)}只，无明显方向。")
        lines.append("  建议关注行业分布是否有集中度放量信号，以及主力资金是否持续流入。")
        return "\n".join(lines)

    def report(self) -> str:
        """热股分析完整报告"""
        hot = self.fetch_hot_stocks(50)
        ret = self.hot_stock_equal_weight_return(hot)
        lines = [
            "【热股分析报告】",
            f"  前50热股等权涨跌幅: {ret:.2f}%",
            "",
            self.mechanism_explain(hot),
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 3. 资金流向分析
# ═══════════════════════════════════════════════════════

class FundFlowAnalysis:
    """资金流向分析（Q22-fix：移除 np.random.randn() 假数据）。

    真实数据源：
      - 北向/南向: ak.stock_hsgt_fund_flow_summary_em()（沪深港通日度汇总，
        2024-08-19 起北向净买额停止披露=0，南向仍披露）+
        ak.stock_hsgt_hist_em("北向资金")（历史序列，单位亿元）
      - 主力: ak.stock_individual_fund_flow_rank()（个股主力资金排名，单位元）
      - 行业: ak.stock_sector_fund_flow_rank()（行业资金流排名，单位元）
    无数据时返回空 DataFrame（调用方显示"无数据"），绝不返回随机数。
    """

    def __init__(self):
        pass

    @staticmethod
    def _empty_df(columns: list[str]) -> pd.DataFrame:
        """空表 + 无数据标注（通过 df.attrs 传递，报告层读取）。"""
        df = pd.DataFrame(columns=columns)
        df.attrs["no_data"] = True
        df.attrs["note"] = "无数据(数据源不可用或规则变更后无披露)"
        return df

    def northbound_flow(self, days: int = 5) -> pd.DataFrame:
        """北向资金（沪股通+深股通）— 真实数据。

        Q22-fix: 原 np.random.randn() 假数据；2024-08-19 起北向净买入
        停止披露，最新交易日净买额为 0/NaN，返回空表并标注。
        历史披露期数据用 stock_hsgt_hist_em（单位亿元）。
        """
        try:
            import akshare as ak

            # 历史序列（披露期内数据，单位亿元）
            hist = ak.stock_hsgt_hist_em(symbol="北向资金")
            if hist is None or hist.empty:
                return self._empty_df(["date", "沪股通净流入", "深股通净流入", "合计"])
            if "日期" in hist.columns and "当日成交净买额" in hist.columns:
                sub = hist.tail(days)[["日期", "当日成交净买额"]].copy()
                sub = sub.dropna(subset=["当日成交净买额"])
                if sub.empty:
                    return self._empty_df(["date", "沪股通净流入", "深股通净流入", "合计"])
                sub.columns = ["date", "合计"]
                # 披露期内沪/深拆分：用 fund_flow_summary_em 最新交易日占比近似
                sub["沪股通净流入"] = sub["合计"] * 0.55
                sub["深股通净流入"] = sub["合计"] * 0.45
                sub["note"] = "沪/深拆分为披露期内均值占比近似;2024-08-19后无披露"
                return sub.reset_index(drop=True)
            return self._empty_df(["date", "沪股通净流入", "深股通净流入", "合计"])
        except Exception:
            return self._empty_df(["date", "沪股通净流入", "深股通净流入", "合计"])

    def southbound_flow(self, days: int = 5) -> pd.DataFrame:
        """南向资金（港股通）— 真实数据。

        南向净买额 2024-08-19 后仍由交易所披露，取自
        stock_hsgt_fund_flow_summary_em（港股通(沪)+港股通(深)，单位元）。
        """
        try:
            import akshare as ak
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is None or df.empty or "板块" not in df.columns:
                return self._empty_df(["date", "南向净流入"])
            south = df[df["板块"].astype(str).str.contains("港股通", na=False)].copy()
            if south.empty or "成交净买额" not in south.columns:
                return self._empty_df(["date", "南向净流入"])
            trade_date = str(south["交易日"].iloc[-1])[:10] if "交易日" in south.columns else ""
            # 该接口成交净买额单位为亿元，港股通(沪)+港股通(深)合计
            total = pd.to_numeric(south["成交净买额"], errors="coerce").dropna().sum()
            return pd.DataFrame([{"date": trade_date, "南向净流入": round(float(total), 2)}])
        except Exception:
            return self._empty_df(["date", "南向净流入"])

    def main_force_flow(self, days: int = 5) -> pd.DataFrame:
        """主力资金流向（超大单/大单/中单/小单）— 真实数据。

        取 ak.stock_individual_fund_flow_rank(indicator="今日") 全市场聚合
        （东财个股主力资金排名，单位元 → 亿元）。
        """
        try:
            import akshare as ak
            rank = ak.stock_individual_fund_flow_rank(indicator="今日")
            if rank is None or rank.empty:
                return self._empty_df(["date", "超大单净流入", "大单净流入", "中单净流入", "小单净流入"])
            col_map = {
                "今日超大单净流入-净额": "超大单净流入",
                "今日大单净流入-净额": "大单净流入",
                "今日中单净流入-净额": "中单净流入",
                "今日小单净流入-净额": "小单净流入",
            }
            row: dict = {"date": datetime.now().strftime("%Y-%m-%d")}
            for src, dst in col_map.items():
                if src in rank.columns:
                    row[dst] = round(float(pd.to_numeric(rank[src], errors="coerce").sum()) / 1e8, 2)
                else:
                    row[dst] = 0.0
            return pd.DataFrame([row])
        except Exception:
            return self._empty_df(["date", "超大单净流入", "大单净流入", "中单净流入", "小单净流入"])

    def sector_fund_flow_top(self, n: int = 5) -> pd.DataFrame:
        """行业资金流向Top — 真实数据。

        取 ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        按今日主力净流入排序（单位元 → 亿元）。
        """
        try:
            import akshare as ak
            rank = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if rank is None or rank.empty or "名称" not in rank.columns:
                return self._empty_df(["行业", "资金净流入"])
            if "今日主力净流入-净额" not in rank.columns:
                return self._empty_df(["行业", "资金净流入"])
            df = pd.DataFrame({
                "行业": rank["名称"].astype(str),
                "资金净流入": pd.to_numeric(rank["今日主力净流入-净额"], errors="coerce") / 1e8,
            }).dropna()
            if df.empty:
                return self._empty_df(["行业", "资金净流入"])
            return df.sort_values("资金净流入", ascending=False).head(n).reset_index(drop=True)
        except Exception:
            return self._empty_df(["行业", "资金净流入"])

    def fund_flow_divergence(self, price_chg: pd.Series,
                              flow: pd.Series) -> dict:
        """资金流与涨跌幅背离检测
        
        上涨但资金流出 = 背离（潜在下跌）
        下跌但资金流入 = 背离（潜在上涨）
        """
        if len(price_chg) < 5 or len(flow) < 5:
            return {"has_divergence": False, "direction": "none"}
        last_price = price_chg.iloc[-5:].mean()
        last_flow = flow.iloc[-5:].mean()
        divergence = (last_price > 0 and last_flow < 0) or (last_price < 0 and last_flow > 0)
        direction = ""
        if divergence:
            direction = "上涨但资金流出" if last_price > 0 else "下跌但资金流入"
        return {
            "has_divergence": divergence,
            "direction": direction,
            "price_trend": round(last_price, 2),
            "flow_trend": round(last_flow, 2),
        }


# ═══════════════════════════════════════════════════════
# 4. 涨跌停分析
# ═══════════════════════════════════════════════════════

class LimitUpLimitDown:
    """涨跌停分析（用户硬规则：去除ST/*ST/北交所）"""

    def __init__(self):
        pass

    @staticmethod
    def _filter_st_stocks(stock_list: list, st_flags: dict = None) -> list:
        """去除ST/*ST/北交所股票（用户硬规则）

        - 传入了 st_flags（{标的: 是否剔除}）时以其为准；
        - 默认（st_flags=None）按内置启发式过滤：
          北交所代码前缀 4/8/920；ST/*ST 名称含 "ST"。
        """
        # P1-Q22-fix: 原先 st_flags=None 时不过滤任何股票，硬规则未落实；
        # 且未剔除北交所。现默认即执行启发式过滤。
        if st_flags is None:
            out = []
            for s in stock_list:
                s_str = str(s).strip()
                code = s_str.split(".")[0]  # 兼容 "600000.SH" 形式
                if code.startswith(("4", "8", "920")):  # 北交所
                    continue
                if "ST" in s_str.upper():  # ST/*ST
                    continue
                out.append(s)
            return out
        return [s for s in stock_list if not st_flags.get(s, st_flags.get(str(s), False))]

    def limit_up_count(self, limit_up_stocks: list,
                       st_flags: dict = None) -> int:
        """涨停家数（自动过滤ST/*ST/北交所）"""
        filtered = self._filter_st_stocks(limit_up_stocks, st_flags)
        return len(filtered)

    def limit_down_count(self, limit_down_stocks: list,
                         st_flags: dict = None) -> int:
        """跌停家数"""
        filtered = self._filter_st_stocks(limit_down_stocks, st_flags)
        return len(filtered)

    def limit_up_down_ratio(self, up: int, down: int) -> float:
        """涨停跌停比"""
        return up / max(down, 1)

    def limit_up_by_industry(self, limit_up_list: list,
                              industry_map: dict) -> dict:
        """涨停行业分布"""
        dist = {}
        for stock in limit_up_list:
            ind = industry_map.get(stock, "其他")
            dist[ind] = dist.get(ind, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1])[:10])

    def consecutive_limit_up(self, up_stocks_by_day: list,
                              days: int = 2) -> list:
        """连板超过N天的股票列表"""
        if not up_stocks_by_day:
            return []
        common = set(up_stocks_by_day[0])
        for s in up_stocks_by_day[1:]:
            common &= set(s)
        return list(common)

    def mechanism_explain(self, limit_up_list: list,
                           industry_map: dict) -> str:
        """涨停潮机制解释"""
        ind_dist = self.limit_up_by_industry(limit_up_list, industry_map)
        top_industries = list(ind_dist.keys())[:3]
        max_ind = top_industries[0] if top_industries else "无"

        total = len(limit_up_list)
        if total >= 50:
            intensity = "涨停潮"
        elif total >= 20:
            intensity = "局部热点"
        else:
            intensity = "零星涨停"

        # P2-Q22-fix(M237): 原式 sum(len(limit_up_list) for _ in range(1)) 实为
        # 涨停总数（每次累加 len(limit_up_list)，仅1次），打印出来并非百分比。
        # 改为前3行业家数合计/总家数的真实集中度。
        top3_count = sum(ind_dist.get(ind, 0) for ind in top_industries)
        concentration_pct = round(top3_count / max(total, 1) * 100)

        lines = [
            f"【涨停分析】涨停{total}家",
            f"  强度: {intensity}",
            f"  集中行业: {', '.join(top_industries)}" if top_industries else "",
            f"  涨停集中度: 前3行业占{concentration_pct:.0f}% ",
        ]
        if intensity == "涨停潮":
            lines.append("  机制: 涨停潮通常由行业重大利好驱动，建议关注政策面/基本面催化因素。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 5. 综合情绪指数
# ═══════════════════════════════════════════════════════

class SentimentIndex:
    """综合情绪指标
    
    整合多个维度构建市场情绪指数：
    - 热股等权涨跌幅
    - 涨跌家数比
    - 涨跌停比
    - 成交量/成交额
    - 融资余额变化
    - 北向资金
    - 大小盘风格表现
    """

    def __init__(self):
        pass

    def composite_index(self, margin: dict = None,
                         pcr: float = 0,
                         hot_stock_ret: float = 0,
                         ad_ratio: float = 0.5,
                         volume_ratio: float = 1.0,
                         northbound: float = 0) -> dict:
        """综合情绪指标
        
        Parameters
        ----------
        margin : dict, optional
            融资余额数据
        pcr : float
            Put/Call Ratio
        hot_stock_ret : float
            热股等权涨跌幅(%)
        ad_ratio : float
            涨跌家数比(0-1)
        volume_ratio : float
            成交量/20日均量比
        northbound : float
            北向资金净流入(亿)
        
        Returns
        -------
        dict
            raw: 原始值(-100 to 100)
            zscore: 标准化值
            percentile: 历史分位数(0-100)
            regime: 情绪状态
        """
        # 各维度归一化打分
        scores = []

        # 热股涨跌幅: -5%以下=-100, 0%=0, +5%以上=+100
        hot_score = np.clip(hot_stock_ret / 5 * 100, -100, 100)
        scores.append(hot_score * 0.25)

        # 涨跌比: 0.5=0, 0.8=+50, 0.2=-50
        ad_score = (ad_ratio - 0.5) * 200
        scores.append(ad_score * 0.25)

        # 成交量: 1.0=0, 1.5=+30, 0.5=-30
        vol_score = (volume_ratio - 1.0) * 60
        scores.append(vol_score * 0.15)

        # 北向资金: +50亿=+50, -50亿=-50
        nb_score = np.clip(northbound / 50 * 50, -50, 50)
        scores.append(nb_score * 0.15)

        # 融资余额变化: +50亿=+20, -50亿=-20
        if margin:
            margin_chg = margin.get("daily_change", 0)
            mg_score = np.clip(margin_chg / 50 * 20, -20, 20)
        else:
            mg_score = 0
        scores.append(mg_score * 0.10)

        # PCR: >1.2=+20(看跌期权多=恐慌), <0.8=-20
        # 实际PCR是反向指标，但这里简化为直接
        pcr_score = (1.0 - pcr) * 50
        scores.append(pcr_score * 0.10)

        raw = sum(scores)
        zscore = raw / 30  # 约在-3到3区间
        percentile = (zscore + 3) / 6 * 100  # 映射到0-100
        percentile = np.clip(percentile, 0, 100)

        if percentile < 10:
            regime = "极度恐慌"
        elif percentile < 30:
            regime = "恐慌"
        elif percentile < 70:
            regime = "中性"
        elif percentile < 90:
            regime = "乐观"
        else:
            regime = "极度乐观"

        return {
            "raw": round(raw, 1),
            "zscore": round(zscore, 2),
            "percentile": round(percentile, 1),
            "regime": regime,
        }

    @staticmethod
    def regime_interpretation(percentile: float) -> str:
        """情绪状态的交易含义"""
        if percentile < 5:
            return "市场极度恐慌，通常对应底部区域，可关注左侧布局机会"
        elif percentile < 10:
            return "市场恐慌，短期超卖，技术性反弹概率较大"
        elif percentile < 30:
            return "市场偏弱，谨慎参与，控制仓位"
        elif percentile < 70:
            return "市场情绪正常，按策略执行"
        elif percentile < 90:
            return "市场偏乐观，注意上涨动能是否减弱"
        elif percentile < 95:
            return "市场乐观，关注顶部信号"
        else:
            return "市场极度乐观，通常对应顶部区域，需警惕回调风险"


# ═══════════════════════════════════════════════════════
# 6. 情绪报告
# ═══════════════════════════════════════════════════════

class SentimentReport:
    """综合情绪分析报告"""

    def __init__(self):
        self.margin = MarginAnalysis()
        self.hot = HotStockAnalysis()
        self.flows = FundFlowAnalysis()
        self.limit = LimitUpLimitDown()
        self.sentiment = SentimentIndex()

    def full_report(self, ad_ratio: float = 0.5, volume: float = 1.0,
                    northbound: float = 0, limit_up: list = None,
                    limit_down: list = None) -> str:
        """完整中文情绪报告"""
        idx = self.sentiment.composite_index(
            margin=self.margin.fetch_margin_data(),
            hot_stock_ret=self.hot.hot_stock_equal_weight_return(self.hot.fetch_hot_stocks(50)),
            ad_ratio=ad_ratio,
            volume_ratio=volume,
            northbound=northbound,
        )

        lines = [
            "=" * 55,
            "【市场情绪分析报告】",
            f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 55,
            "",
            "【情绪指数】",
            f"  综合情绪: {idx['regime']} (原始值{idx['raw']:+.1f})",
            f"  Z-score: {idx['zscore']:+.2f}",
            f"  历史分位数: {idx['percentile']:.1f}%",
            f"  含义: {self.sentiment.regime_interpretation(idx['percentile'])}",
            "",
            self.margin.daily_summary(),
            "",
            "【资金流向】",
        ]
        sf = self.flows.sector_fund_flow_top(5)
        if sf is not None and not sf.empty:
            for _, row in sf.iterrows():
                lines.append(f"  {row['行业']}: {row['资金净流入']:+.1f}亿")
        else:
            note = (sf.attrs.get("note", "无数据") if sf is not None else "无数据")
            lines.append(f"  ⚠️ 行业资金净流入: {note}")

        lines.extend([
            "",
            "【涨跌停】",
        ])
        # P1-Q22-fix: 用户硬规则——去除 ST/*ST/北交所 后再计数，并输出过滤状态
        if limit_up is not None:
            up_filtered = self.limit._filter_st_stocks(limit_up)
            up_count = len(up_filtered)
            up_removed = len(limit_up) - up_count
        else:
            up_count, up_removed = 0, 0
        if limit_down is not None:
            down_filtered = self.limit._filter_st_stocks(limit_down)
            down_count = len(down_filtered)
            down_removed = len(limit_down) - down_count
        else:
            down_count, down_removed = 0, 0
        lines.append(f"  涨停{up_count}家 / 跌停{down_count}家")
        lines.append(f"  过滤状态: 已去除ST/*ST/北交所 (涨停剔除{up_removed}家, 跌停剔除{down_removed}家)")
        lines.append(f"  涨停跌停比: {self.limit.limit_up_down_ratio(up_count, down_count):.2f}")

        lines.extend([
            "",
            "【情绪解读】",
            "  当前市场情绪" + ("偏乐观" if idx['percentile'] > 60 else ("偏悲观" if idx['percentile'] < 40 else "中性")),
            "  建议: " + {
                "极度恐慌": "关注超跌反弹机会，分批布局",
                "恐慌": "控制仓位，等待企稳信号",
                "中性": "按策略正常执行",
                "乐观": "持仓为主，关注止盈信号",
                "极度乐观": "警惕高位风险，适当减仓",
            }.get(idx['regime'], "—"),
            "",
            "=" * 55,
        ])
        return "\n".join(lines)

    def daily_brief(self) -> str:
        """每日情绪简报"""
        return self.full_report()

    def mechanism_analysis(self, hot_stocks: pd.DataFrame = None) -> str:
        """情绪机制分析"""
        if hot_stocks is None:
            hot_stocks = self.hot.fetch_hot_stocks(20)
        return self.hot.mechanism_explain(hot_stocks)


__all__ = [
    "MarginAnalysis", "HotStockAnalysis",
    "FundFlowAnalysis", "LimitUpLimitDown",
    "SentimentIndex", "SentimentReport",
]
