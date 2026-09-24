"""
exposure — 持仓因子暴露分析 (V5)

核心问题：我的持仓在哪些因子上有hidden bet？
  1. 给定持仓列表（股票代码 + 权重）
  2. 计算每只股票在主要因子上的暴露
  3. 加权合成组合级别的因子暴露
  4. 与基准（沪深300）比较，识别主动偏离

对标：Barra 因子模型中的组合暴露分析
"""

from __future__ import annotations
import logging

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from quant_system.utils import safe_float as _safe_float_impl

logger = logging.getLogger(__name__)


# 原 _safe_float 语义：不可解析/非有限值/None 均返回 NaN（表示"不可得"）
def _safe_float(value: Any) -> float:
    return _safe_float_impl(
        value, default=float("nan"), allow_bool=True,
        clean_commas=False, clean_percent=False,
    )

# Q21 修复：timezone() 需要 timedelta 参数（原 timezone(offset=...) 把 Ellipsis 当 offset，导入即崩）
CST = timezone(timedelta(hours=8))

# P1-Q21-fix(H01/H02): 东财全市场快照缓存，避免对每只股票重复抓取 stock_zh_a_spot_em
# （原实现对每只股票每个因子各抓一次，容易触发东财限流且浪费带宽）
_SPOT_SNAPSHOT: dict[str, Any] = {"df": None, "ts": 0.0}
_SPOT_SNAPSHOT_TTL = 300  # 秒

# P2-Q21-fix: 沪深300日收益率序列缓存（供各股 beta 计算复用，
# 避免每只股票各抓一次指数历史；带 TTL）
_BENCHMARK_SNAPSHOT: dict[str, Any] = {"df": None, "ts": 0.0}
_BENCHMARK_SNAPSHOT_TTL = 300  # 秒


def _get_benchmark_returns() -> pd.Series | None:
    """获取沪深300日收益率序列（带 TTL 缓存，供各股 beta 计算复用）。

    Returns:
        以 DatetimeIndex 索引的日收益率 Series；失败返回缓存旧值或 None。
    """
    now = datetime.now(CST).timestamp()
    if (
        _BENCHMARK_SNAPSHOT["df"] is not None
        and now - _BENCHMARK_SNAPSHOT["ts"] < _BENCHMARK_SNAPSHOT_TTL
    ):
        return _BENCHMARK_SNAPSHOT["df"]
    try:
        import akshare as ak
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        if idx is None or idx.empty or "close" not in idx.columns or "date" not in idx.columns:
            return None
        idx = idx.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.set_index("date").sort_index()
        ret = idx["close"].pct_change().dropna()
        _BENCHMARK_SNAPSHOT["df"] = ret
        _BENCHMARK_SNAPSHOT["ts"] = now
        return ret
    except Exception:
        # 刷新失败时复用旧快照，避免 beta 批量计算全挂
        return _BENCHMARK_SNAPSHOT["df"]


def _latest_financial_value(fin: pd.DataFrame | None, indicator: str) -> float:
    """从同花顺财务摘要 DataFrame 中取某指标"最新报告期"的数值。

    实测 akshare 1.18.64 的 stock_financial_abstract_ths 返回结构为：
      rows = 报告期（多期），columns = 指标名（如 净资产收益率/资产负债率），
      数值为带 '%' 的字符串（如 '24.64%'）。
    原代码 fin.get("市盈率-动态") 按列名取列、再 float(整列Series)：
      a) 该接口无 市盈率-动态/市净率/股息率 列 -> get 返回 None；
      b) 净资产收益率 存在但为 25 期 Series，float(Series) 抛 TypeError 被 except 吞掉 -> 恒 0。
    因此统一改为：指标名按子串模糊匹配列，取最后一个可解析数值；取不到返回 NaN。
    """
    if fin is None or fin.empty:
        return float("nan")
    col = None
    for c in fin.columns:
        if indicator in str(c):
            col = c
            break
    if col is None:
        return float("nan")
    frame = fin.copy()
    # 按"报告期"列排序（升序），保证取到的是最新报告期，不依赖接口返回的行序
    if "报告期" in frame.columns:
        frame["_rp"] = pd.to_datetime(frame["报告期"], errors="coerce")
        frame = frame.sort_values("_rp", na_position="last")
    cleaned = frame[col].astype(str).str.replace("%", "", regex=False).str.strip()
    nums = pd.to_numeric(cleaned, errors="coerce").dropna()
    if nums.empty:
        return float("nan")
    return float(nums.iloc[-1])


class FactorExposure:
    """持仓因子暴露分析引擎。

    核心假设：
      组合收益 ≈ Σ(因子暴露 × 因子收益) + 特异性收益
      好的组合 = 有意识地承担想承担的因子风险

    Attributes:
        factor_names: 支持的因子列表
    """

    # 主要风格因子（对标 Barra CNE5）
    FACTOR_NAMES = [
        "beta", "size", "value", "momentum", "quality",
        "volatility", "growth", "liquidity", "leverage",
        "dividend_yield",
    ]

    def __init__(self) -> None:
        self.factor_data: dict[str, pd.DataFrame] = {}
        self._last_update: float = 0
        # P2-Q21-fix: 单次 compute() 内复用个股历史/财务数据，避免
        # "每只股票 × 每个因子" 各抓一次 akshare（10 只持仓 ≈ 40+ 次网络请求）。
        # compute() 入口清空，保证不跨调用返回过期缓存。
        self._hist_cache: dict[str, pd.DataFrame | None] = {}
        self._fin_cache: dict[str, pd.DataFrame | None] = {}

    # ────────────────────────────────────────────────────────
    # 批量数据复用（P2-Q21-fix）
    # ────────────────────────────────────────────────────────

    def _get_hist(self, code: str) -> pd.DataFrame | None:
        """获取个股日线历史（带 compute 级缓存，供 momentum/volatility/liquidity/beta 复用）。"""
        if code in self._hist_cache:
            return self._hist_cache[code]
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            self._hist_cache[code] = df
            return df
        except Exception:
            self._hist_cache[code] = None
            return None

    def _get_financial(self, code: str) -> pd.DataFrame | None:
        """获取同花顺财务摘要（带 compute 级缓存，供 quality/leverage 复用）。"""
        if code in self._fin_cache:
            return self._fin_cache[code]
        try:
            import akshare as ak
            fin = ak.stock_financial_abstract_ths(symbol=code)
            self._fin_cache[code] = fin
            return fin
        except Exception:
            self._fin_cache[code] = None
            return None

    def _get_spot_snapshot(self) -> pd.DataFrame | None:
        """获取东财全市场快照（带 TTL 缓存，供 size/value 等因子复用）。

        stock_zh_a_spot_em 一次返回全市场 代码/名称/市盈率-动态/市净率/总市值 等列，
        相比逐股调用更省且不易限流。失败返回缓存中的旧快照或 None。
        """
        now = datetime.now(CST).timestamp()
        if (
            _SPOT_SNAPSHOT["df"] is not None
            and now - _SPOT_SNAPSHOT["ts"] < _SPOT_SNAPSHOT_TTL
        ):
            return _SPOT_SNAPSHOT["df"]
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            _SPOT_SNAPSHOT["df"] = df
            _SPOT_SNAPSHOT["ts"] = now
            return df
        except Exception:
            # 刷新失败时复用旧快照，避免因子批量计算全挂
            return _SPOT_SNAPSHOT["df"]

    def _calc_value_exposure(self, code: str) -> float:
        """计算价值因子暴露 (PE/ PB 的复合 Z-score)。

        P1-Q21-fix(H01): 原实现用 fin.get("市盈率-动态") 取同花顺财务摘要的列，
        但该接口(1.18.64)无 PE/PB 列 -> 恒 None -> 恒 0。改用东财全市场快照
        stock_zh_a_spot_em 的"市盈率-动态/市净率"列。数据不可得返回 NaN。
        """
        try:
            spot = self._get_spot_snapshot()
            if spot is None or "代码" not in spot.columns:
                return float("nan")
            row = spot[spot["代码"] == code]
            if row.empty:
                return float("nan")
            pe = _safe_float(row.iloc[0].get("市盈率-动态"))
            pb = _safe_float(row.iloc[0].get("市净率"))
            if not np.isfinite(pe) or not np.isfinite(pb) or pe <= 0 or pb <= 0:
                return float("nan")
            pe_z = -max(-3, min(3, (pe - 30) / 20))  # 低PE=高价值暴露
            pb_z = -max(-3, min(3, (pb - 3) / 2))
            return round((pe_z + pb_z) / 2, 2)
        except Exception:
            return float("nan")

    def _calc_momentum_exposure(self, code: str) -> float:
        """计算动量因子暴露 (过去12个月扣除最近1个月的收益)."""
        try:
            df = self._get_hist(code)
            if df is not None and len(df) > 250:
                close = df["收盘"].values.astype(float)
                mom_12m = close[-1] / close[-250] - 1
                mom_1m = close[-1] / close[-20] - 1 if len(df) > 20 else 0
                mom_score = (mom_12m - mom_1m) * 100  # 年化动量
                return round(max(-3, min(3, mom_score / 10)), 2)
        except Exception as e:
            logging.getLogger(__name__).error(f"[exposure] 操作失败: {e}", exc_info=True)
        return 0.0

    def _calc_size_exposure(self, code: str) -> float:
        """计算规模因子暴露 (对数市值标准化).

        P1-Q21-fix(H02): 实测 1.18.64 的 stock_zh_a_spot_em 列名为"总市值"/"流通市值"
        （无"市值"），原代码按 "市值" 精确取列 -> KeyError 被吞 -> size 恒 0。
        改为与 concentration.py get_market_cap 一致的按名称模糊匹配"总市值"。
        """
        try:
            spot = self._get_spot_snapshot()
            if spot is None or "代码" not in spot.columns:
                return float("nan")
            cap_col = None
            for c in spot.columns:
                if "总市值" in str(c):
                    cap_col = c
                    break
            if cap_col is None:
                return float("nan")
            row = spot[spot["代码"] == code]
            if row.empty:
                return float("nan")
            mkt_cap = _safe_float(row.iloc[0][cap_col])
            if not np.isfinite(mkt_cap) or mkt_cap <= 0:
                return float("nan")
            # 标准化: log(市值) -> Z-score
            log_cap = np.log(max(mkt_cap, 1))
            mean_log = np.log(1e10)
            std_log = np.log(5)
            return round(max(-3, min(3, (log_cap - mean_log) / std_log)), 2)
        except Exception:
            return float("nan")

    def _calc_beta_exposure(self, code: str) -> float:
        """计算 Beta 因子暴露 (最近250日相对沪深300，按日期对齐).

        P2-Q21-fix: 原实现取个股/指数各自最后 250 个交易日收益率做协方差，
        无日历对齐（停牌股错位）。现按日期内连接对齐后再取最近 250 个交易日。
        """
        try:
            stock = self._get_hist(code)
            idx_ret = _get_benchmark_returns()
            if stock is None or idx_ret is None or stock.empty or idx_ret.empty:
                return float("nan")
            if "日期" not in stock.columns or "收盘" not in stock.columns:
                return float("nan")
            s = stock.copy()
            s["日期"] = pd.to_datetime(s["日期"])
            s = s.set_index("日期").sort_index()
            s_ret = s["收盘"].pct_change().dropna()
            aligned = pd.DataFrame({"s": s_ret, "m": idx_ret}).dropna().iloc[-250:]
            if len(aligned) < 20:
                return float("nan")
            s_arr = aligned["s"].values
            m_arr = aligned["m"].values
            var_m = float(np.var(m_arr))
            if var_m < 1e-10:
                return float("nan")
            cov = float(np.cov(s_arr, m_arr)[0, 1])
            beta = cov / var_m
            return round(max(-3, min(3, (beta - 1) * 3)), 2)  # 相对1标准化
        except Exception:
            return float("nan")

    def _calc_volatility_exposure(self, code: str) -> float:
        """计算波动率因子暴露 (近60日年化波动率的Z-score)."""
        try:
            df = self._get_hist(code)
            if df is not None and len(df) > 60:
                ret = df["收盘"].pct_change().dropna().values[-60:]
                vol = np.std(ret) * np.sqrt(252)
                # 高波动 = 正暴露
                return round(max(-3, min(3, (vol - 0.3) / 0.1)), 2)
        except Exception as e:
            logging.getLogger(__name__).error(f"[exposure] 操作失败: {e}", exc_info=True)
        return 0.0

    def _calc_quality_exposure(self, code: str) -> float:
        """计算质量因子暴露 (ROE 标准化).

        P1-Q21-fix(H01): 原实现 float(净资产收益率整列Series) 必抛 TypeError 被吞
        -> quality 恒 0。改用 _latest_financial_value 取最新报告期数值。
        P2-Q21-fix: 复用 compute 级财务缓存，避免与 leverage 重复抓取。
        """
        try:
            fin = self._get_financial(code)
            roe = _latest_financial_value(fin, "净资产收益率")
            if not np.isfinite(roe):
                return float("nan")
            return round(max(-1, min(1, (roe - 10) / 10)), 2)
        except Exception:
            return float("nan")

    def _calc_growth_exposure(self, code: str) -> float:
        """计算成长因子暴露 (营收/利润增速)."""
        try:
            import akshare as ak
            inc = ak.stock_profit_sheet_by_report_em(symbol=code)
            if inc is not None and "营业收入" in inc.columns:
                revs = inc["营业收入"].dropna().values.astype(float)
                if len(revs) >= 2:
                    growth = (revs[-1] / max(revs[-2], 1) - 1) * 100
                    return round(max(-3, min(3, growth / 20)), 2)
        except Exception as e:
            logging.getLogger(__name__).error(f"[exposure] 操作失败: {e}", exc_info=True)
        return 0.0

    def _calc_liquidity_exposure(self, code: str) -> float:
        """计算流动性因子暴露 (换手率Z-score)."""
        try:
            df = self._get_hist(code)
            if df is not None and "换手率" in df.columns and len(df) > 20:
                turnover = df["换手率"].dropna().values[-20:]
                avg_turnover = np.mean(turnover)
                return round(max(-3, min(3, (avg_turnover - 3) / 3)), 2)
        except Exception as e:
            logging.getLogger(__name__).error(f"[exposure] 操作失败: {e}", exc_info=True)
        return 0.0

    def _calc_leverage_exposure(self, code: str) -> float:
        """计算杠杆因子暴露 (资产负债率标准化).

        P1-Q21-fix(H01): 原实现 float(资产负债率整列Series) 必抛 TypeError 被吞
        -> leverage 恒 0。改用 _latest_financial_value 取最新报告期数值。
        P2-Q21-fix: 复用 compute 级财务缓存，避免与 quality 重复抓取。
        """
        try:
            fin = self._get_financial(code)
            debt = _latest_financial_value(fin, "资产负债率")
            if not np.isfinite(debt):
                return float("nan")
            return round(max(-3, min(3, (debt - 50) / 20)), 2)
        except Exception:
            return float("nan")

    def _calc_dividend_exposure(self, code: str) -> float:
        """计算股息率因子暴露（最近一期派息方案的股息率）.

        P1-Q21-fix(H01): 原实现 fin.get("股息率") 在同花顺财务摘要中无此列 -> 恒 None
        -> dividend_yield 恒 0。改用东财分红送配详情 stock_fhps_detail_em 的
        "现金分红-股息率"（小数，如 0.0164 表示 1.64%），取最近一个非空方案。
        """
        try:
            import akshare as ak
            d = ak.stock_fhps_detail_em(symbol=code)
            if d is None or d.empty:
                return float("nan")
            dy_col = None
            for c in d.columns:
                if "股息率" in str(c):
                    dy_col = c
                    break
            if dy_col is None:
                return float("nan")
            series = pd.to_numeric(d[dy_col], errors="coerce").dropna()
            if series.empty:
                return float("nan")
            dy_pct = float(series.iloc[-1]) * 100.0  # 小数 -> 百分数
            if not np.isfinite(dy_pct):
                return float("nan")
            return round(max(-3, min(3, (dy_pct - 2) / 2)), 2)
        except Exception:
            return float("nan")

    def _generate_risk_warnings(self, exposures: dict[str, float]) -> list[str]:
        """基于组合因子暴露生成真实风险预警（对标 Barra 组合风格监控）。

        暴露为标准化 Z-score（约 [-3, 3]），|z|>=1.5 视为显著偏离，>=2.0 视为极端：
          - beta/size/momentum/volatility 等单因子极端暴露提示集中风格 bet
          - 全部因子数据不可得时提示数据源问题（区别于"暴露为0"）

        Returns:
            预警文案列表；无显著偏离时返回空列表。
        """
        warnings: list[str] = []
        SIGNIFICANT = 1.5
        EXTREME = 2.0
        templates = [
            ("beta", "Beta暴露{exposure:+.2f}，市场涨跌对组合放大/衰减明显"),
            ("size", "市值暴露{exposure:+.2f}，风格集中于单一市值段"),
            ("value", "估值暴露{exposure:+.2f}，偏价值或偏成长显著"),
            ("momentum", "动量暴露{exposure:+.2f}，追涨或反转风险"),
            ("volatility", "波动率暴露{exposure:+.2f}，高波/低波风格显著"),
            ("quality", "质量暴露{exposure:+.2f}，质量风格显著"),
            ("growth", "成长暴露{exposure:+.2f}，成长风格显著"),
            ("liquidity", "流动性暴露{exposure:+.2f}，流动性风格显著"),
            ("leverage", "杠杆暴露{exposure:+.2f}，杠杆风格显著"),
            ("dividend_yield", "股息率暴露{exposure:+.2f}，红利风格显著"),
        ]
        for factor, tmpl in templates:
            v = exposures.get(factor)
            if v is None or not np.isfinite(v):
                continue
            if abs(v) >= EXTREME:
                warnings.append(f"🔴 因子[{factor}] {tmpl.format(exposure=v)}（>±2.0）")
            elif abs(v) >= SIGNIFICANT:
                warnings.append(f"🟡 因子[{factor}] {tmpl.format(exposure=v)}（>±1.5）")

        # 数据不可得提示：区分"暴露为0"与"算不出"
        missing = [
            f for f in self.FACTOR_NAMES
            if exposures.get(f) is None or not np.isfinite(exposures.get(f))
        ]
        if len(missing) == len(self.FACTOR_NAMES):
            warnings.append("⚠️ 所有因子数据不可得，暴露全为 NaN（检查行情/财务数据源）")
        elif missing:
            warnings.append(f"⚠️ {len(missing)}个因子数据不可得: {', '.join(missing)}")
        return warnings

    def compute(self, holdings: list[dict[str, Any]]) -> dict[str, Any]:
        """计算组合的因子暴露矩阵。

        Args:
            holdings: 持仓列表 [{"code": "000001", "weight": 0.1, "name": "平安银行"}, ...]
                      权重总和应为1.0

        Returns:
            dict:
                - exposures: dict, 因子名 -> 组合暴露值
                - stock_exposures: dict, 股票代码 -> 因子暴露dict
                - active_exposures: dict, 因子名 -> 主动偏离（组合-基准）
                - hidden_bets: list, 最大的非故意因子偏离
                - top_drivers: list, 当前最主要的收益来源因子
                - risk_warnings: list, 因子暴露阈值触发的真实预警
        """
        if not holdings:
            return {"exposures": {}, "stock_exposures": {}, "error": "无持仓数据"}

        codes = [h["code"] for h in holdings if h.get("code")]
        weights = {h["code"]: h.get("weight", 0) for h in holdings}
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        # P2-Q21-fix: 本次 compute 内批量复用个股历史/财务数据（每次计算前清空，
        # 避免把上一次调用的缓存带给不同的持仓集合）
        self._hist_cache = {}
        self._fin_cache = {}

        # 计算单只股票因子暴露
        stock_exposures: dict[str, dict[str, float]] = {}
        # 映射因子 -> 计算方法
        calc_methods = {
            "beta": self._calc_beta_exposure,
            "size": self._calc_size_exposure,
            "value": self._calc_value_exposure,
            "momentum": self._calc_momentum_exposure,
            "volatility": self._calc_volatility_exposure,
            "quality": self._calc_quality_exposure,
            "growth": self._calc_growth_exposure,
            "liquidity": self._calc_liquidity_exposure,
            "leverage": self._calc_leverage_exposure,
            "dividend_yield": self._calc_dividend_exposure,
        }

        for code in codes:
            stock_exp: dict[str, float] = {}
            for fname, method in calc_methods.items():
                try:
                    stock_exp[fname] = method(code)
                except Exception:
                    # P1-Q21-fix(H01): 数据不可得用 NaN 而非 0，避免把"算不出"伪装成"暴露为0"
                    stock_exp[fname] = float("nan")
            stock_exposures[code] = stock_exp

        # 加权合成组合暴露
        # P1-Q21-fix(H01): 个股因子为 NaN（数据不可得）时不参与该因子加权，
        # 按可得个股的权重重新归一化；全部不可得则该因子为 NaN（而非误导性的 0）。
        exposures: dict[str, float] = {}
        for fname in self.FACTOR_NAMES:
            val_w = 0.0
            val_wsum = 0.0
            for code in codes:
                v = stock_exposures.get(code, {}).get(fname)
                if v is None or not np.isfinite(v):
                    continue
                w = weights.get(code, 0.0)
                val_w += w * float(v)
                val_wsum += w
            weighted = val_w / val_wsum if val_wsum > 0 else float("nan")
            exposures[fname] = round(weighted, 3)

        # 主动偏离 = 组合暴露（简版：以0为基准，实际应有沪深300因子暴露）
        active = {k: round(v, 3) for k, v in exposures.items()}

        # 最大的主动偏离（绝对值最大的3个）；NaN（不可得）不参与排序/展示
        sorted_active = sorted(
            [(k, v) for k, v in active.items() if np.isfinite(v)],
            key=lambda x: abs(x[1]), reverse=True
        )
        hidden_bets = [
            {"factor": f, "exposure": e, "direction": "正向偏离" if e > 0 else "负向偏离"}
            for f, e in sorted_active if abs(e) > 0.3
        ]

        # Top drivers
        top = sorted(
            [(f, e) for f, e in exposures.items() if np.isfinite(e) and abs(e) > 0.2],
            key=lambda x: abs(x[1]), reverse=True
        )
        top_drivers = [
            {
                "factor": f,
                "exposure": e,
                "interpretation": self._interpret_exposure(f, e)
            }
            for f, e in top[:5]
        ]

        # P2-Q21-fix: 真实预警逻辑（原实现恒空占位，无任何风险提示）
        risk_warnings = self._generate_risk_warnings(exposures)

        return {
            "timestamp": datetime.now(CST).isoformat(),
            "holdings_count": len(codes),
            "exposures": exposures,
            "active_exposures": active,
            "hidden_bets": hidden_bets,
            "top_drivers": top_drivers,
            "risk_warnings": risk_warnings,
        }

    def _interpret_exposure(self, factor: str, exposure: float) -> str:
        """解释因子暴露的含义。"""
        if factor == "beta":
            if exposure > 0.5: return "高Beta暴露 — 市场上涨时超涨，下跌时超跌"
            if exposure < -0.5: return "低Beta暴露 — 防御性组合"
            return "中性Beta"
        if factor == "size":
            if exposure > 0.5: return "大盘暴露 — 偏向大市值"
            if exposure < -0.5: return "小盘暴露 — 偏向小市值"
            return "市值中性"
        if factor == "value":
            if exposure > 0.5: return "价值暴露 — 偏向低估值"
            if exposure < -0.5: return "成长暴露 — 偏向高估值"
            return "估值中性"
        if factor == "momentum":
            if exposure > 0.5: return "高动量暴露 — 追涨组合"
            if exposure < -0.5: return "反转暴露 — 低吸组合"
            return "动量中性"
        return f"暴露值: {exposure:+.2f}"

    def compare_to_benchmark(self, holdings: list[dict[str, Any]],
                             benchmark: list[dict[str, Any]]) -> dict[str, Any]:
        """与基准比较的主动偏离。"""
        pf = self.compute(holdings)
        bm = self.compute(benchmark)
        active = {}
        for f in self.FACTOR_NAMES:
            diff = pf["exposures"].get(f, 0) - bm["exposures"].get(f, 0)
            active[f] = round(diff, 3)
        return {
            "portfolio_exposures": pf["exposures"],
            "benchmark_exposures": bm["exposures"],
            "active_exposures": active,
            "largest_bets": sorted(
                [{"factor": k, "deviation": v} for k, v in active.items() if abs(v) > 0.2],
                key=lambda x: abs(x["deviation"]), reverse=True
            ),
        }


def main() -> None:
    """示例：计算模拟持仓的因子暴露。"""
    # 模拟持仓
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.15},
        {"code": "000858", "name": "五粮液", "weight": 0.10},
        {"code": "300750", "name": "宁德时代", "weight": 0.08},
        {"code": "601318", "name": "中国平安", "weight": 0.07},
        {"code": "000333", "name": "美的集团", "weight": 0.06},
        {"code": "002415", "name": "海康威视", "weight": 0.05},
        {"code": "600036", "name": "招商银行", "weight": 0.05},
        {"code": "000002", "name": "万科A", "weight": 0.04},
        {"code": "601166", "name": "兴业银行", "weight": 0.04},
        {"code": "002304", "name": "洋河股份", "weight": 0.04},
    ]

    fe = FactorExposure()
    result = fe.compute(holdings)

    logger.info("═" * 65)
    logger.info("  持仓因子暴露诊断")
    logger.info("═" * 65)
    logger.info(f"\n  持仓: {result['holdings_count']}只股票")
    logger.info()

    logger.info("  ── 因子暴露矩阵 ──")
    for f_exp in ["beta", "size", "value", "momentum", "quality",
                  "volatility", "growth", "liquidity", "leverage", "dividend_yield"]:
        val = result["exposures"].get(f_exp, 0)
        bar = "█" * int(abs(val) * 10) + "░" * (30 - int(abs(val) * 10))
        logger.info(f"  {f_exp:<14} {val:>+7.3f}  {bar}")

    logger.info()
    if result.get("hidden_bets"):
        logger.warning("  ⚠️ 非故意因子偏离:")
        for bet in result["hidden_bets"]:
            logger.warning(f"    {bet['factor']}: {bet['exposure']:+.2f} ({bet['direction']})")

    logger.info()
    if result.get("top_drivers"):
        logger.info("  📊 主要收益来源:")
        for d in result["top_drivers"]:
            logger.info(f"    {d['factor']:<12} 暴露{d['exposure']:+.2f}  — {d['interpretation']}")

    logger.info()
    if result.get("risk_warnings"):
        logger.warning("  ⚠️ 风险预警:")
        for w in result["risk_warnings"]:
            logger.warning(f"    • {w}")
    else:
        logger.info("  ✓ 未触发因子暴露预警")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
