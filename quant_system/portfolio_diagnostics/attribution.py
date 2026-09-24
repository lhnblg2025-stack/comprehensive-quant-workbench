"""
attribution — 持仓 PnL 拆解 / Brinson 归因 (V5)

核心问题：我的收益从哪里来？是运气（Beta）还是能力（Alpha/选股）？
  1. 给定持仓（代码 + 权重 + 可选成本价）与区间 [start_date, end_date]
  2. 计算组合总收益、基准（沪深300）收益、超额收益
  3. 用 CAPM 回归把总收益拆成 Beta 收益 + Alpha 收益
  4. 用 Brinson 模型把超额收益拆成 行业配置效应 + 个股选择效应
  5. 找出对收益贡献最大/最小的个股

对标：绩效归因领域的 Brinson-Fachler 模型 + 单因子 CAPM 分解。
"""

from __future__ import annotations
import logging

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CST = timezone(timedelta(hours=8))

# quant_system/ 包根目录 (portfolio_diagnostics 的上一级)
ROOT = Path(__file__).resolve().parent.parent

# 行业分类缺失时的兜底分类名
SECTOR_FALLBACK = "其他"

# 基准指数（沪深300, 新浪代码前缀 sh）
BENCHMARK_SYMBOL = "sh000300"
BENCHMARK_INDEX_CODE = "000300"

TRADING_DAYS_PER_YEAR = 252

# P1-Q21-fix(H03): 计算逐行业基准收益时最多拉取的前 N 大权重成分股，
# 控制全市场 300 只串行拉取造成的耗时与东财限流风险（近似覆盖约六成权重）。
_BENCHMARK_SECTOR_RETURN_MAX_STOCKS = 60


class PnLAttribution:
    """持仓 PnL 拆解引擎（Brinson 归因 + CAPM 分解）。

    核心假设：
      组合总收益 = Beta收益(市场贡献) + Alpha收益(超额能力)
      组合超额收益(相对基准) = 行业配置效应 + 个股选择效应 + 交互效应

    典型用法::

        pa = PnLAttribution()
        result = pa.compute(holdings, "2026-01-01", "2026-07-31")

    Attributes:
        cache: 上一次计算结果缓存
        last_fetch: 上次计算的时间戳（用于 cache_ttl 判断）
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl
        # 个股行业分类缓存：{code: sector_name}，避免重复调用 akshare
        self._sector_cache: dict[str, str] = {}
        # 沪深300 成分股权重缓存（用于近似基准行业权重）
        self._benchmark_weight_cache: dict[str, float] | None = None
        self._benchmark_weight_fetch_time: float = 0.0
        # P1-Q21-fix(H03): 逐行业基准收益缓存（Brinson 行业配置效应需要）
        self._benchmark_sector_returns_cache: dict[str, float] | None = None
        self._benchmark_sector_returns_key: str = ""
        # 沪深300 成分股原始表缓存（权重与逐行业收益共用，避免重复抓取）
        self._benchmark_cons_cache: Any | None = None

    # ────────────────────────── 数据获取 ──────────────────────────

    def fetch_stock_prices(
        self, codes: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """批量获取个股历史日线（前复权）。

        Args:
            codes: 股票代码列表，如 ["600519", "000858"]
            start_date: 起始日期，格式 "YYYY-MM-DD" 或 "YYYYMMDD"
            end_date: 结束日期，格式同上

        Returns:
            dict: {code: DataFrame(index=日期, columns=[..., "收盘", ...])}
                  获取失败的代码不会出现在返回结果中。

        Notes:
            - 使用 adjust="qfq" 前复权，天然处理分红送股对价格的影响。
            - 停牌导致的数据缺口由调用方在对齐日期索引时用前值填充。
        """
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        prices: dict[str, pd.DataFrame] = {}
        try:
            import akshare as ak
        except Exception:
            return prices

        for code in codes:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_fmt,
                    end_date=end_fmt,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    continue
                df = df.copy()
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.set_index("日期").sort_index()
                prices[code] = df
            except Exception as e:
                # 单只股票失败不影响其他股票（新股/停牌/代码错误等）
                logging.getLogger(__name__).error(f"[attribution] 操作失败: {e}", exc_info=True)
                continue
        return prices

    def fetch_benchmark_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取沪深300指数历史日线。

        Args:
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            DataFrame: index=日期, 至少包含 "close" 列；失败时返回空 DataFrame。
        """
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=BENCHMARK_SYMBOL)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        except Exception:
            return pd.DataFrame()

    def get_sector(self, code: str) -> str:
        """获取个股所属行业分类（带缓存，缺失时归类为"其他"）。

        Args:
            code: 股票代码

        Returns:
            str: 行业名称，如 "食品饮料"；无法获取时返回 "其他"。

        Notes:
            akshare 的 stock_board_industry_name_em() 返回的是行业板块列表
            （板块名/代码/成分股数等），并不直接提供"单只股票 -> 行业"的映射。
            实际取行业名使用 stock_individual_info_em() 返回的"行业"字段。
            P2-Q21-fix: 原实现取到"其他"时额外调用 stock_board_industry_name_em()
            仅做存在性探测后 pass——纯浪费 API 调用（且该接口本环境不可达），已删除。
        """
        if code in self._sector_cache:
            return self._sector_cache[code]

        sector = SECTOR_FALLBACK
        try:
            import akshare as ak
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                d = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
                candidate = d.get("行业")
                if candidate:
                    sector = str(candidate).strip() or SECTOR_FALLBACK
        except Exception as e:
            logging.getLogger(__name__).error(f"[attribution] 操作失败: {e}", exc_info=True)

        self._sector_cache[code] = sector
        return sector

    def _fetch_benchmark_sector_weights(self, sector_map: dict[str, str]) -> dict[str, float]:
        """近似计算沪深300的行业权重分布（用于 Brinson 归因的基准侧）。

        通过成分股权重 (index_stock_cons_weight_csindex) 逐个映射到行业，
        再按行业汇总权重。失败时返回空 dict，调用方需要有兜底逻辑。

        Args:
            sector_map: 组合内已知的 {code: sector} 映射，用于复用行业查询缓存

        Returns:
            dict: {sector_name: weight}，权重之和应接近 1.0
        """
        now = _time.time()
        if (
            self._benchmark_weight_cache is not None
            and now - self._benchmark_weight_fetch_time < self.cache_ttl
        ):
            return self._benchmark_weight_cache

        weights: dict[str, float] = {}
        try:
            import akshare as ak
            # P1-Q21-fix(H03): 成分股表缓存，供 _fetch_benchmark_sector_returns 复用
            if self._benchmark_cons_cache is not None and now - self._benchmark_weight_fetch_time < self.cache_ttl:
                cons = self._benchmark_cons_cache
            else:
                cons = ak.index_stock_cons_weight_csindex(symbol=BENCHMARK_INDEX_CODE)
                if cons is not None and not cons.empty:
                    self._benchmark_cons_cache = cons
            if cons is None or cons.empty:
                self._benchmark_weight_cache = {}
                self._benchmark_weight_fetch_time = now
                return {}

            code_col = None
            weight_col = None
            for c in cons.columns:
                if "代码" in str(c):
                    code_col = c
                if "权重" in str(c):
                    weight_col = c
            if code_col is None or weight_col is None:
                self._benchmark_weight_cache = {}
                self._benchmark_weight_fetch_time = now
                return {}

            sector_sum: dict[str, float] = {}
            for _, row in cons.iterrows():
                code = str(row[code_col]).zfill(6)
                try:
                    w = float(row[weight_col]) / 100.0
                except Exception as e:
                    logging.getLogger(__name__).error(f"[attribution] 操作失败: {e}", exc_info=True)
                    continue
                sector = sector_map.get(code) or self.get_sector(code)
                sector_sum[sector] = sector_sum.get(sector, 0.0) + w

            total = sum(sector_sum.values())
            if total > 0:
                weights = {k: v / total for k, v in sector_sum.items()}
        except Exception:
            weights = {}

        self._benchmark_weight_cache = weights
        self._benchmark_weight_fetch_time = now
        return weights

    def _fetch_benchmark_sector_returns(
        self, start_date: str, end_date: str, sector_map: dict[str, str]
    ) -> dict[str, float]:
        """近似计算沪深300各行业的区间收益（供 Brinson 行业配置效应使用）。

        P1-Q21-fix(H03): Brinson 行业配置效应 = (Wp-Wb)*(Rb行业-Rb总)。
        原实现把 Rb行业 恒等于基准总收益 Rb总 -> (Rb行业-Rb总)≡0 -> 行业配置效应恒为0，
        输出误导。此处用 index_stock_cons_weight_csindex 的成分股+权重，
        按行业汇总"加权收益"近似 Rb行业。为避免300只成分股全量拉取，仅取
        权重最大的前 _BENCHMARK_SECTOR_RETURN_MAX_STOCKS 只（覆盖约六成权重），
        覆盖不足的行业不返回，由调用方降级为 NaN+标注。

        Args:
            start_date: 区间起始日 "YYYY-MM-DD"
            end_date: 区间结束日 "YYYY-MM-DD"
            sector_map: {code: sector} 映射（复用已构建的行业查询缓存）

        Returns:
            {sector: 区间收益}（近似值）；任一环节失败返回 {}（调用方需降级处理）。
        """
        cache_key = f"{start_date}|{end_date}"
        if (
            self._benchmark_sector_returns_cache is not None
            and self._benchmark_sector_returns_key == cache_key
        ):
            return self._benchmark_sector_returns_cache

        result: dict[str, float] = {}
        try:
            # 复用 _fetch_benchmark_sector_weights 已抓取的成分股表（避免重复请求）
            now = _time.time()
            if self._benchmark_cons_cache is not None and now - self._benchmark_weight_fetch_time < self.cache_ttl:
                cons = self._benchmark_cons_cache
            else:
                import akshare as ak
                cons = ak.index_stock_cons_weight_csindex(symbol=BENCHMARK_INDEX_CODE)
                if cons is not None and not cons.empty:
                    self._benchmark_cons_cache = cons
            if cons is None or cons.empty:
                return {}
            code_col = next((c for c in cons.columns if "代码" in str(c)), None)
            weight_col = next((c for c in cons.columns if "权重" in str(c)), None)
            if code_col is None or weight_col is None:
                return {}

            # 按权重降序取前 N 大成分股，控制网络请求量
            # （权重列可能为字符串，先转数值再排序，避免字典序错误）
            cons = cons.copy()
            cons["_w"] = pd.to_numeric(cons[weight_col], errors="coerce")
            cons = cons.dropna(subset=["_w"]).sort_values("_w", ascending=False).head(
                _BENCHMARK_SECTOR_RETURN_MAX_STOCKS
            )
            codes = [str(c).zfill(6) for c in cons[code_col]]
            prices = self.fetch_stock_prices(codes, start_date, end_date)
            if not prices:
                return {}

            # 行业内加权收益（权重在行业内归一化）
            sector_ret_sum: dict[str, list[float]] = {}
            sector_ret_w: dict[str, float] = {}
            for _, row in cons.iterrows():
                code = str(row[code_col]).zfill(6)
                w = float(row["_w"]) / 100.0
                df = prices.get(code)
                if df is None or df.empty or "收盘" not in df.columns:
                    continue
                close = df["收盘"].dropna()
                if len(close) < 2:
                    continue
                r = float(close.iloc[-1] / close.iloc[0] - 1)
                sector = sector_map.get(code) or self.get_sector(code)
                sector_ret_sum.setdefault(sector, []).append(w * r)
                sector_ret_w[sector] = sector_ret_w.get(sector, 0.0) + w

            for sector, wsum in sector_ret_w.items():
                if wsum > 0:
                    result[sector] = sum(sector_ret_sum[sector]) / wsum
        except Exception:
            result = {}

        self._benchmark_sector_returns_cache = result
        self._benchmark_sector_returns_key = cache_key
        return result

    # ────────────────────────── 核心计算 ──────────────────────────

    @staticmethod
    def _align_and_fill(price_frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """将各股票的收盘价序列对齐到统一交易日历，停牌用前值填充。

        Args:
            price_frames: {code: DataFrame}，需包含"收盘"列
            calendar: 目标交易日历（通常取基准指数的交易日）

        Returns:
            dict: {code: Series(index=calendar)}，序列已前值填充，
                  头部仍缺失（新股上市前）的保留为 NaN。
        """
        aligned: dict[str, pd.Series] = {}
        for code, df in price_frames.items():
            if "收盘" not in df.columns:
                continue
            s = df["收盘"].reindex(calendar)
            s = s.ffill()  # 停牌填充：沿用停牌前最后一个收盘价
            aligned[code] = s
        return aligned

    @staticmethod
    def capm_regression(stock_returns: np.ndarray, market_returns: np.ndarray) -> dict[str, float]:
        """CAPM 单因子回归，求 beta / alpha。

        模型: r_stock = alpha + beta * r_market + epsilon

        Args:
            stock_returns: 个股（或组合）日收益率序列
            market_returns: 基准日收益率序列，长度需与 stock_returns 一致

        Returns:
            dict:
                - beta: 市场敏感度
                - alpha: 日频截距项（未年化）
                - alpha_annualized: 年化 alpha（按 252 个交易日线性年化）
                - r_squared: 回归拟合优度
        """
        mask = np.isfinite(stock_returns) & np.isfinite(market_returns)
        y = np.asarray(stock_returns, dtype=float)[mask]
        x = np.asarray(market_returns, dtype=float)[mask]

        if len(y) < 5 or np.std(x) == 0:
            return {"beta": 1.0, "alpha": 0.0, "alpha_annualized": 0.0, "r_squared": 0.0}

        var_x = np.var(x, ddof=1)
        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        beta = cov_xy / var_x if var_x > 0 else 1.0
        alpha = float(np.mean(y) - beta * np.mean(x))

        y_hat = alpha + beta * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "beta": round(float(beta), 4),
            "alpha": round(alpha, 6),
            "alpha_annualized": round(alpha * TRADING_DAYS_PER_YEAR, 4),
            "r_squared": round(max(0.0, min(1.0, r_squared)), 4),
        }

    def brinson_decomposition(
        self,
        portfolio_weights: dict[str, float],
        sector_map: dict[str, str],
        stock_returns: dict[str, float],
        benchmark_sector_weights: dict[str, float],
        benchmark_total_return: float,
        benchmark_sector_returns: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Brinson-Fachler 行业归因分解。

        配置效应 = (Wp_sector - Wb_sector) * (Rb_sector - Rb_total)
        选股效应 = Wb_sector * (Rp_sector - Rb_sector)
        交互效应 = (Wp_sector - Wb_sector) * (Rp_sector - Rb_sector)

        Args:
            portfolio_weights: {code: 组合权重}（已归一化）
            sector_map: {code: 行业名}
            stock_returns: {code: 区间收益率（小数）}
            benchmark_sector_weights: {sector: 基准行业权重}
            benchmark_total_return: 基准区间总收益率（小数）
            benchmark_sector_returns: 逐行业基准收益 {sector: 区间收益}。
                P1-Q21-fix(H03): 若不提供，Rb_sector 恒等于 Rb_total 会让
                配置效应数学上恒为 0（误导性输出），因此此时将 sector_allocation
                置为 NaN 并给出 sector_allocation_note 标注"不可计算"。

        Returns:
            dict:
                - sector_allocation: 总配置效应 (%)；不可计算时为 NaN
                - sector_allocation_note: 配置效应不可计算时的说明（可计算时为空串）
                - stock_selection: 总选股效应 (%)
                - interaction: 总交互效应 (%)
                - sector_attribution: 逐行业明细列表
        """
        sector_agg: dict[str, dict[str, float]] = {}
        for code, w in portfolio_weights.items():
            sector = sector_map.get(code, SECTOR_FALLBACK)
            r = stock_returns.get(code)
            if r is None or not np.isfinite(r):
                continue
            entry = sector_agg.setdefault(sector, {"p_weight": 0.0, "p_wret": 0.0})
            entry["p_weight"] += w
            entry["p_wret"] += w * r

        # 基准侧：若无法取得基准行业权重，退化为使用组合自身行业权重
        # （此时配置效应恒为 0，只剩选股效应，属于合理的降级处理）
        if not benchmark_sector_weights:
            total_pw = sum(v["p_weight"] for v in sector_agg.values()) or 1.0
            benchmark_sector_weights = {
                s: v["p_weight"] / total_pw for s, v in sector_agg.items()
            }

        all_sectors = set(sector_agg.keys()) | set(benchmark_sector_weights.keys())
        sector_attribution = []
        total_alloc = 0.0
        total_select = 0.0
        total_interact = 0.0
        alloc_computable = True  # 所有行业都有逐行业基准收益才认为配置效应可计算

        for sector in sorted(all_sectors):
            pw = sector_agg.get(sector, {}).get("p_weight", 0.0)
            p_wret = sector_agg.get(sector, {}).get("p_wret", 0.0)
            p_ret = p_wret / pw if pw > 0 else 0.0
            bw = benchmark_sector_weights.get(sector, 0.0)
            # P1-Q21-fix(H03): 优先用逐行业基准收益；缺失行业用基准总收益近似
            # （该行业配置效应贡献为 0），并在最后把整体配置效应标注为不可计算
            if benchmark_sector_returns and sector in benchmark_sector_returns:
                b_ret = float(benchmark_sector_returns[sector])
                sector_alloc_ok = True
            else:
                b_ret = benchmark_total_return
                sector_alloc_ok = False

            alloc = (pw - bw) * (b_ret - benchmark_total_return)
            select = bw * (p_ret - b_ret)
            interact = (pw - bw) * (p_ret - b_ret)

            if not sector_alloc_ok:
                alloc_computable = False
            total_alloc += alloc
            total_select += select
            total_interact += interact

            sector_attribution.append({
                "sector": sector,
                "allocation_effect": round(alloc * 100, 4) if sector_alloc_ok else float("nan"),
                "selection_effect": round(select * 100, 4),
            })

        sector_attribution.sort(key=lambda x: abs(x["selection_effect"]), reverse=True)

        if not alloc_computable:
            return {
                "sector_allocation": float("nan"),
                "sector_allocation_note": "行业配置效应不可完整计算（缺少逐行业基准收益数据，"
                "原实现以基准总收益近似导致该项恒为0，故不再输出误导性的0）",
                "stock_selection": round(total_select * 100, 4),
                "interaction": round(total_interact * 100, 4),
                "sector_attribution": sector_attribution,
            }

        return {
            "sector_allocation": round(total_alloc * 100, 4),
            "sector_allocation_note": "",
            "stock_selection": round(total_select * 100, 4),
            "interaction": round(total_interact * 100, 4),
            "sector_attribution": sector_attribution,
        }

    def top_contributors(
        self,
        holdings: list[dict[str, Any]],
        stock_returns: dict[str, float],
        weights: dict[str, float],
        n: int = 10,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """按收益贡献（权重 × 区间收益）排序，找出正/负向贡献最大的个股。

        Args:
            holdings: 原始持仓列表（用于取股票名称）
            stock_returns: {code: 区间收益率}
            weights: {code: 归一化权重}
            n: 各方向返回的股票数量

        Returns:
            (top_contributors, bottom_contributors): 两个列表，
            每个元素为 {"code", "name", "contribution", "weight"}
            contribution 单位为 %。
        """
        name_map = {h.get("code"): h.get("name", h.get("code")) for h in holdings}
        contribs = []
        for code, w in weights.items():
            r = stock_returns.get(code)
            if r is None or not np.isfinite(r):
                continue
            contribs.append({
                "code": code,
                "name": name_map.get(code, code),
                "contribution": round(w * r * 100, 4),
                "weight": round(w, 4),
            })

        contribs.sort(key=lambda x: x["contribution"], reverse=True)
        top = contribs[:n]
        bottom = sorted(contribs, key=lambda x: x["contribution"])[:n]
        return top, bottom

    # ────────────────────────── 主入口 ──────────────────────────

    def compute(
        self,
        holdings: list[dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """计算组合的完整 PnL 拆解报告。

        Args:
            holdings: 持仓列表，如
                [{"code": "600519", "weight": 0.15, "name": "贵州茅台",
                  "cost_price": 1800.0}, ...]
                权重不必预先归一化，函数内部会自动归一化。
            start_date: 归因区间起始日期 "YYYY-MM-DD"
            end_date: 归因区间结束日期 "YYYY-MM-DD"

        Returns:
            dict: 见模块文档顶部的输出规格，出错时返回 {"error": str}。
        """
        # P2-Q21-fix: cache_key 加入权重元组——原 key 只含股票代码+日期，
        # TTL 600s 内同持仓集合换权重后返回旧缓存（参考 concentration.py 的
        # sorted((code, weight)) 写法）
        cache_key = f"{sorted((h.get('code'), h.get('weight')) for h in holdings)}|{start_date}|{end_date}"
        now = _time.time()
        if (
            self.cache.get("_key") == cache_key
            and now - self.last_fetch < self.cache_ttl
        ):
            return self.cache

        try:
            if not holdings:
                return {"error": "无持仓数据"}

            codes = [h["code"] for h in holdings if h.get("code")]
            if not codes:
                return {"error": "持仓缺少有效股票代码"}

            raw_weights = {h["code"]: float(h.get("weight", 0) or 0) for h in holdings}
            total_w = sum(raw_weights.values())
            if total_w <= 0:
                return {"error": "持仓权重总和为0，无法归一化"}
            weights = {k: v / total_w for k, v in raw_weights.items()}

            # ── 1. 拉取行情 ──
            price_frames = self.fetch_stock_prices(codes, start_date, end_date)
            bench_df = self.fetch_benchmark_prices(start_date, end_date)

            if bench_df.empty or "close" not in bench_df.columns:
                return {"error": "基准指数数据获取失败"}
            if not price_frames:
                return {"error": "个股行情数据获取失败"}

            calendar = bench_df.index

            # ── 2. 对齐日期 + 停牌填充 ──
            aligned = self._align_and_fill(price_frames, calendar)
            bench_close = bench_df["close"].reindex(calendar).ffill()

            # ── 3. 区间总收益（新股数据不足则记为 NaN，权重照常保留但收益跳过） ──
            stock_returns: dict[str, float] = {}
            for code, s in aligned.items():
                valid = s.dropna()
                if len(valid) < 2:
                    stock_returns[code] = float("nan")
                    continue
                stock_returns[code] = float(valid.iloc[-1] / valid.iloc[0] - 1)

            valid_codes = [c for c in codes if np.isfinite(stock_returns.get(c, float("nan")))]
            if not valid_codes:
                return {"error": "所有个股均无有效行情数据"}

            # 剔除无效数据后重新归一化权重（保证贡献口径一致）
            valid_w_sum = sum(weights.get(c, 0) for c in valid_codes)
            eff_weights = {c: weights.get(c, 0) / valid_w_sum for c in valid_codes} if valid_w_sum > 0 else {}

            total_return = sum(eff_weights[c] * stock_returns[c] for c in valid_codes)
            benchmark_return = float(bench_close.dropna().iloc[-1] / bench_close.dropna().iloc[0] - 1)
            active_return = total_return - benchmark_return

            # ── 4. 日频收益序列（用于 CAPM 回归） ──
            price_matrix = pd.DataFrame({c: aligned[c] for c in valid_codes}).ffill()
            port_index = pd.Series(0.0, index=calendar)
            for c in valid_codes:
                base = price_matrix[c].dropna().iloc[0] if not price_matrix[c].dropna().empty else np.nan
                if not np.isfinite(base) or base == 0:
                    continue
                port_index = port_index.add(
                    eff_weights[c] * (price_matrix[c] / base), fill_value=0
                )
            port_daily_ret = port_index.replace(0, np.nan).pct_change().dropna()
            bench_daily_ret = bench_close.pct_change().dropna()

            common_idx = port_daily_ret.index.intersection(bench_daily_ret.index)
            capm = self.capm_regression(
                port_daily_ret.loc[common_idx].values,
                bench_daily_ret.loc[common_idx].values,
            )

            beta = capm["beta"]
            beta_return = beta * benchmark_return
            alpha_return = total_return - beta_return

            # ── 5. 行业分类 + Brinson 归因 ──
            sector_map = {c: self.get_sector(c) for c in valid_codes}
            benchmark_sector_weights = self._fetch_benchmark_sector_weights(sector_map)
            # P1-Q21-fix(H03): 尝试引入逐行业基准收益，避免行业配置效应恒为0的误导输出
            benchmark_sector_returns = self._fetch_benchmark_sector_returns(
                start_date, end_date, sector_map
            )
            brinson = self.brinson_decomposition(
                eff_weights, sector_map, stock_returns,
                benchmark_sector_weights, benchmark_return,
                benchmark_sector_returns=benchmark_sector_returns,
            )

            # ── 6. 贡献排序 ──
            top, bottom = self.top_contributors(holdings, stock_returns, eff_weights, n=10)

            result: dict[str, Any] = {
                "_key": cache_key,
                "period": {"start": start_date, "end": end_date},
                "total_return": round(total_return * 100, 4),
                "benchmark_return": round(benchmark_return * 100, 4),
                "active_return": round(active_return * 100, 4),
                "attribution": {
                    "beta_return": round(beta_return * 100, 4),
                    "alpha_return": round(alpha_return * 100, 4),
                    "sector_allocation": brinson["sector_allocation"],
                    "sector_allocation_note": brinson.get("sector_allocation_note", ""),
                    "stock_selection": brinson["stock_selection"],
                },
                "capm": capm,
                "sector_attribution": brinson["sector_attribution"],
                "top_contributors": top,
                "bottom_contributors": bottom,
                "excluded_codes": sorted(set(codes) - set(valid_codes)),
                "holdings_count": len(codes),
            }
            self.cache = result
            self.last_fetch = now
            return result
        except Exception as e:
            return {"error": f"PnLAttribution.compute 失败: {e}"}


def main() -> None:
    """示例：计算模拟持仓在指定区间的 PnL 拆解。"""
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.15, "cost_price": 1800.0},
        {"code": "000858", "name": "五粮液", "weight": 0.10, "cost_price": 150.0},
        {"code": "300750", "name": "宁德时代", "weight": 0.08, "cost_price": 200.0},
        {"code": "601318", "name": "中国平安", "weight": 0.07, "cost_price": 45.0},
        {"code": "000333", "name": "美的集团", "weight": 0.06, "cost_price": 55.0},
    ]

    end_dt = datetime.now(CST)
    start_dt = end_dt - timedelta(days=180)

    pa = PnLAttribution()
    result = pa.compute(
        holdings,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    logger.info("═" * 65)
    logger.info("  持仓 PnL 拆解诊断")
    logger.info("═" * 65)

    if "error" in result:
        logger.warning(f"  ⚠️ 计算失败: {result['error']}")
        return

    logger.info(f"\n  区间: {result['period']['start']} ~ {result['period']['end']}")
    logger.info(f"  组合总收益: {result['total_return']:+.2f}%")
    logger.info(f"  基准收益  : {result['benchmark_return']:+.2f}%")
    logger.info(f"  超额收益  : {result['active_return']:+.2f}%")
    logger.info()
    logger.info("  ── 收益拆解 ──")
    attr = result["attribution"]
    logger.info(f"    Beta贡献   : {attr['beta_return']:+.2f}%")
    logger.info(f"    Alpha贡献  : {attr['alpha_return']:+.2f}%")
    logger.info(f"    行业配置   : {attr['sector_allocation']:+.2f}%")
    if attr.get("sector_allocation_note"):
        logger.warning(f"    ⚠️ 行业配置: {attr['sector_allocation_note']}")
    logger.info(f"    个股选择   : {attr['stock_selection']:+.2f}%")
    logger.info(f"    CAPM beta  : {result['capm']['beta']:.3f}  R²={result['capm']['r_squared']:.3f}")

    logger.info()
    logger.info("  ── 正向贡献 Top 5 ──")
    for c in result["top_contributors"][:5]:
        logger.info(f"    {c['name']:<10} {c['contribution']:+.3f}%  (权重{c['weight']:.1%})")

    logger.info()
    logger.info("  ── 负向贡献 Top 5 ──")
    for c in result["bottom_contributors"][:5]:
        logger.info(f"    {c['name']:<10} {c['contribution']:+.3f}%  (权重{c['weight']:.1%})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
