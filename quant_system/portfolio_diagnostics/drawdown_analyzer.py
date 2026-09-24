"""
drawdown_analyzer — 持仓回撤归因 (V5)

核心问题：最大回撤是怎么发生的？哪只股票"背了锅"？多久能回本？
  1. 用持仓权重 + 个股历史价格构建组合净值曲线
  2. 滚动计算回撤序列，定位最大回撤区间
  3. 把回撤贡献拆解到个股层面（谁跌得最多、拖累最大）
  4. 统计历史各次回撤的恢复天数
  5. 历史模拟法计算 VaR / CVaR 等尾部风险指标

对标：机构持仓回撤归因报告 + 历史模拟法风险度量。
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

ROOT = Path(__file__).resolve().parent.parent

TRADING_DAYS_PER_YEAR = 252
DEFAULT_WINDOW = 250


class DrawdownAnalyzer:
    """持仓回撤归因分析引擎。

    核心假设：
      组合净值 = Σ(个股权重 × 个股价格指数)
      回撤 = 1 - 当前净值 / 历史最高净值（滚动峰值）
      回撤贡献可近似拆解到导致净值下跌的各个股头上。

    典型用法::

        da = DrawdownAnalyzer()
        result = da.compute(holdings, analysis_window=250)

    Attributes:
        cache: 上一次计算结果缓存
        last_fetch: 上次计算的时间戳
        cache_ttl: 缓存有效期（秒）
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0.0
        self.cache_ttl = cache_ttl

    # ────────────────────────── 数据获取 ──────────────────────────

    def fetch_stock_prices(self, codes: list[str], window: int) -> dict[str, pd.DataFrame]:
        """批量获取个股历史日线（前复权），回看窗口天数。

        Args:
            codes: 股票代码列表
            window: 回看的自然日天数（会多取一些余量以覆盖非交易日/停牌）

        Returns:
            dict: {code: DataFrame}，包含"收盘"列，index 为日期。
        """
        end_dt = datetime.now(CST)
        # 自然日 -> 交易日有损耗，按 1.6x 余量回溯，并额外加90天缓冲应对长假/停牌
        start_dt = end_dt - timedelta(days=int(window * 1.6) + 90)
        start_fmt = start_dt.strftime("%Y%m%d")
        end_fmt = end_dt.strftime("%Y%m%d")

        prices: dict[str, pd.DataFrame] = {}
        try:
            import akshare as ak
        except Exception:
            return prices

        for code in codes:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_fmt, end_date=end_fmt, adjust="qfq",
                )
                if df is None or df.empty:
                    continue
                df = df.copy()
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df.set_index("日期").sort_index()
                prices[code] = df
            except Exception as e:
                logging.getLogger(__name__).error(f"[drawdown_analyzer] 操作失败: {e}", exc_info=True)
                continue
        return prices

    # ────────────────────────── 核心计算 ──────────────────────────

    @staticmethod
    def build_portfolio_nav(
        price_frames: dict[str, pd.DataFrame], weights: dict[str, float]
    ) -> tuple[pd.Series, pd.DataFrame]:
        """构建组合净值曲线（以统一基准日为 1.0）。

        Args:
            price_frames: {code: DataFrame}，需包含"收盘"列
            weights: {code: 归一化权重}

        Returns:
            (nav, price_index_matrix):
              nav: 组合净值 Series（index=交易日历，起点=1.0）
              price_index_matrix: 每只股票的价格指数矩阵（起点=1.0），
                                   用于逐股回撤贡献计算。

        基准日统一取窗口内所有股票均有效的首个交易日：各股价格指数在同一
        基准日归一为 1.0。修复此前以各股自身首个有效价为基准（base=1.0）导致
        上市/数据起点晚的个股在起点日以 1.0 瞬间并入净值、组合净值人为跳变
        且起点不足 1.0 的问题（Q21 修复）。
        """
        valid = {c: df for c, df in price_frames.items() if "收盘" in df.columns and not df.empty}
        if not valid:
            return pd.Series(dtype=float), pd.DataFrame()

        # 统一交易日历：取所有股票日期的并集，避免遗漏
        calendar = sorted(set().union(*[set(df.index) for df in valid.values()]))
        calendar = pd.DatetimeIndex(calendar)

        # 每只股票对齐到统一日历并前向填充（起始日之前保持 NaN，不回填）
        aligned = pd.DataFrame(index=calendar)
        for code, df in valid.items():
            aligned[code] = df["收盘"].reindex(calendar).ffill()

        # 统一基准日：所有股票均有效的首个交易日（各股 first_valid_index 的最大值）
        first_valid = {c: col.first_valid_index() for c, col in aligned.items()}
        first_valid = {c: d for c, d in first_valid.items() if d is not None}
        if not first_valid or len(first_valid) < len(aligned.columns):
            return pd.Series(dtype=float), pd.DataFrame()
        base_date = max(first_valid.values())
        base_row = aligned.loc[base_date]
        if base_row.isna().any() or (base_row <= 0).any() or not np.isfinite(base_row).all():
            return pd.Series(dtype=float), pd.DataFrame()

        # 价格指数：以统一基准日为 1.0，各股在同一起点归一，起点之后无 NaN
        index_matrix = aligned / base_row
        index_matrix = index_matrix.loc[base_date:]
        index_matrix = index_matrix.replace([np.inf, -np.inf], np.nan)

        # 组合净值 = Σ(权重 × 价格指数)，基准日处 = Σ权重 = 1.0（权重已归一化时）
        nav = index_matrix.mul(pd.Series(weights)).sum(axis=1, min_count=1)
        nav = nav.replace(0, np.nan).dropna()
        if nav.empty:
            return pd.Series(dtype=float), pd.DataFrame()

        # 起点归一化断言：调用方权重未归一化时兜底缩放，保证起点=1.0
        start = float(nav.iloc[0])
        if not np.isclose(start, 1.0, atol=1e-6):
            nav = nav / start
        index_matrix = index_matrix.loc[nav.index]
        return nav, index_matrix

    @staticmethod
    def rolling_drawdown(nav: pd.Series) -> pd.Series:
        """计算滚动回撤序列。

        Args:
            nav: 净值曲线（越高越好）

        Returns:
            Series: 回撤序列，值为负数或0（如 -0.15 表示回撤15%）
        """
        running_max = nav.cummax()
        drawdown = nav / running_max - 1.0
        return drawdown

    @staticmethod
    def _find_max_drawdown_period(nav: pd.Series, drawdown: pd.Series) -> dict[str, Any]:
        """定位最大回撤发生的区间（峰值日 -> 谷值日）。

        Args:
            nav: 净值曲线
            drawdown: rolling_drawdown 的结果

        Returns:
            dict: {"start": 峰值日期, "end": 谷值日期, "max_drawdown": float}
        """
        if drawdown.empty:
            return {"start": None, "end": None, "max_drawdown": 0.0}

        trough_idx = drawdown.idxmin()
        max_dd = float(drawdown.loc[trough_idx])
        # 峰值日 = 谷值之前净值曲线的最高点对应日期
        peak_nav = nav.loc[:trough_idx].max()
        peak_candidates = nav.loc[:trough_idx]
        peak_idx = peak_candidates[peak_candidates == peak_nav].index[-1]

        return {
            "start": str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
            "end": str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx),
            "max_drawdown": round(max_dd * 100, 4),
        }

    @staticmethod
    def _drawdown_contributors(
        index_matrix: pd.DataFrame,
        weights: dict[str, float],
        peak_date: Any,
        trough_date: Any,
        holdings: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """把最大回撤区间的净值下跌拆解到个股。

        贡献度 = 权重 × 个股在区间内的跌幅 / 组合总回撤（归一化后占比）。

        Args:
            index_matrix: build_portfolio_nav 返回的价格指数矩阵
            weights: {code: 归一化权重}
            peak_date: 回撤区间起点（峰值日）
            trough_date: 回撤区间终点（谷值日）
            holdings: 原始持仓列表（取股票名称）
            top_n: 返回贡献最大的前N只

        Returns:
            list[dict]: [{"code", "name", "contribution_to_drawdown",
                          "weight", "stock_drawdown"}, ...]，按贡献降序排列
        """
        if peak_date is None or trough_date is None or index_matrix.empty:
            return []

        name_map = {h.get("code"): h.get("name", h.get("code")) for h in holdings}
        contributions = []
        total_weighted_loss = 0.0
        entries = []

        for code in index_matrix.columns:
            series = index_matrix[code]
            try:
                peak_val = series.loc[:peak_date].dropna().iloc[-1]
                trough_val = series.loc[:trough_date].dropna().iloc[-1]
            except Exception as e:
                logging.getLogger(__name__).error(f"[drawdown_analyzer] 操作失败: {e}", exc_info=True)
                continue
            if not np.isfinite(peak_val) or peak_val == 0:
                continue
            stock_dd = float(trough_val / peak_val - 1.0)
            w = weights.get(code, 0.0)
            weighted_loss = w * stock_dd
            total_weighted_loss += weighted_loss
            entries.append((code, w, stock_dd, weighted_loss))

        if total_weighted_loss == 0:
            return []

        for code, w, stock_dd, weighted_loss in entries:
            contributions.append({
                "code": code,
                "name": name_map.get(code, code),
                "contribution_to_drawdown": round(weighted_loss / total_weighted_loss, 4),
                "weight": round(w, 4),
                "stock_drawdown": round(stock_dd * 100, 4),
            })

        contributions.sort(key=lambda x: x["contribution_to_drawdown"], reverse=True)
        return contributions[:top_n]

    @staticmethod
    def recovery_analysis(nav: pd.Series, drawdown: pd.Series, threshold: float = -0.03) -> dict[str, Any]:
        """统计历史各次回撤事件的恢复天数。

        定义一次"回撤事件"：回撤从0（前高）跌破 threshold，再回到0（创新高）
        为一次完整事件，恢复天数 = 谷值日到回到前高日的交易日数。
        尚未恢复的最近一次回撤不计入统计（因为恢复天数未知）。

        Args:
            nav: 净值曲线
            drawdown: rolling_drawdown 的结果
            threshold: 触发事件的回撤阈值（默认-3%，太浅的回撤不计入统计）

        Returns:
            dict: {"avg_recovery_days": float|None, "worst_recovery_days": int|None,
                   "recovered_events": int, "current_drawdown_days": int}
        """
        if drawdown.empty:
            return {
                "avg_recovery_days": None, "worst_recovery_days": None,
                "recovered_events": 0, "current_drawdown_days": 0,
            }

        in_drawdown = False
        peak_i = 0
        trough_dd = 0.0
        trough_i = 0
        recovery_days: list[int] = []
        values = drawdown.values
        n = len(values)

        for i in range(n):
            dd = values[i]
            if not in_drawdown:
                if dd < threshold:
                    in_drawdown = True
                    trough_dd = dd
                    trough_i = i
            else:
                if dd < trough_dd:
                    trough_dd = dd
                    trough_i = i
                if dd >= -1e-9:  # 回到前高，事件结束
                    recovery_days.append(i - trough_i)
                    in_drawdown = False
                    trough_dd = 0.0

        # 当前是否仍处于未恢复的回撤中
        current_drawdown_days = 0
        if in_drawdown:
            current_drawdown_days = n - 1 - trough_i

        if not recovery_days:
            return {
                "avg_recovery_days": None, "worst_recovery_days": None,
                "recovered_events": 0, "current_drawdown_days": current_drawdown_days,
            }

        return {
            "avg_recovery_days": round(float(np.mean(recovery_days)), 1),
            "worst_recovery_days": int(max(recovery_days)),
            "recovered_events": len(recovery_days),
            "current_drawdown_days": current_drawdown_days,
        }

    @staticmethod
    def compute_var_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> dict[str, float]:
        """历史模拟法计算 VaR / CVaR。

        Args:
            daily_returns: 组合日收益率序列
            confidence: 置信度，默认 0.95

        Returns:
            dict: {var_{conf_pct}, cvar_{conf_pct}, max_single_day_loss,
                   downside_volatility}。VaR/CVaR 为负数（表示亏损方向），
                  单位为小数（非百分号）。

        P2-Q21-fix: 返回键名由 confidence 派生（原实现固定 var_95/cvar_95，
        即使调用方传其他置信度键名也不变，语义不一致）。
        """
        key_suffix = str(int(round(confidence * 100)))
        var_key = f"var_{key_suffix}"
        cvar_key = f"cvar_{key_suffix}"

        empty: dict[str, float] = {
            var_key: 0.0, cvar_key: 0.0,
            "max_single_day_loss": 0.0, "downside_volatility": 0.0,
        }
        r = daily_returns.dropna().values.astype(float)
        if len(r) < 5:
            return empty

        pct = (1 - confidence) * 100
        var = float(np.percentile(r, pct))
        tail = r[r <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        max_loss = float(np.min(r))
        downside = r[r < 0]
        downside_vol = float(np.std(downside) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else 0.0

        return {
            var_key: round(var, 6),
            cvar_key: round(cvar, 6),
            "max_single_day_loss": round(max_loss, 6),
            "downside_volatility": round(downside_vol, 6),
        }

    # ────────────────────────── 主入口 ──────────────────────────

    def compute(
        self, holdings: list[dict[str, Any]], analysis_window: int = DEFAULT_WINDOW
    ) -> dict[str, Any]:
        """计算组合的完整回撤归因诊断报告。

        Args:
            holdings: 持仓列表 [{"code": "600519", "weight": 0.15,
                      "name": "贵州茅台"}, ...]，权重会自动归一化。
            analysis_window: 回看的交易日窗口长度，默认250（约一年）。

        Returns:
            dict: 见模块文档规格，出错时返回 {"error": str}。
        """
        cache_key = f"{sorted((h.get('code'), h.get('weight')) for h in holdings)}|{analysis_window}"
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

            price_frames = self.fetch_stock_prices(codes, analysis_window)
            if not price_frames:
                return {"error": "个股行情数据获取失败"}

            nav, index_matrix = self.build_portfolio_nav(price_frames, weights)
            if nav.empty or len(nav) < 5:
                return {"error": "组合净值数据不足，无法计算回撤"}

            # 截取分析窗口（最近 N 个交易日）
            if len(nav) > analysis_window:
                nav = nav.iloc[-analysis_window:]
                index_matrix = index_matrix.loc[nav.index]

            drawdown = self.rolling_drawdown(nav)
            current_drawdown = float(drawdown.iloc[-1])

            max_dd_period = self._find_max_drawdown_period(nav, drawdown)
            peak_date = max_dd_period["start"]
            trough_date = max_dd_period["end"]
            # 转回 Timestamp 用于索引查找
            peak_ts = pd.Timestamp(peak_date) if peak_date else None
            trough_ts = pd.Timestamp(trough_date) if trough_date else None

            contributors = self._drawdown_contributors(
                index_matrix, weights, peak_ts, trough_ts, holdings,
            )

            recovery = self.recovery_analysis(nav, drawdown)

            daily_returns = nav.pct_change().dropna()
            risk_metrics = self.compute_var_cvar(daily_returns)

            result: dict[str, Any] = {
                "_key": cache_key,
                "max_drawdown": max_dd_period["max_drawdown"],
                "max_drawdown_period": {"start": peak_date, "end": trough_date},
                "current_drawdown": round(current_drawdown * 100, 4),
                "drawdown_contributors": contributors,
                "recovery_analysis": recovery,
                "risk_metrics": {
                    "var_95": round(risk_metrics["var_95"] * 100, 4),
                    "cvar_95": round(risk_metrics["cvar_95"] * 100, 4),
                    "max_single_day_loss": round(risk_metrics["max_single_day_loss"] * 100, 4),
                    "downside_volatility": round(risk_metrics["downside_volatility"] * 100, 4),
                },
                "analysis_window": analysis_window,
                "data_points": len(nav),
            }
            self.cache = result
            self.last_fetch = now
            return result
        except Exception as e:
            return {"error": f"DrawdownAnalyzer.compute 失败: {e}"}


def main() -> None:
    """示例：计算模拟持仓的回撤归因诊断。"""
    holdings = [
        {"code": "600519", "name": "贵州茅台", "weight": 0.20},
        {"code": "000858", "name": "五粮液", "weight": 0.15},
        {"code": "300750", "name": "宁德时代", "weight": 0.15},
        {"code": "601318", "name": "中国平安", "weight": 0.10},
        {"code": "000333", "name": "美的集团", "weight": 0.10},
        {"code": "002415", "name": "海康威视", "weight": 0.10},
        {"code": "600036", "name": "招商银行", "weight": 0.10},
        {"code": "000002", "name": "万科A", "weight": 0.05},
        {"code": "601166", "name": "兴业银行", "weight": 0.05},
    ]

    da = DrawdownAnalyzer()
    result = da.compute(holdings, analysis_window=250)

    logger.info("═" * 65)
    logger.info("  持仓回撤归因诊断")
    logger.info("═" * 65)

    if "error" in result:
        logger.warning(f"  ⚠️ 计算失败: {result['error']}")
        return

    logger.info(f"\n  最大回撤: {result['max_drawdown']:.2f}%")
    p = result["max_drawdown_period"]
    logger.info(f"  回撤区间: {p['start']} ~ {p['end']}")
    logger.info(f"  当前回撤: {result['current_drawdown']:.2f}%")

    logger.info()
    logger.info("  ── 回撤贡献 Top 5 ──")
    for c in result["drawdown_contributors"][:5]:
        logger.info(f"    {c['name']:<10} 贡献{c['contribution_to_drawdown']:.1%}  "
              f"个股回撤{c['stock_drawdown']:+.2f}%  权重{c['weight']:.1%}")

    logger.info()
    logger.info("  ── 恢复分析 ──")
    rec = result["recovery_analysis"]
    if rec["avg_recovery_days"] is not None:
        logger.info(f"    历史平均恢复: {rec['avg_recovery_days']}天  "
              f"最差: {rec['worst_recovery_days']}天  "
              f"事件数: {rec['recovered_events']}")
    else:
        logger.info("    暂无完整恢复事件数据")
    if rec["current_drawdown_days"] > 0:
        logger.info(f"    当前回撤已持续: {rec['current_drawdown_days']}个交易日 (尚未恢复)")

    logger.info()
    logger.info("  ── 风险指标 ──")
    rm = result["risk_metrics"]
    logger.info(f"    95% VaR : {rm['var_95']:.2f}%")
    logger.info(f"    95% CVaR: {rm['cvar_95']:.2f}%")
    logger.info(f"    单日最大跌幅: {rm['max_single_day_loss']:.2f}%")
    logger.info(f"    下行波动率: {rm['downside_volatility']:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
