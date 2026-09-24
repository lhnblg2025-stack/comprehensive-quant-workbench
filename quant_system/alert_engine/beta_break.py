"""
beta_break.py — Beta 突变检测 (V5)

核心问题：组合的市场敏感度（Beta）是否发生了结构性跳变？

Beta 突变通常意味着：
  - 持仓结构发生了显著调整（换股、加减仓）
  - 个股本身的 Beta 特性变化（比如从防御型转为周期型）
  - 市场环境切换导致相关性结构重估

本模块用 60 日滚动 Beta vs 前 60 日 Beta 做对比，变化 > 0.3 视为突变。

对标：风险模型中的 Beta 稳定性监控 / 券商风控 Beta 跳变预警
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

logger = __import__("logging").getLogger(__name__)

BETA_CHANGE_THRESHOLD = 0.3
ROLLING_WINDOW = 60


class BetaBreak:
    """Beta 突变检测引擎——对比组合当前60日滚动Beta与前60日Beta，识别结构性跳变。

    Attributes:
        window: 滚动窗口长度（交易日），默认60
        threshold: Beta 变化触发预警的阈值，默认0.3
    """

    def __init__(self, window: int = ROLLING_WINDOW, threshold: float = BETA_CHANGE_THRESHOLD) -> None:
        self.window = window
        self.threshold = threshold

    # ────────────────────────────────────────────────────────
    # 数据获取
    # ────────────────────────────────────────────────────────

    def _fetch_returns(self, code: str, days: int) -> np.ndarray | None:
        """获取个股日收益率序列。"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is None or len(df) < 5 or "收盘" not in df.columns:
                return None
            close = df["收盘"].values.astype(float)
            ret = np.diff(close) / close[:-1]
            return ret[-days:] if len(ret) >= days else ret
        except Exception as exc:
            logger.debug("获取 %s 收益率失败: %s", code, exc)
            return None

    def _fetch_benchmark_returns(self, days: int) -> np.ndarray | None:
        """获取沪深300日收益率序列作为市场基准。"""
        try:
            import akshare as ak
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            if idx is None or len(idx) < 5 or "close" not in idx.columns:
                return None
            close = idx["close"].values.astype(float)
            ret = np.diff(close) / close[:-1]
            return ret[-days:] if len(ret) >= days else ret
        except Exception as exc:
            logger.debug("获取基准指数收益率失败: %s", exc)
            return None

    # ────────────────────────────────────────────────────────
    # 日历对齐的收益率获取（P1-Q21-fix(H05)）
    # ────────────────────────────────────────────────────────

    def _fetch_returns_with_dates(self, code: str) -> pd.Series | None:
        """获取个股日收益率序列（DatetimeIndex 索引）。

        P1-Q21-fix(H05): 原实现只保留 np 数组，个股停牌时其收益率条数更少但覆盖
        更长的自然日跨度，与指数按位置切片错位。带日期后可按日历 join 对齐。
        """
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is None or len(df) < 5 or "收盘" not in df.columns or "日期" not in df.columns:
                return None
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期").sort_index()
            return df["收盘"].pct_change().dropna()
        except Exception as exc:
            logger.debug("获取 %s 收益率失败: %s", code, exc)
            return None

    def _fetch_benchmark_returns_with_dates(self) -> pd.Series | None:
        """获取沪深300日收益率序列（DatetimeIndex 索引）。"""
        try:
            import akshare as ak
            idx = ak.stock_zh_index_daily(symbol="sh000300")
            if idx is None or len(idx) < 5 or "close" not in idx.columns:
                return None
            idx = idx.copy()
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.set_index("date").sort_index()
            return idx["close"].pct_change().dropna()
        except Exception as exc:
            logger.debug("获取基准指数收益率失败: %s", exc)
            return None

    def _stock_beta_windows(self, code: str, market_full: pd.Series,
                            window: int) -> tuple[float | None, float | None]:
        """计算单只股票"当前窗口"与"前一窗口"的 Beta（按日历对齐）。

        P1-Q21-fix(H05): 个股与基准收益率按日期内连接后，从同一对齐序列上取
        最后 window 行（当前窗口）与其前 window 行（前一窗口）。两窗口长度一致、
        日历起点一致，停牌不再导致个股窗口与指数窗口错位。
        """
        stock_ret = self._fetch_returns_with_dates(code)
        if stock_ret is None:
            return None, None
        aligned = pd.DataFrame({"stock": stock_ret, "market": market_full}).dropna()
        if len(aligned) < window * 2:
            return None, None
        cur = aligned.iloc[-window:]
        prev = aligned.iloc[-2 * window:-window]
        cur_beta = self._calc_beta(cur["stock"].values, cur["market"].values)
        prev_beta = self._calc_beta(prev["stock"].values, prev["market"].values)
        return cur_beta, prev_beta

    def _portfolio_beta_windows(self, holdings: list[dict[str, Any]], market_full: pd.Series,
                                window: int) -> tuple[float | None, float | None]:
        """按持仓权重加权组合的当前/前一窗口 Beta。

        P1-Q21-fix(H05): 替代原来 _portfolio_beta(60日) 与 _portfolio_beta_prev(125日)
        分别按位置切片的做法——两者窗口长度/日历起点不一致。此处用同一对齐序列、
        同一长度 window 计算两窗口，口径统一。
        """
        weights: dict[str, float] = {}
        for h in holdings:
            code = h.get("code")
            w = h.get("weight", 0.0)
            if code:
                weights[code] = weights.get(code, 0.0) + float(w)
        total_w = sum(weights.values())
        if total_w <= 0:
            return None, None
        weights = {k: v / total_w for k, v in weights.items()}

        w_cur = 0.0
        m_cur = 0.0
        w_prev = 0.0
        m_prev = 0.0
        for code, w in weights.items():
            cur_beta, prev_beta = self._stock_beta_windows(code, market_full, window)
            if cur_beta is not None:
                w_cur += w * cur_beta
                m_cur += w
            if prev_beta is not None:
                w_prev += w * prev_beta
                m_prev += w

        cur = w_cur / m_cur if m_cur > 0 else None
        prev = w_prev / m_prev if m_prev > 0 else None
        return cur, prev

    @staticmethod
    def _calc_beta(stock_ret: np.ndarray, market_ret: np.ndarray) -> float | None:
        """用协方差法计算 Beta = Cov(stock, market) / Var(market)。"""
        n = min(len(stock_ret), len(market_ret))
        if n < 5:
            return None
        s = stock_ret[-n:]
        m = market_ret[-n:]
        var_m = float(np.var(m))
        if var_m < 1e-12:
            return None
        cov = float(np.cov(s, m)[0, 1])
        return cov / var_m

    def _portfolio_beta(self, holdings: list[dict[str, Any]], days: int,
                         market_ret_window: np.ndarray) -> float | None:
        """加权计算组合在指定窗口天数内的 Beta（已废弃，仅供向后兼容）。

        P2-Q21-fix: 参数名由 market_ret_full 改为 market_ret_window——原参数名
        与语义不符：调用方传入的是"目标窗口内的市场收益切片"（如当前 60 日），
        并非全量市场收益序列。detect() 已改用 _portfolio_beta_windows（日历对齐），
        本方法不再用于突变检测。
        """
        weights: dict[str, float] = {}
        for h in holdings:
            code = h.get("code")
            w = h.get("weight", 0.0)
            if code:
                weights[code] = weights.get(code, 0.0) + float(w)
        total_w = sum(weights.values())
        if total_w <= 0:
            return None
        weights = {k: v / total_w for k, v in weights.items()}

        weighted_beta = 0.0
        matched_weight = 0.0
        for code, w in weights.items():
            stock_ret = self._fetch_returns(code, days)
            if stock_ret is None:
                continue
            beta = self._calc_beta(stock_ret, market_ret_window)
            if beta is None:
                continue
            weighted_beta += w * beta
            matched_weight += w

        if matched_weight <= 0:
            return None
        return weighted_beta / matched_weight  # 用匹配到数据的权重归一化

    # ────────────────────────────────────────────────────────
    # 主检测方法
    # ────────────────────────────────────────────────────────

    def detect(self, holdings: list[dict[str, Any]]) -> dict[str, Any]:
        """检测组合Beta是否发生突变：当前60日Beta vs 前60日Beta。

        Args:
            holdings: 持仓列表 [{"code": ..., "weight": ..., "name": ...}, ...]

        Returns:
            {
                "current_beta": float | None,
                "previous_beta": float | None,
                "change": float | None,
                "alert": bool,
                "message": str,
                "timestamp": ISO时间,
            }
        """
        result: dict[str, Any] = {
            "current_beta": None, "previous_beta": None, "change": None,
            "alert": False, "message": "", "timestamp": datetime.now(CST).isoformat(),
        }
        if not holdings:
            result["message"] = "持仓为空，无法计算Beta"
            return result

        try:
            # P1-Q21-fix(H05): 使用带日期的基准收益序列做日历对齐，
            # 当前/前一窗口在同一对齐序列上取同一长度（window），口径统一
            market_full = self._fetch_benchmark_returns_with_dates()
            if market_full is None or len(market_full) < self.window * 2:
                result["message"] = "基准数据不足，无法计算Beta突变"
                return result

            current_beta, previous_beta = self._portfolio_beta_windows(
                holdings, market_full, self.window
            )

            result["current_beta"] = round(current_beta, 3) if current_beta is not None else None
            result["previous_beta"] = round(previous_beta, 3) if previous_beta is not None else None

            if current_beta is not None and previous_beta is not None:
                change = round(current_beta - previous_beta, 3)
                result["change"] = change
                if abs(change) > self.threshold:
                    result["alert"] = True
                    direction = "上升" if change > 0 else "下降"
                    result["message"] = (
                        f"组合Beta由{previous_beta:.2f}{direction}至{current_beta:.2f}"
                        f"(变化{change:+.2f})，超过突变阈值{self.threshold}"
                    )
                else:
                    result["message"] = f"组合Beta稳定，当前{current_beta:.2f}，变化{change:+.2f}"
            else:
                result["message"] = "数据不足，无法完成Beta突变对比"
        except Exception as exc:
            logger.warning("Beta突变检测异常: %s", exc)
            result["message"] = f"检测异常: {exc}"

        return result

    def _portfolio_beta_prev(self, holdings: list[dict[str, Any]], window: int,
                              previous_market: np.ndarray) -> float | None:
        """计算组合在"前一窗口"内的Beta（用个股收益率序列的对应切片）。

        注意：P1-Q21-fix(H05) 已废弃——此实现按位置切片、个股与指数窗口日历
        起点不一致（停牌时错位），且与 _portfolio_beta 窗口口径不一致。
        detect() 已改用 _portfolio_beta_windows（日历对齐+同一长度窗口）。
        保留此方法仅用于向后兼容，不应再用于突变检测。
        """
        weights: dict[str, float] = {}
        for h in holdings:
            code = h.get("code")
            w = h.get("weight", 0.0)
            if code:
                weights[code] = weights.get(code, 0.0) + float(w)
        total_w = sum(weights.values())
        if total_w <= 0:
            return None
        weights = {k: v / total_w for k, v in weights.items()}

        weighted_beta = 0.0
        matched_weight = 0.0
        for code, w in weights.items():
            full_ret = self._fetch_returns(code, window * 2 + 5)
            if full_ret is None or len(full_ret) < window + 5:
                continue
            n_total = len(full_ret)
            prev_start = max(0, n_total - 2 * window)
            prev_end = n_total - window
            stock_prev = full_ret[prev_start:prev_end]
            beta = self._calc_beta(stock_prev, previous_market)
            if beta is None:
                continue
            weighted_beta += w * beta
            matched_weight += w

        if matched_weight <= 0:
            return None
        return weighted_beta / matched_weight


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：验证Beta计算公式的正确性，以及无网络场景下的降级行为。"""
    try:
        detector = BetaBreak()
        print("=" * 60)
        print("BetaBreak Beta突变检测 — 测试")
        print("=" * 60)

        # 单元测试：Beta 计算公式（合成数据，Beta应接近1.5）
        np.random.seed(0)
        market_ret = np.random.normal(0, 0.01, 100)
        noise = np.random.normal(0, 0.005, 100)
        stock_ret = 1.5 * market_ret + noise
        beta = detector._calc_beta(stock_ret, market_ret)
        print(f"\n单元测试: 合成数据 Beta = {beta:.3f} (期望≈1.5)")
        assert beta is not None and 1.0 < beta < 2.0, "Beta计算公式异常"

        holdings = [
            {"code": "600519", "name": "贵州茅台", "weight": 0.5},
            {"code": "000001", "name": "平安银行", "weight": 0.5},
        ]
        result = detector.detect(holdings)
        print(f"\n组合Beta突变检测结果:")
        print(f"  current_beta={result['current_beta']} previous_beta={result['previous_beta']}")
        print(f"  change={result['change']} alert={result['alert']}")
        print(f"  message: {result['message']}")

        assert "current_beta" in result and "alert" in result and "message" in result

        print("\n✅ BetaBreak 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ BetaBreak 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
