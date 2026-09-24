"""
style_drift.py — 风格漂移检测 (V5)

核心问题：组合的因子暴露正在悄悄改变吗？

很多"风格漂移"是无意识的——比如价值型基金经理逐渐买入了成长股，
或者组合在下跌中因估值压缩被动地从"大盘"漂移到"中小盘"。
本模块通过短窗口 vs 长窗口的因子暴露对比，量化这种漂移。

对标：晨星 Style Box 漂移分析 / Barra 组合风格监控
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

logger = __import__("logging").getLogger(__name__)

DRIFT_THRESHOLD = 0.3  # 短期 vs 长期暴露差值超过此值触发预警

FACTOR_NAMES = [
    "beta", "size", "value", "momentum", "quality",
    "volatility", "growth", "liquidity",
]


class StyleDrift:
    """风格漂移检测引擎——对比组合短期与长期的因子暴露，识别漂移方向和幅度。

    核心逻辑：
      1. 用组合持仓的历史价格序列，计算各因子在短窗口(30日)、长窗口(90日)
         内的暴露均值（P2-Q21-fix: 两窗口为独立区间——长窗口取短窗口之前的
         历史，互不重叠，避免嵌套窗口在单边趋势中产生方向性偏差）
      2. diff = short_exposure - long_exposure
      3. |diff| > 0.3 视为显著漂移，触发预警

    Attributes:
        drift_threshold: 触发预警的暴露差值阈值
    """

    def __init__(self) -> None:
        self.drift_threshold = DRIFT_THRESHOLD

    # ────────────────────────────────────────────────────────
    # 因子暴露近似计算（基于价格序列，无需外部因子库）
    # ────────────────────────────────────────────────────────

    def _fetch_price_history(self, code: str, days: int = 120) -> np.ndarray | None:
        """获取个股历史收盘价（尽量长，供短/长窗口切片）。

        Returns:
            收盘价 ndarray，若获取失败返回 None
        """
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if df is None or len(df) < 10 or "收盘" not in df.columns:
                return None
            return df["收盘"].values.astype(float)[-days:]
        except Exception as exc:
            logger.debug("获取 %s 价格历史失败: %s", code, exc)
            return None

    def _window_exposure(self, close: np.ndarray) -> dict[str, float]:
        """基于价格窗口计算简化版因子暴露（动量/波动率可直接算，其余置0作为占位近似）。

        为保证在无外部因子库/无网络时也能稳定运行，本函数只用价格序列
        可直接派生的因子（momentum, volatility），其余因子暴露返回 0.0，
        代表"该窗口内不可判定"，不会触发漂移误报。
        """
        exposure = {f: 0.0 for f in FACTOR_NAMES}
        if close is None or len(close) < 5:
            return exposure

        returns = np.diff(close) / close[:-1]
        # 动量：窗口首尾收益率，标准化到 [-3, 3]
        mom = close[-1] / close[0] - 1
        exposure["momentum"] = round(float(max(-3, min(3, mom * 10))), 3)
        # 波动率：窗口年化波动率，标准化
        vol = float(np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0.0
        exposure["volatility"] = round(float(max(-3, min(3, (vol - 0.3) / 0.1))), 3)
        return exposure

    @staticmethod
    def _window_slice(close: np.ndarray, window_short: int, window_long: int,
                      long_window: bool) -> np.ndarray:
        """返回短/长窗口的切片（等长、独立、不重叠）。

        短窗口: close[-window_short:]（最近 window_short 个交易日）
        长窗口: close[-window_long : -(window_long - window_short)]
                （锚定 window_long 个交易日前的 window_short 日区间，
                 默认 30/90 → 前 60~90 日）
        历史不足时，长窗口退化为紧邻短窗口之前的等长区间（仍不重叠）。
        """
        if not long_window:
            return close[-window_short:] if len(close) >= window_short else close
        if len(close) >= window_long:
            start = -window_long
            end = start + window_short
            return close[start:end]
        if len(close) <= window_short:
            return close  # 数据过少，用可用全部（无法保证等长）
        end = -window_short
        start = end - window_short
        if start < -len(close):
            start = -len(close)
        return close[start:end]

    def _portfolio_exposure(self, holdings: list[dict[str, Any]], window_short: int,
                            window_long: int = 90, long_window: bool = False) -> dict[str, float]:
        """加权计算组合因子暴露。

        P2-Q21-fix: 短/长窗口改为等长、独立区间（不重叠）——
          - 短窗口 (long_window=False): 最近 window_short 个交易日
          - 长窗口 (long_window=True): 锚定 window_long 个交易日前的 window_short
            日区间（默认 30/90 → 前 60~90 日）
        原实现 90 日长窗口包含 30 日短窗口（嵌套窗口），平稳上涨时长期累计动量
        恒 > 短期累计动量 → diff 恒为负 → 单边趋势中持续误报"动量下降"。
        改为等长窗口后，恒定上涨速率下两窗口动量接近，不再产生方向性偏差。
        """
        weights: dict[str, float] = {}
        for h in holdings:
            code = h.get("code")
            w = h.get("weight", 0.0)
            if code:
                weights[code] = weights.get(code, 0.0) + float(w)
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        agg = {f: 0.0 for f in FACTOR_NAMES}
        for code, w in weights.items():
            # 需要覆盖长窗口锚点 + 短窗口的历史，才能切出独立长窗口
            need = window_long + window_short + 10
            close = self._fetch_price_history(code, days=need)
            if close is None:
                continue
            close_window = self._window_slice(close, window_short, window_long, long_window)
            exp = self._window_exposure(close_window)
            for f in FACTOR_NAMES:
                agg[f] += w * exp[f]
        return {f: round(v, 3) for f, v in agg.items()}

    # ────────────────────────────────────────────────────────
    # 主检测方法
    # ────────────────────────────────────────────────────────

    def detect(self, holdings: list[dict[str, Any]], window_short: int = 30,
               window_long: int = 90) -> dict[str, Any]:
        """检测组合的风格漂移：短窗口暴露 vs 长窗口暴露的差异。

        P2-Q21-fix: 短/长窗口改为独立区间——短窗口取最近 window_short 个交易日，
        长窗口取 window_long 个交易日（紧邻短窗口之前、不含重叠）。原实现长窗口
        包含短窗口，单边趋势中 diff 恒为负、|diff| 极易 >0.3，产生持续误报。

        Args:
            holdings: 持仓列表 [{"code": ..., "weight": ..., "name": ...}, ...]
            window_short: 短期窗口（交易日），默认30
            window_long: 长期窗口（交易日），默认90

        Returns:
            {
                "drifts": [{"factor", "short_exposure", "long_exposure", "diff"}, ...],
                "alerts": [{"factor", "diff", "severity", "message"}, ...],
                "timestamp": ISO时间,
            }
        """
        result: dict[str, Any] = {"drifts": [], "alerts": [], "timestamp": datetime.now(CST).isoformat()}
        if not holdings:
            return result

        try:
            short_exp = self._portfolio_exposure(holdings, window_short, window_long, long_window=False)
            long_exp = self._portfolio_exposure(holdings, window_short, window_long, long_window=True)
        except Exception as exc:
            logger.warning("风格漂移检测数据获取异常: %s", exc)
            short_exp = {f: 0.0 for f in FACTOR_NAMES}
            long_exp = {f: 0.0 for f in FACTOR_NAMES}

        drifts = []
        alerts = []
        for factor in FACTOR_NAMES:
            s = short_exp.get(factor, 0.0)
            l = long_exp.get(factor, 0.0)
            diff = round(s - l, 3)
            drifts.append({
                "factor": factor,
                "short_exposure": s,
                "long_exposure": l,
                "diff": diff,
            })
            if abs(diff) > self.drift_threshold:
                severity = "high" if abs(diff) > self.drift_threshold * 2 else "medium"
                direction = "上升" if diff > 0 else "下降"
                alerts.append({
                    "factor": factor,
                    "diff": diff,
                    "severity": severity,
                    "message": f"因子[{factor}]暴露短期较长期{direction}{abs(diff):.2f}，超过漂移阈值{self.drift_threshold}",
                })

        result["drifts"] = drifts
        result["alerts"] = alerts
        return result


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：验证漂移检测在无网络/数据缺失场景下不崩溃并返回结构正确。"""
    try:
        detector = StyleDrift()
        print("=" * 60)
        print("StyleDrift 风格漂移检测 — 测试")
        print("=" * 60)

        holdings = [
            {"code": "600519", "name": "贵州茅台", "weight": 0.3},
            {"code": "000001", "name": "平安银行", "weight": 0.3},
            {"code": "300750", "name": "宁德时代", "weight": 0.4},
        ]

        result = detector.detect(holdings, window_short=30, window_long=90)
        print(f"\n因子漂移明细({len(result['drifts'])}个因子):")
        for d in result["drifts"]:
            print(f"  {d['factor']:<12s} short={d['short_exposure']:+.3f} long={d['long_exposure']:+.3f} diff={d['diff']:+.3f}")

        print(f"\n触发预警数: {len(result['alerts'])}")
        for a in result["alerts"]:
            print(f"  [{a['severity'].upper():<6s}] {a['message']}")

        assert "drifts" in result and "alerts" in result and "timestamp" in result
        assert len(result["drifts"]) == len(FACTOR_NAMES)

        # 单元测试：纯数值级别的窗口暴露/漂移判定，不依赖网络
        synthetic_close_short = np.array([100.0, 105.0, 110.0, 120.0, 130.0])
        synthetic_close_long = np.array([100.0] * 90)
        exp_short = detector._window_exposure(synthetic_close_short)
        exp_long = detector._window_exposure(synthetic_close_long)
        assert exp_short["momentum"] > exp_long["momentum"], "上涨窗口动量暴露应更高"

        print("\n✅ StyleDrift 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ StyleDrift 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
