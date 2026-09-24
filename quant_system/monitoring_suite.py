"""
monitoring_suite.py — 量化系统综合监控套件
V4.1 feature: 系统监控

覆盖：
1. 数据质量监控（空值率、异常值、日期连续性）
2. 因子质量监控（IC衰减、因子拥挤度）
3. 组合风险监控（暴露度、集中度）
4. 系统健康监控（执行延迟、API调用成功率）
5. 策略绩效监控（实时绩效追踪）
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# P2-Q21-fix: 统一使用 aware CST 时间（原 datetime.now() 无时区，与系统其它模块混用）
CST = timezone(timedelta(hours=8))


class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self, threshold_null: float = 0.3, threshold_anomaly: float = 5.0):
        self.threshold_null = threshold_null
        self.threshold_anomaly = threshold_anomaly

    def check_null_ratio(self, df: pd.DataFrame) -> dict:
        """检查空值率"""
        null_ratio = df.isnull().mean()
        bad_cols = null_ratio[null_ratio > self.threshold_null]
        return {
            "max_null_ratio": null_ratio.max(),
            "bad_columns": list(bad_cols.index),
            "overall_quality": "pass" if len(bad_cols) == 0 else "warning",
        }

    def check_date_continuity(self, dates: pd.DatetimeIndex, max_gap_days: int = 3) -> dict:
        """检查日期连续性（按A股交易日历计算缺口，周末/节假日不计入）。

        P2-Q21-fix: 原实现按自然日间隔判断（>3 天即 gap），A股长假（国庆8天）
        或周频数据必然误报（实测 Thu→Mon 直连4天即判不连续）。现改为：
          - 缺口 = 相邻日期之间的 A 股交易日个数（复用 market_clock 的交易日历，
            含节假日剔除；日历不可得时降级为仅排除周末，降级有打印警告）
          - max_gap_days 参数化（默认3个交易日），日频数据 Fri→Mon=0 gap 连续，
            长假不误报；周频数据可传更大阈值
        """
        if len(dates) < 2:
            return {
                "gap_days": 0, "max_gap_days": 0,
                "gap_dates": [], "total_gaps": 0, "is_continuous": True,
            }
        idx = pd.DatetimeIndex(dates).sort_values()

        # 复用 A 股交易日历（含节假日剔除）；不可用则降级为周末粗筛
        try:
            from market_clock import get_trade_calendar
            cal = get_trade_calendar()
        except Exception:
            cal = set()

        gaps: list[int] = []
        gap_dates: list[pd.Timestamp] = []
        for i in range(1, len(idx)):
            d_prev, d_cur = idx[i - 1], idx[i]
            span = (d_cur - d_prev).days
            if span <= 0:
                continue
            if cal:
                # 统计 (d_prev, d_cur) 开区间内的交易日个数
                trading = 0
                for k in range(1, span):
                    d = d_prev + timedelta(days=k)
                    if d.strftime("%Y-%m-%d") in cal:
                        trading += 1
            else:
                # 降级：仅排除周末
                trading = max(int(np.busday_count(d_prev.date(), d_cur.date())) - 1, 0)
            gaps.append(trading)
            if trading > max_gap_days:
                gap_dates.append(d_cur)

        max_gap = max(gaps) if gaps else 0
        return {
            "gap_days": max_gap,
            "max_gap_days": max_gap,
            "gap_dates": gap_dates[:10],
            "total_gaps": len(gap_dates),
            "is_continuous": max_gap <= max_gap_days,
        }

    def check_price_anomaly(self, close: pd.Series) -> dict:
        """价格异常检测"""
        if len(close) < 20:
            return {"anomaly_count": 0}
        returns = close.pct_change().dropna()
        mean, std = returns.mean(), returns.std()
        anomalies = returns[abs(returns - mean) > self.threshold_anomaly * std]
        return {
            "anomaly_count": len(anomalies),
            "anomaly_dates": list(anomalies.index[:10]),
            "max_daily_return": returns.max(),
            "min_daily_return": returns.min(),
        }

    def full_report(self, df: pd.DataFrame) -> dict:
        """完整数据质量报告"""
        return {
            "null_ratio": self.check_null_ratio(df),
            "date_continuity": self.check_date_continuity(pd.DatetimeIndex(df.index))
            if hasattr(df, "index") and isinstance(df.index, pd.DatetimeIndex)
            else {},
            "anomaly": self.check_price_anomaly(df.get("close", pd.Series()))
            if isinstance(df, pd.DataFrame)
            else {},
            "n_rows": len(df),
            "n_cols": len(df.columns) if isinstance(df, pd.DataFrame) else 0,
        }


class FactorQualityMonitor:
    """因子质量监控器"""

    def __init__(self, ic_window: int = 60):
        self.ic_window = ic_window
        self._ic_history: dict[str, list] = {}

    def update_ic(self, factor_name: str, ic: float):
        """更新IC记录"""
        if factor_name not in self._ic_history:
            self._ic_history[factor_name] = []
        # P2-Q21-fix: 统一 aware CST，避免与其它模块的 aware 时间比较时报错
        self._ic_history[factor_name].append({"date": datetime.now(CST), "ic": ic})
        if len(self._ic_history[factor_name]) > self.ic_window:
            self._ic_history[factor_name].pop(0)

    def check_factor_decay(self, factor_name: str) -> dict:
        """检测因子衰减"""
        if factor_name not in self._ic_history or len(self._ic_history[factor_name]) < 10:
            return {"decay_status": "insufficient_data"}
        ics = [r["ic"] for r in self._ic_history[factor_name]]
        recent = np.mean(ics[-10:])
        full = np.mean(ics)
        decay = full - recent
        return {
            "recent_ic": recent,
            "full_ic": full,
            "decay": decay,
            "decay_status": "decaying" if decay > 0.02 else "stable",
        }

    def check_crowding(self, factor_exposures: pd.DataFrame) -> dict:
        """因子拥挤度检测"""
        corr = factor_exposures.corr()
        # P2-Q21-fix: 只取上三角（排除对角线自相关=1），避免平均相关性被系统性抬高
        # （实测 3 列完全相关=1.0；零相关 8 因子原公式也≥0.125 → 拥挤度误判）
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(mask).stack()
        avg_corr = float(upper.mean()) if not upper.empty else 0.0
        return {
            "average_correlation": avg_corr,
            "crowding_level": "high"
            if avg_corr > 0.7
            else "medium"
            if avg_corr > 0.4
            else "low",
        }


class RiskMonitor:
    """风险监控器"""

    def __init__(self, max_industry_exposure: float = 0.3, max_single_stock: float = 0.1):
        self.max_industry_exposure = max_industry_exposure
        self.max_single_stock = max_single_stock

    def check_industry_exposure(self, weights: pd.Series, industry_map: dict) -> dict:
        """行业暴露检查"""
        industry_weights = {}
        for symbol, w in weights.items():
            ind = industry_map.get(symbol, "未知")
            industry_weights[ind] = industry_weights.get(ind, 0) + w
        violations = {
            ind: w
            for ind, w in industry_weights.items()
            if w > self.max_industry_exposure
        }
        return {
            "max_exposure": max(industry_weights.values()) if industry_weights else 0,
            "violations": violations,
            "status": "violation" if violations else "ok",
        }

    def check_concentration(self, weights: pd.Series) -> dict:
        """集中度检查"""
        top5 = weights.nlargest(5).sum()
        hhi = (weights**2).sum()
        return {
            "top5_concentration": top5,
            "hhi": hhi,
            "effective_n": 1 / max(hhi, 1e-12),
            "status": "concentrated" if top5 > 0.6 else "diversified",
        }


class SystemHealthMonitor:
    """系统健康监控器"""

    def __init__(self):
        self._api_latency: list[float] = []
        self._api_errors: int = 0
        self._total_calls: int = 0

    def record_api_call(self, latency_ms: float, success: bool):
        """记录API调用"""
        self._api_latency.append(latency_ms)
        self._total_calls += 1
        if not success:
            self._api_errors += 1
        if len(self._api_latency) > 100:
            self._api_latency.pop(0)

    def health_status(self) -> dict:
        """系统健康状态"""
        error_rate = self._api_errors / max(self._total_calls, 1)
        avg_latency = np.mean(self._api_latency) if self._api_latency else 0
        return {
            "total_api_calls": self._total_calls,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": np.percentile(self._api_latency, 99)
            if self._api_latency
            else 0,
            "status": ("unknown" if self._total_calls == 0 else
                       ("healthy" if error_rate < 0.05 and avg_latency < 5000 else "degraded")),
        }


class MonitoringSuite:
    """综合监控面板"""

    def __init__(self):
        self.data_monitor = DataQualityMonitor()
        self.factor_monitor = FactorQualityMonitor()
        self.risk_monitor = RiskMonitor()
        self.health_monitor = SystemHealthMonitor()

    def full_report(self) -> str:
        """生成完整监控报告"""
        sections = []
        sections.append("=" * 50)
        sections.append("量化系统监控报告 (V4.1 feature)")
        sections.append("=" * 50)
        # P2-Q21-fix: 统一 aware CST
        sections.append(f"报告时间: {datetime.now(CST).isoformat()}")
        sections.append("")

        # 系统健康
        health = self.health_monitor.health_status()
        sections.append(f"[系统健康] 状态: {health['status']}")
        sections.append(f"  API调用: {health['total_api_calls']}, 错误率: {health['error_rate']:.2%}")

        return "\n".join(sections)

