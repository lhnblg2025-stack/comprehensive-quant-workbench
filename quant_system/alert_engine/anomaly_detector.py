"""
anomaly_detector.py — 统计异常检测 (V5)

用经典统计方法（Z-score / MAD / IQR）检测组合各维度指标的异常值，
不依赖固定阈值规则，适合发现"没预见到的"新型风险。

三种检测器各有适用场景：
  - Z-score: 假设近似正态分布，对异常值敏感（但异常值本身会拉高均值/标准差）
  - MAD (Median Absolute Deviation): 鲁棒统计量，不受极端值影响，适合厚尾分布
  - IQR (Interquartile Range): 箱线图法，适合非对称分布

对标：量化风控中的统计过程控制 (SPC) / 异常检测层
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


class AnomalyDetector:
    """统计异常检测引擎——扫描持仓组合的多维指标，发现统计意义上的异常。

    Attributes:
        default_zscore_threshold: Z-score 检测默认阈值
        default_mad_threshold: MAD 检测默认阈值
    """

    def __init__(self) -> None:
        self.default_zscore_threshold = 3.0
        self.default_mad_threshold = 3.5

    # ────────────────────────────────────────────────────────
    # 单变量检测方法
    # ────────────────────────────────────────────────────────

    def detect_zscore(self, values: list[float], threshold: float = 3.0) -> list[dict[str, Any]]:
        """Z-score 异常检测：|x - mean| / std > threshold 视为异常。

        Args:
            values: 数值序列
            threshold: Z-score 阈值，默认 3（约99.7%置信区间外）

        Returns:
            异常点列表 [{"index": i, "value": v, "zscore": z, "severity": ...}, ...]
        """
        if not values or len(values) < 2:
            return []
        arr = np.asarray(values, dtype=float)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        if std < 1e-12:
            return []

        results = []
        for i, v in enumerate(arr):
            if np.isnan(v):
                continue
            z = (v - mean) / std
            if abs(z) > threshold:
                results.append({
                    "index": i,
                    "value": round(float(v), 6),
                    "zscore": round(float(z), 3),
                    "method": "zscore",
                    "threshold": threshold,
                    "severity": self._severity_from_ratio(abs(z) / threshold),
                })
        return results

    def detect_mad(self, values: list[float], threshold: float = 3.5) -> list[dict[str, Any]]:
        """MAD（中位数绝对偏差）异常检测——比 Z-score 更鲁棒，不受极端值污染均值/方差。

        修正 Z-score = 0.6745 * (x - median) / MAD
        (0.6745 是使 MAD 对正态分布的渐近一致性系数)

        Args:
            values: 数值序列
            threshold: 修正Z-score阈值，常用 3.5

        Returns:
            异常点列表
        """
        if not values or len(values) < 2:
            return []
        arr = np.asarray(values, dtype=float)
        median = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - median)))
        if mad < 1e-12:
            return []

        results = []
        for i, v in enumerate(arr):
            if np.isnan(v):
                continue
            modified_z = 0.6745 * (v - median) / mad
            if abs(modified_z) > threshold:
                results.append({
                    "index": i,
                    "value": round(float(v), 6),
                    "modified_zscore": round(float(modified_z), 3),
                    "method": "mad",
                    "threshold": threshold,
                    "severity": self._severity_from_ratio(abs(modified_z) / threshold),
                })
        return results

    def detect_iqr(self, values: list[float], k: float = 1.5) -> list[dict[str, Any]]:
        """IQR（四分位距）异常检测——箱线图法，适合非对称/偏态分布。

        异常范围: [Q1 - k*IQR, Q3 + k*IQR] 之外

        Args:
            values: 数值序列
            k: 倍数系数，默认 1.5（标准箱线图），3.0 为"极端异常"

        Returns:
            异常点列表
        """
        if not values or len(values) < 4:
            return []
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) < 4:
            return []
        q1 = float(np.percentile(valid, 25))
        q3 = float(np.percentile(valid, 75))
        iqr = q3 - q1
        if iqr < 1e-12:
            return []
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        results = []
        for i, v in enumerate(arr):
            if np.isnan(v):
                continue
            if v < lower or v > upper:
                dist = max(lower - v, v - upper, 0.0)
                ratio = dist / max(iqr, 1e-8)
                results.append({
                    "index": i,
                    "value": round(float(v), 6),
                    "bounds": (round(lower, 6), round(upper, 6)),
                    "method": "iqr",
                    "threshold": k,
                    "severity": self._severity_from_ratio(ratio),
                })
        return results

    @staticmethod
    def _severity_from_ratio(ratio: float) -> str:
        """根据超出阈值的倍数比例映射严重程度。"""
        if ratio >= 2.0:
            return "high"
        if ratio >= 1.0:
            return "medium"
        return "low"

    # ────────────────────────────────────────────────────────
    # 组合级扫描
    # ────────────────────────────────────────────────────────

    def scan_portfolio(self, holdings: list[dict[str, Any]], window: int = 60) -> list[dict[str, Any]]:
        """扫描组合各维度（权重、日收益率、波动率、成交额等）的统计异常。

        P2-Q21-fix: 明确维度分组语义——每个数值字段（metric）是一个独立的
        检测维度，Z-score/MAD 只在**同一维度内**对横截面持仓做检测，不跨维度
        混算（weight 与 daily_return 量纲不同，不会放进同一个统计量）。
        输出带 dimension 字段标注所属维度组。

        Args:
            holdings: 持仓列表，每条含 code/name 以及若干数值字段：
                weight, daily_return, volatility, daily_turnover,
                pe_ratio, pb_ratio 等（缺失字段自动跳过该维度）
            window: 【已废弃，仅保留兼容】本方法为横截面检测，无历史窗口概念；
                    历史窗口应在调用前由上层聚合好传入 daily_return 等序列。
                    传该参数不产生任何效果，调用方应停止使用。

        Returns:
            [{metric, dimension, code, name, value, threshold, severity, method, timestamp}, ...]
        """
        if not holdings:
            return []

        now = datetime.now(CST).isoformat()
        # 候选可扫描字段（每个字段=一个独立检测维度）
        metrics = ["weight", "daily_return", "volatility", "daily_turnover",
                   "pe_ratio", "pb_ratio", "turnover_rate"]

        anomalies: list[dict[str, Any]] = []

        for metric in metrics:
            values = []
            index_map = []
            for i, h in enumerate(holdings):
                v = h.get(metric)
                if v is not None:
                    try:
                        values.append(float(v))
                        index_map.append(i)
                    except (TypeError, ValueError):
                        continue
            if len(values) < 4:
                continue  # 样本太少，跳过该维度

            mad_hits = self.detect_mad(values, threshold=self.default_mad_threshold)
            zscore_hits = self.detect_zscore(values, threshold=self.default_zscore_threshold)
            hit_indices = {h["index"] for h in mad_hits} | {h["index"] for h in zscore_hits}

            for local_idx in hit_indices:
                orig_idx = index_map[local_idx]
                holding = holdings[orig_idx]
                mad_hit = next((h for h in mad_hits if h["index"] == local_idx), None)
                z_hit = next((h for h in zscore_hits if h["index"] == local_idx), None)
                chosen = mad_hit or z_hit
                anomalies.append({
                    "metric": metric,
                    "dimension": metric,
                    "code": holding.get("code", ""),
                    "name": holding.get("name", ""),
                    "value": chosen["value"],
                    "threshold": chosen["threshold"],
                    "method": chosen["method"],
                    "severity": chosen["severity"],
                    "confirmed_by_both": bool(mad_hit and z_hit),
                    "timestamp": now,
                })

        # 按严重程度排序
        sev_rank = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda a: sev_rank.get(a["severity"], 3))
        return anomalies


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：构造含明显异常值的序列，验证三种检测方法。"""
    try:
        detector = AnomalyDetector()
        print("=" * 60)
        print("AnomalyDetector 统计异常检测 — 测试")
        print("=" * 60)

        # 正常分布 + 2个明显异常
        np.random.seed(42)
        normal_data = list(np.random.normal(0, 1, 30))
        values = normal_data + [15.0, -12.0]

        z_hits = detector.detect_zscore(values, threshold=3.0)
        mad_hits = detector.detect_mad(values, threshold=3.5)
        iqr_hits = detector.detect_iqr(values, k=1.5)

        print(f"\nZ-score 检测: {len(z_hits)} 个异常点")
        for h in z_hits:
            print(f"  index={h['index']} value={h['value']:.2f} z={h['zscore']}")

        print(f"\nMAD 检测: {len(mad_hits)} 个异常点")
        for h in mad_hits:
            print(f"  index={h['index']} value={h['value']:.2f} mz={h['modified_zscore']}")

        print(f"\nIQR 检测: {len(iqr_hits)} 个异常点")
        for h in iqr_hits:
            print(f"  index={h['index']} value={h['value']:.2f} bounds={h['bounds']}")

        assert len(z_hits) >= 1, "Z-score 应至少检出1个异常"

        # 组合扫描
        holdings = [
            {"code": "600519", "name": "贵州茅台", "weight": 0.15, "daily_return": -0.03,
             "daily_turnover": 5e8, "pe_ratio": 35},
            {"code": "000001", "name": "平安银行", "weight": 0.05, "daily_return": 0.01,
             "daily_turnover": 1e8, "pe_ratio": 6},
            {"code": "300750", "name": "宁德时代", "weight": 0.08, "daily_return": -0.08,
             "daily_turnover": 8e8, "pe_ratio": 40},
            {"code": "601318", "name": "中国平安", "weight": 0.06, "daily_return": 0.005,
             "daily_turnover": 3e8, "pe_ratio": 10},
            {"code": "000858", "name": "五粮液", "weight": 0.45, "daily_return": -0.15,
             "daily_turnover": 4e8, "pe_ratio": 200},
        ]
        portfolio_anomalies = detector.scan_portfolio(holdings)
        print(f"\n组合扫描异常数: {len(portfolio_anomalies)}")
        for a in portfolio_anomalies:
            print(f"  [{a['severity'].upper():<6s}] {a['code']} {a['name']} {a['metric']}={a['value']} ({a['method']})")

        print("\n✅ AnomalyDetector 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ AnomalyDetector 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
