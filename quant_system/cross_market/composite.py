"""
composite — 跨市场验证合成 (V5)

将 A-H溢价 / 汇率 / 股债性价比 / 北向资金 / 两融
五个维度的验证结果合成为单一结论。

核心回答："当前市场信号是否有跨市场数据支撑？"
"""

from __future__ import annotations

import re as _re
import time as _time
from datetime import datetime, timezone, timedelta
from typing import Any

from .hk_stock_link import HkStockLink
from .fx_impact import FxImpact
from .bond_equity import BondEquity
from .north_flow_deep import NorthFlowDeep
from .margin_deep import MarginDeep

CST = timezone(timedelta(hours=8))

# P1-Q23-fix(H01): 精确解析信号文本中的"历史XX%分位"，替代"偏贵/偏便宜"
# 子串匹配。信号文本如"溢价处于历史92%分位,A股相对港股明显偏贵,注意回调
# 风险" / "溢价处于历史18%分位,A股相对港股偏便宜(或港股偏贵)"。
_AH_PCT_RE = _re.compile(r"历史\s*(\d+(?:\.\d+)?)\s*%?\s*分位")


class CrossMarketComposite:
    """跨市场验证合成。"""

    def __init__(self) -> None:
        self.hk = HkStockLink()
        self.fx = FxImpact()
        self.bond = BondEquity()
        self.north = NorthFlowDeep()
        self.margin = MarginDeep()
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl: int = 900

    def verify(self, signal_type: str | None = None) -> dict[str, Any]:
        """验证跨市场数据是否支撑当前信号。

        Args:
            signal_type: 信号类型（预留）

        Returns:
            {
                "timestamp": str,
                "validations": {维度: "支撑"/"反对"/"中性"},
                "overall_support": "一致看多" / "分歧" / "一致看空",
                "confidence": "高" / "中" / "低",
                "details": {各维度详情},
                "alerts": [],
            }
        """
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        # 获取各维度数据（失败/无数据 → 记入 degraded，Q23-fix：不再静默降级）
        degraded: list[str] = []

        def _safe_compute(name: str, fn):
            try:
                data = fn()
                if data is None:
                    raise ValueError("compute() 返回 None")
                # 模块内部吞掉异常后返回 error 字典 → 同样视为降级
                if data.get("error"):
                    raise ValueError(str(data["error"])[:80])
                return data
            except Exception:
                degraded.append(name)
                return {"signal": "中性"}

        hk_data = _safe_compute("AH溢价", self.hk.compute)
        fx_data = _safe_compute("汇率影响", self.fx.compute)
        bond_data = _safe_compute("股债性价比", self.bond.compute)
        north_data = _safe_compute("北向资金", self.north.compute)
        margin_data = _safe_compute("两融", self.margin.compute)

        # P2-Q23-fix(M265): 模块返回 ok=False（数据源不可用但未抛异常，如
        # bond_equity "数据源不可用"、fx "仅获取到实时现价"、hk 降级为股票名单）
        # 同样记入 degraded——不再被静默当作"中性"参与计票稀释分母；details 仍
        # 保留真实降级原因，alerts 统一标注。
        for dim, data in (("AH溢价", hk_data), ("汇率影响", fx_data),
                          ("股债性价比", bond_data), ("北向资金", north_data),
                          ("两融", margin_data)):
            if dim not in degraded and isinstance(data, dict) and data.get("ok") is False:
                degraded.append(dim)

        # 规则: 各维度信号 → 看多/看空/中性
        # Q23-fix: 禁止子串匹配（"偏便宜(或港股偏贵)"含"偏贵"子串会把
        # 偏便宜误判为看空/看多方向颠倒）。各维度优先读结构化字段，
        # 文本兜底用精确/有序匹配。
        def classify_signal(data: dict, dim: str) -> str:
            sig = str(data.get("signal", "中性"))
            direction = str(data.get("direction", ""))

            if dim == "AH溢价":
                # P1-Q23-fix(H01): 结构化 premium_percentile 优先
                # （高分位→A股偏贵→看空，低分位→偏便宜→看多）。
                # 文本兜底改用精确解析"历史XX%分位"，禁止子串匹配——
                # "偏便宜(或港股偏贵)"含"偏贵"子串，子串匹配会把高分位
                # 与低分位两种状态都投成同一方向，维度投票恒失真。
                pct = data.get("premium_percentile")
                if isinstance(pct, (int, float)) and not isinstance(pct, bool) and pct >= 0:
                    if pct >= 80:
                        return "看空"
                    if pct <= 20:
                        return "看多"
                    return "中性"
                m = _AH_PCT_RE.search(sig)
                if m:
                    pct = float(m.group(1))
                    if pct >= 80:
                        return "看空"
                    if pct <= 20:
                        return "看多"
                    return "中性"
                return "中性"

            if dim == "汇率影响":
                if "利好出口" in sig:
                    return "看多"
                if "不利于出口" in sig or "利空" in sig:
                    return "看空"
                return "中性"

            if dim == "股债性价比":
                # Q23-fix: 对齐 bond_equity 实际信号文本
                if "性价比较高" in sig:
                    return "看多"
                if "性价比较低" in sig or "债券更具吸引力" in sig:
                    return "看空"
                return "中性"

            if dim == "北向资金":
                # 结构化：方向字段优先；2024-08-19 后净买额未披露 → 中性
                if data.get("net_buy_disclosed") is False:
                    return "中性"
                if direction in ("大幅流入", "流入"):
                    return "看多"
                if direction in ("大幅流出", "流出"):
                    return "看空"
                if "大幅净流入" in sig or "加仓" in sig or "看好" in sig:
                    return "看多"
                if "大幅净流出" in sig or "撤离" in sig:
                    return "看空"
                return "中性"

            if dim == "两融":
                if data.get("ok") is False or data.get("total_margin") is None:
                    return "中性"
                if direction == "加杠杆":
                    return "看多"
                if direction == "去杠杆":
                    return "看空"
                if "加杠杆" in sig:
                    return "看多"
                if "去杠杆" in sig or "过热" in sig:
                    return "看空"
                return "中性"

            return "中性"

        validations = {
            "AH溢价": classify_signal(hk_data, "AH溢价"),
            "汇率影响": classify_signal(fx_data, "汇率影响"),
            "股债性价比": classify_signal(bond_data, "股债性价比"),
            "北向资金": classify_signal(north_data, "北向资金"),
            "两融": classify_signal(margin_data, "两融"),
        }
        # 降级维度不参与计票（Q23-fix）
        # 除异常外，数据缺失/规则变更导致的无效维度同样剔除：
        #   北向净买额 2024-08-19 后停止披露 → 信号不可用
        #   两融数据获取失败(ok=False 或余额缺失) → 信号不可用
        if north_data.get("net_buy_disclosed") is False and "北向资金" not in degraded:
            degraded.append("北向资金")
        if (margin_data.get("ok") is False or margin_data.get("total_margin") is None) \
                and "两融" not in degraded:
            degraded.append("两融")
        for dim in degraded:
            validations[dim] = "中性"

        # 计票（Q23-fix: 分母剔除降级维度，避免死维度稀释投票）
        bullish_count = sum(1 for v in validations.values() if v == "看多")
        bearish_count = sum(1 for v in validations.values() if v == "看空")
        total = len(validations) - len(degraded)

        if bullish_count >= total * 0.6:
            overall = "一致看多"
            confidence = "高" if bullish_count >= total * 0.8 else "中"
        elif bearish_count >= total * 0.6:
            overall = "一致看空"
            confidence = "高" if bearish_count >= total * 0.8 else "中"
        elif bullish_count > bearish_count:
            overall = "偏多分歧"
            confidence = "低"
        elif bearish_count > bullish_count:
            overall = "偏空分歧"
            confidence = "低"
        else:
            overall = "分歧"
            confidence = "低"

        alerts = []
        if total == 0:
            overall = "数据不足"
            confidence = "低"
            alerts.append("⚠️ 全部维度数据不可用 — 跨市场验证无法给出结论")
        elif bullish_count >= total * 0.6:
            overall = "一致看多"
            confidence = "高" if bullish_count >= total * 0.8 else "中"
        elif bearish_count >= total * 0.6:
            overall = "一致看空"
            confidence = "高" if bearish_count >= total * 0.8 else "中"
        elif bullish_count > bearish_count:
            overall = "偏多分歧"
            confidence = "低"
        elif bearish_count > bullish_count:
            overall = "偏空分歧"
            confidence = "低"
        else:
            overall = "分歧"
            confidence = "低"

        if overall == "一致看多":
            alerts.append("🟢 全维度看多 — 历史胜率高,可积极做多")
        elif overall == "一致看空":
            alerts.append("🔴 全维度看空 — 建议减仓避险")
        elif "分歧" in overall:
            alerts.append("⚠️ 跨市场存在分歧 — 注意控制仓位")

        # Q23-fix: 降级维度显式标注，用户可感知
        for dim in degraded:
            alerts.append(f"⚠️ 维度[{dim}]数据不可用,已从计票中剔除")

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "validations": validations,
            "overall_support": overall,
            "confidence": confidence,
            "details": {
                "hk_stock_link": hk_data,
                "fx_impact": fx_data,
                "bond_equity": bond_data,
                "north_flow": north_data,
                "margin": margin_data,
            },
            "degraded": degraded,
            "alerts": alerts,
        }

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    cmc = CrossMarketComposite()
    r = cmc.verify()
    print("═" * 55)
    print("  跨市场验证合成")
    print("═" * 55)
    print(f"  总体判断: {r.get('overall_support', 'N/A')}")
    print(f"  置信度:   {r.get('confidence', 'N/A')}")
    print()
    print("  维度验证:")
    for dim, val in r.get("validations", {}).items():
        marker = "🟢" if val == "看多" else "🔴" if val == "看空" else "⚪"
        print(f"  {marker} {dim}: {val}")
    for a in r.get("alerts", []):
        print(f"  {a}")


if __name__ == "__main__":
    main()
