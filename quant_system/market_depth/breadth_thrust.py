"""
breadth_thrust — 广度推力 (V5)

核心理念：真正的上涨不应该只有少数股票涨。
- 广度推力 = 创新高数 / (创新高+创新低) — 衡量上涨的扩散程度
- 涨跌比 5日平滑 — 消除单日噪音
- 广度推力 > 0.6 + 指数上涨 = 健康的上涨
- 指数上涨但广度推力 < 0.4 = 窄幅上涨（虚假突破信号）

与简单温度的区别：
  温度看"涨了多少"，广度看"多少人一起涨"
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any


CST = timezone(timedelta(hours=8))


class BreadthThrust:
    """广度推力分析。"""

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算当前广度推力。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "breadth_thrust": 0.0,
            "score": 0,
            "direction": "neutral",
            "signals": [],
            "alerts": [],
            # P1-Q22-fix: 各数据源质量标记，缺失/失败必须可见而非静默默认值
            "data_quality": {},
        }

        try:
            import akshare as ak
            today = datetime.now(CST).strftime("%Y%m%d")

            # ── 1. 涨跌家数比 ──
            try:
                # P1-Q22-fix: akshare 1.18.64 无 stock_adv_dec_stats，改用乐咕市场活跃度(含上涨/下跌家数)
                activity = ak.stock_market_activity_legu()
                if activity is not None and "item" in activity.columns:
                    item_map = dict(zip(activity["item"].astype(str), activity["value"]))
                    up = int(float(item_map.get("上涨", 0) or 0))
                    down = int(float(item_map.get("下跌", 0) or 0))
                    total = up + down
                    if total > 0:
                        ad_ratio = up / total
                        result["ad_ratio"] = round(ad_ratio, 3)
                        result["ad_up"] = up
                        result["ad_down"] = down

                        # 广度推力判断
                        if ad_ratio > 0.7:
                            thrust = (ad_ratio - 0.5) * 2  # 0.4 ~ 1.0
                            result["signals"].append(
                                f"广谱上涨({ad_ratio:.0%}股票上涨)"
                            )
                        elif ad_ratio < 0.3:
                            thrust = (ad_ratio - 0.5) * 2  # -0.4 ~ -1.0
                            result["signals"].append(
                                f"广谱下跌({ad_ratio:.0%}股票上涨)"
                            )
                        else:
                            thrust = (ad_ratio - 0.5) * 2  # -0.4 ~ 0.4

                        result["breadth_thrust"] = round(thrust, 3)
                    else:
                        result["data_quality"]["ad"] = "error: 上涨+下跌家数为0"
                else:
                    result["data_quality"]["ad"] = "error: stock_market_activity_legu 无数据"
            except Exception as e:
                # P1-Q22-fix: 降级必须可见——标记缺失而非静默 0
                result["data_quality"]["ad"] = f"error: {e}"

            # ── 2. 新高新低比 ──
            try:
                # P1-Q22-fix: akshare 1.18.64 无 stock_history_new_high/new_low，
                # 改用乐咕全A创新高/新低统计(high20/low20=20日新高/新低家数)
                hl = ak.stock_a_high_low_statistics(symbol="all")
                if hl is not None and len(hl) > 0:
                    row = hl.iloc[-1]
                    new_high = int(row.get("high20", 0) or 0)
                    new_low = int(row.get("low20", 0) or 0)
                    nh_nl_total = new_high + new_low
                    if nh_nl_total > 0:
                        nh_ratio = new_high / nh_nl_total
                        result["nh_nl_ratio"] = round(nh_ratio, 3)
                        result["new_high"] = new_high
                        result["new_low"] = new_low

                        if nh_ratio > 0.7 and result.get("breadth_thrust", 0) > 0.2:
                            result["signals"].append(
                                f"新高{new_high}/新低{new_low} — 多头主导"
                            )
                        elif nh_ratio < 0.3:
                            result["signals"].append(
                                f"新高{new_high}/新低{new_low} — 空头主导"
                            )
                    else:
                        result["data_quality"]["nh_nl"] = "error: 新高+新低数量为0"
                        result["nh_nl_ratio"] = 0.5
                        result["new_high"] = 0
                        result["new_low"] = 0
                else:
                    result["data_quality"]["nh_nl"] = "error: stock_a_high_low_statistics 无数据"
                    result["nh_nl_ratio"] = 0.5
                    result["new_high"] = 0
                    result["new_low"] = 0
            except Exception as e:
                # P1-Q22-fix: 降级必须可见——标记缺失而非静默 0.5
                result["data_quality"]["nh_nl"] = f"error: {e}"
                result["nh_nl_ratio"] = 0.5
                result["new_high"] = 0
                result["new_low"] = 0

            # ── 3. 涨跌停比 ──
            try:
                zt = ak.stock_zt_pool_em(date=today)
                dt = ak.stock_zt_pool_dtgc_em(date=today)
                up_limit = len(zt) if zt is not None else 0
                down_limit = len(dt) if dt is not None else 0
                result["zt_up"] = up_limit
                result["zt_down"] = down_limit
                if up_limit + down_limit > 0:
                    zt_ratio = up_limit / (up_limit + down_limit)
                    result["zt_ratio"] = round(zt_ratio, 3)
            except Exception as e:
                # P1-Q22-fix: 涨跌停数据失败也显式标记
                result["data_quality"]["zt"] = f"error: {e}"

            # ── 4. 综合评分 ──
            bt = result.get("breadth_thrust", 0)
            nh = result.get("nh_nl_ratio", 0.5)
            # 广度推力和新高比加权
            score = bt * 5 + (nh - 0.5) * 5  # -5 ~ +5 scale
            # 涨跌停比修正
            zt_r = result.get("zt_ratio", 0.5)
            score += (zt_r - 0.5) * 3

            result["score"] = round(max(-10, min(10, score)), 1)

            if result["score"] >= 3:
                result["direction"] = "bullish"
            elif result["score"] <= -3:
                result["direction"] = "bearish"
            else:
                result["direction"] = "neutral"

            # P2-Q22-fix(M243): 原 50+score*6.8 线性映射冒充"历史分位"输出。
            # 真实历史分位未实现，改输出字段标注为 score 映射值：
            #   - score_mapped_pctile 为新字段（明确为 score 映射）；
            #   - percentile 保留作兼容别名（值为同一映射值），
            #     并附 percentile_basis/percentile_note 说明其非历史分位。
            mapped = round(max(0, min(100, 50 + result["score"] * 6.8)), 1)
            result["score_mapped_pctile"] = mapped
            result["percentile"] = mapped
            result["percentile_basis"] = "score_mapping"
            result["percentile_note"] = "score线性映射值(50+score*6.8)，非真实历史分位；真实历史分位未实现"

            # ── 预警 ──
            if bt > 0.3 and nh < 0.4:
                result["alerts"].append(
                    "广度背离: 涨多跌少但新高少 — 反弹质量不高"
                )
            if bt < -0.2 and nh > 0.6:
                result["alerts"].append(
                    "背离: 跌多涨少但新高低 — 可能是最后一跌"
                )

        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    bt = BreadthThrust()
    r = bt.compute()
    print("═" * 55)
    print(f"  广度推力: {r.get('breadth_thrust', 'N/A'):>+7.3f}")
    print(f"  综合评分: {r.get('score', 'N/A'):>+5}  |  "
          f"方向: {r.get('direction', 'N/A')}")
    print(f"  分位(score映射,非历史分位): {r.get('score_mapped_pctile', r.get('percentile', 'N/A'))}%")
    print("═" * 55)
    for s in r.get("signals", []):
        print(f"  • {s}")
    for a in r.get("alerts", []):
        print(f"  ⚠️  {a}")
    print(f"  涨跌: {r.get('ad_up', '?')}/{r.get('ad_down', '?')}  "
          f"涨停/跌停: {r.get('zt_up', '?')}/{r.get('zt_down', '?')}  "
          f"新高/新低: {r.get('new_high', '?')}/{r.get('new_low', '?')}")


if __name__ == "__main__":
    main()
