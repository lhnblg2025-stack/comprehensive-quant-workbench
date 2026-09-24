"""
price_sentiment — 基于价格结构的多维度情绪 (V5)

核心逻辑：不直接看涨跌幅，看涨跌结构。
- 涨跌家数比（实时）+ 涨跌停比
- 指数 vs 全 A 等权背离（沪深300涨但全A跌 = 情绪虚假）

输出：
  score: -10~10
  percentile / approx_percentile: 模型近似分位 0~100（正态映射）。
         P1-Q11-fix(H04): 原实现宣称"1年/3年/5年真实历史分位"，但数据源
         ak.stock_market_fund_em 不存在（实测 AttributeError），
         _get_advance_decline_history/_get_percentile 为死代码，已删除；
         真实历史分位暂不可用，percentile 明确为模型近似。
  divergence: 是否与指数走势背离
"""

from __future__ import annotations
import logging

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent

# P1-Q11-fix(H04): 删除死代码 _PERCENTILE_CACHE / _get_advance_decline_history /
# _get_percentile —— 其唯一数据源 ak.stock_market_fund_em 不存在（实测
# AttributeError），且模块内从未使用这些缓存与函数。percentile 改为模型近似，
# 见模块 docstring。


class PriceSentiment:
    """基于价格结构的多维度情绪。"""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl: int = 300  # 5 min

    def compute(self) -> dict[str, Any]:
        """计算当前价格情绪。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "score": 0,
            "direction": "neutral",
            "percentile": 50.0,
            "approx_percentile": 50.0,
            "divergence": False,
            "sub_signals": [],
        }

        try:
            import akshare as ak
            today = datetime.now(CST).strftime("%Y%m%d")

            # ── 1. 涨跌家数比（实时） ──
            try:
                # Q11-fix: ak.stock_adv_dec_stats 不存在（实测 AttributeError），
                # 改用 market_pulse._fetch_adv_dec（stock_zh_a_spot_em 统计）。
                from quant_system.market_pulse import _fetch_adv_dec
                adv_dec = _fetch_adv_dec()
                if adv_dec:
                    up = adv_dec.get('up', 0)
                    down = adv_dec.get('down', 0)
                    total = up + down
                    if total > 0:
                        ad_ratio = up / total
                        # 评分：0.5 = 中性，>0.6 = 积极，<0.4 = 消极
                        ad_score = (ad_ratio - 0.5) * 40  # -20 ~ +20
                        result["ad_ratio"] = round(ad_ratio, 3)
                        result["ad_score"] = round(ad_score, 1)
                        result["sub_signals"].append(
                            f"涨跌比 {ad_ratio:.0%} (得分 {ad_score:+.0f})"
                        )
                    elif adv_dec.get('note'):
                        # P1-Q11-fix(H03): 降级必须可见，不静默吞异常
                        result["sub_signals"].append(
                            f"⚠️ 涨跌比不可用: {adv_dec['note']}")
            except Exception as e:
                logging.getLogger(__name__).error(f"[price_sentiment] 操作失败: {e}", exc_info=True)

            # ── 2. 涨跌停比 + 封板率 ──
            try:
                zt = ak.stock_zt_pool_em(date=today)
                dt = ak.stock_zt_pool_dtgc_em(date=today)
                up_limit = len(zt) if zt is not None else 0
                down_limit = len(dt) if dt is not None else 0
                if up_limit + down_limit > 0:
                    zt_ratio = up_limit / (up_limit + down_limit)
                    zt_score = (zt_ratio - 0.5) * 20  # -10 ~ +10
                    result["zt_ratio"] = round(zt_ratio, 3)
                    result["zt_score"] = round(zt_score, 1)
                    result["sub_signals"].append(
                        f"涨跌停比 {up_limit}/{down_limit} ({zt_ratio:.0%}, 得分 {zt_score:+.0f})"
                    )
                else:
                    result["zt_ratio"] = 0.5
                    result["zt_score"] = 0.0
                    # W2.5 修复: 涨跌停数据不可用，显式告警（不再静默按中性并入 composite）
                    result["sub_signals"].append("⚠️ 涨跌停数据不可用，zt_ratio 按中性兜底")
            except Exception as e:
                logging.getLogger(__name__).error(f"[price_sentiment] 操作失败: {e}", exc_info=True)

            # ── 3. 指数 vs 全A等权背离 ──
            try:
                sh = ak.stock_zh_index_daily(symbol="sh000001")
                if sh is not None and len(sh) > 20:
                    sh_close = sh["close"].values
                    sh_5d = (sh_close[-1] / sh_close[-6] - 1) * 100 if len(sh) >= 6 else 0
                    # 简单近似：用涨跌比方向 vs 指数方向判断背离
                    if sh_5d > 2 and result.get("ad_ratio", 0.5) < 0.4:
                        result["divergence"] = True
                        result["divergence_type"] = "指数涨但涨少跌多 = 虚假上涨"
                        result["sub_signals"].append("⚠️ 指数与广度背离: 指数涨但涨少跌多")  # P2-Q11-fix(L052): 未使用成交量，文案改为广度背离
                    elif sh_5d < -2 and result.get("ad_ratio", 0.5) > 0.6:
                        result["divergence"] = True
                        result["divergence_type"] = "指数跌但涨多跌少 = 恐慌过度"
                        result["sub_signals"].append("⚠️ 指数与广度背离: 指数跌但涨多跌少")  # P2-Q11-fix(L052): 未使用成交量，文案改为广度背离
                    result["sh_5d_pct"] = round(sh_5d, 2)
            except Exception as e:
                logging.getLogger(__name__).error(f"[price_sentiment] 操作失败: {e}", exc_info=True)

            # ── 4. 综合评分 ──
            # P1-Q11-fix(H03): 原 raw_score = ad*0.6 + zt*0.4 后 /2，而
            # ad_score∈[-20,20]、zt_score∈[-10,10]，实际区间仅 ±8 与文档
            # -10~10 不符；且 ad_score 曾因广度 API 失效恒 0 → 综合分恒 ∈[-2,2]
            # → direction 恒 neutral。现将每个分量先归一化到 -10~10 再按权重
            # 合成：ad/2 ∈[-10,10]、zt ∈[-10,10]，保证 score 恰在 [-10,10]。
            ad = result.get("ad_score", 0)
            zt = result.get("zt_score", 0)
            raw_score = (ad / 2) * 0.6 + zt * 0.4
            result["score"] = round(max(-10, min(10, raw_score)), 1)

            # 方向
            if result["score"] >= 3:
                result["direction"] = "bullish"
            elif result["score"] <= -3:
                result["direction"] = "bearish"
            else:
                result["direction"] = "neutral"

            # P1-Q11-fix(H04): 原实现把"50 + score*6.8"的正态模型近似冒充为
            # "历史分位 0~100"。真实历史分位数据源不存在（见模块 docstring），
            # 明确改名为 approx_percentile，同时保留 percentile 键兼容
            # composite.py / tests 调用方，并在文档标注为模型近似。
            approx = round(max(0, min(100, 50 + result["score"] * 6.8)), 1)
            result["approx_percentile"] = approx
            result["percentile"] = approx

        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result

    def get_divergence_alerts(self) -> list[str]:
        """获取背离预警。"""
        data = self.compute()
        alerts = []
        if data.get("divergence"):
            alerts.append(data.get("divergence_type", "背离检测"))
        # 极端情绪预警
        # P1-Q11-fix(H06): score 为 float，原 :+d 整数格式对 float 抛 ValueError，
        # 改为 :+.1f 保留一位小数。
        if data.get("score", 0) >= 7:
            alerts.append(f"极度乐观(评分{data['score']:+.1f}) — 警惕回调")
        elif data.get("score", 0) <= -7:
            alerts.append(f"极度悲观(评分{data['score']:+.1f}) — 关注反弹机会")
        return alerts


def main() -> None:
    """CLI 入口"""
    ps = PriceSentiment()
    result = ps.compute()
    print("═" * 50)
    print(f"  价格情绪评分: {result.get('score', 'N/A'):>+5}  |  "
          f"方向: {result.get('direction', 'N/A'):>8}  |  "
          f"分位: {result.get('percentile', 'N/A')}")
    print("═" * 50)
    for s in result.get("sub_signals", []):
        print(f"  • {s}")
    if result.get("divergence"):
        print(f"  ⚠️ {result.get('divergence_type', '')}")
    alerts = ps.get_divergence_alerts()
    for a in alerts:
        print(f"  🔔 {a}")


if __name__ == "__main__":
    main()
