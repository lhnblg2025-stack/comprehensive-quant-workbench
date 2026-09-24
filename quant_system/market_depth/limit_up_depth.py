"""
limit_up_depth — 涨停板深度分析 (V5)

涨停板数据包含了大量市场情绪信息：
  1. 涨停家数 — 市场热度
  2. 连板高度 — 龙头持续性（最高连板数）
  3. 封板率 — 封板个股/触及涨停个股
  4. 炸板率 — 涨停后打开比例
  5. 题材分布 — 涨停股所属概念集中度

这些信息"温度"无法提供，"涨跌"也看不出来。
"""

from __future__ import annotations
import logging

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

CST = timezone(timedelta(hours=8))


class LimitUpDepth:
    """涨停板深度分析。"""

    def __init__(self, cache_ttl: int = 300) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算涨停板深度指标。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "score": 0,
            "direction": "neutral",
            "signals": [],
            "alerts": [],
            # P2-Q22-fix(M244): 数据质量标记，失败必须可见而非静默默认0
            "data_quality": {},
        }

        try:
            import akshare as ak
            today = datetime.now(CST).strftime("%Y%m%d")

            # ── 1. 涨停/跌停数量和分布 ──
            try:
                zt = ak.stock_zt_pool_em(date=today)
                dt = ak.stock_zt_pool_dtgc_em(date=today)
            except Exception as e:
                # P2-Q22-fix(M244): 原回退逻辑 zt=ak.stock_zt_pool_dtgc_em(...); zt=[]
                # 回退结果立即被清空，主接口失败时涨停/跌停均报0，评分被拉低。
                # 现分别重试涨停池/跌停池、各自校验非空，并在 data_quality 中显式标记。
                result["data_quality"]["zt"] = f"error: 主接口异常: {e}"
                zt = None
                dt = None
                try:
                    zt_fb = ak.stock_zt_pool_em(date=today)
                    zt = zt_fb if zt_fb is not None and not zt_fb.empty else None
                except Exception as e2:
                    result["data_quality"]["zt"] += f" | 涨停池重试失败: {e2}"
                try:
                    dt_fb = ak.stock_zt_pool_dtgc_em(date=today)
                    dt = dt_fb if dt_fb is not None and not dt_fb.empty else None
                except Exception as e3:
                    result["data_quality"]["zt"] += f" | 跌停池重试失败: {e3}"

            up_limit = len(zt) if zt is not None else 0
            down_limit = len(dt) if dt is not None else 0
            result["up_limit_count"] = up_limit
            result["down_limit_count"] = down_limit

            if up_limit > 0:
                result["signals"].append(f"涨停{up_limit}家")
            if down_limit > 0:
                result["signals"].append(f"跌停{down_limit}家")

            # 涨停/跌停比
            total_limit = up_limit + down_limit
            if total_limit > 0:
                up_limit_ratio = up_limit / total_limit
                result["up_limit_ratio"] = round(up_limit_ratio, 3)
            else:
                result["up_limit_ratio"] = 0.5

            # ── 2. 连板分析 ──
            # 使用板块涨停池判断最高连板高度（简化：取涨停股的名字前缀）
            consecutive_info = {"max_consecutive": 0, "consecutive_stocks": []}
            if zt is not None and not zt.empty:
                try:
                    # 涨停股中有 代码、名称、连板数、首次封板时间 等列
                    # P1-Q22-fix: akshare stock_zt_pool_em 列名为"连板数"(非"连板")，且需容错非数字
                    if "连板数" in zt.columns:
                        boards = []
                        for v in zt["连板数"]:
                            try:
                                boards.append(int(float(v)))
                            except (TypeError, ValueError):
                                continue
                        if len(boards) > 0:
                            max_board = max(boards)
                            consecutive_info["max_consecutive"] = max_board
                            result["signals"].append(
                                f"最高连板{max_board}板"
                            )
                            if max_board >= 7:
                                result["alerts"].append(
                                    f"🔥 最高{max_board}连板 — 情绪极度亢奋"
                                )
                            elif max_board >= 5:
                                result["alerts"].append(
                                    f"🔥 最高{max_board}连板 — 短线情绪高涨"
                                )
                    # 涨停时间分布（早盘封板 = 强）
                    # P1-Q22-fix: 列名为"首次封板时间"，值为 HHMMSS 字符串，需解析为分钟数再与 10:00(=600分) 比较
                    if "首次封板时间" in zt.columns:
                        early = 0
                        for t in zt["首次封板时间"]:
                            t_str = str(t).strip()
                            if len(t_str) >= 4 and t_str.isdigit():
                                minutes = int(t_str[:2]) * 60 + int(t_str[2:4])
                                if minutes < 600:
                                    early += 1
                        result["early_limit_pct"] = round(
                            early / max(len(zt), 1) * 100, 1
                        )
                        if result["early_limit_pct"] > 60:
                            result["signals"].append(
                                f"早盘封板率{result['early_limit_pct']:.0f}% — 做多意愿强"
                            )
                except Exception as e:
                    logging.getLogger(__name__).error(f"[limit_up_depth] 操作失败: {e}", exc_info=True)
            result["consecutive"] = consecutive_info

            # ── 3. 题材集中度 ──
            # 统计涨停股的概念板块分布（需要板块数据）
            # 简化实现
            result["theme_concentration"] = "unknown"

            # ── 4. 评分 ──
            score = 0.0

            # 涨停数量评分
            if up_limit > 80:
                score += 4
            elif up_limit > 50:
                score += 3
            elif up_limit > 30:
                score += 2
            elif up_limit > 15:
                score += 1
            elif up_limit < 5:
                score -= 2

            # 跌停数量评分
            if down_limit > 30:
                score -= 4
            elif down_limit > 15:
                score -= 3
            elif down_limit > 8:
                score -= 2
            elif down_limit > 3:
                score -= 1

            # 连板高度修正
            max_b = consecutive_info.get("max_consecutive", 0)
            if max_b >= 7:
                score += 3  # 短线情绪高涨
            elif max_b >= 5:
                score += 2
            elif max_b >= 3:
                score += 1

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

        except Exception as e:
            result["error"] = str(e)

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    lud = LimitUpDepth()
    r = lud.compute()
    print("═" * 55)
    print(f"  涨停板深度")
    print("═" * 55)
    print(f"  涨停: {r.get('up_limit_count', '?')}  "
          f"跌停: {r.get('down_limit_count', '?')}  "
          f"比例: {r.get('up_limit_ratio', '?'):.0%}")
    print(f"  最高连板: {r.get('consecutive', {}).get('max_consecutive', '?')}板")
    print(f"  综合评分: {r.get('score', 'N/A'):>+5}  |  "
          f"方向: {r.get('direction', '?')}")
    print("═" * 55)
    for s in r.get("signals", []):
        print(f"  • {s}")
    for a in r.get("alerts", []):
        print(f"  {a}")


if __name__ == "__main__":
    main()
