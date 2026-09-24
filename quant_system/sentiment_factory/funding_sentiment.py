"""
funding_sentiment — 基于资金面的情绪维度 (V5)

分析两融余额 + 北向资金 + ETF 申赎的结构和方向。
不是看绝对值（两融2万亿还是3万亿），看：
- 两融余额的历史分位（Q11-H05: 改用 margin.py 同源真实历史序列计算）
- 两融增量/减量的方向和速度
- 北向资金的行业偏好变化
- 资金面 vs 大盘的背离

输出：score: -10~10；percentile / approx_percentile 为模型近似分位
（Q11-H04，正态映射，非真实历史分位，标注为模型近似）。
"""

from __future__ import annotations
import logging

import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent

# ── 两融历史余额缓存（供真实分位计算，Q11-H05） ──
_MARGIN_HIST_CACHE: dict[str, tuple[float, list[float]]] = {}
_MARGIN_HIST_TTL = 3600  # 1 小时


def _get_margin_balance_history(days: int = 1095) -> list[float]:
    """获取近 days 日沪市融资余额历史序列（亿元）。

    与 margin.py 同源（ak.stock_margin_sse 支持按日期范围批量返回），
    SSE 序列作为全市场余额的稳定代理（深市无批量历史接口）。
    单位统一换算为亿元；失败返回 []，由上层降级为中性分位并可见标注。
    """
    now = _time.time()
    hit = _MARGIN_HIST_CACHE.get("data")
    if hit and now - hit[0] < _MARGIN_HIST_TTL:
        return hit[1]
    history: list[float] = []
    try:
        import akshare as ak
        import pandas as pd
        start = (datetime.now(CST) - timedelta(days=days)).strftime("%Y%m%d")
        end = datetime.now(CST).strftime("%Y%m%d")
        df = ak.stock_margin_sse(start_date=start, end_date=end)
        if df is None or df.empty:
            return []
        col = next((c for c in df.columns
                    if "融资余额" in str(c) and "融券" not in str(c)), None)
        if col is None:
            return []
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        # 单位归一：>1e9 视为元 → 亿元；否则视为已是亿元
        if vals and max(abs(v) for v in vals) > 1e9:
            history = [v / 1e8 for v in vals]
        else:
            history = vals
    except Exception:
        return []
    if history:
        _MARGIN_HIST_CACHE["data"] = (now, history)
    return history


class FundingSentiment:
    """基于资金面的情绪维度。"""

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def _get_margin_data(self) -> dict[str, Any]:
        """获取两融数据+历史分位。"""
        result: dict[str, Any] = {}
        try:
            # Q11-fix: fetch_margin_detail 不存在（margin.py 无此函数）→ ImportError
            # 静默吞掉后整个两融维度恒 score=0。只导入真实存在的函数。
            from quant_system.margin import fetch_margin_summary

            summary = fetch_margin_summary()
            if summary:
                total = summary.get("total_margin_balance", 0)
                change = summary.get("total_margin_change", 0)
                result["margin_yi"] = total
                result["margin_change_yi"] = change

                # P1-Q11-fix(H05): 原 margin_percentile 硬编码假设两融在
                # 1.5~2.5 万亿间波动（注释自认"模拟历史分位"），与真实历史
                # 无关，pct>90/<10 预警分支永远按假设区间触发。改用
                # margin.py 同源历史余额序列真实计算当前余额分位；SSE 缺失
                # 时回退用全市场总量对比；序列不可用时降级为中性 50 并标注。
                ss_balance = summary.get("ss_margin_balance", 0)
                if ss_balance > 0:
                    current = ss_balance
                    basis = "SSE融资余额历史分位"
                else:
                    current = total
                    basis = "全市场余额(SSE缺失回退total)相对SSE历史分位"
                history = _get_margin_balance_history()
                if history and current > 0:
                    arr = np.array(history)
                    less = float(np.sum(arr < current))
                    eq = float(np.sum(arr == current))
                    result["margin_percentile"] = round(
                        (less + 0.5 * eq) / len(arr) * 100, 1)
                    result["margin_percentile_basis"] = basis
                else:
                    result["margin_percentile"] = 50.0
                    result["margin_percentile_note"] = (
                        "两融历史分位不可用(历史序列为空或当前余额无效),按中性50处理")
                result["margin_direction"] = "add" if change > 50 else (
                    "reduce" if change < -50 else "stable"
                )
                if not summary.get("ok"):
                    result["margin_note"] = summary.get("error") or "两融数据获取失败"
        except Exception as e:
            logging.getLogger(__name__).error(f"[funding_sentiment] 操作失败: {e}", exc_info=True)
        return result

    def _get_north_data(self) -> dict[str, Any]:
        """获取北向资金数据。"""
        result: dict[str, Any] = {}
        try:
            # Q11-fix: fetch_north_detail 不存在（north_flow.py 无此函数）→ ImportError
            # 静默吞掉后北向维度恒 score=0。只导入真实存在的函数。
            from quant_system.north_flow import fetch_north_summary

            summary = fetch_north_summary()
            if summary:
                if "error" in summary:
                    # P1-Q11-fix(H05): 北向接口失败必须可见，不静默降级
                    result["north_note"] = summary.get("error") or "北向数据获取失败"
                else:
                    net = summary.get("total_net_yi", 0)
                    result["north_net_yi"] = net
                    result["north_direction"] = "inflow" if net > 10 else (
                        "outflow" if net < -10 else "neutral"
                    )

                    # P2-Q11-fix(M050): north_3d_net 原为单日净流入近似却命名 3日累计，
                    # 改为显式单日披露字段，避免调用方误读。
                    result["north_net_yi"] = round(net, 1)
                    result["north_period"] = "current_disclosed_day"
        except Exception as e:
            logging.getLogger(__name__).error(f"[funding_sentiment] 操作失败: {e}", exc_info=True)
        return result

    def compute(self) -> dict[str, Any]:
        """计算当前资金情绪。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "score": 0,
            "direction": "neutral",
            "percentile": 50.0,
            "approx_percentile": 50.0,
            "sub_signals": [],
        }

        score = 0.0

        # ── 两融情绪 ──
        margin = self._get_margin_data()
        if margin.get("margin_yi") is not None:  # P2-Q11-fix(L055): 0 余额也应进入两融子维度
            result["margin"] = margin
            direction = margin.get("margin_direction", "stable")
            if direction == "add":
                score += 2
                result["sub_signals"].append(
                    f"两融加仓(余额{margin['margin_yi']:.0f}亿, "
                    f"变动{margin['margin_change_yi']:+.0f}亿)"
                )
            elif direction == "reduce":
                score -= 2
                result["sub_signals"].append(
                    f"两融减仓(余额{margin['margin_yi']:.0f}亿, "
                    f"变动{margin['margin_change_yi']:+.0f}亿)"
                )
            # 极端分位预警
            pct = margin.get("margin_percentile", 50)
            if pct > 90:
                score -= 1
                result["sub_signals"].append(
                    f"两融分位{pct:.0f}% — 杠杆过高预警"
                )
            elif pct < 10:
                score += 1
                result["sub_signals"].append(
                    f"两融分位{pct:.0f}% — 杠杆低位,有上行空间"
                )
        elif margin.get("margin_note"):
            # P1-Q11-fix(H05): 两融数据失败必须可见，不静默降级
            result["sub_signals"].append(f"⚠️ 两融数据不可用: {margin['margin_note']}")

        # ── 北向资金情绪 ──
        north = self._get_north_data()
        if north.get("north_net_yi") is not None:
            result["north"] = north
            direction = north.get("north_direction", "neutral")
            net = north["north_net_yi"]
            if direction == "inflow":
                score += 2 if abs(net) > 50 else 1
                result["sub_signals"].append(
                    f"北向净流入{net:+.0f}亿"
                )
            elif direction == "outflow":
                score -= 2 if abs(net) > 50 else 1
                result["sub_signals"].append(
                    f"北向净流出{net:+.0f}亿"
                )
            # 连续方向判断（简化）
            if abs(net) > 80:
                score += 1 if net > 0 else -1
                result["sub_signals"].append(
                    f"北向大幅{'流入' if net > 0 else '流出'}({abs(net):.0f}亿)"
                )
        elif north.get("north_note"):
            # P1-Q11-fix(H05): 北向数据失败必须可见，不静默降级
            result["sub_signals"].append(f"⚠️ 北向数据不可用: {north['north_note']}")

        # 综合
        result["score"] = round(max(-10, min(10, score)), 1)

        if result["score"] >= 2:
            result["direction"] = "bullish"
        elif result["score"] <= -2:
            result["direction"] = "bearish"
        else:
            result["direction"] = "neutral"

        # P1-Q11-fix(H04): percentile 为模型近似（正态映射），非真实历史分位，
        # 明确改名为 approx_percentile 并保留 percentile 键兼容调用方。
        approx = round(max(0, min(100, 50 + result["score"] * 6.8)), 1)
        result["approx_percentile"] = approx
        result["percentile"] = approx

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    fs = FundingSentiment()
    result = fs.compute()
    print("═" * 50)
    print(f"  资金情绪评分: {result.get('score', 'N/A'):>+5}  |  "
          f"方向: {result.get('direction', 'N/A'):>8}  |  "
          f"分位: {result.get('percentile', 'N/A')}")
    print("═" * 50)
    for s in result.get("sub_signals", []):
        print(f"  • {s}")
    if "margin" in result:
        m = result["margin"]
        print(f"  两融: {m.get('margin_yi', 'N/A')}亿  "
              f"变动{m.get('margin_change_yi', 'N/A'):+}")
    if "north" in result:
        n = result["north"]
        print(f"  北向: {n.get('north_net_yi', 'N/A'):+}亿")


if __name__ == "__main__":
    main()
