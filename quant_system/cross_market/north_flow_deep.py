"""
north_flow_deep — 北向资金深度分析 (V5)

比简单"净流入/净流出"更深入：
  1. 当日/5日/20日/年内累计净流入
  2. 年内累计的历史分位
  3. 行业偏好（哪些行业被增持最多）—— 无现成数据源，返回空 + 说明
  4. 个股偏好（哪些个股被增持最多）

数据源: akshare.stock_hsgt_hist_em("北向资金") + stock_hsgt_fund_flow_summary_em()

Q23-fix（2026-08-02，V6 P0 修复 B 组）：
  - CRITICAL: 原调 ak.stock_hsgt_north_flow_em()/stock_hsgt_industry_em() 在
    akshare 1.18.64 均不存在（AttributeError 被静默吞掉）→ 模块恒返回全零
    默认值且 error 为空，对外表现"正常"。现改调现存接口：
      * stock_hsgt_hist_em("北向资金")：历史序列，含 日期/当日成交净买额/
        买入成交额/卖出成交额；2024-08-19 后净买额字段为 NaN（披露规则变更）。
      * stock_hsgt_fund_flow_summary_em()：沪深港通日度汇总（北向净买额=0，
        南向仍有值）。
    - 净买入类指标在 2024-08-19 后显式标注 net_buy_disclosed=False，
      口径改为成交额（买入+卖出，仍披露）。
  - MEDIUM: ytd_net 原为全历史求和（无年份过滤）→ 按当前年份过滤；
    "历史分位"原为硬编码均值2000亿/标准差800亿的 logistic 假分位
    → 改按历史年度 YTD 的真实分位，样本不足返回 None。
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone

import pandas as pd  # V11 审计修复（Medium）: 原缺 import，南向分支 pd.to_numeric NameError
from pathlib import Path
from typing import Any

import numpy as np

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent

# 北向净买入披露规则变更日
DISCLOSURE_CUTOFF = "2024-08-19"


class NorthFlowDeep:
    """北向资金深度分析。"""

    def __init__(self, cache_ttl: int = 600) -> None:
        self.cache: dict[str, Any] = {}
        self.last_fetch: float = 0
        self.cache_ttl = cache_ttl

    def compute(self) -> dict[str, Any]:
        """计算北向资金深度指标。"""
        now = _time.time()
        if now - self.last_fetch < self.cache_ttl and self.cache:
            return self.cache

        result: dict[str, Any] = {
            "timestamp": datetime.now(CST).isoformat(),
            "daily_net": None,
            "5d_net": None,
            "20d_net": None,
            "ytd_net": None,
            "ytd_percentile": None,
            "turnover_yi": None,          # 当日成交额（2024-08-19 后仍披露）
            "southbound_net_yi": None,    # 南向净买额（仍披露）
            "net_buy_disclosed": True,    # 2024-08-19 后为 False
            "industry_preference": [],
            "top_stocks": [],
            "direction": "数据不足",
            "signal": "",
            "ok": False,
            "notes": [],
        }

        errors: list[str] = []
        try:
            import akshare as ak

            # ── 北向历史序列（stock_hsgt_hist_em 存在，替代已不存在的
            #    stock_hsgt_north_flow_em）──
            hist: Any = None
            try:
                hist = ak.stock_hsgt_hist_em(symbol="北向资金")
            except Exception as e:
                errors.append(f"北向历史序列获取失败: {repr(e)[:80]}")

            if hist is not None and not hist.empty:
                # 列：日期/当日成交净买额/买入成交额/卖出成交额/历史累计净买额...
                date_col = next((c for c in hist.columns if "日期" in str(c)), None)
                net_col = next((c for c in hist.columns if "当日成交净买额" in str(c)), None)
                buy_col = next((c for c in hist.columns if "买入成交额" in str(c)), None)
                sell_col = next((c for c in hist.columns if "卖出成交额" in str(c)), None)

                if date_col:
                    hist = hist.copy()
                    hist["_date"] = hist[date_col].astype(str)
                    # 归一化日期为 YYYY-MM-DD
                    hist["_date"] = hist["_date"].str[:10]

                def _series_float(col: str | None) -> list[float]:
                    if col is None or col not in hist.columns:
                        return []
                    return [float(v) for v in hist[col].dropna().tolist()]

                net_series = _series_float(net_col)
                buy_series = _series_float(buy_col)
                sell_series = _series_float(sell_col)

                # 规则变更判定：最新数据日期 >= 2024-08-19 且该日净买额为 NaN/0
                # （注意：不能用 dropna 后的序列判定——NaN 被剔除后序列尾值会
                # 落在披露期最后一天，导致误判为"仍披露"）
                latest_date = ""
                last_raw_net = float("nan")
                if date_col and len(hist) > 0:
                    latest_date = str(hist["_date"].iloc[-1])[:10]
                    if net_col is not None:
                        try:
                            last_raw_net = float(hist[net_col].iloc[-1])
                        except (ValueError, TypeError):
                            last_raw_net = float("nan")
                disclosed = True
                if latest_date >= DISCLOSURE_CUTOFF and (
                        last_raw_net != last_raw_net or last_raw_net == 0.0):
                    disclosed = False
                result["net_buy_disclosed"] = disclosed

                # 当日净买额（变更后为 None，不冒充信号）
                if disclosed and net_series:
                    result["daily_net"] = round(float(net_series[-1]), 1)
                elif not disclosed:
                    result["notes"].append(
                        "2024-08-19起北向净买入停止披露,净买额指标下线,改用成交额口径")

                # 5日/20日累计（仅披露期数据；变更后净买额口径下线）
                if disclosed and net_series:
                    result["5d_net"] = round(float(sum(net_series[-5:])), 1)
                    result["20d_net"] = round(float(sum(net_series[-20:])), 1)
                if buy_series and sell_series:
                    n = min(len(buy_series), len(sell_series))
                    if n > 0:
                        result["turnover_yi"] = round(
                            float(buy_series[-1] + sell_series[-1]), 1)

                # 年内累计（按当前年份过滤，Q23-fix）
                current_year = datetime.now(CST).year
                if date_col and net_series and len(hist) == len(net_series):
                    ytd_vals = [
                        net_series[i] for i in range(len(hist))
                        if str(hist["_date"].iloc[i]).startswith(str(current_year))
                        and net_series[i] == net_series[i]  # 剔除 NaN
                    ]
                    if ytd_vals:
                        result["ytd_net"] = round(float(sum(ytd_vals)), 1)

                # 年内分位（真实历史分位：当前年 YTD 与历史各年 YTD 比，Q23-fix）
                if date_col and net_series and len(hist) == len(net_series):
                    year_ytd: dict[str, float] = {}
                    for i in range(len(hist)):
                        ds = str(hist["_date"].iloc[i])
                        yr = ds[:4]
                        v = net_series[i]
                        if v != v:  # NaN
                            continue
                        year_ytd[yr] = year_ytd.get(yr, 0.0) + v
                    if str(current_year) in year_ytd:
                        cur = year_ytd[str(current_year)]
                        prev_ytd = [v for k, v in year_ytd.items()
                                    if k != str(current_year) and k >= "2017"]
                        if len(prev_ytd) >= 3:
                            pct = float((np.asarray(prev_ytd) <= cur).mean()) * 100
                            result["ytd_percentile"] = round(max(0, min(100, pct)), 1)

            # ── 南向资金（stock_hsgt_fund_flow_summary_em，南向净买额仍披露）──
            try:
                summary_df = ak.stock_hsgt_fund_flow_summary_em()
                if summary_df is not None and not summary_df.empty:
                    south = summary_df[
                        summary_df["板块"].astype(str).str.contains("港股通", na=False)
                    ] if "板块" in summary_df.columns else None
                    if south is not None and not south.empty:
                        if "成交净买额" in south.columns:
                            # 港股通(沪)+港股通(深) 合计（南向净买额仍披露，
                            # 该接口成交净买额单位为亿元）
                            s_vals = pd.to_numeric(south["成交净买额"], errors="coerce").dropna()
                            if not s_vals.empty:
                                result["southbound_net_yi"] = round(
                                    float(s_vals.sum()), 1)
            except Exception as e:
                errors.append(f"沪深港通汇总获取失败: {repr(e)[:80]}")

            # ── 行业偏好（原 stock_hsgt_industry_em 不存在，无替代数据源）──
            result["industry_preference"] = []
            result["notes"].append(
                "行业资金流无现成数据源(原stock_hsgt_industry_em在akshare 1.18.64不存在),行业偏好维度下线")

            # ── 方向判断（仅在净买额披露时）──
            daily = result.get("daily_net")
            if daily is not None and result.get("net_buy_disclosed"):
                if daily > 30:
                    result["direction"] = "大幅流入"
                elif daily > 10:
                    result["direction"] = "流入"
                elif daily < -30:
                    result["direction"] = "大幅流出"
                elif daily < -10:
                    result["direction"] = "流出"
                else:
                    result["direction"] = "中性"
            elif not result.get("net_buy_disclosed"):
                result["direction"] = "净买额未披露(2024-08-19起)"

            # ── 综合信号（仅真实数据）──
            if daily is not None and result.get("net_buy_disclosed"):
                if daily > 50:
                    result["signal"] = f"北向大幅净流入{daily:.0f}亿 — 外资积极加仓A股"
                elif daily < -50:
                    result["signal"] = f"北向大幅净流出{abs(daily):.0f}亿 — 外资撤离"
                elif result.get("ytd_percentile") is not None and result["ytd_percentile"] > 70:
                    result["signal"] = "北向年内累计处于高位 — 外资整体看好"
                else:
                    result["signal"] = "北向资金流动中性"
            elif not result.get("net_buy_disclosed"):
                result["signal"] = "北向净买入2024-08-19起停止披露,净买额信号不可用"
            else:
                result["signal"] = "北向数据不足"

            result["ok"] = (result["daily_net"] is not None
                            or result["turnover_yi"] is not None
                            or result["southbound_net_yi"] is not None)

        except Exception as e:
            errors.append(repr(e)[:200])

        if errors:
            result["error"] = "; ".join(errors)
        result["data_errors"] = errors

        self.cache = result
        self.last_fetch = now
        return result


def main() -> None:
    nf = NorthFlowDeep()
    r = nf.compute()
    print("═" * 55)
    print("  北向资金深度")
    print("═" * 55)
    daily = r.get("daily_net")
    print(f"  当日净流: {daily if daily is not None else '未披露/N/A'}")
    print(f"  5日累计:  {r.get('5d_net', 'N/A')}")
    print(f"  20日累计: {r.get('20d_net', 'N/A')}")
    print(f"  年内累计: {r.get('ytd_net', 'N/A')}")
    pct = r.get("ytd_percentile")
    print(f"  年内分位: {pct if pct is not None else '样本不足/N/A'}%")
    print(f"  当日成交额: {r.get('turnover_yi', 'N/A')}亿")
    print(f"  南向净买: {r.get('southbound_net_yi', 'N/A')}亿")
    print(f"  方向: {r.get('direction', 'N/A')}")
    print(f"  信号: {r.get('signal', 'N/A')}")
    for note in r.get("notes", []):
        print(f"  ℹ️ {note}")
    if r.get("error"):
        print(f"  ⚠️ {r['error']}")


if __name__ == "__main__":
    main()
