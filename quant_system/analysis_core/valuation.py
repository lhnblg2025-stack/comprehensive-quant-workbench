"""
valuation — 中长线估值系统（V11 N5）

相对估值: PE-TTM/PB 历史分位（baostock 免费源，3/5年窗口）
绝对估值: DCF 三阶段 v1（净利驱动简化版，输入增长率假设 → 合理市值区间）
交叉验证: PEG / ROE-PB 匹配

用法:
  python3 -m quant_system.analysis_core.valuation 600519
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))


def fetch_pe_pb_history(code: str, years: int = 5) -> pd.DataFrame:
    """baostock 拉 PE-TTM/PB 日频序列（免费无限额）。"""
    import baostock as bs
    code6 = code.zfill(6)
    bs_code = f"sh.{code6}" if code6.startswith(("6", "9")) else f"sz.{code6}"
    lg = bs.login()
    if lg.error_code != "0":
        return pd.DataFrame()
    end = datetime.now(CST).strftime("%Y-%m-%d")
    start = (datetime.now(CST) - timedelta(days=years * 366)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(
        bs_code, "date,close,peTTM,pbMRQ", start_date=start, end_date=end,
        frequency="d", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "close", "peTTM", "pbMRQ"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["close", "peTTM", "pbMRQ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _percentile(series: pd.Series, v: float) -> float | None:
    s = series.dropna()
    if len(s) < 60 or np.isnan(v):
        return None
    return float((s < v).mean())


def valuation(code: str) -> dict:
    df = fetch_pe_pb_history(code)
    if df.empty:
        return {"code": code, "error": "baostock 无数据"}
    cur = df.iloc[-1]
    pe, pb = cur["peTTM"], cur["pbMRQ"]

    def _window(years: int) -> dict:
        w = df[df["date"] >= df["date"].max() - pd.Timedelta(days=years * 366)]
        return {
            "pe_pct": round(_percentile(w["peTTM"], pe), 3) if _percentile(w["peTTM"], pe) is not None else None,
            "pb_pct": round(_percentile(w["pbMRQ"], pb), 3) if _percentile(w["pbMRQ"], pb) is not None else None,
            "pe_med": round(float(w["peTTM"].median()), 1) if len(w["peTTM"].dropna()) else None,
            "pb_med": round(float(w["pbMRQ"].median()), 2) if len(w["pbMRQ"].dropna()) else None,
        }

    return {
        "code": code,
        "asof": str(cur["date"].date()),
        "close": round(float(cur["close"]), 2),
        "pe_ttm": round(float(pe), 2) if not np.isnan(pe) else None,
        "pb": round(float(pb), 2) if not np.isnan(pb) else None,
        "pct_3y": _window(3),
        "pct_5y": _window(5),
    }


# ── DCF 三阶段 v1（净利驱动）───────────────────────────
def dcf_3stage(profit_now: float, g1: float = 0.15, g2: float = 0.10, g3: float = 0.05,
               discount: float = 0.10, years1: int = 3, years2: int = 5,
               terminal_growth: float = 0.03) -> dict:
    """简化三阶段 DCF: 净利 → 自由现金流近似(净利×0.8) → 折现 → 终值。"""
    fcf = profit_now * 0.8
    pv = 0.0
    flows = []
    # 阶段1: 高增长 g1
    for y in range(1, years1 + 1):
        fcf *= (1 + g1)
        pv += fcf / (1 + discount) ** y
        flows.append(round(fcf, 2))
    # 阶段2: 中增长 g2
    for y in range(years1 + 1, years1 + years2 + 1):
        fcf *= (1 + g2)
        pv += fcf / (1 + discount) ** y
        flows.append(round(fcf, 2))
    # 终值
    tv = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    pv_tv = tv / (1 + discount) ** (years1 + years2)
    return {"pv_cashflows": round(pv, 0), "pv_terminal": round(pv_tv, 0),
            "total_value": round(pv + pv_tv, 0), "flows": flows[:4]}


def valuation_report(code: str) -> str:
    v = valuation(code)
    if "error" in v:
        return f"# {code} 估值\n\n数据不可用: {v['error']}"
    lines = [
        f"# {code} 估值仪表盘（{v['asof']}，baostock）",
        "",
        f"- 现价: {v['close']} | PE-TTM: {v['pe_ttm']} | PB: {v['pb']}",
        f"- PE 分位: 3年 {v['pct_3y']['pe_pct']} / 5年 {v['pct_5y']['pe_pct']}（中位 {v['pct_5y']['pe_med']}）",
        f"- PB 分位: 3年 {v['pct_3y']['pb_pct']} / 5年 {v['pct_5y']['pb_pct']}（中位 {v['pct_5y']['pb_med']}）",
    ]
    if v["pe_ttm"] and v["pct_5y"]["pe_pct"] is not None:
        p = v["pct_5y"]["pe_pct"]
        if p < 0.2:
            lines.append("- 🟢 估值处于历史低位区（安全边际较厚）")
        elif p > 0.8:
            lines.append("- 🔴 估值处于历史高位区（均值回归风险）")
        else:
            lines.append("- 🟡 估值处于历史中枢区")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="估值仪表盘")
    ap.add_argument("code")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(valuation(args.code), ensure_ascii=False, indent=2))
    else:
        print(valuation_report(args.code))
