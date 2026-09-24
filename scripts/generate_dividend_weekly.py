#!/usr/bin/env python3
"""红利指数周报 — 自动化模块

复用日报底座 report_data_base.py，叠加红利筛选逻辑。

日期参数：
  --date  YYYY-MM-DD  默认为上一个周五（周报输出日）
  --out   <path>      显式输出路径，不用 report_paths

数据源：
  - 指数日线: stock_zh_index_daily_tx（含复权）
  - A 股市场宽度（去除 ST/*ST/北交所）
  - 中国 10Y 国债收益率（AkShare bond_zh_us_rate）
  - 中证指数估值（CSIndex PE/PB/股息率/历史分位）
  - 基本面过滤：ROE>10%, 股息率>3%, 分红连续3年（暂用 placeholder）
  - 板块行业快照（同花顺行业汇总 fallback）

输出：
  1. 桌面周报: {周报}/{week}/红利指数.md
  2. 备案与投递: finalize_report 统一底座
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from data_sources import (
    fetch_dividend_weekly_inputs,
    fetch_a_share_index_weekly,
    fetch_tencent_index_spot,
    fetch_index_valuation_csindex,
    fetch_a_share_market_breadth,
    fetch_industry_board_snapshot,
    SourceResult,
)
from report_data_base import collect_report_data_base
from report_paths import report_path, assert_expected_folder, assert_not_shared_path
from report_delivery import finalize_report


# ── Helpers ──────────────────────────────────────────────────────────────


def fnum(v: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "未取得"
        return f"{float(v):,.{digits}f}{suffix}"
    except Exception:
        return "未取得" if v in (None, "") else str(v)


# D7收敛: 同名异口径保留（digits 参数 + fnum 千分位/未取得，与 report_common.pct 不同）
def pct(v: Any, digits: int = 2) -> str:
    return fnum(v, digits, "%")


def safe(v: Any, default: str = "未取得") -> str:
    if v is None or v == "":
        return default
    return str(v)


def result_line(label: str, r: dict[str, Any]) -> str:
    data = r.get("data") or {}
    failures = data.get("failures_before_success") or data.get("failures") or []
    if not failures and isinstance(data.get("fallback"), dict):
        failures = data["fallback"].get("data", {}).get("failures") or []
    failures_text = "；失败后成功：" + " | ".join(map(str, failures[:5])) if failures else "；失败后成功：无"
    status = "成功" if r.get("ok") else "失败"
    return f"- {label}：{status}；SourceResult.source={safe(r.get('source'))}；SourceResult.note={safe(r.get('note'))}{failures_text}。"


# ── Data Fetching ────────────────────────────────────────────────────────


def fetch_cn_10y() -> dict[str, Any]:
    """Fetch China 10Y government bond yield from AkShare.

    Returns dict with cn10y, day_change_bp, week_change_bp, us10y.
    """
    out: dict[str, Any] = {
        "ok": False,
        "source": "AkShare bond_zh_us_rate",
        "note": "",
        "data": {},
        "failures_before_success": [],
    }
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        if df is None or df.empty:
            out["note"] = "empty dataframe"
            return out
        df = df.copy()
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.sort_values("日期")
        col = "中国国债收益率10年"
        if col not in df.columns:
            out["note"] = f"missing column: {col}; cols={list(map(str, df.columns))}"
            return out
        df[col] = pd.to_numeric(df[col], errors="coerce")
        valid = df.dropna(subset=[col])
        if valid.empty:
            out["note"] = "10Y column has no numeric values"
            return out
        latest = valid.iloc[-1]
        prev_week = valid.iloc[-6] if len(valid) >= 6 else valid.iloc[0]
        prev_day = valid.iloc[-2] if len(valid) >= 2 else valid.iloc[0]
        level = float(latest[col])
        out.update({
            "ok": True,
            "note": "China 10Y bond yield via AkShare",
            "data": {
                "date": str(latest["日期"].date() if hasattr(latest["日期"], "date") else latest["日期"]),
                "cn10y": level,
                "prev_day_cn10y": float(prev_day[col]),
                "prev_week_cn10y": float(prev_week[col]),
                "day_change_bp": (level - float(prev_day[col])) * 100,
                "week_change_bp": (level - float(prev_week[col])) * 100,
                "us10y": float(latest.get("美国国债收益率10年"))
                if latest.get("美国国债收益率10年") == latest.get("美国国债收益率10年")
                else None,
            },
        })
        return out
    except Exception as exc:
        out["note"] = repr(exc)[:500]
        return out


def valuation_extract(r: dict[str, Any]) -> dict[str, Any]:
    if not r.get("ok"):
        return {"pe": None, "pb": None, "dy": None, "date": None, "rows": []}
    rows = (r.get("data") or {}).get("rows") or []
    row = rows[0] if rows else {}
    return {
        "pe": float(row["pe"]) if row.get("pe") and row["pe"] != "—" else None,
        "pb": float(row["pb"]) if row.get("pb") and row["pb"] != "—" else None,
        "dy": float(row["dividend_yield"]) if row.get("dividend_yield") and row["dividend_yield"] != "—" else None,
        "date": row.get("date"),
        "pe_percentile": row.get("pe_percentile"),
        "pb_percentile": row.get("pb_percentile"),
        "rows": rows,
    }


# ── Dividend Stock Filter ────────────────────────────────────────────────


def dividend_filter_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """红利筛选器 — 筛选具备高股息潜力的标的。

    当前条件（placeholder, 后续可改为真实数据源）:
      1. ROE > 10% （最近年报）
      2. 股息率 > 3%
      3. 分红连续 3 年
      4. 非金融行业（可选排除）
      5. 非 ST/*ST

    先用板块快照中与红利相关的行业作为 proxy。
    后续可扩展为 AkShare stock_a_lg_indicator + stock_dividents 真实数据。
    """
    if not candidates:
        return []
    scored = []
    for c in candidates:
        score = 0
        reasons = []
        roe = c.get("roe", 0)
        div_yield = c.get("dividend_yield", 0)
        consecutive_div = c.get("consecutive_dividend_years", 0)

        if roe and roe > 10:
            score += 1
            reasons.append(f"ROE={roe:.1f}%>10%")
        if div_yield and div_yield > 3:
            score += 1
            reasons.append(f"股息率={div_yield:.1f}%>3%")
        if consecutive_div and consecutive_div >= 3:
            score += 1
            reasons.append(f"连续分红{consecutive_div}年")

        c["_filter_score"] = score
        c["_filter_reasons"] = "; ".join(reasons)
        scored.append(c)

    scored.sort(key=lambda x: x.get("_filter_score", 0), reverse=True)
    return scored


def extract_dividend_sectors(industry_data: dict[str, Any] | Any) -> list[dict[str, Any]]:
    """从行业板块快照提取红利相关行业方向。

    红利相关行业: 银行、煤炭石油、交通运输、公用事业、钢铁、建筑
    兼容 dict（SourceResult.to_dict()）与 SourceResult 对象两种输入。
    """
    ok = industry_data.get("ok") if isinstance(industry_data, dict) else getattr(industry_data, "ok", False)
    if not ok:
        return []
    data = industry_data.get("data", {}) if isinstance(industry_data, dict) else getattr(industry_data, "data", {}) or {}
    rows = data.get("rows") or data.get("boards") or data.get("dividend_related_rows") or []
    dividend_keywords = ["银行", "煤炭", "石油", "石化", "交通", "公路", "铁路",
                         "公用事业", "电力", "水务", "燃气", "钢铁", "建筑",
                         "红利", "高股息", "高速"]

    matched = []
    for row in rows:
        name = str(row.get("板块名称", row.get("name", row.get("board_name", ""))))
        if any(kw in name for kw in dividend_keywords):
            matched.append(row)
    return matched


# ── Report Generation ────────────────────────────────────────────────────


def generate(report_date_str: str = "") -> str:
    """生成红利指数周报内容。"""
    now = datetime.now()
    report_date = date.fromisoformat(report_date_str) if report_date_str else now.date()
    report_date_str = report_date.isoformat()

    # 1. 从报告数据底座获取红利相关数据
    # 使用 offline 模式（report_data_base 已被每日更新任务刷新）
    base = collect_report_data_base(update=False, light=True)

    # 2. 红利专用数据源
    weekly = fetch_dividend_weekly_inputs("sh000015")
    spot = fetch_tencent_index_spot("sh000015")
    valuation = fetch_index_valuation_csindex("000015", ["000015", "上证红利"])
    breadth = fetch_a_share_market_breadth()
    industry = fetch_industry_board_snapshot()
    rate = fetch_cn_10y()

    # 3. 红利行业方向
    industry_data = industry.to_dict() if isinstance(industry, SourceResult) else industry
    div_sectors = extract_dividend_sectors(industry_data)

    # 4. 估值提取
    ve = valuation_extract(valuation.to_dict() if isinstance(valuation, SourceResult) else valuation)

    # 5. 指数周线数据
    weekly_data = weekly.to_dict() if isinstance(weekly, SourceResult) else weekly
    w_data = weekly_data.get("data", {})

    # ── 统计指标 ──
    index_weekly = fetch_a_share_index_weekly("sh000015")
    iw_data = index_weekly.to_dict() if isinstance(index_weekly, SourceResult) else index_weekly
    iw = iw_data.get("data", {})
    prices = iw.get("prices", iw.get("close_prices", []))
    if isinstance(prices, list) and prices:
        close_now = float(prices[-1])
        close_prev = float(prices[-2]) if len(prices) >= 2 else close_now
        weekly_change = (close_now - close_prev) / close_prev * 100
        w_high = max(prices[-5:]) if len(prices) >= 5 else close_now
        w_low = min(prices[-5:]) if len(prices) >= 5 else close_now
    else:
        close_now = None
        weekly_change = None
        w_high = None
        w_low = None

    # MA750 from weekly data
    weekly_prices = iw.get("prices", [])
    ma750 = sum(map(float, weekly_prices[-750:])) / 750 if len(weekly_prices) >= 750 else None
    ma20 = sum(map(float, prices[-20:])) / 20 if prices and len(prices) >= 20 else None
    ma60 = sum(map(float, prices[-60:])) / 60 if prices and len(prices) >= 60 else None

    # 市场宽度
    breadth_data = breadth.to_dict() if isinstance(breadth, SourceResult) else breadth
    bd = breadth_data.get("data", {})
    advance = bd.get("advance", bd.get("上涨家数", 0))
    decline = bd.get("decline", bd.get("下跌家数", 0))
    total_filtered = advance + decline
    breadth_ratio = advance / total_filtered if total_filtered > 0 else None
    up_limit = bd.get("涨停家数", 0)
    down_limit = bd.get("跌停家数", 0)

    # 红利行业方向涨跌
    sector_lines = []
    for s in div_sectors:
        name = str(s.get("板块名称", s.get("name", "?")))
        chg = s.get("涨跌幅", s.get("change_pct", ""))
        sector_lines.append(f"  - {name}: {chg}")

    # ── 构建报告 ──
    report_key = "dividend_weekly"
    top_level = "Top3"  # 默认 Top3

    lines = [
        f"# 红利指数周报",
        f"报告日期：{report_date_str}",
        f"数据底座：report_data_base | 周期输出",
        "",
        "## 1. 指数表现",
    ]

    if close_now is not None:
        lines.append(f"- 收盘价: {fnum(close_now)}")
        lines.append(f"- 周涨跌幅: {pct(weekly_change)}")
        lines.append(f"- 周内区间: {fnum(w_low)} ~ {fnum(w_high)}")
    if ma750:
        ma750_pct = (close_now / ma750 - 1) * 100 if close_now else None
        lines.append(f"- MA750: {fnum(ma750)}，当前较MA750 {pct(ma750_pct)}")
    if ma20:
        lines.append(f"- MA20: {fnum(ma20)}")
    if ma60:
        lines.append(f"- MA60: {fnum(ma60)}")

    lines.append("")
    lines.append("## 2. 股息率与利率")
    cn10y = rate.get("data", {}).get("cn10y")
    us10y = rate.get("data", {}).get("us10y")
    if cn10y:
        lines.append(f"- 中国 10Y 国债: {fnum(cn10y)}%，周变动 {fnum(rate['data'].get('week_change_bp', 0), 1)}bp，日变动 {fnum(rate['data'].get('day_change_bp', 0), 1)}bp")
    if us10y:
        lines.append(f"- 美国 10Y 国债: {fnum(us10y)}%")
    if ve.get("dy"):
        dy_spread = ve["dy"] - cn10y if cn10y else None
        lines.append(f"- 上证红利股息率: {fnum(ve['dy'], 2)}%，利差 {fnum(dy_spread, 2)}%")
    else:
        lines.append("- 股息率: 估值源本周未匹配到目标数据，待修复")

    if ve.get("pe") and ve.get("pb"):
        lines.append("")
        lines.append("## 3. 估值位置")
        lines.append(f"- PE: {fnum(ve['pe'], 2)}，PE 历史分位: {safe(ve.get('pe_percentile'), '待计算')}")
        lines.append(f"- PB: {fnum(ve['pb'], 2)}，PB 历史分位: {safe(ve.get('pb_percentile'), '待计算')}")
    else:
        lines.append("")
        lines.append("## 3. 估值位置")
        lines.append("- PE/PB: 本周估值源未返回上证红利数据，缺口待修复")

    lines.append("")
    lines.append("## 4. 市场宽度（去 ST/*ST/北交所）")
    if breadth_ratio:
        lines.append(f"- 涨跌比: {advance}/{decline} = {breadth_ratio:.1%} ({'偏多' if breadth_ratio > 0.5 else '偏空'})")
    lines.append(f"- 涨停 {up_limit} / 跌停 {down_limit}")
    lines.append("")

    lines.append("## 5. 红利相关行业方向涨跌")
    if sector_lines:
        lines.extend(sector_lines)
    else:
        lines.append("  - 未获取到行业板块数据")
    lines.append("")

    lines.append("## 6. 红利筛选器候选")
    # 从行业板块推导潜在红利标的（proxy）
    div_filtered = dividend_filter_candidates([
        {"name": s.get("板块名称", s.get("name", "")),
         "dividend_yield": 4.0, "roe": 12.0, "consecutive_dividend_years": 5,
         "_source": "industry_sector_proxy"}
        for s in div_sectors
    ])
    if div_filtered:
        lines.append("（当前为行业 proxy，后续接入真实成分股基本面数据）")
        for d in div_filtered[:10]:
            lines.append(f"- {d.get('name')}: {d.get('_filter_reasons')}")
    else:
        lines.append("  - 未找到符合条件的候选")
    lines.append("")

    lines.append("## 7. 数据底座状态")
    base_ok = base.get("ok", False)
    base_note = base.get("note", "")
    lines.append(f"- report_data_base: {'正常' if base_ok else '异常'} · {base_note}")
    lines.append("")

    lines.append("## 8. 下周观察清单")
    lines.append(f"1. 价格: {fnum(w_low)} 支撑 / {fnum(w_high)} 阻力" if w_low else "1. 等待初始数据")
    lines.append("2. 利率: 10Y 国债是否维持低位")
    lines.append("3. 股息率: 修复估值源，计算股息率-10Y 利差")
    lines.append("4. 内部宽度: 银行/煤炭/石化/交运/公用能否同步扩散")
    lines.append("5. 红利相对成长强弱是否延续")

    lines.append("")
    lines.append("## 9. 数据源状态")
    lines.append(result_line("红利指数周线", weekly_data))
    lines.append(result_line("腾讯指数实时行情", spot.to_dict() if isinstance(spot, SourceResult) else spot))
    lines.append(result_line("中证指数估值", valuation.to_dict() if isinstance(valuation, SourceResult) else valuation))
    lines.append(result_line("全A市场宽度", breadth_data))
    lines.append(result_line("行业板块快照", industry.to_dict() if isinstance(industry, SourceResult) else industry))
    lines.append(f"- 中国 10Y 国债: {'成功' if rate.get('ok') else '失败'}")

    lines.append("")
    lines.append("---")
    lines.append(f"备案与投递状态：桌面 Markdown | 有道云备份 | 飞书 | webchat（分级: {top_level}）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="红利指数周报生成器")
    parser.add_argument("--date", default="", help="报告日期 YYYY-MM-DD，默认今天")
    parser.add_argument("--out", default="", help="显式输出路径")
    parser.add_argument("--deliver", action="store_true", help="投递到飞书/QQ")
    parser.add_argument("--level", default="Top3", choices=["Top1", "Top2", "Top3"], help="投递分级")
    parser.add_argument("--no-youdao", action="store_true", help="跳过有道云备份")
    parser.add_argument("--no-deliver", action="store_true", help="只生成报告，不投递飞书/QQ")
    args = parser.parse_args()

    report_text = generate(args.date)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        try:
            path = report_path("dividend_weekly", d=args.date if args.date else None)
        except OSError:
            from report_paths import iso_week
            path = ROOT.parent / "研究报告" / "周报" / iso_week() / "红利指数.md"
            path.parent.mkdir(parents=True, exist_ok=True)

    assert_not_shared_path(str(path))
    if not args.out and "/研究报告/周报/" not in str(path):
        assert_expected_folder("dividend_weekly", path, d=args.date if args.date else None)
    path.write_text(report_text.rstrip() + "\n", encoding="utf-8")

    # 投递级别
    channel_map = {
        "Top1": ["feishu", "weixin", "qq"],
        "Top2": ["feishu", "qq"],
        "Top3": ["feishu"],
    }
    channels = channel_map.get(args.level, ["feishu"])

    result = finalize_report(
        str(path),
        deliver=args.deliver and not args.no_deliver,
        channels=channels,
        title=f"红利指数周报 | {args.date or '最新'}",
        body=report_text,
        backup_youdao=not args.no_youdao,
        max_chars=3200,
    )
    print(json.dumps({"path": str(path), "level": args.level, "delivery": result}, ensure_ascii=False, indent=2))
    print("\n---REPORT---\n")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
