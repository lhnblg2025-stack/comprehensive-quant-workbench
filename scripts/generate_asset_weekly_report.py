#!/usr/bin/env python3
"""
Weekly asset report generator for cron tasks.

Replaces the legacy weekly_report.py with a proper report pipeline integration.
Each task (gold_weekly, muyuan_weekly, etc.) generates a focused markdown report
and uses report_delivery.finalize_report for archiving and delivery.
"""
from __future__ import annotations
import logging

import argparse
import json
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from report_delivery import finalize_report  # noqa: E402
from report_contract import make_report_contract, validate_and_write_contract  # noqa: E402


ASSET_MAPPING = {
    "红利指数": ("1B0015", "sh000015"),
    "牧原股份": ("002714", "sz002714"),
    "紫金矿业": ("601899", "sh601899"),
    "中金黄金": ("600489", "sh600489"),
}

TASK_ALIASES = {
    "gold_weekly": ["黄金"],
    "muyuan_weekly": ["牧原股份"],
    "pig_weekly": ["牧原股份"],
    "china_internet_weekly": ["中概互联"],
    "muyuan_capacity": ["牧原股份"],
    "pig_capacity": ["牧原股份"],
    "zijin_weekly": ["紫金矿业"],
}


# D7收敛: 与 legacy weekly_report.py 同名近口径（本文件为活动版真源，含 len(p)<5 与异常兜底），保留
# get_weekly_ma750 键为 rows（weekly_report 为 weekly_rows），异口径保留。
def fetch_tencent_price(code: str) -> dict | None:
    """Fetch real-time price from Tencent quote API."""
    mapping = {"1B0015": "sh000015", "002714": "sz002714", "601899": "sh601899", "600489": "sh600489"}
    tc = mapping.get(code, code)
    url = f"https://qt.gtimg.cn/q={tc}"
    try:
        r = requests.get(url, timeout=10)
        r.encoding = "gbk"
        m = re.search(r'"([^"]+)"', r.text.strip())
        if not m:
            return None
        p = m.group(1).split("~")
        if len(p) < 5:
            return None
        return {
            "name": p[1],
            "close": float(p[3]) if p[3] else 0,
            "pct": round((float(p[3]) / float(p[4]) - 1) * 100, 2) if p[3] and p[4] else 0,
            "high": float(p[33]) if len(p) > 33 and p[33] else 0,
            "low": float(p[34]) if len(p) > 34 and p[34] else 0,
            "volume": p[6] if len(p) > 6 else "0",
            "quote_date": p[30] if len(p) > 30 and p[30] else None,
            "quote_time": p[31] if len(p) > 31 and p[31] else None,
            "source": "qt.gtimg.cn",
        }
    except Exception as e:
        return None


def get_weekly_ma750() -> tuple:
    """红利指数周K MA750 from data_sources."""
    try:
        from data_sources import fetch_a_share_index_weekly
        res = fetch_a_share_index_weekly("sh000015", ma_windows=[750])
        if not res.ok:
            return None, None, 0
        latest = res.data.get("latest") or {}
        close = latest.get("close")
        ma750 = latest.get("ma750")
        weekly_rows = int(res.data.get("rows") or 0)
        if ma750 is None:
            return None, round(float(close), 2) if close is not None else None, weekly_rows
        return round(float(ma750), 2), round(float(close), 2), weekly_rows
    except Exception:
        return None, None, 0


def _read_market_series(asset: str, limit: int = 60) -> list[dict]:
    """Read real warehouse history for decision context; never fabricate missing values."""
    try:
        import pandas as pd
        path = ROOT / "data_warehouse" / "market" / f"commodity__{asset}.parquet"
        if not path.exists():
            return []
        frame = pd.read_parquet(path).tail(limit)
        out = []
        for _, row in frame.iterrows():
            vals = list(row.values)
            if len(vals) < 2:
                continue
            try:
                out.append({"date": str(vals[0])[:10], "value": float(vals[-1])})
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return []


def _decision_section(title: str, data: dict, asset: str) -> list[str]:
    rows = _read_market_series(asset)
    values = [x["value"] for x in rows]
    lines = [f"## {title}：从数据到行动"]
    if values:
        latest = values[-1]
        def ret(n):
            return (latest / values[-n-1] - 1) * 100 if len(values) > n and values[-n-1] else None
        lines += [f"- 最新观测：{latest:.2f}，数据截至 {rows[-1]['date']}。",
                  f"- 5日变化：{ret(5):+.2f}%" if ret(5) is not None else "- 5日变化：数据不足。",
                  f"- 20日变化：{ret(20):+.2f}%" if ret(20) is not None else "- 20日变化：数据不足。"]
        ma20 = sum(values[-20:]) / min(20, len(values))
        lines.append(f"- 趋势状态：{'站上20日均线' if latest >= ma20 else '跌破20日均线'}（MA20={ma20:.2f}）。")
    else:
        lines.append("- 数据状态：该资产历史序列不可用，以下不生成方向性结论。")
    lines += ["- 偏多触发：价格重新站稳短中期均线，且美元/实际利率等外部约束不继续恶化。",
              "- 偏空触发：价格跌破关键支撑并伴随成交放大，或宏观约束与资金流同时转差。",
              "- 执行动作：触发前只保留观察仓；触发后分两次建立，不在单日急涨后追价；失效即降回观察。"]
    return lines


def get_gold_london_price() -> dict | None:
    """获取伦敦金价格（新浪黄金TD）"""
    try:
        url = "https://hq.sinajs.cn/list=au9999"
        r = requests.get(url, timeout=10, headers={"Referer": "https://finance.sina.com.cn"})
        r.encoding = "gbk"
        if "au9999" in r.text:
            parts = r.text.split(",")
            if len(parts) > 3:
                return {
                    "name": "AU9999",
                    "price": float(parts[3]),
                    "pct": round((float(parts[3]) / float(parts[2]) - 1) * 100, 2) if parts[2] else 0,
                }
    except Exception as e:
        logging.getLogger(__name__).error(f"[generate_asset_weekly_report] 操作失败: {e}", exc_info=True)
    return None


def generate_gold_weekly() -> str:
    """Generate a decision-grade gold macro report, not a price memo."""
    dstr = date.today().isoformat()
    lines = ["# 黄金宏观周报", "", f"生成日期：{dstr}", "", "## 本周结论", "- 先判断金融条件，再判断金价：美元与实际利率同步上行时，黄金反弹不作为追涨依据；美元回落、实际利率回落且ETF/央行需求稳定时，才允许提高配置权重。", "- 执行分层：核心仓只在趋势和宏观约束同向时持有；交易仓必须等回踩承接，不用单日涨跌决定仓位。", ""]
    gold = get_gold_london_price()
    lines += ["## 价格与趋势"]
    lines.append(f"- AU9999：{gold['price']}（{gold['pct']:+.2f}%）" if gold else "- AU9999：实时源不可用，沿用历史仓库并标注日期。")
    lines += _decision_section("黄金价格", {}, "gold")
    for title, asset in (("白银联动", "silver"), ("原油通胀通道", "crude")):
        lines += _decision_section(title, {}, asset)
    lines += ["## 宏观传导与情景", "- 宽松共振：美元回落、实际利率回落、通胀预期稳定，黄金与黄金股可提高观察权重。", "- 避险但金融条件偏紧：地缘或信用风险推升金价，但美元/实际利率仍强，黄金只做防御，不追高弹性股。", "- 再通胀压制：油价上行推高通胀，联储路径转鹰、实际利率上行，黄金和黄金股降权，等待实际利率拐点。", "", "## A股映射", "- 黄金ETF：看金价趋势、成交额和份额是否同向；价格上涨但份额下降，按获利了结处理。", "- 黄金股：看金价弹性、产量兑现、成本和估值，不把金价上涨直接等同于个股买点。", "- 紫金/山东黄金/中金黄金等资源股：必须叠加铜、汇率、成本和公告风险，单一黄金因子不足以执行。", "", "## 下周行动卡", "1. 若美元和实际利率同时回落，黄金回踩不破支撑且ETF份额回升：分两次建立观察仓。", "2. 若金价创新高但实际利率同步上行、ETF份额减少：不追，等待价格与资金重新同步。", "3. 若跌破20日线并伴随成交放大：交易仓退出，核心仓只保留经过风险预算的底仓。", "4. 若数据源缺失或日期错位：只输出观察，不把旧数据改标为本周数据。", "", "## 数据与口径", "- 价格、商品历史和宏观数据按真实观测日期使用；缺失项不补零。", "- 本报告的仓位结论需要与A股统一决策快照、ETF/行业资金图和风险门控交叉确认。", ""]
    return "\n".join(lines)


def generate_muyuan_weekly() -> str:
    """Generate muyuan-focused weekly report markdown."""
    today = date.today()
    dstr = today.strftime("%Y-%m-%d")
    lines = [f"# 牧原股份周报", f"", f"生成日期：{dstr}", ""]

    price = fetch_tencent_price("002714")
    if price:
        lines.append(f"## 股价")
        lines.append(f"- 收盘：{price['close']}（{price['pct']:+.2f}%）")
        lines.append(f"- 最高：{price['high']}，最低：{price['low']}")
        zhongfu_low, zhongfu_high = 25.00, 31.17
        lines.append(f"- 预警区间：{zhongfu_low} ~ {zhongfu_high}（除权后）")
        if zhongfu_low <= price['close'] <= zhongfu_high:
            lines.append(f"- ⚠️ 在预警区间内！")
        elif price['close'] < zhongfu_low:
            lines.append(f"- ⚠️ 跌破{zhongfu_low}元抄底下界！")
        else:
            lines.append(f"- ✅ 正常，距预警上界 {round(price['close'] - zhongfu_high, 2)}元")
    else:
        lines.append(f"- 数据获取失败")

    lines.append("")
    lines.append("## 执行纪律")
    lines.append("- 本报告由 scripts/generate_asset_weekly_report.py 专用生成器生成。")
    lines.append("- 投递与备份由 report_delivery.finalize_report 统一处理。")
    lines.append("")
    return "\n".join(lines)


def generate_pig_weekly() -> str:
    """Generate pig cycle weekly report."""
    today = date.today()
    dstr = today.strftime("%Y-%m-%d")
    day_of_month = today.day
    lines = [f"# 生猪周期周报", f"", f"生成日期：{dstr}", ""]

    price = fetch_tencent_price("002714")
    if price:
        lines.append(f"## 牧原股份")
        lines.append(f"- 收盘：{price['close']}（{price['pct']:+.2f}%）")
    else:
        lines.append(f"- 牧原股份数据获取失败")

    if day_of_month >= 10 and day_of_month <= 12:
        lines.append("")
        lines.append("📅 本月10号已到，请执行生猪产能出清状况分析")

    lines.append("")
    lines.append("## 执行纪律")
    lines.append("- 本报告由 scripts/generate_asset_weekly_report.py 专用生成器生成。")
    lines.append("- 生猪周期跟踪需结合能繁母猪存栏/猪粮比/出栏体重等数据。")
    lines.append("- 投递与备份由 report_delivery.finalize_report 统一处理。")
    lines.append("")
    return "\n".join(lines)


def generate_china_internet_weekly() -> str:
    """Generate China internet ETF weekly report."""
    today = date.today()
    dstr = today.strftime("%Y-%m-%d")
    lines = [f"# 159792 港股通互联网ETF 定投周报", f"", f"生成日期：{dstr}", ""]

    try:
        from data_sources import fetch_yfinance_commodities
        symbols = ["^HSI", "^HSTECH"]
        for sym in symbols:
            res = fetch_yfinance_commodities(sym)
            if res.ok:
                data = res.data or {}
                latest = data.get("latest") or {}
                r = latest.get("close")
                ret1d = data.get("return_1d_pct")
                lines.append(f"- {sym}：收盘 {r}，1日涨跌幅 {ret1d}%")
            else:
                lines.append(f"- {sym}：数据获取失败")
    except Exception:
        lines.append("- 恒生/恒生科技数据获取失败")

    lines.append("")
    lines.append("## 执行纪律")
    lines.append("- 本报告由 scripts/generate_asset_weekly_report.py 专用生成器生成。")
    lines.append("- 投递与备份由 report_delivery.finalize_report 统一处理。")
    lines.append("")
    return "\n".join(lines)


TASK_GENERATORS = {
    "gold_weekly": generate_gold_weekly,
    "muyuan_weekly": generate_muyuan_weekly,
    "pig_weekly": generate_pig_weekly,
    "china_internet_weekly": generate_china_internet_weekly,
    # capacity tasks use the same generator as weekly (minimal content)
    "muyuan_capacity": generate_muyuan_weekly,
    "pig_capacity": generate_pig_weekly,
}


def _asset_contract(task: str, markdown: str, as_of: str | None = None) -> dict:
    """Build a contract sidecar while preserving the established Markdown body."""
    specs = {
        "gold_weekly": ("gold", "黄金", "commodity"),
        "zijin_weekly": ("601899", "紫金矿业", "stock"),
        "muyuan_weekly": ("002714", "牧原股份", "stock"),
        "pig_weekly": ("pig-cycle", "生猪周期", "industry"),
        "china_internet_weekly": ("159792", "港股通互联网ETF", "index"),
        "muyuan_capacity": ("002714", "牧原股份产能", "stock"),
        "pig_capacity": ("pig-cycle", "生猪周期产能", "industry"),
    }
    subject_id, name, kind = specs.get(task, (task, task, "market"))
    observed = as_of or _first_report_date(markdown) or date.today().isoformat()
    source = {"id": "src-asset-report", "name": "本地资产周报数据适配器", "kind": "market", "status": "ok" if "数据获取失败" not in markdown and "实时源不可用" not in markdown else "fallback", "observed_at": observed, "locator": "scripts/generate_asset_weekly_report.py", "note": "正文保留历史格式，契约字段为机器可读投研视图"}
    evidence = [{"claim": "资产周报正文已生成", "value": task, "unit": "text", "observed_at": observed, "source_ref": "src-asset-report", "quality": "primary"}]
    if "数据获取失败" in markdown or "实时源不可用" in markdown:
        gaps = [{"field": "realtime_quote", "reason": "source_failed", "impact": "不形成当前价格方向结论", "fallback": "使用明确标注日期的历史仓库", "as_of": observed}]
    else:
        gaps = []
    return make_report_contract("weekly" if task.endswith("weekly") or task.endswith("capacity") else "weekly", name + "周报", observed, subject={"id": subject_id, "name": name, "kind": kind}, report_id=f"asset:{task}:{observed}", period_start=observed, period_end=observed, summary={"stance": "observe", "summary": "报告正文已生成；动作需等待价格、资金和基本面证据同日确认。", "horizon": "weeks", "confidence": None}, evidence=evidence, transmission=[{"from": "价格/供需/金融条件", "to": name, "mechanism": "先验证数据日期与来源状态，再评估盈利、估值或配置含义", "direction": "mixed", "evidence_refs": ["ev-1"], "confidence": None}], risks=[{"description": "实时或宏观证据可能缺失", "trigger": "关键数据未更新或日期错位", "impact": "只保留观察，不扩大风险暴露", "severity": "medium", "evidence_refs": ["ev-1"]}], actions=[{"action": "validate", "target": name, "condition": "价格/趋势与资金或供需证据同日确认", "invalidated_by": "来源失败、日期错位或风险限制触发", "horizon": "next_week", "risk_limit": "按纸面交易风控执行", "evidence_refs": ["ev-1"]}], data_gaps=gaps, sources=[source], chapters=[{"title": "核心结论", "conclusion": "仅作研究与决策辅助，不直接执行交易。", "evidence": ["ev-1"], "implication": "把实时价格、历史序列与金融条件分开审阅", "next_check": "下周更新真实观测日并复核触发条件"}], metadata={"task": task, "legacy_markdown": True})


def _first_report_date(text: str) -> str | None:
    # 审计修复: 原正则写成 r"20\\d{2}-..." 会匹配字面量反斜杠d，永远抓不到日期，
    # 导致 as_of 回退 date.today()，把旧数据标成当天。必须匹配真实日期字符。
    match = re.search(r"20\d{2}-\d{2}-\d{2}", text or "")
    return match.group(0) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate asset weekly report")
    parser.add_argument("--task", required=True, help="task id (gold_weekly, muyuan_weekly, etc.)")
    parser.add_argument("--out", required=True, help="output file path")
    parser.add_argument("--no-finalize", action="store_true", help="skip report_delivery.finalize_report")
    args = parser.parse_args()

    if args.task not in TASK_GENERATORS:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        return 1

    report = TASK_GENERATORS[args.task]()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    contract_error = None
    try:
        contract = _asset_contract(args.task, report)
        validate_and_write_contract(out, contract)
    except Exception as exc:
        contract_error = str(exc)[:240]

    delivery_status = None
    delivery_error = None
    if not args.no_finalize and finalize_report is not None:
        try:
            delivery_status = finalize_report(out)
        except Exception as e:
            delivery_error = str(e)[:240]
            print(f"finalize_report error: {e}", file=sys.stderr)

    result = {"task": args.task, "path": str(out), "bytes": out.stat().st_size,
              "status": "failed" if contract_error else "ok",
              "contract": str(out) + ".research.json" if not contract_error else None,
              "delivery": "failed" if delivery_error else "attempted" if not args.no_finalize else "skipped"}
    if contract_error:
        result["contract_error"] = contract_error
    if delivery_error:
        result["delivery_error"] = delivery_error
    print(json.dumps(result, ensure_ascii=False))
    return 1 if contract_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
