"""
meta_reviewer — Meta-Reviewer 元审查器（多空矛盾焦点图, V12.1 方案阶段2）

独立于 11 专家之外的裁判 Agent：
输入  = generated/ 下所有报告（battle_map/fusion/risk_report/valuation/daily_report 等
        md/json）的核心结论（看多/看空/中性 + 置信度 + 依赖数据 as_of）；
        报告分散在 ROOT/generated 与 ROOT/quant_system/generated 两个目录，全部扫描
输出  = 《多空矛盾焦点图》（json + md 双格式，落盘 generated/meta_review/YYYYMMDD/）：
        - 3 个做多逻辑 + 3 个做空逻辑（按置信度排序）
        - 每个逻辑标注来源报告、数据时效性（依赖数据陈旧 > 3 天 → 置信度 × 0.7）
        - 多空矛盾焦点：哪些逻辑互相冲突（如 valuation 看多 vs risk 看空）

特性：
- 降级：某份报告缺失 → 跳过不阻塞；全部缺失 → 输出空焦点图 + note
- 防前视：只读文件名日期 ≤ 目标日期的报告（无日期文件名 → 用内容日期过滤，
  仍无日期 → 跳过并计数）
- 全 mock 无网络；ROOT 用 parent.parent.parent（统一）

用法：
  python3 -m quant_system.analysis_core.meta_reviewer --date 2026-08-10
  python3 -m quant_system.analysis_core.meta_reviewer   # 缺省取最近交易日
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "generated"
REPORT_DIRS = [REPORT_DIR, ROOT / "quant_system" / "generated"]
STALE_DAYS = 3
STALE_PENALTY = 0.7
TOP_N = 3
CST = timezone(timedelta(hours=8))

# 期望出现的报告类型（用于降级说明「哪些专家没交卷」）
EXPECTED_TYPES = [
    "battle_map", "risk", "valuation", "daily", "fusion", "trend", "emotion",
    "factor", "cycle", "vpa", "chart", "behavior", "macro",
]

TYPE_LABELS = {
    "battle_map": "作战地图",
    "risk": "风险纪律",
    "valuation": "基本面估值",
    "daily": "短线日报",
    "fusion": "信号融合",
    "trend": "三屏趋势",
    "emotion": "情绪周期",
    "factor": "因子轮动",
    "cycle": "市场周期",
    "vpa": "量价筹码",
    "chart": "形态识别",
    "behavior": "行为金融",
    "macro": "宏观周期",
}

STANCE_CN = {"long": "看多", "short": "看空", "neutral": "中性"}

_DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_STOCK_CODE_RE = re.compile(r"\(\d{6}\)")


# ---------------------------------------------------------------- 基础工具
def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def extract_date_from_filename(name: str) -> date | None:
    """文件名日期：valuation_report_2026-08-11.md → 2026-08-11。"""
    m = _DATE_RE.search(name)
    return _parse_date(m.group(0)) if m else None


def extract_date_from_content(text: str) -> date | None:
    """内容日期兜底：标题行 / 日期字段 / 数据日期。"""
    for pat in (
        r"^#.*?(20\d{2}-\d{2}-\d{2})",
        r"\"date\"\s*:\s*\"(20\d{2}-\d{2}-\d{2})\"",
        r"日期[:：]\s*(20\d{2}-\d{2}-\d{2})",
        r"数据日期\s*(20\d{2}-\d{2}-\d{2})",
        r"生成时间[:：]\s*(20\d{2}-\d{2}-\d{2})",
    ):
        m = re.search(pat, text[:2000], re.M)
        if m:
            d = _parse_date(m.group(1))
            if d:
                return d
    return None


def _as_confidence(x: float | None) -> float | None:
    """82 → 0.82；0.57 → 0.57。"""
    if x is None:
        return None
    return round(x / 100.0, 3) if x > 1 else round(x, 3)


def _map_stance(s: str | None) -> str:
    """文本 → long/short/neutral。"""
    s = str(s or "")
    if any(k in s for k in ("看多", "做多", "进攻", "加仓", "布局", "持仓", "上涨", "主升", "启动", "多")):
        return "long"
    if any(k in s for k in ("看空", "做空", "防守", "减仓", "清仓", "空仓", "下跌", "退潮", "回避", "空")):
        return "short"
    return "neutral"


def _aggregate_stance(stocks: list[tuple[str, float | None]]) -> tuple[str, float | None]:
    """[(stance, conf)] → 多数立场 + 该立场均值置信度；平局 → 中性。"""
    if not stocks:
        return "neutral", None
    counts = {"long": 0, "short": 0, "neutral": 0}
    for st, _ in stocks:
        counts[st] += 1
    majority = max(counts, key=counts.get)
    if majority == "neutral" or (counts["long"] == counts["short"] and counts["long"] > 0):
        vals = [c for _, c in stocks if c is not None]
        return "neutral", (round(sum(vals) / len(vals), 3) if vals else None)
    vals = [c for st, c in stocks if st == majority and c is not None]
    conf = round(sum(vals) / len(vals), 3) if vals else None
    return majority, conf


# ---------------------------------------------------------------- 各报告解析器
def _parse_battle_map_json(text: str, fdate: date | None) -> dict | None:
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or "recommended" not in d:
        return None
    rec = d.get("recommended", "")
    conf = _as_confidence(d.get("confidence"))
    return {
        "report_type": "battle_map",
        "stance": _map_stance(rec),
        "confidence": conf,
        "as_of": _parse_date(d.get("analyze_date") or d.get("date")) or fdate,
        "summary": f"作战地图建议: {rec}（置信 {conf}）",
    }


def _parse_battle_map_md(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"\*{0,2}建议\*{0,2}[:：]\s*\*{0,2}(进攻|试错|防守)\*{0,2}", text)
    m2 = re.search(r"\*{0,2}置信度\*{0,2}[:：]\s*([0-9.]+)", text)
    if not m:
        return None
    return {
        "report_type": "battle_map",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m2.group(1))) if m2 else None,
        "as_of": fdate,
        "summary": f"作战地图建议: {m.group(1)}",
    }


def _parse_emotion(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"结论[:：]\s*\*{0,2}(看多|看空|中性)\*{0,2}", text)
    m2 = re.search(r"置信\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None
    return {
        "report_type": "emotion",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m2.group(1))) if m2 else None,
        "as_of": fdate,
        "summary": f"情绪周期结论: {m.group(1)}（置信 {m2.group(1) if m2 else '?'}%）",
    }


def _parse_risk(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"操作建议[:：]\s*\*{0,2}(加仓|减仓|清仓|观望|持仓|半仓|空仓)\*{0,2}", text)
    m2 = re.search(r"置信度\s*([0-9.]+)", text)
    if not m:
        return None
    return {
        "report_type": "risk",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m2.group(1))) if m2 else None,
        "as_of": fdate,
        "summary": f"风险纪律操作建议: {m.group(1)}",
    }


def _parse_valuation(text: str, fdate: date | None) -> dict | None:
    headers = list(re.finditer(r"^###\s+.+?\(\d{6}\)\s*—\s*(看多|看空|中性)", text, re.M))
    if not headers:
        return None
    stocks: list[tuple[str, float | None]] = []
    for i, h in enumerate(headers):
        seg = text[h.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        cm = re.search(r"\*{0,2}综合\*{0,2}[:：]\s*\*{0,2}(看多|看空|中性)\*{0,2}\s*[（(]置信\s*([0-9.]+)%?[）)]", seg)
        stocks.append((_map_stance(h.group(1)), _as_confidence(float(cm.group(2))) if cm else None))
    stance, conf = _aggregate_stance(stocks)
    as_dates = [d for d in (_parse_date(x) for x in re.findall(r"财务口径 as_of[:：]\s*(20\d{2}-\d{2}-\d{2})", text)) if d]
    return {
        "report_type": "valuation",
        "stance": stance,
        "confidence": conf,
        "as_of": max(as_dates) if as_dates else fdate,
        "summary": f"基本面估值{len(stocks)}只个股聚合: {STANCE_CN[stance]}（均值置信 {conf}）",
    }


def _parse_vpa(text: str, fdate: date | None) -> dict | None:
    headers = list(re.finditer(r"^###\s+.+?\(\d{6}\)\s*—\s*(看多|看空|中性)", text, re.M))
    if not headers:
        return None
    stocks: list[tuple[str, float | None]] = []
    for i, h in enumerate(headers):
        seg = text[h.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        cm = re.search(r"结论[:：]\s*\*{0,2}(看多|看空|中性)\*{0,2}\s*[（(]置信\s*([0-9.]+)%?[）)]", seg)
        stocks.append((_map_stance(h.group(1)), _as_confidence(float(cm.group(2))) if cm else None))
    stance, conf = _aggregate_stance(stocks)
    return {
        "report_type": "vpa",
        "stance": stance,
        "confidence": conf,
        "as_of": fdate,
        "summary": f"量价筹码{len(stocks)}只个股聚合: {STANCE_CN[stance]}（均值置信 {conf}）",
    }


def _parse_trend(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"综合评级[:：]\s*\*{0,2}(多|空|震荡)\*{0,2}\s*[（(]置信\s*([0-9.]+)[）)]", text)
    if not m:
        return None
    return {
        "report_type": "trend",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m.group(2))),
        "as_of": fdate,
        "summary": f"三屏趋势综合评级: {m.group(1)}（置信 {m.group(2)}）",
    }


def _parse_factor(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"信号[:：]\s*\*{0,2}(多|空|震荡)\*{0,2}\s*[（(]?(看多|看空)?[）)]?\s*\|\s*置信度\s*([0-9.]+)", text)
    if not m:
        return None
    return {
        "report_type": "factor",
        "stance": _map_stance(m.group(2) or m.group(1)),
        "confidence": _as_confidence(float(m.group(3))),
        "as_of": fdate,
        "summary": f"因子轮动信号: {m.group(2) or m.group(1)}（置信度 {m.group(3)}）",
    }


def _parse_cycle(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"信号\s*/\s*置信[:：]\s*(看多|看空|中性|多|空|震荡)\s*/\s*([0-9.]+)", text)
    if not m:
        m2 = re.search(r"策略建议[:：]\s*\*{0,2}(布局|观望|防守|进攻)\*{0,2}", text)
        if not m2:
            return None
        return {
            "report_type": "cycle",
            "stance": _map_stance(m2.group(1)),
            "confidence": None,
            "as_of": fdate,
            "summary": f"市场周期策略建议: {m2.group(1)}",
        }
    return {
        "report_type": "cycle",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m.group(2))),
        "as_of": fdate,
        "summary": f"市场周期信号: {m.group(1)}（置信 {m.group(2)}）",
    }


def _parse_chart(text: str, fdate: date | None) -> dict | None:
    stocks: list[tuple[str, float | None]] = []
    for m in re.finditer(r"^###\s+\d{6}\s+\S+\s*—\s*\*{0,2}(多|空|震荡)\*{0,2}\s*置信\s*(\d+)%", text, re.M):
        stocks.append((_map_stance(m.group(1)), _as_confidence(float(m.group(2)))))
    if not stocks:
        return None
    stance, conf = _aggregate_stance(stocks)
    return {
        "report_type": "chart",
        "stance": stance,
        "confidence": conf,
        "as_of": fdate,
        "summary": f"形态识别{len(stocks)}只个股聚合: {STANCE_CN[stance]}（均值置信 {conf}）",
    }


def _parse_behavior(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"结论[:：]\s*\*{0,2}(看多|看空|中性)\*{0,2}\s*\|\s*置信\s*(\d+)%", text)
    if not m:
        return None
    return {
        "report_type": "behavior",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m.group(2))),
        "as_of": fdate,
        "summary": f"行为金融结论: {m.group(1)}（置信 {m.group(2)}%）",
    }


def _parse_macro(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"^\*{0,2}(滞胀|衰退|避险|复苏|过热|扩张|中性过渡|中性)\b", text, re.M)
    m2 = re.search(r"置信度[:：]\s*\*{0,2}([0-9.]+)", text)
    if not m:
        return None
    return {
        "report_type": "macro",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m2.group(1))) if m2 else None,
        "as_of": fdate,
        "summary": f"宏观模式判定: {m.group(1)}",
    }


def _parse_daily(text: str, fdate: date | None) -> dict | None:
    m = re.search(r"阶段[:：]\s*\*{0,2}(主升|上涨|下跌|震荡|退潮|冰点|启动|修复|下跌末期)\*{0,2}\s*\(?置信度\s*([0-9.]+)", text)
    if not m:
        return None
    return {
        "report_type": "daily",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m.group(2))),
        "as_of": fdate,
        "summary": f"短线日报情绪阶段: {m.group(1)}（置信度 {m.group(2)}）",
    }


def _parse_generic_md(text: str, fdate: date | None) -> dict | None:
    """兜底：任意 md 含「结论: 看多/看空/中性」即提取。"""
    m = re.search(r"结论[:：]\s*\*{0,2}(看多|看空|中性)\*{0,2}", text)
    if not m:
        return None
    m2 = re.search(r"置信(?:度)?\s*[:：]?\s*\*{0,2}([0-9.]+)\s*%?", text)
    return {
        "report_type": "generic",
        "stance": _map_stance(m.group(1)),
        "confidence": _as_confidence(float(m2.group(1))) if m2 else None,
        "as_of": fdate,
        "summary": f"报告结论: {m.group(1)}",
    }


def _parse_generic_json(text: str, fdate: date | None) -> dict | None:
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    key = next((k for k in ("conclusion", "stance", "recommended", "verdict") if k in d), None)
    if key is None:
        return None
    st = _map_stance(d[key])
    if st == "neutral" and "recommended" in d:
        st = _map_stance(d["recommended"])
    conf = _as_confidence(d.get("confidence"))
    return {
        "report_type": "generic",
        "stance": st,
        "confidence": conf,
        "as_of": _parse_date(d.get("as_of") or d.get("date")) or fdate,
        "summary": f"JSON 报告结论: {d[key]}（置信 {conf}）",
    }


# 文件名模式 → 解析器（顺序即优先级）
TYPE_PATTERNS: list[tuple[str, str, callable]] = [
    ("battle_map", r"^battle_map_\d{4}-\d{2}-\d{2}\.json$", _parse_battle_map_json),
    ("battle_map", r"^battle_map_\d{4}-\d{2}-\d{2}\.md$", _parse_battle_map_md),
    ("risk", r"^risk_report_\d{4}-\d{2}-\d{2}\.md$", _parse_risk),
    ("valuation", r"^valuation_report_\d{4}-\d{2}-\d{2}\.md$", _parse_valuation),
    ("daily", r"^short_term_daily\.md$", _parse_daily),
    ("trend", r"^trend_report_\d{4}-\d{2}-\d{2}\.md$", _parse_trend),
    ("emotion", r"^emotion_report_\d{4}-\d{2}-\d{2}\.md$", _parse_emotion),
    ("factor", r"^factor_report_\d{4}-\d{2}-\d{2}\.md$", _parse_factor),
    ("cycle", r"^cycle_report_\d{4}-\d{2}-\d{2}\.md$", _parse_cycle),
    ("vpa", r"^vpa_report_\d{4}-\d{2}-\d{2}\.md$", _parse_vpa),
    ("chart", r"^chart_report_\d{4}-\d{2}-\d{2}\.md$", _parse_chart),
    ("behavior", r"^behavior_report_\d{4}-\d{2}-\d{2}\.md$", _parse_behavior),
    ("macro", r"^macro_report_\d{4}-\d{2}-\d{2}\.md$", _parse_macro),
]


def classify_report(path: Path, text: str, fdate: date | None) -> dict | None:
    """按文件名类型解析单份报告 → 结论 dict 或 None。"""
    name = path.name
    for rtype, pat, parser in TYPE_PATTERNS:
        if re.match(pat, name):
            return parser(text, fdate)
    if path.suffix == ".json":
        return _parse_generic_json(text, fdate)
    if path.suffix == ".md":
        return _parse_generic_md(text, fdate)
    return None


def _apply_staleness(concl: dict, target: date) -> dict:
    """依赖数据陈旧 > 3 天 → 置信度 × 0.7。"""
    as_of = concl.get("as_of")
    raw = concl.get("confidence")
    concl["confidence_raw"] = raw
    if as_of:
        days = (target - as_of).days
        concl["stale_days"] = days
        if raw is not None and days > STALE_DAYS:
            concl["stale"] = True
            concl["confidence"] = round(raw * STALE_PENALTY, 3)
            return concl
    concl["stale"] = False
    if as_of is None:
        concl["stale_days"] = None
    return concl


def latest_report_date(report_dir: Path | None = None) -> date:
    """最近交易日 = 报告文件名中 ≤ 今天的最大日期；无 → 今天。"""
    today = datetime.now().date()
    dirs = [report_dir] if report_dir else REPORT_DIRS
    found = []
    for d in dirs:
        if d.is_dir():
            for p in d.iterdir():
                dt = extract_date_from_filename(p.name)
                if dt and dt <= today:
                    found.append(dt)
    return max(found) if found else today


def scan_reports(target: date | None = None, report_dir: Path | None = None) -> dict:
    """扫描 generated/（双目录）→ 结论列表 + 统计。

    同一报告类型存在多份历史文件时，取 as_of 最新的一份（每专家一份结论），
    其余计入 duplicates_collapsed（防历史文件污染当日焦点图）。
    """
    dirs = [report_dir] if report_dir is not None else REPORT_DIRS
    target = target or latest_report_date(report_dir)
    conclusions: list[dict] = []
    inputs = {"scanned": 0, "parsed": 0, "future_skipped": 0,
              "no_date_skipped": 0, "duplicates_collapsed": 0,
              "unparsed": [], "missing_reports": []}
    for report_dir in dirs:
        if not report_dir.is_dir():
            continue
        for path in sorted(report_dir.iterdir()):
            if path.suffix not in (".md", ".json") or not path.is_file():
                continue
            inputs["scanned"] += 1
            fdate = extract_date_from_filename(path.name)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                inputs["unparsed"].append(path.name)
                continue
            if fdate is None:
                fdate = extract_date_from_content(text)
            if fdate is None:
                inputs["no_date_skipped"] += 1
                continue
            if fdate > target:  # 防前视
                inputs["future_skipped"] += 1
                continue
            concl = classify_report(path, text, fdate)
            if concl is None:
                inputs["unparsed"].append(path.name)
                continue
            inputs["parsed"] += 1
            concl["source"] = path.name
            concl["as_of"] = concl.get("as_of") or fdate
            concl["confidence"] = round(concl["confidence"], 3) if concl.get("confidence") is not None else 0.5
            conclusions.append(_apply_staleness(concl, target))
    # 每报告类型保留 as_of 最新一份
    dedup: dict[str, dict] = {}
    for c in sorted(conclusions, key=lambda c: (c["as_of"] or date.min, c["source"]),
                    reverse=True):
        key = c["report_type"]
        if key in dedup:
            inputs["duplicates_collapsed"] += 1
        else:
            dedup[key] = c
    conclusions = list(dedup.values())
    parsed_types = {c["report_type"] for c in conclusions}
    for rt in EXPECTED_TYPES:
        if rt not in parsed_types:
            inputs["missing_reports"].append(TYPE_LABELS.get(rt, rt))
    return {"conclusions": conclusions, "inputs": inputs, "target": target}


# ---------------------------------------------------------------- 焦点图组装
def build_focus_map(target: date | None = None, report_dir: Path | None = None,
                    write: bool = False) -> dict:
    """组装《多空矛盾焦点图》。write=True 时落盘 json+md。"""
    out_root = report_dir if report_dir is not None else REPORT_DIR
    scanned = scan_reports(target=target, report_dir=report_dir)
    target = scanned["target"]
    conclusions = scanned["conclusions"]
    inputs = scanned["inputs"]

    longs = sorted([c for c in conclusions if c["stance"] == "long"],
                   key=lambda c: (c["confidence"], c["source"]), reverse=True)[:TOP_N]
    shorts = sorted([c for c in conclusions if c["stance"] == "short"],
                    key=lambda c: (c["confidence"], c["source"]), reverse=True)[:TOP_N]

    conflicts = []
    for lo in longs:
        for sh in shorts:
            severity = round(min(lo["confidence"], sh["confidence"]), 3)
            conflicts.append({
                "long_report": lo["source"], "long_type": lo["report_type"],
                "long_confidence": lo["confidence"],
                "short_report": sh["source"], "short_type": sh["report_type"],
                "short_confidence": sh["confidence"],
                "severity": severity,
                "description": f"{TYPE_LABELS.get(lo['report_type'], lo['report_type'])}看多 ↔ "
                               f"{TYPE_LABELS.get(sh['report_type'], sh['report_type'])}看空",
            })
    conflicts.sort(key=lambda c: (c["severity"], c["long_report"], c["short_report"]), reverse=True)

    note = None
    if not conclusions:
        note = f"未找到任何 ≤ {target} 的可用报告，输出空焦点图（降级）。"
    elif not longs and not shorts:
        note = "有报告但无明确多空结论（全部中性/无法解析），输出空逻辑焦点图。"

    focus = {
        "date": str(target),
        "generated_at": datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "report_dir": str(out_root),
        "inputs": inputs,
        "long_logics": [_logic_entry(i, c) for i, c in enumerate(longs, 1)],
        "short_logics": [_logic_entry(i, c) for i, c in enumerate(shorts, 1)],
        "conflicts": conflicts,
        "note": note,
    }
    if write:
        _write_outputs(focus, out_root)
    return focus


def _logic_entry(rank: int, c: dict) -> dict:
    return {
        "rank": rank,
        "stance": STANCE_CN[c["stance"]],
        "report_type": c["report_type"],
        "report_type_label": TYPE_LABELS.get(c["report_type"], c["report_type"]),
        "report": c["source"],
        "confidence": c["confidence"],
        "confidence_raw": c.get("confidence_raw"),
        "as_of": str(c["as_of"]) if c.get("as_of") else None,
        "stale": c["stale"],
        "stale_days": c.get("stale_days"),
        "summary": c.get("summary", ""),
    }


def _write_outputs(focus: dict, report_dir: Path) -> tuple[Path, Path]:
    out_dir = report_dir / "meta_review" / focus["date"].replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"meta_review_{focus['date']}.json"
    mp = out_dir / f"meta_review_{focus['date']}.md"
    jp.write_text(json.dumps(focus, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(render_md(focus), encoding="utf-8")
    return jp, mp


# ---------------------------------------------------------------- MD 渲染
def _render_logics(items: list[dict]) -> list[str]:
    lines = []
    for it in items:
        staleness = (f" ⚠️ 陈旧 {it['stale_days']} 天 → 置信度 {it['confidence_raw']}×0.7={it['confidence']}"
                     if it["stale"] else " ✅ 新鲜")
        lines.append(
            f"{it['rank']}. **{it['report_type_label']}** `{it['report']}` — {it['stance']} "
            f"置信 {it['confidence']} | as_of {it['as_of']} | {staleness}"
        )
        lines.append(f"   - {it['summary']}")
    return lines


def render_md(focus: dict) -> str:
    d = focus["date"]
    inp = focus["inputs"]
    lines = [
        f"# 🧭 多空矛盾焦点图 — {d}",
        "",
        f"> Meta-Reviewer 元审查器 | 扫描 {inp['scanned']} 份报告 → 解析 {inp['parsed']} 份 | "
        f"生成 {focus['generated_at']}",
        "",
        "## 🔴 做多逻辑（Top 3, 按置信度排序）",
    ]
    lines += _render_logics(focus["long_logics"]) or ["- （无）"]
    lines += ["", "## 🟢 做空逻辑（Top 3, 按置信度排序）"]
    lines += _render_logics(focus["short_logics"]) or ["- （无）"]
    lines += ["", "## ⚔️ 多空矛盾焦点"]
    if focus["conflicts"]:
        for c in focus["conflicts"]:
            lines.append(
                f"- ⚔️ {c['description']} — 冲突强度 {c['severity']} "
                f"（`{c['long_report']}` 看多 {c['long_confidence']} ↔ "
                f"`{c['short_report']}` 看空 {c['short_confidence']}）"
            )
    else:
        lines.append("- （无）")
    lines += ["", "## 📉 数据时效性"]
    stale = [it for it in focus["long_logics"] + focus["short_logics"] if it["stale"]]
    if stale:
        for it in stale:
            lines.append(
                f"- ⚠️ `{it['report']}` as_of {it['as_of']}（陈旧 {it['stale_days']} 天 > 3）"
                f"→ 置信度 {it['confidence_raw']}×0.7 = {it['confidence']}"
            )
    else:
        lines.append("- 全部依赖数据新鲜（≤ 3 天），无降权。")
    lines += ["", "## ⚠️ 降级说明"]
    if inp["future_skipped"]:
        lines.append(f"- 防前视跳过未来报告: {inp['future_skipped']} 份")
    if inp["no_date_skipped"]:
        lines.append(f"- 无日期信息跳过: {inp['no_date_skipped']} 份")
    if inp["duplicates_collapsed"]:
        lines.append(f"- 同类历史报告仅取最新: 折叠 {inp['duplicates_collapsed']} 份")
    for u in inp["unparsed"]:
        lines.append(f"- 无法解析结论跳过: `{u}`")
    for miss in inp["missing_reports"]:
        lines.append(f"- 缺失报告（跳过不阻塞）: {miss}")
    if focus.get("note"):
        lines.append(f"- ℹ️ {focus['note']}")
    lines += ["", "---", "*Meta-Reviewer | 裁判视角, 多空矛盾即风险焦点*"]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Meta-Reviewer 元审查器（多空矛盾焦点图）")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，缺省取最近交易日")
    args = ap.parse_args(argv)
    target = _parse_date(args.date) if args.date else None
    focus = build_focus_map(target=target, report_dir=None, write=True)
    print(render_md(focus))
    print(f"\n已落盘: {REPORT_DIR / 'meta_review' / focus['date'].replace('-', '')}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
