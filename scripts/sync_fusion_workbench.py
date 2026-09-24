#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将统一决策快照写回单文件融合研报，并刷新飞书正文。"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "Desktop" / "研报共享" / "A股融合研报_2026-08-21.html"
OUT_ROOT = ROOT.parent / "Desktop" / "研报共享"
MARK_START = "<!-- unified-decision-start -->"
MARK_END = "<!-- unified-decision-end -->"

# Kept local so the standalone report has the same terminal tokens as V13 Web.
FUSION_TERMINAL_THEME = """
<style id="v13-terminal-report-theme">
:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--line:#30363d;--navy:#161b22;--teal:#58a6ff;--red:#f6465d;--green:#2ebd85;--gold:#d29922;--font-num:"SF Mono","JetBrains Mono",Consolas,"Liberation Mono",monospace}
body{background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}.app{grid-template-columns:216px minmax(0,1fr) 260px}.side,.right{background:var(--panel);border-color:var(--line)}.side{padding:18px 12px}.brand{font:700 15px/1.3 var(--font-num);letter-spacing:.08em;color:var(--teal)}.side .sub,.report small,.navtitle{color:var(--muted)}.nav button,.report{border-radius:6px;color:#c9d1d9}.nav button:hover,.nav .active{background:#1c2128;color:var(--ink)}.main{max-width:1100px;padding:22px 26px 48px}.top{position:sticky;top:0;z-index:20;padding:10px 0;background:rgba(13,17,23,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}.eyebrow{color:var(--teal);font:700 11px var(--font-num);letter-spacing:.08em}.btn{background:transparent;border-color:var(--line);border-radius:6px;color:var(--ink)}.primary{background:var(--teal);border-color:var(--teal);color:#08111c}.hero{background:var(--panel);border:1px solid var(--teal);border-left:4px solid var(--teal);border-radius:8px}.hero p,.score small{color:var(--muted)}.score,.metric b,td{font-variant-numeric:tabular-nums}.metrics{gap:8px}.metric,.section,.block{background:var(--panel)!important;border-color:var(--line)!important;border-radius:8px!important}.metric{border-left:1px solid var(--line)!important}.section{padding:16px;margin-bottom:12px}.section h2{font-size:15px;border-left:3px solid var(--teal);padding-left:8px}.section h2 span,.sub{color:var(--muted)!important}.right{padding:16px 12px}.block{padding:12px}.decision{border-left:3px solid var(--gold);background:#1c2128}.topic{border-color:var(--line)}.topic button{background:transparent;border:1px solid var(--line);border-radius:4px;color:var(--muted)}table{font-size:12px}th,td{border-color:var(--line);padding:8px 10px}th{background:#1c2128;color:#c9d1d9}tbody tr:hover{background:rgba(255,255,255,.03)}textarea,input{background:#0d1117;color:var(--ink);border-color:var(--line);border-radius:6px}details{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:8px}summary{color:#c9d1d9;font-weight:650}@media(max-width:900px){.app{grid-template-columns:1fr}.side,.right{display:none}.main{padding:14px}}
</style>
"""


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def esc(value: Any) -> str:
    return html.escape("-" if value is None or value == "" else str(value), quote=True)


def fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "-"


def js_excerpt(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) > limit:
        text = text[:limit] + "\n...截断，完整文件见 workspace/generated ..."
    return esc(text)


def research_path_for(snapshot: dict) -> Path:
    date = ((snapshot.get("evidence") or {}).get("research") or {}).get("as_of") or snapshot.get("as_of")
    return ROOT / "generated" / f"research_flow_{date}.json"


def candidate_line(item: dict) -> str:
    """Render one stock as an auditable action card, not a score-only list item."""
    ev = item.get("research_evidence") or {}
    stock_flow = item.get("stock_fund_flow") or {}
    flow_match = item.get("flow_match") or {}
    strategy = item.get("execution_strategy") or {}
    positive = "、".join(item.get("short_term_reasons") or []) or "暂无资金/价格正证据"
    blockers = "、".join(item.get("short_term_blockers") or item.get("trade_reasons") or []) or "暂无额外拦截"
    stock_net = stock_flow.get("main_net_yi")
    stock_flow_text = f"逐股主力{fmt(stock_net, 2)}亿" if stock_net is not None else stock_flow.get("status", "无逐股主力流")
    research_text = (
        f"命中{ev.get('reports', 0)}篇/有效提取{ev.get('decision_extracted_reports', 0)}篇"
        f"，评级{ev.get('rating') or '-'}，催化{'、'.join(ev.get('top_catalysts') or []) or '-'}"
    )
    return (
        f"### {item.get('name') or item.get('code')}（{item.get('code')}）｜{item.get('trade_label')}｜短线分 {fmt(item.get('short_term_score'), 1)}\n"
        f"- 资金：行业 {flow_match.get('name') or '未匹配'} {fmt(flow_match.get('net_yi'), 2)}亿；{stock_flow_text}；成交额 {fmt(item.get('amount_yi'), 2)}亿\n"
        f"- 为什么关注：{positive}\n"
        f"- 为什么不能直接买：{blockers}\n"
        f"- 研报证据：{research_text}。研报只解释催化和预期差，不替代盘口确认\n"
        f"- 次日触发：{strategy.get('entry') or '竞价、成交和行业资金同时增强后再评估'}\n"
        f"- 加仓条件：{strategy.get('add') or '未定义，不加仓'}\n"
        f"- 失效/退出：{strategy.get('stop') or '资金转弱或价格结构破坏即停止执行'}"
    )


def build_feishu(snapshot: dict) -> str:
    """渲染盘后飞书正文：所有数据项均保留状态与执行影响。"""
    market = snapshot.get("market") or {}
    evidence = snapshot.get("evidence") or {}
    research = evidence.get("research") or {}
    regulatory = evidence.get("regulatory") or {}
    holders = evidence.get("holders") or {}
    decay = evidence.get("factor_decay") or {}
    counts = snapshot.get("counts") or {}
    coverage = snapshot.get("intraday_coverage") or {}
    source_chain = snapshot.get("source_chain") or {}
    money_flow = snapshot.get("money_flow") or {}
    etf_activity = snapshot.get("etf_activity") or {}
    source_dates = snapshot.get("source_dates") or {}
    flags = market.get("risk_flags") or []
    force_index = float(market.get("force_index") or 0)
    coverage_status = coverage.get("status", "unknown")
    gate_items = []
    if force_index < 35:
        gate_items.append(f"资金合力 {fmt(force_index, 1)} < 35：候选降为观察，须待竞价与成交确认")
    if coverage_status != "complete":
        gate_items.append(f"盘中覆盖={coverage_status}：禁止将候选升格为可执行")
    if decay.get("status") not in ("ok", "available", "ready"):
        gate_items.append(f"日内因子={decay.get('status', 'unavailable')}：不纳入权重")
    hard_risk = int(counts.get("hard_risk_candidates") or 0)
    if hard_risk:
        gate_items.append(f"监管硬风险拦截 {hard_risk} 个候选：不因强势信号放行")
    gate_status = "BLOCKED" if coverage_status != "complete" else ("CAUTION" if gate_items else "OPEN")
    top_themes = "、".join(
        f"{x.get('concept')}({x.get('reports')}篇/{(x.get('leader') or {}).get('leader') or '无龙头'})"
        for x in (research.get("concepts") or [])[:6]
    ) or "暂无"
    holder_text = "；".join(
        f"{x.get('name') or x.get('code')} {str(x.get('as_of'))[:10]} 户数{fmt(x.get('holder_count'),0)} 变化{fmt(x.get('change_pct'),2)}%"
        for x in (holders.get("items") or [])[:4]
    ) or "暂无"
    risk_text = "；".join(
        f"{x.get('name')}({x.get('code')}) 硬风险{x.get('tier1')}条 最新{x.get('latest')}"
        for x in (regulatory.get("top_codes") or [])[:5]
    ) or "暂无"
    candidate_pool = snapshot.get("short_term_candidates") or snapshot.get("observation_candidates") or snapshot.get("opportunities") or []
    opp_text = "\n".join(candidate_line(item) for item in candidate_pool[:8]) or "- 暂无具备完整证据链的候选"
    confirmed_mainlines = [x for x in (snapshot.get("mainlines") or []) if x.get("level") == "主线"]
    mainline_text = "；".join(
        f"{x.get('industry') or x.get('name')}：资金{fmt(x.get('net_yi'),2)}亿，候选{x.get('candidate_count',0)}只，梯队{x.get('zt',0)}只/最高{x.get('max_board',0)}板"
        for x in confirmed_mainlines[:5]
    ) or "无主题通过同日资金与扩散/梯队双确认"
    top_in = money_flow.get("top_in") or []
    top_out = money_flow.get("top_out") or []
    money_text = "；".join(f"{x.get('name') or x.get('sector') or x.get('industry')} {fmt(x.get('main_net_yi') or x.get('net_yi') or x.get('value'),2)}亿" for x in top_in[:5]) or "暂无可用净流入排行"
    outflow_text = "；".join(f"{x.get('name') or x.get('sector') or x.get('industry')} {fmt(x.get('main_net_yi') or x.get('net_yi') or x.get('value'),2)}亿" for x in top_out[:5]) or "暂无可用净流出排行"
    source_text = "、".join(k for k, available in source_chain.items() if available) or "无可用来源"
    gate_lines = [f"- {item}" for item in gate_items] or ["- 未触发额外门控；仍须遵守单票风险预算与次日确认。"]
    extracted = int(research.get("decision_extracted_reports") or 0)
    parsed_reports = int(research.get("parsed_reports") or research.get("reports") or 0)
    extraction_rate = float(research.get("decision_extraction_rate") or (extracted / parsed_reports if parsed_reports else 0))
    lines = [
        f"# 次日交易决策｜{snapshot.get('as_of', '-')}｜GATE {gate_status}",
        "## 数据摘要",
        f"- {('当前没有可直接执行标的，次日先观察资金是否回流主线。' if gate_status != 'OPEN' else '市场门控开放，但仍只执行满足竞价、成交和行业资金三项触发的候选。')}",
        "## 一句话结论",
        f"- 市场温度 {fmt(market.get('temperature'), 1)}，资金合力 {fmt(force_index, 1)}，上涨宽度 {fmt(float(market.get('breadth') or 0) * 100, 1)}%；{((snapshot.get('scope_detail') or {}).get('execution') or '仅作分析与决策辅助，不连接券商执行')}",
        "## 资金去了哪里",
        f"- 已确认主线：{mainline_text}",
        f"- 净流入前排：{money_text}",
        f"- 净流出前排：{outflow_text}",
        "- 资金口径：逐股主力流/行业主力流优先；成交额×涨跌方向仅标记为代理，不冒充主力净流入。",
        "## 执行门控",
        "## 为什么现在不能直接做",
        *gate_lines,
        f"- 数据覆盖：全市场 {counts.get('scanned', 0)}，盘中信号 {coverage.get('signal_evaluated', 0)}/{coverage.get('snapshot_rows', 0)} ({coverage_status})；硬风险候选 {hard_risk}。",
        "## 逐股行动卡",
        opp_text,
        "## 研报真正提供了什么",
        f"- 入库 {research.get('reports', 0)} 篇，结构化解析 {parsed_reports} 篇，提取出评级/目标价/催化等决策字段 {extracted} 篇（{extraction_rate:.1%}）。未提取出决策字段的资料只算索引覆盖，不给候选加分。",
        f"- 覆盖代码 {research.get('all_codes', 0)}，前排主题 {top_themes}。主题热度不等于可交易主线，必须由同日资金和扩散/梯队确认。",
        "## 风险结构",
        "## 风险与数据缺口",
        f"- 市场风险：{'；'.join(flags) if flags else '暂无额外市场风险标记'}",
        f"- 监管风险：索引 {regulatory.get('total_hits', 0)} 条／代码 {regulatory.get('codes', 0)} 个；硬风险 {((regulatory.get('tier_counts') or {}).get('硬风险', 0))} 条。样本：{risk_text}",
        f"- 筹码状态：股东户数 {holders.get('available', 0)}/{holders.get('total', 0)} 可用；{holder_text}",
        f"- 因子状态：{decay.get('status', 'unavailable')}，有效窗口 {decay.get('n_windows', 0)}；{decay.get('reason') or '形成有效窗口后才纳入权重'}。可用来源：{source_text}",
        "## 次日复核顺序",
        "- 先看竞价强弱，再看开盘前两轮5分钟成交与行业资金是否同步增强，最后核对个股是否守住开盘价/分时均价；任一环节失败，不新增进攻仓。",
    ]
    return "\n".join(lines)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_section(snapshot: dict, research_raw: dict) -> str:
    market = snapshot.get("market") or {}
    evidence = snapshot.get("evidence") or {}
    research = evidence.get("research") or {}
    holders = evidence.get("holders") or {}
    regulatory = evidence.get("regulatory") or {}
    decay = evidence.get("factor_decay") or {}
    counts = snapshot.get("counts") or {}
    flags = market.get("risk_flags") or []
    holder_stat = f"{holders.get('available', 0)}/{holders.get('total', 0)}"
    concept_rows = [
        [x.get("concept"), x.get("reports"), (x.get("leader") or {}).get("leader"), (x.get("leader") or {}).get("signal"), (x.get("leader") or {}).get("zt_cnt")]
        for x in (research.get("concepts") or [])[:20]
    ]
    holder_rows = [
        [x.get("code"), x.get("name"), x.get("as_of"), fmt(x.get("holder_count"), 0), f"{fmt(x.get('change_pct'), 2)}%", x.get("signal"), x.get("source")]
        for x in (holders.get("items") or [])
    ]
    risk_rows = [
        [x.get("code"), x.get("name"), x.get("count"), x.get("tier1"), x.get("latest"), ((x.get("recent") or [{}])[0]).get("type"), ((x.get("recent") or [{}])[0]).get("title")]
        for x in (regulatory.get("top_codes") or [])[:30]
    ]
    opp_rows = []
    for item in (snapshot.get("opportunities") or []):
        ev = item.get("research_evidence") or {}
        opp_rows.append([
            item.get("code"), item.get("name"), item.get("board"), item.get("trade_label"), item.get("decision"),
            item.get("decision_score"), item.get("mainline_match"), ev.get("reports", 0), "、".join(ev.get("top_concepts") or []),
            "；".join(item.get("trade_reasons") or []),
        ])
    source_rows = [[k, "可用" if v else "未接入"] for k, v in (snapshot.get("source_chain") or {}).items()]
    flags_html = "".join(f"<li>{esc(flag)}</li>" for flag in flags) or "<li>暂无额外风险标记</li>"
    parsed_sample = (research_raw.get("parsed") or [])[:90]
    raw_payload = {
        "decision_snapshot": snapshot,
        "research_flow_summary": {
            "date": research_raw.get("date"),
            "n_reports": research_raw.get("n_reports"),
            "n_ocr": research_raw.get("n_ocr"),
            "all_concepts": (research_raw.get("all_concepts") or [])[:100],
            "all_codes": (research_raw.get("all_codes") or [])[:160],
            "all_companies": (research_raw.get("all_companies") or [])[:220],
            "chain_hits": (research_raw.get("chain_hits") or [])[:80],
            "leader_hits": (research_raw.get("leader_hits") or [])[:80],
            "parsed_sample": parsed_sample,
        },
    }
    return f'''{MARK_START}<section class="section" id="unified-decision"><h2>统一决策快照 <span>{esc(snapshot.get('as_of'))} · 同口径证据层</span></h2><p>市场：{esc(market.get('emotion_stage'))} · 温度 {fmt(market.get('temperature'),1)} · 资金合力 {fmt(market.get('force_index'),1)} · 宽度 {fmt(float(market.get('breadth') or 0)*100,1)}%。候选 {esc(counts.get('scanned'))}，主板可执行初筛 {esc(counts.get('execution_candidates'))}，硬风险拦截 {esc(counts.get('hard_risk_candidates'))}。</p><div class="metrics"><div class="metric"><b>{esc(research.get('reports'))}</b><small>真实研报/IMA资料</small></div><div class="metric"><b>{esc(research.get('all_codes'))}</b><small>研究覆盖代码</small></div><div class="metric"><b>{esc(regulatory.get('total_hits'))}</b><small>监管/问询索引</small></div><div class="metric"><b>{esc(holder_stat)}</b><small>股东数据可用</small></div></div><h3>当前风险门控</h3><ul>{flags_html}</ul><h3>候选执行矩阵</h3>{table(['代码','名称','板块','标签','决策','评分','主线匹配','研报数','研报主题','拦截/备注'], opp_rows)}<h3>研报主题与龙头证据</h3>{table(['主题','研报数','龙头','信号','涨停数'], concept_rows)}<h3>股东户数</h3>{table(['代码','名称','报告期','户数','变化','解释','来源'], holder_rows)}<h3>监管/问询风险清单</h3>{table(['代码','名称','总数','硬风险','最新','类型','最近公告'], risk_rows)}<h3>日内因子衰减</h3><p>状态：<b>{esc(decay.get('status'))}</b> · 窗口 {esc(decay.get('n_windows', 0))} · 原因 {esc(decay.get('reason') or '有效窗口形成后才纳入权重')}</p><h3>来源链</h3>{table(['模块','状态'], source_rows)}<details open><summary>决策证据原始摘录（用于复盘审计）</summary><pre style="white-space:pre-wrap;max-height:900px;overflow:auto;font-size:11px">{js_excerpt(raw_payload, 185000)}</pre></details></section>{MARK_END}'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(ROOT / "generated" / "decision_snapshot_after_close_2026-08-26.json"))
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    snapshot_path = Path(args.snapshot)
    source = Path(args.source)
    out = Path(args.out) if args.out else OUT_ROOT / f"A股融合研报_{snapshot_path.stem.rsplit('_', 1)[-1]}.html"
    snapshot = load_json(snapshot_path)
    if not snapshot:
        raise SystemExit(f"snapshot unavailable: {snapshot_path}")
    research_raw = load_json(research_path_for(snapshot))
    raw = source.read_text(encoding="utf-8")
    if 'id="v13-terminal-report-theme"' not in raw:
        raw = raw.replace("</head>", FUSION_TERMINAL_THEME + "</head>", 1)
    section = build_section(snapshot, research_raw)
    pattern = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), re.S)
    if pattern.search(raw):
        raw = pattern.sub(section, raw, count=1)
    elif "</main>" in raw:
        raw = raw.replace("</main>", section + "</main>", 1)
    else:
        raise SystemExit("fusion HTML does not contain </main>")
    feishu = build_feishu(snapshot)
    textarea = re.compile(r"(<textarea[^>]*id=[\"']feishu[\"'][^>]*>).*?(</textarea>)", re.S | re.I)
    if textarea.search(raw):
        raw = textarea.sub(lambda m: m.group(1) + html.escape(feishu) + m.group(2), raw, count=1)
    # Keep the legacy right rail from contradicting the current unified gate.
    if float((snapshot.get("market") or {}).get("force_index") or 0) < 35:
        raw = raw.replace("<strong>试探性进攻</strong><span class=\"sub\">仓位 18–30% · 只做主线前排 · 海外转弱即收缩</span>", "<strong>谨慎观察</strong><span class=\"sub\">资金合力偏弱 · 主线未确认不进攻 · 只等盘口承接</span>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw, encoding="utf-8")
    print(json.dumps({"ok": True, "path": str(out), "bytes": out.stat().st_size, "feishu_chars": len(feishu), "snapshot": str(snapshot_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
