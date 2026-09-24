"""
daily_report — 短线全景日报（V11 交付层）

合成: 情绪周期卡 + 连板天梯 + 四路资金合力 + 社交情绪 + 次日推演 + 决策卡片
输出: Markdown（后续接 HTML 看板 / PNG 长图）

用法:
  python3 -m quant_system.analysis_core.daily_report --today [--save]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import emotion_cycle, scenario, decision_card  # noqa: E402
from quant_system.product_contract import PRODUCT_VERSION  # noqa: E402
from quant_system.analysis_core.social_sentiment import collect_all  # noqa: E402
from quant_system.analysis_core.fund_forces import latest as forces_latest  # noqa: E402
from quant_system.analysis_core.ladder import latest as ladder_latest  # noqa: E402


def _ladder_block(as_of: str | None = None) -> str:
    df = ladder_latest(3, as_of=as_of)
    lines = []
    for _, r in df.iloc[::-1].iterrows():
        ladder = {int(k): int(v) for k, v in json.loads(r["ladder_json"]).items()}
        ladder_str = " | ".join(f"{k}板:{v}" for k, v in sorted(ladder.items(), reverse=True))
        gap = [n for n in range(2, max(ladder)) if n not in ladder] if ladder else []
        gap_str = f"⚠️断层@{gap}" if gap else "无断层"
        lines.append(f"- {r['date'].date()}: 涨停{r['zt_cnt']} 炸板{r['zb_cnt']} 跌停{r['dt_cnt']} "
                     f"最高{r['max_board']}板 | 天梯: {ladder_str} | {gap_str}")
    return "\n".join(lines)


def _forces_block(as_of: str | None = None) -> str:
    try:
        df = forces_latest(as_of=as_of)
    except Exception as e:
        return f"- 资金合力数据不可用: {str(e)[:80]}"
    lines = []
    for _, r in df.iloc[::-1].iterrows():
        md = r.get("margin_delta")
        md_str = f" 融资Δ{md:.1f}亿" if pd_notna(md) else ""
        lines.append(f"- {r['date'].date()}: 合力{r['force_index']} | "
                     f"游资{r['youzi_net']/1e8:.2f}亿 机构{r['jg_net']/1e8:.2f}亿 "
                     f"北向{r['north_net']/1e8:.2f}亿{md_str}")
    return "\n".join(lines)


def pd_notna(x) -> bool:
    import pandas as pd
    return pd.notna(x)


def _social_block() -> str:
    rows = collect_all()
    out = []
    for r in rows:
        if not r["ok"]:
            out.append(f"- {r['platform']}: ⚠️ {r['error'][:50]}")
            continue
        v = json.loads(r["value"])
        if r["platform"] == "theme_heat":
            top = " > ".join(f"{x['industry']}涨停{x['zt_cnt']}" for x in v["industries"][:5])
            out.append(f"- 题材热度: {top}")
        elif "sentiment" in v:
            out.append(f"- {r['platform']}: 情感分{v['sentiment']} | Top: {v.get('top', [])[:4]}")
    return "\n".join(out)


def _research_block(as_of: str | None = None) -> str:
    """研报工作流摘要，严格按日报数据日读取。"""
    try:
        from quant_system.analysis_core import research_flow
        date = research_flow._resolve_date(as_of)
        p = ROOT / "generated" / f"research_flow_{date}.json"
        used_date = date
        if not p.exists():
            # 回退最近研究产物（如非交易日/未跑当天）
            cands = sorted((ROOT / "generated").glob("research_flow_*.json"))
            eligible = []
            for candidate in cands:
                candidate_date = candidate.stem.replace("research_flow_", "")[:10]
                if not as_of or candidate_date <= as_of:
                    eligible.append((candidate, candidate_date))
            if not eligible:
                return ""
            p, used_date = eligible[-1]
        rf = json.loads(p.read_text(encoding="utf-8"))
        if not rf.get("n_reports"):
            return ""
        lines = [f"- 研报 {rf.get('n_reports')} 篇 / OCR {rf.get('n_ocr', 0)} 张"
                 + (f"（数据日 {used_date}）" if used_date != date else "")]
        concepts = rf.get("all_concepts", [])[:6]
        if concepts:
            lines.append(f"- 涉及概念: {'、'.join(concepts)}")
        # 研报看好的公司
        companies = rf.get("all_companies", [])[:8]
        if companies:
            lines.append(f"- 研报提及公司: {'、'.join(c.split('(')[0] for c in companies)}")
        # 概念 → 成分股数量
        cstocks = rf.get("concept_stocks", {})
        if isinstance(cstocks, dict) and cstocks:
            brief = "、".join(f"{k}({len(v)}只)" for k, v in list(cstocks.items())[:6])
            lines.append(f"- 研报概念成分股: {brief}")
        for h in rf.get("chain_hits", [])[:4]:
            if h.get("broad_concept"):
                continue  # 宽泛题材的链信号降权，不在日报主列表展示
            lines.append(f"- 产业链: {h.get('concept')} → {h.get('chain')}({h.get('layer')}) "
                         f"温度={h.get('temperature')} 信号={h.get('signal')}")
        leader_hits = [h for h in rf.get("leader_hits", []) if not h.get("concept", "").startswith(("BK", "THS"))]
        for h in leader_hits[:4]:
            lines.append(f"- 龙头扩散: {h.get('concept')} {h.get('signal')} "
                         f"龙头={h.get('leader')} 涨停={h.get('zt_cnt')}")
        actual_date = rf.get("_leader_actual_date")
        if actual_date and actual_date != date:
            lines.append(f"- ⚠️ 龙头数据为最近交易日 {actual_date}（非今日）")
        return "\n".join(lines)
    except Exception:
        return ""


def build_report(as_of: str | None = None) -> str:
    emo = emotion_cycle.run_today(as_of)
    data_date = emo["date"]
    fc = scenario.scenario_forecast(data_date)
    card = decision_card.build_card(data_date)

    p = fc["scenarios"]
    scen = "\n".join(
        f"  {k}({v['prob']:.0%}): {'; '.join(v['triggers'][:2])} → {v['op']}"
        for k, v in p.items())

    lines = [
        f"# 📊 短线全景日报 {emo['date']} · {PRODUCT_VERSION}",
        "",
        "## 🌡️ 情绪周期",
        f"- 阶段: **{emo['stage_cn']}** (置信度 {emo['confidence']})",
        f"- 证据: {', '.join(emo['evidence'])}",
        f"- 明日转移: {emo['transition_probs']}",
        "",
        "## 🪜 连板天梯（近3日）",
        _ladder_block(data_date),
        "",
        "## 💧 四路资金合力",
        _forces_block(data_date),
        "",
        "## 📱 社交情绪",
        _social_block(),
        "",
        "## 🔮 次日推演（三情景）",
        scen,
        f"- 相似日底稿: {[s['date'] + '→次日涨停' + str(s['next_zt_cnt']) for s in fc['similar_days'][:3]]}",
        "",
        "## 📋 决策卡片",
        decision_card.print_card(card),
        "",
        "## 📚 研报工作流",
        _research_block(data_date) or "- 暂无研报工作流数据（未生成 research_flow 或研报为空）",
        "",
        "## 🎯 预测入库",
        f"- 已写入预测记录库: 情绪转移 + 三情景概率（进化闭环）",
        "",
        "---",
        f"*{PRODUCT_VERSION} 自动生成 | 预测均为概率判断，需验证变量确认*",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V11 短线全景日报")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--save", action="store_true", help="保存到 generated/")
    args = ap.parse_args()
    report = build_report()
    print(report)
    if args.save:
        out = ROOT / "generated" / "short_term_daily.md"
        out.write_text(report, encoding="utf-8")
        print(f"\n已保存: {out}")
