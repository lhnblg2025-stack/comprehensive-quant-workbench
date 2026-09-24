#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日复盘决策研判引擎（2026-08-21 重构 —— 从数据罗列到决策语言）

把融合链信号翻译成操盘手语言：
- 定调：情绪×资金×龙头 → 明日操作基调（积极/中性/防守）+ 仓位
- 主线研判：共振主线 → 明日关注方向 + 延续/切换判断
- 操作清单：从选股池提炼操作建议（进攻池/观察池/规避池 + 理由）
- 风险预案：次日触发条件 → 应对动作

输入: 融合链 blocks（市场温度/强势方向/龙头情绪/龙虎榜/多格局/模型）
输出: 决策卡 dict（明确定调 + 操作 + 风险预案）
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("decision_engine")

# 情绪阶段 → 操作基调
EMOTION_POSTURE = {
    "冰点": "防御性布局", "修复": "试探性进攻", "发酵": "积极进攻",
    "复苏": "积极进攻", "高潮": "进攻但防分歧", "分歧": "谨慎中性",
    "退潮": "防守撤退", "恐慌": "防守观望",
}
# 情绪 → 建议仓位区间（基础）
EMOTION_POSITION = {
    "冰点": (0.0, 0.3), "修复": (0.2, 0.5), "发酵": (0.4, 0.7),
    "复苏": (0.4, 0.7), "高潮": (0.3, 0.6), "分歧": (0.1, 0.4),
    "退潮": (0.0, 0.2), "恐慌": (0.0, 0.1),
}
# 资金合力 → 加分/减分
FUND_SCORE = lambda fi: 1 if fi >= 60 else (0.5 if fi >= 40 else 0)  # noqa: E731


def _num(v, d=0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _stage(blocks: dict) -> str:
    emo = (blocks.get("market_temperature") or {}).value.get("components", {}).get("emotion", {})
    return str(emo.get("stage_cn") or "?")


def _fund(blocks: dict) -> dict:
    return (blocks.get("market_temperature") or {}).value.get("components", {}).get("fund", {}) or {}


def _directions(blocks: dict) -> list:
    return (blocks.get("strong_direction") or {}).value.get("directions", []) or []


def _ladder(blocks: dict) -> dict:
    return (blocks.get("leader_sentiment") or {}).value.get("components", {}).get("ladder", {}) or {}


def _pool(blocks: dict, key: str) -> list:
    return (blocks.get("stock_picks") or {}).value.get("pools", {}).get(key, []) or []


def _verdict(blocks: dict) -> dict:
    return (blocks.get("model_verdict") or {}).value.get("verdict", {}) or {}


_MAIN_KEYWORDS = ["医药", "医疗", "生物", "基因", "疫苗", "创新药", "器械",
                "半导体", "芯片", "AI", "算力", "机器人", "汽车", "军工", "储能",
                "光伏", "消费", "食品", "软件", "通信", "电子", "低空", "数据"]


def _mainline_leaders(blocks, main_names: list, top_n: int = 6) -> list:
    """从涨停池提取主线板块的连板龙头个股（行业∩主线 + board_count 排序）。

    这是"进攻池必须含具体龙头股"的关键——比仅列概念名可执行得多。
    返回 [{name, code, board, industry, action}, ...]
    """
    try:
        import pandas as pd
        import sys as _sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parent.parent
        _base = _root / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"
        if not _base.exists():
            return []
        df = pd.read_parquet(_base)
        df["date"] = pd.to_datetime(df["date"])
        d = df[df["date"] == df["date"].max()].copy()
        if d.empty or "industry" not in d.columns:
            return []
        # 主线匹配: 概念名与行业双匹配
        mains = [m for m in main_names if m and "无明确" not in m]
        if not mains:
            return []
        # 行业匹配主线关键词
        kw = []
        for m in mains:
            kw += [m] + [k for k in _MAIN_KEYWORDS if k in m]
        kw = list(set(kw))
        ind = d["industry"].astype(str)
        mask = ind.apply(lambda x: any(k in x or x in k for k in kw if len(k) >= 2))
        med = d[mask].sort_values("board_count", ascending=False)
        if med.empty:
            # 放宽: 涨停池主线名称词直接匹配
            nm = d["name"].astype(str)
            mask2 = nm.apply(lambda x: any(k in x for k in kw if len(k) >= 2))
            med = d[mask2].sort_values("board_count", ascending=False)
        out = []
        for _, r in med.head(top_n).iterrows():
            board = int(r.get("board_count") or 0)
            out.append({
                "name": r.get("name"), "code": str(r.get("code", "")),
                "board": board, "industry": r.get("industry"),
                "action": f"{board}连板龙头" if board >= 2 else "涨停启动",
            })
        return out
    except Exception as _e:  # noqa: BLE001
        print(f"[decision] 主线龙头提取失败: {str(_e)[:60]}")
        return []


def make_decision(blocks: dict) -> dict:
    """核心：信号 → 决策卡。"""
    stage = _stage(blocks)
    fund = _fund(blocks)
    ladder = _ladder(blocks)
    dirs = _directions(blocks)
    verdict = _verdict(blocks)
    zt = _num(ladder.get("zt_cnt"))
    max_board = int(_num(ladder.get("max_board"), 1))
    fi = _num(fund.get("force_index"))
    youzi = _num(fund.get("youzi_net"))
    overheat = _num((blocks.get("leader_sentiment") or {}).value.get("components", {}).get("leader", {}).get("overheat"))

    # ── 1. 定调 ─────────────────────────────────────────
    posture = EMOTION_POSTURE.get(stage, "中性观望")
    pos_lo, pos_hi = EMOTION_POSITION.get(stage, (0.1, 0.4))

    # 修正：资金合力弱 → 下调；游资净买为正 → 加底气
    fund_adj = ""
    if fi < 40:
        pos_hi = max(pos_lo, pos_hi - 0.1)
        fund_adj = "资金合力弱，压一档仓位"
    elif youzi > 0:
        pos_hi = min(0.8, pos_hi + 0.05)
        fund_adj = f"游资净买{youzi/1e8:.1f}亿，略加底气"

    # 赚钱效应：涨停家数与炸板率
    zb_rate = _num(ladder.get("zb_rate"), 0)
    effect = "赚钱效应尚可" if zt >= 60 and zb_rate < 0.25 else ("赚钱效应一般" if zt >= 30 else "赚钱效应弱")
    if zb_rate >= 0.35:
        effect += "（炸板偏高，追高需谨慎）"

    # ── 2. 主线研判 ─────────────────────────────────────
    main_lines = [d for d in dirs if d.get("level") == "主线"]
    main_txt = "、".join(d.get("name", "") for d in main_lines[:3]) if main_lines else "无明确主线（题材快速轮动）"
    # 主线质量：主线数量 + 最高板高度
    main_quality = "强" if (main_lines and max_board >= 4) else ("中" if main_lines else "弱")
    if max_board >= 5:
        main_quality = "很强"

    # ── 3. 操作清单（从池提炼）─────────────────────────
    # 进攻池 v2: 主线龙头个股(行业∩连板) 优先 → 主线概念 → 龙虎榜资金补充
    attack = []
    seen = set()
    _mains = [d.get("name") for d in main_lines if d.get("name")]
    leaders = _mainline_leaders(blocks, _mains, top_n=5)
    for l in leaders:
        nm = f'{l["name"]}({l["code"]})'
        if nm not in seen:
            attack.append({"name": nm, "type": f"主线{l['industry']}",
                           "action": l["action"], "why": f"板块{l['industry']}"})
            seen.add(nm)
    for d in main_lines[:3]:
        nm = d.get("name") or d.get("concept")
        if nm and nm not in seen:
            _lv = str(d.get('level') or '主线')
            attack.append({"name": nm, "type": _lv, "action": "关注低吸/接力", "why": f"score{d.get('score')}"})
            seen.add(nm)
    for s in _pool(blocks, "lhb_stocks")[:3]:
        nm = s.get("name"); cd = s.get("code")
        key = f"{nm}({cd})"
        if nm and key not in seen:
            attack.append({"name": key, "type": "龙虎榜", "action": "观察溢价承接",
                           "why": f"净买{_num(s.get('net')):.2f}亿"})
            seen.add(key)
    # 观察池：RS 强势（趋势未确认）
    observe = [{"name": f"{s.get('name')}({s.get('code')})", "type": "RS强势",
                "action": "待回踩关注", "why": f"RS榜#{s.get('rank')}"}
               for s in _pool(blocks, "rs_stocks")[:4]]
    # 规避池：过热信号 + 炸板风险
    avoid = []
    if overheat and overheat > 0:
        avoid.append({"name": f"过热概念({overheat}个)", "type": "情绪", "action": "规避追高", "why": "扩散过热，谨防龙头见顶"})
    if zb_rate >= 0.35:
        avoid.append({"name": "高炸板方向", "type": "情绪", "action": "不追涨停", "why": f"炸板率{zb_rate:.0%}"})

    # ── 4. 风险预案 ─────────────────────────────────────
    risks = []
    if stage in ("高潮", "发酵"):
        risks.append({"trigger": "最高板断板+炸板率升破40%", "action": "降仓离场，兑现"})
        risks.append({"trigger": "竞价最高板低开>3%", "action": "清仓非龙头"})
    if stage in ("修复", "复苏"):
        risks.append({"trigger": "涨停数回落至30以下", "action": "停止进攻，转防守"})
    if fi < 30:
        risks.append({"trigger": "资金合力持续<30", "action": "仓位压至3成内"})
    if max_board <= 3:
        risks.append({"trigger": "最高板持续<4板", "action": "情绪不支撑连板，做低吸不做接力"})

    # ── 汇总决策卡 ──────────────────────────────────────
    return {
        "date": (blocks.get("market_temperature") or {}).value.get("date", "?"),
        "定调": {
            "posture": posture,
            "position": f"{pos_lo*100:.0f}-{pos_hi*100:.0f}%",
            "fund_adj": fund_adj,
            "basis": f"{stage}·{effect}·最高{max_board}板·合力{fi:.0f}·{verdict.get('consensus','震荡')}",
        },
        "主线研判": {
            "main": main_txt,
            "quality": main_quality,
            "note": f"最高{max_board}板，连板顶格{'继续打开' if max_board >= 4 else '受限'}"
                    + (f"，梯队完整" if zt >= 60 else ""),
        },
        "操作清单": {"attack": attack[:6], "observe": observe, "avoid": avoid},
        "风险预案": risks,
        "决策依据": {
            "情绪阶段": stage, "涨停": int(zt), "最高板": max_board,
            "炸板率": round(zb_rate, 2), "资金合力": round(fi, 0),
            "游资净买(亿)": round(youzi/1e8, 1), "过热概念": int(overheat or 0),
            "专家共识": verdict.get("consensus"),
        },
    }


def render_decision_md(dec: dict) -> str:
    """决策卡 → Markdown（决策先行）。"""
    L = []
    L.append("## 🎯 今日决策（结论先行）")
    d = dec.get("定调", {})
    L.append(f"- **明日定调**: {d.get('posture')} | 建议仓位 **{d.get('position')}**")
    if d.get("fund_adj"):
        L.append(f"- 资金提示: {d['fund_adj']}")
    L.append(f"- 定调依据: {d.get('basis')}")
    L.append("")
    L.append("### 🔍 主线研判")
    m = dec.get("主线研判", {})
    L.append(f"- 主线: **{m.get('main')}**（质量{m.get('quality')}）")
    if m.get("note"):
        L.append(f"- {m['note']}")
    L.append("")
    op = dec.get("操作清单", {})
    L.append("### ✅ 进攻清单（优先关注）")
    if op.get("attack"):
        for a in op["attack"]:
            L.append(f"- **{a['name']}** [{a['type']}] {a['action']} — {a['why']}")
    else:
        L.append("- 无主线加持标的，等待主线明确")
    L.append("")
    L.append("### 👀 观察清单（回踩后关注）")
    if op.get("observe"):
        for o in op["observe"]:
            L.append(f"- {o['name']} [{o['type']}] {o['action']} — {o['why']}")
    else:
        L.append("- 暂无")
    L.append("")
    L.append("### ⛔ 规避清单")
    if op.get("avoid"):
        for a in op["avoid"]:
            L.append(f"- **{a['name']}** [{a['type']}] {a['action']} — {a['why']}")
    else:
        L.append("- 暂无显著规避方向")
    L.append("")
    risks = dec.get("风险预案", [])
    L.append("### 🚨 风险预案（触发即执行）")
    if risks:
        for r in risks:
            L.append(f"- 若 {r['trigger']} → **{r['action']}**")
    else:
        L.append("- 暂无特定预案，按定调执行")
    return "\n".join(L)