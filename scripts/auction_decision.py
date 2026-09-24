#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9:25 集合竞价短线决策报告。

只使用已经落盘的作战图和行情快照。没有权威竞价成交数据时，输出条件决策，
不把盘前预案伪装成真实竞价结果。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
CST = timezone(timedelta(hours=8))


def _latest_before(pattern: str, date: str) -> Path | None:
    candidates = []
    for path in GEN.glob(pattern):
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
        if match and match.group(1) <= date:
            candidates.append((match.group(1), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _auction_data_status() -> dict[str, Any]:
    """Report whether authoritative 9:25 auction data is available."""
    # No local auction dataset is accepted as authoritative unless it has the
    # standard fields; this prevents a prior-day snapshot from being reused.
    candidates = [GEN / f"auction_{datetime.now(CST):%Y-%m-%d}.json"]
    for path in candidates:
        value = _read(path)
        if value.get("as_of") == datetime.now(CST).strftime("%Y-%m-%d") and value.get("authoritative"):
            return {"available": True, "label": "权威竞价数据已接入", "as_of": value.get("as_of")}
    return {"available": False, "label": "权威竞价成交数据未接入", "as_of": None}


def build_decision(date: str | None = None, save: bool = True) -> dict[str, Any]:
    """Build one detailed, auditable auction decision report."""
    day = date or datetime.now(CST).strftime("%Y-%m-%d")
    battle_path = _latest_before("battle_map_*.json", day)
    battle = _read(battle_path)
    auction = _auction_data_status()
    action_card = battle.get("action_card") or battle.get("recommended") or "谨慎观望"
    position = battle.get("position_range") or "轻仓"
    bid_watch = list(battle.get("bid_watch") or [])
    attack = list((battle.get("attack_groups") or {}).get("core") or [])
    observe = list((battle.get("attack_groups") or {}).get("observe") or [])
    risks = list(battle.get("risk_watch") or [])
    degraded = list(battle.get("degraded") or [])
    mainline_names = [str(row.get("name") or row.get("concept") or row.get("industry") or "")
                      for row in attack if isinstance(row, dict)]
    if not mainline_names:
        mainline_names = [str(row.get("name") or row.get("concept") or "")
                          for row in bid_watch if isinstance(row, dict) and row.get("name")]

    reasoning = [
        f"市场总基调为“{action_card}”，建议仓位{position}；竞价阶段先验证市场是否允许进攻，不因单只股票高开就扩大风险。",
        f"主线候选只按概念/题材观察：{('、'.join(mainline_names[:5]) or '当前没有通过证据门槛的概念方向')}。申万一级、二级行业只作为归属信息，不作为主线确认依据。",
        "9:15-9:20 只观察委托变化和高开方向，不把可撤单阶段的挂单量当成真实承接；异常放大但快速撤单的信号降级。",
        "9:20-9:25 重点看不可撤单阶段的价格稳定性、成交是否逐步增加、核心概念内是否有扩散，单只高开而同概念没有跟随时不追。",
        "9:25 之后先看开盘价能否守住，再看前两轮十分钟成交和概念资金是否同步增强；任一项不成立，结论保持观察。",
        "高开超过预案阈值但已经接近涨停的标的，优先视为兑现风险而不是买入机会；低开、炸板或核心股弱于概念平均表现时，执行减仓/回避。",
    ]
    checks = [
        {"阶段": "9:15-9:20", "检查": "可撤单阶段", "动作": "只记录方向，不下结论；撤单异常或报价剧烈跳变，标记为噪声。"},
        {"阶段": "9:20-9:25", "检查": "不可撤单阶段", "动作": "确认核心概念是否同步、价格是否稳定；没有扩散就不追单。"},
        {"阶段": "9:25", "检查": "竞价定价", "动作": "记录高开/低开、成交额、概念内排名和风险股表现，形成开盘前最终等级。"},
        {"阶段": "9:25-9:30", "检查": "开盘承接", "动作": "守住开盘价且概念资金增强才允许试探，否则继续观察。"},
        {"阶段": "9:30-9:40", "检查": "首轮确认", "动作": "十分钟全量扫描结果与概念扩散一致，才允许从观察升级为条件候选。"},
    ]
    report = {
        "ok": bool(battle),
        "schema_version": "auction-decision.v1",
        "date": day,
        "phase": "集合竞价决策",
        "as_of": day if battle else None,
        "generated_at": datetime.now(CST).isoformat(),
        "action_card": action_card,
        "position_range": position,
        "authoritative_data": auction,
        "data_note": "当前报告为盘前作战图转化的条件决策；权威竞价成交数据未接入时，不展示伪造的竞价价格、成交量或封单数。",
        "mainline_concepts": mainline_names[:8],
        "bid_watch": bid_watch[:8],
        "attack_concepts": attack[:8],
        "observe_concepts": observe[:8],
        "risk_watch": risks[:8],
        "reasoning": reasoning,
        "checks": checks,
        "decision_rules": [
            "满足：核心概念至少两只同步、竞价价格稳定、开盘后守住开盘价且十分钟成交/资金增强，才进入条件候选。",
            "不满足：只有单股高开、概念没有扩散、资金数据非同日、或竞价接近涨停，保持观察，不追高。",
            "风险触发：最高板低开超过2%、核心股炸板、炸板率快速上升或市场宽度转弱，立即降低仓位并停止新增进攻仓。",
        ],
        "degraded": degraded,
    }
    if save:
        GEN.mkdir(parents=True, exist_ok=True)
        (GEN / f"auction_decision_{day}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report for logs or notification without internal paths."""
    lines = [
        f"# 集合竞价短线决策（{report.get('date', '未知日期')}）",
        f"**总基调：{report.get('action_card', '谨慎观望')} · 建议仓位：{report.get('position_range', '轻仓')}**",
        f"数据口径：{report.get('data_note', '')}",
        "",
        "## 决策理由",
        *[f"{index}. {text}" for index, text in enumerate(report.get("reasoning", []), 1)],
        "",
        "## 分时检查与动作",
        *[f"- {row.get('阶段')}：{row.get('检查')} → {row.get('动作')}" for row in report.get("checks", [])],
        "",
        "## 执行规则",
        *[f"- {text}" for text in report.get("decision_rules", [])],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成集合竞价短线决策")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    result = build_decision(args.date)
    print(render_markdown(result))
