#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pattern_agent — 规律面专家（multi_agent 第 6 位专家）

把 pattern_engine 规律库（当日 confirmed+validating 规律）包装成与
multi_agent 其余 5 位专家同构的 view dict，并并入加权投票：

  输入: pattern_engine.load_knowledge_base() → 当日 confirmed+validating 规律
  信号合成: 当日 confirmed 规律按 (置信 × 方向) 折算投票权重
    up         → 多方 +1
    down       → 空方 -1
    risk_off   → 市场级防守规律，按当前温度方向映射（见 _risk_off_vote）:
        温度 up(修复/发酵/高潮, 风险偏好上行) → 记入空方（过热+防守=顶部警示）
        温度 down(冰点/退潮, 风险偏好下行)   → 记入多方（低位防守=超跌确认, 均值回归）
        温度 neutral(分歧/未知)              → 不计入多空比
    small/large（风格切换）→ 不构成多空投票，仅列证据
  validating 规律 → 尚无足够历史胜率，仅列入 evidence/patterns，不参与加权
  多空比 = 多方权重和 / 空方权重和；>1.5 看多 / <0.67 看空 / 否则中性

与 multi_agent.agent_views() 兼容：提供 pattern_agent.view(date) 返回同结构 dict
（agent/view/confidence/evidence/weight/status/detail + patterns 明细）。

用法:
  python3 -m quant_system.analysis_core.pattern_agent --date 2026-08-11
  python3 -m quant_system.analysis_core.pattern_agent --date 2026-08-11 --multi
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.pattern_engine import load_knowledge_base  # noqa: E402
from quant_system.analysis_core.common import num, today  # noqa: E402

CST = timezone(timedelta(hours=8))

AGENT_NAME = "规律面"
BASE_WEIGHTS = {"规律面": 1.0}   # view() 默认权重（multi_agent 仲裁权重独立维护于 multi_agent.BASE_WEIGHTS）
OUT_DIR = Path(__file__).resolve().parent.parent / "generated"

BULL_RATIO = 1.5      # 多空比 > 1.5 → 看多
BEAR_RATIO = 0.67     # 多空比 < 0.67 → 看空
MAX_CONF = 0.92

def _temperature_direction(date: str | None = None) -> tuple[str, str]:
    """市场温度方向：up=风险偏好上行 / down=风险偏好下行 / neutral=方向不明。

    取 emotion_system.detect(date) 的温度值（0-100，随 date 回放无前视）：
    温度 ≥60（活跃+）→ up；≤40（冰点/低迷）→ down；其余 → neutral。
    """
    try:
        from quant_system.analysis_core.emotion_system import EmotionSystem
        raw = EmotionSystem().detect(date)
    except Exception as e:
        return "neutral", f"温度不可用({str(e)[:60]})"
    temp = raw.get("temperature")
    if temp is None:
        return "neutral", "温度缺失"
    band = raw.get("temp_band") or ""
    cn = raw.get("stage_cn") or ""
    if temp >= 60:
        return "up", f"温度{temp:.0f}({band}/{cn})（风险偏好上行）"
    if temp <= 40:
        return "down", f"温度{temp:.0f}({band}/{cn})（风险偏好下行）"
    return "neutral", f"温度{temp:.0f}({band}/{cn})（方向不明）"


def _risk_off_vote(temp_dir: str) -> float:
    """市场级 risk_off 防守规律按当前温度方向映射为多空投票。

    - 温度上行（过热/风险偏好上行）：防守信号 = 顶部警示 → 记入空方
    - 温度下行（冰点/退潮/风险偏好下行）：防守信号 = 低位超跌确认（均值回归）→ 记入多方
    - 方向不明：不参与多空比
    """
    if temp_dir == "up":
        return -1.0
    if temp_dir == "down":
        return 1.0
    return 0.0


def _pattern_summary(row: pd.Series) -> dict:
    bs: dict = {}
    try:
        bs = json.loads(row.get("backtest_stats") or "{}")
    except Exception as e:
        logging.getLogger(__name__).error(f"[pattern_agent] 操作失败: {e}", exc_info=True)
    return {
        "pattern_id": row.get("pattern_id"),
        "pattern_type": row.get("pattern_type"),
        "name": row.get("name"),
        "direction": row.get("direction"),
        "confidence": num(row.get("confidence")),
        "status": row.get("status"),
        "win_rate_5": num(bs.get("win_rate_5")),
        "n_obs_5": bs.get("samples_5") if isinstance(bs, dict) else None,
    }


def view(date: str | None = None) -> dict:
    """规律面专家观点，与 multi_agent.agent_views() 各 view dict 同构。"""
    day = date or today()
    try:
        kb = load_knowledge_base()
        if kb is None or kb.empty:
            raise ValueError("规律库为空")
    except Exception as e:
        return {"agent": AGENT_NAME, "signal": "震荡", "view": "震荡", "confidence": 0.0,
                "evidence": [f"规律库读取异常: {str(e)[:120]}"], "patterns": [],
                "weight": BASE_WEIGHTS[AGENT_NAME], "status": "degraded",
                "detail": {"date": day, "error": str(e)[:200]}}

    rows = kb[(kb["signal_date"].astype(str) == day) &
              (kb["status"].isin(["confirmed", "validating"]))]
    if rows.empty:
        return {"agent": AGENT_NAME, "signal": "震荡", "view": "震荡", "confidence": 0.25,
                "evidence": [f"{day} 当日无 confirmed/validating 规律，规律面无票"],
                "patterns": [], "weight": BASE_WEIGHTS[AGENT_NAME], "status": "ok",
                "detail": {"date": day, "confirmed": 0, "validating": 0}}

    temp_dir, temp_note = _temperature_direction(day)
    confirmed = rows[rows["status"] == "confirmed"]
    validating = rows[rows["status"] == "validating"]

    bull = 0.0
    bear = 0.0
    vote_notes: list[str] = []
    for _, r in confirmed.iterrows():
        conf = num(r.get("confidence"), 0.5)
        d = r.get("direction")
        if d == "up":
            bull += conf
        elif d == "down":
            bear += conf
        elif d == "risk_off":
            v = _risk_off_vote(temp_dir)
            if v > 0:
                bull += conf
            elif v < 0:
                bear += conf
            vote_notes.append(f"risk_off[{r.get('name')}] 按温度方向({temp_dir})映射→"
                              f"{'多方' if v > 0 else ('空方' if v < 0 else '不投票')}")
        # small/large（风格切换）→ 不构成多空投票，仅列证据

    if bull + bear > 0:
        ratio = bull / bear if bear > 0 else float("inf")
        if ratio > BULL_RATIO:
            view_ = "多"
        elif ratio < BEAR_RATIO:
            view_ = "空"
        else:
            view_ = "震荡"
        agreement = max(bull, bear) / (bull + bear)
        conf = min(MAX_CONF, 0.35 + 0.35 * agreement + min(0.20, 0.04 * len(confirmed)))
    else:
        view_, ratio, conf = "震荡", 0.0, 0.30
    conf = round(conf, 2)
    ratio_txt = "∞" if ratio == float("inf") else f"{ratio:.2f}"

    ev = [f"{day} 当日 confirmed {len(confirmed)} 条 / validating {len(validating)} 条",
          f"多方权重 {bull:.2f} / 空方权重 {bear:.2f} → 多空比 {ratio_txt} → {view_}",
          temp_note]
    ev.extend(vote_notes)
    for _, r in confirmed.head(8).iterrows():
        ev.append(f"✅confirmed[{r.get('name')}] {r.get('direction')} 置信{num(r.get('confidence'), 0):.2f}")
    for _, r in validating.head(5).iterrows():
        ev.append(f"⏳validating[{r.get('name')}] {r.get('direction')} 置信{num(r.get('confidence'), 0):.2f}")

    patterns = [_pattern_summary(r) for _, r in rows.iterrows()]
    return {"agent": AGENT_NAME, "signal": view_, "view": view_, "confidence": conf,
            "evidence": ev, "patterns": patterns,
            "weight": BASE_WEIGHTS[AGENT_NAME], "status": "ok",
            "detail": {"date": day, "temperature_dir": temp_dir,
                       "confirmed": int(len(confirmed)), "validating": int(len(validating)),
                       "bull_weight": round(bull, 3), "bear_weight": round(bear, 3),
                       "ratio": None if ratio == float("inf") else round(ratio, 3),
                       "n_patterns": int(len(rows))}}


def detect(date: str | None = None) -> dict:
    """轻量 detect：规律面无独立检测管线，复用 view() 全量结果（含 patterns/detail）。"""
    return view(date)


def report(date: str | None = None, out_dir: Path | str | None = None) -> Path:
    """轻量报告：复用 view() 写 generated/pattern_report_{date}.md，返回文件路径。"""
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    v = view(date)
    day = (v.get("detail") or {}).get("date") or date or today()
    lines = [f"# 规律面报告 — {day}", "",
             f"- 结论: **{v['view']}** | 置信 {v['confidence']:.0%} | 状态 {v['status']}", ""]
    lines += [f"- {e}" for e in v.get("evidence") or []]
    path = out / f"pattern_report_{day}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_with_multi_agent(date: str | None = None) -> dict:
    """multi_agent.arbitrate 结果 + 规律面证据，供 battle_map/决策卡消费。"""
    from quant_system.analysis_core.multi_agent import arbitrate
    res = arbitrate(date)
    res["pattern_agent"] = view(date or res.get("date") or None)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="规律面专家（multi_agent 第 6 位）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD")
    ap.add_argument("--multi", action="store_true", help="合并 multi_agent 仲裁结果")
    args = ap.parse_args()
    if args.multi:
        print(json.dumps(run_with_multi_agent(args.date), ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(view(args.date), ensure_ascii=False, indent=2, default=str))
