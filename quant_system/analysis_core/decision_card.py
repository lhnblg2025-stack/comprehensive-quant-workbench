"""
decision_card — 决策层（V11 辅助决策核心）

3 分钟可读完、可直接执行。硬规则:
  - 冰点/退潮 只允许 空仓/试错 档（防手痒）
  - 每个机会必须带 买入条件 + 止损，没有条件的禁止上榜
  - 所有档位随情绪周期/资金合力自动收紧

用法:
  python3 -m quant_system.analysis_core.decision_card --today
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_DAILY_STATS  # noqa: E402
from quant_system.analysis_core.emotion_cycle import run_today  # noqa: E402
from quant_system.analysis_core.scenario import scenario_forecast  # noqa: E402

# 阶段 → (基调, 仓位区间) —— 已按陈小群心法校准(2026-08-09 RAG审计)
# 知识来源: 陈小群"启动≤20%试错/发酵30-50%加仓/高潮≤20%兑现/退潮0%空仓"
STAGE_STANCE = {
    "ice":        ("空仓观望", "0-20%"),
    "repair":     ("轻仓试错", "10-20%"),
    "ferment":    ("进攻加仓", "30-60%"),
    "climax":     ("高潮兑现", "10-30%"),
    "divergence": ("防守", "10-30%"),
    "ebb":        ("空仓防守", "0-5%"),
}


def _shanghai_ma300_guard(as_of: str | None = None) -> dict:
    """短线决策与融合决策共用的上证MA300长期趋势门控。"""
    path = ROOT / "data_warehouse" / "market" / "index_daily_上证指数.parquet"
    try:
        df = pd.read_parquet(path).sort_values("date")
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)] if as_of else df
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 300:
            return {"state": "unknown", "reason": "上证MA300历史不足", "cap": "0-30%"}
        last, ma300 = float(close.iloc[-1]), float(close.tail(300).mean())
        if last < ma300:
            return {"state": "below", "reason": f"上证{last:.2f}跌破MA300({ma300:.2f})", "cap": "0-20%"}
        return {"state": "above", "reason": f"上证{last:.2f}站上MA300({ma300:.2f})", "cap": None}
    except Exception as exc:
        return {"state": "unknown", "reason": f"上证MA300不可用: {str(exc)[:60]}", "cap": "0-30%"}



def _top_ladder(as_of: str | None = None) -> str:
    """从指定数据日天梯提取最高板结构描述。"""
    df = pd.read_parquet(ZT_DAILY_STATS)
    if as_of:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)]
    if df.empty:
        return "天梯数据不可用"
    import json as _json
    ladder = _json.loads(df.iloc[-1]["ladder_json"])
    if not ladder:
        return "今日无涨停"
    top = max(ladder)
    return f"最高{top}板({ladder[top]}家)"


def build_card(as_of: str | None = None) -> dict:
    emo = run_today(as_of)
    stage = emo["stage"]
    sc = scenario_forecast(emo["date"])
    stance, pos = STAGE_STANCE[stage]
    long_guard = _shanghai_ma300_guard(emo["date"])
    if long_guard["state"] == "below":
        stance, pos = "防守观望", "0-20%"
    elif long_guard["state"] == "unknown" and stage == "ferment":
        stance, pos = "轻仓试错", "0-30%"

    # 机会榜（规则版 v2: 陈小群心法校准——高潮期兑现不进攻）
    opportunities = []
    risks = []
    if stage in ("ferment",):
        opportunities.append({
            "target": f"主线龙头（{_top_ladder(emo['date'])}）",
            "logic": "发酵期加仓主线龙头（心法: 发酵30-50%逐步加码）",
            "buy_cond": "竞价>3%且不炸板",
            "stop": "单笔亏损>5%无条件止损",
            "cap": "15%",
        })
        risks.append("高位断层股禁接力（晋级率断崖）")
    elif stage == "climax":
        opportunities.append({
            "target": "兑现为主，仅留龙头活口",
            "logic": "高潮期逐步兑现降仓（心法: 高潮≤20%兑现保护利润）",
            "buy_cond": "不追高，只减不加",
            "stop": "首阴/炸板即走",
            "cap": "5%",
        })
        risks.append("高潮期追高=接盘（心法: 卖在一致）")
    elif stage == "repair":
        opportunities.append({
            "target": "低位首板（修复期晋级率回升方向）",
            "logic": "冰点后修复，首板溢价转正（心法: 启动期≤20%试错首板）",
            "buy_cond": "首板放量且板块涨停≥3",
            "stop": "次日溢价<0 即走；单笔亏损>5%止损",
            "cap": "10%",
        })
    elif stage == "divergence":
        opportunities.append({
            "target": "唯一逆势品种（分歧期卡位）",
            "logic": "分歧期强者恒强，只做唯一性（心法: 买在分歧）",
            "buy_cond": "逆势上板且封单>流通市值2%",
            "stop": "炸板即走",
            "cap": "8%",
        })
    else:  # ice / ebb
        opportunities.append({
            "target": "空仓（硬规则）",
            "logic": "冰点/退潮期胜率低，等待冰点信号（心法: 会空仓的才是祖师爷）",
            "buy_cond": "涨停家数回升至35+且溢价转正",
            "stop": "-",
            "cap": "0%",
        })

    # 风险榜（心法: 单笔亏损>5%无条件止损）
    risks.append("单笔亏损>5%无条件止损（心法铁律）")
    risks.append("涨停家数 <50 立即降仓" if stage in ("ferment", "climax") else "等待情绪修复信号")
    risks.append("炸板率>40% 清掉所有非龙头")
    risks.append("昨日涨停平均竞价溢价<0 → 今日禁打板")

    # 条件单（竞价/盘中触发；高潮期兑现优先）
    cond_orders = [
        {"trigger": "竞价: 昨日最高板高开<1%", "action": "减半仓"},
        {"trigger": "盘中: 炸板率>40%", "action": "清非龙头"},
        {"trigger": "盘中: 主线板块涨停家数>15", "action": "加仓主线二线"},
    ]
    if stage == "climax":
        cond_orders.append({"trigger": "盘中: 总浮盈回撤>3%", "action": "兑现一半（卖在一致）"})

    if long_guard["state"] == "below":
        opportunities = [{
            "target": "观察仓/已有强势核心",
            "logic": "上证跌破MA300，短线情绪信号服从长期趋势硬门控",
            "buy_cond": "上证重新站回MA300且市场广度连续改善",
            "stop": "指数继续走弱或个股跌破关键位立即退出",
            "cap": "20%",
        }]
        risks.insert(0, long_guard["reason"] + "，禁止进攻")
        cond_orders = [c for c in cond_orders if "加仓" not in c["action"]]
    elif long_guard["state"] == "unknown":
        risks.insert(0, long_guard["reason"] + "，禁止积极进攻")

    return {
        "date": emo["date"],
        "stance": stance,
        "position_range": pos,
        "position_authority": "decision_card+MA300_guard",
        "long_trend_guard": long_guard,
        "basis": f"情绪周期={emo['stage_cn']}(置信{emo['confidence']}) | "
                 f"明日乐观概率{sc['scenarios']['乐观']['prob']:.0%}",
        "opportunities": opportunities,
        "risks": risks,
        "condition_orders": cond_orders,
        "auction_checklist": sc["auction_checklist"],
        "emotion_evidence": emo["evidence"],
    }


def print_card(card: dict) -> str:
    lines = [
        f"📋 决策卡片 {card['date']}",
        "─" * 40,
        f"总基调: {card['stance']}  仓位: {card['position_range']}",
        f"依据: {card['basis']}",
        "",
        "🎯 机会榜",
    ]
    for i, o in enumerate(card["opportunities"], 1):
        lines.append(f"{i}. {o['target']} | {o['logic']}")
        lines.append(f"   买入: {o['buy_cond']} | 止损: {o['stop']} | 上限: {o['cap']}")
    lines += ["", "⛔ 风险榜"]
    for r in card["risks"]:
        lines.append(f"- {r}")
    lines += ["", "🔀 条件单"]
    for c in card["condition_orders"]:
        lines.append(f"- {c['trigger']} → {c['action']}")
    lines += ["", "👀 竞价观察清单 (9:25)"]
    for a in card["auction_checklist"]:
        lines.append(f"- {a}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="决策卡片")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    card = build_card()
    if args.json:
        print(json.dumps(card, ensure_ascii=False, indent=2, default=str))
    else:
        print(print_card(card))
