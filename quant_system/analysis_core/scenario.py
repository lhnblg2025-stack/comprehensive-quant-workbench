"""
scenario — 推演层（V11 预测分析核心）

铁律: 每条预测必须带 时间窗 + 方向 + 概率 + 验证变量。
概率优先来自历史统计（相似日检索），LLM 只负责叙述，不负责编概率。

输出:
  1. 明日情绪周期转移概率（来自情绪周期机的历史转移矩阵）
  2. 相似日检索: 按 (情绪阶段, 涨停家数档, 最高板档) 匹配历史，
     给出次日 涨停家数/溢价/最高板 的实际分布 → 三情景概率底稿
  3. 三情景推演: 乐观/中性/悲观 × 概率 × 触发条件 × 操作预案
  4. 竞价观察清单（9:25 必看项）
  5. 预测自动入库（predictions 记录库）

用法:
  python3 -m quant_system.analysis_core.scenario --today
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.emotion_cycle import run_history, STAGE_CN  # noqa: E402
from quant_system.analysis_core import predictions  # noqa: E402


def _zt_bucket(zt: int) -> str:
    if zt < 35:
        return "低(<35)"
    if zt < 70:
        return "中(35-70)"
    if zt < 100:
        return "高(70-100)"
    return "极高(≥100)"


def similar_days(k: int = 5, as_of: str | None = None) -> list[dict]:
    """相似日检索，严格截至 as_of，避免历史报告读取未来交易日。"""
    df = run_history()
    if as_of:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)]
    last = df.iloc[-1]
    key = (last["stage"], _zt_bucket(int(last["zt_cnt"])), int(last["max_board"]))
    mask = (df["stage"] == key[0]) & \
           (df["zt_cnt"].map(_zt_bucket) == key[1]) & \
           (df["max_board"] == key[2])
    cand = df[mask]
    # 排除自身: 仅当最后一个匹配行确实是最新交易日时才删除（2026-08-10 审计修复:
    # 原 iloc[:-1] 在今日不匹配时会误删最新真实相似日）
    if not cand.empty and cand.index[-1] == df.index[-1]:
        cand = cand.iloc[:-1]
    out = []
    for _, r in cand.tail(k).iterrows():
        nxt = df[df.index == r.name + 1]
        if nxt.empty:
            continue
        n = nxt.iloc[0]
        out.append({
            "date": str(r["date"].date()),
            "stage": STAGE_CN[r["stage"]],
            "zt_cnt": int(r["zt_cnt"]),
            "next_zt_cnt": int(n["zt_cnt"]),
            "next_premium": round(float(n["premium"]), 2) if not pd.isna(n["premium"]) else None,
            "next_max_board": int(n["max_board"]),
        })
    return out


def scenario_forecast(as_of: str | None = None) -> dict:
    """生成次日三情景推演，所有输入严格截至同一 as_of。"""
    emo = __import__("quant_system.analysis_core.emotion_cycle", fromlist=["run_today"]).run_today(as_of)
    as_of = emo["date"]
    sims = similar_days(k=5, as_of=as_of)

    # 相似日次日表现统计 → 概率底稿
    nxt_zt = [s["next_zt_cnt"] for s in sims]
    nxt_prem = [s["next_premium"] for s in sims if s["next_premium"] is not None]
    if nxt_zt:
        p_up = sum(1 for v in nxt_zt if v > 100) / len(nxt_zt)
        p_flat = sum(1 for v in nxt_zt if 60 <= v <= 100) / len(nxt_zt)
        p_down = 1 - p_up - p_flat
    else:
        p_up, p_flat, p_down = 0.4, 0.35, 0.25

    stage = emo["stage"]
    # 阶段先验修正（历史转移矩阵已有，这里取前2）
    trans = emo.get("transition_probs", {})

    base_trigger = {
        "乐观": ["昨日最高板竞价高开>3%且封单强", "主线板块竞价涨停家数≥5", "涨停家数放量突破前日"],
        "中性": ["最高板竞价平开，承接尚可", "主线分歧但低位晋级正常"],
        "悲观": ["最高板低开或竞价炸板", "昨日涨停平均竞价溢价<0", "外围大跌传导"],
    }
    ops = {
        "乐观": "持有主线龙头，可加仓主线二线（总仓位上限内）",
        "中性": "降半仓，换仓至低位晋级股，不追高",
        "悲观": "清仓非龙头，空仓或轻仓试错，等冰点",
    }

    return {
        "date": emo["date"],
        "emotion_stage": emo["stage_cn"],
        "transition_probs": trans,
        "similar_days": sims,
        "sample_ok": bool(nxt_zt),  # W2.5: 无相似日样本时概率为硬编码兜底，下游应降置信
        "scenarios": {
            "乐观": {"prob": round(p_up, 2), "triggers": base_trigger["乐观"], "op": ops["乐观"]},
            "中性": {"prob": round(p_flat, 2), "triggers": base_trigger["中性"], "op": ops["中性"]},
            "悲观": {"prob": round(p_down, 2), "triggers": base_trigger["悲观"], "op": ops["悲观"]},
        },
        "auction_checklist": [
            "昨日最高板竞价涨跌幅与量比（9:25）",
            "昨日涨停股平均竞价溢价（>3%强，<0 弱）",
            "主线题材龙头竞价封单量",
            "跌停板竞价家数（>10 警惕）",
            "外围: 美股/港股/A50期货",
        ],
        "verify_vars": ["次日涨停家数", "次日最高板数", "次日炸板率", "次日昨日涨停溢价"],
    }


def make_predictions(forecast: dict) -> list[str]:
    """推演结果自动入库（预测记录库）。"""
    ids = []
    # 情绪周期转移预测
    for label, prob in forecast["transition_probs"].items():
        ids.append(predictions.add_prediction(
            target_type="情绪", target="明日情绪阶段", direction=label,
            probability=prob, horizon_days=1,
            source_module="emotion_cycle",
            verify_vars=["次日情绪阶段判定"], note="情绪转移矩阵"))
    # 三情景概率预测（以乐观情景为方向代表）
    ids.append(predictions.add_prediction(
        target_type="指数", target="市场次日涨停家数",
        direction="乐观情景", probability=forecast["scenarios"]["乐观"]["prob"],
        horizon_days=1, scenario=forecast["scenarios"],
        verify_vars=forecast["verify_vars"], source_module="scenario",
        note=("相似日检索底稿" if forecast.get("sample_ok")
              else "相似日样本缺失，概率为硬编码兜底(降置信)")))
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="推演层")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--no-store", action="store_true", help="不写入预测记录库")
    args = ap.parse_args()
    fc = scenario_forecast()
    print(json.dumps(fc, ensure_ascii=False, indent=2, default=str))
    if not args.no_store:
        ids = make_predictions(fc)
        print("\n[入库]", len(ids), "条预测")
