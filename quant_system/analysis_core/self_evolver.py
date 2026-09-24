"""
self_evolver — 规则自进化引擎（V11 每周日校准）

系统自己发现新规则 → 回测验证 → 样本外通过才部署，从"人工调参"升级为"自进化"。

规则模板库（预置骨架 + 参数搜索）:
  R1  炸板率连续{N}日下降 且 最高板≥{M} → 打板仓位系数×{K}
  R2  涨停家数{op}{V} 且 溢价{op2}{V2} → 次日仓位系数{pos}
  R3  情绪{stage} 且 资金合力{op}{F}(亿) → {action}
  R4  主线数≥{N}(梯队数代理) 且 题材爆发≥{M}(最高板代理) → 温度+{T}
参数搜索空间: N∈{1..5}, M∈{3..7}, K∈{1.2,1.3}, op∈{>,<,>=}, V∈{30,45,60,80,100},
             F∈{50,60,70}, T∈{5,8,10}

数据源（先读列名）:
  data_warehouse/market/zt_daily_stats.parquet
  列: date / zt_cnt / zb_rate / premium / max_board / zt_amount / jr1 / ladder_json ...
  派生代理: 资金合力=zt_amount(亿元)、主线数=ladder_json 梯队数、题材爆发=max_board

回测对齐（严禁前视）:
  T 日特征（含 ≤T 的滚动/环比）→ 预测 T+1 涨停家数。
  标签 = zt_cnt.shift(-1)，T 日触发与 T 日标签严格错开，同日触发即视为前视 bug。
  训练窗口 2020-2025；样本外 2026 至今。
  通过标准: 训练期相对差值>10% 且 样本外相对差值>5% 且 触发次数≥20 且 OOS 触发≥3。

部署:
  通过的规则写入 generated/evolved_rules.json（保留历史，新规则 append，
  旧规则样本外失效则移除；无通过规则时不写空文件）。
  active_rules(date) 返回当日触发的已部署规则 [{rule_id, multiplier, reason}]，
  multiplier 限幅 0.8-1.2，battle_map 可读作"进化规则"节乘入仓位系数连乘。

用法:
  python3 -m quant_system.analysis_core.self_evolver --search --max-combos 50
  python3 -m quant_system.analysis_core.self_evolver --today [--date 2026-08-07]
  python3 -m quant_system.analysis_core.self_evolver --weekly
"""

from __future__ import annotations
import logging

import argparse
import json
import random
import sys
from datetime import timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_DAILY_STATS  # noqa: E402
from quant_system.analysis_core.emotion_cycle import STAGES, classify as _classify  # noqa: E402
from quant_system.analysis_core.common import today  # noqa: E402

CST = timezone(timedelta(hours=8))
OUT_DIR = ROOT / "generated"
DEPLOY_FILE = OUT_DIR / "evolved_rules.json"

TRAIN_YEARS = (2020, 2025)   # 训练窗口
OOS_YEAR = 2026              # 样本外窗口起点
PASS_TRAIN_DIFF = 0.10       # 训练期差值 >10%
PASS_OOS_DIFF = 0.05         # 样本外差值 >5%
MIN_HITS = 20                # 触发次数 ≥20
MIN_OOS_HITS = 3             # 样本外触发 ≥3（防单点噪声）
MULT_LO, MULT_HI = 0.8, 1.2  # 部署 multiplier 限幅

# 参数搜索空间
R1_N = range(1, 6)           # 连续下降 N 日
R1_M = range(3, 8)           # 最高板 ≥M
R1_K = (1.2, 1.3)            # 打板仓位系数
R2_OPS = (">", "<", ">=")
R2_V = (30, 45, 60, 80, 100)      # 涨停家数阈值
R2_OPS2 = (">=", "<")
R2_V2 = (0.0, 1.0, 2.0, 3.0)      # 溢价阈值(%)
R2_POS = (0.8, 1.0, 1.2)          # 次日仓位系数
R3_F = (50, 60, 70)               # 资金合力阈值(亿元)
R3_ACTION = (("加仓", 1.2), ("减仓", 0.8), ("空仓", 0.5))
R4_T = (5, 8, 10)                 # 温度增量


# ────────────────────────────────────────────────────────────
# 数据加载（进程内缓存）
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    """读取 zt_daily_stats 全历史并派生 T 日特征（只依赖 ≤T 的行）。"""
    df = pd.read_parquet(ZT_DAILY_STATS)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    # 环比变化（T vs T-1，仍属 T 日信息，无前视）
    df["zt_chg"] = df["zt_cnt"].diff().fillna(0)
    df["mb_chg"] = df["max_board"].diff().fillna(0)
    # 缺失值回退 3 日均值（与 emotion_cycle.run_history 一致）
    df["premium"] = df["premium"].fillna(df["premium_t3"])
    df["jr1"] = df["jr1"].fillna(df["jr1_t3"])
    # 情绪阶段（T 日特征判定）
    df["stage"] = df.apply(lambda r: _classify(r)["stage"], axis=1)
    # 派生代理特征
    df["fund_force"] = df["zt_amount"] / 1e8                      # 资金合力(亿元)
    df["main_lines"] = df["ladder_json"].apply(_ladder_levels)    # 主线数=梯队数
    df["theme_boom"] = df["max_board"]                            # 题材爆发=最高板
    return df


def _ladder_levels(s: object) -> int:
    """ladder_json 的梯队数（R4 主线数代理）。"""
    try:
        if isinstance(s, str) and s:
            return len(json.loads(s))
    except Exception as e:
        logging.getLogger(__name__).error(f"[self_evolver] 操作失败: {e}", exc_info=True)
    return 0


# ────────────────────────────────────────────────────────────
# 候选规则生成
# ────────────────────────────────────────────────────────────
def _build_candidates() -> list[dict]:
    """网格展开全部候选规则（模板 + 参数 + 可读文本 + multiplier）。"""
    cands: list[dict] = []
    for n in R1_N:
        for m in R1_M:
            for k in R1_K:
                cands.append({
                    "rule_id": f"R1_zb{n}日降_mb≥{m}_k{k}",
                    "template": "R1",
                    "params": {"N": n, "M": m, "K": k},
                    "text": f"炸板率连续{n}日下降 且 最高板≥{m} → 打板仓位系数×{k}",
                    "multiplier": float(k),
                })
    for op in R2_OPS:
        for v in R2_V:
            for op2 in R2_OPS2:
                for v2 in R2_V2:
                    for pos in R2_POS:
                        cands.append({
                            "rule_id": f"R2_zt{op}{v}_prem{op2}{v2}_pos{pos}",
                            "template": "R2",
                            "params": {"op": op, "V": v, "op2": op2, "V2": v2, "pos": pos},
                            "text": f"涨停家数{op}{v} 且 溢价{op2}{v2}% → 次日仓位系数×{pos}",
                            "multiplier": float(pos),
                        })
    for stage in STAGES:
        for op in R2_OPS:
            for f in R3_F:
                for action, mult in R3_ACTION:
                    cands.append({
                        "rule_id": f"R3_{stage}_force{op}{f}_{action}",
                        "template": "R3",
                        "params": {"stage": stage, "op": op, "F": f, "action": action},
                        "text": f"情绪[{stage}] 且 资金合力{op}{f}亿 → {action}",
                        "multiplier": float(mult),
                    })
    for n in R1_N:
        for m in R1_M:
            for t in R4_T:
                cands.append({
                    "rule_id": f"R4_lines≥{n}_boom≥{m}_T+{t}",
                    "template": "R4",
                    "params": {"N": n, "M": m, "T": t},
                    "text": f"主线数≥{n}(梯队) 且 题材爆发≥{m}(最高板) → 温度+{t}",
                    "multiplier": 1.0 + t / 100.0,
                })
    return cands


# ────────────────────────────────────────────────────────────
# 触发判定（T 日特征 → 布尔序列，只向后看）
# ────────────────────────────────────────────────────────────
def _cmp(series: pd.Series, op: str, val: float) -> pd.Series:
    if op == ">":
        return series > val
    if op == "<":
        return series < val
    if op == ">=":
        return series >= val
    raise ValueError(f"未知比较符: {op}")


def _trigger(df: pd.DataFrame, cand: dict) -> np.ndarray:
    """返回 T 日触发掩码。只用 T 及 T 之前的数据（diff/rolling 均向后看）。"""
    p = cand["params"]
    tpl = cand["template"]
    if tpl == "R1":
        dec = (df["zb_rate"].diff() < 0).fillna(False)
        trig = dec.rolling(p["N"], min_periods=p["N"]).sum() >= p["N"]
        trig = trig & (df["max_board"] >= p["M"])
    elif tpl == "R2":
        trig = _cmp(df["zt_cnt"], p["op"], p["V"]) & _cmp(df["premium"], p["op2"], p["V2"])
    elif tpl == "R3":
        trig = df["stage"].eq(p["stage"]) & _cmp(df["fund_force"], p["op"], p["F"])
    elif tpl == "R4":
        trig = (df["main_lines"] >= p["N"]) & (df["theme_boom"] >= p["M"])
    else:
        raise ValueError(f"未知模板: {tpl}")
    return np.asarray(trig.fillna(False), dtype=bool)


# ────────────────────────────────────────────────────────────
# 回测（T 日特征预测 T+1，严禁同日）
# ────────────────────────────────────────────────────────────
def _backtest(df: pd.DataFrame, trigger: np.ndarray) -> dict:
    """统计触发日次日涨停家数 vs 非触发日。

    标签严格用 shift(-1): T 行触发 → T+1 行 zt_cnt，同日触发即前视。
    """
    nxt = df["zt_cnt"].shift(-1)          # T+1 标签
    train = df["year"].between(*TRAIN_YEARS).to_numpy()
    oos = (df["year"] >= OOS_YEAR).to_numpy()

    def _diff(mask: np.ndarray) -> tuple[float, float, float]:
        hit = float(nxt[mask & trigger].mean())
        base = float(nxt[mask & ~trigger].mean())
        if not np.isfinite(base) or base == 0 or not np.isfinite(hit):
            return float("nan"), hit, base
        return hit / base - 1.0, hit, base

    train_diff, train_hit, train_base = _diff(train)
    oos_diff, oos_hit, oos_base = _diff(oos)
    return {
        "train_diff": train_diff,
        "oos_diff": oos_diff,
        "hits": int(trigger.sum()),
        "train_hits": int((train & trigger).sum()),
        "oos_hits": int((oos & trigger).sum()),
        "train_mean": train_hit, "train_base": train_base,
        "oos_mean": oos_hit, "oos_base": oos_base,
        "label": "T+1 zt_cnt",
    }


def _passed(st: dict) -> bool:
    return (
        np.isfinite(st["train_diff"]) and st["train_diff"] > PASS_TRAIN_DIFF
        and np.isfinite(st["oos_diff"]) and st["oos_diff"] > PASS_OOS_DIFF
        and st["hits"] >= MIN_HITS and st["oos_hits"] >= MIN_OOS_HITS
    )


def _eval_candidate(df: pd.DataFrame, cand: dict) -> dict:
    st = _backtest(df, _trigger(df, cand))
    return {
        "rule_id": cand["rule_id"],
        "template": cand["template"],
        "text": cand.get("text", cand["rule_id"]),
        "params": cand["params"],
        "multiplier": cand.get("multiplier", 1.0),
        "train_diff": round(st["train_diff"], 4) if np.isfinite(st["train_diff"]) else None,
        "oos_diff": round(st["oos_diff"], 4) if np.isfinite(st["oos_diff"]) else None,
        "hits": st["hits"],
        "train_hits": st["train_hits"],
        "oos_hits": st["oos_hits"],
        "deployed": bool(_passed(st)),
    }


# ────────────────────────────────────────────────────────────
# 对外接口
# ────────────────────────────────────────────────────────────
def search_rules(max_combos: int = 200) -> list[dict]:
    """网格搜索 + 回测 → 候选规则列表（已通过者排前）。

    全组合上千 → 超限时用固定种子随机采样，保证单次运行 <60s。
    结果存 generated/evolved_rules_{date}.json。
    """
    df = _load_data()
    cands = _build_candidates()
    if len(cands) > max_combos:
        cands = random.Random(42).sample(cands, max_combos)
    results = [_eval_candidate(df, c) for c in cands]
    results.sort(key=lambda r: (not r["deployed"], -(r["train_diff"] or 0.0)))
    payload = {
        "date": today(),
        "window": {"train": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}", "oos": f"{OOS_YEAR}-"},
        "max_combos": max_combos,
        "count": len(results),
        "passed": [r for r in results if r["deployed"]],
        "candidates": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"evolved_rules_{today()}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _load_deployed() -> list[dict]:
    if DEPLOY_FILE.exists():
        try:
            data = json.loads(DEPLOY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def active_rules(date: str | None = None) -> list[dict]:
    """当日触发的已部署规则: [{rule_id, multiplier, reason}]。multiplier 限幅 0.8-1.2。"""
    df = _load_data()
    if date is None:
        date = str(df["date"].iloc[-1].date())
    target = pd.Timestamp(date)
    pos = df.index[df["date"] <= target]
    if len(pos) == 0:
        return []
    idx = int(pos[-1])
    out: list[dict] = []
    for r in _load_deployed():
        cand = {"template": r["template"], "params": r["params"]}
        if _trigger(df, cand)[idx]:
            mult = min(MULT_HI, max(MULT_LO, float(r.get("multiplier", 1.0))))
            out.append({
                "rule_id": r["rule_id"],
                "multiplier": mult,
                "reason": r.get("text", r["rule_id"]),
            })
    return out


def run_weekly() -> dict:
    """周日全流程: 搜索 → 筛选 → 写 evolved_rules.json（保留历史，旧规则样本外失效移除）。"""
    df = _load_data()
    results = search_rules(max_combos=200)
    passed = [r for r in results if r["deployed"]]
    existing = _load_deployed()

    # 旧规则: 用当前全历史重算，样本外失效则移除；未知模板保守保留
    kept: list[dict] = []
    for r in existing:
        if r.get("template") in ("R1", "R2", "R3", "R4"):
            st = _backtest(df, _trigger(df, r))
            if not _passed(st):
                continue  # 样本外失效 → 移除
        kept.append(r)

    # 新通过规则 append（去重）
    new_ids = {r["rule_id"] for r in kept}
    for r in passed:
        if r["rule_id"] in new_ids:
            continue
        kept.append({
            "rule_id": r["rule_id"], "template": r["template"], "text": r["text"],
            "params": r["params"], "multiplier": r["multiplier"],
            "train_diff": r["train_diff"], "oos_diff": r["oos_diff"], "hits": r["hits"],
            "deployed": True, "deployed_at": today(), "last_verified": today(),
        })

    if not kept:
        # 无通过规则 → evolved_rules.json 保持原样，不写空文件
        return {
            "date": today(), "status": "no_rules",
            "kept": len(existing), "passed_new": len(passed),
            "message": "无通过规则，evolved_rules.json 保持原样",
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_FILE.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "date": today(), "status": "deployed",
        "kept": len(kept), "passed_new": len(passed),
        "rules": [{"rule_id": r["rule_id"], "multiplier": r["multiplier"]} for r in kept],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="规则自进化引擎")
    ap.add_argument("--search", action="store_true", help="网格搜索 + 回测")
    ap.add_argument("--max-combos", type=int, default=200)
    ap.add_argument("--today", action="store_true", help="当日触发的已部署规则")
    ap.add_argument("--date", default=None)
    ap.add_argument("--weekly", action="store_true", help="周日全流程部署")
    args = ap.parse_args()
    if args.search:
        res = search_rules(max_combos=args.max_combos)
        print(f"{len(res)} 条候选, 通过 {sum(r['deployed'] for r in res)} 条")
        for r in res:
            if r["deployed"]:
                print(f"  ✅ {r['rule_id']} train_diff={r['train_diff']} oos_diff={r['oos_diff']} hits={r['hits']}")
    elif args.weekly:
        print(json.dumps(run_weekly(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(active_rules(args.date), ensure_ascii=False, indent=2))
