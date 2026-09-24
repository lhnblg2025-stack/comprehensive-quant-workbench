"""
predictions — 预测记录库与自动验证（V11 进化闭环）

铁律: 没有记录就没有进化。
所有推演层的预测必须入库，事后自动/手动验证，滚动统计分模块胜率，
低于阈值的模块自动降权（供决策层读取）。

schema:
  pred_id, created_at, horizon_days, target_type(指数/个股/题材/情绪),
  target, direction, probability, scenario_json, verify_vars_json,
  source_module, status(pending/correct/wrong/void), actual, verified_at, note

用法:
  python3 -m quant_system.analysis_core.predictions --add '{...json...}'
  python3 -m quant_system.analysis_core.predictions --verify '{pred_id, correct, actual}'
  python3 -m quant_system.analysis_core.predictions --report
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import PREDICTIONS  # noqa: E402

CST = timezone(timedelta(hours=8))

COLS = ["pred_id", "created_at", "horizon_days", "target_type", "target",
        "direction", "probability", "scenario_json", "verify_vars_json",
        "source_module", "status", "actual", "verified_at", "note"]


def _load() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(PREDICTIONS)


def _save(df: pd.DataFrame) -> None:
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLS].sort_values("created_at").reset_index(drop=True)
    df.to_parquet(PREDICTIONS, index=False)


def add_prediction(*, target_type: str, target: str, direction: str,
                   probability: float | None = None, horizon_days: int = 1,
                   scenario: dict | None = None, verify_vars: list | None = None,
                   source_module: str = "unknown", note: str = "") -> str:
    """新增一条预测，返回 pred_id。"""
    pred_id = f"P{datetime.now(CST).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    row = {
        "pred_id": pred_id,
        "created_at": datetime.now(CST).isoformat(),
        "horizon_days": horizon_days,
        "target_type": target_type,
        "target": target,
        "direction": direction,
        "probability": probability,
        "scenario_json": json.dumps(scenario or {}, ensure_ascii=False),
        "verify_vars_json": json.dumps(verify_vars or [], ensure_ascii=False),
        "source_module": source_module,
        "status": "pending",
        "actual": None,
        "verified_at": None,
        "note": note,
    }
    df = _load()
    new = pd.DataFrame([row])
    if df.empty:
        df = new
    else:
        df = pd.concat([df, new], ignore_index=True)
    _save(df)
    return pred_id


def verify(pred_id: str, correct: bool | None, actual: str | float | None = None,
           note: str = "") -> bool:
    """验证一条预测。correct=None 表示作废（如情景未触发）。"""
    df = _load()
    idx = df.index[df["pred_id"] == pred_id]
    if len(idx) == 0:
        print(f"[predictions] 未找到 {pred_id}")
        return False
    i = idx[0]
    df.loc[i, "status"] = "correct" if correct else ("void" if correct is None else "wrong")
    df.loc[i, "actual"] = str(actual) if actual is not None else None
    df.loc[i, "verified_at"] = datetime.now(CST).isoformat()
    if note:
        df.loc[i, "note"] = note
    _save(df)
    return True


def report() -> dict:
    df = _load()
    if df.empty:
        return {"total": 0}
    done = df[df["status"].isin(["correct", "wrong"])]
    res = {"total": len(df), "pending": int((df["status"] == "pending").sum()),
           "verified": len(done)}
    if len(done):
        res["overall_hit"] = round(float((done["status"] == "correct").mean()), 3)
    # 分模块胜率
    mod = []
    for m, g in done.groupby("source_module"):
        mod.append({"module": m, "n": len(g),
                    "hit": round(float((g["status"] == "correct").mean()), 3)})
    res["by_module"] = mod
    # 概率质量（Brier 简化版: 命中样本的 (1-p)^2 + 未命中 p^2）
    prob = done[done["probability"].notna() & done["probability"].notna()]
    if len(prob):
        p = prob["probability"].astype(float)
        y = (prob["status"] == "correct").astype(int)
        res["brier"] = round(float(((p - y) ** 2).mean()), 4)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="预测记录库")
    ap.add_argument("--add", type=str, default=None)
    ap.add_argument("--verify", type=str, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.add:
        d = json.loads(args.add)
        print("pred_id:", add_prediction(**d))
    if args.verify:
        d = json.loads(args.verify)
        verify(d["pred_id"], d.get("correct"), d.get("actual"), d.get("note", ""))
    if args.report:
        print(json.dumps(report(), ensure_ascii=False, indent=2))
