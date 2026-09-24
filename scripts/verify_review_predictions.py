#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信号后验闭环 · 次日验证（2026-08-21）

对照次日指数实际涨跌，验证融合链前一日入库的 market_trend 预测：
  up   → 次日指数收涨  → correct；否则 wrong
  down → 次日指数收跌  → correct；否则 wrong
  flat → 次日指数±0.3%内 → correct；否则 wrong

用法:
  python3 scripts/verify_review_predictions.py [--date YYYY-MM-DD]  # 默认验证截至昨日所有 pending
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("verify_review_predictions")
CST = timezone(timedelta(hours=8))


def _load_preds() -> list[dict]:
    from quant_system.analysis_core import predictions  # noqa: PLC0415
    df = predictions._load()
    return df.to_dict("records") if df is not None and len(df) else []


def _next_day_close(target_date: str) -> float | None:
    """下一次易日 index_daily close（作为实际涨跌基准）。返回次日收盘 vs 当日收盘的涨跌%。"""
    import pandas as pd
    p = ROOT / "data_warehouse" / "market" / "index_daily.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    t = pd.Timestamp(target_date)
    after = df[df["date"] > t]
    if not len(after):
        return None
    nxt = after.iloc[0]
    # 找当日收盘（若无当日，用前一个收盘）
    same = df[df["date"] <= t]
    if not len(same):
        return None
    cur = same.iloc[-1]
    if cur["close"] == 0 or pd.isna(cur["close"]):
        return None
    return float((nxt["close"] / cur["close"] - 1) * 100)  # 次日涨跌%


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="验证 target_date<=该日的 pending 预测（默认昨日）")
    args = ap.parse_args()

    from quant_system.analysis_core import predictions  # noqa: PLC0415
    preds = _load_preds()
    pending = [p for p in preds if p.get("status") == "pending"
               and p.get("target_type") == "market_trend"]
    if not pending:
        print("无待验证预测")
        return 0

    ok = wrong = void = 0
    for p in pending:
        t = str(p.get("target", ""))[:10]
        if args.date and t > args.date:
            continue
        # 次日涨跌
        nxt = _next_day_close(t)
        if nxt is None:
            continue  # 次日数据未出，留待下次
        d = p.get("direction")
        correct = None
        if d == "up":
            correct = nxt > 0
        elif d == "down":
            correct = nxt < 0
        else:  # flat
            correct = abs(nxt) <= 0.3
        pid = p.get("pred_id")
        predictions.verify(pid, correct, actual=round(nxt, 3),
                           note=f"次日涨跌{nxt:+.2f}%")
        if correct:
            ok += 1
        else:
            wrong += 1
        logger.info("[verify] %s %s→%s 次日%+.2f%% %s",
                    pid, t, d, nxt, "✅" if correct else "❌")

    total = ok + wrong
    hit = f"{ok/total:.0%}" if total else "N/A"
    print(f"\n验证完成: 正确 {ok} / 错误 {wrong}（命中率 {hit}）")
    if total:
        r = predictions.report()
        logger.info("全库命中率: %s", r.get("overall_hit"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())