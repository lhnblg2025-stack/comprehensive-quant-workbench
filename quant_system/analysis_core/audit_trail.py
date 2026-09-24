"""
audit_trail — 决策留痕审计（V11 机构级可回溯）

每份作战地图/每笔模拟单/每次预警 → 追加一条不可变 JSONL 记录。
半年后回看: "当时看到了什么 → 基于什么 → 输出了什么 → 后来怎样"。

schema:
  ts, kind(battle_map/order/alert), date,
  inputs{emotion, regime, macro, confidence, resonance_top3},
  output{position, recommended, core_actions},
  user_feedback(预留: null/👍/👎), note

输出: generated/audit_trail.jsonl + query 接口

用法:
  python3 -m quant_system.analysis_core.audit_trail --log-battle-map --date 2026-08-07
  python3 -m quant_system.analysis_core.audit_trail --query 2026-08-07
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))
TRAIL = ROOT / "generated" / "audit_trail.jsonl"


def append(kind: str, date: str, inputs: dict, output: dict,
           user_feedback: str | None = None, note: str = "") -> None:
    rec = {
        "ts": datetime.now(CST).isoformat(),
        "kind": kind, "date": date,
        "inputs": inputs, "output": output,
        "user_feedback": user_feedback, "note": note,
    }
    TRAIL.parent.mkdir(parents=True, exist_ok=True)
    # 并发写保护（fcntl.flock）: 防止多进程交错写坏 JSONL
    try:
        import fcntl
        with open(TRAIL, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        with open(TRAIL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def log_battle_map(date: str | None = None) -> bool:
    """把当日作战地图 JSON 落成留痕记录。"""
    if date is None:
        date = datetime.now(CST).date().isoformat()
    p = ROOT / "generated" / f"battle_map_{date}.json"
    if not p.exists():
        print(f"[audit_trail] 无作战地图: {date}")
        return False
    bm = json.loads(p.read_text(encoding="utf-8"))
    append(
        kind="battle_map", date=date,
        inputs={
            "emotion": bm.get("emotion_stage"),
            "regime": bm.get("regime"),
            "macro": bm.get("macro_veto"),
            "confidence": bm.get("confidence"),
            "resonance_top3": [g["name"] for g in bm.get("attack_groups", {}).get("core", [])[:3]],
        },
        output={
            "position": bm.get("position_range"),
            "recommended": bm.get("recommended"),
            "core_actions": [g["name"] for g in bm.get("attack_groups", {}).get("core", [])[:5]],
            "risk_watch": [r["name"] for r in bm.get("risk_watch", [])[:3]],
        },
        note="auto",
    )
    print(f"[audit_trail] 已留痕 battle_map {date}")
    return True


def query(date: str | None = None, kind: str | None = None, limit: int = 20) -> list[dict]:
    if not TRAIL.exists():
        return []
    out = []
    for line in TRAIL.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception as e:
            logging.getLogger(__name__).error(f"[audit_trail] 操作失败: {e}", exc_info=True)
            continue
        if date and r.get("date") != date:
            continue
        if kind and r.get("kind") != kind:
            continue
        out.append(r)
    return out[-limit:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="决策留痕审计")
    ap.add_argument("--log-battle-map", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--query", default=None)
    args = ap.parse_args()
    if args.log_battle_map:
        log_battle_map(args.date)
    if args.query:
        rows = query(date=args.query)
        print(f"共 {len(rows)} 条留痕:")
        for r in rows:
            print(f"  [{r['ts'][:16]}] {r['kind']} {r['date']} | "
                  f"情绪{r['inputs'].get('emotion')} 宏观{r['inputs'].get('macro')} "
                  f"置信{r['inputs'].get('confidence')} | 仓位{r['output'].get('position')}")
