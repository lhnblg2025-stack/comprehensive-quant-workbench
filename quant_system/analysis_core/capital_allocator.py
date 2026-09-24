"""
capital_allocator — 双账户资金分配（V11 战略底仓 vs 战术机动）

S 账户（战略底仓, 默认70%): 中长线，估值分位驱动，月换手 ≤2 次
T 账户（战术机动, 默认30%): 短线，作战地图驱动，跟随情绪周期

融合输出: 最终仓位建议 = "S账户 XX% 仓位(标的) + T账户 XX% 仓位(标的)"

数据: valuation(PE分位) + battle_map(短线) + config 默认比例

用法:
  python3 -m quant_system.analysis_core.capital_allocator --date 2026-08-07 [--s-ratio 0.7]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CST = timezone(timedelta(hours=8))


def allocate(date: str | None = None, s_ratio: float = 0.7) -> dict:
    date = date or datetime.now(CST).date().isoformat()
    t_ratio = 1.0 - s_ratio

    # ── S 账户: 中长线（估值分位 < 30% 才开仓）──
    s_account = {"ratio": s_ratio, "mode": "估值驱动", "targets": [], "note": ""}
    try:
        from quant_system.analysis_core.macro_veto import _pe_percentile
        pct = _pe_percentile(date)
        s_account["market_pe_percentile"] = pct
        if pct is not None and pct < 0.3:
            s_account["note"] = f"沪深300 PE分位 {pct:.0%} < 30% → 底仓可加"
        elif pct is not None:
            s_account["note"] = f"沪深300 PE分位 {pct:.0%} ≥ 30% → 底仓不动/减"
        else:
            s_account["note"] = "PE分位数据不可用，底仓维持现状"
    except Exception as e:
        s_account["note"] = f"估值数据不可用: {str(e)[:60]}"

    # ── T 账户: 短线（作战地图）──
    t_account = {"ratio": t_ratio, "mode": "作战地图", "targets": [], "note": ""}
    bm_path = ROOT / "generated" / f"battle_map_{date}.json"
    if bm_path.exists():
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        t_account["targets"] = [{"name": g["name"], "leader": (g.get("leader") or {}).get("name"),
                                 "strategy": g["strategy"]}
                                for g in bm.get("attack_groups", {}).get("core", [])[:5]]
        t_account["position_range"] = bm.get("position_range")
        t_account["note"] = f"短线置信度 {bm.get('confidence')}, 建议 {bm.get('recommended')}"
    else:
        t_account["note"] = "无作战地图（先运行 battle_map）"

    res = {
        "date": date,
        "s_account": s_account,
        "t_account": t_account,
        "final_advice": (f"S账户({s_ratio:.0%})底仓: {s_account['note']} | "
                         f"T账户({t_ratio:.0%})机动: {t_account['note']}"),
    }
    out = ROOT / "generated" / f"capital_alloc_{date}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="双账户资金分配")
    ap.add_argument("--date", default=None)
    ap.add_argument("--s-ratio", type=float, default=0.7)
    args = ap.parse_args()
    r = allocate(args.date, args.s_ratio)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
