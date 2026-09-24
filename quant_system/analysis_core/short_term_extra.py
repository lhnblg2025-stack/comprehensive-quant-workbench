"""
short_term_extra — 短线引擎补全（V11 遗留任务）

M5 老龙断板 → 新龙卡位检测
M8 筹码结构（市值分布/封单比/换手）
M9 战术识别（打板/接力/龙头/反包/低吸/首板挖掘）

数据: zt_pool_history.parquet + zt_daily_stats.parquet + zt_pool_em_daily.parquet
用法:
  python3 -m quant_system.analysis_core.short_term_extra --all
  python3 -m quant_system.analysis_core.short_term_extra --breakout
"""

from __future__ import annotations
import logging

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_HISTORY  # noqa: E402


# ── M5 老龙断板 → 新龙卡位 ─────────────────────────────
def detect_breakout_takeover(k: int = 5) -> list[dict]:
    """近 k 个交易日的老龙断板 + 新龙卡位事件。

    断板: T日最高板 < T-1日最高板 且 T-1日最高板股票不在 T日涨停池
    新龙: 断板日/次日，同概念板块内新出现的首板/二板（卡位候选）
    """
    h = pd.read_parquet(ZT_HISTORY)
    h = h[h["is_zt"]].copy()
    # 概念映射（优先同花顺，回退东财）
    cm_file = MARKET_DIR / "concept_member_ths.parquet"
    if not cm_file.exists():
        cm_file = MARKET_DIR.parent / "classification" / "concept_member.parquet"
    if cm_file.exists():
        cm = pd.read_parquet(cm_file)
        cm["code"] = cm["code"].astype(str).str.zfill(6)
        code2concept: dict[str, set[str]] = {}
        for c, g in cm.groupby("code"):
            code2concept[c] = set(g["concept"])
    else:
        code2concept = {}

    dates = sorted(h["date"].unique())
    events = []
    for i in range(1, len(dates)):
        d_prev, d_cur = dates[i - 1], dates[i]
        if d_cur - d_prev > pd.Timedelta(days=5):
            continue
        prev_zt = h[h["date"] == d_prev]
        cur_zt = h[h["date"] == d_cur]
        prev_max = int(prev_zt["board_count"].max()) if len(prev_zt) else 0
        cur_max = int(cur_zt["board_count"].max()) if len(cur_zt) else 0
        if cur_max >= prev_max or prev_max < 3:
            continue  # 无断板
        # 老龙（T-1最高板股）
        old_leaders = prev_zt[prev_zt["board_count"] == prev_max]["code"].tolist()
        cur_codes = set(cur_zt["code"])
        broken = [c for c in old_leaders if c not in cur_codes]
        if not broken:
            continue
        # 新龙候选: 当日新晋首板/二板，与老龙同概念
        takeover = []
        for c in cur_zt["code"]:
            bc = int(cur_zt[cur_zt["code"] == c]["board_count"].iloc[0])
            if bc > 2:
                continue
            shared = set()
            for b in broken:
                shared |= (code2concept.get(c, set()) & code2concept.get(b, set()))
            if shared:
                takeover.append({"code": c, "board": bc, "shared_concepts": sorted(shared)[:3]})
        if takeover:
            events.append({
                "date": str(d_cur.date()),
                "old_leader_max": prev_max,
                "new_max": cur_max,
                "old_leaders": broken,
                "takeover_candidates": takeover[:5],
            })
    return events[-k:]


# ── M8 筹码结构 ─────────────────────────────────────────
def chip_structure() -> dict:
    """涨停股筹码画像: 市值分布 + 封单比 + 换手分布。"""
    h = pd.read_parquet(ZT_HISTORY)
    last = h["date"].max()
    day = h[(h["date"] == last) & h["is_zt"]].copy()
    out = {"date": str(last.date()), "zt_cnt": len(day)}
    # 流通市值: 优先 EM 池（真实值），回退 kline 估算
    em = MARKET_DIR / "zt_pool_em_daily.parquet"
    mv = None
    if em.exists():
        ed = pd.read_parquet(em)
        ed_last = ed["date"].max()
        z = ed[(ed["date"] == ed_last) & ed["is_zt"]]
        if len(z) and "float_mv" in z.columns:
            mv = pd.to_numeric(z["float_mv"], errors="coerce") / 1e8  # 元→亿
    if mv is None:
        mv = day["float_mv"].dropna()
    mv = mv.dropna()
    if len(mv):
        out["mv_dist"] = {
            "小盘<50亿": int((mv < 50).sum()),
            "中盘50-200亿": int(((mv >= 50) & (mv < 200)).sum()),
            "大盘≥200亿": int((mv >= 200).sum()),
        }
    # 封单比（EM 池: 封板资金/流通市值）
    if em.exists():
        ed = pd.read_parquet(em)
        ed_last = ed["date"].max()
        z = ed[(ed["date"] == ed_last) & ed["is_zt"]]
        if len(z) and "seal_fund" in z.columns:
            ratio = pd.to_numeric(z["seal_fund"], errors="coerce") / \
                    pd.to_numeric(z["float_mv"], errors="coerce").replace(0, np.nan)
            out["seal_ratio"] = {"强封(>5%)": int((ratio > 0.05).sum()),
                                 "中封(1-5%)": int(((ratio >= 0.01) & (ratio <= 0.05)).sum()),
                                 "弱封(<1%)": int((ratio < 0.01).sum())}
        if "turnover" in z.columns:
            tr = pd.to_numeric(z["turnover"], errors="coerce").dropna()
            out["turnover_mean"] = round(float(tr.mean()), 2)
    return out


# ── M9 战术识别 ─────────────────────────────────────────
def tactic_identify() -> dict:
    """当前市场主导战术（规则库 v1）。"""
    stats = pd.read_parquet(MARKET_DIR / "zt_daily_stats.parquet")
    r = stats.iloc[-1]
    zb_rate = r["zb_rate"] if pd.notna(r["zb_rate"]) else 0
    zt, mb = int(r["zt_cnt"]), int(r["max_board"])
    jr2 = r["jr2"] if pd.notna(r["jr2"]) else 0
    jr1 = r["jr1"] if pd.notna(r["jr1"]) else 0

    scores = {}
    # 打板: 封板率高 + 首板多 + 高度 2-3
    scores["打板"] = (1 - zb_rate) * 0.4 + (0.3 if 2 <= mb <= 3 else 0)
    # 接力: 2进3 晋级率高
    scores["接力"] = jr2 * 0.8 + (0.2 if zt >= 60 else 0)
    # 龙头: 高度≥5 + 梯队完整
    scores["龙头"] = (0.6 if mb >= 5 else 0) + (0.3 if zt >= 80 else 0)
    # 低吸: 炸板率高 + 跌停少
    scores["低吸"] = zb_rate * 0.5 + (0.3 if int(r["dt_cnt"]) <= 5 else 0)
    # 首板挖掘: 首板占比高
    try:
        import json as _json
        ladder = {int(k): v for k, v in _json.loads(r["ladder_json"]).items()}
        first_ratio = ladder.get(1, 0) / max(zt, 1)
        scores["首板挖掘"] = first_ratio * 0.7
    except Exception as e:
        logging.getLogger(__name__).error(f"[short_term_extra] 操作失败: {e}", exc_info=True)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return {"date": str(r["date"].date()),
            "tactics": [{"tactic": t, "score": round(s, 2)} for t, s in ranked],
            "dominant": ranked[0][0] if ranked else "观望",
            "note": {"zb_rate": round(float(zb_rate), 2), "zt": zt, "max_board": mb}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="短线补全 M5/M8/M9")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--breakout", action="store_true")
    ap.add_argument("--chip", action="store_true")
    ap.add_argument("--tactic", action="store_true")
    args = ap.parse_args()
    if args.breakout or args.all:
        print("[M5 老龙断板→新龙卡位]")
        print(json.dumps(detect_breakout_takeover(), ensure_ascii=False, indent=1)[:1200])
    if args.chip or args.all:
        print("\n[M8 筹码结构]")
        print(json.dumps(chip_structure(), ensure_ascii=False, indent=1))
    if args.tactic or args.all:
        print("\n[M9 战术识别]")
        print(json.dumps(tactic_identify(), ensure_ascii=False, indent=1))
    if not (args.all or args.breakout or args.chip or args.tactic):
        ap.print_help()
