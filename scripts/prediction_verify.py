#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测验证闭环（2026-08-22 融合升级 —— 修复"进化闭环死亡"）

daily_review_chain 只入库预测、从不验证（101 pending / 0 verified）→ 校准权重恒空。
本模块自动回填：
- 取全部 pending 的 target_type=market_trend 预测（target=被预测的交易日）
- 用沪深300(index_daily.parquet) 找 target 的下一交易日收盘 → ret = close(next)/close(target)-1
- direction up/down/flat（阈 ±0.3%）对照实际 → predictions.verify() 回填 status/actual

用法:
  python3 scripts/prediction_verify.py            # 自动验证全部可验证 pending
  python3 scripts/prediction_verify.py --date 2026-08-21   # 指定参考日(一般为复盘日)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))

logger = logging.getLogger("prediction_verify")
CST = timezone(timedelta(hours=8))

# 判定阈值：|ret| <= FLAT_EPS 记 flat
FLAT_EPS = 0.003

_INDEX_PATH = ROOT / "data_warehouse" / "market" / "index_daily.parquet"


def _load_index_returns() -> tuple[dict, list]:
    """沪深300 日线 → {date: next_trade_date} 与 {date: ret(next/cur)-1}。"""
    import pandas as pd
    df = pd.read_parquet(_INDEX_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    close = df["close"].astype(float).tolist()
    nxt_map: dict[str, str] = {}
    ret_map: dict[str, float] = {}
    for i in range(len(dates) - 1):
        nxt_map[dates[i]] = dates[i + 1]
        if close[i]:
            ret_map[dates[i]] = round((close[i + 1] / close[i]) - 1.0, 4)
    return nxt_map, ret_map


def _judge(direction: str, ret: float) -> bool:
    d = str(direction or "").strip().lower()
    if d in ("up", "多", "乐观", "进攻"):
        return ret > FLAT_EPS
    if d in ("down", "空", "悲观", "清仓", "防守"):
        return ret < -FLAT_EPS
    if d in ("flat", "震荡", "中性"):
        return abs(ret) <= FLAT_EPS
    return False  # 未知方向：不作废也不判对（保守）


def void_stale_non_trend(days: int = 7) -> int:
    """作废无法自动验证的陈旧预测（target 为描述文本、非交易日期的类型）。

    情绪/指数描述类预测（target="明日情绪阶段"等）无法对照指数涨跌自动验证，
    留置 pending 会永久污染 pending 数、拉低校准信任——超期直接作废(void)，
    让 pending 队列诚实反映"待下一个交易日验证"的市场趋势预测。
    返回作废条数。
    """
    from quant_system.analysis_core import predictions  # noqa: PLC0415
    from datetime import timedelta
    df = predictions._load()  # noqa: SLF001
    pend = df[df["status"] == "pending"]
    mask = pend["target_type"] != "market_trend"
    if mask.any():
        cutoff = pd.Timestamp.now(tz=CST) - timedelta(days=days)
        pend = pend[mask]
        if len(pend):
            created = pd.to_datetime(pend["created_at"])
            old = pend[created < cutoff]
            n = 0
            for _, row in old.iterrows():
                try:
                    predictions.verify(row["pred_id"], None, note="描述类目标不可自动验证, 超期作废")
                    n += 1
                except Exception:  # noqa: BLE001
                    continue
            return n
    return 0


def verify_pending(review_date: str | None = None, limit: int = 0,
                   void_stale: bool = True) -> dict:
    """验证全部可验证的 pending market_trend 预测。

    review_date: 复盘日（不参与判定，仅用于日志/产物命名）。
    void_stale: 同时作废超期(>7天)不可自动验证的描述类预测，保持队列诚实。
    返回 {verified_n, correct_n, hit, by_module, skipped_n, note}
    """
    from quant_system.analysis_core import predictions  # noqa: PLC0415
    voided = 0
    if void_stale:
        try:
            voided = void_stale_non_trend()
        except Exception as e:  # noqa: BLE001
            logger.warning("作废陈旧预测失败: %s", str(e)[:80])
    nxt_map, ret_map = {}, {}
    try:
        nxt_map, ret_map = _load_index_returns()
    except Exception as e:  # noqa: BLE001
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": 0, "voided_n": voided, "note": f"index_daily 不可用: {str(e)[:80]}"}

    # 2026-08-22 稳定化: 结构检查前置（先确认预测库可读再统计，防缺列渗透到 report()）
    raw_df = None
    try:
        raw_df = predictions._load()  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": 0, "voided_n": voided, "note": f"预测库损坏: {str(e)[:80]}"}
    if raw_df is None or raw_df.empty:
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": 0, "voided_n": voided, "note": "预测库为空"}
    if not {"status", "target_type", "pred_id"}.issubset(raw_df.columns):
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": 0, "voided_n": voided, "note": "预测库结构异常(缺列/空)"}

    rep = predictions.report()
    if not rep.get("total"):
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": 0, "voided_n": voided, "note": "预测库为空"}

    df = raw_df
    pending_n = int((df["status"] == "pending").sum())
    pend = df[(df["status"] == "pending") & (df["target_type"] == "market_trend")].copy()
    if pend.empty:
        return {"verified_n": 0, "correct_n": 0, "hit": None, "by_module": [],
                "skipped_n": pending_n,
                "voided_n": voided, "note": "无 market_trend pending（已全部闭环）"}

    verified_n = correct_n = 0
    missed: list[str] = []
    for _, row in pend.iterrows():
        tgt = str(row.get("target") or "")[:10]
        nxt = nxt_map.get(tgt)
        if not nxt or tgt not in ret_map:
            missed.append(tgt)
            continue
        ret = ret_map[tgt]
        ok = _judge(row.get("direction"), ret)
        try:
            predictions.verify(row["pred_id"], ok, actual=ret,
                               note=f"沪深300 {tgt}→{nxt} {ret:+.2%}")
            verified_n += 1
            correct_n += 1 if ok else 0
        except Exception as e:  # noqa: BLE001
            logger.warning("验证 %s 失败: %s", row["pred_id"], str(e)[:80])

    done = predictions.report()
    by_module = done.get("by_module", [])
    hit = done.get("overall_hit")
    note = f"本次验证 {verified_n} 条" + (f"（跳过 {len(missed)} 条无下一交易日: {missed[:3]}…）" if missed else "")
    return {
        "verified_n": verified_n, "correct_n": correct_n,
        "hit": round(correct_n / verified_n, 3) if verified_n else None,
        "by_module": by_module,
        "total": done.get("total"), "pending": done.get("pending"), "brier": done.get("brier"),
        "skipped_n": len(missed), "voided_n": voided, "note": note,
    }


def render_verify_md(v: dict) -> str:
    if not v:
        return ""
    L = ["## 📈 预测验证闭环"]
    note = v.get("note") or ""
    if not v.get("verified_n"):
        L.append(f"- {note}（pending 剩余 {v.get('pending', 0)}）")
        if v.get("by_module"):
            pass
        return "\n".join(L)
    L.append(f"- 本次验证 **{v['verified_n']}** 条 · 判对 {v.get('correct_n', 0)} 条 · 总命中率 **{v.get('hit')}**（Brier {v.get('brier', '-')}）")
    L.append(f"- pending 剩余: {v.get('pending', 0)} / 累计 {v.get('total', 0)}")
    mods = v.get("by_module") or []
    if mods:
        L.append("- 分模块命中率:")
        for m in mods:
            warn = " ⚠️降权" if m["n"] >= 5 and m["hit"] < 0.5 else ""
            L.append(f"  - {m['module']}: {m['hit']}（n={m['n']}）{warn}")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    date = None
    args = sys.argv[1:]
    if "--date" in args:
        date = args[args.index("--date") + 1]
    r = verify_pending(date)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(render_verify_md(r))