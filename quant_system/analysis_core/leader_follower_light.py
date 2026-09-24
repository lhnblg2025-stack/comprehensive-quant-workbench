#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""龙头扩散 · 小拆散轻量版（2026-08-21 用户要求"把 leader_follower 做小拆散"）

原 leader_follower.run_today() 在云端死挂(>120s)，原因是全量概念扩散计算沉重。
本模块把能力拆为 3 个独立小函数，各自只读本地 parquet、秒级返回、可单独调用：

  1. zt_board_stats(date)     → 涨停池板高分布(高标/连板/最高板)
  2. concept_diffusion(date)  → 概念扩散表(轻量: 仅概念×涨停家数&最高板, 不做全扩散)
  3. leader_signal(date)      → 龙头信号(过热/独苗, 基于 diffusion 判定)

任何单卡不用跑全链；collectors 只调需要的函数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
ZT_EM_DAILY = ROOT / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"


def _latest_zt_day(max_rows: int = 400) -> Optional[object]:
    """读最新交易日涨停池（本地 parquet，秒级）。"""
    import pandas as pd
    if not ZT_EM_DAILY.exists():
        return None
    df = pd.read_parquet(ZT_EM_DAILY)
    df["date"] = pd.to_datetime(df["date"])
    day = df[df["date"] == df["date"].max()].copy()
    return day if len(day) else None


def zt_board_stats(date: str | None = None) -> dict:
    """涨停池板高分布（小函数①）：高标/连板/最高板/梯队断层。"""
    day = _latest_zt_day()
    if day is None or "board_count" not in day.columns:
        return {"error": "无涨停池数据", "zt_cnt": 0, "max_board": 0}
    bc = day["board_count"].fillna(0)
    boards = {int(k): int(v) for k, v in bc.value_counts().to_dict().items()}
    max_board = max(boards) if boards else 0
    zt_cnt = int(len(day))
    lianban = int((bc >= 2).sum())
    high = int((bc >= 3).sum())
    # 梯队断层: 2板→最高板之间空档
    gap = max_board >= 3 and any(k not in boards for k in range(2, max_board))
    return {
        "zt_cnt": zt_cnt, "max_board": max_board, "lianban": lianban,
        "high_board": high, "board_dist": boards, "ladder_gap": bool(gap),
    }


def concept_diffusion(date: str | None = None, top_n: int = 12) -> list[dict]:
    """概念/行业扩散（小函数②）：涨停行业×涨停家数&最高板。
    2026-08-21 修复: 原用 theme_cycle.load_concept_map 云端死挂(>150s);
    改用涨停池自带 industry 列(纯本地 parquet, 秒级, 零依赖)。"""
    day = _latest_zt_day()
    if day is None:
        return []
    if "industry" not in day.columns:
        return []
    rows: dict[str, dict] = {}
    for r in day.itertuples(index=False):
        ind = str(getattr(r, "industry", "") or "未知")
        bcnt = int(getattr(r, "board_count", 0) or 1)
        e = rows.setdefault(ind, {"concept": ind, "zt_cnt": 0, "max_board": 0})
        e["zt_cnt"] += 1
        e["max_board"] = max(e["max_board"], bcnt)
    out = sorted(rows.values(), key=lambda x: (-x["max_board"], -x["zt_cnt"]))[:top_n]
    for o in out:
        o["level"] = "主线" if (o["zt_cnt"] >= 4 or o["max_board"] >= 3) else "支线"
    return out


def leader_signal(date: str | None = None) -> dict:
    """龙头信号（小函数③）：过热/独苗/连板健康度。"""
    st = zt_board_stats(date)
    if "error" in st:
        return st
    max_board = st["max_board"]
    lianban = st["lianban"]
    high = st["high_board"]
    overheat = high > 4  # 高标拥挤示意
    alone = max_board >= 2 and st.get("board_dist", {}).get(max_board, 0) == 1
    # 信号文本
    if max_board >= 5:
        txt = "高度打开"
    elif max_board == 4:
        txt = "4板高度,延续观察"
    elif max_board <= 2:
        txt = "高度受限,无连板主线"
    else:
        txt = f"{max_board}板梯队"
    return {
        "max_board": max_board, "lianban": lianban, "high_board": high,
        "overheat": overheat, "alone_top": alone, "signal": txt,
        "note": f"涨停{st['zt_cnt']} 连板{lianban} 高标{high} 最高{max_board}板"
                + (" ⚠️高标拥挤" if overheat else "") + (" ⚠️断层" if st["ladder_gap"] else ""),
    }


def quick_summary(date: str | None = None) -> dict:
    """组合三小函数 → 一次性轻量龙头概况（替代 run_today）。"""
    return {
        "stats": zt_board_stats(date),
        "concepts": concept_diffusion(date, top_n=8),
        "signal": leader_signal(date),
    }


if __name__ == "__main__":
    import sys as _s
    _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=== zt_board_stats ===")
    print(zt_board_stats())
    print("=== concept_diffusion ===")
    for c in concept_diffusion()[:6]:
        print(f"  {c['concept']}: 涨停{c['zt_cnt']} 最高{c['max_board']}板 {c['level']}")
    print("=== leader_signal ===")
    print(leader_signal())