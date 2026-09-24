#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合决策引擎 · 辅助（主线龙头个股提取，2026-08-21）

从涨停池按 (行业∩主线 + 连板数) 提取龙头个股，供融合选股打分使用。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MAIN_KEYWORDS = ["医药", "医疗", "生物", "基因", "疫苗", "创新药", "器械",
                  "半导体", "芯片", "AI", "算力", "机器人", "汽车", "军工", "储能",
                  "光伏", "消费", "食品", "软件", "通信", "电子", "低空", "数据"]


def mainline_leaders(blocks: dict, main_names: list, top_n: int = 6) -> list:
    """从涨停池提取主线板块的连板龙头个股。"""
    try:
        import pandas as pd
        _base = ROOT / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"
        if not _base.exists():
            return []
        df = pd.read_parquet(_base)
        df["date"] = pd.to_datetime(df["date"])
        d = df[df["date"] == df["date"].max()].copy()
        if d.empty or "industry" not in d.columns:
            return []
        # CRITICAL-3: 只保留当日真实涨停(is_zt=True)，炸板/跌停不得当龙头
        if "is_zt" in d.columns:
            d = d[d["is_zt"] == True]  # noqa: E712
        if d.empty:
            return []  # 主线当日无涨停 → 显式空(调用方提示)
        mains = [m for m in main_names if m and "无明确" not in m]
        if not mains:
            return []
        # MAJOR-8: 单方向行业匹配(industry 含关键词), 剔除过宽词
        _WIDE = {"AI", "国产", "自主", "新时代", "数字"}
        kw = []
        for m in mains:
            for k in m.split():
                kk = k.strip()
                if len(kk) >= 2 and kk not in _WIDE:
                    kw.append(kk)
            kw += [k for k in _MAIN_KEYWORDS if k in m and len(k) >= 2]
        kw = list({k for k in kw if len(k) >= 2})
        # 若主线名直接出现在行业内, 优先精确包含
        ind = d["industry"].astype(str)
        mask = ind.apply(lambda x: any(k in x for k in kw))
        med = d[mask].sort_values("board_count", ascending=False)
        if med.empty:
            nm = d["name"].astype(str)
            mask2 = nm.apply(lambda x: any(k in x for k in kw))
            med = d[mask2].sort_values("board_count", ascending=False)
        out = []
        for _, r in med.head(top_n).iterrows():
            out.append({
                "name": r.get("name"), "code": str(r.get("code", "")),
                "board": int(r.get("board_count") or 0), "industry": r.get("industry"),
            })
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[fusion_helpers] 龙头提取失败: {str(e)[:60]}")
        return []