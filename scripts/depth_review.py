#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度复盘增强（2026-08-22 —— 用户提供25节深度模板，本模块实现可自动化章节）

用现有数据(涨停池/指数)产出模板要求的深度章节:
 1. zt_board_table()   → 涨停板块分布-有龙分级(模板1.9: 板块/涨停数/龙头梯队)
 2. zt_reason_keywords() → 涨停原因关键词Top(模板1.9③)
 3. core_emotion()     → 核心情绪指标(模板1.10: 空间龙头/封板率/温度档)
 4. breadth_stats()    → 量能结构(模板1.3: 涨跌家数/中位数/上涨占比)
 5. benchmark_vs()     → 基准指标AlphaBot阈值对比(模板1.6)
全部纯本地parquet, 秒级, 缺失降级。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MK = ROOT / "data_warehouse" / "market"


def _read_zt():
    import pandas as pd
    p = MK / "zt_pool_em_daily.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    d = df[df["date"] == df["date"].max()].copy()
    if not len(d):
        return None
    if "is_zt" in d.columns:
        d = d[d["is_zt"] == True]  # noqa: E712
    return d


def zt_board_table(top_n: int = 10) -> list[dict]:
    """涨停板块分布-有龙分级（板块/涨停数/最高板/龙头名）。"""
    d = _read_zt()
    if d is None or "industry" not in d.columns:
        return []
    rows = []
    for ind, g in d.groupby("industry"):
        bc = g["board_count"].fillna(0)
        # 龙头=最高板中首个(按封单时间/代码)
        leader = g.sort_values("board_count", ascending=False).iloc[0]
        rows.append({"板": ind, "涨停": int(len(g)), "最高板": int(bc.max()),
                     "龙头": str(leader.get("name", "")),
                     "龙头板数": int(leader.get("board_count", 0) or 0)})
    rows.sort(key=lambda x: (-x["最高板"], -x["涨停"]))
    return rows[:top_n]


def zt_reason_keywords(top_n: int = 8) -> list[str]:
    """热点关键词Top（涨停池无 reason 字段 → 用行业计数近似, 2026-08-22）。"""
    d = _read_zt()
    if d is None:
        return []
    from collections import Counter
    cnt = Counter()
    if "industry" in d.columns:
        for ind, g in d.groupby("industry"):
            cnt[ind] = len(g)
    return [w for w, _ in cnt.most_common(top_n)]


def core_emotion() -> dict:
    """核心情绪指标: 空间龙头/封板率/连板数/温度档。"""
    d = _read_zt()
    if d is None:
        return {}
    bc = d["board_count"].fillna(0)
    zt_n = int(len(d))
    zb_n = 0
    # 炸板数需要原始池含炸板
    p = MK / "zt_pool_em_daily.parquet"
    import pandas as pd
    raw = pd.read_parquet(p)
    raw["date"] = pd.to_datetime(raw["date"])
    day = raw[raw["date"] == raw["date"].max()]
    if "is_zb" in day.columns:
        zb_n = int(day["is_zb"].fillna(False).sum())
    seal_rate = round(zt_n / (zt_n + zb_n) * 100, 1) if (zt_n + zb_n) else 0
    max_b = int(bc.max()) if len(bc) else 0
    leader = d.sort_values("board_count", ascending=False).iloc[0]
    return {
        "zt": zt_n, "zb": zb_n, "seal_rate": seal_rate, "max_board": max_b,
        "space_leader": f'{leader.get("name","")}{max_b}板' if max_b else "-",
        "lianban": int((bc >= 2).sum()),
    }


def breadth_stats() -> dict:
    """量能结构: 涨跌家数(规范降级: 无全市场spot时用涨停池近似+标注)。"""
    try:
        p = MK / "spot.parquet"
        if not p.exists():
            # 降级: 涨停池=活跃家数(不利于直接当涨跌家数, 标注)
            d = _read_zt()
            zt = len(d) if d is not None else 0
            return {"up": zt, "down": 0, "median": None, "up_ratio": None,
                    "note": "全市场spot缺失, 用涨停家数近似(涨跌家数不可得)"}
        import pandas as pd
        sp = pd.read_parquet(p)
        up = int((sp["pct_chg"] > 0).sum())
        dn = int((sp["pct_chg"] < 0).sum())
        med = float(sp["pct_chg"].median())
        return {"up": up, "down": dn, "median": round(med, 2),
                "up_ratio": round(up / (up + dn) * 100, 1) if (up + dn) else 0}
    except Exception as e:  # noqa: BLE001
        return {"note": f"breadth失败: {str(e)[:50]}"}


def benchmark_vs() -> list[dict]:
    """基准指标阈值对比（AlphaBot简化阈值）。"""
    d = _read_zt()
    if d is None:
        return []
    import pandas as pd
    broad = breadth_stats()
    zt_n = len(d)
    up_n = broad.get("up", 0)
    _ratio = f"{zt_n/max(up_n,1)*100:.1f}%" if up_n else "N/A"
    out = [
        {"指标": "涨停占上涨比", "阈值": "4.4%", "实测": _ratio,
         "判定": "🟢" if (up_n and zt_n/max(up_n,1)*100 >= 4.4) else ("🔴" if up_n else "—")},
        {"指标": "空间高度", "阈值": "≥4板", "实测": f"{core_emotion().get('max_board','-')}板",
         "判定": "🟢" if core_emotion().get("max_board", 0) >= 4 else "🔴"},
        {"指标": "连板数", "阈值": "≥15", "实测": str(core_emotion().get("lianban", "-")),
         "判定": "🟢" if core_emotion().get("lianban", 0) >= 15 else "🔴"},
    ]
    return out


def chain_heat_stage(top_n: int = 6) -> list[dict]:
    """产业链热度阶段(模板1.7.6②): 板块涨停数+高度+炸板→导入/成长/成熟/过热。"""
    d = _read_zt()
    if d is None or "industry" not in d.columns:
        return []
    import pandas as pd
    # 拉原始含炸板
    raw = pd.read_parquet(MK / "zt_pool_em_daily.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    day = raw[raw["date"] == raw["date"].max()]
    out = []
    for ind, g in d.groupby("industry"):
        zt = int(len(g))
        bc = g["board_count"].fillna(0)
        mb = int(bc.max())
        zb = int(day[(day["industry"] == ind) & (day["is_zb"].fillna(False))].__len__()) if "is_zb" in day.columns else 0
        # 阶段判定
        if mb >= 4 and zt >= 5:
            stage = "过热(兑现期)" if zb > zt * 0.3 else "主升"
        elif mb == 3 and zt >= 3:
            stage = "成长中期"
        elif zt >= 4:
            stage = "成长期"
        elif zt >= 2 and mb >= 2:
            stage = "导入期"
        else:
            stage = "萌芽"
        out.append({"板块": ind, "涨停": zt, "最高板": mb, "炸板": zb, "阶段": stage})
    out.sort(key=lambda x: (-x["最高板"], -x["涨停"]))
    return out[:top_n]


def depth_block() -> dict:
    """组合: 深度数据(供HTML研报接入)。"""
    return {
        "zt_board": zt_board_table(),
        "reason_keywords": zt_reason_keywords(),
        "emotion": core_emotion(),
        "breadth": breadth_stats(),
        "benchmark": benchmark_vs(),
        "chain_stage": chain_heat_stage(),
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(depth_block(), ensure_ascii=False, indent=1)[:900])