"""
theme_cycle — 题材生命周期（V11 短线 M4）

数据融合:
  zt_pool_history.parquet（全历史涨停，2020→今）× concept_member.parquet（code→概念）
  → 每个概念板块的每日涨停家数序列 → 阶段判定 + 主线/支线/一日游

五阶段: 启动 → 爆发 → 分歧 → 回流 → 退潮
判定规则（v1，随预测记录库校准）:
  启动: 板块涨停 0→≥2（首次出现）
  爆发: 涨停 ≥5 且连续增长
  分歧: 涨停环比下降 ≥40% 或 最高板断板
  回流: 分歧后 1-3 日内涨停回升
  退潮: 涨停归零 或 连续3日降至 ≤1

主线判定: 连续 ≥3 日涨停 ≥3 家；支线: 2日 2-4 家；一日游: 单日 ≥5 次日 ≤1

用法:
  python3 -m quant_system.analysis_core.theme_cycle --build [--days 30]
  python3 -m quant_system.analysis_core.theme_cycle --today
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_HISTORY  # noqa: E402

OUT = MARKET_DIR / "theme_cycle.parquet"
CONCEPT_MEMBER = MARKET_DIR.parent / "classification" / "concept_member.parquet"
CONCEPT_MEMBER_THS = MARKET_DIR.parent / "classification" / "concept_member_ths.parquet"
CONCEPT_BOARD = MARKET_DIR.parent / "classification" / "concept_board.parquet"

STAGE_CN = {"start": "启动", "burst": "爆发", "split": "分歧", "back": "回流", "ebb": "退潮"}

# 概念源选择: 默认同时消费东财 504 板块 + THS 361 细分概念（命名空间隔离防同名重复统计）
CONCEPT_SOURCES = ("em", "ths")
THS_PREFIX = "THS:"

# 泛化/交易属性板块（非题材，过滤掉）
# 泛化板块过滤统一复用 resonance_scorer 词表（2026-08-10 审计: 本地旧词表
# 漏掉 中报预增/央国企改革/机构重仓 等，导致泛化板块混入主线）
from quant_system.analysis_core.resonance_scorer import is_broad_board  # noqa: E402


@lru_cache(maxsize=1)
def _load_concept_map_cached() -> dict[str, list[str]]:
    """code → 概念板块列表（东财 BK + THS 双源合并，THS 概念带 'THS:' 前缀）。

    各源独立容错: 东财文件缺失/损坏不影响 THS，反之亦然。
    70884+ 行概念表被多模块整表重复读取，lru_cache 缓存原始 dict；返回值勿修改。
    """
    merged: dict[str, list[str]] = {}
    loaded: list[str] = []
    for src in CONCEPT_SOURCES:
        if src == "em":
            path, label = CONCEPT_MEMBER, "东财"
        else:
            path, label = CONCEPT_MEMBER_THS, "THS"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            df["code"] = df["code"].astype(str).str.zfill(6)
            for c, g in df.groupby("code"):
                cons = g["concept"].tolist()
                if src == "ths":
                    cons = [f"{THS_PREFIX}{x}" for x in cons]
                merged.setdefault(c, []).extend(cons)
            loaded.append(label)
        except Exception as e:  # noqa: BLE001
            print(f"[theme] {label} 概念源读取失败: {type(e).__name__}: {e}")
    if loaded:
        print(f"[theme] 概念源: {' + '.join(loaded)}")
    return merged


def load_concept_map() -> dict[str, list[str]]:
    """code → 概念板块列表（东财 BK + THS 双源合并，THS 概念带 'THS:' 前缀）。

    复用 _load_concept_map_cached 的 lru_cache；返回 dict 浅拷贝，
    避免调用方修改污染缓存（内层 list 仍为共享引用，请只读）。
    """
    return dict(_load_concept_map_cached())


def _stage_of(series: pd.Series, i: int) -> tuple[str, float]:
    """对某板块的涨停家数序列在 i 位置判阶段。返回 (阶段, 置信度)。"""
    v = series.iloc[i]
    prev = series.iloc[i - 1] if i >= 1 else 0
    p3 = series.iloc[i - 3:i].tolist() if i >= 3 else series.iloc[:i].tolist()

    if v == 0:
        if len([x for x in p3 if x >= 3]) >= 2:
            return "ebb", 0.8   # 曾经活跃后归零 → 退潮
        return "ebb", 0.4
    if v >= 5 and v > prev:
        # 爆发: 今日大爆发
        return "burst", 0.7
    if prev >= 5 and v < prev * 0.6:
        return "split", 0.75    # 环比下降 ≥40%
    if v >= 2 and prev <= 1:
        return "start", 0.6
    if i >= 2 and series.iloc[i - 2] >= 5 and v < series.iloc[i - 2] and v >= 2:
        return "back", 0.6      # 分歧后回升 → 回流
    if v <= 1 and len(p3) and max(p3) <= 1:
        return "ebb", 0.5
    return "split" if prev > v else ("burst" if v >= 3 else "start"), 0.3


def build_theme_cycle(days: int = 60, min_zt: int = 2) -> pd.DataFrame:
    """构建题材生命周期表（滚动: 只保留近 days 天 + 活跃板块）。"""
    if not ZT_HISTORY.exists():
        raise FileNotFoundError("先运行 zt_pool_history --reconstruct")
    # 与 ladder 共用“历史K线 + 东财三池最新尾部”数据口径，避免题材周期停在旧交易日。
    from quant_system.analysis_core.ladder import load_history
    zt = load_history()
    zt = zt[zt["is_zt"]].copy()

    cmap = load_concept_map()
    if not cmap:
        print("[theme] 无 concept_member / concept_member_ths，仅输出涨停池行业热度")
        return pd.DataFrame()

    # 展开: 每只涨停股 → 其所属概念（一个股可能属多板块）
    rows = []
    for _, r in zt.iterrows():
        for c in cmap.get(r["code"], []):
            rows.append((r["date"], c, r["board_count"]))
    long = pd.DataFrame(rows, columns=["date", "concept", "board_count"])

    # 每日每板块涨停家数 + 最高板
    daily = long.groupby(["date", "concept"]).agg(
        zt_cnt=("board_count", "size"),
        max_board=("board_count", "max"),
    ).reset_index()

    # 板块名映射
    board_names = {}
    board_pct = {}
    if CONCEPT_BOARD.exists():
        b = pd.read_parquet(CONCEPT_BOARD)
        board_names = dict(zip(b["board_code"], b["board_name"]))
        board_pct = dict(zip(b["board_code"], b["pct_chg"]))

    # 逐板块判定阶段（近 days 天）
    # 真实交易日对齐（2026-08-10 审计: freq="B" 会把节假日插入 zt_cnt=0 假行，阶段判定错位）
    from quant_system.analysis_core.config import ZT_DAILY_STATS
    try:
        _tcal = pd.to_datetime(pd.read_parquet(ZT_DAILY_STATS, columns=["date"])["date"]).unique()
    except Exception:
        _tcal = None
    out_rows = []
    for concept, g in daily.groupby("concept"):
        g = g.sort_values("date").reset_index(drop=True)
        if g["zt_cnt"].max() < min_zt:
            continue
        # 窗口: 以该概念最新活跃日为终点，往前取最近 days 个交易日
        # （2026-08-10 审计: 原实现 tail 截涨停日后再 reindex 全部交易日，窗口失效）
        last_td = g["date"].max()
        if _tcal is not None:
            dates = [d for d in _tcal if d <= last_td][-days:]
        else:
            dates = list(pd.date_range(g["date"].min(), g["date"].max(), freq="B"))[-days:]
        g = g[g["date"].isin(dates)]
        if g.empty:
            continue
        full = g.set_index("date").reindex(dates).fillna(0)
        full["max_board"] = full["max_board"].fillna(0).astype(int)
        s = full["zt_cnt"]
        for i in range(len(full)):
            st, conf = _stage_of(s, i)
            out_rows.append({
                "date": full.index[i], "concept": concept,
                "zt_cnt": int(s.iloc[i]), "max_board": int(full["max_board"].iloc[i]),
                "stage": st, "stage_cn": STAGE_CN[st], "confidence": round(conf, 2),
            })

    out = pd.DataFrame(out_rows)
    if len(out):
        out["board_name"] = out["concept"].map(board_names).fillna(out["concept"])
        out["board_pct"] = out["concept"].map(board_pct).fillna(np.nan)
        out = out[~out["board_name"].map(is_broad_board)]
        out["role"] = ""
        # role 只对最新交易日计算，但窗口用该概念最近 5 日（跨日期）
        # （2026-08-10 审计: 原实现把基于最后5日窗口的 role 赋给全部历史行 → 前视泄漏）
        last_date = out["date"].max()
        for concept, g in out.groupby("concept"):
            seq = g.sort_values("date")["zt_cnt"].tolist()
            n3 = sum(1 for x in seq[-5:-1] if x >= 3)
            if n3 >= 3:
                role = "主线"
            elif n3 >= 1 or (len(seq) >= 3 and seq[-2] >= 3):
                role = "支线"
            else:
                role = "一日游"
            out.loc[(out["date"] == last_date) & (out["concept"] == concept), "role"] = role
        out = out.sort_values(["date", "zt_cnt"], ascending=[True, False]).reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[theme] {len(out)} 行（近{days}日，活跃板块）→ {OUT}")
    return out


def today_themes(top_n: int = 10) -> dict:
    """当日题材热度榜 + 主线/支线/一日游判定。"""
    if not OUT.exists():
        raise FileNotFoundError("先运行 --build")
    df = pd.read_parquet(OUT)
    last = df["date"].max()
    day = df[df["date"] == last].sort_values("zt_cnt", ascending=False).head(top_n)

    # 主线/支线/一日游（基于近5日连续性）
    res = []
    recent = df[df["date"] >= last - pd.Timedelta(days=10)]
    for _, r in day.iterrows():
        g = recent[recent["concept"] == r["concept"]].sort_values("date")
        seq = g["zt_cnt"].tolist()
        n3 = len([x for x in seq[-4:-1] if x >= 3]) if len(seq) >= 4 else 0
        if n3 >= 3:
            role = "主线"
        elif n3 >= 1 or len(seq) >= 3 and seq[-2] >= 3:
            role = "支线"
        else:
            role = "一日游/新题材"
        res.append({"board": r["board_name"], "concept": r["concept"],
                    "zt_cnt": int(r["zt_cnt"]), "max_board": int(r["max_board"]),
                    "stage": r["stage_cn"], "role": role,
                    "board_pct": round(float(r["board_pct"]), 2) if pd.notna(r.get("board_pct")) else None})
    return {"date": str(last.date()), "themes": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="题材生命周期")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_theme_cycle(days=args.days)
    if args.today:
        print(json.dumps(today_themes(), ensure_ascii=False, indent=2))
    if not (args.build or args.today):
        ap.print_help()
