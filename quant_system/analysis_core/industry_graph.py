"""
industry_graph — 产业链传导图谱（V11 联动观察引擎）

用真实数据构建板块关联图（networkx 语义，落盘为边表 parquet）:

  节点: 概念板块（东财 BK 代码）
  边:   双向，权重 = 涨停传导强度 lift

  lift(A→B) = P(B 次日活跃 | A 当日活跃) / P(B 活跃基准)
  (活跃 = 该概念当日涨停家数 ≥3)
  + 成分股重叠 Jaccard 作为辅助加固（lift 优先，重叠辅助）

  过滤: lift ≥ 2.2(MIN_LIFT) 且 共现样本 ≥ 10(MIN_SAMPLES) 且 与基准比有增量 → 保留
  (注: 审计 P2-8 对齐——docstring 原写 ≥1.8/≥5 已漂移, 实际代码常量为 2.2/10, 已同步)

应用（供 battle_map 联动观察栏）:
  - 主线板块 X 当日爆发 → 自动列出传导最强的 3-5 个关联板块（次日接力候选）
  - 个股异动 → 所属概念 → 同概念/传导板块的联动标的

输出: generated/industry_graph.parquet (边表) + industry_graph.json (图摘要)

用法:
  python3 -m quant_system.analysis_core.industry_graph --build      # 重建图（周更）
  python3 -m quant_system.analysis_core.industry_graph --links PCB  # 查询板块关联
  python3 -m quant_system.analysis_core.industry_graph --stock 600183
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.resonance_scorer import is_broad_board  # noqa: E402

THEME_CYCLE = MARKET_DIR / "theme_cycle.parquet"
CONCEPT_MEMBER = MARKET_DIR.parent / "classification" / "concept_member.parquet"
CONCEPT_MEMBER_THS = MARKET_DIR.parent / "classification" / "concept_member_ths.parquet"
EDGE_OUT = ROOT / "generated" / "industry_graph.parquet"
SUMMARY_OUT = ROOT / "generated" / "industry_graph.json"

MIN_SAMPLES = 10       # 共现样本下限
MIN_LIFT = 2.2        # 传导强度下限
ACTIVE_TH = 3         # 活跃阈值: 概念当日涨停 ≥3

# 概念源: 默认同时消费东财 504 板块 + THS 361 细分概念（THS 加 'THS:' 前缀隔离命名空间）
CONCEPT_SOURCES = ("em", "ths")
THS_PREFIX = "THS:"


def _concept_map() -> dict[str, list[str]]:
    """code → 概念列表（东财 BK + THS 双源合并，THS 概念带 'THS:' 前缀）。

    各源独立容错: 东财文件缺失/损坏不影响 THS，反之亦然。
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
            print(f"[industry_graph] {label} 概念源读取失败: {type(e).__name__}: {e}")
    if loaded:
        print(f"[industry_graph] 概念源: {' + '.join(loaded)}")
    return merged


def _concept_sets() -> dict[str, set[str]]:
    """concept → 成分代码集合（双源合并，THS 概念带 'THS:' 前缀）。"""
    sets: dict[str, set[str]] = {}
    for code, cons in _concept_map().items():
        for con in cons:
            sets.setdefault(con, set()).add(code)
    return sets

def build_graph() -> pd.DataFrame:
    tc = pd.read_parquet(THEME_CYCLE)
    tc["date"] = pd.to_datetime(tc["date"])
    # 过滤泛化板块
    tc = tc[~tc["board_name"].map(lambda x: is_broad_board(str(x)) if pd.notna(x) else True)]
    # 只保留至少出现过一次活跃的概念
    tc["active"] = tc["zt_cnt"].fillna(0) >= ACTIVE_TH

    concepts = sorted(tc["concept"].unique())
    name_map = _name_map()

    # 概念 × 日期 活跃矩阵
    pivot = tc.pivot_table(index="date", columns="concept", values="active", aggfunc="max").fillna(False)
    pivot = pivot.sort_index()
    dates = list(pivot.index)
    if len(dates) < 10:
        print(f"[industry_graph] 样本不足: {len(dates)} 个交易日")
        return pd.DataFrame()

    # 基准活跃概率 P(B)
    base_prob = pivot.mean()
    # A 活跃 → B 次日活跃 共现
    active_prev = pivot.iloc[:-1].values.astype(bool)
    active_next = pivot.iloc[1:].values.astype(bool)

    n = len(concepts)
    edges = []
    for i in range(n):
        a = concepts[i]
        a_act = active_prev[:, i]
        if a_act.sum() < MIN_SAMPLES:
            continue
        for j in range(n):
            if i == j:
                continue
            b = concepts[j]
            b_act_next = active_next[:, j]
            both = int(np.sum(a_act & b_act_next))
            if both < MIN_SAMPLES:
                continue
            pb = base_prob.get(b, 0.0)
            pa = a_act.mean()
            if pa <= 0 or pb <= 0:
                continue
            cond = both / (a_act.sum())
            lift = cond / pb
            if lift >= MIN_LIFT and cond > pb * 1.3:
                edges.append({
                    "src": a, "src_name": name_map.get(a, a),
                    "dst": b, "dst_name": name_map.get(b, b),
                    "lift": round(float(lift), 2),
                    "cond_prob": round(float(cond), 3),
                    "base_prob": round(float(pb), 3),
                    "samples": int(both),
                })

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        print("[industry_graph] 无有效传导边（样本不足或市场未形成传导）")
        return edge_df

    edge_df = edge_df.sort_values(["src", "lift"], ascending=[True, False]).reset_index(drop=True)

    # 成分重叠 Jaccard 辅助（2026-08-10 审计: 原在 to_parquet 之后计算从未落盘，
    # 且 >3000 边时长度不匹配崩溃 → 改为落盘前计算全部边）
    sets = _concept_sets()
    jacs = []
    for _, e in edge_df.iterrows():
        sa, sb = sets.get(e["src"], set()), sets.get(e["dst"], set())
        jacs.append(round(len(sa & sb) / len(sa | sb), 3) if sa and sb else 0.0)
    edge_df["overlap"] = jacs

    EDGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_parquet(EDGE_OUT, index=False)

    summary = {
        "nodes": len(concepts), "edges": len(edge_df),
        "trade_days": len(dates),
        "max_lift": float(edge_df["lift"].max()),
        "top_edges": edge_df.head(20)[["src_name", "dst_name", "lift", "samples"]].to_dict("records"),
        "note": "lift = P(次日活跃|当日活跃)/P(活跃基准); 活跃=概念涨停≥3家",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[industry_graph] 图构建完成: {len(concepts)} 节点 / {len(edge_df)} 边 / {len(dates)} 交易日")
    return edge_df


def _name_map() -> dict:
    """concept(BK) → 板块名称（concept_board 优先，theme_cycle 兜底）。"""
    nm: dict = {}
    cb = MARKET_DIR.parent / "classification" / "concept_board.parquet"
    if cb.exists():
        df = pd.read_parquet(cb)
        nm = dict(zip(df["board_code"], df["board_name"]))
    if not nm and THEME_CYCLE.exists():
        tc = pd.read_parquet(THEME_CYCLE).drop_duplicates("concept")
        nm = dict(zip(tc["concept"], tc["board_name"]))
    return nm


def build_graph_from_history(start: str = "2020-01-01") -> pd.DataFrame:
    """全量历史重建（zt_pool_history 长表 + concept_member）→ 更稳健的传导矩阵。

    向量化: M (T×N) 活跃矩阵, 共现 = M_prev.T @ M_next
    """
    from quant_system.analysis_core.config import ZT_HISTORY
    h = pd.read_parquet(ZT_HISTORY)
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["date"] >= pd.Timestamp(start)]
    zt = h[h["is_zt"]].copy()
    zt["code"] = zt["code"].astype(str).str.zfill(6)

    code2con = {c: set(cons) for c, cons in _concept_map().items()}

    # 每板块每日涨停家数
    recs = []
    for code, cons in code2con.items():
        sub = zt[zt["code"] == code]
        if sub.empty:
            continue
        d = sub.groupby("date").size()
        for con in cons:
            recs.append(pd.DataFrame({"date": d.index, "concept": con, "n": d.values}))
    if not recs:
        print("[industry_graph] 无成分映射数据")
        return pd.DataFrame()
    agg = pd.concat(recs, ignore_index=True).groupby(["date", "concept"]).sum().reset_index()
    agg["active"] = agg["n"] >= ACTIVE_TH
    pivot = agg.pivot_table(index="date", columns="concept", values="active", aggfunc="max").fillna(False)
    pivot = pivot.sort_index()
    M = pivot.values.astype(bool)
    concepts = list(pivot.columns)
    T, N = M.shape
    if T < 30:
        print(f"[industry_graph] 历史样本不足: {T} 交易日")
        return pd.DataFrame()

    A_prev, B_next = M[:-1], M[1:]
    both = A_prev.T.astype(int) @ B_next.astype(int)          # N×N 共现矩阵
    a_sum = A_prev.sum(0).astype(float)
    base = M.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = both / a_sum[:, None]
        lift = cond / base[None, :]
    lift = np.where((both >= MIN_SAMPLES) & (base > 0) & (a_sum[:, None] > 0), lift, 0)

    # 名称映射（concept_board 优先）
    name_map = _name_map()
    # 过滤: 泛化概念不建边（保留节点）
    keep = [c for c in concepts if not is_broad_board(str(name_map.get(c, "")))]
    keep_idx = [concepts.index(c) for c in keep]
    concepts = [concepts[i] for i in keep_idx]
    M = M[:, keep_idx]
    # 审计 2026-08-16：lift/cond/base/both 也必须按 keep_idx 重排，
    # 否则下面循环用过滤后索引访问原始矩阵 → 索引错位、输出错误边
    keep_arr = np.asarray(keep_idx, dtype=int)
    lift = lift[np.ix_(keep_arr, keep_arr)]
    cond = cond[np.ix_(keep_arr, keep_arr)]
    both = both[np.ix_(keep_arr, keep_arr)]
    base = base[keep_arr]
    N = len(concepts)
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j or lift[i, j] <= 0:
                continue
            if lift[i, j] < MIN_LIFT:
                continue
            if is_broad_board(str(name_map.get(concepts[j], ""))):
                continue
            edges.append({
                "src": concepts[i], "src_name": name_map.get(concepts[i], concepts[i]),
                "dst": concepts[j], "dst_name": name_map.get(concepts[j], concepts[j]),
                "lift": round(float(lift[i, j]), 2),
                "cond_prob": round(float(cond[i, j]), 3),
                "base_prob": round(float(base[j]), 3),
                "samples": int(both[i, j]),
            })
    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        return edge_df
    edge_df = edge_df.sort_values(["src", "lift"], ascending=[True, False]).reset_index(drop=True)
    # 成分重叠 Jaccard（全量重建同样需要 overlap 落盘，保证 links 输出一致）
    sets = _concept_sets()
    jacs = []
    for _, e in edge_df.iterrows():
        sa, sb = sets.get(e["src"], set()), sets.get(e["dst"], set())
        jacs.append(round(len(sa & sb) / len(sa | sb), 3) if sa and sb else 0.0)
    edge_df["overlap"] = jacs
    EDGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_parquet(EDGE_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps({
        "nodes": N, "edges": len(edge_df), "trade_days": T,
        "note": "全量历史重建", "start": start,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[industry_graph] 全量重建: {N} 节点 / {len(edge_df)} 边 / {T} 交易日")
    return edge_df


def _load_edges() -> pd.DataFrame:
    if not EDGE_OUT.exists():
        return pd.DataFrame()
    return pd.read_parquet(EDGE_OUT)


def links(concept_name: str, top_k: int = 8) -> list[dict]:
    """查询板块的传导关联（入向+出向合并）。"""
    edges = _load_edges()
    if edges.empty:
        return []
    mask = edges["src_name"].astype(str).str.contains(concept_name, na=False) | \
           edges["dst_name"].astype(str).str.contains(concept_name, na=False)
    sub = edges[mask]
    out = []
    for _, e in sub.sort_values("lift", ascending=False).head(top_k).iterrows():
        out.append({
            "src": e["src_name"], "dst": e["dst_name"],
            "lift": e["lift"], "samples": e["samples"],
            "overlap": e.get("overlap", 0.0),
        })
    return out


def stock_links(stock_code: str, top_k: int = 8) -> dict:
    """个股联动: 所属概念 → 传导关联板块 → 关联个股样例。"""
    cons = _concept_map().get(stock_code.zfill(6), [])
    if not cons:
        return {"stock": stock_code, "concepts": [], "links": []}
    name_map = {}
    tc = pd.read_parquet(THEME_CYCLE)
    tc = tc.drop_duplicates("concept")
    name_map = dict(zip(tc["concept"], tc["board_name"]))

    edges = _load_edges()
    linked: list[dict] = []
    if not edges.empty:
        for con in cons:
            mask = (edges["src"] == con) | (edges["dst"] == con)
            for _, e in edges[mask].sort_values("lift", ascending=False).head(5).iterrows():
                other = e["dst"] if e["src"] == con else e["src"]
                linked.append({
                    "concept": name_map.get(other, other),
                    "lift": e["lift"], "samples": e["samples"],
                })
    return {
        "stock": stock_code,
        "concepts": [name_map.get(c, c) for c in cons[:6]],
        "links": linked[:top_k],
    }


def print_links(concept_name: str) -> str:
    ls = links(concept_name)
    if not ls:
        return f"板块 '{concept_name}' 无传导关联（图未构建或传导弱）"
    lines = [f"🔗 产业链传导关联: {concept_name}", "-" * 50]
    for x in ls:
        lines.append(f"  {x['src']} → {x['dst']}  lift={x['lift']} (样本{x['samples']}, 重叠{x['overlap']:.0%})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="产业链传导图谱")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--links", default=None)
    ap.add_argument("--stock", default=None)
    ap.add_argument("--rebuild-history", action="store_true", help="用 zt_pool_history 全量重建")
    args = ap.parse_args()
    if args.build:
        build_graph()
    elif args.rebuild_history:
        build_graph_from_history()
    elif args.links:
        print(print_links(args.links))
    elif args.stock:
        print(json.dumps(stock_links(args.stock), ensure_ascii=False, indent=2))
    else:
        print("用法: --build | --links 板块名 | --stock 代码")
