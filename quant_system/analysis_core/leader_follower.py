"""
leader_follower — 龙头-跟风扩散度监控（V11 短线）

监控板块内"龙头-跟风"结构:
  - 跟风股大面积涨停（扩散度>0.6）且龙头非最高板梯队 → 跟风过热，龙头见顶风险
  - 扩散度<0.2 且龙头>=3板 → 独苗行情，退潮前兆
  - 涨停梯队断层（2板扎堆但3板断层）→ 接力意愿弱，写入 reason

数据源（全部本地，禁止网络）:
  data_warehouse/market/zt_pool_em_daily.parquet    当日东财三池（is_zt/board_count/name/code/industry）
  data_warehouse/classification/concept_member.parquet  code→概念（theme_cycle.load_concept_map，东财+THS 双源）
  data_warehouse/classification/concept_board.parquet   官方 leader_name/leader_code（board_code/board_name）

判定:
  活跃概念 = 当日涨停 >= 3 家；DataFrame 同时保留 1-2 家的临界概念
  （无跟风/无 leader → diffusion=None、signal=正常，不参与信号）
  扩散度 = 跟风涨停家数 / 龙头连板数，归一 0-1（超过 1 截断为 1）

用法:
  python3 -m quant_system.analysis_core.leader_follower --today
  python3 -m quant_system.analysis_core.leader_follower --date 2026-08-07
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR, ZT_EM_DAILY  # noqa: E402

# 泛化/交易属性板块（融资融券/深股通/昨日涨停等）非题材，过滤掉，避免噪声刷屏
# （与 theme_cycle 一致，统一复用 resonance_scorer 词表）
from quant_system.analysis_core.resonance_scorer import is_broad_board  # noqa: E402
from quant_system.analysis_core.theme_cycle import load_concept_map  # noqa: E402

CONCEPT_MEMBER = MARKET_DIR.parent / "classification" / "concept_member.parquet"
CONCEPT_BOARD = MARKET_DIR.parent / "classification" / "concept_board.parquet"

ACTIVE_MIN_ZT = 3      # 活跃概念: 当日涨停 >= 3 家
OVERHEAT_DIFF = 0.6    # 扩散度 > 0.6 且龙头非最高板梯队 → 过热
ALONE_DIFF = 0.2       # 扩散度 < 0.2 且龙头 >= 3 板 → 独苗
ALONE_MIN_BOARD = 3    # 独苗判定所需龙头最低连板数

_COLS = [
    "concept", "board_name", "leader", "leader_boards", "zt_cnt", "follower_cnt",
    "diffusion", "signal", "max_board", "board_dist", "ladder_gap", "active", "stocks",
]


# ── 数据装载 ─────────────────────────────────────────────
def _resolve_date(date: str | None = None) -> str:
    """解析日期，None → 涨停池最新交易日。"""
    if date is not None:
        return str(pd.Timestamp(date).date())
    if not ZT_EM_DAILY.exists():
        return str(pd.Timestamp.today().date())
    zt = pd.read_parquet(ZT_EM_DAILY, columns=["date"])
    if zt.empty:
        return str(pd.Timestamp.today().date())
    return str(pd.to_datetime(zt["date"]).max().date())


def _load_zt_day(day: str) -> pd.DataFrame:
    """当日涨停池（仅 is_zt），code 统一 6 位。"""
    if not ZT_EM_DAILY.exists():
        return pd.DataFrame()
    zt = pd.read_parquet(ZT_EM_DAILY)
    zt = zt[pd.to_datetime(zt["date"]).dt.strftime("%Y-%m-%d") == day]
    if zt.empty:
        return zt
    # 2026-08-21 审计: is_zt 加 fillna(False) 守卫——一旦与历史表合并出现 NaN，
    # bool(nan)=True 且 Series 含 NaN 掩码会抛 ValueError（emotion_system 已守卫，此处对齐）。
    zt = zt[zt["is_zt"].fillna(False)].copy()
    zt["code"] = zt["code"].astype(str).str.zfill(6)
    return zt


def _board_map(day: str) -> pd.DataFrame:
    """概念板块官方 leader 快照，按 board_code 索引。

    2026-08-21 审计修复: 原实现无当日快照时回退"最新快照"，会把未来日期的
    leader 归属回填历史(--date 分析过去时泄露未来龙头，前视)。改为只取 ts<=day
    的最近快照；无 ≤day 快照则返回空表（leader 置 None，不做未来回填）。
    """
    b = pd.read_parquet(CONCEPT_BOARD)
    if "ts" in b.columns:
        ts = pd.to_datetime(b["ts"]).dt.strftime("%Y-%m-%d")
        avail = b[ts <= day]
        if len(avail):
            b = avail[ts == avail["ts"].max()] if "ts" in avail.columns else avail
        else:
            return pd.DataFrame()
    return b.drop_duplicates("board_code").set_index("board_code")


def _code2concepts() -> dict[str, list[str]]:
    """code → 所属概念列表（一个股可属多概念）。

    复用 theme_cycle.load_concept_map（lru_cache 缓存整表读取；东财 BK + THS
    双源合并，THS 概念带 'THS:' 前缀）。概念集比原单东财源更大，属预期增强。
    """
    return load_concept_map()


# ── 信号判定 ─────────────────────────────────────────────
def _signal(diffusion: float | None, leader_boards: int | None, max_board: int) -> str:
    if diffusion is None or leader_boards is None:
        return "正常"                                   # 无 leader / 无跟风，不参与信号
    if diffusion > OVERHEAT_DIFF and leader_boards < max_board:
        return "过热"                                   # 跟风过热，龙头见顶风险
    if diffusion < ALONE_DIFF and leader_boards >= ALONE_MIN_BOARD:
        return "独苗"                                   # 独苗行情，退潮前兆
    return "健康"


def _severity(r) -> float:
    """异常程度分: 过热 > 独苗；同信号内扩散度越极端越靠前。"""
    if r.signal == "过热":
        return 3.0 + float(r.diffusion)                                       # 3.6 ~ 4.0
    if r.signal == "独苗":
        return 2.0 + (ALONE_DIFF - float(r.diffusion)) + min(int(r.leader_boards), 9) * 0.05
    return 0.0


def _reason(r) -> str:
    parts = []
    if r.signal == "过热":
        parts.append(
            f"跟风过热：{r.follower_cnt}家跟风涨停、扩散度{r.diffusion:.2f}，"
            f"龙头{r.leader}({r.leader_boards}板)未进最高板梯队（概念最高{r.max_board}板）"
        )
    elif r.signal == "独苗":
        parts.append(
            f"独苗行情：龙头{r.leader}({r.leader_boards}板)仅{r.follower_cnt}家跟风、"
            f"扩散度{r.diffusion:.2f}，退潮前兆"
        )
    if r.ladder_gap:
        dist = "、".join(f"{k}板×{v}" for k, v in r.board_dist.items() if v)
        parts.append(f"涨停梯队断层({dist})，接力意愿弱")
    return "；".join(parts)


# ── 对外接口 ─────────────────────────────────────────────
def diffusion_index(date: str | None = None) -> pd.DataFrame:
    """当日各概念扩散度表: concept/board_name/leader/leader_boards/zt_cnt/
    follower_cnt/diffusion/signal + max_board/board_dist/ladder_gap/active/stocks。"""
    if not (ZT_EM_DAILY.exists() and CONCEPT_MEMBER.exists() and CONCEPT_BOARD.exists()):
        return pd.DataFrame(columns=_COLS)
    day = _resolve_date(date)
    zt = _load_zt_day(day)
    if zt.empty:
        return pd.DataFrame(columns=_COLS)
    board = _board_map(day)
    code2concepts = _code2concepts()

    # concept → 当日涨停股
    concept_zt: dict[str, list[dict]] = {}
    for r in zt.itertuples(index=False):
        bcnt = int(r.board_count) if pd.notna(r.board_count) else 1
        for c in code2concepts.get(r.code, []):
            concept_zt.setdefault(c, []).append({"code": r.code, "name": r.name, "boards": bcnt})

    rows = []
    for concept, stocks in concept_zt.items():
        zt_cnt = len(stocks)
        stocks_sorted = sorted(stocks, key=lambda s: (-s["boards"], s["code"]))
        max_board = stocks_sorted[0]["boards"]
        dist: dict[int, int] = {}
        for s in stocks:
            dist[s["boards"]] = dist.get(s["boards"], 0) + 1
        board_dist = {k: dist[k] for k in sorted(dist)}
        ladder_gap = any(k in dist and dist[k] > 0 and (k + 1) not in dist
                         for k in range(2, max_board))

        b = board.loc[concept] if concept in board.index else None
        board_name = concept if b is None else b["board_name"]
        if b is not None and is_broad_board(board_name):
            continue  # 泛化/交易属性板块，不参与龙头-跟风扩散度
        leader_code = None
        leader_name = None
        if b is not None:
            if pd.notna(b["leader_code"]):
                leader_code = str(b["leader_code"]).zfill(6)
            if pd.notna(b["leader_name"]):
                leader_name = str(b["leader_name"])

        leader = None
        if leader_code is not None:
            by_code = {s["code"]: s for s in stocks}
            if leader_code in by_code:                    # 官方龙头当日涨停 → 直接作龙头
                hit = by_code[leader_code]
                leader = {"code": leader_code, "name": leader_name or hit["name"], "boards": hit["boards"]}
            else:                                         # 官方龙头未涨停 → 概念内最高板股
                leader = dict(stocks_sorted[0])

        leader_boards = leader["boards"] if leader else None
        follower_cnt = zt_cnt - 1 if leader else 0
        diffusion = None
        if leader_boards and follower_cnt > 0:
            diffusion = round(min(follower_cnt / leader_boards, 1.0), 3)   # 归一 0-1

        if leader:
            rest = [s for s in stocks_sorted if s["code"] != leader["code"]]
            ordered = [leader] + rest
        else:
            ordered = stocks_sorted
        stocks_list = [{"code": s["code"], "name": s["name"], "boards": s["boards"]} for s in ordered]

        rows.append({
            "concept": concept,
            "board_name": board_name,
            "leader": leader["name"] if leader else None,
            "leader_boards": leader_boards,
            "zt_cnt": zt_cnt,
            "follower_cnt": follower_cnt,
            "diffusion": diffusion,
            "signal": _signal(diffusion, leader_boards, max_board),
            "max_board": max_board,
            "board_dist": board_dist,
            "ladder_gap": ladder_gap,
            "active": zt_cnt >= ACTIVE_MIN_ZT,
            "stocks": stocks_list,
        })
    return pd.DataFrame(rows, columns=_COLS)


def top_signals(date: str | None = None, n: int = 10) -> list[dict]:
    """按异常程度排序的信号: {concept, signal, reason, stocks}。"""
    df = diffusion_index(date)
    if df.empty or n <= 0:
        return []
    abn = df[df["signal"].isin(("过热", "独苗"))].copy()
    if abn.empty:
        return []
    abn["severity"] = abn.apply(_severity, axis=1)
    abn = abn.sort_values(["severity", "leader_boards"], ascending=False).head(n)
    return [
        {"concept": r.concept, "signal": r.signal, "reason": _reason(r), "stocks": r.stocks}
        for r in abn.itertuples(index=False)
    ]


def _record(r) -> dict:
    return {
        "concept": r.concept,
        "board_name": r.board_name,
        "leader": r.leader,
        "leader_boards": r.leader_boards,
        "zt_cnt": int(r.zt_cnt),
        "follower_cnt": int(r.follower_cnt),
        "diffusion": r.diffusion,
        "signal": r.signal,
        "max_board": int(r.max_board),
        "board_dist": r.board_dist,
        "ladder_gap": bool(r.ladder_gap),
        "active": bool(r.active),
        "stocks": r.stocks,
    }


def run_today() -> dict:
    """汇总当日扩散度，存 generated/leader_follower_{date}.json。"""
    day = _resolve_date(None)
    df = diffusion_index(day)
    if df.empty:
        res = {"date": day, "active_concepts": 0, "overheat_cnt": 0, "alone_cnt": 0,
               "top_signals": [], "diffusion": []}
    else:
        res = {
            "date": day,
            "active_concepts": int(df["active"].sum()),
            "overheat_cnt": int((df["signal"] == "过热").sum()),
            "alone_cnt": int((df["signal"] == "独苗").sum()),
            "top_signals": top_signals(day),
            "diffusion": [_record(r) for r in df.itertuples(index=False)],
        }
    out = ROOT / "generated" / f"leader_follower_{day}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="龙头-跟风扩散度监控")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认最新交易日")
    ap.add_argument("--top", type=int, default=10, help="top_signals 数量")
    args = ap.parse_args()
    if args.date:
        _df = diffusion_index(args.date)
        print(_df.to_string(index=False))
        print("\ntop_signals:\n" + json.dumps(top_signals(args.date, args.top),
                                              ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_today(), ensure_ascii=False, indent=2))
