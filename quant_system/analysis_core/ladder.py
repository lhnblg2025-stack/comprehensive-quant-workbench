"""
ladder — 连板天梯计算器（V11 W1-2）

从 zt_pool_history.parquet（K线重建长表）计算每日：
  - 涨停/炸板/跌停家数（含 ST 口径 + 去 ST 口径）
  - 连板天梯分布（1板..N板）+ 断层标记
  - 炸板率 + 3日趋势
  - 四档晋级率（1进2/2进3/3进4/高位）+ 3日趋势
  - 昨日涨停今日溢价（premium）+ 亏钱效应（大面家数）
  - 涨停股成交额合计（量能代理）
  - 最高板

输出: zt_daily_stats.parquet（每日一行，情绪周期/推演层的地基）

用法:
  python3 -m quant_system.analysis_core.ladder --build
  python3 -m quant_system.analysis_core.ladder --latest
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

from quant_system.analysis_core.config import ZT_HISTORY, ZT_DAILY_STATS  # noqa: E402


def load_history() -> pd.DataFrame:
    """读取历史涨停长表，并合并东财三池最新尾部，防止天梯停在旧交易日。"""
    if not ZT_HISTORY.exists():
        raise FileNotFoundError(f"先运行 zt_pool_history --reconstruct: {ZT_HISTORY}")
    history = pd.read_parquet(ZT_HISTORY)
    em_path = ROOT / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"
    if not em_path.exists():
        return history
    try:
        em = pd.read_parquet(em_path)
        required = {"date", "code", "board_count", "is_zt", "is_zb", "is_dt", "is_st"}
        if not required.issubset(em.columns):
            return history
        # EM三池是短线精确源：重叠日期以EM为准，补足历史长表缺失的最新交易日。
        cols = [c for c in history.columns if c in em.columns]
        tail = em[cols].copy()
        for c in history.columns:
            if c not in tail.columns:
                tail[c] = np.nan
        tail = tail[history.columns]
        combined = pd.concat([history, tail], ignore_index=True)
        combined = combined.drop_duplicates(["date", "code"], keep="last")
        return combined.sort_values(["date", "code"]).reset_index(drop=True)
    except Exception:
        return history


def build_daily_stats(recent_days: int | None = None) -> pd.DataFrame:
    """构建每日聚合统计。

    recent_days=None → 全量重建；recent_days=N → 滚动增量：
    只重算最近 N 个交易日并拼接历史，旧行自动裁掉。
    """
    h = load_history()
    h = h.sort_values(["date", "code"]).reset_index(drop=True)
    dates = pd.Index(h["date"].unique())

    zt = h[h["is_zt"]].copy()
    zb = h[h["is_zb"]].copy()
    dt = h[h["is_dt"]].copy()

    # ── 涨停集合（次日视角）: code -> board_count
    zt_next = zt[["date", "code", "board_count"]].rename(
        columns={"date": "date_n", "board_count": "board_n"})
    zt_next["date_n"] = zt_next["date_n"].shift(-1)  # 占位，下面按日期对齐
    # 用 merge_asof 太慢，改用字典: date -> {code: board_count}
    zt_map: dict[pd.Timestamp, dict[str, int]] = {}
    for d, g in zt.groupby("date"):
        zt_map[d] = dict(zip(g["code"], g["board_count"]))

    rows = []
    for d in dates:
        d_next = dates[dates.get_loc(d) + 1] if dates.get_loc(d) + 1 < len(dates) else None
        zt_d = zt[zt["date"] == d]
        zb_d = zb[zb["date"] == d]
        dt_d = dt[dt["date"] == d]

        zt_cnt = len(zt_d)
        zt_cnt_nost = int(zt_d["is_st"].eq(False).sum())
        zb_cnt = len(zb_d)
        dt_cnt = len(dt_d)
        dt_cnt_nost = int(dt_d["is_st"].eq(False).sum())

        max_board = int(zt_d["board_count"].max()) if zt_cnt else 0
        ladder: dict[int, int] = {}
        for n, c in zt_d["board_count"].value_counts().items():
            ladder[int(n)] = int(c)

        zb_rate = zb_cnt / (zt_cnt + zb_cnt) if (zt_cnt + zb_cnt) > 0 else np.nan

        # 溢价/亏钱效应：T 日涨停股在 T+1 日的表现（next_pct 即 T+1 涨幅）
        nxt = zt_d["next_pct"].dropna()
        premium = float(nxt.mean()) if len(nxt) else np.nan
        big_loss = int((nxt < -5.0).sum()) if len(nxt) else 0

        # 晋级率: T日 board==n 的涨停股，在 T+1 是否仍涨停且 board==n+1
        jr = {}
        if d_next is not None:
            next_map = zt_map.get(d_next, {})
            for n in [1, 2, 3]:
                pool = zt_d.loc[zt_d["board_count"] == n, "code"]
                denom = len(pool)
                hit = sum(1 for c in pool if next_map.get(c) == n + 1)
                jr[f"jr{n}"] = hit / denom if denom else np.nan
            hi_pool = zt_d.loc[zt_d["board_count"] >= 4, "code"]
            hi_denom = len(hi_pool)
            hi_hit = sum(1 for c in hi_pool if next_map.get(c, 0) >= 5)
            jr["jr_high"] = hi_hit / hi_denom if hi_denom else np.nan
        else:
            jr = {"jr1": np.nan, "jr2": np.nan, "jr3": np.nan, "jr_high": np.nan}

        rows.append({
            "date": d,
            "zt_cnt": zt_cnt, "zt_cnt_nost": zt_cnt_nost,
            "zb_cnt": zb_cnt, "dt_cnt": dt_cnt, "dt_cnt_nost": dt_cnt_nost,
            "max_board": max_board,
            "ladder_json": json.dumps(ladder, ensure_ascii=False),
            "zb_rate": zb_rate,
            "premium": premium,
            "big_loss_cnt": big_loss,
            "zt_amount": float(zt_d["amount"].sum()) if zt_cnt else 0.0,
            **jr,
        })

    out = pd.DataFrame(rows)
    # 3日趋势列
    for col in ["zb_rate", "premium", "jr1", "jr2", "jr3", "jr_high", "zt_cnt", "max_board"]:
        out[f"{col}_t3"] = out[col].rolling(3).mean()

    # 滚动增量：保留历史 + 只替换最近 recent_days 天
    if recent_days is not None and ZT_DAILY_STATS.exists():
        old = pd.read_parquet(ZT_DAILY_STATS)
        first_new = out["date"].iloc[-recent_days]
        old = old[old["date"] < first_new]
        out = pd.concat([old, out.tail(recent_days)], ignore_index=True)

    out.to_parquet(ZT_DAILY_STATS, index=False)
    print(f"[ladder] 完成: {len(out)} 个交易日 → {ZT_DAILY_STATS}")
    return out


def latest(n: int = 1, as_of: str | None = None) -> pd.DataFrame:
    if not ZT_DAILY_STATS.exists():
        raise FileNotFoundError("先运行 ladder --build")
    df = pd.read_parquet(ZT_DAILY_STATS)
    if as_of:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)]
    return df.tail(n)


def print_latest() -> None:
    df = latest(5).copy()
    for _, r in df.iloc[::-1].iterrows():
        ladder = {int(k): int(v) for k, v in json.loads(r["ladder_json"]).items()}
        ladder_str = " | ".join(f"{k}板:{v}" for k, v in sorted(ladder.items(), reverse=True))
        print(f"{r['date'].date()} 涨停{r['zt_cnt']}(去ST{r['zt_cnt_nost']}) 炸板{r['zb_cnt']} "
              f"跌停{r['dt_cnt']} 最高{r['max_board']}板 炸板率{r['zb_rate']:.1%} "
              f"溢价{r['premium']:.2f}% 1进2:{r['jr1']:.0%} 2进3:{r['jr2']:.0%}")
        print(f"   天梯: {ladder_str}")
        print(f"   断层: {gap_of(ladder)}")


def gap_of(ladder: dict[int, int]) -> str:
    """断层标记: 从最高板往下找第一个空缺。"""
    if not ladder:
        return "无涨停"
    top = max(ladder)
    gaps = [n for n in range(2, top) if n not in ladder]
    return f"{top}板下断层@{gaps}" if gaps else "无断层"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="连板天梯计算器")
    ap.add_argument("--build", action="store_true", help="全量重建")
    ap.add_argument("--update", type=int, default=0, help="滚动增量：只重算最近 N 个交易日")
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_daily_stats()
    if args.update:
        build_daily_stats(recent_days=args.update)
    if args.latest or not (args.build or args.update):
        print_latest()
