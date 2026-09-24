"""
game_theory_model — 游资博弈论建模（V11）

把打板/接力/锁仓的"游资博弈"形式化为博弈矩阵，全部用本地历史数据校准，
零网络零 LLM。两阶段:

  阶段1 calibrate_matrix()  全历史博弈矩阵校准
    参与方: 打板客(首板入场)、接力客(二板/三板入场)、锁仓客(连板龙头持有至断板)、砸盘方(获利兑现)
    全部参数从 zt_daily_stats / zt_pool_history 实际历史收益分布估计:
      - 首板成功率: 首板封板/炸板 次日溢价均值与胜率（分封板/炸板）
      - 接力成功率: 二板→三板、三板→四板晋级率（分板块强弱: 涨停家数高于/低于中位数）
      - 锁仓收益:  连板龙头(≥3连板)从首板到断板累计收益分布
      - 砸盘损失:  断板日平均回撤
    输出: generated/game_matrix.json

  阶段2 current_game()     当日格局映射
    当日状态(最高板/涨停家数/炸板率/晋级率/昨日涨停溢价) → 历史最近似 20 个交易日
    的后续 3 日实际表现 → 输出当前最优策略 {strategy, ev, win_rate, confidence, reason}
    窗口内锁仓口径: 当日最高板龙头买入持有至断板的实际收益（避免"只看最终成龙者"的幸存偏差）

收益口径（与 zt_pool_history.next_pct 一致，即 T+1 收盘涨跌幅，作为"次日开盘溢价"的近似）:
  - 打板: 当日买入首板封板股，次日卖出（收益=次日溢价，近似用 next_pct）
  - 接力: 当日买入二板/三板封板股，次日卖出
  - 锁仓: 首板收盘买入连板龙头，持有至断板日收盘（cum_ret=∏(1+next_pct/100)-1）
  - 空仓: 等待，EV=0
策略池剔除 ST（游资惯例，ST 涨跌幅机制不同）。

用法:
  from quant_system.analysis_core.game_theory_model import calibrate_matrix, current_game, run_report
  calibrate_matrix()        # 阶段1 → 博弈矩阵（带 generated/game_matrix.json 缓存）
  current_game()            # 阶段2 → 当前建议
  run_report("2026-08-07")  # 汇总 → generated/game_theory_{date}.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import ZT_DAILY_STATS, ZT_HISTORY  # noqa: E402

CST = timezone(__import__("datetime").timedelta(hours=8))

MIN_TRADING_DAYS = 100          # 数据不足阈值
SIMILAR_DAYS = 20               # 相似日数量
HOLD_MIN_BOARDS = 3             # "连板龙头"最低连板数
FWD_WINDOW = 3                  # 后续观察窗口（交易日）
CACHE = ROOT / "generated" / "game_matrix.json"
FEATURES = ["zt_cnt", "max_board", "zb_rate", "jr1", "premium"]

_cache: dict = {}


def _load(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 zt_daily_stats / zt_pool_history（模块级缓存，避免重复读盘）。"""
    if not force and _cache.get("loaded"):
        return _cache["stats"], _cache["hist"]
    if not ZT_DAILY_STATS.exists() or not ZT_HISTORY.exists():
        raise FileNotFoundError(f"数据缺失: {ZT_DAILY_STATS} / {ZT_HISTORY}")
    stats = pd.read_parquet(ZT_DAILY_STATS)
    hist = pd.read_parquet(
        ZT_HISTORY,
        columns=["date", "code", "board_count", "is_zt", "is_zb", "is_st", "next_pct"],
    )
    stats["date"] = pd.to_datetime(stats["date"])
    hist["date"] = pd.to_datetime(hist["date"])
    _cache.update(loaded=True, stats=stats, hist=hist)
    return stats, hist


def _streak_engine(hist: pd.DataFrame, start: pd.Timestamp
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 zt_pool_history 回溯连板序列（引擎）。

    序列规则: 同一 code 的涨停行，若后一行 board_count == 前一行+1 视为连板延续；
    board_count==1（或前一行未晋级）即为新序列起点。

    返回:
      seg  — 序列级: code/start/end/boards/n_days/cum_ret(首板收盘→断板日收盘,%)/break_ret/closed
      rows — 行级(涨停行): date/code/board_count/fwd_ret(该行收盘买入→断板日收盘,%，在途为NaN)/closed
    """
    h = hist[hist["date"] >= start].sort_values(["code", "date"]).reset_index(drop=True)
    zt = h[h["is_zt"]].drop_duplicates(subset=["date", "code"]).copy()
    empty_seg = pd.DataFrame(columns=["code", "start", "end", "boards", "n_days",
                                      "cum_ret", "break_ret", "closed"])
    if zt.empty:
        return empty_seg, zt

    g = zt.groupby("code", sort=False)
    same_code = g["date"].shift(-1).notna()
    nxt_board = g["board_count"].shift(-1)
    zt["promoted"] = (same_code & (nxt_board == zt["board_count"] + 1)).values
    prev_prom = zt["promoted"].astype(float).pipe(
        lambda s2: s2.groupby(zt["code"], sort=False).shift(1)).fillna(0.0).astype(bool).values
    is_start = (zt["board_count"] == 1) | (~prev_prom)
    zt["streak_id"] = is_start.cumsum()

    # 序列内累计收益: cumprod(1+next_pct/100)，末行 next_pct 即断板日/下一个涨停日收益
    w = (zt["next_pct"].fillna(0.0) / 100.0 + 1.0)
    cs = w.groupby(zt["streak_id"]).cumprod()
    last = zt.groupby("streak_id").tail(1).set_index("streak_id")

    seg = pd.DataFrame({
        "code": zt.groupby("streak_id")["code"].first(),
        "start": zt.groupby("streak_id")["date"].first(),
        "end": last["date"],
        "boards": last["board_count"],
        "n_days": zt.groupby("streak_id")["board_count"].count(),
        "cum_ret": (cs.groupby(zt["streak_id"]).last() - 1.0) * 100.0,
        "break_ret": last["next_pct"],
        "closed": last["next_pct"].notna(),  # 数据末端仍在途的序列不可观测
    }).reset_index(drop=True)

    # 行级: 从该行收盘买入持有至断板日收盘的收益 = 序列总乘积 / 该行之前乘积 - 1
    cp_last = cs.groupby(zt["streak_id"]).last().reindex(zt["streak_id"]).values
    cp_prev = cs.groupby(zt["streak_id"]).shift(1).fillna(1.0)
    zt["fwd_ret"] = (cp_last / cp_prev - 1.0) * 100.0
    closed_map = last["next_pct"].notna()  # Series indexed by streak_id
    zt["closed"] = zt["streak_id"].map(closed_map).fillna(False).astype(bool)
    zt.loc[~zt["closed"], "fwd_ret"] = np.nan
    zt["days_to_break"] = (zt.groupby("streak_id")["date"].transform("size") -
                           zt.groupby("streak_id").cumcount())
    zt.loc[~zt["closed"], "days_to_break"] = np.nan
    rows = zt[["date", "code", "board_count", "fwd_ret", "days_to_break",
               "closed"]].reset_index(drop=True)
    return seg, rows


def _build_streaks(hist: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """序列级连板回溯（校准用），见 _streak_engine。"""
    seg, _ = _streak_engine(hist, start)
    return seg


def _summarize(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"n": 0, "ev": None, "win_rate": None, "avg_ret": None}
    return {
        "n": int(len(r)),
        "ev": round(float(r.mean()), 3),
        "win_rate": round(float((r > 0).mean()), 3),
        "avg_ret": round(float(r.mean()), 3),
    }


def _jr_split(stats: pd.DataFrame) -> dict:
    """晋级率分板块强弱（涨停家数 ≥ 中位数 = 强）。"""
    j = stats.dropna(subset=["jr2", "jr3"]).copy()
    out = {"overall": {"n": 0, "jr2": None, "jr3": None}}
    if j.empty:
        return out
    med = float(j["zt_cnt"].median())
    out["overall"] = {"n": int(len(j)),
                      "jr2": round(float(j["jr2"].mean()), 3),
                      "jr3": round(float(j["jr3"].mean()), 3)}
    for lbl, cond in [("strong", j["zt_cnt"] >= med), ("weak", j["zt_cnt"] < med)]:
        s = j[cond]
        out[lbl] = {"n": int(len(s)),
                    "jr2": round(float(s["jr2"].mean()), 3),
                    "jr3": round(float(s["jr3"].mean()), 3)}
    return out


def _source_meta() -> dict:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in (ZT_HISTORY, ZT_DAILY_STATS)}


def calibrate_matrix(start: str = "2024-01-01", use_cache: bool = True) -> dict:
    """阶段1: 全历史博弈矩阵校准（含 generated/game_matrix.json 缓存，源数据变化才重算）。

    start: 校准窗口起点。默认 2024-01-01（近期生态）；传 "2020-01-01" 可全历史。
    """
    start_ts = pd.Timestamp(start)
    meta = _source_meta()

    if use_cache and CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if (cached.get("meta") == meta and
                    pd.Timestamp(cached.get("window_start", "1900-01-01")) == start_ts and
                    cached.get("status") == "ok"):
                return cached
        except Exception:
            CACHE.unlink(missing_ok=True)

    stats, hist = _load()
    st = stats[stats["date"] >= start_ts].copy()
    if len(st) < MIN_TRADING_DAYS:
        return {"status": "insufficient_data", "days": int(len(st)),
                "reason": f"可用交易日 {len(st)} < {MIN_TRADING_DAYS}"}

    zt = hist[(hist["date"] >= start_ts) & (hist["is_zt"]) & (~hist["is_st"])]

    # ── 打板: 首板封板股次日收益；炸板首板仅作对照（次日回撤）──
    sealed = zt[zt["board_count"] == 1]["next_pct"]
    broken = hist[(hist["date"] >= start_ts) & (hist["is_zb"]) &
                  (hist["board_count"] == 1) & (~hist["is_st"])]["next_pct"]
    touches = len(sealed.dropna()) + len(broken.dropna())
    seal_rate = len(sealed.dropna()) / touches if touches else None
    first_board = _summarize(sealed)
    first_board["sealed"] = _summarize(sealed)
    first_board["broken"] = _summarize(broken)
    first_board["seal_rate"] = round(float(seal_rate), 3) if seal_rate else None

    # ── 接力: 二板/三板封板股次日收益 + 晋级率（分强弱）──
    relay = _summarize(zt[zt["board_count"].isin([2, 3])]["next_pct"])
    relay["jr"] = _jr_split(st)

    # ── 锁仓: 连板龙头(≥3板) 首板→断板累计收益分布 ──
    seg = _build_streaks(hist, start_ts)
    hold_pool = seg[(seg["boards"] >= HOLD_MIN_BOARDS) & seg["closed"] & seg["cum_ret"].notna()]
    hold = _summarize(hold_pool["cum_ret"])
    if hold["n"]:
        hold["median_ret"] = round(float(hold_pool["cum_ret"].median()), 3)
        hold["p25"] = round(float(hold_pool["cum_ret"].quantile(0.25)), 3)
        hold["p75"] = round(float(hold_pool["cum_ret"].quantile(0.75)), 3)
        hold["hold_days_avg"] = round(float(hold_pool["n_days"].mean()), 2)
        hold["per_day_ev"] = round(float((hold_pool["cum_ret"] / hold_pool["n_days"]).mean()), 3)
    else:
        hold.update({"median_ret": None, "p25": None, "p75": None,
                     "hold_days_avg": None, "per_day_ev": None})

    # ── 砸盘: 断板日平均回撤（所有已断板序列）──
    brk = seg[seg["closed"] & seg["break_ret"].notna()]
    smash = _summarize(brk["break_ret"])

    # ── 主导策略: 按可比口径（打板/接力为单日，锁仓折算每持有日，空仓=0）──
    candidates = {
        "first_board": first_board["ev"],
        "relay": relay["ev"],
        "hold": hold.get("per_day_ev"),
        "cash": 0.0,
    }
    valid = {k: v for k, v in candidates.items() if v is not None}
    dominant = max(valid, key=valid.get) if valid else None

    out = {
        "status": "ok",
        "calibrated_at": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"),
        "window_start": str(start_ts.date()),
        "window_end": str(st["date"].max().date()),
        "days": int(len(st)),
        "meta": meta,
        "first_board": first_board,
        "relay": relay,
        "hold": hold,
        "smash": smash,
        "dominant": dominant,
        "note": ("EV=次日溢价均值(%)，next_pct口径; 策略池剔除ST; "
                 "锁仓EV为首板收盘持有至断板收盘累计收益; 主导按锁仓单日EV可比口径"),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _daily_ev_series(hist: pd.DataFrame, start: pd.Timestamp) -> dict[str, pd.Series]:
    """按日聚合 打板/接力 的次日溢价均值，供阶段2窗口快速查询。"""
    zt = hist[(hist["date"] >= start) & (hist["is_zt"]) & (~hist["is_st"])]
    fb = zt[zt["board_count"] == 1].groupby("date")["next_pct"].mean()
    relay = zt[zt["board_count"].isin([2, 3])].groupby("date")["next_pct"].mean()
    return {"fb": fb, "relay": relay}


def _window_ev(days: pd.Index, pos: int, daily: dict[str, pd.Series],
               rows: pd.DataFrame, win: int = FWD_WINDOW) -> dict:
    """相似日 d 的后续 win 个交易日窗口内各策略实际 EV（%）。

    fb/relay: 窗口内每日策略次日溢价均值再取平均；
    hold: 窗口内每个交易日"当日最高板龙头"买入持有至断板的实际收益
    （累计 hold / 单日 hold_pd / 持有天数 hold_days；无在途数据则 None）；
    cash: 恒 0。
    """
    if pos + 1 >= len(days):
        return {"fb": None, "relay": None, "hold": None, "hold_pd": None,
                "hold_days": None, "cash": 0.0}
    w_dates = days[pos + 1: pos + 1 + win]
    out: dict = {"cash": 0.0}
    for k in ("fb", "relay"):
        vals = daily[k].reindex(w_dates).dropna()
        out[k] = round(float(vals.mean()), 3) if len(vals) else None
    if rows is not None and len(rows):
        rets, pds, dts = [], [], []
        for d in w_dates:
            g = rows[rows["date"] == d]
            if g.empty:
                continue
            mb = int(g["board_count"].max())
            sel = g.loc[(g["board_count"] == mb) & g["fwd_ret"].notna()]
            if len(sel):
                rets.append(float(sel["fwd_ret"].mean()))
                pds.append(float((sel["fwd_ret"] / sel["days_to_break"]).mean()))
                dts.append(float(sel["days_to_break"].mean()))
        out["hold"] = round(float(np.mean(rets)), 3) if rets else None
        out["hold_pd"] = round(float(np.mean(pds)), 3) if pds else None
        out["hold_days"] = round(float(np.mean(dts)), 2) if dts else None
    return out


def current_game(date: str | None = None) -> dict:
    """阶段2: 当日格局 → 历史相似日后续3日表现 → 最优策略建议。

    相似度: [涨停家数, 最高板, 炸板率, 1进2晋级率, 昨日涨停溢价] 缺失用3日均值回填，
    历史z-score标准化后欧氏距离，取前 SIMILAR_DAYS 个。
    建议策略: 各相似日窗口内实际 EV 最高的策略（空仓=0），按跨相似日平均 EV 取优。
    """
    stats, hist = _load()
    if date is None:
        date = str(stats["date"].max().date())
    target = pd.Timestamp(date)
    if target not in set(stats["date"]):
        return {"status": "insufficient_data", "reason": f"交易日不存在: {date}"}

    st = stats[stats["date"] <= target].copy()
    if len(st) < MIN_TRADING_DAYS:
        return {"status": "insufficient_data", "days": int(len(st)),
                "reason": f"可用交易日 {len(st)} < {MIN_TRADING_DAYS}"}

    start_ts = st["date"].iloc[0]
    for c in ("zb_rate", "premium", "jr1"):
        # 末行 _t3 可能仍为 NaN（rolling 窗口内有效值不足），再 ffill 兜底
        st[c] = st[c].fillna(st[f"{c}_t3"]).ffill()
    feat = st[FEATURES].replace([np.inf, -np.inf], np.nan)
    mu, sd = feat.mean(), feat.std().replace(0.0, 1.0)
    norm = (feat - mu) / sd

    row = norm.loc[st["date"] == target].iloc[0]
    dist = ((norm - row) ** 2).sum(axis=1).pow(0.5)
    mask = st["date"] < target
    similar = pd.Series(dist[mask].values, index=st.loc[mask, "date"]).nsmallest(SIMILAR_DAYS)
    if len(similar) == 0:
        return {"status": "insufficient_data", "reason": "无历史相似日"}

    days = pd.Index(stats["date"].sort_values())
    pos_map = {d: i for i, d in enumerate(days)}
    daily = _daily_ev_series(hist, start_ts)
    _, zt_rows = _streak_engine(hist, start_ts)

    rows = []
    for d, ddist in similar.items():
        w = _window_ev(days, pos_map[d], daily, zt_rows)
        cand = {"fb": w["fb"], "relay": w["relay"], "hold": w["hold_pd"],
                "cash": 0.0}  # 单日可比口径（锁仓按每持有日）
        best = max(cand, key=lambda k: (cand[k] is not None, cand[k]))
        rows.append({"date": str(d.date()), "distance": round(float(ddist), 3),
                     **{f"{k}_ev": w[k] for k in w}, "best": best})

    df = pd.DataFrame(rows)
    def _agg_col(key: str) -> dict:
        vals = pd.to_numeric(df[f"{key}_ev"], errors="coerce").dropna()
        if len(vals) == 0:
            return {"ev": None, "win_rate": None, "n": 0, "std": None}
        return {"ev": float(vals.mean()), "win_rate": float((vals > 0).mean()),
                "n": int(len(vals)), "std": float(vals.std())}

    agg = {k: _agg_col(k) for k in ("fb", "relay", "hold", "hold_pd", "hold_days")}
    # 空仓: EV=0；胜率 = 相似日中所有策略单日EV均≤0（空仓最优）的占比
    daily_evs = df[["fb_ev", "relay_ev", "hold_pd_ev"]].apply(pd.to_numeric, errors="coerce")
    cash_best = daily_evs.max(axis=1).fillna(0.0) <= 0
    agg["cash"] = {"ev": 0.0, "win_rate": round(float(cash_best.mean()), 3),
                   "n": len(df), "std": 0.0}

    # 选优池: 仅四类可比策略（锁仓按单日EV），累计/持有天数仅作信息展示
    playable = {"fb": agg["fb"]["ev"], "relay": agg["relay"]["ev"],
                "hold": agg["hold_pd"]["ev"], "cash": 0.0}
    def _ev(k: str) -> float:
        v = playable[k]
        return v if v is not None else float("-inf")
    strategy = max(playable, key=_ev)
    rep = agg["hold_pd"] if strategy == "hold" else agg[strategy]
    agree = float((df["best"] == strategy).mean()) if len(df) else 0.0
    evs = sorted((playable[k] for k in playable if playable[k] is not None), reverse=True)
    second_ev = evs[1] if len(evs) > 1 else 0.0
    stability = 1.0 - min(1.0, (rep["std"] or 0.0) / (abs(rep["ev"]) + 1.0))
    edge = abs(rep["ev"]) / (abs(rep["ev"]) + abs(second_ev) + 1e-9) if rep["ev"] != 0 else 0.0
    confidence = float(np.clip(0.30 * agree + 0.15 * len(df) / SIMILAR_DAYS +
                               0.30 * stability + 0.25 * edge, 0.05, 0.95))
    confidence = round(confidence, 3)

    cur = st.loc[st["date"] == target].iloc[0]
    def _f(x, d=2):
        return None if pd.isna(x) else round(float(x), d)
    def _pct(x, d=1):
        return "NA" if pd.isna(x) else f"{x * 100:.{d}f}%"
    def _r(v, d=3):
        return None if v is None else round(v, d)
    hold_txt = (f"锁仓EV {_r(agg['hold']['ev'])}%(单日{_r(agg['hold_pd']['ev'])}%)"
                if agg["hold"]["ev"] is not None else "锁仓EV NA")
    reason = (f"今日状态: 最高板{int(cur['max_board'])}板/涨停{int(cur['zt_cnt'])}家/"
              f"炸板率{_pct(cur['zb_rate'])}/1进2晋级率{_pct(cur['jr1'])}/"
              f"昨日涨停溢价{_f(cur['premium'])}%。"
              f"近{len(df)}个相似日后续3日: 打板EV {_r(agg['fb']['ev'])}%、接力EV {_r(agg['relay']['ev'])}%、"
              f"{hold_txt}、空仓0；相似日{strategy}单日平均EV {_r(rep['ev'])}%"
              f"(胜率{rep['win_rate']:.0%})。")
    name = {"fb": "打板", "relay": "接力", "hold": "锁仓", "cash": "空仓"}[strategy]
    reason += f"建议: {name}。"

    return {
        "status": "ok",
        "date": str(target.date()),
        "state": {k: _f(cur[k]) for k in FEATURES},
        "strategy": name,
        "strategy_key": strategy,
        "ev": round(rep["ev"], 3),
        "win_rate": round(rep["win_rate"], 3),
        "confidence": confidence,
        "reason": reason,
        "similar_days": rows,
        "agg": {k: ({"days": _r(v["ev"])} if k == "hold_days" else
                    {"ev": _r(v["ev"]), "win_rate": _r(v["win_rate"]), "n": v["n"]})
                for k, v in agg.items()},
    }


def run_report(date: str | None = None) -> dict:
    """汇总: 阶段1矩阵 + 阶段2建议 → generated/game_theory_{date}.json。"""
    matrix = calibrate_matrix()
    game = current_game(date)
    d = game.get("date") or date or datetime.now(CST).strftime("%Y-%m-%d")
    out = {"status": "ok", "matrix": matrix, "game": game}
    if matrix.get("status") != "ok" or game.get("status") != "ok":
        out["status"] = "insufficient_data"
    p = ROOT / "generated" / f"game_theory_{d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="游资博弈论建模")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--current", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    kw = {"use_cache": not args.no_cache}
    if args.report:
        print(json.dumps(run_report(args.date), ensure_ascii=False, indent=1))
    elif args.current:
        print(json.dumps(current_game(args.date), ensure_ascii=False, indent=1))
    elif args.calibrate or not (args.current or args.report):
        print(json.dumps(calibrate_matrix(**kw), ensure_ascii=False, indent=1))
