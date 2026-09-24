#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缠论分析 Skills（2026-08-22 —— epub 做成可调用技能）

基于《图解缠论》解析的规则，把 K 线数据判为缠论结构:
 1. structure_judge(df)    → 走势结构判定(下跌-盘整-下跌/盘整-上涨等)
 2. buy_point_scan(df)     → 买点扫描(中短线买点特征: 下跌-横盘-下跌后企稳)
 3. chan_thoughts(df, q)   → 结合 RAG 检索给出缠论视角解读
 4. chan_report_md(date)   → 缠论分析段(供研报)

数据源: 本地 K 线/index parquet, 全秒级。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _load_index_df(name: str = "沪深300"):
    import pandas as pd
    p = ROOT / "data_warehouse" / "market" / f"index_daily_{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def structure_judge(df, lookback: int = 60) -> dict:
    """走势结构判定: 用高低点序列识别"下跌-盘整-下跌"等(简化)."""
    import pandas as pd
    c = df["close"].astype(float).tail(lookback)
    h = df["high"].astype(float).tail(lookback)
    l = df["low"].astype(float).tail(lookback)
    # 分段: 找显著高低点(±2%)
    pivots = []
    for i in range(2, len(c) - 2):
        if c.iloc[i] >= max(c.iloc[i-2:i+1]) * 1.02:
            pivots.append(("高", float(c.iloc[i])))
        elif c.iloc[i] <= min(c.iloc[i-2:i+1]) * 0.98:
            pivots.append(("低", float(c.iloc[i])))
    # 趋势方向
    seg = [p[0] for p in pivots[-8:]]
    last_chg = float((c.iloc[-1] / c.iloc[max(0, len(c)-10)] - 1) * 100)
    # 结构判定
    if seg[-2:] == ["低", "高"] and last_chg > 0:
        struct = "上涨段(高-低结构)"
    elif seg.count("低") >= 3 and last_chg < 0:
        struct = "下跌-盘整-下跌(下盘下)"
    elif seg[-2:] == ["高", "低"] and last_chg < 0:
        struct = "下跌段(高-低)"
    else:
        struct = "盘整区"
    # 相对位置(距60日高低)
    h60, l60 = float(h.max()), float(l.min())
    pos = (float(c.iloc[-1]) - l60) / (h60 - l60) * 100
    return {"结构": struct, "枢轴数": len(pivots), "近10日涨幅": round(last_chg, 1),
            "位置分位": round(pos, 0), "60日高低": f"{l60:.0f}-{h60:.0f}"}


def buy_point_scan(df, lookback: int = 40) -> list[dict]:
    """中短线买点扫描(简化): 下跌-横盘-下跌后企稳+距低点近。"""
    import pandas as pd
    c = df["close"].astype(float).tail(lookback)
    l = df["low"].astype(float).tail(lookback)
    l60 = float(c.min())
    # 近5日是否企稳(未创新低+回升)
    recent = c.tail(5)
    new_low = recent.min() < c.tail(10).min()
    rebound = recent.iloc[-1] > recent.iloc[0]
    near_low = (float(recent.iloc[-1]) - l60) / l60 * 100 < 5
    sigs = []
    if not new_low and rebound and near_low:
        sigs.append({"信号": "下跌企稳反弹", "级别": "中短线", "位置": f"距60日低{((float(recent.iloc[-1])/l60-1)*100):.1f}%"})
    return sigs


# ═══════════════════════════════════════════════════════════
# 缠论笔段自动生成（分型→笔→线段→中枢, 2026-08-22 用户"缠论图我没有，你要丰富"）
# ═══════════════════════════════════════════════════════════
def _fractals(df, n: int = 3) -> list[dict]:
    """顶/底分型（K线高低点相对窗口极值, 幅度过滤去噪）。"""
    import pandas as pd
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    dates = df["date"].astype(str).str[:10].tolist()
    fxs = []
    for i in range(n, len(df) - n):
        left_h = highs.iloc[i - n:i].max()
        right_h = highs.iloc[i + 1:i + n + 1].max()
        left_l = lows.iloc[i - n:i].min()
        right_l = lows.iloc[i + 1:i + n + 1].min()
        hi, lo = float(highs.iloc[i]), float(lows.iloc[i])
        # 顶: 高于左右窗口 + 幅度>0.3%
        if hi > max(left_h, right_h) and (hi / max(lo, 1) - 1) * 100 > 0.3:
            fxs.append({"date": dates[i], "type": "顶", "price": hi, "i": i})
        # 底: 低于左右窗口 + 幅度>0.3%
        elif lo < min(left_l, right_l) and (min(left_l, right_l) / max(lo, 1) - 1) * 100 > 0.3:
            fxs.append({"date": dates[i], "type": "底", "price": lo, "i": i})
    return fxs


def _bi(fxs: list[dict]) -> list[dict]:
    """笔: 交替顶底分型连接(价格差>阈值)。"""
    bis = []
    last = None
    for fx in fxs:
        if last is None:
            last = fx
            continue
        # 交替且幅度够(>0.5%)
        if fx["type"] != last["type"]:
            diff = (fx["price"] / last["price"] - 1) * 100
            if abs(diff) > 0.5:
                bis.append({"from": last["date"], "to": fx["date"],
                            "from_p": last["price"], "to_p": fx["price"],
                            "dir": "上" if fx["price"] > last["price"] else "下",
                            "chg": round(diff, 2)})
                last = fx
    return bis


def _zhongshu(bis: list[dict]) -> list[dict]:
    """中枢: 连续3笔重叠区间。"""
    zs = []
    for i in range(len(bis) - 2):
        a = bis[i]
        b = bis[i + 1]
        c = bis[i + 2]
        lo = max(min(a["from_p"], a["to_p"]), min(c["from_p"], c["to_p"]))
        hi = min(max(a["from_p"], a["to_p"]), max(c["from_p"], c["to_p"]))
        if hi > lo:  # 重叠
            zs.append({"start": a["from"], "end": c["to"], "low": round(lo, 1), "high": round(hi, 1)})
    return zs


def chan_bigscan(name: str = "沪深300", days: int = 180) -> dict:
    """缠论结构扫描: 分型/笔/中枢 + 当前结构判定。"""
    df = _load_index_df(name)
    if df is None:
        return {}
    df = df.tail(days)
    fxs = _fractals(df)
    bis = _bi(fxs)
    zs = _zhongshu(bis)
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    # 当前相对最近中枢
    cur_zs = "无中枢"
    cur_pos = ""
    if zs:
        last_zs = zs[-1]
        if last > last_zs["high"]:
            cur_zs = f"{last_zs['low']}-{last_zs['high']}"
            cur_pos = "中枢上方(突破)"
        elif last < last_zs["low"]:
            cur_zs = f"{last_zs['low']}-{last_zs['high']}"
            cur_pos = "中枢下方(弱势)"
        else:
            cur_zs = f"{last_zs['low']}-{last_zs['high']}"
            cur_pos = "中枢内(震荡)"
    # 笔数据(供画图)
    bi_pts = [{"x": b["from"], "y": b["from_p"]} for b in bis] + [{"x": bis[-1]["to"], "y": bis[-1]["to_p"]}] if bis else []
    return {
        "name": name, "fractals": len(fxs), "bis": len(bis), "zhongshu": zs,
        "last": round(last, 1), "cur_zs": cur_zs, "cur_pos": cur_pos,
        "bi_pairs": [[b["from"], b["to"]] for b in bis[-12:]],  # 近12笔(画图用)
        "bi_prices": [b["from_p"] for b in bis[-12:]] + ([bis[-1]["to_p"]] if bis else []),
        "note": f"自动缠论: {len(fxs)}分型 {len(bis)}笔 {len(zs)}中枢 → {cur_pos}",
    }


def chan_weekly(name: str = "沪深300") -> dict:
    """周线级别缠论: 周线重采样后分型/笔/中枢(做大做强-多级别)."""
    import pandas as pd
    df = _load_index_df(name)
    if df is None:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    w = df.set_index("date").resample("W").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna().tail(80)
    w = w.reset_index().rename(columns={"index": "date"})
    if "date" not in w.columns:
        w["date"] = w.index
    fxs = _fractals(w, n=2)
    bis = _bi(fxs)
    zs = _zhongshu(bis)
    last = float(w["close"].iloc[-1])
    cur = "无中枢"
    pos = ""
    if zs:
        z = zs[-1]
        cur = f"{z['low']}-{z['high']}"
        pos = "上攻" if last > z["high"] else ("下方" if last < z["low"] else "中枢内")
    return {"name": name + "周线", "fractals": len(fxs), "bis": len(bis), "zhongshu": zs[-2:],
            "last": round(last, 1), "cur_zs": cur, "cur_pos": pos,
            "note": f"周线缠论: {len(fxs)}分型 {len(bis)}笔 {len(zs)}中枢 → {pos}({cur})"}


def chan_all_indices() -> list[dict]:
    """多指数缠论总览(上证50/沪深300/创业板/科创50/中证500/中证1000 日线+周线)."""
    out = []
    for nm in ("上证50", "沪深300", "创业板指", "科创50", "中证500", "中证1000"):
        try:
            d = chan_bigscan(nm, days=180)
            w = chan_weekly(nm)
            out.append({"index": nm, "day_pos": d.get("cur_pos", "-"), "day_zs": d.get("cur_zs", "-"),
                        "week_pos": w.get("cur_pos", "-"), "week_zs": w.get("cur_zs", "-"),
                        "bis": d.get("bis", 0), "last": d.get("last", 0)})
        except Exception:
            continue
    return out


def chan_all_md() -> str:
    L = ["## 📐 多指数缠论总览（日线+周线）"]
    for x in chan_all_indices():
        L.append(f"- {x['index']}: {x['last']} 日线{x['day_pos']}({x['day_zs']}) 周线{x['week_pos']}({x['week_zs']})")
    return "\n".join(L)


# 用户缠论图(2026-08-18 上证5分钟笔结构)关键位
USER_CHAN_LEVELS = {"强势突破": 3968, "中等不破": 3903, "弱势支撑": 3884, "关键位": 3838}
# 长周期均线(替代固定图点位, 动态关键位)
LONG_MA = [5, 10, 144, 300, 750]


def _long_mas(df) -> dict:
    """长周期均线值(5/10/144/300/750 取可用)。"""
    import pandas as pd
    c = df["close"].astype(float)
    out = {}
    for n in LONG_MA:
        if len(c) >= n:
            out[f"MA{n}"] = round(float(c.tail(n).mean()), 1)
    return out


def script_judge_by_levels(df) -> dict:
    """缠论三剧本判定(混合: 用户图关键位 + 长周期均线): 
    关键位取自均线动态(MA144≈中期/MA300/750≈长期), 剧本=用户图规则。"""
    import pandas as pd
    c = df["close"].astype(float)
    last = float(c.iloc[-1])
    mas = _long_mas(df)
    # 关键位: 优先长周期均线(动态), 缺乏用用户图固定位
    strong = mas.get("MA144", USER_CHAN_LEVELS["强势突破"])
    weak = mas.get("MA300", USER_CHAN_LEVELS["弱势支撑"])
    if len(c) >= 300 and mas.get("MA300"):
        strong = mas["MA144"]
        weak = mas["MA300"]
    mid = (strong + weak) / 2
    # 剧本判定
    if last >= strong:
        script, op, act = "剧本1 强势", "突破上行", "🟢 偏多延续(站上MA144)"
    elif last >= mid:
        script, op, act = "剧本2 中等", "中枢震荡", "🟠 震荡(MA144-MA300间)"
    else:
        script, op, act = "剧本3 弱势", "下探支撑", "🔴 退潮(跌破MA300)"
    return {
        "last": round(last, 1), "mas": mas, "script": script, "op": op, "act": act,
        "levels": {"MA144": strong, "MA300": weak, "中位": round(mid, 1)},
        "user_levels": USER_CHAN_LEVELS,
        "note": "关键位=长周期均线(144/300/750)动态, 对应缠论图关键位; 亦标用户图3968/3903/3884/3838",
    }


def chan_report_md(date: str | None = None) -> str:
    """缠论分析段 → Markdown(供研报接入, 对应模板1.6.5)."""
    df = _load_index_df("沪深300")
    if df is None:
        return "## 📐 缠论分析\n- 指数数据缺失"
    st = structure_judge(df)
    bp = buy_point_scan(df)
    sj = script_judge_by_levels(df)
    L = ["## 📐 缠论分析（用户图精确剧本 + 均线关键位）"]
    L.append(f"- {sj['note']}")
    L.append(f"- 当前位置 {sj['last']} → **{sj['script']}** {sj['op']} {sj['act']}")
    L.append(f"- 均线关键位: " + " · ".join(f"{k}={v}" for k, v in sj["levels"].items()))
    L.append(f"- 用户图关键位: " + " · ".join(f"{k}={v}" for k, v in USER_CHAN_LEVELS.items()))
    L.append(f"- 结构: {st['结构']} · 近10日{st['近10日涨幅']:+.1f}% · 均线 " + " ".join(f"{k}={v}" for k, v in list(sj["mas"].items())[:6]))
    if bp:
        for b in bp:
            L.append(f"- 🎯 买点: {b['信号']} [{b['级别']}] {b['位置']}")
    else:
        L.append("- 当前无缠论买点信号(需企稳反弹结构)")
    # 关联RAG知识
    try:
        from chan_rag import query_md  # noqa: PLC0415
        if "下盘下" in st["结构"] or "盘整" in st["结构"]:
            L.append("")
            L.append(query_md("下跌 盘整 底部", top=1))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(chan_report_md())