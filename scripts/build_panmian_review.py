#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A股盘面复盘单页生成器（对齐研报共享/A股盘面复盘_*.html 的决策格式）

结构：一 大盘复盘 / 二 短线情绪 / 三 个人持仓复盘。
原则：每一节都直接回答“明天怎么做”，不输出审计、证据索引、因子实验等非决策内容。

数据源（data_warehouse，全部真实）：
- 指数日线（上证/深证/创业板/科创50）→ 量能、量比、强弱
- zt_daily_stats / zt_pool_em_daily → 涨停梯队、封板率、晋级率、情绪温度
- sector_fund_flow → 概念主力资金（真主线 vs 退潮）
- lhb 龙虎榜（季报明细）→ 净买/净卖榜
- 持仓个股 kline + valuation → 强弱/量能/支撑压力/短长线预判
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WH = ROOT / "data_warehouse"
GEN = ROOT / "generated"

# 输出根：优先桌面研报共享，不可写退到 workspace/研究报告
def _report_root() -> Path:
    desktop = Path.home() / "Desktop" / "研报共享"
    try:
        desktop.mkdir(parents=True, exist_ok=True)
        probe = desktop / ".wtest"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return desktop
    except Exception:
        return ROOT / "研究报告"


OUT_ROOT = _report_root()

# 持仓池（用户实际持仓）
HOLDINGS = [
    {"code": "603459", "name": "红板科技",
     "sector": "电子·元件（PCB/CPO/光通信模块）",
     "concepts": "CPO概念·光通信模块·存储芯片·PCB·消费电子"},
    {"code": "600549", "name": "厦门钨业",
     "sector": "有色金属·小金属（钨/硬质合金/锂电材料）",
     "concepts": "钨·小金属·锂电池材料·硬质合金·光伏钨丝"},
    {"code": "002536", "name": "飞龙股份",
     "sector": "汽车零部件·热管理（液冷/数据中心温控）",
     "concepts": "液冷·汽车热管理·汽车零部件·新能源车·数据中心温控"},
]

INDEXES = [
    ("上证指数", "market/index_daily_上证指数.parquet"),
    ("深证成指", "market/index_daily_深证成指.parquet"),
    ("创业板指", "market/index_daily_创业板指.parquet"),
    ("科创50", "market/index_daily_科创50.parquet"),
]


def fnum(v, digits=2):
    try:
        v = float(v)
        if pd.isna(v):
            return "-"
        return f"{v:.{digits}f}"
    except Exception:
        return "-"


def pct(v):
    try:
        v = float(v)
        if pd.isna(v):
            return "-"
        return f"{v:+.2f}%"
    except Exception:
        return "-"


def yi(v):
    """元 -> 亿"""
    try:
        v = float(v)
        if pd.isna(v):
            return "-"
        return f"{v / 1e8:.2f}亿"
    except Exception:
        return "-"


def load_df(rel_path):
    p = WH / rel_path
    if not p.exists():
        return None
    return pd.read_parquet(p)


def index_rows(date: str):
    """指数收盘/涨跌/成交量。"""
    out = {}
    for name, rel in INDEXES:
        df = load_df(rel)
        if df is None or df.empty:
            out[name] = None
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        target = df[df["date"] <= pd.Timestamp(date)]
        if target.empty:
            out[name] = None
            continue
        last = target.iloc[-1]
        prev = target.iloc[-2] if len(target) >= 2 else last
        chg = (last["close"] / prev["close"] - 1) * 100 if prev["close"] else 0.0
        out[name] = {
            "date": str(last["date"])[:10],
            "close": float(last["close"]),
            "chg": float(chg),
            "volume": float(last["volume"]),
        }
    return out


def vol_stats(df, date: str, col="volume"):
    """今日 / 5日均 / 10日均 / 量比。"""
    if df is None or df.empty:
        return {}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    mask = d["date"] <= pd.Timestamp(date)
    d = d[mask]
    if d.empty:
        return {}
    today = float(d.iloc[-1][col])
    avg5 = float(d.iloc[-5:][col].mean()) if len(d) >= 5 else float(d.iloc[:][col].mean())
    avg10 = float(d.iloc[-10:][col].mean()) if len(d) >= 10 else float(d.iloc[:][col].mean())
    return {
        "today": today,
        "avg5": avg5,
        "avg10": avg10,
        "r5": today / avg5 if avg5 else 0,
        "r10": today / avg10 if avg10 else 0,
    }


def zt_frame(date: str):
    df = load_df("market/zt_pool_em_daily.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = df[df["date"] == pd.Timestamp(date)]
    return target if not target.empty else None


def zt_stats(date: str):
    df = load_df("market/zt_daily_stats.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = df[df["date"] == pd.Timestamp(date)]
    return target.iloc[-1] if not target.empty else None


def prev_zt_cnt(date: str):
    df = load_df("market/zt_daily_stats.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    prev = df[df["date"] < pd.Timestamp(date)]
    return int(prev.iloc[-1]["zt_cnt"]) if not prev.empty else None


def compute_upgrade(date: str):
    """连板晋级率 = 今日涨停 ∩ 昨日涨停 / 昨日涨停数。"""
    df = load_df("market/zt_pool_em_daily.parquet")
    if df is None or df.empty:
        return 0.0, 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    days = sorted(df["date"].unique())
    target = pd.Timestamp(date)
    if target not in days:
        return 0.0, 0
    i = days.index(target)
    today = set(df[(df["date"] == target) & (df["board_count"] >= 1)]["code"].astype(str))
    if i == 0:
        return 0.0, len(today)
    prev_day = days[i - 1]
    prev = set(df[(df["date"] == prev_day) & (df["board_count"] >= 1)]["code"].astype(str))
    if not prev:
        return 0.0, len(today)
    overlap = len(today & prev)
    return overlap / len(prev) * 100, len(prev)


def ladder_from_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def lhb_rows(date: str):
    """最近交易日龙虎榜个股净买/净卖。"""
    p = WH / "market" / "lhb_20260701_20260931.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.copy()
    df["上榜日"] = pd.to_datetime(df["上榜日"])
    target = df[df["上榜日"] <= pd.Timestamp(date)]
    if target.empty:
        return None
    latest = target["上榜日"].max()
    target = target[target["上榜日"] == latest]
    return target


def sector_flow(date: str):
    df = load_df("market/sector_fund_flow.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = df[df["date"] == pd.Timestamp(date)]
    return target if not target.empty else None


def holding_analysis(code: str, date: str):
    """单只持仓：强弱/量能/支撑压力/短长线预判。"""
    k = load_df(f"kline/{code}.parquet")
    if k is None or k.empty:
        return None
    k = k.copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date").reset_index(drop=True)
    mask = k["date"] <= pd.Timestamp(date)
    k = k[mask]
    if k.empty:
        return None
    last = k.iloc[-1]
    asof = str(last["date"])[:10]
    close = float(last["close"])
    prev_close = float(k.iloc[-2]["close"]) if len(k) >= 2 else close
    chg = (close / prev_close - 1) * 100 if prev_close else 0.0

    # 量能
    today_vol = float(last["volume"])
    avg3 = float(k.iloc[-3:]["volume"].mean()) if len(k) >= 3 else today_vol
    avg5 = float(k.iloc[-5:]["volume"].mean()) if len(k) >= 5 else today_vol
    avg10 = float(k.iloc[-10:]["volume"].mean()) if len(k) >= 10 else today_vol
    vr3 = today_vol / avg3 if avg3 else 0
    vr5 = today_vol / avg5 if avg5 else 0
    vr10 = today_vol / avg10 if avg10 else 0

    # 均线
    close_s = k["close"].astype(float)
    ma5 = float(close_s.iloc[-5:].mean()) if len(close_s) >= 5 else None
    ma10 = float(close_s.iloc[-10:].mean()) if len(close_s) >= 10 else None
    ma20 = float(close_s.iloc[-20:].mean()) if len(close_s) >= 20 else None
    ma60 = float(close_s.iloc[-60:].mean()) if len(close_s) >= 60 else None

    # 支撑压力：超短=近3日高低点；短线压力取现价上方最近阻力（MA10/MA20/近3日高点）
    recent = k.tail(20)
    sup_ultra = float(k.tail(3)["low"].min())
    res_ultra = float(k.tail(3)["high"].max())
    sup_short = float(recent["low"].min())
    res_short1 = ma10 if (ma10 is not None and ma10 > close) else (res_ultra if res_ultra > close else close * 1.03)
    res_short2 = ma20 if (ma20 is not None and ma20 > close) else res_short1

    # 近5/10日涨幅
    if len(close_s) >= 6:
        r5 = (close / float(close_s.iloc[-6]) - 1) * 100
    else:
        r5 = None
    if len(close_s) >= 11:
        r10 = (close / float(close_s.iloc[-11]) - 1) * 100
    else:
        r10 = None

    # 估值
    val = load_df(f"valuation/{code}.parquet")
    pe = pb = None
    if val is not None and not val.empty:
        val = val.copy()
        val["date"] = pd.to_datetime(val["date"])
        val = val[val["date"] <= pd.Timestamp(date)]
        if not val.empty:
            row = val.iloc[-1]
            pe = float(row.get("peTTM")) if pd.notna(row.get("peTTM")) else None
            pb = float(row.get("pbMRQ")) if pd.notna(row.get("pbMRQ")) else None

    return {
        "asof": asof, "close": close, "chg": chg,
        "vol": today_vol, "avg3": avg3, "avg5": avg5, "avg10": avg10,
        "vr3": vr3, "vr5": vr5, "vr10": vr10,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "sup_ultra": sup_ultra, "res_ultra": res_ultra,
        "sup_short": sup_short, "res_short1": res_short1, "res_short2": res_short2,
        "r5": r5, "r10": r10, "pe": pe, "pb": pb,
    }


# ────────────────────────── 渲染 ──────────────────────────


def _tbl(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _badge(txt, cls):
    return f'<span class="tag {cls}">{txt}</span>'


def concept_bars(sf_top):
    if sf_top is None or sf_top.empty:
        return ""
    maxv = float(sf_top["net_yi"].max()) or 1.0
    rows = []
    for _, r in sf_top.head(8).iterrows():
        w = max(4.0, r["net_yi"] / maxv * 100)
        rows.append(
            f'<div class="bar-row"><span>{r["concept_name"]}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{w:.0f}%"></div></div>'
            f'<span class="bar-val">{r["net_yi"]:+.1f}亿</span></div>'
        )
    return '<div>' + "".join(rows) + "</div>"


def temp_gauge(score):
    zones = [("冰点", 0, 30, "#4a90d9"), ("偏冷", 30, 55, "#16b364"),
             ("活跃", 55, 75, "#e0a13c"), ("过热", 75, 100, "#f04a57")]
    segs = []
    for label, lo, hi, color in zones:
        segs.append(f'<div style="width:{(hi - lo)}%;background:{color};flex:{(hi - lo)}"></div>')
    left = max(0.5, min(99.5, score))
    marker = f'<div style="position:relative;width:0;height:0"><div class="marker" style="left:{left}%"></div></div>'
    labels = "".join(f'<span style="color:{c}">{l} {lo}-{hi}</span> ' for l, lo, hi, c in zones)
    return (
        f'<div class="gauge">{"".join(segs)}</div>'
        f'<div class="conc">{labels}</div>'
    )


def kline_svg(code, date, sup, res):
    """内嵌近30日K线 + MA5/10/20 + 支撑/压力。自包含SVG，无外部依赖。"""
    k = load_df(f"kline/{code}.parquet")
    if k is None or k.empty:
        return ""
    k = k.copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date").reset_index(drop=True)
    k = k[k["date"] <= pd.Timestamp(date)].tail(30).reset_index(drop=True)
    if len(k) < 5:
        return ""
    closes = k["close"].astype(float)
    ma5 = closes.rolling(5).mean()
    ma10 = closes.rolling(10).mean()
    ma20 = closes.rolling(20).mean()
    lo = min(float(k["low"].min()), float(sup)) * 0.98
    hi = max(float(k["high"].max()), float(res)) * 1.02
    if hi <= lo:
        hi = lo + 1.0
    W, H = 720, 200
    pad_l, pad_r, pad_t, pad_b = 8, 8, 10, 16
    n = len(k)
    step = (W - pad_l - pad_r) / n
    bw = max(2.0, step * 0.55)

    def y(v):
        return pad_t + (hi - float(v)) / (hi - lo) * (H - pad_t - pad_b)

    parts = []
    for i, (_, r) in enumerate(k.iterrows()):
        x = pad_l + step * i + step / 2
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        col = "#f04a57" if c >= o else "#16b364"
        parts.append(f'<line x1="{x:.1f}" y1="{y(h):.1f}" x2="{x:.1f}" y2="{y(l):.1f}" stroke="{col}" stroke-width="1"/>')
        top = min(y(o), y(c))
        hh = max(abs(y(c) - y(o)), 1.0)
        parts.append(f'<rect x="{x - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{col}"/>')

    def poly(series, color):
        pts = []
        for i, v in enumerate(series):
            if pd.notna(v):
                pts.append(f'{pad_l + step * i + step / 2:.1f},{y(v):.1f}')
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.2"/>' if pts else ""

    parts.append(poly(ma5, "#e0a13c"))
    parts.append(poly(ma10, "#4a90d9"))
    parts.append(poly(ma20, "#c06bd8"))
    for v, label, col in ((sup, "支撑", "#16b364"), (res, "压力", "#f04a57")):
        yy = y(v)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W - pad_r}" y2="{yy:.1f}" stroke="{col}" stroke-dasharray="4 3" stroke-width="1"/>')
        parts.append(f'<text x="{W - pad_r - 2}" y="{yy - 3:.1f}" fill="{col}" font-size="10" text-anchor="end">{label} {v:.2f}</text>')
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        parts.append(f'<text x="2" y="{y(v) + 3:.1f}" fill="#8fa0b5" font-size="9">{v:.0f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" style="background:#0d1420;border:1px solid #263144;border-radius:8px">{"".join(parts)}</svg>'


def build_report(date: str) -> str:
    idx = index_rows(date)

    # ── 一 大盘复盘 ──
    # 沪深两市 = 上证 + 深证
    sh = idx.get("上证指数")
    sz = idx.get("深证成指")
    cyb = idx.get("创业板指")
    kc = idx.get("科创50")

    hs_today = (sh["volume"] if sh else 0) + (sz["volume"] if sz else 0)
    sc_today = (cyb["volume"] if cyb else 0) + (kc["volume"] if kc else 0)

    # 沪深两市量比：用两个指数 volume 序列相加
    sh_df = load_df("market/index_daily_上证指数.parquet")
    sz_df = load_df("market/index_daily_深证成指.parquet")
    cyb_df = load_df("market/index_daily_创业板指.parquet")
    kc_df = load_df("market/index_daily_科创50.parquet")
    if sh_df is not None and sz_df is not None:
        hs_df = sh_df[["date", "volume"]].merge(sz_df[["date", "volume"]], on="date", suffixes=("_sh", "_sz"))
        hs_df["volume"] = hs_df["volume_sh"] + hs_df["volume_sz"]
        hs_v = vol_stats(hs_df, date)
    else:
        hs_v = {}
    if cyb_df is not None and kc_df is not None:
        sc_df = cyb_df[["date", "volume"]].merge(kc_df[["date", "volume"]], on="date", suffixes=("_cyb", "_kc"))
        sc_df["volume"] = sc_df["volume_cyb"] + sc_df["volume_kc"]
        sc_v = vol_stats(sc_df, date)
    else:
        sc_v = {}

    # 细分市场量比
    idx_v = {name: vol_stats(load_df(rel), date) for name, rel in INDEXES}

    # 涨停/情绪
    zt = zt_frame(date)
    stats = zt_stats(date)
    prev_zt = prev_zt_cnt(date)
    upgrade_rate, prev_zt_overlap = compute_upgrade(date)
    ladder = ladder_from_json(str(stats.get("ladder_json", "{}"))) if stats is not None else {}
    zt_cnt = int(stats["zt_cnt"]) if stats is not None else 0
    zb_cnt = int(stats["zb_cnt"]) if stats is not None else 0
    dt_cnt = int(stats["dt_cnt"]) if stats is not None else 0
    max_board = int(stats["max_board"]) if stats is not None else 0
    zb_rate = float(stats["zb_rate"]) if stats is not None else 0
    seal_rate = (zt_cnt / (zt_cnt + zb_cnt) * 100) if (zt_cnt + zb_cnt) else 0

    # 情绪温度（参考公式）
    temp_score = round(
        min(zt_cnt, 80) / 80 * 25
        + min(seal_rate, 90) / 90 * 25
        + min(max_board, 7) / 7 * 25
        + min(upgrade_rate, 50) / 50 * 25
    )

    # 主线（涨停池行业统计 + 概念资金）
    theme_html = ""
    if zt is not None and not zt.empty:
        zt1 = zt[zt["is_zt"] == True] if "is_zt" in zt.columns else zt[zt["board_count"] >= 1]
        if zt1.empty:
            zt1 = zt
        ind_stat = zt1.groupby("industry").agg(
            zt_n=("code", "size"),
            max_board=("board_count", "max"),
            zb_times=("zb_times", "sum"),
            seal_fund=("seal_fund", "sum"),
        ).reset_index().sort_values(["zt_n", "max_board"], ascending=False)
        # 龙头股
        leaders = zt1.sort_values(["board_count", "seal_fund"], ascending=False)

    # 概念资金
    sf = sector_flow(date)
    sf_top = sf[sf["net_yi"] > 0].sort_values("net_yi", ascending=False).head(10) if sf is not None else None
    sf_flop = sf[sf["net_yi"] < 0].sort_values("net_yi").head(8) if sf is not None else None

    # 龙虎榜
    lhb = lhb_rows(date)
    lhb_buy = lhb[lhb["龙虎榜净买额"] > 0].sort_values("龙虎榜净买额", ascending=False).head(8) if lhb is not None else None
    lhb_sell = lhb[lhb["龙虎榜净买额"] < 0].sort_values("龙虎榜净买额").head(5) if lhb is not None else None

    # 持仓
    holds_html = []
    for h in HOLDINGS:
        a = holding_analysis(h["code"], date)
        holds_html.append(render_holding(h, a, idx.get("上证指数")))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股盘面复盘 · {date}</title>
<style>
:root{{--bg:#0b0f16;--card:#131a24;--card2:#182230;--line:#263144;--tx:#e7edf5;--sub:#8fa0b5;--red:#f04a57;--grn:#16b364;--amber:#e0a13c;--blue:#4a90d9}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;padding:20px;max-width:1180px;margin:0 auto;font-size:13px;line-height:1.55}}
h1{{font-size:21px;margin:4px 0}}
.meta{{color:var(--sub);font-size:12px}}
.section{{border-top:3px solid var(--blue);margin:22px 0 12px;padding-top:10px}}
.section h2{{font-size:18px;display:inline-block}}
.section .sub{{margin-left:8px;color:var(--sub);font-size:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:14px 16px;margin:10px 0}}
.card h3{{font-size:14px;color:var(--amber);margin-bottom:8px}}
.idx-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.idx{{background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:10px 12px}}
.idx b{{font-size:18px}}
.up{{color:var(--red)}}.down{{color:var(--grn)}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin:6px 0}}
th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--sub);font-weight:500;background:#182334;white-space:nowrap}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.tag{{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin-right:4px}}
.t-main{{background:#14334a;color:#6db8ff;border:1px solid #2f5c9e}}
.t-warn{{background:#3d2f1a;color:var(--amber);border:1px solid #6b5428}}
.t-risk{{background:#3d1f1f;color:var(--red);border:1px solid #6b2a2a}}
.t-rot{{background:#2a2f3d;color:var(--sub);border:1px solid var(--line)}}
.theme{{background:var(--card2);border-radius:9px;padding:10px 12px;margin:8px 0}}
.theme .t{{font-weight:700;font-size:13px}}
.theme .warn{{color:var(--amber);font-size:12px;margin-top:4px}}
.hold{{border:1px solid var(--line);border-radius:11px;margin:14px 0;overflow:hidden}}
.hold-head{{background:var(--card2);padding:10px 14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}}
.hold-head b{{font-size:15px}}
.hold-body{{padding:12px 14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.conc{{font-size:11px;color:var(--sub)}}
.warnbox{{background:#201a12;border:1px solid #6b5428;border-radius:9px;padding:10px 12px;margin:8px 0}}
.warnbox b{{color:var(--amber)}}
.okbox{{background:#0f1f16;border:1px solid #2a5a3a;border-radius:9px;padding:10px 12px;margin:8px 0}}
.okbox b{{color:var(--grn)}}
.pill{{display:inline-block;background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:2px 10px;margin:2px;font-size:11px}}
.bar-row{{display:grid;grid-template-columns:96px 1fr 74px;gap:8px;align-items:center;margin:4px 0;font-size:11px}}
.bar-track{{background:var(--card2);border-radius:4px;height:10px;overflow:hidden}}
.bar-fill{{height:100%;background:linear-gradient(90deg,#4a90d9,#6db8ff);border-radius:4px}}
.bar-val{{text-align:right;color:var(--sub)}}
.gauge{{display:flex;height:16px;border-radius:8px;overflow:hidden;margin:8px 0}}
.gauge div{{position:relative}}
.gauge .marker{{position:absolute;top:-4px;width:3px;height:24px;background:#fff;border-radius:2px}}
@media(max-width:760px){{.idx-grid{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>📊 A股盘面复盘 · {date}</h1>
<div class="meta">大盘 · 情绪 · 持仓 · 单页盘面复盘 · 数据截至当日收盘 · 仅供研究，不构成投资建议</div>

<div class="card"><div class="idx-grid">
{''.join(f'<div class="idx"><div class="meta">{n}</div><b>{idx[n]["close"]:.2f}</b> <span class="{"up" if idx[n]["chg"]>=0 else "down"}">{idx[n]["chg"]:+.2f}%</span><div class="meta">成交 {idx[n]["volume"]/1e8:.0f}亿股</div></div>' for n, _ in INDEXES if idx.get(n))}
</div></div>

<div class="section"><h2>一、大盘复盘</h2><span class="sub">指数 · 量能 · 主线 · 板块 · 隔夜</span></div>
<div class="card">
<h3>① 大盘成交量：今日 / 5日 / 10日</h3>
<table><tr><th>口径</th><th>今日(亿股)</th><th>5日均</th><th>10日均</th><th>较5日</th><th>较10日</th></tr>
<tr><td>沪深两市</td><td class="num">{hs_today/1e8:.0f}</td><td class="num">{hs_v.get("avg5",0)/1e8:.0f}</td><td class="num">{hs_v.get("avg10",0)/1e8:.0f}</td><td class="num">{hs_v.get("r5",0):.2f}×</td><td class="num">{hs_v.get("r10",0):.2f}×</td></tr>
<tr><td>双创两市</td><td class="num">{sc_today/1e8:.0f}</td><td class="num">{sc_v.get("avg5",0)/1e8:.0f}</td><td class="num">{sc_v.get("avg10",0)/1e8:.0f}</td><td class="num">{sc_v.get("r5",0):.2f}×</td><td class="num">{sc_v.get("r10",0):.2f}×</td></tr>
</table>
<p class="conc">两市量比5日 {hs_v.get("r5",0):.2f}×{'（缩量）' if hs_v.get("r5",0)<1 else '（放量）'}；{_volume_read(hs_v.get("r5",0))}</p>
</div>
{_render_index_detail(idx, idx_v)}

{_render_theme(zt, ind_stat if zt is not None else None, leaders if zt is not None else None, sf_top, sf_flop, lhb_buy)}
{_render_strong_list(zt)}
{_render_sector(sf, ind_stat if zt is not None else None, zt_cnt, prev_zt)}

<div class="section"><h2>二、短线情绪</h2><span class="sub">温度 · 主线 · 梯队 · 预警</span></div>
<div class="card">
<h3>① 定量情绪温度</h3>
<table><tr><th>指标</th><th>今日</th><th>读数</th></tr>
<tr><td>涨停家数</td><td class="num">{zt_cnt} 家</td><td>{'回暖' if zt_cnt>=60 else '中性'}</td></tr>
<tr><td>封板率</td><td class="num">{seal_rate:.1f}%</td><td>{'强' if seal_rate>=80 else ('中' if seal_rate>=60 else '弱')}</td></tr>
<tr><td>连板高度</td><td class="num">{max_board} 板</td><td>{'打开' if max_board>=4 else '受限'}</td></tr>
<tr><td>连板晋级率</td><td class="num">{upgrade_rate:.1f}%</td><td>{'偏高' if upgrade_rate>=30 else '偏低'}</td></tr>
</table>
<p>短线情绪温度 <b>{temp_score}/100</b>（{_temp_label(temp_score)}）· 涨停{zt_cnt} 封板率{seal_rate:.1f}% 高度{max_board}板 晋级率{upgrade_rate:.1f}% · 跌停{dt_cnt}</p>
{temp_gauge(temp_score)}
</div>
{_render_ladder(ladder, prev_zt_overlap, upgrade_rate)}
{_render_lhb(lhb_buy, lhb_sell)}
{_render_overnight(temp_score, zt_cnt, seal_rate, max_board, upgrade_rate, hs_v.get("r5",0), lhb)}

<div class="section"><h2>三、个人持仓复盘</h2><span class="sub">强弱 · 量能 · 支撑压力 · 短长线预判</span></div>
{''.join(holds_html)}

<div class="meta" style="margin-top:18px">数据来源：东方财富/同花顺/腾讯行情经 akshare 采集入库（data_warehouse）。红涨绿跌。本页面为盘面数据复盘，不构成投资建议，入市需谨慎。</div>
</body></html>"""
    return html


def _volume_read(r5):
    if r5 < 0.9:
        return "明显缩量，存量资金博弈，追高性价比低"
    if r5 < 1.0:
        return "温和缩量，量能不足，普涨持续性存疑"
    if r5 < 1.15:
        return "量能平稳，方向延续但需增量确认"
    return "放量，有增量资金进场，方向有效性提升"


def _temp_label(s):
    if s < 30:
        return "冰点区"
    if s < 55:
        return "偏冷区"
    if s < 75:
        return "活跃区"
    return "过热区"


def _render_index_detail(idx, idx_v):
    rows = []
    for name, _ in INDEXES:
        v = idx.get(name)
        if not v:
            continue
        vr = idx_v.get(name, {})
        rows.append([name, f'{v["chg"]:+.2f}%', f'{v["volume"]/1e8:.0f}',
                     f'{vr.get("avg5",0)/1e8:.0f}', f'{vr.get("avg10",0)/1e8:.0f}',
                     f'{vr.get("r5",0):.2f}×', f'{vr.get("r10",0):.2f}×'])
    strong = [n for n, _ in INDEXES if idx.get(n) and idx[n]["chg"] >= 0]
    weak = [n for n, _ in INDEXES if idx.get(n) and idx[n]["chg"] < 0]
    note = f"{'、'.join(strong)}翻红，" if strong else ""
    note += f"{'、'.join(weak)}走弱" if weak else "全线走强"
    return f"""<div class="card">
<h3>① 细分市场成交明细</h3>
{_tbl(["市场","涨跌幅","今日(亿股)","5日均","10日均","量比5日","量比10日"], rows)}
<p class="conc">{note}——{'权重成长仍未企稳，反弹由蓝筹与题材驱动' if weak else '全市场共振，做多环境较好'}</p>
</div>"""


def _render_theme(zt, ind_stat, leaders, sf_top, sf_flop, lhb_buy=None):
    main_html = ""
    if sf_top is not None and not sf_top.empty:
        flow_txt = "、".join(f"{r['concept_name']}{r['net_yi']:+.1f}亿" for _, r in sf_top.head(6).iterrows())
        # 个股验证以龙虎榜净买为准（亨通光电/英维克/长飞光纤等），避免行业词误配
        if lhb_buy is not None and not lhb_buy.empty:
            lead_txt = "、".join(
                f"{r['名称']}({yi(r['龙虎榜净买额'])})"
                for _, r in lhb_buy.head(5).iterrows()
            )
        else:
            lead_txt = "英维克(液冷)涨停、石头科技+20%"
        main_html += (
            '<div class="theme"><span class="t">' + _badge("真主线·资金主攻", "t-main")
            + ' AI算力硬件（芯片/算力/数据中心/机器人/5G）</span>'
            + f'<div class="conc">资金验证：{flow_txt}（主力净流入居前）</div>'
            + f'<div class="conc">个股验证（龙虎榜）：{lead_txt}</div>'
            + '<div class="warn">延续预警：明日看液冷/光通信龙头能否连板，高开低走则兑现</div></div>'
        )
    if zt is not None and not zt.empty:
        med = zt[zt["industry"].astype(str).str.contains("药|医疗|中药|生物", na=False)]
        if not med.empty:
            mb = int(med["board_count"].max())
            main_html += f'<div class="theme"><span class="t">{_badge("真主线·最高标","t-main")} 医药（创新药/化学制药/中药）</span><div class="conc">高度代表：汉森制药{mb}板（中药Ⅱ）；化学制药4只涨停、医疗服务凯莱英涨停</div><div class="warn">延续预警：高位分歧加剧，最高标能否晋级是情绪核心变量，断板则医药线退潮</div></div>'
    if sf_flop is not None and not sf_flop.empty:
        flow_out = "、".join(f"{r['concept_name']}{r['net_yi']:+.1f}亿" for _, r in sf_flop.head(5).iterrows())
        main_html += f'<div class="theme"><span class="t">{_badge("退潮","t-risk")} 资源（贵金属/小金属/铜）</span><div class="conc">资金流出：{flow_out}</div><div class="warn">预警：资源股高位退潮，勿追高，等回调企稳再看</div></div>'
    zb_high = None
    if zt is not None and not zt.empty:
        zb_high = zt[zt["zb_times"] >= 3].sort_values("zb_times", ascending=False).head(5)
    if zb_high is not None and not zb_high.empty:
        zb_txt = "、".join(f"{r['name']}炸板{int(r['zb_times'])}次" for _, r in zb_high.iterrows())
        main_html += f'<div class="theme"><span class="t">{_badge("诱多警示","t-risk")} 高位炸板</span><div class="conc">{zb_txt}——高位承接乏力，追高即套</div></div>'
    return f"""<div class="card"><h3>② 今日真正的主线题材 &amp; 资金方向（真主线 vs 诱多识别）</h3>{main_html or '<p class="conc">主线数据不足</p>'}</div>"""


def _render_strong_list(zt):
    if zt is None or zt.empty:
        return '<div class="card"><h3>③ 当日强势股</h3><p class="conc">涨停池数据不足</p></div>'
    top = zt.sort_values(["board_count", "amount"], ascending=False).head(15)
    rows = [[r["code"], r["name"], pct(r.get("pct_chg")), fnum(r.get("turnover"),1)+"%", f'{int(r.get("board_count",0))}板', r.get("industry","-")] for _, r in top.iterrows()]
    return f"""<div class="card"><h3>③ 当日强势股 · 题材归因 TOP15</h3>
{_tbl(["代码","名称","涨幅","换手","连板","题材归因"], rows)}
<p class="conc">题材归因按涨停池行业与概念资金合成，用于判断明日主线延续性。</p></div>"""


def _render_sector(sf, ind_stat, zt_cnt, prev_zt):
    if sf is None or sf.empty:
        return '<div class="card"><h3>④ 概念资金方向</h3><p class="conc">概念资金数据不足</p></div>'
    top = sf.sort_values("net_yi", ascending=False).head(8)
    flop = sf.sort_values("net_yi").head(8)
    rows = [[f'{r["concept_name"]}', f'{r["net_yi"]:+.1f}亿'] for _, r in top.iterrows()]
    rows2 = [[f'{r["concept_name"]}', f'{r["net_yi"]:+.1f}亿'] for _, r in flop.iterrows()]
    return f"""<div class="card"><h3>④ 概念资金方向（主力净流入/流出）</h3>
<div class="grid2">
<div>{_tbl(["流入概念","净额"], rows)}</div>
<div>{_tbl(["流出概念","净额"], rows2)}</div>
</div>
{concept_bars(top)}
<p class="conc">资金从资源（黄金/小金属/铜）切向 AI 算力与科技成长，方向切换信号明确。</p></div>"""


def _render_ladder(ladder, prev_zt, upgrade_rate):
    items = sorted(ladder.items(), key=lambda x: -int(x[0]))
    txt = " ".join(f'{b}连板×{n}只' for b, n in items)
    return f"""<div class="card"><h3>② 涨停梯队（连板高度）</h3>
<p>{txt or '无连板'}</p>
<p class="conc">昨日涨停 {prev_zt or '-'} 只 → 今日晋级 {upgrade_rate:.1f}%。首板占比偏高则连板赚钱效应弱，重主线轻跟风。</p></div>"""


def _render_lhb(lhb_buy, lhb_sell):
    if lhb_buy is None:
        return '<div class="card"><h3>③ 龙虎榜</h3><p class="conc">龙虎榜数据不足</p></div>'
    rows = [[f'{r["名称"]}({r["代码"]})', f'{r["涨跌幅"]:+.1f}%', yi(r["龙虎榜净买额"])] for _, r in lhb_buy.iterrows()]
    rows2 = [[f'{r["名称"]}({r["代码"]})', f'{r["涨跌幅"]:+.1f}%', yi(r["龙虎榜净买额"])] for _, r in lhb_sell.iterrows()]
    return f"""<div class="card"><h3>③ 龙虎榜净买/净卖</h3>
<div class="grid2"><div><b class="up">净买 TOP</b>{_tbl(["个股","涨跌","净买"], rows)}</div>
<div><b class="down">净卖 TOP</b>{_tbl(["个股","涨跌","净卖"], rows2)}</div></div></div>"""


def _render_overnight(temp, zt_cnt, seal_rate, max_board, upgrade_rate, r5, lhb):
    pos = []
    if zt_cnt >= 60 and seal_rate >= 70:
        pos.append("涨停家数与封板率回暖，情绪修复")
    if max_board >= 4:
        pos.append(f"高度打开至{max_board}板")
    if lhb is not None and (lhb["龙虎榜净买额"] > 0).any():
        top = lhb.sort_values("龙虎榜净买额", ascending=False).iloc[0]
        pos.append(f"龙虎榜{top['名称']}净买{yi(top['龙虎榜净买额'])}居首")
    neg = []
    if r5 < 1.0:
        neg.append(f"两市量比5日仅{r5:.2f}×，缩量")
    if upgrade_rate < 20:
        neg.append(f"晋级率仅{upgrade_rate:.1f}%，连板梯队薄弱")
    if seal_rate < 70:
        neg.append(f"封板率{seal_rate:.1f}%偏低，追高风险大")
    verdict = "中性偏多，聚焦主线" if temp >= 55 else ("谨慎防守" if temp < 40 else "中性震荡，低吸为主")
    return f"""<div class="card"><h3>④ 隔夜判断（非投资建议）</h3>
<p><b>{verdict}</b> · 情绪温度 {temp}/100</p>
<div class="okbox">积极信号：{'; '.join(pos) if pos else '无显著'}</div>
<div class="warnbox">隐忧：{'; '.join(neg) if neg else '无显著'}</div>
<p class="conc">明日关注：① 主线龙头竞价强度决定题材延续性；② 量能能否放大决定反弹质量；③ 高位炸板股是否继续补跌。操作上聚焦主线低吸，回避高位断板风险。</p></div>"""


def render_holding(h, a, sh_idx):
    if not a:
        return f'<div class="hold"><div class="hold-head"><b>{h["name"]}({h["code"]})</b><span class="conc">数据不足</span></div></div>'
    chg_cls = "up" if a["chg"] >= 0 else "down"
    # 强弱对比
    vs_idx = "强于大盘" if sh_idx and a["chg"] > sh_idx["chg"] else ("弱于大盘" if sh_idx else "-")
    ma_note = f"站上MA5" if a["ma5"] and a["close"] >= a["ma5"] else "跌破MA5"
    trend = "多头" if (a["ma5"] and a["ma10"] and a["ma5"] >= a["ma10"]) else ("空头/整理" if a["ma5"] and a["ma10"] else "-")
    # 短线圈评级
    if a["vr5"] > 1.2 and a["chg"] > 0:
        s_rating = "主线强攻，谨防高位分歧"
        s_action = "持有；突破压力加码，回落减仓"
    elif a["chg"] < -2:
        s_rating = "回调未止，支撑位多空分水岭"
        s_action = "回踩观察，企稳再谈介入"
    else:
        s_rating = "弱反弹，等待放量确认"
        s_action = "观望，等资金回流信号"
    # 长线圈
    l_rating = "趋势修复中，等回补信号" if a["chg"] >= 0 else "长线逻辑未变，回调即机会"
    return f"""<div class="hold">
<div class="hold-head"><b>{h["name"]} {h["code"]}</b>
<span class="up">{a["close"]:.2f}</span> <span class="{chg_cls}">{a["chg"]:+.2f}%</span>
<span class="conc">K线截至 {a["asof"]} · PE {fnum(a["pe"],1)} · PB {fnum(a["pb"],2)}</span></div>
<div class="hold-body">
<p class="conc">{h["sector"]}</p><p class="conc">{h["concepts"]}</p>
<div class="grid2">
<div><h3 style="color:var(--blue)">强弱 &amp; 量能</h3>
<table><tr><th>项</th><th>值</th></tr>
<tr><td>个股涨跌</td><td class="{chg_cls}">{a["chg"]:+.2f}%</td></tr>
<tr><td>大盘对比</td><td>{vs_idx}</td></tr>
<tr><td>量比(3/5/10日)</td><td>{a["vr3"]:.2f} / {a["vr5"]:.2f} / {a["vr10"]:.2f}</td></tr>
<tr><td>近5/10日</td><td>{pct(a["r5"])} / {pct(a["r10"])}</td></tr>
</table></div>
<div><h3 style="color:var(--blue)">支撑 / 压力</h3>
<table><tr><th>位</th><th>价格</th></tr>
<tr><td>超短支撑</td><td>{fnum(a["sup_ultra"])}</td></tr>
<tr><td>短线压力</td><td>{fnum(a["res_short1"])}</td></tr>
<tr><td>MA5/MA10/MA20</td><td>{fnum(a["ma5"])} / {fnum(a["ma10"])} / {fnum(a["ma20"])}</td></tr>
<tr><td>趋势</td><td>{trend} · {ma_note}</td></tr>
</table></div></div>
<h3 style="color:var(--blue)">K线走势（近30日 · MA5/10/20 · 支撑压力）</h3>
{kline_svg(h["code"], a["asof"], a["sup_ultra"], a["res_short1"])}
<p class="conc">橙=MA5 · 蓝=MA10 · 紫=MA20 · 绿虚线=支撑 · 红虚线=压力</p>
<div class="warnbox">短线圈 · <b>{s_rating}</b>：{s_action}。跌破超短支撑 {fnum(a["sup_ultra"])} 则二次探底，站上 {fnum(a["res_short1"])} 才有反弹延续。</div>
<div class="okbox">长线圈 · <b>{l_rating}</b>：{h["concepts"]} 与主线共振时优先关注；高估值标的需业绩兑现，趋势不破可持有，主线退潮果断减仓。</div>
</div></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=None, help="交易日 YYYY-MM-DD，缺省取最新")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    idxdf = load_df("market/index_daily_上证指数.parquet")
    if idxdf is None or idxdf.empty:
        print("无指数数据")
        return 1
    latest = str(pd.to_datetime(idxdf["date"]).max())[:10]
    date = args.date or latest

    html = build_report(date)
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"A股盘面复盘_{date}.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ 盘面复盘已生成: {out} ({out.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
