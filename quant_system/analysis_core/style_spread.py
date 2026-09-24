#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""style_spread — 指数风格剪刀差复盘（大盘价值 vs 小盘成长）

设计依据（skills）:
  - elder-technical-analysis: 指数趋势/宽度验证（新高新低、趋势方向）
  - a-share-market-state-monitor: 五层看盘框架的趋势层/宽度层——风格判定回答
    "权重搭台还是题材唱戏"：大小盘是否共振、剪刀差方向与极端程度。
  - intraday-trading-strategies: 复盘输出的标准化表格。

数据源（data_warehouse/market/）:
  - index_daily_{name}.parquet : B3 按指数分文件日线（上证50/沪深300/创业板指/
    科创50/中证500/中证1000，由 scripts/fetch_index_daily.py 落盘），优先读取
  - index_daily.parquet : 沪深300 日线（已验证与 index_pe.parquet 的沪深300点位一致）
  - index_pe.parquet    : 上证50/沪深300/中证500/中证1000 点位（列 指数=收盘点位）
  - 创业板指(sz399006)/科创50(sh000688) 本地无文件 → akshare 直连兜底
    （ak.stock_zh_index_daily），失败则跳过并在报告中注明，
    小盘组回退以 中证1000/中证500 替代。

用法:
  python3 -m quant_system.analysis_core.style_spread
  python3 -m quant_system.analysis_core.style_spread --report
  python3 -m quant_system.analysis_core.style_spread --date 2026-08-07 --report
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace（与 analysis_core.config 同约定）
MARKET_DIR = ROOT / "data_warehouse" / "market"
# 报告输出目录：仓库根 generated/（本环境 workspace/generated 为只读，
# 现有 rs_strength/breakout 报告亦落在仓库 generated/）
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"

# 剪刀差判定阈值（百分点）：|大盘5日累计 - 小盘5日累计| 超过该值视为风格偏离
SPREAD_EPS = 0.5
SPREAD_EXTREME = 2.0

# (显示名, 类别, 本地源, akshare symbol)
# 本地源: index_daily=index_daily.parquet; index_pe=index_pe.parquet(列 指数);
#         none=本地无文件，仅 akshare 兜底
INDEX_SPECS = [
    ("上证50",   "大盘价值", "index_pe",    "sh000016"),
    ("沪深300",  "大盘价值", "index_daily", "sh000300"),
    ("创业板指", "小盘成长", "none",        "sz399006"),
    ("科创50",   "小盘成长", "none",        "sh000688"),
    ("中证500",  "中盘参考", "index_pe",    "sh000905"),
    ("中证1000", "小盘参考", "index_pe",    "sh000852"),
]

LARGE_GROUP = ("上证50", "沪深300")
SMALL_GROUP = ("创业板指", "科创50")
SMALL_FALLBACK = ("中证1000", "中证500")  # 创业板/科创 不可用时的替代


# ────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────
def _read_daily_close(p: Path) -> pd.Series | None:
    """通用 index_daily*.parquet 读取 → date 索引 close 序列（异常一律 None）。"""
    try:
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if not {"date", "close"}.issubset(df.columns):
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date")
        return df.set_index("date")["close"].astype(float).sort_index()
    except Exception:
        return None


def _load_named_index_daily(name: str) -> pd.Series | None:
    """B3: 按指数分文件日线 index_daily_{name}.parquet（优先源）。"""
    return _read_daily_close(MARKET_DIR / f"index_daily_{name}.parquet")


def _load_local_index(name: str, local: str) -> pd.Series | None:
    """本地指数收盘序列（date index -> close）。

    统一异常兜底：文件损坏/缺列/读取失败一律返回 None，由 load_index
    降级到 akshare 或跳过该指数，不向上抛异常。
    """
    try:
        if local == "index_daily":
            return _read_daily_close(MARKET_DIR / "index_daily.parquet")
        if local == "index_pe":
            p = MARKET_DIR / "index_pe.parquet"
            if not p.exists():
                return None
            df = pd.read_parquet(p)
            if not {"日期", "指数", "symbol"}.issubset(df.columns):
                return None
            df = df[df["symbol"] == name]
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.dropna(subset=["日期", "指数"]).drop_duplicates("日期")
            return df.set_index("日期")["指数"].astype(float).sort_index()
    except Exception:
        return None
    return None


def _load_akshare_index(symbol: str) -> pd.Series | None:
    """akshare 直连兜底：stock_zh_index_daily（date/open/high/low/close/volume）。"""
    import akshare as ak

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(8)
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
    finally:
        socket.setdefaulttimeout(old)
    if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date")
    return df.set_index("date")["close"].astype(float).sort_index()


def load_index(name: str, local: str, ak_symbol: str, target: pd.Timestamp
               ) -> tuple[pd.Series | None, str]:
    """加载指数序列（截断至 target），返回 (序列, 数据源说明)。"""
    note = ""
    local_err = ""
    s = None
    # 1) B3 优先: 按指数分文件日线 index_daily_{name}.parquet
    try:
        s = _load_named_index_daily(name)
        if s is not None and len(s) >= 6:
            note = f"index_daily_{name}.parquet(本地)"
    except Exception as e:
        local_err = str(e)[:60]
    # 2) 缺失时降级现有 index_pe/index_daily 逻辑
    if s is None or len(s) < 6:
        try:
            s_old = _load_local_index(name, local)
            if s_old is not None and len(s_old) >= 6:
                s = s_old
                note = "index_daily.parquet(本地)" if local == "index_daily" else (
                    "index_pe.parquet(本地)" if local == "index_pe" else "")
        except Exception as e:  # _load_local_index 已兜底，此处双保险
            local_err = local_err or str(e)[:60]
    # 3) akshare 直连兜底
    if s is None or len(s) < 6:
        try:
            s = _load_akshare_index(ak_symbol)
            note = "akshare 直连" + (f"（本地读取失败: {local_err}）" if local_err else "")
        except Exception as e:
            note = f"akshare 失败: {str(e)[:60]}"
    if s is None or len(s) < 6:
        return None, note + "（样本不足）"
    s = s[s.index <= target]
    if len(s) < 6:
        return None, note + f"（{target:%Y-%m-%d} 前样本不足）"
    return s, note


# ────────────────────────────────────────────────────────────
# 统计与风格判定
# ────────────────────────────────────────────────────────────
def index_stats(s: pd.Series) -> dict:
    """当日涨跌幅 + 5 日累计（%）。"""
    c = s.to_numpy(dtype=float)
    t, prev = c[-1], c[-2]
    pct1 = (t / prev - 1.0) * 100.0 if prev > 0 else np.nan
    pct5 = (t / c[-6] - 1.0) * 100.0 if len(c) >= 6 and c[-6] > 0 else np.nan
    return {"close": float(t), "pct1": float(pct1), "pct5": float(pct5),
            "asof": s.index[-1]}


def _mean_pct(rows: dict[str, dict], names: list[str], key: str) -> float | None:
    vals = [rows[n][key] for n in names if n in rows and np.isfinite(rows[n][key])]
    return float(np.mean(vals)) if vals else None


def judge_style(rows: dict[str, dict]) -> dict:
    """风格判定：大盘价值 vs 小盘成长（优先 5 日剪刀差）。"""
    large = _mean_pct(rows, list(LARGE_GROUP), "pct5")
    small_names = [n for n in SMALL_GROUP if n in rows]
    fallback = False
    if not small_names:
        small_names = [n for n in SMALL_FALLBACK if n in rows]
        fallback = True
    small = _mean_pct(rows, small_names, "pct5")
    if large is None or small is None:
        return {"style": "数据不足", "diff5": None, "diff1": None,
                "small_names": small_names, "fallback": fallback, "breadth": "—",
                "extreme": False}

    d5 = large - small
    d1 = _mean_pct(rows, list(LARGE_GROUP), "pct1") - _mean_pct(rows, small_names, "pct1")
    if d5 > SPREAD_EPS:
        style = "大盘价值占优"
    elif d5 < -SPREAD_EPS:
        style = "小盘成长占优"
    else:
        style = "风格均衡"
    if large > 0 and small > 0:
        breadth = "大小盘共振上行"
    elif large < 0 and small < 0:
        breadth = "大小盘共振回落"
    else:
        breadth = "大小盘分化"
    return {"style": style, "diff5": d5, "diff1": d1,
            "small_names": small_names, "fallback": fallback, "breadth": breadth,
            "extreme": bool(abs(d5) > SPREAD_EXTREME)}


# ────────────────────────────────────────────────────────────
# 输出
# ────────────────────────────────────────────────────────────
def _fmt(v: float | None) -> str:
    return f"{v:+.2f}%" if v is not None and np.isfinite(v) else "—"


def _latest_local_date() -> pd.Timestamp:
    """默认目标日：本地各指数最新数据日（akshare 不可用时的最近数据锚点）。"""
    candidates = []
    for name, _, local, _ in INDEX_SPECS:
        s = _load_named_index_daily(name)
        if s is None or not len(s):
            s = _load_local_index(name, local)
        if s is not None and len(s):
            candidates.append(s.index.max())
    return pd.Timestamp(max(candidates)).normalize() if candidates else pd.Timestamp.now().normalize()


def render_markdown(target: pd.Timestamp, rows: dict[str, dict], judge: dict,
                    notes: list[str]) -> str:
    date = target.strftime("%Y-%m-%d")
    style_line = (f"风格判定: **{judge['style']}** | 5日剪刀差(大盘-小盘) {_fmt(judge['diff5'])} "
                  f"| 当日剪刀差 {_fmt(judge['diff1'])} | 宽度: {judge['breadth']}")
    if judge.get("extreme"):
        style_line += " | ⚠ 风格极端分化"
    lines = [
        f"# 指数风格剪刀差复盘 — {date}",
        "",
        f"- {style_line}",
        f"- 小盘组: {'/'.join(judge['small_names'])}"
        + ("（创业板/科创50 数据不可用，以中证替代）" if judge["fallback"] else ""),
        "- 数据截至: 各指数自身最新交易日（见下表），日期为分析参考日",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "| 指数 | 类别 | 收盘点位 | 当日涨跌 | 5日累计 | 数据截至 | 数据源 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, _, local, ak_symbol in INDEX_SPECS:
        r = rows.get(name)
        if r is None:
            lines.append(f"| {name} | {_category(name)} | — | — | — | — | 不可用 |")
            continue
        lines.append(
            f"| {name} | {_category(name)} | {r['close']:.2f} | {_fmt(r['pct1'])} | "
            f"{_fmt(r['pct5'])} | {r['asof']:%Y-%m-%d} | {r['source']} |"
        )
    lines += ["", "---",
              "*解读（五层框架-趋势/宽度层）: 5日剪刀差 > +0.5pp 判大盘价值占优，"
              "< -0.5pp 判小盘成长占优，否则均衡；|剪刀差| ≥ 2pp 视为极端分化，追涨容错率低。*"]
    return "\n".join(lines) + "\n"


def _category(name: str) -> str:
    for n, cat, _, _ in INDEX_SPECS:
        if n == name:
            return cat
    return "—"


def main() -> None:
    ap = argparse.ArgumentParser(description="指数风格剪刀差复盘（大盘价值 vs 小盘成长）")
    ap.add_argument("--report", action="store_true", help="写入 generated/style_spread_{date}.md")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认各指数最新数据）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    args = ap.parse_args()

    if args.date:
        target = pd.Timestamp(args.date).normalize()
    else:
        target = _latest_local_date()
    print(f"[信息] 分析日期 {target.date()}（各指数取截至该日的最新数据）", flush=True)

    rows: dict[str, dict] = {}
    notes: list[str] = []
    for name, cat, local, ak_symbol in INDEX_SPECS:
        s, source = load_index(name, local, ak_symbol, target)
        if s is None:
            notes.append(f"{name}: {source}")
            print(f"[跳过] {name} — {source}")
            continue
        st = index_stats(s)
        st["source"] = source
        st["category"] = cat
        rows[name] = st
        print(f"[读取] {name:<6} 收{st['close']:>10.2f} 当日{_fmt(st['pct1']):>8} "
              f"5日{_fmt(st['pct5']):>8} 截至{st['asof']:%Y-%m-%d} ({source})")

    judge = judge_style(rows)
    extreme_note = " | ⚠ 风格极端分化" if judge.get("extreme") else ""
    print(f"\n[风格判定] {judge['style']} | 5日剪刀差 {_fmt(judge['diff5'])} "
          f"| 当日剪刀差 {_fmt(judge['diff1'])} | {judge['breadth']}{extreme_note}")
    if judge["fallback"]:
        print("[注] 创业板指/科创50 本地无文件且 akshare 不可用，小盘组以 "
              f"{'/'.join(judge['small_names'])} 替代")

    print("\n[指数涨跌幅表]")
    print(f"{'指数':<8}{'类别':<8}{'收盘':>10}{'当日涨跌':>10}{'5日累计':>10}{'截至':>12}{'数据源':<12}")
    for name, _, _, _ in INDEX_SPECS:
        r = rows.get(name)
        if r is None:
            print(f"{name:<8}{_category(name):<8}{'—':>10}{'—':>10}{'—':>10}{'—':>12}不可用")
            continue
        asof = f"{r['asof']:%Y-%m-%d}"
        print(f"{name:<8}{r['category']:<8}{r['close']:>10.2f}{_fmt(r['pct1']):>10}"
              f"{_fmt(r['pct5']):>10}{asof:>12} {r['source'][:16]:<16}")

    if args.report:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"style_spread_{target:%Y-%m-%d}.md"
        out.write_text(render_markdown(target, rows, judge, notes), encoding="utf-8")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
