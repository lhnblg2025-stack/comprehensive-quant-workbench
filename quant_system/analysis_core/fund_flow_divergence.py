#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fund_flow_divergence — 个股大单流向 vs 价格背离（东财资金流辅助增强）

方法（skills/mainline-volume-capital-flow-trading 的"真假承接/出货识别"落地为量化口径）:
  涨跌幅取自本地日K（收盘价近似）；主力/超大单净流入取自东财 push2delay 个股资金流
  日线接口（f52=主力净额, f56=超大单净额，单位=元），取目标交易日或最近交易日：
    涨 > +3% 且主力净流入 < 0  → ⚠️ 量价背离（涨但主力流出，疑似出货/对倒）
    跌 < -3% 且主力净流入 > 0  → ✅ 主力承接（跌但主力流入，疑似吸筹）
    其余                        → 正常

口径标注:
  东财资金流按"单笔成交金额"机械分桶（超大单>100万/大单20-100万/中单4-20万/小单<4万，
  双向统计）估算主力方向，是估算口径，仅供参考，不代表真实机构意图。

防御:
  - 接口/解析失败 → 该股标记 unavailable，不中断整体
  - 代码前缀归一化: SH600519→1.600519, SZ000001→0.000001, 920/4/8 北交所→标注跳过
  - 本地 kline 缺失 → 涨跌幅不可用，判定为数据不足

数据契约:
  data_warehouse/kline/{6位代码}.parquet  日K（当日涨跌幅）
  东财 fflow/kline/get（klt=101 日线，lmt=0 全量）

用法:
  python3 -m quant_system.analysis_core.fund_flow_divergence --watch 600519
  python3 -m quant_system.analysis_core.fund_flow_divergence            # 缺省读 config/watchlist.json
  python3 -m quant_system.analysis_core.fund_flow_divergence --watch 600519 --date 2026-08-07
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.common import load_names, read_kline_window  # noqa: E402

KLINE_DIR = ROOT / "data_warehouse" / "kline"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
WATCHLIST_FILE = CONFIG_DIR / "watchlist.json"
WATCHLIST_TEMPLATE = {"watch": ["600519", "000001"]}

CODE_RE = re.compile(r"^\d{6}$")
UP_PCT = 3.0        # 上涨阈值 %
DOWN_PCT = -3.0     # 下跌阈值 %

FFLOW_URL = ("https://push2delay.eastmoney.com/api/qt/stock/fflow/kline/get"
             "?secid={secid}&fields1=f1,f2,f3,f7"
             "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
             "&klt=101&lmt=0&ut=b2884a393a59ad64002292a3e90d46a5")
FFLOW_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}

# fields2 顺序: f51=日期, f52=主力净额, f53=小单净额, f54=中单净额, f55=大单净额, f56=超大单净额
IDX_DATE = 0
IDX_MAIN = 1
IDX_SUPER = 5


# ────────────────────────────────────────────────────────────
# watchlist 配置（缺省 watch 列表）
# ────────────────────────────────────────────────────────────
def load_watchlist_config(config_file: Path | None = None) -> list[str]:
    """读 config/watchlist.json（{"watch": ["600519", "000001"]}）。

    文件不存在则创建模板并返回模板代码；格式非法/为空返回 []。
    """
    path = Path(config_file) if config_file else WATCHLIST_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(WATCHLIST_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            return [str(c).zfill(6) for c in WATCHLIST_TEMPLATE["watch"]]
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("watch") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    codes = [str(c).strip().zfill(6) for c in raw if str(c).strip()]
    return [c for c in codes if CODE_RE.match(c)]


# ────────────────────────────────────────────────────────────
# 代码归一化
# ────────────────────────────────────────────────────────────
def normalize_code(code: str) -> tuple[str | None, str]:
    """6位/带前缀代码 → 东财 secid（"1.600519"/"0.000001"）。

    规则: 6/9 开头=沪→1；0/3 开头=深→0；4/8/92 开头=北交所→跳过；其余无法识别。
    返回 (secid 或 None, 错误说明)。
    """
    c = str(code).strip().upper().replace(".", "")
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    if not c.isdigit():
        return None, "代码非数字"
    if c.startswith(("4", "8", "92")):
        return None, "北交所跳过"
    if c.startswith(("6", "9")):
        return f"1.{c}", ""
    if c.startswith(("0", "3")):
        return f"0.{c}", ""
    return None, "无法识别市场前缀"


# ────────────────────────────────────────────────────────────
# 东财资金流拉取
# ────────────────────────────────────────────────────────────
def fetch_fund_flow(secid: str, target_date: pd.Timestamp | None = None,
                    timeout: float = 8.0) -> dict:
    """拉取东财个股资金流日线，返回最近（或目标）交易日主力/超大单净额（元）。"""
    url = FFLOW_URL.format(secid=secid)
    r = requests.get(url, headers=FFLOW_HEADERS, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return {"ok": False, "error": "接口无数据"}
    rows = [ln.split(",") for ln in klines if ln]
    if not rows:
        return {"ok": False, "error": "接口行解析失败"}

    want = str(target_date.date()) if target_date is not None else None
    pick = rows[-1]
    for row in reversed(rows):
        if len(row) > IDX_SUPER and row[IDX_DATE] == want:
            pick = row
            break

    def _num(x: str) -> float | None:
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    main_net = _num(pick[IDX_MAIN]) if len(pick) > IDX_MAIN else None
    super_net = _num(pick[IDX_SUPER]) if len(pick) > IDX_SUPER else None
    if main_net is None:
        return {"ok": False, "error": "主力净额字段缺失"}
    return {"ok": True, "error": "", "date": pick[IDX_DATE],
            "main_net": main_net, "super_net": super_net}


# ────────────────────────────────────────────────────────────
# 腾讯云国内IP通道数据（固定通道, 2026-08-22）
# ────────────────────────────────────────────────────────────
FUND_FLOW_DIR = ROOT / "data_warehouse" / "market" / "fund_flow"

_ff_cache: dict[str, dict[str, dict]] | None = None  # {code: {date: row}}


def _load_ff_channel() -> dict[str, dict[str, dict]]:
    """读 data_warehouse/market/fund_flow/fund_flow_*.parquet（云端 crawler 回传）。

    返回 {code: {date(YYYY-MM-DD): {main_net, super_net}}}；无数据 → {}。"""
    global _ff_cache
    if _ff_cache is not None:
        return _ff_cache
    _ff_cache = {}
    try:
        import glob as _gff
        files = sorted(_gff.glob(str(FUND_FLOW_DIR / "fund_flow_*.parquet")))
        if not files:
            return _ff_cache
        for fp in files:
            df = pd.read_parquet(fp)
            if not {"code", "date", "main_net"}.issubset(df.columns):
                continue
            for r in df.itertuples(index=False):
                code = str(r.code).zfill(6)
                d = str(r.date)[:10]
                _ff_cache.setdefault(code, {})[d] = {
                    "main_net": float(r.main_net),
                    "super_net": float(getattr(r, "super_net", 0) or 0),
                }
    except Exception:
        _ff_cache = {}
    return _ff_cache


def fetch_fund_flow_channel(code: str, target_date: pd.Timestamp | None = None) -> dict:
    """通道优先: 本地回传的云端资金流数据（国内IP抓取, 境外直连替代）。

    命中目标日期（或最近一日）→ 返回与 fetch_fund_flow 同构 dict；无 → {"ok": False}。"""
    try:
        tbl = _load_ff_channel()
        rows = tbl.get(str(code).zfill(6))
        if not rows:
            return {"ok": False, "error": "通道无该股"}
        if target_date is not None:
            key = str(pd.Timestamp(target_date).date())
            row = rows.get(key)
        else:
            # 取最新日期
            key = max(rows.keys())
            row = rows[key]
        if row is None:
            # 无目标日 → 取最近一日
            key = max(rows.keys())
            row = rows[key]
        return {"ok": True, "error": "", "date": key,
                "main_net": row["main_net"], "super_net": row["super_net"],
                "source": "cloud_channel"}
    except Exception as e:
        return {"ok": False, "error": f"通道读取失败: {str(e)[:60]}"}


def fetch_fund_flow_by_code(code: str, target_date: pd.Timestamp | None = None,
                            timeout: float = 8.0) -> dict:
    """通道优先 → 直连东财兜底。失败/跳过返回防御性 dict，不抛异常。"""
    secid, note = normalize_code(code)
    if secid is None:
        return {"ok": False, "skipped": True, "note": note, "error": note}
    # 1) 腾讯云国内IP通道（crawler_fund_flow 每日回传）
    ch = fetch_fund_flow_channel(code, target_date)
    if ch.get("ok"):
        ch["code"] = code
        return ch
    # 2) 直连东财（境外IP可能断连, 失败留给上层标记 unavailable）
    try:
        out = fetch_fund_flow(secid, target_date, timeout=timeout)
        out["code"] = code
        return out
    except Exception as e:
        return {"ok": False, "unavailable": True, "error": f"{type(e).__name__}: {str(e)[:80]}",
                "code": code}


# ────────────────────────────────────────────────────────────
# 背离判定
# ────────────────────────────────────────────────────────────
def judge_divergence(pct_chg: float | None, main_net: float | None) -> str:
    """涨跌幅 + 主力净额 → 背离标记。任一缺失 → 数据不足。"""
    if pct_chg is None or main_net is None or not (np.isfinite(pct_chg) and np.isfinite(main_net)):
        return "数据不足"
    if pct_chg > UP_PCT and main_net < 0:
        return "⚠️ 量价背离(涨但主力流出，疑似出货/对倒)"
    if pct_chg < DOWN_PCT and main_net > 0:
        return "✅ 主力承接(跌但主力流入，疑似吸筹)"
    return "正常"


def local_pct_chg(code: str, target: pd.Timestamp | None = None) -> tuple[float | None, str]:
    """本地日K最近两日收盘近似涨跌幅（%）。缺失返回 (None, 原因)。"""
    path = KLINE_DIR / f"{code}.parquet"
    if not path.exists():
        return None, "kline文件缺失"
    window = (pd.Timestamp(target) if target is not None else pd.Timestamp.now()).normalize()
    try:
        df = read_kline_window(path, ["date", "close"], window - pd.Timedelta(days=15))
    except Exception as e:
        return None, f"kline读取失败: {str(e)[:60]}"
    if df is None or df.empty:
        return None, "kline无数据"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    df = df[df["date"] <= window]
    if len(df) < 2:
        return None, "kline样本不足"
    c0 = float(df["close"].astype(float).iloc[-2])
    c1 = float(df["close"].astype(float).iloc[-1])
    if c0 <= 0:
        return None, "kline前收无效"
    return (c1 / c0 - 1.0) * 100.0, ""


# ────────────────────────────────────────────────────────────
# 格式化与输出
# ────────────────────────────────────────────────────────────
def fmt_amount(v: float | None) -> str:
    """元 → 亿/万 可读格式。"""
    if v is None or not np.isfinite(v):
        return "—"
    if abs(v) >= 1e8:
        return f"{v / 1e8:+.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:+.1f}万"
    return f"{v:+.0f}元"


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2f}%" if v is not None and np.isfinite(v) else "—"


def analyze_one(code: str, target: pd.Timestamp | None) -> dict:
    """单股: 本地涨跌幅 + 东财资金流 → 判定。"""
    pct, pct_note = local_pct_chg(code, target)
    ff = fetch_fund_flow_by_code(code, target)
    row = {"code": code, "pct_chg": pct, "pct_note": pct_note, "ff": ff}
    if ff.get("skipped"):
        row["verdict"] = ff.get("note", "跳过")
        row["status"] = "skip"
    elif not ff.get("ok"):
        row["verdict"] = "unavailable"
        row["status"] = "unavailable"
    else:
        row["verdict"] = judge_divergence(pct, ff.get("main_net"))
        row["status"] = "ok" if row["verdict"] != "数据不足" else "partial"
    return row


def check_divergence(code: str, target_date: pd.Timestamp | None = None) -> dict:
    """watch_card 兼容契约: 单股本地涨跌幅 + 东财资金流 → 背离信号。

    返回 {"code", "pct_chg", "main_net", "super_net", "date", "signal", "note"}。
    接口失败 → signal="unavailable", note=错误摘要（不抛异常）。
    """
    pct, pct_note = local_pct_chg(code, target_date)
    ff = fetch_fund_flow_by_code(code, target_date)
    if ff.get("skipped"):
        return {"code": code, "pct_chg": pct, "main_net": None, "super_net": None,
                "date": None, "signal": ff.get("note", "跳过"), "note": ff.get("note", "")}
    if not ff.get("ok"):
        return {"code": code, "pct_chg": pct, "main_net": None, "super_net": None,
                "date": None, "signal": "unavailable",
                "note": str(ff.get("error", "接口失败"))}
    return {"code": code, "pct_chg": pct, "main_net": ff.get("main_net"),
            "super_net": ff.get("super_net"), "date": ff.get("date"),
            "signal": judge_divergence(pct, ff.get("main_net")), "note": pct_note}


def render_terminal(rows: list[dict], names: dict[str, str], target: pd.Timestamp) -> str:
    lines = [f"[大单流向背离] {target:%Y-%m-%d} | 源: 东财push2delay fflow/kline（f52主力/f56超大单，元）"]
    lines += ["  代码    名称         资金日期       涨跌幅     主力净额       超大单净额      判定"]
    for r in rows:
        code = r["code"]
        name = names.get(code, "")[:6]
        ff = r["ff"]
        if r["status"] == "skip":
            lines.append(f"  {code:<6} {name:<10} —            —          —             —             {ff.get('note', '跳过')}")
            continue
        if r["status"] == "unavailable":
            lines.append(f"  {code:<6} {name:<10} —            {_fmt_pct(r['pct_chg']):>8} "
                         f"{'—':>12} {'—':>14}  unavailable（{ff.get('error', '接口失败')[:30]}）")
            continue
        lines.append(f"  {code:<6} {name:<10} {ff.get('date', '—'):<12} {_fmt_pct(r['pct_chg']):>8} "
                     f"{fmt_amount(ff.get('main_net')):>12} {fmt_amount(ff.get('super_net')):>14}  {r['verdict']}")
    lines += ["---",
              "*口径: 东财按单笔金额机械分桶估算主力/超大单方向（f52主力=f55大单+f56超大单），"
              "仅供参考，不代真实机构意图。*"]
    return "\n".join(lines)


def render_markdown(rows: list[dict], names: dict[str, str], target: pd.Timestamp) -> str:
    lines = [f"# 大单流向背离 — {target:%Y-%m-%d}", "",
             f"- 股票数: {len(rows)} | 数据源: 东财 push2delay 个股资金流日线（f52主力/f56超大单，单位=元）",
             "- 涨跌幅: 本地日K收盘价近似口径", "",
             "| 代码 | 名称 | 资金日期 | 涨跌幅 | 主力净额 | 超大单净额 | 判定 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        code = r["code"]
        name = names.get(code, "")
        ff = r["ff"]
        if r["status"] == "skip":
            lines.append(f"| {code} | {name} | — | — | — | — | {ff.get('note', '跳过')} |")
            continue
        if r["status"] == "unavailable":
            lines.append(f"| {code} | {name} | — | {_fmt_pct(r['pct_chg'])} | — | — | "
                         f"unavailable（{ff.get('error', '接口失败')[:50]}） |")
            continue
        lines.append(f"| {code} | {name} | {ff.get('date', '—')} | {_fmt_pct(r['pct_chg'])} | "
                     f"{fmt_amount(ff.get('main_net'))} | {fmt_amount(ff.get('super_net'))} | {r['verdict']} |")
    lines += ["---",
              "*口径: 东财资金流按单笔成交金额机械分桶估算（主力=大单+超大单），为估算口径，"
              "仅供参考，不代表真实机构意图。*"]
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="个股大单流向 vs 价格背离")
    ap.add_argument("--watch", default=None, help="股票代码，逗号分隔（缺省读 config/watchlist.json）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认 kline 最新交易日）")
    ap.add_argument("--no-report", action="store_true", help="不写 generated/fund_flow_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录（默认仓库 generated/）")
    args = ap.parse_args()

    if args.watch:
        codes = [c.strip().zfill(6) for c in args.watch.split(",") if c.strip()]
        bad = [c for c in codes if not CODE_RE.match(c)]
        if bad:
            print(f"[错误] 非法代码: {','.join(bad)}（需6位数字）")
            sys.exit(1)
        if not codes:
            print("[错误] --watch 不能为空")
            sys.exit(1)
    else:
        codes = load_watchlist_config()
        if not codes:
            print("[错误] config/watchlist.json 缺失/格式非法，且未提供 --watch")
            sys.exit(1)
        print(f"[提示] --watch 缺省，读取 {WATCHLIST_FILE}")

    target = pd.Timestamp(args.date).normalize() if args.date else None
    if target is None:
        # 用所有 watch 股票本地 kline 的最新共同日期；无 kline 时回退今天
        cands = []
        for code in codes:
            p = KLINE_DIR / f"{code}.parquet"
            if not p.exists():
                continue
            try:
                d = pd.read_parquet(p, columns=["date"])
                m = pd.to_datetime(d["date"], errors="coerce").max()
                if m is not None and pd.notna(m):
                    cands.append(m)
            except Exception as e:
                logging.getLogger(__name__).error(f"[fund_flow_divergence] 操作失败: {e}", exc_info=True)
                continue
        target = pd.Timestamp(min(cands)).normalize() if cands else pd.Timestamp.now().normalize()
    else:
        target = target.normalize()

    names, _ = load_names()
    t0 = time.time()
    rows = [analyze_one(c, target) for c in codes]
    elapsed = time.time() - t0
    ok_n = sum(1 for r in rows if r["status"] == "ok")

    print(f"[大单流向背离] {target.date()} | {len(codes)} 只（可用 {ok_n}）| 耗时 {elapsed:.1f}s")
    print()
    print(render_terminal(rows, names, target))

    if not args.no_report:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"fund_flow_{target:%Y-%m-%d}.md"
        out.write_text(render_markdown(rows, names, target), encoding="utf-8")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
