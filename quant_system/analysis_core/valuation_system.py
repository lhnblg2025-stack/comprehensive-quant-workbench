#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""valuation_system — 体系8 基本面估值系统（V11 N8）

方法论（skills/dcf-valuation-mastery + earnings-quality-analysis +
dcf-value-driver-sensitivity + factor-investing-framework）:
  - DCF: 自由现金流折现（简版 Gordon）: FCF 5年均值 ÷ (WACC 10% - g 3%)
         → 内在价值 vs 现价 折价/溢价%，附 WACC×g 敏感性（value-driver-sensitivity）。
  - 盈余质量（Penman）: 现金流/利润比、应收周转 vs 营收、ROE、毛利率稳定性 → 0-10 分。
  - 因子框架: 价值因子（PE/PB/PS）近5年分位 → 低估(<30%)/合理(30-70%)/高估(>70%)。

五层检测（每只标的）:
  1. 估值分位: PE/PB/PS 近5年分位（报告期口径，价格×每股/TTM 归母净利派生）。
  2. 盈利质量评分: 经营现金流/净利>1、应收增速<营收增速、ROE>10%、毛利率稳定 → 0-10分。
  3. 业绩预告: 读 data_warehouse 业绩预告 parquet（缺失则用净利同比代理）→ '业绩支撑'/'业绩雷'。
  4. DCF 简版: FCF 5年均值 ÷ (WACC 10% - g 3%) → 内在价值 vs 现价 折价/溢价%。
  5. 综合: 估值分位 × 盈利质量 × 业绩方向 → 低估优质(看多)/高估劣质(看空)/中性。

防御:
  - 财务数据缺失 → 标注 '无财务数据'；单标的任一环节失败 → 跳过该股，不影响整体。
  - 财务数据可能陈旧（季报口径）→ 每只标的标注 as_of（最新报告期）与快照日期。
  - 本地库仅一期快照（如 601899）→ 估值分位/DCF 标注样本不足，不硬算。
  - RAG 索引未构建/检索失败 → 方法论依据标注不可用，不中断。

数据契约（全部本地优先，零网络）:
  ~/.quant_system/financial.db  financial_indicators（80指标×多期，含 pe_ttm/pb/ps_ttm）
  data_store（SQLite/parquet 日K，近5年）→ 价格与估值分位
  data_warehouse/**/业绩预告 parquet（可选；缺失降级净利同比代理）

统一接口:
  from quant_system.analysis_core.valuation_system import ValuationSystem
  vs = ValuationSystem(watch=["601899", "600519"])
  vs.detect() / vs.view() / vs.report()

用法:
  python3 -m quant_system.analysis_core.valuation_system --watch 601899,600519
  python3 -m quant_system.analysis_core.valuation_system --watch 601899,600519 --json
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # RAG 本地向量模型，避免网络重试

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace 上级（quant_system 包所在）
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.rs_strength import load_names  # noqa: E402

CST = timezone(timedelta(hours=8))

FIN_DB = Path.home() / ".quant_system" / "financial.db"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "generated"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
WATCHLIST_FILE = CONFIG_DIR / "watchlist.json"
WATCHLIST_TEMPLATE = {"watch": ["600519", "000001"]}
CODE_RE = re.compile(r"^\d{6}$")

# ── 方法论常量 ────────────────────────────────────────────
WACC = 0.10          # 折现率（简版取固定 10%）
GROWTH_G = 0.03      # 永续增长（低于名义经济增速）
PCT_LOW, PCT_HIGH = 0.30, 0.70   # 估值分位 低估/高估 阈值
PCT_MIN_N = 8        # 分位所需最少样本数（报告期点数）
HIST_YEARS = 5       # 估值分位窗口
DCF_MIN_YEARS = 3    # DCF 所需最少年度 FCF 样本
QUALITY_OK = 7.0     # 盈利质量 优质 阈值（0-10）
QUALITY_BAD = 4.0    # 盈利质量 劣质 阈值
COVERED_MIN = 2      # 至少覆盖的子项数，少于则质量等级标注"数据不足"
PRICE_LOOKBACK_DAYS = 1300  # ≈5 个交易年（估值分位/DCF 用日K）

RAG_QUERY = "DCF 估值 盈利质量 业绩预告"
RAG_K = 3


# ════════════════════════════════════════════════════════════
# 本地财务数据（financial.db，只读，离线可用）
# ════════════════════════════════════════════════════════════
def _db() -> sqlite3.Connection | None:
    if not FIN_DB.exists():
        return None
    try:
        return sqlite3.connect(str(FIN_DB), timeout=5)
    except Exception:
        return None


def _norm_date(v: object) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(str(v), errors="coerce")
        return ts if pd.notna(ts) else None
    except Exception:
        return None


def _is_report_period(d: object) -> bool:
    """报告期日期（YYYYMMDD 8位）优先；'YYYY-MM-DD' 视为数据快照日期。"""
    s = str(d)
    return len(s) == 8 and "-" not in s


def _indicator_history(code: str, indicator: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """financial_indicators 单指标全历史 → DataFrame[date, value]（按日期升序）。

    as_of 非空时截断 date ≤ as_of（防前视：不读分析日之后的财务/快照）。
    """
    db = _db()
    if db is None:
        return pd.DataFrame(columns=["date", "value"])
    try:
        rows = db.execute(
            "SELECT date, value FROM financial_indicators "
            "WHERE symbol = ? AND indicator = ?",
            (code, indicator),
        ).fetchall()
        db.close()
    except Exception:
        return pd.DataFrame(columns=["date", "value"])
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows, columns=["date", "value"])
    # 报告期标记: 原始日期串为 8 位无横线（如 20260331）→ True；快照（如 2026-07-29）→ False
    df["is_period"] = [len(str(r[0])) == 8 and "-" not in str(r[0]) for r in rows]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).drop_duplicates("date").sort_values("date")
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    return df


def _latest_report(df: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
    """最新报告期取值（报告期 YYYYMMDD 优先，快照兜底）。返回 (报告期, 值)。"""
    if df.empty:
        return None, None
    d = df[df["is_period"]] if "is_period" in df.columns else df
    if d.empty:
        d = df
    last = d.iloc[-1]
    return pd.Timestamp(last["date"]), float(last["value"])


def _load_indicators(code: str, indicators: list[str],
                     as_of: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    return {ind: _indicator_history(code, ind, as_of=as_of) for ind in indicators}


def _ttm_from_cum(hist: pd.DataFrame) -> pd.DataFrame:
    """累计口径归母净利/营收 → TTM 序列（报告期粒度）。

    ttm(t) = 累计(t) + 上年年报 - 上年同期累计
    """
    if hist.empty:
        return hist
    h = hist.copy()
    h["year"] = h["date"].dt.year
    h["month"] = h["date"].dt.month
    h["ymd"] = h["date"].dt.strftime("%m%d")
    annual = h[h["ymd"] == "1231"].set_index("year")["value"]
    rows = []
    for _, r in h.iterrows():
        prev_annual = annual.get(r["year"] - 1)
        same_prev = h[(h["year"] == r["year"] - 1) & (h["ymd"] == r["ymd"])]
        if prev_annual is None or same_prev.empty:
            continue
        same_prev_v = float(same_prev.iloc[0]["value"])
        if pd.isna(prev_annual) or pd.isna(same_prev_v):
            continue
        rows.append({"date": r["date"], "value": float(r["value"]) + float(prev_annual) - same_prev_v})
    out = pd.DataFrame(rows, columns=["date", "value"]).dropna()
    return out


def _shares_series(np_hist: pd.DataFrame, eps_hist: pd.DataFrame) -> pd.Series:
    """股本序列: shares = 归母净利累计 / 每股收益（同报告期口径）。"""
    if np_hist.empty or eps_hist.empty:
        return pd.Series(dtype=float)
    m = pd.merge(np_hist, eps_hist, on="date", suffixes=("_np", "_eps"))
    if m.empty:
        return pd.Series(dtype=float)
    denom = m["value_eps"].replace(0, np.nan)
    shares = pd.Series(m["value_np"].to_numpy() / denom.to_numpy(),
                       index=pd.DatetimeIndex(m["date"]))
    shares = shares.dropna()
    return shares[shares > 0]


# ════════════════════════════════════════════════════════════
# 价格（data_store 近5年日K，零网络优先）
# ════════════════════════════════════════════════════════════
def _price_history(code: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """本地日K（≤as_of，防前视：分析日当天可用，未来行情剔除）。"""
    try:
        from quant_system.data_store import get_store
        df = get_store().get(code, days=PRICE_LOOKBACK_DAYS, force_refresh=False)
    except Exception:
        df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce"),
                        "close": pd.to_numeric(df["close"], errors="coerce")})
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    if as_of is not None:
        out = out[out["date"] <= pd.Timestamp(as_of)]
    return out


def _price_at(price_df: pd.DataFrame, ts: pd.Timestamp, max_gap_days: int = 60) -> float | None:
    """报告期后最近交易日收盘价（缺价/间隔过久 → None）。"""
    if price_df.empty:
        return None
    f = price_df[price_df["date"] >= ts]
    if f.empty:
        return None
    row = f.iloc[0]
    if (row["date"] - ts).days > max_gap_days:
        return None
    return float(row["close"])


# ════════════════════════════════════════════════════════════
# 1. 估值分位（PE/PB/PS 近5年，报告期口径）
# ════════════════════════════════════════════════════════════
def _percentile(series: pd.Series, v: float | None) -> float | None:
    s = series.dropna()
    if len(s) < PCT_MIN_N or v is None or not np.isfinite(v):
        return None
    return float((s < v).mean())


def _bias(pct: float | None) -> str | None:
    if pct is None:
        return None
    if pct < PCT_LOW:
        return "低估"
    if pct > PCT_HIGH:
        return "高估"
    return "合理"


def valuation_percentile(code: str, inds: dict[str, pd.DataFrame],
                         price_df: pd.DataFrame) -> dict:
    """PE/PB/PS 近5年分位 + 低估/合理/高估 标注。

    口径: 报告期累计净利/营收 → TTM；shares = 归母净利/每股收益；
    pe = 价格×shares/TTM净利，pb = 价格/每股净资产，ps = 价格×shares/TTM营收。
    历史点位取报告期后最近交易日价格（≤60天），窗口=近5年。
    """
    out = {"pe": None, "pb": None, "ps": None, "pe_pct": None, "pb_pct": None,
           "ps_pct": None, "bias": None, "bias_detail": "", "n": 0}
    np_hist = inds.get("net_profit_parent")
    rev_hist = inds.get("revenue")
    eps_hist = inds.get("eps")
    bvps_hist = inds.get("bvps")
    if price_df.empty:
        out["bias_detail"] = "行情历史不足(本地K线缺失)"
        return out
    if np_hist is None or np_hist.empty:
        out["bias_detail"] = "财务历史不足(仅快照, 无多期净利)"
        return out

    np_ttm = _ttm_from_cum(np_hist)
    rev_ttm = _ttm_from_cum(rev_hist) if rev_hist is not None else pd.DataFrame()
    shares = _shares_series(np_hist, eps_hist)
    if np_ttm.empty or shares.empty:
        out["bias_detail"] = "净利/股本历史不足(TTM派生失败)"
        return out

    latest_np_date = pd.Timestamp(np_ttm["date"].max())
    win_start = latest_np_date - pd.Timedelta(days=365 * HIST_YEARS)
    np_ttm = np_ttm[np_ttm["date"] >= win_start]
    rev_ttm = rev_ttm[rev_ttm["date"] >= win_start] if not rev_ttm.empty else rev_ttm

    pe_s, pb_s, ps_s = [], [], []
    for _, r in np_ttm.iterrows():
        t = pd.Timestamp(r["date"])
        px = _price_at(price_df, t)
        sh = shares.get(t)
        if px is None or pd.isna(sh):
            continue
        npv, revv = float(r["value"]), None
        if npv > 0:
            pe_s.append(px * sh / npv)
        if not rev_ttm.empty:
            m = rev_ttm[rev_ttm["date"] == t]
            if not m.empty and float(m.iloc[0]["value"]) > 0:
                revv = float(m.iloc[0]["value"])
                ps_s.append(px * sh / revv)
        if bvps_hist is not None:
            b = bvps_hist[bvps_hist["date"] == t]
            if not b.empty and float(b.iloc[0]["value"]) > 0:
                pb_s.append(px / float(b.iloc[0]["value"]))
    out["n"] = max(len(pe_s), len(pb_s), len(ps_s))
    if out["n"] < PCT_MIN_N:
        out["bias_detail"] = f"近5年报告期样本 {out['n']}<{PCT_MIN_N}，分位不适用"
        return out

    # 当前点: 最新报告期 TTM × 最新价
    px_now = float(price_df["close"].iloc[-1])
    t_now = pd.Timestamp(np_ttm["date"].max())
    sh_now = shares.get(t_now)
    np_now = float(np_ttm[np_ttm["date"] == t_now].iloc[0]["value"])
    out["pe"] = round(px_now * sh_now / np_now, 2) if (sh_now and np_now > 0) else None
    out["pb"] = None
    if bvps_hist is not None and not bvps_hist.empty:
        b = bvps_hist[bvps_hist["date"] == t_now]
        if not b.empty and float(b.iloc[0]["value"]) > 0:
            out["pb"] = round(px_now / float(b.iloc[0]["value"]), 2)
    out["ps"] = None
    if not rev_ttm.empty:
        m = rev_ttm[rev_ttm["date"] == t_now]
        if not m.empty and float(m.iloc[0]["value"]) > 0 and sh_now:
            out["ps"] = round(px_now * sh_now / float(m.iloc[0]["value"]), 2)

    pe_sr, pb_sr, ps_sr = pd.Series(pe_s), pd.Series(pb_s), pd.Series(ps_s)
    out["pe_pct"] = _percentile(pe_sr, out["pe"])
    out["pb_pct"] = _percentile(pb_sr, out["pb"])
    out["ps_pct"] = _percentile(ps_sr, out["ps"])

    biases = [_bias(x) for x in (out["pe_pct"], out["pb_pct"], out["ps_pct"])]
    biases = [b for b in biases if b]
    if biases:
        from collections import Counter
        out["bias"] = Counter(biases).most_common(1)[0][0]
        detail = []
        if out["pe_pct"] is not None:
            detail.append(f"PE分位{out['pe_pct']:.0%}")
        if out["pb_pct"] is not None:
            detail.append(f"PB分位{out['pb_pct']:.0%}")
        if out["ps_pct"] is not None:
            detail.append(f"PS分位{out['ps_pct']:.0%}")
        out["bias_detail"] = "，".join(detail)
    return out


# ════════════════════════════════════════════════════════════
# 2. 盈利质量评分（Penman 盈余质量，0-10）
# ════════════════════════════════════════════════════════════
def _ratio_score(v: float | None, bins: list[tuple[float, float]]) -> float:
    """按阈值档位给分（bins: [(下限, 得分), ...] 取第一个命中的档）。"""
    if v is None or not np.isfinite(v):
        return 0.0
    for lo, score in bins:
        if v >= lo:
            return score
    return 0.0


def quality_score(inds: dict[str, pd.DataFrame]) -> dict:
    """四项各 2.5 分，合计 0-10；covered=有数据子项数（<2 时等级标注数据不足）。

    1. 经营现金流/净利 > 1        —— ocf_to_net_profit（缺则 TTM 口径推算）
    2. 应收增速 < 营收增速        —— 应收周转率同季同比不降（周转降=应收增速偏高）
    3. ROE > 10%                 —— 加权/摊薄 ROE
    4. 毛利率稳定                 —— 近8期毛利率标准差小
    """
    comps: dict[str, dict] = {}
    covered = 0

    # 1. 现金含量
    _, ocf_np = _latest_report(inds.get("ocf_to_net_profit", pd.DataFrame()))
    if ocf_np is None:
        ocf_t, ocf_v = _latest_report(inds.get("ocf_total", pd.DataFrame()))
        np_t, np_v = _latest_report(inds.get("net_profit_parent", pd.DataFrame()))
        if ocf_v is not None and np_v is not None and np_v > 0:
            ocf_np = ocf_v / np_v
            comps["cash"] = {"note": "OCF/归母净利(报告期累计口径)"}
    else:
        comps["cash"] = {"note": "OCF/净利(库内指标)"}
    if ocf_np is not None:
        covered += 1
        comps["cash"].update({"value": round(ocf_np, 3),
                              "score": _ratio_score(ocf_np, [(1.0, 2.5), (0.7, 1.5), (0.5, 0.8)]),
                              "ok": ocf_np > 1})
    else:
        comps["cash"] = {"score": 0.0, "ok": None, "note": "缺现金流/净利数据"}

    # 2. 应收 vs 营收（周转率同季同比）
    rec_t, rec_v = _latest_report(inds.get("receivables_turnover", pd.DataFrame()))
    rec_prev = None
    if rec_v is not None and rec_t is not None:
        h = inds.get("receivables_turnover", pd.DataFrame())
        same_q = h[(h["date"].dt.year == rec_t.year - 1) &
                   (h["date"].dt.strftime("%m%d") == rec_t.strftime("%m%d"))]
        if not same_q.empty:
            rec_prev = float(same_q.iloc[0]["value"])
    if rec_v is not None and rec_prev is not None and rec_prev > 0:
        covered += 1
        chg = rec_v / rec_prev - 1.0
        comps["receivables"] = {
            "value": round(rec_v, 1), "yoy": round(chg, 3),
            "score": _ratio_score(1 - chg, [(0.95, 2.5), (0.80, 1.25)]),
            "ok": chg >= -0.05, "note": f"应收周转率同季同比{chg:+.1%}"}
    else:
        comps["receivables"] = {"score": 0.0, "ok": None, "note": "缺应收周转历史"}

    # 3. ROE
    roe_d, roe_v = _latest_report(inds.get("roe", pd.DataFrame()))
    if roe_v is None:
        roe_d, roe_v = _latest_report(inds.get("roe_full", pd.DataFrame()))
    if roe_v is not None:
        covered += 1
        comps["roe"] = {"value": round(roe_v, 2),
                        "score": _ratio_score(roe_v, [(10.0, 2.5), (8.0, 1.5), (5.0, 0.8)]),
                        "ok": roe_v > 10, "note": f"ROE {roe_v:.1f}%"}
    else:
        comps["roe"] = {"score": 0.0, "ok": None, "note": "缺 ROE 数据"}

    # 4. 毛利率稳定
    gm_hist = inds.get("gross_margin", pd.DataFrame())
    gm_tail = gm_hist["value"].dropna().tail(8)
    if len(gm_tail) >= 2:
        covered += 1
        gstd = float(gm_tail.std())
        comps["gross_margin"] = {
            "value": round(float(gm_tail.mean()), 2), "std": round(gstd, 2),
            "score": _ratio_score(10.0 - gstd, [(7.0, 2.5), (4.0, 1.5), (0.0, 0.8)]),
            "ok": gstd < 3, "note": f"近{len(gm_tail)}期毛利率均值{float(gm_tail.mean()):.1f}% 标准差{gstd:.1f}pp"}
    else:
        comps["gross_margin"] = {"score": 0.0, "ok": None,
                                 "note": f"毛利率样本不足({len(gm_tail)}<2)"}

    score = round(sum(c["score"] for c in comps.values()), 1)
    if score >= QUALITY_OK and covered >= 3:
        level = "优质"
    elif score <= QUALITY_BAD and covered >= COVERED_MIN:
        level = "劣质"
    elif covered < COVERED_MIN:
        level = "数据不足"
    else:
        level = "中性"
    return {"score": score, "level": level, "covered": covered,
            "components": comps}


# ════════════════════════════════════════════════════════════
# 3. 业绩预告 / 快报
# ════════════════════════════════════════════════════════════
_FORECAST_COLUMNS = ("预告类型", "业绩预告类型", "预警类型", "业绩变动方向",
                     "预告净利润变动", "业绩预告", "预告摘要")


def _find_forecast_parquets() -> list[Path]:
    """在 data_warehouse 下找含业绩预告列的 parquet（缺失 → 空）。"""
    base = ROOT / "data_warehouse"
    if not base.exists():
        return []
    out = []
    for p in base.rglob("*.parquet"):
        try:
            cols = _parquet_columns(p)
        except Exception as e:
            logging.getLogger(__name__).error(f"[valuation_system] 操作失败: {e}", exc_info=True)
            continue
        if any(str(c) in _FORECAST_COLUMNS or "预告" in str(c) or "预警" in str(c) for c in cols):
            out.append(p)
    return out


def _parquet_columns(p: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(p)).schema.names


def earnings_forecast(code: str, profit_yoy: float | None,
                      as_of: pd.Timestamp | None = None) -> dict:
    """业绩预告/快报：优先 parquet 库（公告日期 ≤ as_of 过滤），缺失 → 净利同比代理。

    parquet 无公告日期列时无法截断 → detail 显式标注 as_of（提示可能含分析日后披露）。
    """
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else None
    parquet_hit = None
    for p in _find_forecast_parquets():
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            logging.getLogger(__name__).error(f"[valuation_system] 操作失败: {e}", exc_info=True)
            continue
        if df is None or df.empty:
            continue
        code_col = next((c for c in df.columns if str(c).lower() in
                         ("code", "symbol", "股票代码", "证券代码")), None)
        if code_col is None:
            continue
        df = df[df[code_col].astype(str).str.zfill(6) == code.zfill(6)]
        if df.empty:
            continue
        ann_col = next((c for c in df.columns if any(k in str(c) for k in
                        ("公告日期", "公告日", "披露日期", "发布日期",
                         "announce", "ann_date", "notice_date"))), None)
        if ann_col is not None and cutoff is not None:
            df[ann_col] = pd.to_datetime(df[ann_col], errors="coerce")
            df = df[df[ann_col] <= cutoff]
            if df.empty:
                continue  # 分析日尚无披露，不引入未来预告
        type_col = next((c for c in df.columns if "预告类型" in str(c) or "预警类型" in str(c)), None)
        val_col = next((c for c in df.columns if "净利润变动" in str(c) or "变动方向" in str(c)), None)
        latest = df.iloc[-1]
        label = None
        if type_col is not None:
            t = str(latest.get(type_col, ""))
            if any(k in t for k in ("预增", "略增", "续盈", "扭亏", "减亏")):
                label = "业绩支撑"
            elif any(k in t for k in ("预减", "略减", "首亏", "续亏", "增亏", "预亏")):
                label = "业绩雷"
            else:
                label = "中性"
        elif val_col is not None:
            v = pd.to_numeric(latest.get(val_col), errors="coerce")
            if pd.notna(v):
                label = "业绩支撑" if v > 0 else ("业绩雷" if v < 0 else "中性")
        if label is not None:
            date_note = ""
            if cutoff is not None:
                if ann_col is not None:
                    date_note = f" 公告≤{cutoff.date()}"
                else:
                    date_note = f" as_of={cutoff.date()}(无公告日期列)"
            parquet_hit = {"file": str(p), "label": label,
                           "type": (str(latest.get(type_col, ""))[:20] if type_col else ""),
                           "date_note": date_note}
            break

    if parquet_hit:
        return {"label": parquet_hit["label"], "source": "parquet",
                "detail": (f"业绩预告库 {parquet_hit['file']} {parquet_hit['type']}"
                           f"{parquet_hit['date_note']}").strip()}

    # 代理: 净利同比
    if profit_yoy is None:
        return {"label": "业绩方向未知", "source": "proxy",
                "detail": "无业绩预告库且缺净利同比，方向未知"}
    suspect = abs(profit_yoy) > 80
    label = "业绩支撑" if profit_yoy > 0 else "业绩雷"
    detail = f"无业绩预告库，净利同比代理 {profit_yoy:+.1f}%"
    if suspect:
        detail += "（数值异常|同比口径存疑，仅供参考）"
    return {"label": label, "source": "proxy", "detail": detail}


# ════════════════════════════════════════════════════════════
# 4. DCF 简版（Gordon）: FCF 5年均值 ÷ (WACC - g)
# ════════════════════════════════════════════════════════════
def dcf_simple(inds: dict[str, pd.DataFrame],
               price: float | None) -> dict:
    out = {"fcf_avg": None, "intrinsic": None, "wacc": WACC, "g": GROWTH_G,
           "discount_pct": None, "note": "", "sensitivity": None, "ok": False}
    def _annual(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df[df["date"].dt.strftime("%m%d") == "1231"]["value"].dropna().tail(HIST_YEARS)

    fcff = inds.get("fcff_per_share", pd.DataFrame())
    annual = _annual(fcff)
    src = "每股自由现金流(FCFE/FCFF,年报口径)"
    if len(annual) < DCF_MIN_YEARS:
        # 降级: 每股经营现金流近似 FCF
        annual = _annual(inds.get("cash_flow_per_share", pd.DataFrame()))
        src = "每股经营现金流近似FCF(缺自由现金流)"
    if len(annual) < DCF_MIN_YEARS:
        out["note"] = f"FCF 年度样本 {len(annual)}<{DCF_MIN_YEARS}，DCF 不适用"
        return out

    fcf_avg = float(annual.mean())
    intrinsic = fcf_avg / (WACC - GROWTH_G)
    out.update({"ok": True, "fcf_avg": round(fcf_avg, 2),
                "intrinsic": round(intrinsic, 2),
                "note": f"{src}: 近{len(annual)}年均值 {fcf_avg:.2f} ÷ "
                        f"(WACC {WACC:.0%} - g {GROWTH_G:.0%})",
                "sensitivity": _dcf_sensitivity(annual)})
    if price is not None and price > 0:
        out["discount_pct"] = round(price / intrinsic - 1.0, 3)
    return out


def _dcf_sensitivity(annual_fcf: pd.Series) -> dict:
    """WACC × g 敏感性（dcf-value-driver-sensitivity）：内在价值矩阵。"""
    fcf_avg = float(annual_fcf.mean())
    grid = {}
    for w in (0.08, 0.10, 0.12):
        row = {}
        for g in (0.02, 0.03, 0.04):
            if w > g:
                row[f"{g:.0%}"] = round(fcf_avg / (w - g), 0)
        grid[f"wacc_{w:.0%}"] = row
    return grid


# ════════════════════════════════════════════════════════════
# 5. 综合
# ════════════════════════════════════════════════════════════
def compose(val: dict, quality: dict, earnings: dict, dcf: dict) -> dict:
    bias, qlevel, elabel = val.get("bias"), quality["level"], earnings["label"]
    reasons: list[str] = []
    if bias:
        reasons.append(f"估值分位={bias}")
    else:
        reasons.append(f"估值分位=未知({val.get('bias_detail', '样本不足')})")
    reasons.append(f"盈利质量={quality['score']}分/{qlevel}")
    reasons.append(f"业绩方向={elabel}({earnings['source']})")
    if dcf.get("ok") and dcf.get("discount_pct") is not None:
        reasons.append(f"DCF{dcf['discount_pct'] * 100:+.0f}%")

    signal, conf = "中性", 0.4
    if bias == "低估" and qlevel == "优质" and elabel in ("业绩支撑", "业绩方向未知"):
        signal = "看多"
        conf = 0.45 + 0.05 * quality["score"]
        if elabel == "业绩支撑":
            conf += 0.10
        if val.get("pe_pct") is not None and val["pe_pct"] < 0.15:
            conf += 0.10
        if elabel == "业绩雷":
            signal, conf = "中性", 0.4
            reasons.append("业绩雷对冲低估优质 → 中性")
    elif bias == "高估" and qlevel == "劣质" and elabel in ("业绩雷", "业绩方向未知"):
        signal = "看空"
        conf = 0.45 + 0.05 * (10 - quality["score"])
        if elabel == "业绩雷":
            conf += 0.10
        if val.get("pe_pct") is not None and val["pe_pct"] > 0.85:
            conf += 0.10
        if elabel == "业绩支撑":
            signal, conf = "中性", 0.4
            reasons.append("业绩支撑对冲高估劣质 → 中性")
    elif qlevel == "数据不足":
        conf = 0.30
        reasons.append("财务覆盖不足，置信度下调")

    # DCF 大幅溢价提示（不否决综合方向，仅下调置信）
    conf = min(conf, 0.95)
    if dcf.get("ok") and dcf.get("discount_pct") is not None and dcf["discount_pct"] > 0.30:
        conf *= 0.85
        reasons.append(f"DCF简版溢价{dcf['discount_pct'] * 100:.0f}%，安全边际不足")
    conf = round(conf, 2)
    return {"signal": signal, "confidence": conf, "reason": "；".join(reasons)}


# ════════════════════════════════════════════════════════════
# ValuationSystem 统一接口
# ════════════════════════════════════════════════════════════
class ValuationSystem:
    """基本面估值系统统一接口: detect(date)/report(date)/view(date)。"""

    def __init__(self, watch: list[str] | None = None,
                 out_dir: Path | None = None) -> None:
        self.watch = [str(c).zfill(6) for c in (watch or WATCHLIST_TEMPLATE["watch"])]
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self.names, _ = load_names()
        self._detect_cache: dict | None = None
        self._cache_key: str | None = None
        self._rag: list[dict] | None = None

    # ---- RAG 方法论依据（懒加载 + 缓存） ----
    def _rag_basis(self) -> list[dict]:
        if self._rag is None:
            try:
                from quant_system.analysis_core.knowledge_rag import search
                self._rag = search(RAG_QUERY, k=RAG_K)
            except Exception as e:
                self._rag = [{"file": "(rag不可用)", "cat": "—", "score": 0.0,
                              "text": f"RAG检索失败: {str(e)[:80]}"}]
        return self._rag

    # ---- 单标的分析 ----
    def _analyze_one(self, code: str, date: str | None = None) -> dict:
        row = {"code": code, "name": self.names.get(code, ""), "ok": False,
               "error": "", "signal": "中性", "confidence": 0.0, "evidence": []}
        try:
            cutoff = pd.Timestamp(date).normalize() if date else None
            inds = _load_indicators(code, [
                "net_profit_parent", "revenue", "eps", "bvps", "roe", "roe_full",
                "gross_margin", "ocf_to_net_profit", "ocf_total", "receivables_turnover",
                "fcff_per_share", "cash_flow_per_share", "profit_growth_yoy",
            ], as_of=cutoff)
            has_fin = any(not df.empty for df in inds.values())
            if not has_fin:
                row.update({"error": "无财务数据（本地库无该标的）",
                            "evidence": ["财务数据缺失"]})
                return row

            as_of, _ = _latest_report(inds.get("net_profit_parent", pd.DataFrame()))
            if as_of is None:
                for df in inds.values():
                    if not df.empty:
                        as_of = pd.Timestamp(df["date"].max())
                        break
            row["as_of"] = str(as_of.date()) if as_of is not None else "未知"

            price_df = _price_history(code, as_of=cutoff)
            price = float(price_df["close"].iloc[-1]) if not price_df.empty else None
            row["price"] = round(price, 2) if price is not None else None
            row["price_date"] = str(price_df["date"].iloc[-1].date()) if not price_df.empty else None
            if price is None:
                row["evidence"].append("行情不可用（本地K线缺失），DCF折溢价跳过")

            val = valuation_percentile(code, inds, price_df)
            quality = quality_score(inds)
            _, profit_yoy = _latest_report(inds.get("profit_growth_yoy", pd.DataFrame()))
            earnings = earnings_forecast(code, profit_yoy, as_of=cutoff)
            dcf = dcf_simple(inds, price)
            comp = compose(val, quality, earnings, dcf)

            ev = [
                f"估值分位: {val['bias'] if val['bias'] else '未知'}（{val['bias_detail']}）",
                f"盈利质量: {quality['score']}/10 {quality['level']}（覆盖{quality['covered']}/4项）",
                f"业绩方向: {earnings['label']}（{earnings['detail']}）",
            ]
            if dcf.get("ok"):
                dp = dcf["discount_pct"]
                ev.append(f"DCF简版: 内在价值≈{dcf['intrinsic']}，现价{'折价' if dp and dp < 0 else '溢价'}"
                          f" {abs(dp) * 100:.0f}%（{dcf['note']}）")
            else:
                ev.append(f"DCF简版: 不适用（{dcf['note']}）")
            if as_of is not None:
                ev.append(f"财务口径 as_of={row['as_of']}（季报可能陈旧）")
            ev.append(f"综合: {comp['signal']}（{comp['reason']}）")

            row.update({
                "ok": True, "signal": comp["signal"], "confidence": comp["confidence"],
                "valuation": val, "quality": quality, "earnings": earnings,
                "dcf": dcf, "composite": comp, "evidence": ev,
                "profit_yoy": round(profit_yoy, 2) if profit_yoy is not None else None,
            })
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        return row

    def _run_detect(self, date: str | None) -> dict:
        t0 = time.time()
        rows = [self._analyze_one(c, date) for c in self.watch]
        elapsed = time.time() - t0
        ok_rows = [r for r in rows if r["ok"]]
        signals = {"看多": 0, "看空": 0, "中性": 0}
        for r in ok_rows:
            signals[r["signal"]] = signals.get(r["signal"], 0) + 1
        eff_date = date or datetime.now(CST).date().isoformat()
        return {"date": eff_date, "rows": rows,
                "summary": {"date": eff_date, "total": len(rows),
                            "ok": len(ok_rows), "failed": len(rows) - len(ok_rows),
                            "signals": signals, "elapsed": round(elapsed, 1)},
                "rag": self._rag_basis()}

    def _detect_cached(self, date: str | None) -> dict:
        key = date or "auto"
        if self._cache_key != key:
            self._detect_cache = self._run_detect(date)
            self._cache_key = key
        return self._detect_cache

    def detect(self, date: str | None = None) -> dict:
        """全量检测：每只标的 估值分位/盈利质量/业绩预告/DCF/综合。"""
        return self._detect_cached(date)

    def view(self, date: str | None = None) -> dict:
        """多空观点（供 multi_agent 使用）: {agent, signal, view, confidence, evidence}。"""
        det = self._detect_cached(date)
        rows = [r for r in det["rows"] if r["ok"]]
        if not rows:
            return {"agent": "基本面估值", "signal": "中性", "view": "震荡",
                    "confidence": 0.0, "evidence": ["无有效标的（全部跳过）"],
                    "weight": 1.0, "status": "degraded", "detail": det}
        scores = {"看多": 1.0, "看空": -1.0, "中性": 0.0}
        wsum = sum(scores[r["signal"]] * r["confidence"] for r in rows) / len(rows)
        if wsum > 0.15:
            signal = "看多"
        elif wsum < -0.15:
            signal = "看空"
        else:
            signal = "中性"
        conf = round(min(0.95, 0.35 + 0.45 * abs(wsum) +
                         0.05 * min(1.0, len(rows) / 5.0)), 2)
        top = sorted(rows, key=lambda r: -abs({"看多": 1, "看空": -1, "中性": 0}[r["signal"]]
                                              * r["confidence"]))[:5]
        evidence = [f"有效标的 {len(rows)}/{len(det['rows'])} 只，加权分 {wsum:+.2f}"]
        for r in top:
            evidence.append(f"{r['name'] or r['code']}({r['code']}) {r['signal']} "
                            f"置信{r['confidence']:.2f}：{r['composite']['reason']}")
        return {"agent": "基本面估值", "signal": signal,
                "view": {"看多": "多", "看空": "空", "中性": "震荡"}.get(signal, "震荡"),
                "confidence": conf, "evidence": evidence,
                "weight": 1.0, "status": "ok", "detail": det}

    # ---- Markdown 报告 ----
    def render_markdown(self, det: dict) -> str:
        s = det["summary"]
        lines = [f"# 基本面估值系统 — {s['date']}", "",
                 f"- 标的: {s['total']} 只 | 可用: {s['ok']} 只 | 跳过: {s['failed']} 只 | "
                 f"耗时 {s['elapsed']}s | 信号: {s['signals']}",
                 "- 方法论: DCF 自由现金流折现（简版 Gordon，WACC 10% / g 3%）+ "
                 "盈余质量（Penman: 现金流/利润比、应收/营收、ROE、毛利率稳定性）+ "
                 "价值因子（PE/PB/PS 近5年分位）",
                 "- 数据源: 本地 financial.db（80指标×多期）+ data_store 日K（近5年）；"
                 "业绩预告读 data_warehouse parquet（缺失时净利同比代理）",
                 "- 财务口径: 报告期（季报可能陈旧，每只标注 as_of）", "",
                 "## 方法论依据（RAG）", ""]
        for i, r in enumerate(det.get("rag") or [], 1):
            lines.append(f"{i}. `{r.get('file', '')}` ({r.get('cat', '—')}, "
                         f"score {r.get('score', 0)}) — "
                         f"{str(r.get('text', ''))[:120].replace(chr(10), ' ')}")
        if not det.get("rag"):
            lines.append("_无检索结果_")

        lines += ["", "## 汇总", "",
                  "| 代码 | 名称 | 估值分位 | 盈利质量 | 业绩方向 | DCF | 信号 | 置信 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in det["rows"]:
            if not r["ok"]:
                lines.append(f"| {r['code']} | {r['name']} | — | — | — | — | 跳过 | "
                             f"{r['error'][:30]} |")
                continue
            val_txt = r["valuation"]["bias"] or "未知"
            dcf_txt = "不适用"
            if r["dcf"].get("ok"):
                dp = r["dcf"]["discount_pct"]
                dcf_txt = f"{r['dcf']['intrinsic']:.0f}({'折价' if dp and dp < 0 else '溢价'}{abs(dp or 0) * 100:.0f}%)"
            lines.append(f"| {r['code']} | {r['name']} | {val_txt} | "
                         f"{r['quality']['score']}/10 {r['quality']['level']} | "
                         f"{r['earnings']['label']} | {dcf_txt} | "
                         f"{r['signal']} | {r['confidence']:.2f} |")

        lines += ["", "## 明细", ""]
        for r in det["rows"]:
            title = f"{r['name']}({r['code']}) — {r['signal']}" if r["ok"] else \
                f"{r['name']}({r['code']}) — 跳过"
            lines.append(f"### {title}")
            if not r["ok"]:
                lines.append(f"- 错误: {r['error']}")
                lines.append("")
                continue
            lines.append(f"- 现价: {r['price']}（{r['price_date']}） | 财务口径 as_of: {r['as_of']}")
            v = r["valuation"]
            v3 = (v.get('pe'), v.get('pb'), v.get('ps'))
            v3_txt = f"（PE {v3[0]} / PB {v3[1]} / PS {v3[2]}）" if any(v3) else ""
            lines.append(f"- **估值分位**: {v['bias'] or '未知'} — {v['bias_detail']}{v3_txt}")
            q = r["quality"]
            lines.append(f"- **盈利质量**: {q['score']}/10 **{q['level']}**（覆盖 {q['covered']}/4）")
            for name, c in q["components"].items():
                vv = c.get("value")
                vv_txt = f" = {vv}" if vv is not None else ""
                lines.append(f"  - {name}: {c['score']}/2.5{vv_txt} — {c['note']}")
            e = r["earnings"]
            lines.append(f"- **业绩预告**: **{e['label']}**（{e['detail']}）")
            d = r["dcf"]
            if d.get("ok"):
                dp = d["discount_pct"]
                rel = f"现价{'折价' if dp and dp < 0 else '溢价'}{abs(dp or 0) * 100:.0f}%" \
                    if dp is not None else "现价不可用"
                lines.append(f"- **DCF简版**: 内在价值≈{d['intrinsic']} | {rel} | {d['note']}")
                if d.get("sensitivity"):
                    sens = "；".join(f"{k.split('_')[1]}×g{list(v.keys())[0]}={list(v.values())[0]}"
                                     for k, v in list(d["sensitivity"].items())[:2])
                    lines.append(f"  - 敏感性（节选）: {sens}")
            else:
                lines.append(f"- **DCF简版**: 不适用（{d['note']}）")
            lines.append(f"- **综合**: **{r['signal']}**（置信 {r['confidence']:.0%}）— {r['composite']['reason']}")
            lines.append("")
        return "\n".join(lines)

    def report(self, date: str | None = None) -> Path:
        """生成并保存 generated/valuation_report_{date}.md，返回文件路径。"""
        det = self._detect_cached(date)
        md = self.render_markdown(det)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / f"valuation_report_{det['date']}.md"
        out.write_text(md, encoding="utf-8")
        return out


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def _load_watchlist() -> list[str]:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not WATCHLIST_FILE.exists():
            WATCHLIST_FILE.write_text(
                json.dumps(WATCHLIST_TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            return [str(c) for c in WATCHLIST_TEMPLATE["watch"]]
        data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("watch") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [str(c).strip().zfill(6) for c in raw if str(c).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="体系8 基本面估值系统（DCF + 盈余质量 + 价值因子）")
    ap.add_argument("--watch", default=None, help="股票代码，逗号分隔（缺省读 config/watchlist.json）")
    ap.add_argument("--date", default=None, help="报告日期 YYYY-MM-DD（默认今日 CST）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 详情")
    ap.add_argument("--no-report", action="store_true", help="不写 generated/valuation_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录")
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
        codes = _load_watchlist()
        if not codes:
            print("[错误] config/watchlist.json 缺失/格式非法，且未提供 --watch")
            sys.exit(1)
        print(f"[提示] --watch 缺省，读取 {WATCHLIST_FILE}")

    system = ValuationSystem(watch=codes, out_dir=Path(args.out_dir))
    det = system.detect(args.date)
    s = det["summary"]
    print(f"[基本面估值] {s['date']} | {s['total']} 只（可用 {s['ok']}，跳过 {s['failed']}）"
          f" | 信号 {s['signals']} | 耗时 {s['elapsed']}s")
    print()
    print(f"  {'代码':<6} {'名称':<8} {'估值分位':<8} {'质量':<12} {'业绩':<8} {'信号':<4} 置信")
    for r in det["rows"]:
        if not r["ok"]:
            print(f"  {r['code']:<6} {r['name'][:8]:<8} {'跳过':<8} {r['error'][:40]}")
            continue
        val_txt = r["valuation"]["bias"] or "未知"
        q_txt = f"{r['quality']['score']}/{r['quality']['level']}"
        print(f"  {r['code']:<6} {r['name'][:8]:<8} {val_txt:<8} {q_txt:<12} "
              f"{r['earnings']['label']:<8} {r['signal']:<4} {r['confidence']:.2f}")

    if args.json:
        print(json.dumps(det, ensure_ascii=False, indent=2, default=str))

    if not args.no_report:
        out = system.report(args.date)
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
