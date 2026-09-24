"""
fusion — 信号融合层（V11）

把 情绪周期 × 连板天梯 × 四路资金 × 题材 × 社交 融合成:
  1. 市场综合温度 (0-100) + 状态标签
  2. 交叉信号（多模块共振/背离 → 高价值预警）

融合原则: 单一模块可能出错，多模块共振才可信；背离必须显式标注。

用法:
  python3 -m quant_system.analysis_core.fusion --today
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402
from quant_system.analysis_core.data_contract import format_pct_value  # noqa: E402
from quant_system.analysis_core import alternative_data  # noqa: E402
from quant_system.analysis_core.emotion_cycle import run_today  # noqa: E402

CST = timezone(timedelta(hours=8))
OUT = MARKET_DIR / "fusion.parquet"
SW_FIRST_HIST = ROOT / "data_warehouse" / "industry" / "sw_first_hist.parquet"
SW_FIRST = ROOT / "data_warehouse" / "industry" / "sw_first.parquet"
CSI_INDUSTRY_HIST = ROOT / "data_warehouse" / "industry" / "csi_industry_hist.parquet"
CSI_INDUSTRY = ROOT / "data_warehouse" / "industry" / "csi_industry.parquet"
CPI_YEARLY = ROOT / "data_warehouse" / "macro" / "cpi_yearly.parquet"
M2_YEARLY = ROOT / "data_warehouse" / "macro" / "m2_yearly.parquet"

# 情绪阶段 → 温度基准
STAGE_TEMP = {"ice": 15, "repair": 40, "ferment": 65, "climax": 85, "divergence": 45, "ebb": 12}


def _load_latest(path: str, cols: list[str] | None = None) -> pd.DataFrame | None:
    p = MARKET_DIR / path
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=cols) if cols else pd.read_parquet(p)
    return df


def read_fusion_latest(ref: str | pd.Timestamp, cols: list[str] | None = None) -> dict | None:
    """fusion.parquet ≤ref 最新一期。返回 {"date": "YYYY-MM-DD", <请求列>...} 或 None；异常（含损坏文件）返回 None。"""
    try:
        if cols is None:
            df = _load_latest("fusion.parquet")
        else:
            read_cols = ["date"] + [c for c in cols if c != "date"]
            df = _load_latest("fusion.parquet", read_cols)
    except Exception as e:  # noqa: BLE001
        print(f"[fusion] 读取 fusion.parquet 失败，降级返回 None: {e}")
        return None
    if df is None or len(df) == 0:
        return None
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        ref_ts = pd.Timestamp(ref)
        hist = df[df["date"] <= ref_ts].sort_values("date")
        if hist.empty:
            return None
        row = hist.iloc[-1]
        out: dict = {"date": row["date"].strftime("%Y-%m-%d")}
        for c in row.index:
            if c != "date":
                out[c] = row[c]
        return out
    except Exception:  # noqa: BLE001
        return None


def _data_lag(df: pd.DataFrame | None, ref_date: str) -> int | None:
    """数据相对参考日落后几个交易日（None=无数据）。"""
    if df is None or len(df) == 0:
        return None
    last = pd.Timestamp(df["date"].max())
    ref = pd.Timestamp(ref_date)
    # 交易日间隔用日数近似（≤3 视为新鲜）
    return max(0, (ref - last).days)


def _as_of_lag(as_of: str | None, ref_date: str) -> int | None:
    """as_of（YYYYMMDD / YYYY-MM-DD / YYYY-MM 月度）落后 ref_date 的天数；None=不可比。"""
    if not as_of:
        return None
    try:
        s = str(as_of).strip().replace("-", "")
        if len(s) == 6:  # 月度宏观 YYYYMM → 月初 YYYYMM01
            s = s + "01"
        return max(0, (pd.Timestamp(ref_date[:10]) - pd.Timestamp(s)).days)
    except Exception:  # noqa: BLE001
        return None


def _industry_breadth(ref_date: str) -> dict | None:
    """申万一级行业广度：最新交易日收盘 vs 上一交易日收盘自算涨跌幅。

    行业名称映射用 sw_first.parquet 的 行业代码(801010.SI)→行业名称。
    """
    if not SW_FIRST_HIST.exists():
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": f"缺失 {SW_FIRST_HIST}"}
    df = pd.read_parquet(SW_FIRST_HIST, columns=["代码", "日期", "收盘"])
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"])
    ref = pd.Timestamp(ref_date[:10])
    hist = df[df["日期"] <= ref]
    if not len(hist):
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": f"sw_first_hist 无 ≤{ref_date} 数据"}
    dates = sorted(hist["日期"].unique())
    if len(dates) < 2:
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": "sw_first_hist 交易日不足 2 日"}
    latest, prev = dates[-1], dates[-2]
    cur = hist[hist["日期"] == latest][["代码", "收盘"]].copy()
    pre = hist[hist["日期"] == prev][["代码", "收盘"]].copy()
    merged = cur.merge(pre, on="代码", suffixes=("_cur", "_prev"))
    merged = merged[merged["收盘_prev"] > 0]
    if not len(merged):
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": "sw_first_hist 无有效收盘价对"}
    merged["pct"] = (merged["收盘_cur"] / merged["收盘_prev"] - 1) * 100
    name_map: dict[str, str] = {}
    if SW_FIRST.exists():
        try:
            nm = pd.read_parquet(SW_FIRST)
            name_map = dict(zip(nm["行业代码"].astype(str).str.replace(".SI", "", regex=False),
                                nm["行业名称"].astype(str)))
        except Exception:  # noqa: BLE001
            name_map = {}
    merged["name"] = merged["代码"].astype(str).map(name_map).fillna(merged["代码"].astype(str))
    top_up = merged.sort_values("pct", ascending=False).head(3)
    top_down = merged.sort_values("pct", ascending=True).head(3)
    return {
        "status": "available",
        "as_of": latest.strftime("%Y-%m-%d"),
        "n": int(len(merged)),
        "breadth": round(float((merged["pct"] > 0).mean()), 4),
        "top_up": [{"name": r["name"], "pct": round(float(r["pct"]), 2),
                    "pct_text": format_pct_value(r["pct"], unit="pct")}
                   for _, r in top_up.iterrows()],
        "top_down": [{"name": r["name"], "pct": round(float(r["pct"]), 2),
                      "pct_text": format_pct_value(r["pct"], unit="pct")}
                     for _, r in top_down.iterrows()],
        "reason": "申万一级行业最新交易日涨跌幅（收盘价自算）",
    }

def _csi_breadth(ref_date: str) -> dict:
    """中证行业指数广度：最新交易日（≤ref）行业指数上涨占比 + 领涨/领跌 top3。

    涨跌幅优先用现成的'涨跌幅'列，NaN 行回退到收盘价自算（最新 vs 上一交易日）。
    行业范围以 csi_industry.parquet 的行业指数清单为准（缺失时退回全表）。
    """
    if not CSI_INDUSTRY_HIST.exists():
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": f"缺失 {CSI_INDUSTRY_HIST}"}
    df = pd.read_parquet(CSI_INDUSTRY_HIST,
                         columns=["日期", "指数代码", "指数中文简称", "收盘", "涨跌幅"])
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"])
    ref = pd.Timestamp(ref_date[:10])
    hist = df[df["日期"] <= ref]
    if not len(hist):
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": f"csi_industry_hist 无 ≤{ref_date} 数据"}
    dates = sorted(hist["日期"].unique())
    latest = dates[-1]
    cur = hist[hist["日期"] == latest].copy()
    # 行业范围: csi_industry.parquet 的行业指数代码清单（78 个）
    name_map: dict[str, str] = {}
    ind_codes: set[str] = set()
    if CSI_INDUSTRY.exists():
        try:
            nm = pd.read_parquet(CSI_INDUSTRY, columns=["指数代码", "指数简称"])
            name_map = dict(zip(nm["指数代码"].astype(str), nm["指数简称"].astype(str)))
            ind_codes = set(name_map)
        except Exception:  # noqa: BLE001
            name_map = {}
    if ind_codes:
        cur = cur[cur["指数代码"].astype(str).isin(ind_codes)]
    if not len(cur):
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": "csi_industry_hist 最新交易日无行业指数"}
    # 涨跌幅: 现成列优先；NaN 用收盘自算（对上一交易日）
    # 2026-08-21 复核(critical 修复): 中证行业 涨跌幅 列本就是百分数(全史 median_abs≈0.87,
    # 范围 -14~17，|值|>5 仅 1.4%)；曾用"当日 median_abs<1 则×100"启发式导致 63% 交易日
    # 被误放 100 倍(-1.26%→-126%)，top_up/top_down 的 pct/pct_text 失真。已删除启发式，
    # 统一百分数口径(与 pct_calc 自算一致)。若未来源切换为小数口径，需按数据契约显式转换。
    if len(dates) >= 2:
        prev = hist[hist["日期"] == dates[-2]][["指数代码", "收盘"]].rename(
            columns={"收盘": "收盘_prev"})
        cur = cur.merge(prev, on="指数代码", how="left")
        cur["pct_calc"] = (cur["收盘"] / cur["收盘_prev"] - 1) * 100
        cur["pct"] = cur["涨跌幅"].where(cur["涨跌幅"].notna(), cur["pct_calc"])
    else:
        cur["pct"] = cur["涨跌幅"]
    cur = cur.dropna(subset=["pct"])
    if not len(cur):
        return {"status": "unavailable", "as_of": None, "breadth": None,
                "top_up": [], "top_down": [], "reason": "csi_industry_hist 最新交易日无有效涨跌幅"}
    cur["name"] = cur["指数代码"].astype(str).map(name_map).fillna(cur["指数中文简称"])
    top_up = cur.sort_values("pct", ascending=False).head(3)
    top_down = cur.sort_values("pct", ascending=True).head(3)
    return {
        "status": "available",
        "as_of": latest.strftime("%Y-%m-%d"),
        "n": int(len(cur)),
        "breadth": round(float((cur["pct"] > 0).mean()), 4),
        "top_up": [{"name": r["name"], "pct": round(float(r["pct"]), 2),
                    "pct_text": format_pct_value(r["pct"], unit="pct")}
                   for _, r in top_up.iterrows()],
        "top_down": [{"name": r["name"], "pct": round(float(r["pct"]), 2),
                      "pct_text": format_pct_value(r["pct"], unit="pct")}
                     for _, r in top_down.iterrows()],
        "reason": "中证行业指数最新交易日涨跌幅（现成涨跌幅，NaN 用收盘自算）",
    }


MACRO_STALE_DAYS = 45  # 月度宏观（CPI/M2）允许的最大落后天数，超过视为陈旧


def _macro_latest_yoy(path: Path, col_prefs: list[str],
                      ref_date: str | None = None,
                      value_range: tuple[float, float] | None = None,
                      ) -> tuple[float | None, str | None]:
    """读月度宏观 parquet，取 ≤ref_date 的最新一期同比（%）。月份形如 '2026年07月份'。

    value_range: 若提供 (lo, hi)，仅接受 lo<=v<=hi 的值（防回退到'今值/数值'时拿非同比当同比）。
    """
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    col = next((c for c in col_prefs if c in df.columns), None)
    if col is None or "月份" not in df.columns or not len(df):
        return None, None
    ref = pd.Timestamp(ref_date[:10]) if ref_date else None
    rows: list[tuple[int, int, float]] = []
    for _, r in df.iterrows():
        mm = re.match(r"(\d{4})年(\d{1,2})月", str(r["月份"]))
        v = pd.to_numeric(r.get(col), errors="coerce")
        if not (mm and pd.notna(v)):
            continue
        if value_range is not None and not (value_range[0] <= v <= value_range[1]):
            continue  # 越界视为非同比/异常，丢弃
        y, mo = int(mm.group(1)), int(mm.group(2))
        # 月份转日期（月初）比较：只取 ≤ref 的最新一期
        if ref is not None and pd.Timestamp(y, mo, 1) > ref:
            continue
        rows.append((y, mo, float(v)))
    if not rows:
        return None, None
    rows.sort()
    y, mo, v = rows[-1]
    return v, f"{y}-{mo:02d}"


def _macro_env(ref_date: str | None = None) -> dict | None:
    """宏观月度环境：≤ref_date 最新一期 CPI 同比 + M2 同比 → 环境标签。

    返回结构 status/label/cpi/m2/as_of；缺失 → None 不参与；
    任一路径落后 ref 超过 MACRO_STALE_DAYS 天 → status='stale'（仅展示，不参与温度修正）。
    """
    try:
        cpi, cpi_as_of = _macro_latest_yoy(
            CPI_YEARLY, ["全国-同比增长", "今值", "数值", "同比增长"], ref_date,
            value_range=(-5.0, 10.0))  # CPI 同比合理域
        m2, m2_as_of = _macro_latest_yoy(
            M2_YEARLY, ["货币和准货币(M2)-同比增长", "今值", "数值", "同比增长"], ref_date,
            value_range=(0.0, 30.0))  # M2 同比合理域
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "label": None, "cpi": None, "m2": None,
                "as_of": None, "reason": f"{type(e).__name__}: {e}"}
    if cpi is None and m2 is None:
        return None
    # 标签口径与温度修正一致：通缩/滞胀 显式标注，不再落到"温和"
    if cpi is not None and cpi < 0:
        label = "通缩"
    elif cpi is not None and cpi > 3 and m2 is not None and m2 > 10:
        label = "滞胀"
    else:
        labels: list[str] = []
        if cpi is not None and cpi > 3:
            labels.append("通胀偏高")
        if m2 is not None and m2 > 10:
            labels.append("流动性宽松")
        label = "+".join(labels) if labels else "温和"
    as_of = cpi_as_of or m2_as_of
    status = "available"
    if ref_date and as_of:
        lags = []
        for a in (cpi_as_of, m2_as_of):
            if not a:
                continue
            try:
                lags.append(max(0, (pd.Timestamp(ref_date[:10]) - pd.Timestamp(a)).days))
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).error(f"[fusion] 操作失败: {e}", exc_info=True)
        if lags and max(lags) > MACRO_STALE_DAYS:
            status = "stale"
    return {"status": status, "label": label, "cpi": cpi, "m2": m2, "as_of": as_of}


def fuse_today() -> dict:
    emo = run_today()
    ref_date = emo["date"]
    stats = _load_latest("zt_daily_stats.parquet")
    forces = _load_latest("fund_forces.parquet")
    themes = _load_latest("theme_cycle.parquet")
    social = _load_latest("social_sentiment.parquet")

    # ── 数据新鲜度检查（2026-08-10 审计补全）─────────────────────────
    # 任何关键输入落后 >3 日：温度标记降级且该输入不参与加权，禁止静默用旧数据。
    # 参考日 = 情绪周期数据日；情绪基准本身落后（相对今天）也整体降级。
    stale: list[str] = []
    today_str = datetime.now(CST).strftime("%Y-%m-%d")
    emo_lag = (pd.Timestamp(today_str) - pd.Timestamp(ref_date)).days
    emo_stale = emo_lag > 3
    if emo_stale:
        stale.append(f"情绪周期基准落后{emo_lag}日(实际{ref_date})")

    forces_lag = _data_lag(forces, ref_date)
    stats_lag = _data_lag(stats, ref_date)
    themes_lag = _data_lag(themes, ref_date)
    social_lag = _data_lag(social, ref_date)
    if forces_lag is not None and forces_lag > 3:
        stale.append(f"资金合力落后{forces_lag}日(实际{forces['date'].max().date()})")
        forces = None  # 旧资金数据不参与融合，避免静默污染温度
    if stats_lag is not None and stats_lag > 3:
        stale.append(f"涨停统计落后{stats_lag}日(实际{stats['date'].max().date()})")
        stats = None
    if themes_lag is not None and themes_lag > 3:
        stale.append(f"题材周期落后{themes_lag}日(实际{themes['date'].max().date()})")
        themes = None
    if social_lag is not None and social_lag > 3:
        stale.append(f"社交情绪落后{social_lag}日(实际{social['date'].max()})")
        social = None  # 社交仅作参考，陈旧时整体剔除，避免报虚假"数据源可用"

    # ── 另类数据（云源 + 本地代理）────────────────────────────
    alt: dict = {"status": "unavailable", "reason": "alternative_data 不可用"}
    lp: dict = {"status": "unavailable"}
    try:
        alt = alternative_data.cloud_sources(ref_date)
    except Exception as e:  # noqa: BLE001
        alt = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
    try:
        lp = alternative_data.local_proxies(ref_date)
    except Exception as e:  # noqa: BLE001
        lp = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

    cninfo_ok = alt.get("status") == "available"
    cn_lag = _as_of_lag(alt.get("date"), ref_date)
    if cninfo_ok and cn_lag is not None and cn_lag > 3:
        stale.append(f"巨潮公告云快照落后{cn_lag}日(as_of={alt.get('date')})")
        cninfo_ok = False
    hr = alt.get("hot_rank") or {}
    hr_ok = hr.get("status") == "available"
    hr_lag = _as_of_lag(hr.get("as_of"), ref_date)
    if hr_ok and hr_lag is not None and hr_lag > 3:
        stale.append(f"东财热榜云快照落后{hr_lag}日(as_of={hr.get('as_of')})")
        hr_ok = False
    sc = alt.get("social") or {}
    sc_ok = sc.get("status") == "available"
    sc_lag = _as_of_lag(sc.get("as_of"), ref_date)
    if sc_ok and sc_lag is not None and sc_lag > 3:
        stale.append(f"社交云快照落后{sc_lag}日(as_of={sc.get('as_of')})")
        sc_ok = False
    lp_ok = lp.get("status") == "available"
    lp_lag = _as_of_lag(lp.get("date"), ref_date)
    if lp_ok and lp_lag is not None and lp_lag > 3:
        stale.append(f"本地代理基准落后{lp_lag}日(as_of={lp.get('date')})")
        lp_ok = False

    # ── 行业广度（申万一级，最新交易日收盘自算）────────────────
    industry: dict | None = None
    try:
        industry = _industry_breadth(ref_date)
    except Exception as e:  # noqa: BLE001
        industry = {"status": "unavailable", "as_of": None, "breadth": None,
                    "top_up": [], "top_down": [], "reason": f"{type(e).__name__}: {e}"}
    if industry is not None and industry.get("status") == "available":
        ind_lag = _as_of_lag(industry.get("as_of"), ref_date)
        if ind_lag is not None and ind_lag > 3:
            stale.append(f"申万行业最新交易日落后{ind_lag}日({industry.get('as_of')})")
            industry = None
    # 中证行业广度（78 行业指数，与申万互为校验）
    csi_industry: dict | None = None
    try:
        csi_industry = _csi_breadth(ref_date)
    except Exception as e:  # noqa: BLE001
        csi_industry = {"status": "unavailable", "as_of": None, "breadth": None,
                        "top_up": [], "top_down": [], "reason": f"{type(e).__name__}: {e}"}
    if csi_industry is not None and csi_industry.get("status") == "available":
        csi_lag = _as_of_lag(csi_industry.get("as_of"), ref_date)
        if csi_lag is not None and csi_lag > 3:
            stale.append(f"中证行业最新交易日落后{csi_lag}日({csi_industry.get('as_of')})")
            csi_industry = None

    # ── 基础温度: 情绪阶段基准 ± 修正 ──
    emo_stage = emo["stage"] if not emo_stale else None  # 审计 2026-08-16：陈旧情绪基准不参与温度/交叉信号
    temp = float(STAGE_TEMP.get(emo_stage, 40)) if emo_stage else 40.0
    signals: list[str] = []
    if stale:
        signals.append("⚠️ 数据降级: " + "; ".join(stale))

    if stats is not None:
        r = stats.iloc[-1]
        # 炸板率修正（炸板率高 → 降）
        if pd.notna(r.get("zb_rate")) and r["zb_rate"] > 0.35:
            temp -= 15
            signals.append(f"炸板率{r['zb_rate']:.0%}偏高→降温")
        # 溢价修正
        if pd.notna(r.get("premium")) and r["premium"] < 0:
            temp -= 10
            signals.append(f"昨日涨停溢价{r['premium']:.1f}%为负→降温")

    if forces is not None and len(forces):
        fi = float(forces.iloc[-1]["force_index"])
        temp = temp * 0.7 + fi * 0.3
        if fi >= 90:
            signals.append(f"四路资金合力{fi:.0f}极强")
        elif fi < 50:
            signals.append(f"四路资金合力{fi:.0f}偏弱/背离")

    if themes is not None and len(themes):
        last = themes["date"].max()
        day = themes[themes["date"] == last]
        main_lines = day[day["role"] == "主线"]
        # theme_cycle 落盘 stage 为英文键（burst/start/...），中文在 stage_cn
        burst = day[day["stage"] == "burst"] if "stage" in day else day[day.get("stage_cn", "") == "爆发"]
        if len(main_lines) >= 2 and len(burst) >= 2:
            temp += 8
            signals.append(f"{len(burst)}个题材爆发中")
        elif len(day) == 0:
            temp -= 8
            signals.append("题材熄火")

    # 社交（如果有可用数据；只统计最新日期行，避免把历史 ok 混入当日可用数）
    if social is not None and len(social):
        latest_social = social[social["date"] == social["date"].max()]
        ok = latest_social[latest_social["ok"]]
        n_platform = int(latest_social["platform"].nunique())
        if len(ok):
            signals.append(f"社交数据源可用{len(ok)}/{n_platform}")

    # 本地代理温度修正（仅 available 且新鲜 ≤3 日参与）
    if lp_ok:
        proxies = lp.get("proxies", {}) or {}
        rules = {"supply_chain_heat": (3, 3), "consumption_heat": (3, 3),
                 "new_stock_activity": (2, 2), "retail_focus": (0, 3)}
        for key, (up, down) in rules.items():
            p = proxies.get(key)
            if not p:
                continue
            sig = p.get("signal")
            if sig == "升温" and up:
                temp += up
                signals.append(f"{key} 升温→+{up}")
            elif sig == "降温" and down:
                temp -= down
                signals.append(f"{key} 降温→-{down}")

    # 申万 × 中证 双源校验（都可用且方向一致 → 双源一致；分歧 → 修正减半）
    industry_confidence: str | None = None
    csi_b: float | None = None
    if (industry is not None and industry.get("status") == "available"
            and csi_industry is not None and csi_industry.get("status") == "available"):
        sw_b = industry.get("breadth")
        csi_b = csi_industry.get("breadth")
        if sw_b is not None and csi_b is not None:
            industry_confidence = "双源一致" if (sw_b >= 0.5) == (csi_b >= 0.5) else "分歧"

    # ── 行业广度校验（申万 vs 中证 口径一致性提示，仅展示不改温度）────────
    # 两源最新交易日 + 上涨家数比例差 >15pct → 标注"口径分歧"。
    industry_check: dict | None = None
    if (industry is not None and industry.get("status") == "available"
            and csi_industry is not None and csi_industry.get("status") == "available"):
        sw_b = industry.get("breadth")
        csi_b = csi_industry.get("breadth")
        if sw_b is not None and csi_b is not None:
            diff = abs(sw_b - csi_b)
            label = "口径分歧" if diff > 0.15 else "口径一致"
            industry_check = {
                "sw": {"as_of": industry.get("as_of"), "breadth": round(float(sw_b), 4)},
                "csi": {"as_of": csi_industry.get("as_of"), "breadth": round(float(csi_b), 4)},
                "diff_pct": round(diff * 100, 1),
                "label": label,
                "note": (f"申万广度{format_pct_value(sw_b, digits=0, unit='decimal')}（{industry.get('as_of')}） vs "
                         f"中证广度{format_pct_value(csi_b, digits=0, unit='decimal')}（{csi_industry.get('as_of')}），"
                         f"差{diff * 100:.1f}pct → {label}"),
            }
            signals.append(f"行业广度校验: {industry_check['note']}")

    # 行业广度温度修正（申万基准；分歧时减半）
    if industry is not None and industry.get("status") == "available":
        breadth = industry.get("breadth")
        if breadth is not None:
            delta = 0.0
            if breadth >= 0.7:
                delta = 5.0
            elif breadth <= 0.3:
                delta = -5.0
            if industry_confidence == "分歧":
                note = (f"行业口径分歧(申万{format_pct_value(breadth, digits=0, unit='decimal')} vs "
                        f"中证{format_pct_value(csi_b, digits=0, unit='decimal')})")
                if delta:
                    delta = delta / 2
                    note += " → 修正减半"
                signals.append(note)
            elif industry_confidence == "双源一致":
                signals.append(f"行业广度双源一致(申万{format_pct_value(breadth, digits=0, unit='decimal')} vs "
                               f"中证{format_pct_value(csi_b, digits=0, unit='decimal')})")
            if delta > 0:
                temp += delta
                signals.append(f"行业广度{format_pct_value(breadth, digits=0, unit='decimal')}≥70%→+{delta:g}（普涨）")
            elif delta < 0:
                temp += delta
                signals.append(f"行业广度{format_pct_value(breadth, digits=0, unit='decimal')}≤30%→{delta:g}（普跌）")

    # ── 宏观月度环境（轻量标签 + 小修正）────────────────
    macro_env: dict | None = None
    try:
        macro_env = _macro_env(ref_date)
    except Exception as e:  # noqa: BLE001
        macro_env = {"status": "unavailable", "label": None, "cpi": None, "m2": None,
                     "as_of": None, "reason": f"{type(e).__name__}: {e}"}
    if macro_env is not None and macro_env.get("status") == "available":
        cpi = macro_env.get("cpi")
        m2 = macro_env.get("m2")
        if cpi is not None and cpi < 0:
            temp -= 3
            signals.append(f"CPI同比{cpi:.1f}%转负(通缩)→-3")
        if cpi is not None and cpi > 3 and m2 is not None and m2 > 10:
            temp -= 3
            signals.append("流动性宽松+通胀偏高 → 滞胀风险-3")
        elif m2 is not None and m2 > 10:
            if cpi is not None and 0 <= cpi <= 3:
                signals.append(f"流动性宽松(M2同比{m2:.1f}%)+通胀温和(CPI同比{cpi:.1f}%) → 环境健康（不修正）")
            else:
                signals.append(f"流动性宽松(M2同比{m2:.1f}%)")
        elif cpi is not None and cpi > 3:
            signals.append(f"通胀偏高(CPI同比{cpi:.1f}%)")
        elif cpi is not None and m2 is not None and 0 <= cpi <= 3 and m2 <= 10:
            signals.append(f"宏观环境温和(CPI同比{cpi:.1f}%/M2同比{m2:.1f}%)")
        elif cpi is not None and m2 is None:
            signals.append(f"CPI同比{cpi:.1f}%")
    elif macro_env is not None and macro_env.get("status") == "stale":
        lag = _as_of_lag(macro_env.get("as_of"), ref_date)
        lag_note = f"落后{lag}日" if lag is not None else "陈旧"
        signals.append(f"宏观环境{macro_env.get('label')}（{lag_note}, as_of={macro_env.get('as_of')}）"
                       " → 仅展示，不参与温度修正")

    temp = int(max(0, min(100, temp)))
    if temp >= 75:
        tag = "🔥 高温（进攻区，防高潮转分歧）"
    elif temp >= 55:
        tag = "🌤 常温（平衡区）"
    elif temp >= 35:
        tag = "🌧 低温（防守区）"
    else:
        tag = "🥶 冰点（空仓区）"

    # ── 交叉信号 ──
    cross = []
    if emo_stage in ("ferment", "climax") and forces is not None and len(forces) and forces.iloc[-1]["force_index"] >= 75:
        cross.append({"type": "共振", "level": "强", "msg": "情绪向上+资金合力强 → 顺势做多窗口"})
    if emo_stage in ("divergence", "ebb") and forces is not None and len(forces) and forces.iloc[-1]["force_index"] < 50:
        cross.append({"type": "共振", "level": "强", "msg": "情绪退潮+资金流出 → 防守确认"})
    if emo_stage == "climax" and stats is not None and pd.notna(stats.iloc[-1].get("zb_rate")) and stats.iloc[-1]["zb_rate"] > 0.3:
        cross.append({"type": "背离", "level": "⚠️", "msg": "高潮期炸板率抬升 → 高潮末段，防次日分歧"})
    if themes is not None and len(themes):
        last = themes["date"].max()
        day = themes[themes["date"] == last]
        if len(day) and day["zt_cnt"].max() >= 10 and emo_stage in ("divergence", "ebb"):
            cross.append({"type": "背离", "level": "⚠️", "msg": "指数情绪退潮但题材局部活跃 → 结构性机会"})
    if cninfo_ok and alt.get("signal") == "高扰动" and emo_stage in ("ferment", "climax"):
        cross.append({"type": "背离", "level": "⚠️", "msg": "公告高扰动+情绪高潮 → 事件驱动末段，防次日分歧"})
    if (industry is not None and industry.get("status") == "available"
            and industry.get("breadth") is not None and industry["breadth"] >= 0.7
            and forces is not None and len(forces) and forces.iloc[-1]["force_index"] >= 75):
        cross.append({"type": "共振", "level": "强", "msg": "行业普涨+资金合力强 → 全面做多窗口"})
    if hr_ok and hr.get("avg_pct") is not None and hr["avg_pct"] < 0 and emo_stage == "ferment":
        cross.append({"type": "背离", "level": "⚠️", "msg": "热股赚钱效应为负但情绪发酵 → 关注度与收益背离"})

    if industry is not None and industry.get("status") == "available":
        industry_out = dict(industry)
    else:
        industry_out = {"status": "unavailable", "as_of": None, "breadth": None,
                        "top_up": [], "top_down": [],
                        "reason": (industry or {}).get("reason", "无数据")}
    if csi_industry is not None and csi_industry.get("status") == "available":
        industry_out["csi"] = csi_industry
    else:
        industry_out["csi"] = {"status": "unavailable",
                               "as_of": (csi_industry or {}).get("as_of"),
                               "breadth": None, "top_up": [], "top_down": [],
                               "reason": (csi_industry or {}).get("reason", "无数据")}
    if industry_confidence:
        industry_out["confidence"] = industry_confidence
    if industry_check is not None:
        industry_out["check"] = industry_check

    return {
        "date": emo["date"],
        "temperature": temp,
        "tag": tag,
        "emotion_stage": emo["stage_cn"],
        "signals": signals,
        "cross": cross,
        "alt": {
            "status": alt.get("status", "unavailable"),
            "cninfo": {"status": "available" if cninfo_ok else "unavailable",
                       "signal": alt.get("signal") if cninfo_ok else None,
                       "as_of": alt.get("date"),
                       "total_announcements": alt.get("total_announcements") if cninfo_ok else None},
            "hot_rank": hr if hr_ok else {**hr, "status": "unavailable"},
            "social": sc if sc_ok else {**sc, "status": "unavailable"},
            "local_proxies": {"status": "available" if lp_ok else lp.get("status", "unavailable"),
                              "date": lp.get("date") if lp_ok else None,
                              "signal": lp.get("signal") if lp_ok else None,
                              "proxy_signals": {k: v.get("signal")
                                                for k, v in (lp.get("proxies") or {}).items()}},
        },
        "industry": industry_out,
        "macro_env": (macro_env if macro_env is not None
                      and macro_env.get("status") in ("available", "stale")
                      else {"status": "unavailable", "label": None, "cpi": None, "m2": None,
                            "as_of": (macro_env or {}).get("as_of"),
                            "reason": (macro_env or {}).get("reason", "无数据")}),
    }


def store(res: dict) -> None:
    row = pd.DataFrame([{**res, "signals": json.dumps(res["signals"], ensure_ascii=False),
                         "cross": json.dumps(res["cross"], ensure_ascii=False),
                         "alt": json.dumps(res.get("alt", {}), ensure_ascii=False, default=str),
                         "industry": json.dumps(res.get("industry", {}), ensure_ascii=False, default=str),
                         "macro_env": json.dumps(res.get("macro_env", {}), ensure_ascii=False, default=str)}])
    if OUT.exists():
        old = pd.read_parquet(OUT)
        row = pd.concat([old, row], ignore_index=True)
    row = row.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    row.to_parquet(OUT, index=False)
    print(f"[fusion] 已写入 → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="信号融合层")
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()
    res = fuse_today()
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    if args.today:
        store(res)
