#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_learner — 宏观规律自主学习器（域B：消费者信心指数 → 社零 → 消费板块传导链）
=======================================================================
目标：自动发现 房价→消费者信心指数(CCI)→社零→消费板块 的传导链规律并给出权重建议。

输入（只读，data_warehouse 或 MACRO_DATA_DIR 覆盖）:
  - macro/consumer_confidence.parquet  列: 月份 / 消费者信心指数[-指数值]（可含 来源）
  - macro/retail_sales_yoy.parquet     列: 月份 / 同比增长
  - macro/new_house_price.parquet      列: 日期 / 城市 / 新建商品住宅价格指数-同比
  - data_warehouse/industry/sw_first_hist.parquet  列: 代码/日期/收盘（31 申万一级行业）

防前视红线（本模块逐处强制）:
  - CCI/房价/社零 为月度宏观，发布滞后 1 个月：as_of ≤ 分析月-1 个月
  - 行业收益只取 ≤ as_of（分析日）；"下月超额"仅用历史已完成的 t+1 月
  - 任何 future 信息不进入样本

输出（运行时写，MACRO_LEARNER_OUT_DIR 可覆盖；自查写 /tmp，本机真实落盘）:
  - generated/macro_learner_{date}.json + .md
  - generated/macro_learner_history.parquet（历史追加）

用法:
  python3 -m quant_system.analysis_core.macro_learner --date 2026-08-11
  python3 -m quant_system.analysis_core.macro_learner --date 2026-08-11 --json
  MACRO_LEARNER_OUT_DIR=/tmp/macro_learner_test python3 -m ...（自查覆盖输出目录）
"""
from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace 根（与 battle_map/multi_agent/pattern_engine 一致）
OUT_DIR = Path(os.environ.get("MACRO_LEARNER_OUT_DIR", str(ROOT / "generated")))
DATA_DIR = Path(os.environ.get("MACRO_DATA_DIR", str(ROOT / "data_warehouse")))

CST = timezone(timedelta(hours=8))

# 阈值候选（CCI 水平，历史均值 108.47 附近的传统心理分位）
CCI_THRESHOLDS = [85, 86, 87, 88, 89, 90, 92, 95]
# 统计门槛
MIN_CORR_N = 36          # 相关分析最少样本（月）
MIN_THRESH_N = 24        # 阈值分组每侧最少样本（月）
P_THRESH = 0.10          # 阈值 t 检验显著性
MIN_CCI_N = 24           # CCI 最少月数，不足只出点位观察
EXCESS_FLOOR = 0.015     # 超额绝对值 ≥ 1.5%（0.015 小数）才给板块权重建议
# 历史均值（akshare macro_china_xfzxx 2007-01~2026-06 全样本均值，公开数据源）
CCI_HIST_MEAN = 108.47

# 消费类行业（申万一级 6 个）
CONSUMER_SECTORS = ["食品饮料", "家用电器", "商贸零售", "医药生物", "社会服务", "美容护理"]


# ── 小工具 ──────────────────────────────────────────────
def _ts_month(v) -> pd.Timestamp | None:
    """解析为月首 Timestamp，失败 None。"""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return pd.Timestamp(v.year, v.month, 1)
    if isinstance(v, (datetime,)):
        return pd.Timestamp(v.year, v.month, 1)
    s = str(v).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.search(r"(\d{4})-(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    m = re.match(r"^(\d{4})(\d{2})$", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return None


def _ym(v) -> str | None:
    ts = _ts_month(v)
    return ts.strftime("%Y-%m") if ts is not None else None


def _load_parquet(rel: str) -> pd.DataFrame | None:
    p = DATA_DIR / rel
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ macro_learner 读取失败 {p}: {type(e).__name__} {str(e)[:100]}", flush=True)
        return None


def _month_series(df: pd.DataFrame | None, date_col: str, val_col: str,
                  max_ym: str | None = None) -> pd.Series:
    """从数据框构建 月份(YYYY-MM)→值 序列，可选按 max_ym 截断（as_of 防前视）。"""
    if df is None or df.empty or date_col not in df.columns or val_col not in df.columns:
        return pd.Series(dtype=float, name=val_col)
    s = pd.to_numeric(df[val_col], errors="coerce")
    idx = pd.Series([_ym(x) for x in df[date_col]], index=df.index)
    out = pd.Series(s.to_numpy(), index=idx).dropna()
    out = out[~out.index.isna()]
    out = out[out.index.astype(str) != "None"]
    out.index = pd.Index([str(x) for x in out.index])
    if max_ym is not None:
        out = out[out.index <= max_ym]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.name = val_col
    return out


def _industry_monthly_excess(max_ts: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """31 申万一级行业月收益 + 等权超额。返回 (excess_by_sector, meta)。

    - 月收益 = 月末收盘 / 上月末收盘 - 1（对齐自然月）
    - 超额 = 行业月收益 - 31 行业等权月收益
    - 只取日期 ≤ max_ts（as_of 防前视）
    """
    df = _load_parquet("industry/sw_first_hist.parquet")
    if df is None or df.empty:
        return pd.DataFrame(), {}
    need = {"代码", "日期", "收盘"}
    if not need.issubset(df.columns):
        return pd.DataFrame(), {}
    d = df[["代码", "日期", "收盘"]].copy()
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    if max_ts is not None:
        d = d[d["日期"] <= max_ts]
    d = d.dropna(subset=["日期", "收盘"])
    d["代码"] = d["代码"].astype(str).str.replace(r"\.SI$", "", regex=True)
    # 代码 → 行业名称（用 sw_first 快照映射，31 个申万一级）
    meta: dict = {}
    sw = _load_parquet("industry/sw_first.parquet")
    if sw is not None and "行业代码" in sw.columns and "行业名称" in sw.columns:
        for _, row in sw.iterrows():
            code = str(row["行业代码"]).replace(".SI", "")
            meta[code] = str(row["行业名称"])
    d["行业"] = d["代码"].map(meta)
    d = d.dropna(subset=["行业"])
    if d["行业"].nunique() != 31:
        return pd.DataFrame(), {"degraded": f"申万一级行业数={d['行业'].nunique()}，期望 31"}
    # 月末收盘
    d["ym"] = d["日期"].dt.strftime("%Y-%m")
    last = d.sort_values("日期").groupby(["行业", "ym"])["收盘"].last().reset_index()
    last = last.sort_values(["行业", "ym"])
    rets = {}
    for sector, g in last.groupby("行业"):
        vals = g["收盘"].to_numpy()
        r = pd.Series(np.nan, index=g["ym"])
        if len(vals) > 1:
            r.iloc[1:] = vals[1:] / vals[:-1] - 1.0
        rets[sector] = r
    mat = pd.DataFrame(rets).dropna(how="all")
    eq = mat.mean(axis=1)  # 31 行业等权月收益
    excess = mat.sub(eq, axis=0)
    return excess, meta


def _pearson(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    """对齐索引计算 Pearson 相关，返回 (r, n)。"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    n = len(df)
    if n < 2:
        return None, 0
    r = float(np.corrcoef(df["a"], df["b"])[0, 1])
    if not np.isfinite(r):
        return None, n
    return r, n


def _welch_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """双样本 Welch t 检验（不等方差），返回 (t, p)。

    优先用 scipy（已装），缺失时回退 numpy 手写近似（Lentz 连分数，
    统计门槛判定够用，见 domainB_audit.md 校验）。
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, 1.0
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    se = np.sqrt(vx / nx + vy / ny)
    if se <= 0 or not np.isfinite(se):
        return np.nan, 1.0
    t = (mx - my) / se
    try:
        from scipy import stats
        p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        return t, min(max(p, 0.0), 1.0)
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[macro_learner] 操作失败: {e}", exc_info=True)
    # Welch–Satterthwaite 自由度（回退路径：分子补上，公式与 scipy 一致）
    numer = (vx / nx + vy / ny) ** 2
    denom = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = (numer / denom) if denom > 0 else nx + ny - 2
    df = min(max(df, 1.0), nx + ny - 2)
    xb = df / (df + t * t)
    try:
        from scipy.special import betainc
        p = float(betainc(df / 2, 0.5, xb))
    except Exception:  # noqa: BLE001
        p = _betainc_approx(df / 2, 0.5, xb)
    return t, float(2.0 * (1.0 - p))


def _betainc_approx(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 I_x(a,b)，连分数 Lentz 法（数值食谱 betai 公式，含 ÷a）。

    仅作 scipy 缺失时的兜底；结果 ∈[0,1]，小 x 走级数前项不溢出。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import exp, lgamma, log
    if x > (a + 1) / (a + b + 2):  # 对称性：I_x(a,b)=1-I_(1-x)(b,a)
        return 1.0 - _betainc_approx(b, a, 1.0 - x)
    ln_pre = (lgamma(a + b) - lgamma(a) - lgamma(b)
              + a * log(x) + b * log(1.0 - x))
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-8:
            break
    val = exp(ln_pre) * h / a  # 旧实现漏掉 ÷a，小 x 时偏差约 a 倍（概率>1）
    if val != val or val < 0.0 or val > 1.0:
        val = 0.0 if val != val else max(0.0, min(1.0, val))
    return val


# ── 对外接口 ────────────────────────────────────────────
def load_latest_learner_result(date: str | None = None) -> dict | None:
    """读取 ≤date 最新 generated/macro_learner_*.json（只读，缺失返回 None）。"""
    pat = re.compile(r"macro_learner_(\d{4}-\d{2}-\d{2})\.json$")
    best: tuple[str, Path] | None = None
    try:
        for p in OUT_DIR.glob("macro_learner_*.json"):
            m = pat.search(p.name)
            if not m:
                continue
            if date is not None and m.group(1) > date:
                continue
            if best is None or m.group(1) > best[0]:
                best = (m.group(1), p)
    except OSError:
        return None
    if best is None:
        return None
    try:
        return json.loads(best[1].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def format_ai_rules(res: dict | None, max_rules: int = 2) -> list[str]:
    """把规律结果格式化为一行一条的 evidence 文本（供 multi_agent 追加）。"""
    if not res:
        return []
    out: list[str] = []
    th = res.get("thresholds") or []
    for t in th[:max_rules]:
        direction = "高信心利好" if t.get("direction") == "高信心利好" else "低信心利空"
        # 符号：高信心利好 → CCI≥thr 时为正；低信心利空 → CCI<thr 时为正
        sign = "≥" if direction == "高信心利好" else "<"
        if direction == "高信心利好":
            best = t.get("pos_excess", 0)
        else:
            best = t.get("neg_excess", 0)
        out.append(
            f"AI规律: CCI{sign}{t.get('threshold')} 时 {t.get('sector')} 下月超额{best * 100:+.1f}% "
            f"(n={t.get('n_pos', 0)}+{t.get('n_neg', 0)}, p={t.get('p_value', 1):.2f}, {direction})"
        )
    adj = res.get("sector_adjustments") or []
    for a in adj[: max_rules - len(out)]:
        out.append(f"AI规律: {a.get('sector')} 权重建议 factor={a.get('factor')} ({a.get('reason')})")
    return out


def discover(date: str | None = None, write: bool = True) -> dict:
    """核心入口：发现宏观规律并输出建议（详见模块 docstring）。"""
    dstr = date or datetime.now(CST).strftime("%Y-%m-%d")
    try:
        ref = pd.Timestamp(dstr)
    except Exception:  # noqa: BLE001
        ref = pd.Timestamp(datetime.now(CST).strftime("%Y-%m-%d"))
    analysis_ym = ref.strftime("%Y-%m")
    # 月度宏观发布滞后 1 个月：as_of = 分析月 - 1 个月
    as_of_ym = (ref.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")

    degraded: list[str] = []

    # ── 数据准备 ──────────────────────────────────────
    cci_raw = _load_parquet("macro/consumer_confidence.parquet")
    cci_col = next((c for c in ["消费者信心指数", "消费者信心指数-指数值"]
                    if cci_raw is not None and c in cci_raw.columns), None)
    cci = _month_series(cci_raw, "月份", cci_col or "", max_ym=as_of_ym) if cci_col else pd.Series(dtype=float)
    if cci.empty:
        degraded.append("CCI 数据不足：consumer_confidence.parquet 缺失/空（需 月份 + 消费者信心指数 列）")

    retail_raw = _load_parquet("macro/retail_sales_yoy.parquet")
    retail = _month_series(retail_raw, "月份", "同比增长", max_ym=as_of_ym)
    if retail.empty:
        degraded.append("社零数据不足：retail_sales_yoy.parquet 缺失/空（需 月份 + 同比增长 列）")

    hp_raw = _load_parquet("macro/new_house_price.parquet")
    hp = pd.Series(dtype=float)
    if hp_raw is not None and not hp_raw.empty and "新建商品住宅价格指数-同比" in hp_raw.columns:
        hp_s = pd.to_numeric(hp_raw["新建商品住宅价格指数-同比"], errors="coerce")
        idx = pd.Series([_ym(x) for x in hp_raw["日期"]], index=hp_raw.index)
        tmp = pd.Series(hp_s.to_numpy(), index=idx).dropna()
        tmp.index = pd.Index([str(x) for x in tmp.index])
        tmp = tmp[tmp.index != "None"]
        # 全城市月同比均值 = 房价同比（数值型）
        hp = tmp.groupby(level=0).mean()
        hp = hp[hp.index <= as_of_ym]
        hp = hp[~hp.index.duplicated(keep="last")].sort_index()
        hp.name = "房价同比"
    if hp.empty:
        degraded.append("房价数据不足：new_house_price.parquet 缺失/空（需 日期/城市/新建商品住宅价格指数-同比）")

    excess, ind_meta = _industry_monthly_excess(max_ts=ref)
    if excess is None or excess.empty:
        degraded.append(f"行业收益不足：sw_first_hist.parquet 缺失/空（需 31 个申万一级）{ind_meta or ''}")
    # 行业月收益上限：≤ 分析月（as_of）；下月超额仅取历史已完成的 t+1 月
    excess = excess[excess.index <= analysis_ym] if not excess.empty else excess
    # 防前视：分析当月可能未走完（如 08-11 只有 8 月上旬），t+1=当月 的超额不算
    # 完整历史 → 训练用的最大 t = 行业数据最后一月 - 1
    max_excess_ym = excess.index.max() if not excess.empty else None
    complete_cap = None
    if max_excess_ym is not None:
        ym_ts = pd.Timestamp(max_excess_ym + "-01")
        complete_cap = (ym_ts - pd.Timedelta(days=1)).strftime("%Y-%m")

    # ── 1. CCI 水平 vs 各行业同月/下月超额相关 ─────────
    correlations: list[dict] = []
    if not cci.empty and not excess.empty:
        for sector in excess.columns:
            same_r, same_n = _pearson(cci, excess[sector])
            # 下月超额：CCI[t] vs 超额[t+1]，仅用历史已完成月份（t+1 ≤ complete_cap）
            base = excess[sector]
            if complete_cap is not None:
                base = base[base.index <= complete_cap]
            lag = base.shift(-1)
            lag = lag.reindex(cci.index).dropna()
            nxt_r, nxt_n = _pearson(cci, lag)
            rec: dict = {"sector": sector, "same_month_r": same_r, "same_month_n": same_n}
            if nxt_n >= MIN_CORR_N and nxt_r is not None:
                rec["next_month_r"] = nxt_r
                rec["next_month_n"] = nxt_n
            if same_n >= MIN_CORR_N or nxt_n >= MIN_CORR_N:
                correlations.append(rec)
    else:
        degraded.append("相关分析跳过：CCI 或行业超额样本不足")

    # ── 2. 阈值自动发现 ──────────────────────────────
    thresholds: list[dict] = []
    if not cci.empty and not excess.empty:
        # 训练样本：CCI[t] 对 超额[t+1]，t+1 ≤ complete_cap（无未来/无未走完月份）
        # 防前视：先截断 excess 再 shift（与第 1 节相关分析一致），避免未完成月份进入
        excess_capped = excess[excess.index <= complete_cap] if complete_cap is not None else excess
        lag_excess = excess_capped.shift(-1)
        lag_excess = lag_excess.reindex(cci.index)
        pairs = pd.concat([cci.rename("cci"), lag_excess], axis=1).dropna()
        if len(cci) < MIN_CCI_N:
            degraded.append("CCI 样本 < 24 个月：只出点位观察，不出阈值结论")
        elif len(pairs) >= MIN_CCI_N:
            for sector in excess.columns:
                best = None
                for thr in CCI_THRESHOLDS:
                    low = pairs[pairs["cci"] < thr][sector]
                    high = pairs[pairs["cci"] >= thr][sector]
                    if len(low) < MIN_THRESH_N or len(high) < MIN_THRESH_N:
                        continue
                    t, p = _welch_t(low.to_numpy(dtype=float), high.to_numpy(dtype=float))
                    if p >= P_THRESH:
                        continue
                    diff = abs(float(low.mean()) - float(high.mean()))
                    if best is None or diff > best["_diff"]:
                        best = {"sector": sector, "threshold": thr,
                                "pos_excess": round(float(high.mean()), 4),
                                "neg_excess": round(float(low.mean()), 4),
                                "n_pos": int(len(high)), "n_neg": int(len(low)),
                                "p_value": round(p, 4),
                                "direction": "高信心利好" if float(high.mean()) > float(low.mean()) else "低信心利空",
                                "_diff": diff}
                if best:
                    best.pop("_diff", None)
                    thresholds.append(best)
        else:
            degraded.append("阈值发现跳过：配对样本不足")
    else:
        degraded.append("阈值发现跳过：CCI 或行业超额样本不足")

    # ── 3. 传导链（各段独立验证，不跨段拼接因果）──────
    transmission: dict = {}
    # 房价同比 → CCI（同月 + 房价 t 对 CCI t+1）
    if not hp.empty and not cci.empty:
        same_r, same_n = _pearson(hp, cci)
        cci_lag = cci.shift(-1).reindex(hp.index)
        lead_r, lead_n = _pearson(hp, cci_lag)
        transmission["house_price_to_cci"] = {
            "same_month": {"r": same_r, "n": same_n},
            "lead_1m": {"r": lead_r, "n": lead_n},
            "note": "房价同比(城市均值) → CCI：同月相关 + 房价 t 对 CCI t+1",
        }
    else:
        transmission["house_price_to_cci"] = {"note": "数据不足跳过", "degraded": "房价或 CCI 缺失"}
    # CCI → 社零同比（同月 + 1 月领先）
    if not cci.empty and not retail.empty:
        same_r, same_n = _pearson(cci, retail)
        retail_lag = retail.shift(-1).reindex(cci.index)
        lead_r, lead_n = _pearson(cci, retail_lag)
        transmission["cci_to_retail"] = {
            "same_month": {"r": same_r, "n": same_n},
            "lead_1m": {"r": lead_r, "n": lead_n},
            "note": "CCI → 社零同比：同月 + CCI t 对社零 t+1",
        }
    else:
        transmission["cci_to_retail"] = {"note": "数据不足跳过", "degraded": "CCI 或社零缺失"}
    # 社零同比 → 消费类行业 下月超额
    retail_to_consumer: dict = {}
    if not retail.empty and not excess.empty:
        for sector in CONSUMER_SECTORS:
            if sector not in excess.columns:
                continue
            nxt = excess[sector].shift(-1)
            if complete_cap is not None:
                nxt = nxt[nxt.index <= complete_cap]
            nxt = nxt.reindex(retail.index).dropna()
            r, n = _pearson(retail, nxt)
            retail_to_consumer[sector] = {"r": r, "n": n}
        transmission["retail_to_consumer"] = retail_to_consumer
    else:
        transmission["retail_to_consumer"] = {"degraded": "社零或行业超额缺失"}

    # ── 4. 当前落点 ──────────────────────────────────
    current: dict = {"latest_cci": None, "latest_month": None, "mom_chg": None,
                     "pct_rank": None, "vs_hist_mean": None, "hist_mean": CCI_HIST_MEAN}
    if not cci.empty:
        latest_ym = cci.index.max()
        latest = float(cci.loc[latest_ym])
        prev_ym = cci.index[cci.index.get_loc(latest_ym) - 1] if cci.index.get_loc(latest_ym) > 0 else None
        mom = (float(cci.loc[prev_ym]) if prev_ym is not None else np.nan)
        pct_rank = float((cci < latest).mean())
        current = {
            "latest_cci": round(latest, 2),
            "latest_month": latest_ym,
            "mom_chg": round(latest - mom, 2) if prev_ym is not None else None,
            "pct_rank": round(pct_rank, 3),
            "vs_hist_mean": round(latest - CCI_HIST_MEAN, 2),
            "hist_mean": CCI_HIST_MEAN,
        }

    # ── 5. 板块权重建议 ──────────────────────────────
    sector_adjustments: list[dict] = []
    if current.get("latest_cci") is not None and thresholds:
        latest = current["latest_cci"]
        for t in thresholds:
            sector = t["sector"]
            thr = t["threshold"]
            direction = t["direction"]
            pos = t["pos_excess"]   # 高信心侧（CCI ≥ thr）均值
            neg = t["neg_excess"]   # 低信心侧（CCI < thr）均值
            magnitude = abs(pos - neg)
            if magnitude < EXCESS_FLOOR:
                continue
            in_high = latest >= thr
            side_excess = pos if in_high else neg
            if direction == "高信心利好":
                # 高信心侧超额为正 → 高信心利好；当前低信心 = 利空
                if in_high:
                    factor = round(min(1.15, 1.0 + (magnitude - EXCESS_FLOOR) * 0.08), 3)
                    reason = (f"CCI{latest}≥{thr} 高信心侧下月超额 {pos*100:+.2f}%（n={t['n_pos']}, "
                              f"p={t['p_value']:.2f}）→ 增配")
                else:
                    factor = round(max(0.70, 0.85 - (magnitude - EXCESS_FLOOR) * 0.08), 3)
                    reason = (f"CCI{latest}<{thr} 低信心侧下月超额 {neg*100:+.2f}%（n={t['n_neg']}, "
                              f"p={t['p_value']:.2f}）→ 减配")
            else:
                # 高信心侧超额为负 → 高信心利空；当前低信心 = 相对利好
                if in_high:
                    factor = round(max(0.70, 0.85 - (magnitude - EXCESS_FLOOR) * 0.08), 3)
                    reason = (f"CCI{latest}≥{thr} 高信心侧下月超额 {pos*100:+.2f}%（n={t['n_pos']}, "
                              f"p={t['p_value']:.2f}）→ 减配")
                else:
                    factor = round(min(1.15, 1.0 + (magnitude - EXCESS_FLOOR) * 0.08), 3)
                    reason = (f"CCI{latest}<{thr} 低信心侧下月超额 {neg*100:+.2f}%（n={t['n_neg']}, "
                              f"p={t['p_value']:.2f}）→ 增配")
            sector_adjustments.append({
                "sector": sector, "factor": factor, "reason": reason, "threshold": thr,
                "direction": direction, "pos_excess": pos, "neg_excess": neg,
                "n_pos": t["n_pos"], "n_neg": t["n_neg"], "p_value": t["p_value"],
                "trigger_side": "high" if in_high else "low",
                "excess": round(side_excess, 4),
            })
        # 仅保留消费类 6 行业（传导链对象）
        sector_adjustments = [a for a in sector_adjustments if a["sector"] in CONSUMER_SECTORS]
        if not sector_adjustments:
            degraded.append("无触发阈值的消费类行业（阈值规律存在但当前落点未触发/超额<1.5%）")

    result = {
        "date": dstr,
        "generated_at": datetime.now(CST).isoformat(timespec="seconds"),
        "analysis_month": analysis_ym,
        "as_of_month": as_of_ym,
        "cci": current,
        "correlations": correlations,
        "thresholds": thresholds,
        "transmission": transmission,
        "sector_adjustments": sector_adjustments,
        "degraded": degraded,
    }
    if write:
        _write_outputs(result)
    return result


def _write_outputs(result: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date = result["date"]
    json_path = OUT_DIR / f"macro_learner_{date}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = OUT_DIR / f"macro_learner_{date}.md"
    md_path.write_text(_render_md(result), encoding="utf-8")
    # 历史追加
    hist = OUT_DIR / "macro_learner_history.parquet"
    row = pd.DataFrame([{
        "date": date,
        "generated_at": result["generated_at"],
        "result_json": json.dumps(result, ensure_ascii=False),
    }])
    if hist.exists():
        try:
            old = pd.read_parquet(hist)
            old = old[old["date"] != date]  # 同日重跑覆盖
            row = pd.concat([old, row], ignore_index=True)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ macro_learner 历史 parquet 读取失败，本次将覆盖: {type(e).__name__} {str(e)[:100]}", flush=True)
    row.to_parquet(hist, index=False)


def _render_md(result: dict) -> str:
    lines = [
        f"# 宏观规律自主学习（macro_learner）{result['date']}",
        "",
        f"> 分析月 {result['analysis_month']} | as_of（宏观滞后1月）{result['as_of_month']} | "
        f"生成 {result['generated_at']}",
        "",
    ]
    c = result.get("cci") or {}
    if c.get("latest_cci") is not None:
        lines += [
            "## 当前落点",
            f"- CCI（{c.get('latest_month')}）: **{c['latest_cci']}**  环比 {c.get('mom_chg')}  "
            f"历史分位 {c.get('pct_rank')}  相对均值 {c.get('vs_hist_mean')}（均值 {c.get('hist_mean')}）",
            "",
        ]
    th = result.get("thresholds") or []
    if th:
        lines += ["## 触发的阈值规律（CCI → 行业下月超额）", "| 行业 | 阈值 | 低信心均值 | 高信心均值 | n低/n高 | p | 方向 |", "|---|---|---|---|---|---|---|"]
        for t in th:
            sign = "≥" if t.get("direction") == "高信心利好" else "<"
            lines.append(f"| {t['sector']} | CCI{sign}{t['threshold']} | {t['neg_excess']*100:+.1f}% | "
                         f"{t['pos_excess']*100:+.1f}% | {t['n_neg']}/{t['n_pos']} | {t['p_value']:.2f} | {t['direction']} |")
        lines.append("")
    adj = result.get("sector_adjustments") or []
    if adj:
        lines += ["## 板块权重建议", ""]
        for a in adj:
            lines.append(f"- **{a['sector']}**: factor **{a['factor']}** — {a['reason']}")
        lines.append("")
    tr = result.get("transmission") or {}
    if tr:
        lines += ["## 传导链（各段独立验证）", ""]
        hp2cci = tr.get("house_price_to_cci") or {}
        if "same_month" in hp2cci:
            lines.append(f"- 房价同比 → CCI: 同月 r={hp2cci['same_month'].get('r')} (n={hp2cci['same_month'].get('n')})；"
                         f"1月领先 r={hp2cci.get('lead_1m', {}).get('r')} (n={hp2cci.get('lead_1m', {}).get('n')})")
        c2r = tr.get("cci_to_retail") or {}
        if "same_month" in c2r:
            lines.append(f"- CCI → 社零: 同月 r={c2r['same_month'].get('r')} (n={c2r['same_month'].get('n')})；"
                         f"1月领先 r={c2r.get('lead_1m', {}).get('r')} (n={c2r.get('lead_1m', {}).get('n')})")
        r2c = tr.get("retail_to_consumer") or {}
        if isinstance(r2c, dict) and r2c and "degraded" not in r2c:
            parts = [f"{k}: r={v.get('r')} (n={v.get('n')})" for k, v in r2c.items()]
            lines.append(f"- 社零 → 消费行业下月超额: {'；'.join(parts)}")
        lines.append("")
    corr = result.get("correlations") or []
    if corr:
        lines += ["## CCI 与行业超额相关（≥36月）", "| 行业 | 同月 r | 同月 n | 下月 r | 下月 n |", "|---|---|---|---|---|"]
        for c in corr:
            lines.append(f"| {c['sector']} | {c.get('same_month_r')} | {c.get('same_month_n')} | "
                         f"{c.get('next_month_r')} | {c.get('next_month_n')} |")
        lines.append("")
    deg = result.get("degraded") or []
    if deg:
        lines += ["## 降级说明", ""]
        for x in deg:
            lines.append(f"- ⚠️ {x}")
        lines.append("")
    lines += ["---", "*macro_learner 自动生成 | 规律为统计相关/分组检验，不构成投资建议*", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观规律自主学习器")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认今天(CST)")
    ap.add_argument("--json", action="store_true", help="打印结果 JSON")
    ap.add_argument("--no-write", action="store_true", help="不写输出文件（仅返回）")
    args = ap.parse_args()
    r = discover(args.date, write=not args.no_write)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str), flush=True)
    else:
        print(f"date={r['date']} analysis_month={r.get('analysis_month')} "
              f"cci={r.get('cci', {}).get('latest_cci')} thresholds={len(r.get('thresholds') or [])} "
              f"adjustments={len(r.get('sector_adjustments') or [])} degraded={r.get('degraded')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
