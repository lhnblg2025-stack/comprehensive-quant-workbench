#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factor_rotation_system — 体系9 因子轮动系统

方法论（skills/factor-investing-alpha-factors + factor-investing-framework
        + active-alpha-forecast-combination）:
  因子定义    : 动量/价值/质量/规模/低波 五大风格因子（截面 z-score，winsorize 1%/99%）
  因子收益    : 每因子按截面分位构建多空组合（前20%多 / 后20%空），近20日收益 + 截面IC
  因子动量    : 各因子近20日多空收益排名 → 当前占优因子 TOP3（因子轮动信号）
  拥挤度      : 因子截面相关度（去冗余）+ 因子暴露秩稳定性换手代理（数据不足标注）
  风格映射    : 占优因子 → 风格（动量=强势题材 / 价值=红利大盘 / 规模=小盘 / 低波=防御）
  综合        : 占优因子组合加权收益 → 看多(成长/价值/小盘)/看空 + 风格建议
  因子库复用  : factor_zoo.FACTOR_META/list_factors/compute_factor_ic（截面IC与族映射）
  RAG         : knowledge_rag.search('因子轮动 动量 价值 风格', k=3)

输入:
  data_warehouse/kline/{6位代码}.parquet        抽样500只日K（close/outstanding_share）
  data_warehouse/financial/{6位代码}.parquet    可选财务（净资产收益率/每股收益 → 质量/价值）

统一接口:
  from quant_system.analysis_core.factor_rotation_system import FactorRotationSystem
  frs = FactorRotationSystem()
  res = frs.detect(date=None, limit=500)     # 因子截面+多空收益+占优因子
  path = frs.report(date=None, limit=500)    # 写 generated/factor_report_{date}.md
  view = frs.view(date=None)                 # multi_agent 兼容 {agent, signal, ...}

用法:
  python3 -m quant_system.analysis_core.factor_rotation_system --limit 500 --report
  python3 -m quant_system.analysis_core.factor_rotation_system --limit 500 --view --date 2026-08-10
"""

from __future__ import annotations
import logging

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # noqa: F401
except Exception:  # pragma: no cover
    pq = None

DATA_ROOT = Path(__file__).resolve().parent.parent.parent          # workspace（data_warehouse 所在）
REPO_ROOT = Path(__file__).resolve().parent.parent                 # 仓库根（generated/ 输出目录）
KLINE_DIR = DATA_ROOT / "data_warehouse" / "kline"
FIN_DIR = DATA_ROOT / "data_warehouse" / "financial"
DEFAULT_OUT_DIR = REPO_ROOT / "generated"

sys.path.insert(0, str(DATA_ROOT))
from quant_system.analysis_core.common import (  # noqa: E402
    kline_files,
    load_names,
    read_kline_window,
    signal_view,
)

# 测试兼容别名（tests/test_factor_rotation_system.py 引用模块级 _signal_view）
_signal_view = signal_view

CODE_RE = re.compile(r"^\d{6}$")

# ── 参数 ─────────────────────────────────────────────────────────
DEFAULT_SAMPLE = 500          # 抽样股数（性能: 全流程 <60s）
LOOKBACK_CAL_DAYS = 170       # 日历回看窗口（≈115 交易日，覆盖 t-70 用于换手代理）
MIN_BARS = 22                 # 至少 22 根K线（20日动量/收益窗口）
MIN_BARS_MOM60 = 62           # 60日动量最少样本
FIN_LAG_DAYS = 45             # 财报滞后假设（避免最新财报未披露即被使用的前视偏差）
LS_QUANTILE = 0.20            # 多空分位（前20%多 / 后20%空）
RET_WINDOW = 20               # 近20日收益窗口
TURNOVER_LOOKBACK = 10        # 换手代理: 距当前 10 个交易日的暴露秩对比
MIN_FACTOR_N = 30             # 因子有效样本下限（低于 → 数据不足，不参与排名）
MIN_SAMPLE_OK = 30            # 有效样本下限（低于 → 整体降级）
RAG_QUERY = "因子轮动 动量 价值 风格"
RAG_K = 3
VIEW_WEIGHT = 1.0             # view() 默认权重（multi_agent 仲裁权重在 multi_agent.BASE_WEIGHTS 维护）
PROGRESS_EVERY = 200

# 五大风格因子: 名称 -> (中文名, 方向定义说明)
FACTOR_NAMES: dict[str, str] = {
    "momentum": "动量",
    "value": "价值",
    "quality": "质量",
    "size": "规模",
    "low_vol": "低波",
}

# 占优因子 → 风格映射
STYLE_MAP: dict[str, dict[str, str]] = {
    "momentum": {"style": "强势题材/趋势成长", "mode": "进攻",
                 "desc": "追强势题材、趋势延续（动量占优）"},
    "value": {"style": "红利大盘/低估值", "mode": "价值",
              "desc": "低估值高股息大盘蓝筹（价值回归占优）"},
    "quality": {"style": "绩优蓝筹", "mode": "防御",
                "desc": "高ROE核心资产（质量防御占优）"},
    "size": {"style": "小盘", "mode": "进攻",
             "desc": "小市值弹性（规模因子占优，注意流动性/退市风险）"},
    "low_vol": {"style": "防御", "mode": "防御",
                "desc": "低波动防守配置（低波占优）"},
}

# 因子 → factor_zoo 因子库族映射（复用现有因子库做交叉验证）
FACTOR_ZOO_MAP: dict[str, list[str]] = {
    "momentum": ["mom_1m", "mom_3m", "mom_6m", "mom_12m"],
    "value": ["ep", "bp", "pe_ttm", "pb", "div_yield"],
    "quality": ["roe", "roa", "gross_margin", "net_margin"],
    "size": [],   # factor_zoo 无规模因子，用 kline outstanding_share 独立计算
    "low_vol": ["vol_20d", "beta_60d", "idio_vol_60d", "max_dd_12m"],
}


# ────────────────────────────────────────────────────────────
# 数据读取
# ────────────────────────────────────────────────────────────
def read_financial(path: Path, as_of: pd.Timestamp | None = None) -> dict[str, float | None]:
    """读财务 parquet → {roe, eps}。取 日期 ≤ as_of-FIN_LAG_DAYS 的最新一条（防前视）。

    文件缺失/读取失败/无有效期 → 返回空 dict（调用方标注数据不足）。
    """
    out: dict[str, float | None] = {}
    try:
        cols = ["日期", "净资产收益率(%)", "摊薄每股收益(元)"]
        if pq is not None:
            df = pq.read_table(path, columns=cols).to_pandas()
        else:
            df = pd.read_parquet(path, columns=cols)
        if df.empty:
            return out
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        cutoff = (as_of or pd.Timestamp.now()).normalize() - pd.Timedelta(days=FIN_LAG_DAYS)
        latest = df[df["日期"] <= cutoff]
        if latest.empty:
            return out
        row = latest.sort_values("日期").iloc[-1]
        roe = row.get("净资产收益率(%)")
        if pd.notna(roe):
            out["roe"] = float(roe)
        eps = row.get("摊薄每股收益(元)")
        if pd.notna(eps):
            out["eps"] = float(eps)
    except Exception as e:
        logging.getLogger(__name__).error(f"[factor_rotation_system] 操作失败: {e}", exc_info=True)
    return out


# ────────────────────────────────────────────────────────────
# 因子计算（纯价格/规模，复用 factor_zoo 做 IC 与族映射）
# ────────────────────────────────────────────────────────────
def _winsorize_zscore(s: pd.Series) -> pd.Series:
    """截面 z-score：1%/99% winsorize → 标准化 → clip ±3。"""
    vals = pd.to_numeric(s, errors="coerce")
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    trimmed = vals.clip(lo, hi)
    std = trimmed.std()
    if not np.isfinite(std) or std <= 1e-12:
        return trimmed * 0.0
    return ((trimmed - trimmed.mean()) / std).clip(-3, 3)


def _raw_factors_from_close(close: np.ndarray, share: float | None) -> dict[str, float | None]:
    """从单只股票K线序列计算原始因子值（截至序列末尾）。

    momentum 用 20/60 日收益；low_vol 用 20 日对数收益 std（负值，高=低波）；
    size 用 -ln(流通市值)（高=小盘）。
    """
    out: dict[str, float | None] = {
        "mom20": None, "mom60": None, "vol20": None, "size": None,
    }
    close = np.asarray(close, dtype=float)
    close = close[np.isfinite(close)]
    if len(close) < MIN_BARS or close[-1] <= 0:
        return out
    if len(close) >= MIN_BARS and close[-MIN_BARS] > 0:
        out["mom20"] = float(close[-1] / close[-MIN_BARS] - 1.0)
    if len(close) >= MIN_BARS_MOM60 and close[-MIN_BARS_MOM60] > 0:
        out["mom60"] = float(close[-1] / close[-MIN_BARS_MOM60] - 1.0)
    logr = np.diff(np.log(np.maximum(close, 1e-12)))
    if len(logr) >= RET_WINDOW:
        out["vol20"] = -float(np.std(logr[-RET_WINDOW:]) * (252.0 ** 0.5))
    if share is not None and np.isfinite(share) and share > 0 and close[-1] > 0:
        out["size"] = -float(np.log(close[-1] * share))
    return out


def _factor_cross_section(
    stocks: dict[str, dict], at_offset: int = 0
) -> tuple[pd.DataFrame, dict[str, int]]:
    """截面因子值（z-score 后五大因子） + 各因子有效样本数。

    stocks: {code: {"close": np.ndarray, "share": float|None, "fin": {roe, eps}}}
    at_offset: 0=当前日, >0 = 往前 N 个交易日（用于换手代理的暴露秩）
    """
    raw = {}
    for code, s in stocks.items():
        close = s["close"]
        if at_offset > 0:
            if len(close) <= at_offset:
                continue
            close = close[:-at_offset]
        r = _raw_factors_from_close(close, s.get("share"))
        r["roe"] = None
        r["ep"] = None
        fin = s.get("fin") or {}
        if at_offset == 0:
            if fin.get("roe") is not None:
                r["roe"] = float(fin["roe"])
            if fin.get("eps") is not None and len(close) and close[-1] > 0:
                r["ep"] = float(fin["eps"]) / float(close[-1])
        raw[code] = r
    if not raw:
        return pd.DataFrame(), {}

    df = pd.DataFrame.from_dict(raw, orient="index")
    coverage = {c: int(df[c].notna().sum()) for c in ("mom20", "mom60", "vol20", "size", "roe", "ep")}

    mom = pd.DataFrame(index=df.index)
    mom["mom20"] = _winsorize_zscore(df["mom20"])
    mom["mom60"] = _winsorize_zscore(df["mom60"])
    factors = pd.DataFrame(index=df.index)
    # 动量轮动信号使用60日窗口，避免与下方20日验证收益同源导致排名虚高。
    # mom20仍保留在原始计算中供诊断，但不进入本轮综合排名。
    factors["momentum"] = mom["mom60"]

    # 价值: 优先财务 EP（PE倒数）；财务不足(>=30)时用价格动量代理 -mom60（标注 proxy）
    value_proxy = coverage["ep"] >= MIN_FACTOR_N
    if value_proxy:
        factors["value"] = _winsorize_zscore(df["ep"])
    else:
        factors["value"] = _winsorize_zscore(-df["mom60"])
    factors["quality"] = _winsorize_zscore(df["roe"])
    factors["size"] = _winsorize_zscore(df["size"])
    factors["low_vol"] = _winsorize_zscore(df["vol20"])
    return factors, coverage


# ────────────────────────────────────────────────────────────
# 因子收益 / IC / 拥挤度
# ────────────────────────────────────────────────────────────
def _factor_ls_return(values: pd.Series, ret20: pd.Series, q: float = LS_QUANTILE,
                      min_n: int = MIN_FACTOR_N) -> float:
    """截面分位多空组合近20日收益: 前q分位做多 / 后q分位做空。"""
    v = values.reindex(ret20.index)
    m = v.notna() & ret20.notna()
    if int(m.sum()) < min_n:
        return 0.0
    long_mask = v >= v[m].quantile(1 - q)
    short_mask = v <= v[m].quantile(q)
    long_ret = ret20[long_mask & m].mean()
    short_ret = ret20[short_mask & m].mean()
    if pd.isna(long_ret) or pd.isna(short_ret):
        return 0.0
    return float(long_ret - short_ret)


def _cross_section_ic(name: str, values: pd.Series, ret20: pd.Series) -> float:
    """截面IC：复用 factor_zoo.compute_factor_ic（因子库统一口径）。"""
    try:
        from quant_system import factor_zoo
        return float(factor_zoo.compute_factor_ic(name, values, ret20))
    except Exception:
        aligned = pd.concat([values, ret20], axis=1, join="inner").dropna()
        if len(aligned) < 10:
            return 0.0
        x = aligned.iloc[:, 0].astype(float)
        y = aligned.iloc[:, 1].astype(float)
        if x.nunique() <= 1 or y.nunique() <= 1:
            return 0.0
        return float(x.corr(y, method="spearman")) or 0.0


def _crowding(factors: pd.DataFrame, factors_past: pd.DataFrame | None,
              coverage: dict[str, int]) -> dict:
    """拥挤度: 因子截面相关度（去冗余）+ 暴露秩稳定性换手代理（可选）。"""
    out: dict = {"cross_corr": 0.0, "top3_corr": 0.0, "turnover_proxy": None,
                 "note": "", "available": True}
    if len(factors) < 10:
        out.update({"available": False, "note": "截面样本不足，拥挤度不可用"})
        return out
    corr = factors.corr().abs()
    vals = corr.values[np.triu_indices(len(corr), k=1)]
    if len(vals):
        out["cross_corr"] = round(float(np.nanmean(vals)), 3)
    if len(corr) >= 3:
        tri = corr.values[np.triu_indices(len(corr), k=1)]
        out["top3_corr"] = round(float(np.nanmean(tri)), 3)
    # 换手代理: 1 - 秩相关(现暴露 vs 10日前暴露) 均值
    if factors_past is not None and len(factors_past) >= 10:
        stab = []
        for c in factors.columns:
            common = factors.index.intersection(factors_past.index)
            if len(common) < 10:
                continue
            a = factors.loc[common, c].rank()
            b = factors_past.loc[common, c].rank()
            if a.nunique() <= 1 or b.nunique() <= 1:
                continue
            r = a.corr(b, method="spearman")
            if pd.notna(r):
                stab.append(float(r))
        if stab:
            out["turnover_proxy"] = round(1.0 - float(np.mean(stab)), 3)
        else:
            out["note"] = "暴露历史不足，换手代理未计算"
    else:
        out["note"] = "暴露历史不足，换手代理未计算（仅截面相关度）"
    return out


# ────────────────────────────────────────────────────────────
# 风格映射 / 综合
# ────────────────────────────────────────────────────────────
def _style_suggestion(top: list[dict], composite: dict) -> dict:
    """占优因子 → 风格映射 + 综合风格建议。"""
    styles = []
    for t in top:
        meta = STYLE_MAP.get(t["factor"], {"style": t["factor"], "mode": "观察", "desc": ""})
        styles.append({"factor": FACTOR_NAMES.get(t["factor"], t["factor"]),
                       "style": meta["style"], "mode": meta["mode"], "desc": meta["desc"]})
    primary = styles[0] if styles else {"factor": "", "style": "观察", "mode": "观察", "desc": ""}
    if composite.get("signal") == "空":
        suggestion = {
            "primary": {"style": "防御/红利低波", "mode": "防御",
                        "desc": "占优因子组合亏损，风格转防御，降低进攻仓位"},
            "secondary": primary,
            "note": "因子整体走弱（多空收益为负）→ 风险偏好收缩，倾向防御风格",
        }
    else:
        suggestion = {
            "primary": primary,
            "secondary": styles[1] if len(styles) > 1 else primary,
            "note": "占优因子组合盈利 → 沿占优因子风格配置，兼顾第二占优因子",
        }
    return suggestion


def _composite_signal(top: list[dict]) -> dict:
    """综合: 占优因子组合加权收益 → 多/空/震荡。"""
    if not top:
        return {"signal": "震荡", "view": "震荡", "weighted_ret": 0.0, "note": "无有效因子"}
    weights = [0.5, 0.3, 0.2]
    wret = sum(w * t["ls_return"] for w, t in zip(weights, top))
    top1 = top[0]["ls_return"]
    if wret > 0 and top1 > 0:
        signal, view = "多", "看多"
        note = "占优因子组合近20日收益为正 → 沿占优风格做多"
    elif wret < 0 and top1 < 0:
        signal, view = "空", "看空"
        note = "占优因子组合近20日收益为负 → 风格退潮，规避/防御"
    else:
        signal, view = "震荡", "震荡"
        note = "占优因子多空收益方向不一致 → 轮动切换期，观望"
    return {"signal": signal, "view": view, "weighted_ret": round(float(wret), 4), "note": note}


def _confidence(meta: dict, top: list[dict], crowding: dict) -> float:
    """置信度: 样本覆盖 + 占优因子强度 + 拥挤惩罚 + 数据完整度。"""
    conf = 0.45
    n_ok = meta.get("parsed", 0)
    conf += min(0.15, n_ok / max(1, meta.get("sample_target", DEFAULT_SAMPLE)) * 0.15)
    if top:
        top1 = abs(top[0]["ls_return"])
        conf += min(0.15, top1 * 0.5)
        if len(top) >= 3 and all(t["ls_return"] > 0 for t in top):
            conf += 0.05
        elif len(top) >= 3 and all(t["ls_return"] < 0 for t in top):
            conf += 0.05
    if crowding.get("available") and crowding.get("cross_corr", 0) > 0.6:
        conf -= 0.10
    cov = meta.get("factor_coverage", {})
    if cov:
        ratio = np.mean([min(1.0, cov.get(f, 0) / max(1, n_ok)) for f in FACTOR_NAMES])
        conf -= (1.0 - ratio) * 0.10
    return round(float(np.clip(conf, 0.15, 0.9)), 2)


# ────────────────────────────────────────────────────────────
# 主类
# ────────────────────────────────────────────────────────────
class FactorRotationSystem:
    """因子轮动系统：detect() 检测 / report() 写 md / view() 输出 multi_agent 兼容观点。"""

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self._names: dict[str, str] | None = None
        self._rag_cache: dict | None = None

    def _name_map(self) -> dict[str, str]:
        if self._names is None:
            self._names, _ = load_names()
        return self._names

    def _rag(self) -> dict:
        """RAG 方法论依据: knowledge_rag.search('因子轮动 动量 价值 风格', k=3)。"""
        if self._rag_cache is not None:
            return self._rag_cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from quant_system.analysis_core import knowledge_rag
            hits = knowledge_rag.search(RAG_QUERY, k=RAG_K)
            if hits:
                self._rag_cache = {
                    "query": RAG_QUERY, "available": True, "basis": "检索可用",
                    "hits": [{"file": h.get("file"), "cat": h.get("cat", ""),
                              "score": h.get("score"),
                              "summary": (h.get("text") or "")[:200]} for h in hits],
                }
            else:
                self._rag_cache = {"query": RAG_QUERY, "available": False,
                                   "basis": "检索无命中", "hits": []}
        except Exception as e:
            self._rag_cache = {"query": RAG_QUERY, "available": False, "hits": [],
                               "basis": "检索不可用", "error": str(e)[:120]}
        return self._rag_cache

    def _resolve_target(self, files: list[Path], date: str | None) -> pd.Timestamp:
        """目标日期：--date 优先；否则样本K线最新共同交易日（取最近30只的最大日期）。"""
        if date:
            return pd.Timestamp(date).normalize()
        kline_max = None
        for f in files[: min(30, len(files))]:
            try:
                d = pd.read_parquet(f, columns=["date"])
                m = pd.to_datetime(d["date"], errors="coerce").max()
                if m is not None and (kline_max is None or m > kline_max):
                    kline_max = m
            except Exception as e:
                logging.getLogger(__name__).error(f"[factor_rotation_system] 操作失败: {e}", exc_info=True)
                continue
        return (kline_max or pd.Timestamp("today")).normalize()

    # ── detect ───────────────────────────────────────────
    def detect(self, date: str | None = None, limit: int | None = None) -> dict:
        """因子轮动检测：截面因子 → 多空收益 → 占优因子TOP3 → 风格映射 → 综合。

        返回 {date, factors, top_factors, crowding, style, composite, meta, rag}。
        """
        t0 = time.time()
        limit = limit or DEFAULT_SAMPLE
        files = kline_files(limit)
        names = self._name_map()
        target = self._resolve_target(files, date)
        window_start = target - pd.Timedelta(days=LOOKBACK_CAL_DAYS)

        meta = {
            "target_date": str(target.date()), "universe": "fullA_sample",
            "sample_target": len(files), "parsed": 0, "skip_read_error": 0,
            "skip_empty": 0, "skip_no_data": 0, "insufficient": 0,
            "elapsed_sec": 0.0, "status": "ok",
            "factor_coverage": {}, "value_source": "unknown",
        }

        stocks: dict[str, dict] = {}
        for i, f in enumerate(files, 1):
            code = f.stem
            if not f.exists():
                meta["skip_read_error"] += 1
                continue
            try:
                df = read_kline_window(f, ["date", "close", "outstanding_share"], window_start)
            except Exception:
                meta["skip_read_error"] += 1
                continue
            if df is None or df.empty:
                meta["skip_empty"] += 1
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            df = df[df["date"] <= target]
            if len(df) < MIN_BARS:
                meta["insufficient"] += 1
                continue
            closes = df["close"].astype(float).to_numpy()
            closes = closes[np.isfinite(closes)]
            if len(closes) < MIN_BARS:
                meta["insufficient"] += 1
                continue
            share = df["outstanding_share"].dropna()
            share_val = float(share.iloc[-1]) if len(share) else None
            stocks[code] = {"close": closes, "share": share_val, "fin": {}}
            meta["parsed"] += 1
            if i % PROGRESS_EVERY == 0:
                print(f"[进度] {i}/{len(files)} | 有效 {meta['parsed']} | "
                      f"耗时 {time.time() - t0:.0f}s", flush=True)

        # 财务（质量/价值）: 仅对已入样股票读取，失败静默 → 标注数据不足
        if stocks:
            for code in list(stocks):
                try:
                    fin = read_financial(FIN_DIR / f"{code}.parquet", as_of=target)
                    stocks[code]["fin"] = fin
                except Exception:
                    stocks[code]["fin"] = {}
        fin_cov = sum(1 for s in stocks.values() if s["fin"].get("roe") is not None)
        meta["financial_roe_cov"] = fin_cov

        factors, coverage = _factor_cross_section(stocks, at_offset=0)
        factors_past, _ = _factor_cross_section(stocks, at_offset=TURNOVER_LOOKBACK)
        meta["factor_coverage"] = coverage
        meta["value_source"] = ("财务EP(PE倒数)" if coverage.get("ep", 0) >= MIN_FACTOR_N
                                else "价格动量代理(-60日)")

        # 近20日收益截面（与因子同源，避免再次读盘）
        ret20 = pd.Series({code: (s["close"][-1] / s["close"][-MIN_BARS] - 1.0)
                           for code, s in stocks.items()
                           if len(s["close"]) >= MIN_BARS and s["close"][-MIN_BARS] > 0})
        ret20 = ret20.replace([np.inf, -np.inf], np.nan).dropna()

        factor_rows = []
        for fname in FACTOR_NAMES:
            values = factors[fname] if fname in factors.columns else pd.Series(dtype=float)
            ls_ret = _factor_ls_return(values, ret20)
            ic = _cross_section_ic(fname, values.reindex(ret20.index), ret20)
            n_valid = int(values.notna().sum())
            factor_rows.append({
                "factor": fname, "name": FACTOR_NAMES[fname],
                "ls_return": ls_ret, "ic": ic, "n_valid": n_valid,
                "data_ok": n_valid >= MIN_FACTOR_N,
                "zoo_factors": FACTOR_ZOO_MAP.get(fname, []),
            })

        ok_rows = [r for r in factor_rows if r["data_ok"]]
        top = sorted(ok_rows, key=lambda r: -r["ls_return"])[:3]
        for i, t in enumerate(top, 1):
            t["rank"] = i

        composite = _composite_signal(top)
        style = _style_suggestion(top, composite)
        crowd = _crowding(factors, factors_past, coverage)

        if meta["parsed"] < MIN_SAMPLE_OK:
            meta["status"] = "degraded"
            composite = {"signal": "震荡", "view": "震荡", "weighted_ret": 0.0,
                         "note": f"有效样本 {meta['parsed']} < {MIN_SAMPLE_OK}，检测降级"}
        conf = _confidence(meta, top, crowd)
        meta["elapsed_sec"] = round(time.time() - t0, 1)

        return {
            "date": str(target.date()),
            "factors": factor_rows,
            "top_factors": top,
            "crowding": crowd,
            "style": style,
            "composite": composite,
            "confidence": conf,
            "meta": meta,
            "rag": self._rag(),
        }

    # ── report ───────────────────────────────────────────
    def report(self, date: str | None = None, limit: int | None = None,
               out_dir: str | Path | None = None) -> Path:
        """检测并写 generated/factor_report_{date}.md，返回文件路径。"""
        res = self.detect(date=date, limit=limit)
        out = Path(out_dir) if out_dir else self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"factor_report_{res['date']}.md"
        path.write_text(render_markdown(res), encoding="utf-8")
        return path

    # ── view ─────────────────────────────────────────────
    def view(self, date: str | None = None, limit: int | None = None) -> dict:
        """multi_agent 兼容观点 {agent, signal, view, confidence, evidence, weight, status, detail}。

        任何异常降级为震荡，不崩溃。
        """
        try:
            res = self.detect(date=date, limit=limit)
        except Exception as e:
            return self._degraded_view(f"检测异常: {str(e)[:120]}")
        rag_basis = res.get("rag", {}).get("basis", "检索不可用")
        top = res.get("top_factors", [])
        composite = res.get("composite", {})
        if not top:
            return self._degraded_view("无有效因子样本", rag_basis=rag_basis, date=res["date"])
        signal = composite.get("signal", "震荡")
        evidence = self._evidence_from_res(res)
        evidence.append(f"RAG: {rag_basis}")
        return {
            "agent": "因子轮动", "signal": signal, "view": signal_view(composite.get("view", signal)),
            "confidence": float(res.get("confidence", 0.3)),
            "evidence": evidence, "weight": VIEW_WEIGHT, "status": res["meta"].get("status", "ok"),
            "detail": {
                "date": res["date"], "universe": res["meta"].get("universe"),
                "top_factors": [t["factor"] for t in top],
                "factor_returns": {t["factor"]: t["ls_return"] for t in res.get("factors", [])},
                "style": res.get("style", {}).get("primary", {}),
                "crowding": res.get("crowding", {}),
                "weighted_ret": composite.get("weighted_ret"),
                "sample": res["meta"].get("parsed"),
                "elapsed_sec": res["meta"].get("elapsed_sec"),
            },
            "rag_basis": rag_basis,
        }

    def _degraded_view(self, msg: str, rag_basis: str = "检索不可用",
                       date: str | None = None) -> dict:
        return {"agent": "因子轮动", "signal": "震荡", "view": "震荡", "confidence": 0.0,
                "evidence": [msg, f"RAG: {rag_basis}"], "status": "degraded",
                "weight": VIEW_WEIGHT, "detail": {"date": date or ""}, "rag_basis": rag_basis}

    @staticmethod
    def _evidence_from_res(res: dict) -> list[str]:
        ev: list[str] = []
        top = res.get("top_factors", [])
        for t in top:
            st = STYLE_MAP.get(t["factor"], {})
            ev.append(f"占优因子#{t.get('rank', 0)} {t['name']}({t['factor']}): "
                      f"多空收益 {t['ls_return']:+.2%} IC {t['ic']:+.3f} "
                      f"→ 风格[{st.get('style', '观察')}]")
        crowd = res.get("crowding", {})
        if crowd.get("available"):
            ev.append(f"拥挤度: 截面相关 {crowd.get('cross_corr', 0):.2f}"
                      + (f" 换手代理 {crowd.get('turnover_proxy'):.2f}"
                         if crowd.get("turnover_proxy") is not None else "")
                      + (" ⚠高相关拥挤" if crowd.get("cross_corr", 0) > 0.6 else ""))
        else:
            ev.append(f"拥挤度: {crowd.get('note', '数据不足')}")
        composite = res.get("composite", {})
        ev.append(f"综合: {composite.get('signal', '震荡')} | {composite.get('note', '')}"
                  f"（加权收益 {composite.get('weighted_ret', 0):+.2%}）")
        st = res.get("style", {})
        primary = st.get("primary", {})
        if primary:
            ev.append(f"风格建议: {primary.get('style', '')}（{primary.get('desc', '')}）")
        meta = res.get("meta", {})
        ev.append(f"样本: {meta.get('parsed')}/{meta.get('sample_target')} 只"
                  f" | 价值来源 {meta.get('value_source', '')}"
                  f" | 耗时 {meta.get('elapsed_sec')}s")
        return ev


# ────────────────────────────────────────────────────────────
# 报告渲染
# ────────────────────────────────────────────────────────────
def _fmt_pct(v: float) -> str:
    return f"{v:+.2%}" if np.isfinite(v) else "--"


def _fmt_ic(v: float) -> str:
    return f"{v:+.3f}" if np.isfinite(v) else "--"


def render_markdown(res: dict) -> str:
    meta = res["meta"]
    rag = res.get("rag", {})
    lines = [
        f"# 因子轮动报告 {res['date']}",
        "",
        f"- 生成: {meta.get('target_date')} | 抽样 {meta.get('sample_target')} 只"
        f"（有效 {meta.get('parsed')}） | 耗时 {meta.get('elapsed_sec')}s",
        f"- 状态: {meta.get('status')} | 价值因子来源: {meta.get('value_source')}",
        f"- RAG 依据: `{rag.get('query', RAG_QUERY)}` → {rag.get('basis', '检索不可用')}",
        "",
        "## 一、综合结论",
        "",
        f"- 信号: **{res['composite'].get('signal')}**（{res['composite'].get('view')}）"
        f" | 置信度 {res.get('confidence'):.2f}",
        f"- 占优因子组合加权收益(近20日): {_fmt_pct(res['composite'].get('weighted_ret', 0))}",
        f"- 说明: {res['composite'].get('note', '')}",
        "",
        "### 风格映射",
        "",
    ]
    st = res.get("style", {})
    primary = st.get("primary", {})
    secondary = st.get("secondary", {})
    if primary:
        lines.append(f"- 主风格: **{primary.get('style')}**（{primary.get('mode')}）— {primary.get('desc')}")
    if secondary:
        lines.append(f"- 次风格: {secondary.get('style')}（{secondary.get('mode')}）— {secondary.get('desc')}")
    lines.append(f"- 备注: {st.get('note', '')}")
    lines += ["", "## 二、因子收益（截面多空组合, 近20日）", "",
              "| 排名 | 因子 | 多空收益(近20日) | 截面IC | 覆盖样本 | 数据 | factor_zoo 映射 |",
              "|---|---|---|---|---|---|---|"]
    for t in sorted(res["factors"], key=lambda r: -r["ls_return"]):
        zoo = ",".join(t.get("zoo_factors", [])) if t.get("zoo_factors") else "—(独立计算)"
        lines.append(f"| {t.get('rank', '') or '—'} | {t['name']} | {_fmt_pct(t['ls_return'])} | "
                     f"{_fmt_ic(t['ic'])} | {t['n_valid']} | "
                     f"{'✓' if t['data_ok'] else '⚠数据不足'} | {zoo} |")
    lines += ["", "## 三、占优因子 TOP3（因子动量）", ""]
    top = res.get("top_factors", [])
    if top:
        for t in top:
            m = STYLE_MAP.get(t["factor"], {})
            lines.append(f"- **{t['name']}** 多空收益 {_fmt_pct(t['ls_return'])} "
                         f"IC {_fmt_ic(t['ic'])} → 风格[{m.get('style', '观察')}]")
    else:
        lines.append("- 无有效因子样本，无法给出占优因子。")
    lines += ["", "## 四、拥挤度", ""]
    crowd = res.get("crowding", {})
    if crowd.get("available"):
        lines.append(f"- 因子截面平均相关度: {crowd.get('cross_corr', 0):.3f}"
                     + ("（⚠ >0.6 拥挤，因子间冗余度高）" if crowd.get("cross_corr", 0) > 0.6 else ""))
        tp = crowd.get("turnover_proxy")
        if tp is not None:
            lines.append(f"- 暴露换手代理(10日秩变化): {tp:.3f}"
                         + ("（⚠ 换手偏高，拥挤风险上升）" if tp > 0.5 else ""))
        if crowd.get("note"):
            lines.append(f"- 备注: {crowd['note']}")
    else:
        lines.append(f"- {crowd.get('note', '数据不足，拥挤度不可用')}")
    lines += ["", "## 五、因子库复用（factor_zoo）", ""]
    lines.append("- 截面IC统一走 `factor_zoo.compute_factor_ic`（spearman，与因子库同口径）。")
    lines.append("- 五大类因子 ↔ factor_zoo 族映射：")
    for fname, zoo_fs in FACTOR_ZOO_MAP.items():
        zh = FACTOR_NAMES[fname]
        lines.append(f"  - {zh}({fname}) ↔ {', '.join(zoo_fs) if zoo_fs else '独立计算(规模=流通市值)'}")
    lines += ["", "## 六、RAG 依据", ""]
    for h in rag.get("hits", [])[:RAG_K]:
        lines.append(f"- [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')}")
    if not rag.get("hits"):
        lines.append(f"- {rag.get('basis', '检索不可用')}")
    lines += ["", "## 七、防御与局限", "",
              "- 抽样失败/样本不足(<30) → 状态 degraded，信号降级为震荡。",
              "- 财务数据不足时: 价值因子用价格动量代理(-60日)并标注，质量因子缺失标注。",
              "- 多空收益为历史已实现收益（近20日），非前瞻收益，仅供因子轮动强度参考。",
              "- 动量因子与收益窗口同源（mom20≈近20日收益），其多空收益/IC 天然偏高，"
              "解读时以相对排序为主。",
              "- 财报滞后假设: 仅用 `日期 ≤ 目标-45天` 的最新财报，规避前视偏差。",
              ""]
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="体系9 因子轮动系统（动量/价值/质量/规模/低波）")
    ap.add_argument("--limit", type=int, default=DEFAULT_SAMPLE, help="抽样股数（默认500）")
    ap.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD（默认样本最新共同交易日）")
    ap.add_argument("--report", action="store_true", help="写入 generated/factor_report_{date}.md")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="报告输出目录")
    ap.add_argument("--view", action="store_true", help="输出 multi_agent 兼容 view JSON")
    args = ap.parse_args()

    frs = FactorRotationSystem(out_dir=args.out_dir)
    res = frs.detect(date=args.date, limit=args.limit)
    meta = res["meta"]
    print(f"[信息] 目标日期 {res['date']} | 抽样 {meta['sample_target']} 只"
          f"（有效 {meta['parsed']}） | 状态 {meta['status']} | 耗时 {meta['elapsed_sec']}s",
          flush=True)
    print(f"[信息] 价值因子来源: {meta['value_source']} | 财务ROE覆盖 {meta.get('financial_roe_cov', 0)} 只")

    print("\n[因子收益] 截面多空组合(前20%多/后20%空)近20日收益 + IC:")
    for t in sorted(res["factors"], key=lambda r: -r["ls_return"]):
        flag = " ✓" if t["data_ok"] else " ⚠数据不足"
        print(f"  {t['name']:<4} 多空 {t['ls_return']:+.2%}  IC {t['ic']:+.3f}"
              f"  样本 {t['n_valid']}{flag}")

    print("\n[占优因子 TOP3]（因子动量）→ 风格映射:")
    for t in res["top_factors"]:
        m = STYLE_MAP.get(t["factor"], {})
        print(f"  #{t['rank']} {t['name']}({t['factor']}) 多空 {t['ls_return']:+.2%} "
              f"IC {t['ic']:+.3f} → 风格[{m.get('style', '观察')}]")

    crowd = res["crowding"]
    print(f"\n[拥挤度] 截面相关 {crowd.get('cross_corr', 0):.3f}"
          + (f" 换手代理 {crowd['turnover_proxy']:.3f}" if crowd.get("turnover_proxy") is not None else "")
          + (f" | {crowd.get('note')}" if crowd.get("note") else ""))

    comp = res["composite"]
    primary = res["style"].get("primary", {})
    print(f"\n[综合] 信号 {comp['signal']} | 置信 {res['confidence']:.2f} | {comp['note']}")
    if primary:
        print(f"[风格建议] 主风格 {primary.get('style')}（{primary.get('mode')}）— {primary.get('desc')}")

    rag = res.get("rag") or {}
    print(f"\n[RAG] {rag.get('query', '')} → {rag.get('basis', '检索不可用')}")
    for h in rag.get("hits", [])[:RAG_K]:
        print(f"  [{h.get('score')}] ({h.get('cat', '')}) {h.get('file')}")

    if args.report:
        path = frs.report(date=args.date, limit=args.limit, out_dir=args.out_dir)
        print(f"\n已保存: {path}")
    if args.view:
        v = frs.view(date=args.date, limit=args.limit)
        print(f"\n[VIEW] {json.dumps(v, ensure_ascii=False, indent=2, default=str)}")


if __name__ == "__main__":
    main()
