#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_analysis.py — 数据融合 + 深化分析 + AI 辅助决策引擎
=========================================================
把 data_warehouse 里分散的板块（K线/估值/财务/事件/资金流/涨停池/宏观）
融合成「个股全景档案」，并基于多维度规则打分生成 AI 辅助决策建议。

数据来源（data_warehouse/）:
  kline/<code>.parquet       日K: date open high low close volume amount turnover outstanding_share
  valuation/<code>.parquet   估值: date peTTM pbMRQ psTTM pcfNcfTTM
                             （无 total_mv/float_mv 列；流通市值由 kline 的
                             close × outstanding_share 推导，见 _derive_float_mv_yi）
  financial/<code>.parquet   财务: 中文列名季度报表
  events/<name>.parquet      事件: block大宗交易/龙虎榜等
  market/zt_pool/*.parquet   涨停池逐日
  market/zt_pool_strong/*.parquet  强势股池逐日
  macro/*.parquet            宏观: pmi cpi m2 shibor lpr gdp 等
  oneoff/*.parquet           一次性: a_all_pb a_ttm_lyr 等

AI 决策逻辑（规则引擎，非量化模型）:
  五个维度打分 → 加权合成 → 结论模板生成自然语言建议
  1. 趋势(30%): MA20/60/144/300 多空排列 + 动量
  2. 估值(20%): PE/PB 历史分位
  3. 财务质量(20%): ROE/毛利率/营收净利增速/负债率
  4. 资金/热度(20%): 成交额变化/换手/涨停池出现频率/大宗
  5. 事件/宏观(10%): 龙虎榜/宏观环境(PMI/流动性)
"""
from __future__ import annotations
import logging

import json
import math
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pypinyin import lazy_pinyin, Style
    _HAS_PINYIN = True
except Exception:  # noqa: BLE001
    _HAS_PINYIN = False

WAREHOUSE = Path(__file__).resolve().parent.parent / "data_warehouse"

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _read_parquet(rel: str) -> Optional[pd.DataFrame]:
    """按 data_warehouse 相对路径读 parquet，失败返回 None"""
    p = WAREHOUSE / rel
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def _norm_code(code: str) -> str:
    """规范化股票代码：600000 / 600000.SH / SH600000 → 600000"""
    s = str(code).strip().upper()
    s = re.sub(r"\.(SH|SZ|BJ)$", "", s)
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s.zfill(6) if s.isdigit() else s


def load_name_map() -> dict[str, str]:
    """加载 code → name 映射（反转 stock_name_map.json）"""
    m = Path(__file__).resolve().parent.parent / "quant_system" / "stock_name_map.json"
    if not m.exists():
        return {}
    try:
        raw = json.loads(m.read_text(encoding="utf-8"))
        return {v: k for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


_NAME_MAP_CACHE: dict[str, str] = {}


def stock_name(code: str) -> str:
    global _NAME_MAP_CACHE
    if not _NAME_MAP_CACHE:
        _NAME_MAP_CACHE = load_name_map()
    return _NAME_MAP_CACHE.get(_norm_code(code), "")


# ---------------------------------------------------------------------------
# 中文模糊搜索（拼音增强）
# ---------------------------------------------------------------------------

def fuzzy_search_stocks(query: str, limit: int = 20) -> list[dict]:
    """中文模糊搜索：支持中文子串 / 拼音全拼 / 拼音首字母 / 代码前缀

    返回 [{"name","code","match_type","score"}]
    """
    q = query.strip().lower()
    if not q:
        return []
    m = Path(__file__).resolve().parent.parent / "quant_system" / "stock_name_map.json"
    if not m.exists():
        return []
    try:
        name_map = json.loads(m.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    results: list[dict] = []
    q_digits = q if q.isdigit() else ""

    for name, code in name_map.items():
        score = 0
        mtype = ""
        # 1) 代码精确/前缀
        if q_digits:
            if code == q_digits:
                score = 100
                mtype = "code_exact"
            elif code.startswith(q_digits):
                score = 80
                mtype = "code_prefix"
        # 2) 中文精确/包含
        if q in name:
            cand = 95 if name == q else 70
            if cand > score:
                score, mtype = cand, "name_contains"
        # 3) 拼音匹配
        if _HAS_PINYIN and score < 60:
            py = lazy_pinyin(name, style=Style.NORMAL)
            full = "".join(py)
            initials = "".join(w[0] for w in py if w)
            if q == full:
                score, mtype = 90, "pinyin_full"
            elif q in full:
                score, mtype = 60, "pinyin_contains"
            elif q == initials:
                score, mtype = 65, "pinyin_initials"
            elif len(q) >= 2 and q in initials:
                score, mtype = 50, "pinyin_initials_part"
        if score > 0:
            results.append({"name": name, "code": code, "match_type": mtype, "score": score})

    results.sort(key=lambda x: (-x["score"], x["code"]))
    return results[:limit]


# ---------------------------------------------------------------------------
# 单股数据加载
# ---------------------------------------------------------------------------

def load_kline(code: str) -> Optional[pd.DataFrame]:
    df = _read_parquet(f"kline/{_norm_code(code)}.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_valuation(code: str) -> Optional[pd.DataFrame]:
    df = _read_parquet(f"valuation/{_norm_code(code)}.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # 兼容两种 schema：老格式(pe_ttm/pb/ps) 与 新格式(peTTM/pbMRQ/psTTM)
    # 新格式由服务器新抓取覆盖（000001 等 860 只），缺列名会导致 KeyError
    rename = {"peTTM": "pe_ttm", "pbMRQ": "pb", "psTTM": "ps", "pcfNcfTTM": "pcf"}
    df = df.rename(columns=rename)
    # 部分历史 parquet 曾写入重复字段名；取最新列，避免 Series 无法转成标量并阻断画像。
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df.sort_values("date").reset_index(drop=True)


def load_financial(code: str) -> Optional[pd.DataFrame]:
    df = _read_parquet(f"financial/{_norm_code(code)}.parquet")
    if df is None or df.empty:
        return None
    df = df.copy()
    # 2026-08-22 稳定化: schema 变更时按多种日期列名兼容, 无日期列则不清洗直接返回
    date_col = None
    for cand in ("日期", "date", "报告期", "REPORT_DATE", "report_date", "END_DATE", "ann_date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        try:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            # AkShare 财务表常带“常用指标”汇总行和空日期占位行；这些行
            # 不能被当作最新季度，否则会把画像日期变成 NaT 并丢失有效财务分数。
            df = df.assign(date=parsed).loc[parsed.notna()].copy()
            return df.sort_values("date").reset_index(drop=True)
        except Exception:  # noqa: BLE001
            pass  # 日期解析失败 → 保持原样
    return df.reset_index(drop=True)


def load_events(code: str) -> list[dict]:
    """个股相关事件：大宗交易/龙虎榜/涨停池出现记录"""
    evs: list[dict] = []
    code6 = _norm_code(code)
    # 大宗交易
    b = _read_parquet("events/block.parquet")
    if b is not None and "证券代码" in b.columns:
        sub = b[b["证券代码"].astype(str).str.zfill(6) == code6]
        for _, r in sub.tail(10).iterrows():
            evs.append({
                "type": "大宗交易",
                "date": str(r.get("交易日期", ""))[:10],
                "detail": f"{r.get('证券简称','')} 成交价{r.get('成交价','')} 成交额{r.get('成交额','')}",
            })
    # 涨停池出现
    zt_dir = WAREHOUSE / "market" / "zt_pool"
    if zt_dir.is_dir():
        hits = 0
        last_date = ""
        for f in sorted(zt_dir.glob("*.parquet"))[-60:]:
            try:
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                col = "代码" if "代码" in df.columns else ("证券代码" if "证券代码" in df.columns else None)
                if col and (df[col].astype(str).str.zfill(6) == code6).any():
                    hits += 1
                    last_date = f.stem
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).error(f"[stock_analysis] 操作失败: {e}", exc_info=True)
                continue
        if hits:
            evs.append({"type": "涨停池", "count_60d": hits, "last_date": last_date,
                        "detail": f"近60交易日出现{hits}次，最近{last_date}"})
    return evs


# ---------------------------------------------------------------------------
# 技术指标
# ---------------------------------------------------------------------------

def add_ma(df: pd.DataFrame, windows=(5, 10, 20, 60, 144, 300)) -> pd.DataFrame:
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    return df


def _finite_float(value) -> Optional[float]:
    """把可空/异常数值统一成有限 float；空值不得阻断整张画像。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_finite(values) -> bool:
    """判断一组字段是否至少有一个真实有限数值。"""
    return any(_finite_float(value) is not None for value in values)


def _rounded_float(value, digits: int = 2) -> Optional[float]:
    result = _finite_float(value)
    return round(result, digits) if result is not None else None


def _pct(x, y) -> Optional[float]:
    x_value, y_value = _finite_float(x), _finite_float(y)
    if x_value is None or y_value is None or y_value == 0:
        return None
    return round((x_value / y_value - 1) * 100, 2)


# ---------------------------------------------------------------------------
# 核心：个股全景档案
# ---------------------------------------------------------------------------

def _derive_float_mv_detail(kline: Optional[pd.DataFrame]) -> tuple[Optional[float], Optional[str], str]:
    """Derive float market value and expose the share snapshot date/status."""
    if kline is None or kline.empty:
        return None, None, "missing"
    if "close" not in kline.columns or "outstanding_share" not in kline.columns:
        return None, None, "schema_error"
    km = kline[kline["close"].notna() & kline["outstanding_share"].notna()].copy()
    if km.empty:
        return None, None, "missing_share_snapshot"
    close = _finite_float(km.iloc[-1]["close"])
    share = _finite_float(km.iloc[-1]["outstanding_share"])
    if close is None or share is None or share <= 0 or close <= 0:
        return None, None, "invalid_share_snapshot"
    raw_date = km.iloc[-1].get("date")
    as_of = pd.to_datetime(raw_date, errors="coerce")
    return round(close * share / 1e8, 1), as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else None, "available"


def _derive_float_mv_yi(kline: Optional[pd.DataFrame]) -> Optional[float]:
    """Compatibility wrapper for close × outstanding_share / 1e8."""
    return _derive_float_mv_detail(kline)[0]


def build_stock_profile(code: str) -> dict:
    """融合 K线+估值+财务+事件+资金 → 全景档案"""
    code6 = _norm_code(code)
    name = stock_name(code6)
    kline = load_kline(code6)
    val = load_valuation(code6)
    fin = load_financial(code6)
    events = load_events(code6)

    profile: dict = {
        "code": code6,
        "name": name or code6,
        "available": bool(kline is not None and not kline.empty),
        "data_status": "available" if kline is not None and not kline.empty else "missing",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_as_of": None,
        "source_dates": {},
        "quote": {}, "trend": {}, "valuation": {}, "financial": {},
        "liquidity": {}, "events": events, "commodity": {}, "signals": [], "score": {},
        "ai_advice": {},
    }
    # 即使行情缺失也保留稳定的来源字段，调用方可区分缺失与字段漂移。
    profile["source_dates"]["quote"] = None

    # ---- 行情 ----
    if kline is not None and not kline.empty:
        last = kline.iloc[-1]
        prev = kline.iloc[-2] if len(kline) > 1 else None
        quote_date = str(last["date"].date())
        profile["data_as_of"] = quote_date
        profile["source_dates"]["quote"] = quote_date
        profile["quote"] = {
            "date": quote_date,
            "close": round(float(last["close"]), 2),
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "volume": float(last.get("volume", 0) or 0),
            "amount": float(last.get("amount", 0) or 0),
            "turnover": round(float(last.get("turnover", 0) or 0) * 100, 2),
            "chg_pct": _pct(last["close"], prev["close"]) if prev is not None else None,
        }
        # 趋势
        k2 = add_ma(kline)
        last_row = k2.iloc[-1]
        ma20, ma60, ma144, ma300 = (last_row.get(f"ma{w}") for w in (20, 60, 144, 300))
        close = float(last["close"])
        profile["trend"] = {
            "ma20": round(float(ma20), 2) if ma20 and not math.isnan(ma20) else None,
            "ma60": round(float(ma60), 2) if ma60 and not math.isnan(ma60) else None,
            "ma144": round(float(ma144), 2) if ma144 and not math.isnan(ma144) else None,
            "ma300": round(float(ma300), 2) if ma300 and not math.isnan(ma300) else None,
            "above_ma20": bool(close > ma20) if ma20 and not math.isnan(ma20) else None,
            "above_ma60": bool(close > ma60) if ma60 and not math.isnan(ma60) else None,
            "above_ma144": bool(close > ma144) if ma144 and not math.isnan(ma144) else None,
            "above_ma300": bool(close > ma300) if ma300 and not math.isnan(ma300) else None,
            "mom20": _pct(close, ma20),
            "mom60": _pct(close, ma60),
            "high_52w": round(float(kline["high"].tail(250).max()), 2),
            "low_52w": round(float(kline["low"].tail(250).min()), 2),
            "pos_52w": round((close - kline["low"].tail(250).min()) /
                             max(kline["high"].tail(250).max() - kline["low"].tail(250).min(), 1e-9) * 100, 1),
        }
        # 流动性（近20日）
        v20 = kline["volume"].tail(20)
        a20 = kline["amount"].tail(20)
        profile["liquidity"] = {
            "vol_ma5": round(float(kline["volume"].tail(5).mean()), 0),
            "vol_ma20": round(float(v20.mean()), 0),
            "vol_ratio_5_20": round(float(v20.tail(5).mean() / max(v20.mean(), 1)), 2),
            "amount_ma5_yi": round(float(a20.tail(5).mean()) / 1e8, 2),
            "amount_ma20_yi": round(float(a20.mean()) / 1e8, 2),
        }

    # ---- 估值 ----
    if val is not None and not val.empty:
        v = val.dropna(subset=["pe_ttm", "pb"])
        if not v.empty:
            last_v = v.iloc[-1]
            pe, pb = _finite_float(last_v.get("pe_ttm")), _finite_float(last_v.get("pb"))
            # v 已按 pe_ttm/pb 非空筛选；防御性检查避免 schema 变更阻断画像。
            if pe is None or pb is None:
                logging.getLogger(__name__).warning("valuation row has invalid pe/pb for %s", code6)
                pe = pe if pe is not None else 0.0
                pb = pb if pb is not None else 0.0
            # 流通市值(亿元)：实测 valuation parquet 无 total_mv/float_mv 列
            # （仅 date/peTTM/pbMRQ/psTTM/pcfNcfTTM）。真实可得的市值来源取 kline 的
            # close × outstanding_share（与 emotion/ic 面板 value_picks 口径一致），
            # 取最近一行同时有 close 且 outstanding_share 非空者。
            float_mv_yi, float_mv_as_of, float_mv_status = _derive_float_mv_detail(kline)
            valuation_date = str(last_v["date"].date())
            profile["source_dates"]["valuation"] = valuation_date
            profile["valuation"] = {
                "date": valuation_date,
                "float_mv_as_of": float_mv_as_of,
                "float_mv_status": float_mv_status,
                "pe_ttm": _rounded_float(pe),
                "pe_lyr": _rounded_float(last_v.get("pe_lyr")),
                "pb": _rounded_float(pb),
                "ps": _rounded_float(last_v.get("ps")),
                # 兼容旧字段名（前端/下游若读 total_mv_yi 不再恒 0）
                "total_mv_yi": float_mv_yi,
                "float_mv_yi": float_mv_yi,
                "pe_percentile_3y": _percentile(v["pe_ttm"].dropna().tail(750), pe),
                "pb_percentile_3y": _percentile(v["pb"].dropna().tail(750), pb),
            }

    # ---- 财务 ----
    if fin is not None and not fin.empty:
        profile["financial"] = _financial_summary(fin)

    # ---- 行业商品/期货证据 ----
    try:
        from quant_web.industry_commodity import build_industry_commodity_evidence
        profile["commodity"] = build_industry_commodity_evidence(code6)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("commodity evidence unavailable for %s: %s", code6, exc)
        profile["commodity"] = {"status": "error", "error": str(exc)[:160], "items": []}

    # ---- 评分与决策 ----
    if not profile["available"]:
        profile["coverage"] = {"available_dimensions": 0, "total_dimensions": len(_WEIGHTS), "ratio": 0.0}
        profile["score"] = {}
        profile["ai_advice"] = {}
        profile["signals"] = []
        return profile
    profile["score"] = _score_profile(profile)
    profile["coverage"] = profile["score"].get("coverage", {})
    profile["ai_advice"] = _make_advice(profile)
    profile["signals"] = _collect_signals(profile)
    return profile


def _percentile(series: pd.Series, value: float) -> Optional[float]:
    if series is None or len(series) < 60 or math.isnan(value):
        return None
    return round(float((series <= value).mean() * 100), 1)


# ---------------------------------------------------------------------------
# 财务摘要
# ---------------------------------------------------------------------------

_FIN_MAP = {
    "roe": ["净资产收益率(%)", "净资产收益率", "加权净资产收益率(%)", "摊薄净资产收益率(%)"],
    "gross_margin": ["主营业务利润率(%)", "销售毛利率(%)", "毛利率(%)"],
    "rev_yoy": ["主营业务收入增长率(%)", "营业收入增长率(%)", "营业总收入增长率(%)"],
    "profit_yoy": ["净利润增长率(%)", "净利润同比(%)", "扣除非经常性损益后的净利润增长率(%)"],
    "debt_ratio": ["资产负债率(%)"],
    "eps": ["摊薄每股收益(元)", "每股收益(元)", "基本每股收益(元)"],
    "ocf_ps": ["每股经营性现金流(元)"],
}


def _pick_col(df: pd.DataFrame, keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _financial_summary(fin: pd.DataFrame) -> dict:
    out: dict = {}
    cols = {
        "roe": _pick_col(fin, _FIN_MAP["roe"]),
        "gross_margin": _pick_col(fin, _FIN_MAP["gross_margin"]),
        "rev_yoy": _pick_col(fin, _FIN_MAP["rev_yoy"]),
        "profit_yoy": _pick_col(fin, _FIN_MAP["profit_yoy"]),
        "debt_ratio": _pick_col(fin, _FIN_MAP["debt_ratio"]),
        "eps": _pick_col(fin, _FIN_MAP["eps"]),
        "ocf_ps": _pick_col(fin, _FIN_MAP["ocf_ps"]),
    }
    # 仅使用有有效报告期的最后一行；每个指标独立过滤 NaN，避免
    # 某一列缺失把整个财务维度渲染成“NaN”或误判为已覆盖。
    latest = fin.iloc[-1] if not fin.empty else None
    if latest is None:
        return out
    if "date" in latest.index:
        report_date = pd.to_datetime(latest.get("date"), errors="coerce")
        if pd.notna(report_date):
            out["report_date"] = report_date.strftime("%Y-%m-%d")
    available_fields = []
    missing_fields = []
    for key, col in cols.items():
        if col and col in latest.index:
            fv = _finite_float(latest.get(col))
            if fv is not None:
                out[key] = round(fv, 2)
                available_fields.append(key)
            else:
                missing_fields.append(key)
        else:
            missing_fields.append(key)
    out["available_fields"] = available_fields
    out["missing_fields"] = missing_fields
    out["coverage"] = round(len(available_fields) / len(cols), 3) if cols else 0.0
    out["sample_size"] = int(len(fin))
    # 同比趋势：最近两期营收/净利增速
    if cols["rev_yoy"] and len(fin) >= 2:
        value = _finite_float(fin.iloc[-2].get(cols["rev_yoy"]))
        if value is not None:
            out["rev_yoy_prev"] = round(value, 2)
    if cols["profit_yoy"] and len(fin) >= 2:
        value = _finite_float(fin.iloc[-2].get(cols["profit_yoy"]))
        if value is not None:
            out["profit_yoy_prev"] = round(value, 2)
    return out


# ---------------------------------------------------------------------------
# 评分引擎（6 维加权：商品维度只在有真实行业暴露数据时生效）
# ---------------------------------------------------------------------------

# 牧原等周期公司必须把产品和饲料成本放进相对重要的业务维度；
# 具体品种仍按可用覆盖率折减，缺失期货不会被补成中性。
_WEIGHTS = {"trend": 0.25, "valuation": 0.16, "financial": 0.20, "liquidity": 0.14, "events": 0.07, "commodity": 0.18}


def _runtime_weights() -> tuple[dict[str, float], dict]:
    """Return the business prior plus a bounded IC/OOS calibration overlay."""
    try:
        from quant_web.ic_evidence import calibrated_dimension_weights
        weights, evidence = calibrated_dimension_weights(_WEIGHTS)
        return {key: float(weights.get(key, value)) for key, value in _WEIGHTS.items()}, evidence
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("IC calibration unavailable: %s", exc)
        return dict(_WEIGHTS), {"status": "error", "fallback": str(exc)[:160], "dimension_weights": dict(_WEIGHTS)}


def _dimension_valid(key: str, value) -> bool:
    if key == "trend":
        return isinstance(value, dict) and any(value.get(k) is not None for k in ("above_ma20", "above_ma60", "mom20", "pos_52w"))
    if key == "valuation":
        return isinstance(value, dict) and any(value.get(k) is not None for k in ("pe_percentile_3y", "pb_percentile_3y", "pe_ttm", "pb"))
    if key == "financial":
        return isinstance(value, dict) and any(value.get(k) is not None for k in ("roe", "profit_yoy", "rev_yoy", "debt_ratio", "ocf_ps"))
    if key == "liquidity":
        return isinstance(value, dict) and any(value.get(k) is not None for k in ("vol_ratio_5_20", "amount_ma5_yi", "amount_ma20_yi"))
    if key == "events":
        return isinstance(value, list) and bool(value)
    if key == "commodity":
        # partial means at least one fresh contract is available. Missing/stale
        # contracts reduce the effective weight and remain visible in the audit.
        return (isinstance(value, dict)
                and value.get("status") in {"available", "partial"}
                and float(value.get("available_weight") or 0) > 0)
    return False


def _score_profile(p: dict) -> dict:
    """各维度 0-100 分 + 加权总分"""
    s: dict = {}
    # 1) 趋势
    t = p.get("trend", {})
    trend = 50.0
    if t.get("above_ma20") is not None:
        trend = sum([
            20 if t.get("above_ma20") else 10,
            20 if t.get("above_ma60") else 10,
            15 if t.get("above_ma144") else 8,
            10 if t.get("above_ma300") else 5,
        ])
        mom = t.get("mom20") or 0
        trend += max(-10, min(15, mom / 2))
        pos = t.get("pos_52w")
        if pos is not None:
            trend += 10 if pos >= 75 else (5 if pos >= 50 else (-5 if pos >= 25 else -10))
    if _dimension_valid("trend", t):
        s["trend"] = round(max(0, min(100, trend)), 1)

    # 2) 估值
    v = p.get("valuation", {})
    val_score = 50.0
    if v.get("pe_percentile_3y") is not None:
        pct = v["pe_percentile_3y"]
        val_score = 80 if pct <= 20 else (65 if pct <= 40 else (50 if pct <= 60 else (35 if pct <= 80 else 20)))
        if v.get("pe_ttm") and v["pe_ttm"] < 0:
            val_score = 30  # 亏损股估值打折
    if _dimension_valid("valuation", v):
        s["valuation"] = round(val_score, 1)

    # 3) 财务
    f = p.get("financial", {})
    fin_score = 50.0
    if f.get("roe") is not None:
        roe = f["roe"]
        fin_score = 80 if roe >= 15 else (65 if roe >= 10 else (50 if roe >= 5 else (35 if roe > 0 else 15)))
    if f.get("profit_yoy") is not None:
        py = f["profit_yoy"]
        fin_score += 10 if py >= 30 else (5 if py >= 10 else (-5 if py < 0 else 0))
    if f.get("debt_ratio") is not None:
        dr = f["debt_ratio"]
        fin_score += 5 if dr <= 40 else (0 if dr <= 60 else -10)
    if f.get("ocf_ps") is not None and f.get("eps"):
        if f["ocf_ps"] > 0 and f["ocf_ps"] >= f["eps"] * 0.8:
            fin_score += 5
    if _dimension_valid("financial", f):
        s["financial"] = round(max(0, min(100, fin_score)), 1)

    # 4) 流动性/资金
    liq = p.get("liquidity", {})
    liq_score = 50.0
    if liq.get("vol_ratio_5_20") is not None:
        vr = liq["vol_ratio_5_20"]
        liq_score = 70 if vr >= 1.5 else (60 if vr >= 1.2 else (50 if vr >= 0.8 else 35))
    if liq.get("amount_ma5_yi") is not None and liq["amount_ma5_yi"] >= 2:
        liq_score += 10  # 大额成交更活跃
    if _dimension_valid("liquidity", liq):
        s["liquidity"] = round(max(0, min(100, liq_score)), 1)

    # 5) 事件
    evs = p.get("events", [])
    ev_score = 50.0
    for e in evs:
        if e.get("type") == "涨停池":
            c = e.get("count_60d", 0)
            ev_score = 80 if c >= 5 else (65 if c >= 2 else 55)
        elif e.get("type") == "大宗交易":
            ev_score = min(90, ev_score + 5)
    if _dimension_valid("events", evs):
        s["events"] = round(ev_score, 1)

    # 6) 行业商品/期货：商品篮子已在 industry_commodity 中按业务方向计算。
    commodity = p.get("commodity", {})
    if _dimension_valid("commodity", commodity) and commodity.get("score") is not None:
        s["commodity"] = round(float(commodity["score"]), 1)

    # 商品篮子覆盖率影响其名义权重：只观测到生猪时，不把缺失的饲料
    # 暴露重新包装成完整商品分数；剩余有效维度再按实际权重归一。
    dimension_keys = [key for key in _WEIGHTS if key != "commodity" or "commodity" in p]
    base_weights, ic_evidence = _runtime_weights()
    calibrated_weight_total = sum(base_weights.get(key, 0.0) for key in dimension_keys)
    if "commodity" in s:
        commodity_coverage = max(0.0, min(1.0, float(commodity.get("coverage") or 0.0)))
        base_weights["commodity"] = base_weights["commodity"] * commodity_coverage
    valid_dimensions = [
        key for key in dimension_keys
        if key in s and _dimension_valid(key, p.get(key, evs if key == "events" else {}))
        and base_weights.get(key, 0) > 0
    ]
    available_weights = {key: base_weights[key] for key in valid_dimensions}
    weight_sum = sum(available_weights.values())
    total = sum(s[key] * weight for key, weight in available_weights.items()) / weight_sum if weight_sum else None
    nominal_sum = calibrated_weight_total
    coverage_ratio = weight_sum / nominal_sum if nominal_sum else 0.0
    # total 是“已知维度重归一化分”，coverage_adjusted_score 是保守下界；
    # 后者用于提醒数据不完整时不要把高分当成完整画像结论。
    adjusted = total * coverage_ratio if total is not None else None
    s["total"] = round(total, 1) if total is not None else None
    s["raw_score"] = s["total"]
    s["coverage_adjusted_score"] = round(adjusted, 1) if adjusted is not None else None
    s["score_semantics"] = "raw_score=有效维度重归一化；coverage_adjusted_score=按名义权重覆盖率折减的保守下界，缺失不视为中性"
    s["weights"] = {key: round(weight / weight_sum, 4) for key, weight in available_weights.items()} if weight_sum else {}
    s["nominal_weights"] = {key: round(weight, 4) for key, weight in available_weights.items()}
    s["business_prior_weights"] = {key: round(weight, 4) for key, weight in _WEIGHTS.items()}
    s["ic_calibrated_weights"] = {key: round(float(base_weights.get(key, 0)), 4) for key in dimension_keys}
    s["ic_evidence_status"] = ic_evidence.get("status")
    s["unavailable_dimensions"] = [key for key in dimension_keys if key not in valid_dimensions]
    s["dimension_contributions"] = {
        key: round(s[key] * weight / weight_sum, 2) for key, weight in available_weights.items()
    } if weight_sum else {}
    s["dimension_status"] = {
        key: ("available" if key in valid_dimensions else "missing_or_invalid") for key in dimension_keys
    }
    s["coverage"] = {"available_dimensions": len(available_weights), "total_dimensions": len(dimension_keys),
                      "ratio": round(len(available_weights) / len(dimension_keys), 2) if dimension_keys else 0.0,
                      "nominal_weight_covered": round(weight_sum, 4),
                      "calibrated_weight_total": round(nominal_sum, 4),
                      "coverage_ratio": round(coverage_ratio, 4)}
    s["level"] = ("强" if total >= 70 else ("偏强" if total >= 60 else ("中性" if total >= 45 else ("偏弱" if total >= 35 else "弱")))) if total is not None else "不可用"
    return s


# ---------------------------------------------------------------------------
# AI 辅助决策（自然语言建议）
# ---------------------------------------------------------------------------

def _make_advice(p: dict) -> dict:
    s = p.get("score", {})
    total = s.get("total")
    if total is None:
        return {
            "stance": "不可用", "color": "gray", "total_score": None,
            "summary": f"{p.get('name', p.get('code', ''))} 数据覆盖不足，未形成评分结论。",
            "reasons": [], "risks": ["关键评分维度缺失，禁止将缺失视为中性"],
            "disclaimer": "数据不足，不构成投资建议",
        }
    t = p.get("trend", {})
    v = p.get("valuation", {})
    f = p.get("financial", {})
    liq = p.get("liquidity", {})
    name = p.get("name", p.get("code", ""))

    # 结论
    if total >= 70:
        stance, color = "看多", "red"
    elif total >= 60:
        stance, color = "偏多", "orange"
    elif total >= 45:
        stance, color = "中性观望", "gray"
    elif total >= 35:
        stance, color = "偏空", "green"
    else:
        stance, color = "看空", "darkgreen"

    reasons: list[str] = []
    # 趋势
    if t.get("above_ma20") is True and t.get("above_ma60") is True:
        reasons.append(f"股价站上MA20/MA60，短线趋势偏强（现价{t.get('ma20')}上方）")
    elif t.get("above_ma20") is False:
        reasons.append(f"股价跌破MA20（{t.get('ma20')}），短线动能转弱")
    if t.get("above_ma144") is True and t.get("above_ma300") is True:
        reasons.append("站上MA144/MA300 长期均线，中期趋势向上")
    elif t.get("above_ma144") is False and t.get("ma144") is not None:
        reasons.append(f"处于MA144（{t.get('ma144')}）下方，中期趋势偏弱")
    if t.get("pos_52w") is not None:
        if t["pos_52w"] >= 75:
            reasons.append(f"股价处于52周高位区间（{t['pos_52w']}%分位），强势但需防回调")
        elif t["pos_52w"] <= 25:
            reasons.append(f"股价处于52周低位区间（{t['pos_52w']}%分位），关注反转机会")

    # 估值
    if v.get("pe_percentile_3y") is not None:
        pct = v["pe_percentile_3y"]
        if pct <= 20:
            reasons.append(f"PE_TTM {v.get('pe_ttm')} 处于近3年 {pct}% 分位，估值偏低")
        elif pct >= 80:
            reasons.append(f"PE_TTM {v.get('pe_ttm')} 处于近3年 {pct}% 分位，估值偏高")
        else:
            reasons.append(f"PE_TTM {v.get('pe_ttm')} 处于近3年 {pct}% 分位，估值中性")
    elif v.get("pe_ttm") is not None:
        reasons.append(f"PE_TTM {v.get('pe_ttm')}（估值数据较短）")

    # 财务
    if f.get("roe") is not None:
        reasons.append(f"最新ROE {f['roe']}%")
    if f.get("profit_yoy") is not None:
        yoy = f["profit_yoy"]
        reasons.append(f"净利润同比 {yoy}%" + ("，盈利加速" if yoy >= 20 else ("，盈利下滑" if yoy < 0 else "")))
    if f.get("rev_yoy") is not None:
        reasons.append(f"营收同比 {f['rev_yoy']}%")
    if f.get("debt_ratio") is not None:
        reasons.append(f"资产负债率 {f['debt_ratio']}%")

    # 商品价格/原料成本
    commodity = p.get("commodity", {})
    for item in commodity.get("items", []):
        if item.get("status") == "available" and item.get("return_20d_pct") is not None:
            direction = "产品价格正向" if float(item.get("direction", 0)) > 0 else "原料成本负向"
            reasons.append(f"{item.get('name')}近20日{float(item['return_20d_pct']):+.2f}%（{direction}，权重{item.get('weight')}）")

    # 资金
    if liq.get("vol_ratio_5_20") is not None:
        vr = liq["vol_ratio_5_20"]
        reasons.append("近5日成交量为20日均量" + ("1.2倍以上，资金活跃" if vr >= 1.2 else ("，量能萎缩" if vr < 0.8 else "，量能平稳")))
    if liq.get("amount_ma5_yi") is not None:
        reasons.append(f"近5日日均成交额 {liq['amount_ma5_yi']} 亿元")

    # 事件
    for e in p.get("events", []):
        reasons.append(e.get("detail", ""))

    # 风险提示
    risks: list[str] = []
    if t.get("pos_52w") is not None and t["pos_52w"] >= 90:
        risks.append("股价接近52周新高，追高风险大")
    if v.get("pe_percentile_3y") is not None and v["pe_percentile_3y"] >= 90:
        risks.append("估值处于历史极高位")
    if f.get("profit_yoy") is not None and f["profit_yoy"] < -30:
        risks.append("净利润大幅下滑")
    if f.get("debt_ratio") is not None and f["debt_ratio"] > 70:
        risks.append("负债率偏高")
    if not risks:
        risks.append("综合风险中等，注意大盘系统性波动")

    return {
        "stance": stance,
        "color": color,
        "total_score": total,
        "summary": f"{name}（{p.get('code')}）综合评分 {total}/100，结论：{stance}。",
        "reasons": reasons[:10],
        "risks": risks[:5],
        "commodity_score": commodity.get("score"),
        "commodity_coverage": commodity.get("coverage"),
        "commodity_methodology": commodity.get("methodology"),
        "coverage_note": f"有效维度 {s.get('coverage', {}).get('available_dimensions', 0)}/{s.get('coverage', {}).get('total_dimensions', 0)}；缺失维度不计分，实际权重按可用证据重归一。",
        "disclaimer": "AI 辅助分析基于历史数据规则引擎，不构成投资建议",
    }


def _collect_signals(p: dict) -> list[dict]:
    """关键信号列表（看多/看空/中性）"""
    sig: list[dict] = []
    t, v, f, liq = p.get("trend", {}), p.get("valuation", {}), p.get("financial", {}), p.get("liquidity", {})

    def add(kind, text, weight):
        sig.append({"type": kind, "text": text, "weight": weight})

    if t.get("above_ma20") and t.get("above_ma60"):
        add("bull", "MA20/MA60 多头排列", 2)
    if t.get("above_ma144") and t.get("above_ma300"):
        add("bull", "长期均线多头（MA144/300上方）", 2)
    if t.get("above_ma20") is False:
        add("bear", "跌破MA20", 1)
    if v.get("pe_percentile_3y") is not None and v["pe_percentile_3y"] <= 20:
        add("bull", f"PE处于近3年低位（{v['pe_percentile_3y']}%分位）", 1)
    if v.get("pe_percentile_3y") is not None and v["pe_percentile_3y"] >= 80:
        add("bear", f"PE处于近3年高位（{v['pe_percentile_3y']}%分位）", 1)
    if f.get("roe") is not None and f["roe"] >= 15:
        add("bull", f"高ROE {f['roe']}%", 1)
    if f.get("profit_yoy") is not None and f["profit_yoy"] >= 30:
        add("bull", f"净利高增 {f['profit_yoy']}%", 1)
    if f.get("profit_yoy") is not None and f["profit_yoy"] < 0:
        add("bear", f"净利下滑 {f['profit_yoy']}%", 1)
    if liq.get("vol_ratio_5_20") is not None and liq["vol_ratio_5_20"] >= 1.5:
        add("bull", "放量（量比≥1.5）", 1)
    for e in p.get("events", []):
        if e.get("type") == "涨停池" and e.get("count_60d", 0) >= 3:
            add("bull", f"近期多次涨停（{e['count_60d']}次/60日）", 1)
    if not sig:
        add("neutral", "无明显多空信号", 0)
    return sig


# ---------------------------------------------------------------------------
# 宏观环境摘要（供前端联动）
# ---------------------------------------------------------------------------

def _find_m2_quantity_col(m2: pd.DataFrame) -> Optional[str]:
    """在货币供应 parquet 中定位 M2 总量(数量)列。

    正确列：名称含 "M2" 且含 "数量"(或以亿元为单位、非同比/环比增速)，
    且不含 "M0"。避免取到 M0-环比增长/同比等增速列。
    """
    if m2 is None or m2.empty:
        return None
    for c in m2.columns:
        cs = str(c)
        if "M2" in cs and "M0" not in cs and ("数量" in cs or "亿元" in cs):
            return c
    # 再宽松一档：含 M2 但非同比/环比增速
    for c in m2.columns:
        cs = str(c)
        if "M2" in cs and "同比" not in cs and "环比" not in cs and "M0" not in cs:
            return c
    return None


def macro_snapshot() -> dict:
    """宏观快照：PMI/CPI/M2/SHIBOR/LPR 最新值 + 趋势
    注意：akshare 宏观接口常倒序（最新在头部），统一取最新日期行。"""
    out: dict = {}
    try:
        pmi = _read_parquet("macro/pmi.parquet")
        if pmi is not None and not pmi.empty:
            # 找含“制造业”的指数列
            col = None
            for c in pmi.columns:
                if "制造业" in str(c) and ("指数" in str(c) or "PMI" in str(c).upper()):
                    col = c
                    break
            if col is None and len(pmi.columns) > 1:
                col = pmi.columns[1]
            if col is not None:
                last = pmi.iloc[0] if str(pmi.iloc[0].iloc[0]).startswith("202") else pmi.iloc[-1]
                val = float(last[col])
                out["pmi"] = {"date": str(last.iloc[0])[:10], "value": round(val, 1),
                              "expansion": val >= 50}
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[stock_analysis] 操作失败: {e}", exc_info=True)
    try:
        m2 = _read_parquet("macro/m2_yearly.parquet")
        if m2 is None:
            m2 = _read_parquet("macro/money_supply.parquet")
        if m2 is not None and not m2.empty:
            # 实测 m2_yearly.parquet 为倒序（首行=最新），列为：
            #   [月份, 货币和准货币(M2)-数量(亿元), M2-同比, M2-环比, M1-数量,.., 流通中的现金(M0)-数量, M0-同比, M0-环比]
            # 老代码取 last.iloc[-1]（M0-环比增长，如 0.347）冒充 M2 总量——错误。
            # 正确取 "货币和准货币(M2)-数量(亿元)" 列。
            m2 = m2.copy()
            date_col = "月份" if "月份" in m2.columns else m2.columns[0]
            qty_col = _find_m2_quantity_col(m2)
            if qty_col is None:
                qty_col = "货币和准货币(M2)-数量(亿元)" if "货币和准货币(M2)-数量(亿元)" in m2.columns else None
            if qty_col is None:
                # last resort: 跳过首列(月份)后选含“M2”且非环比/同比的列
                cand = [c for c in m2.columns if c != date_col and "M2" in str(c)
                        and "同比" not in str(c) and "环比" not in str(c)]
                qty_col = cand[0] if cand else m2.columns[1]
            last = m2.sort_values(date_col).iloc[-1]
            raw_date = str(last[date_col])
            m = re.search(r"(\d{4})\s*[年\-/]\s*(\d{1,2})\s*[月\-/]?", raw_date)
            if m:
                date_str = f"{m.group(1)}-{int(m.group(2)):02d}"
            else:
                date_str = str(raw_date)[:10]
            out["m2"] = {"date": date_str, "value": round(float(last[qty_col]), 2),
                         "unit": "亿元", "column": qty_col,
                         "note": "货币和准货币(M2)-数量(亿元)"}
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[stock_analysis] 操作失败: {e}", exc_info=True)
    try:
        shibor = _read_parquet("macro/shibor_all.parquet")
        if shibor is not None and not shibor.empty:
            col = "O/N-定价" if "O/N-定价" in shibor.columns else shibor.columns[1]
            last = shibor.iloc[-1]
            out["shibor"] = {"date": str(last.iloc[0])[:10], "value": round(float(last[col]), 3)}
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[stock_analysis] 操作失败: {e}", exc_info=True)
    try:
        lpr = _read_parquet("macro/lpr.parquet")
        if lpr is not None and not lpr.empty:
            col = "LPR1Y" if "LPR1Y" in lpr.columns else lpr.columns[1]
            last = lpr.iloc[-1]
            out["lpr"] = {"date": str(last.iloc[0])[:10], "value": round(float(last[col]), 2)}
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error(f"[stock_analysis] 操作失败: {e}", exc_info=True)
    return out


# ---------------------------------------------------------------------------
# 数据仓库覆盖度报告
# ---------------------------------------------------------------------------

def warehouse_status() -> dict:
    """数据仓库各板块覆盖度/新鲜度"""
    out: dict = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "boards": {}}
    for board in ["kline", "valuation", "financial", "macro", "market", "oneoff", "events"]:
        d = WAREHOUSE / board
        if not d.is_dir():
            continue
        files = list(d.rglob("*.parquet")) + list(d.rglob("*.csv"))
        n = len(files)
        size_mb = round(sum(f.stat().st_size for f in files) / 1e6, 1)
        newest = ""
        if n:
            newest = max(f.stat().st_mtime for f in files)
            newest = datetime.fromtimestamp(newest).strftime("%Y-%m-%d")
        out["boards"][board] = {"files": n, "size_mb": size_mb, "newest": newest}
    out["kline_stocks"] = len(list((WAREHOUSE / "kline").glob("*.parquet"))) if (WAREHOUSE / "kline").is_dir() else 0
    return out


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "002714"
    prof = build_stock_profile(code)
    print(json.dumps(prof, ensure_ascii=False, indent=2)[:3000])
