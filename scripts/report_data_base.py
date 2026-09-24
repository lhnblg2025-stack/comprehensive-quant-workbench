#!/usr/bin/env python3
"""Shared report data base for market/macro cron reports.

This module gives cron reports a common, persistent data layer instead of
letting every prompt rediscover sources from scratch. It deliberately degrades
gracefully: each source is captured as a SourceResult-like record, failures are
saved, and the latest usable cache remains available for reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from quant_system.utils import safe_float as _safe_float_impl
except ImportError:
    _safe_float_impl = None
BASE_DIR = ROOT / "generated" / "report_data_base"
DAILY_DIR = BASE_DIR / date.today().strftime("%Y%m%d")
LATEST_JSON = BASE_DIR / "latest.json"
LATEST_MD = BASE_DIR / "latest_summary.md"
PROBE_LATEST_JSON = ROOT / "generated" / "data_source_probe" / "latest.json"
LATEST_SINGLE_ITEM_MAX_AGE_SECONDS = 18 * 3600
FALLBACK_SUCCESS_MAX_AGE_SECONDS = 14 * 24 * 3600
BASE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)


CORE_SOURCE_NAMES = {
    "cpi_ppi",
    "social_finance_m2",
    "official_source_registry",
    "fed_data_sources",
    "a_share_index_daily_sh000001",
    "a_share_market_breadth",
    "industry_board_snapshot",
    "dividend_weekly_inputs",
}

# Supplemental maps may help a report find backup endpoints, but they are not
# promoted into the core evidence set unless the data-source probe accepts them.
SUPPORTING_CONTEXT_SOURCE_NAMES = {
    "cross_asset_fallback_sources",
}

DISCLOSURE_FALLBACK_SOURCE_NAMES = {
    "pmi",
    "customs",
    "international_oil_sources",
}

PENDING_LIVE_PARSER_NOTES = {
    "pmi": "archive-only desktop fallback; add live NBS PMI parser before core promotion",
    "customs": "archive-only desktop fallback; add live GACC/customs parser before core promotion",
    "international_oil_sources": "source registry includes official endpoints but probe still detected archive fallback; parse OPEC/EIA live tables before core promotion",
}


@dataclass(frozen=True)
class DataStatus:
    name: str
    ok: bool
    source: str
    note: str = ""
    latest_date: str | None = None
    cache_path: str | None = None
    error: str | None = None


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def _safe_float(value: Any) -> float | None:
    """D1 收敛: 复用 quant_system.utils.safe_float（import 失败时保留原实现）。

    utils 参数映射: default=None（None/空/失败→None）、clean_percent/clean_commas=False
    （原实现不清洗）、finite=True（nan/inf→None）、allow_bool=True（bool 按数值，
    对齐 float(True)=1.0）；原实现 round(out, 6) 在转发层保留。
    """
    if _safe_float_impl is not None:
        out = _safe_float_impl(
            value,
            default=None,
            clean_percent=False,
            clean_commas=False,
            finite=True,
            allow_bool=True,
        )
        return round(out, 6) if out is not None else None
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return round(out, 6)
    except Exception:
        return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.\-]+", "_", name)[:120].strip("_") or "data"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_seconds(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _cache_path(bucket: str, key: str) -> Path:
    return DAILY_DIR / bucket / f"{_safe_name(key)}.json"


def _payload_ok(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    return bool(status.get("ok"))


def _latest_payload(bucket: str, key: str) -> dict[str, Any] | None:
    latest = _read_json(LATEST_JSON) or {}
    try:
        previous = latest["data"][bucket][key]
    except Exception:
        return None
    return previous if _payload_ok(previous) else None


def _is_cache_fresh(bucket: str, key: str, max_age_seconds: float = 18 * 3600) -> bool:
    """Check if a daily cache file for (bucket, key) exists and is within TTL."""
    path = _cache_path(bucket, key)
    age = _age_seconds(path)
    if age is None:
        return False
    return age < max_age_seconds


def _recent_success_payload(bucket: str, key: str, max_age_seconds: int = FALLBACK_SUCCESS_MAX_AGE_SECONDS) -> dict[str, Any] | None:
    safe = _safe_name(key)
    candidates = sorted(BASE_DIR.glob(f"20*/{bucket}/{safe}.json"), reverse=True)
    for path in candidates:
        age = _age_seconds(path)
        if age is None or age > max_age_seconds:
            continue
        payload = _read_json(path)
        if _payload_ok(payload):
            payload.setdefault("status", {})["note"] = str(payload.get("status", {}).get("note") or "") + f"; loaded from recent successful cache {path.parent.parent.name}"
            return payload
    return None


def _fresh_payload(bucket: str, key: str, max_age_seconds: int) -> dict[str, Any] | None:
    today_path = _cache_path(bucket, key)
    age = _age_seconds(today_path)
    if age is not None and age <= max_age_seconds:
        payload = _read_json(today_path)
        if _payload_ok(payload):
            payload.setdefault("status", {})["note"] = str(payload.get("status", {}).get("note") or "") + "; loaded from report_data_base cache"
            return payload
    latest_age = _age_seconds(LATEST_JSON)
    latest_max_age_seconds = min(max_age_seconds, LATEST_SINGLE_ITEM_MAX_AGE_SECONDS)
    if latest_age is not None and latest_age <= latest_max_age_seconds:
        previous = _latest_payload(bucket, key)
        if previous:
            previous.setdefault("status", {})["note"] = str(previous.get("status", {}).get("note") or "") + "; loaded from latest successful report_data_base snapshot"
            return previous
    return _recent_success_payload(bucket, key)


def _fresh_aggregate(bucket: str, key: str, max_age_seconds: int) -> dict[str, Any] | None:
    def usable(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if bucket == "macro":
            admission = payload.get("_source_admission") if isinstance(payload.get("_source_admission"), dict) else {}
            audit = admission.get("admission_audit") if isinstance(admission.get("admission_audit"), dict) else {}
            return "_source_admission" in payload and "_disclosure_fallbacks" in payload and "_supporting_context" in payload and "clean" in audit
        if bucket == "market":
            return any(isinstance(item, dict) and item.get("admission") for item in payload.values())
        return True

    today_path = _cache_path(bucket, key)
    age = _age_seconds(today_path)
    if age is not None and age <= max_age_seconds:
        payload = _read_json(today_path)
        if usable(payload):
            for item in payload.values():
                if isinstance(item, dict) and isinstance(item.get("status"), dict):
                    item["status"]["note"] = str(item["status"].get("note") or "") + "; loaded from report_data_base cache"
            return payload
    latest_age = _age_seconds(LATEST_JSON)
    if latest_age is None or latest_age > max_age_seconds:
        return None
    latest = _read_json(LATEST_JSON)
    try:
        previous = latest["data"][bucket]
    except Exception:
        return None
    if usable(previous):
        for item in previous.values():
            if isinstance(item, dict) and isinstance(item.get("status"), dict):
                item["status"]["note"] = str(item["status"].get("note") or "") + "; loaded from latest report_data_base snapshot"
        return previous
    return None


def _load_source_admission() -> dict[str, Any]:
    probe = _read_json(PROBE_LATEST_JSON) or {}
    results = probe.get("results") if isinstance(probe.get("results"), list) else []
    by_name = {str(item.get("name")): item for item in results if isinstance(item, dict) and item.get("name")}

    probe_available = bool(results)
    probe_recommended = {name for name, item in by_name.items() if bool(item.get("recommended_for_core"))}
    protected_allowlist = set(CORE_SOURCE_NAMES)
    promoted_core_names = protected_allowlist & probe_recommended if probe_available else protected_allowlist
    allowlist_not_recommended = protected_allowlist - promoted_core_names
    recommended_not_allowlisted = probe_recommended - protected_allowlist

    def entry(name: str, usage: str) -> dict[str, Any]:
        item = by_name.get(name, {})
        probe_flag = item.get("recommended_for_core") if item else None
        recommended = bool(probe_flag) if item else usage == "core"
        pending_note = PENDING_LIVE_PARSER_NOTES.get(name)
        reason = item.get("recommendation_reason") or pending_note
        if usage == "blocked_core_candidate" and not reason:
            reason = "blocked: source is on protected core allowlist, but latest probe did not recommend it for core promotion"
        if usage == "supporting_context" and not reason:
            reason = "supporting context only; use for endpoint discovery or fallback disclosure, not as core evidence"
        return {
            "name": name,
            "ok": bool(item.get("ok", usage in {"core", "disclosure_fallback", "supporting_context"})),
            "recommended_for_core": recommended,
            "probe_recommended_for_core": probe_flag,
            "allowed_core_name": name in CORE_SOURCE_NAMES,
            "promoted_to_core": usage == "core",
            "usage": usage,
            "row_count": item.get("row_count"),
            "latest_date": item.get("latest_date"),
            "source": item.get("source"),
            "reason": reason,
            "pending_live_parser": pending_note,
        }

    core = {name: entry(name, "core") for name in sorted(promoted_core_names)}
    blocked = {name: entry(name, "blocked_core_candidate") for name in sorted(allowlist_not_recommended)}
    fallback = {name: entry(name, "disclosure_fallback") for name in sorted(DISCLOSURE_FALLBACK_SOURCE_NAMES)}
    supporting = {name: entry(name, "supporting_context") for name in sorted(SUPPORTING_CONTEXT_SOURCE_NAMES)}
    return {
        "probe_path": str(PROBE_LATEST_JSON),
        "probe_generated_at": probe.get("generated_at"),
        "probe_summary": probe.get("summary", {}),
        "probe_available": probe_available,
        "recommended_source_names": sorted(probe_recommended),
        "protected_core_allowlist": sorted(protected_allowlist),
        "admission_audit": {
            "clean": not allowlist_not_recommended and not recommended_not_allowlisted,
            "promoted_core_count": len(core),
            "allowlist_count": len(protected_allowlist),
            "probe_recommended_count": len(probe_recommended),
            "allowlist_not_recommended": sorted(allowlist_not_recommended),
            "recommended_not_allowlisted": sorted(recommended_not_allowlisted),
        },
        "core_sources": core,
        "blocked_core_candidates": blocked,
        "disclosure_fallback_sources": fallback,
        "supporting_context_sources": supporting,
        "pending_live_parsers": PENDING_LIVE_PARSER_NOTES,
        "rule": "Only core_sources are promoted into core report logic. Disclosure fallbacks may be cited with stale/archive caveats until live parsers are added.",
    }


def _annotate_admission(name: str, item: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return item
    meta = None
    for group in ("core_sources", "blocked_core_candidates", "disclosure_fallback_sources", "supporting_context_sources"):
        sources = admission.get(group, {}) if isinstance(admission, dict) else {}
        if isinstance(sources, dict) and name in sources:
            meta = sources[name]
            break
    if meta:
        item["admission"] = meta
    return item


def _normalize_columns(df: Any) -> Any:
    rename = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "开盘": "open",
        "开盘价": "open",
        "最高": "high",
        "最高价": "high",
        "最低": "low",
        "最低价": "low",
        "收盘": "close",
        "收盘价": "close",
        "最新价": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    if "date" not in df.columns:
        raise ValueError(f"missing date column: {list(df.columns)}")
    if "close" not in df.columns:
        raise ValueError(f"missing close column: {list(df.columns)}")
    import pandas as pd

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    for col in ["open", "high", "low", "volume", "amount", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df.drop_duplicates(subset=["date"], keep="last")


def _period_features(
    df: Any,
    *,
    close_col: str = "close",
    period_suffix: str,
    return_periods: list[int],
    ma_periods: list[int],
    volatility_period: int,
    annualization: float,
    range_period: int,
    drawdown_period: int,
) -> dict[str, Any]:
    import pandas as pd

    if df is None or df.empty or close_col not in df.columns:
        return {}
    close = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if close.empty:
        return {}

    latest_date = None
    if "date" in df.columns:
        latest_date = str(pd.to_datetime(df.iloc[-1]["date"]).date())
    features: dict[str, Any] = {
        "latest": _safe_float(close.iloc[-1]),
        "latest_date": latest_date,
        "rows": int(len(close)),
    }
    for n in return_periods:
        if len(close) > n:
            features[f"return_{n}{period_suffix}_pct"] = _safe_float((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100)
    for n in ma_periods:
        if len(close) >= n:
            ma = close.rolling(n).mean().iloc[-1]
            features[f"ma{n}"] = _safe_float(ma)
            features[f"distance_ma{n}_pct"] = _safe_float((close.iloc[-1] / ma - 1) * 100) if ma else None
    if len(close) > volatility_period:
        ret = close.pct_change().dropna()
        vol = _safe_float(ret.tail(volatility_period).std() * annualization * 100)
        features[f"volatility_{volatility_period}{period_suffix}_annualized_pct"] = vol
        return_key = f"return_{volatility_period}{period_suffix}_pct"
        features[f"momentum_{volatility_period}{period_suffix}_score"] = _safe_float((features.get(return_key) or 0) / (vol or 1))
    if len(close) >= range_period:
        range_close = close.tail(range_period)
        high = range_close.max()
        low = range_close.min()
        features[f"drawdown_from_{range_period}{period_suffix}_high_pct"] = _safe_float((close.iloc[-1] / high - 1) * 100) if high else None
        features[f"distance_from_{range_period}{period_suffix}_low_pct"] = _safe_float((close.iloc[-1] / low - 1) * 100) if low else None
    if len(close) >= drawdown_period:
        drawdown_close = close.tail(drawdown_period)
        peak = drawdown_close.cummax()
        dd = drawdown_close / peak - 1
        features[f"max_drawdown_{drawdown_period}{period_suffix}_pct"] = _safe_float(dd.min() * 100)
    return features


def _resample_ohlcv(df: Any, rule: str) -> Any:
    import pandas as pd

    if df is None or df.empty or "date" not in df.columns:
        return df
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").set_index("date")
    agg: dict[str, str] = {"close": "last"}
    for col, how in [("open", "first"), ("high", "max"), ("low", "min"), ("volume", "sum"), ("amount", "sum")]:
        if col in work.columns:
            agg[col] = how
    out = work.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
    return out[["date", *[c for c in out.columns if c != "date"]]]


def _series_features(df: Any, *, close_col: str = "close") -> dict[str, Any]:
    if df is None or df.empty or close_col not in df.columns:
        return {}

    daily = _period_features(
        df,
        close_col=close_col,
        period_suffix="d",
        return_periods=[1, 5, 20, 60, 120, 252],
        ma_periods=[20, 60, 144, 300, 750],
        volatility_period=20,
        annualization=math.sqrt(252),
        range_period=60,
        drawdown_period=252,
    )
    weekly_df = _resample_ohlcv(df, "W-FRI")
    monthly_df = _resample_ohlcv(df, "ME")
    weekly = _period_features(
        weekly_df,
        close_col=close_col,
        period_suffix="w",
        return_periods=[1, 4, 13, 26, 52, 104],
        ma_periods=[10, 20, 40, 80, 156],
        volatility_period=13,
        annualization=math.sqrt(52),
        range_period=52,
        drawdown_period=156,
    )
    monthly = _period_features(
        monthly_df,
        close_col=close_col,
        period_suffix="m",
        return_periods=[1, 3, 6, 12, 24, 36],
        ma_periods=[6, 12, 24, 36, 60],
        volatility_period=12,
        annualization=math.sqrt(12),
        range_period=36,
        drawdown_period=60,
    )
    return {
        **daily,
        "timeframes": {
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly,
        },
    }


def _status_from_result(result: Any, name: str, cache_path: Path | None = None) -> DataStatus:
    data = getattr(result, "data", {}) if result is not None else {}
    latest = None
    if isinstance(data, dict):
        latest = data.get("end") or data.get("date") or data.get("period") or data.get("latest_date")
    return DataStatus(
        name=name,
        ok=bool(getattr(result, "ok", False)),
        source=str(getattr(result, "source", "")),
        note=str(getattr(result, "note", "")),
        latest_date=str(latest) if latest else None,
        cache_path=str(cache_path) if cache_path else None,
        error=None if bool(getattr(result, "ok", False)) else str(getattr(result, "note", "")),
    )


ASSET_UNIVERSE: dict[str, dict[str, dict[str, str]]] = {
    "a_share_indices": {
        "000001.SH": {"name": "上证指数", "source_symbol": "sh.000001"},
        "399001.SZ": {"name": "深证成指", "source_symbol": "sz.399001"},
        "399006.SZ": {"name": "创业板指", "source_symbol": "sz.399006"},
        "000300.SH": {"name": "沪深300", "source_symbol": "sh.000300"},
        "000905.SH": {"name": "中证500", "source_symbol": "sh.000905"},
        "000852.SH": {"name": "中证1000", "source_symbol": "sh.000852"},
        "399303.SZ": {"name": "国证2000", "source_symbol": "sz.399303"},
        "000015.SH": {"name": "上证红利", "source_symbol": "sh.000015"},
    },
    "a_share_stocks": {
        "002714.SZ": {"name": "牧原股份", "source_symbol": "002714"},
        "601899.SH": {"name": "紫金矿业A", "source_symbol": "601899"},
    },
    "etfs": {
        "159792.SZ": {"name": "港股通互联网ETF", "source_symbol": "159792"},
    },
    "hk_stocks": {
        "02899.HK": {"name": "紫金矿业H", "source_symbol": "02899"},
        "00700.HK": {"name": "腾讯控股", "source_symbol": "00700"},
        "09988.HK": {"name": "阿里巴巴-W", "source_symbol": "09988"},
        "03690.HK": {"name": "美团-W", "source_symbol": "03690"},
        "01810.HK": {"name": "小米集团-W", "source_symbol": "01810"},
    },
    "yahoo_assets": {
        "GC=F": {"name": "COMEX黄金期货", "source_symbol": "GC=F"},
        "SI=F": {"name": "COMEX白银期货", "source_symbol": "SI=F"},
        "HG=F": {"name": "COMEX铜期货", "source_symbol": "HG=F"},
        "CL=F": {"name": "WTI原油期货", "source_symbol": "CL=F"},
        "BZ=F": {"name": "Brent原油期货", "source_symbol": "BZ=F"},
        "DX-Y.NYB": {"name": "美元指数", "source_symbol": "DX-Y.NYB"},
        "^HSI": {"name": "恒生指数", "source_symbol": "^HSI"},
        "^HSTECH": {"name": "恒生科技指数", "source_symbol": "^HSTECH"},
        "2800.HK": {"name": "盈富基金", "source_symbol": "2800.HK"},
        "KWEB": {"name": "中概互联网ETF-KWEB", "source_symbol": "KWEB"},
    },
}


YAHOO_FALLBACK_SYMBOLS: dict[str, str] = {
    "a_share_stocks-002714": "002714.SZ",
    "a_share_stocks-601899": "601899.SS",
    "etfs-159792": "159792.SZ",
}


FRED_SERIES: dict[str, str] = {
    "fed_funds": "FEDFUNDS",
    "us10y": "DGS10",
    "us2y": "DGS2",
    "real_10y": "DFII10",
    "breakeven_10y": "T10YIE",
    "dollar_trade_weighted": "DTWEXBGS",
    "wti_spot": "DCOILWTICO",
    "brent_spot": "DCOILBRENTEU",
}


MODEL_LIBRARY: dict[str, dict[str, Any]] = {
    "macro_four_driver": {
        "skills": ["macro-four-driver-asset-map", "big-cycle-empire", "long-term-debt-cycle", "intermarket-analysis"],
        "inputs": ["PMI", "CPI/PPI", "社融M2", "海关", "油价", "美元", "实际利率"],
        "outputs": ["增长", "通胀", "信用", "外需", "资产映射"],
    },
    "greenspan_reaction": {
        "skills": [
            "greenspan-central-bank-reaction-function",
            "greenspan-financial-imbalances",
            "greenspan-productivity-long-cycle",
            "greenspan-crisis-liquidity-transmission",
        ],
        "inputs": ["通胀", "就业/PMI", "信用", "期限利差", "真实利率", "美元流动性"],
        "outputs": ["央行反应函数", "金融失衡", "生产率周期", "危机传导"],
    },
    "trend_breadth_sentiment": {
        "skills": ["market-breadth-appel", "bull-bear-reversal", "volume-price-analysis", "appel-market-cycle-segmentation"],
        "inputs": ["A股宽度", "指数MA144/MA300/MA750", "成交额", "涨跌停", "热股等权"],
        "outputs": ["情绪分数", "趋势状态", "风格扩散", "退潮风险"],
    },
    "dividend_spread": {
        "skills": ["fixed-income-valuation", "investment-risk-premium-estimation", "relative-valuation-multiples"],
        "inputs": ["上证红利价格", "股息率", "10Y国债", "PB/PE", "行业权重"],
        "outputs": ["股债利差", "估值分位", "价值陷阱风险", "仓位纪律"],
    },
    "gold_real_rate_usd": {
        "skills": ["intermarket-analysis", "fiat-money-inflation-anchor", "fixed-income-valuation"],
        "inputs": ["黄金", "实际利率", "名义利率", "通胀预期", "美元", "油价", "央行购金"],
        "outputs": ["黄金驱动拆解", "金股弹性", "风险冲击保护价值"],
    },
    "oil_supply_demand_curve": {
        "skills": ["commodity-cyclical-valuation", "futures-momentum-roll-yield", "intermarket-analysis"],
        "inputs": ["WTI", "Brent", "期限结构", "EIA库存", "OPEC供需", "美元", "实际利率"],
        "outputs": ["供需缺口", "库存压力", "通胀传导", "行业利润分配"],
    },
    "hog_capacity_cycle": {
        "skills": ["hog-production-cycle-analysis", "long-term-debt-cycle", "earnings-quality-analysis"],
        "inputs": ["猪价", "仔猪", "母猪", "猪粮比", "养殖利润", "冻品库存", "LH期货"],
        "outputs": ["产能去化", "盈利拐点", "现金流压力", "估值触发"],
    },
    "zijin_commodity_sensitivity": {
        "skills": ["commodity-cyclical-valuation", "dcf-valuation-mastery", "earnings-quality-analysis", "intermarket-analysis"],
        "inputs": ["黄金", "铜", "美元", "油价", "A/H价差", "产量", "成本", "资本开支"],
        "outputs": ["利润弹性", "估值情景", "项目风险", "商品周期暴露"],
    },
}


TASK_MODEL_VIEWS: dict[str, dict[str, Any]] = {
    "A股量化日报": {
        "models": ["trend_breadth_sentiment", "macro_four_driver", "greenspan_reaction"],
        "feature_buckets": ["a_share_indices", "market", "macro", "fred"],
        "causal_chains": [
            "成交额/成交量 -> 宽度与涨跌停 -> 热股等权/全A等权 -> 情绪分数与赚钱效应",
            "300MA/144MA趋势 -> 风格指数强弱 -> 板块扩散或退潮 -> 仓位纪律",
            "社融M2/利率/美元 -> 折现率与流动性 -> 大盘/小盘/微盘风格切换",
        ],
    },
    "5-宏观数据周报": {
        "models": ["macro_four_driver", "greenspan_reaction", "gold_real_rate_usd", "oil_supply_demand_curve"],
        "feature_buckets": ["macro", "fred", "yahoo_assets", "a_share_indices"],
        "causal_chains": [
            "增长/通胀/信用/外需 -> 政策反应函数 -> 债券/黄金/RMB/A股风格",
            "油价 -> PPI与通胀预期 -> 实际利率 -> 黄金与成长股估值",
            "美元/美债实际利率 -> 外部流动性 -> 港股互联网与商品资产风险偏好",
        ],
    },
    "1-红利指数周报": {
        "models": ["dividend_spread", "trend_breadth_sentiment", "macro_four_driver"],
        "feature_buckets": ["a_share_indices", "market", "fred", "macro"],
        "causal_chains": [
            "红利股息率-10Y利率利差 -> 相对吸引力 -> 加仓/等待纪律",
            "MA750位置 -> 趋势风险 -> 估值修复或价值陷阱识别",
            "信用与经济增长 -> 高股息盈利稳定性 -> 防御资产性价比",
        ],
    },
    "2-牧原股份周报": {
        "models": ["hog_capacity_cycle", "macro_four_driver", "trend_breadth_sentiment"],
        "feature_buckets": ["a_share_stocks", "macro", "market", "fred"],
        "causal_chains": [
            "猪价/仔猪价/能繁母猪 -> 产能去化阶段 -> 牧原盈利拐点",
            "养殖利润 -> 现金流/债务压力 -> 估值容忍度",
            "牧原股价趋势/量能 -> 产业验证强弱 -> 25-31元观察区触发条件",
        ],
    },
    "3-紫金矿业周报": {
        "models": ["zijin_commodity_sensitivity", "gold_real_rate_usd", "macro_four_driver", "greenspan_reaction"],
        "feature_buckets": ["a_share_stocks", "hk_stocks", "yahoo_assets", "fred", "macro"],
        "causal_chains": [
            "金价/铜价/美元 -> 紫金收入与利润弹性 -> 估值情景",
            "实际利率/通胀预期 -> 黄金估值 -> 紫金贵金属业务弹性",
            "油价/汇率/项目成本 -> 成本曲线 -> 盈利安全边际",
        ],
    },
    "4-黄金宏观周报": {
        "models": ["gold_real_rate_usd", "greenspan_reaction", "macro_four_driver"],
        "feature_buckets": ["yahoo_assets", "fred", "macro"],
        "causal_chains": [
            "实际利率 -> 黄金机会成本 -> 金价趋势",
            "美元指数 -> 非美购买力与风险偏好 -> 金价波动",
            "油价/通胀预期/央行反应 -> 实际利率路径 -> 黄金配置价值",
        ],
    },
    "159792-定投周报": {
        "models": ["macro_four_driver", "trend_breadth_sentiment", "greenspan_reaction"],
        "feature_buckets": ["etfs", "hk_stocks", "yahoo_assets", "fred", "macro"],
        "causal_chains": [
            "美元/美债利率/HKD流动性 -> 港股估值折现率 -> 159792定投节奏",
            "腾讯/阿里/美团/小米趋势 -> ETF基本面验证 -> 0.428-0.55区间纪律",
            "国内信用与消费修复 -> 互联网盈利预期 -> 南向资金风险偏好",
        ],
    },
    "生猪周期周报": {
        "models": ["hog_capacity_cycle", "macro_four_driver", "trend_breadth_sentiment"],
        "feature_buckets": ["macro", "a_share_stocks", "market"],
        "causal_chains": [
            "能繁母猪/仔猪/猪价 -> 产能去化 -> 周期位置",
            "猪粮比/养殖利润 -> 主体现金流压力 -> 去化速度",
            "猪价对CPI食品项 -> 政策与消费预期 -> 养殖股估值",
        ],
    },
    "宏观-PMI数据解读": {
        "models": ["macro_four_driver", "greenspan_reaction", "trend_breadth_sentiment"],
        "feature_buckets": ["macro", "a_share_indices", "market"],
        "causal_chains": [
            "生产/新订单/新出口订单 -> 增长动能 -> A股周期/成长风格",
            "库存/价格分项 -> 补库或去库阶段 -> 商品与利润传导",
            "PMI就业/订单 -> 政策反应 -> 利率与风险偏好",
        ],
    },
    "宏观-海关进出口数据解读": {
        "models": ["macro_four_driver", "greenspan_reaction", "oil_supply_demand_curve"],
        "feature_buckets": ["macro", "yahoo_assets", "fred"],
        "causal_chains": [
            "出口增速/地区结构 -> 外需强弱 -> RMB与制造链景气",
            "进口量价/能源账单 -> PPI与利润分配 -> 商品链影响",
            "贸易顺差 -> 外汇流动性 -> 港股与A股风险偏好",
        ],
    },
    "宏观-CPI-PPI数据解读": {
        "models": ["macro_four_driver", "greenspan_reaction", "gold_real_rate_usd", "hog_capacity_cycle"],
        "feature_buckets": ["macro", "fred", "yahoo_assets"],
        "causal_chains": [
            "CPI核心/服务/食品 -> 真实需求与通胀黏性 -> 政策空间",
            "PPI上中下游 -> 利润转移 -> 周期/制造/消费板块影响",
            "猪价/油价 -> CPI/PPI扰动 -> 实际利率 -> 黄金与债券",
        ],
    },
    "宏观-社融M2数据解读": {
        "models": ["macro_four_driver", "greenspan_reaction", "dividend_spread"],
        "feature_buckets": ["macro", "fred", "a_share_indices", "market"],
        "causal_chains": [
            "社融/信贷结构 -> 实体融资需求 -> A股盈利预期",
            "M1-M2差 -> 资金活化或空转 -> 红利/成长相对占优",
            "政府债/企业中长贷 -> 宽信用质量 -> 利率与风险偏好",
        ],
    },
    "生猪产能出清-每月10号": {
        "models": ["hog_capacity_cycle", "macro_four_driver"],
        "feature_buckets": ["macro", "a_share_stocks"],
        "causal_chains": [
            "能繁母猪 -> 未来供给 -> 猪价弹性",
            "仔猪/母猪价格 -> 补栏意愿 -> 产能去化是否反转",
            "养殖利润/现金流 -> 行业出清强度 -> 牧原估值触发",
        ],
    },
    "牧原股份-产能出清月度检查": {
        "models": ["hog_capacity_cycle", "trend_breadth_sentiment"],
        "feature_buckets": ["a_share_stocks", "macro", "market"],
        "causal_chains": [
            "行业产能去化 -> 牧原出栏/成本/现金流 -> 盈利拐点",
            "猪价与成本差 -> 单头利润 -> 估值弹性",
            "股价趋势/成交量 -> 基本面预期验证 -> 操作纪律",
        ],
    },
    "月度原油市场报告": {
        "models": ["oil_supply_demand_curve", "macro_four_driver", "greenspan_reaction", "gold_real_rate_usd"],
        "feature_buckets": ["yahoo_assets", "fred", "macro"],
        "causal_chains": [
            "WTI/Brent/期限结构 -> 供需与库存压力 -> 油价趋势",
            "油价 -> PPI/CPI与通胀预期 -> 央行反应函数",
            "美元/实际利率 -> 商品金融条件 -> 能源链利润分配",
        ],
    },
    "桌面文件结构审计（每周日）": {
        "models": [],
        "feature_buckets": [],
        "causal_chains": ["仅执行桌面分类审计；不使用投研模型。"],
    },
}


def model_views_for_task(task: str | None = None) -> dict[str, Any]:
    if task:
        return {task: TASK_MODEL_VIEWS.get(task, {})}
    return TASK_MODEL_VIEWS


def fetch_yahoo_chart(symbol: str, *, range_: str = "2y", interval: str = "1d", max_age_seconds: int = 18 * 3600) -> dict[str, Any]:
    cached = _fresh_payload("yahoo_assets", symbol, max_age_seconds)
    if cached:
        return cached
    import pandas as pd
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        resp = requests.get(url, params={"range": range_, "interval": interval}, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        result = (resp.json().get("chart", {}).get("result") or [])[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or quote.get("close") or []
        rows = []
        for i, ts in enumerate(timestamps):
            close = _safe_float(adj[i] if i < len(adj) else None)
            if close is None:
                continue
            rows.append(
                {
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                    "open": _safe_float((quote.get("open") or [None])[i] if i < len(quote.get("open") or []) else None),
                    "high": _safe_float((quote.get("high") or [None])[i] if i < len(quote.get("high") or []) else None),
                    "low": _safe_float((quote.get("low") or [None])[i] if i < len(quote.get("low") or []) else None),
                    "close": close,
                    "volume": _safe_float((quote.get("volume") or [None])[i] if i < len(quote.get("volume") or []) else None),
                }
            )
        df = _normalize_columns(pd.DataFrame(rows))
        payload = {
            "status": asdict(DataStatus(symbol, True, url, "Yahoo chart API", str(df.iloc[-1]["date"].date()), str(_cache_path("yahoo_assets", symbol)))),
            "features": _series_features(df),
            "rows_tail": df.tail(260).to_dict(orient="records"),
        }
    except Exception as exc:
        payload = {
            "status": asdict(DataStatus(symbol, False, url, "Yahoo chart API failed", None, str(_cache_path("yahoo_assets", symbol)), repr(exc)[:220])),
            "features": {},
            "rows_tail": [],
        }
    _write_json(_cache_path("yahoo_assets", symbol), payload)
    return payload


def fetch_akshare_daily(kind: str, symbol: str, *, max_age_seconds: int = 18 * 3600) -> dict[str, Any]:
    key = f"{kind}-{symbol}"
    cached = _fresh_payload(kind, key, max_age_seconds)
    if cached:
        return cached
    import akshare as ak
    import pandas as pd

    today = date.today().strftime("%Y%m%d")
    failures: list[str] = []
    calls: list[tuple[str, Callable[[], Any]]] = []
    if kind == "a_share_stocks":
        calls = [("AkShare stock_zh_a_hist", lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20180101", end_date=today, adjust="qfq"))]
    elif kind == "etfs":
        calls = [("AkShare fund_etf_hist_em", lambda: ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date="20180101", end_date=today, adjust="qfq"))]
    elif kind == "hk_stocks":
        calls = [("AkShare stock_hk_hist", lambda: ak.stock_hk_hist(symbol=symbol, period="daily", start_date="20180101", end_date=today, adjust="qfq"))]
    else:
        raise ValueError(f"unsupported akshare kind: {kind}")

    payload: dict[str, Any] | None = None
    for source, fn in calls:
        try:
            raw = fn()
            if raw is None or raw.empty:
                failures.append(f"{source}: empty")
                continue
            df = _normalize_columns(raw)
            payload = {
                "status": asdict(DataStatus(key, True, source, "AkShare daily history", str(df.iloc[-1]["date"].date()), str(_cache_path(kind, key)))),
                "features": _series_features(df),
                "rows_tail": df.tail(260).to_dict(orient="records"),
                "failures_before_success": failures,
            }
            break
        except Exception as exc:
            failures.append(f"{source}: {repr(exc)[:180]}")
    if payload is None:
        yahoo_symbol = YAHOO_FALLBACK_SYMBOLS.get(key)
        if yahoo_symbol:
            fallback = fetch_yahoo_chart(yahoo_symbol, max_age_seconds=max_age_seconds)
            if _payload_ok(fallback):
                payload = fallback
                payload["status"] = dict(payload.get("status") or {})
                payload["status"].update(
                    {
                        "name": key,
                        "source": f"Yahoo chart API fallback ({yahoo_symbol})",
                        "note": "AkShare failed; using Yahoo chart fallback",
                        "cache_path": str(_cache_path(kind, key)),
                    }
                )
                payload["failures_before_success"] = failures
        if payload is None:
            payload = {
                "status": asdict(DataStatus(key, False, "AkShare", "all AkShare daily history sources failed", None, str(_cache_path(kind, key)), "; ".join(failures)[:500])),
                "features": {},
                "rows_tail": [],
                "failures": failures,
            }
    _write_json(_cache_path(kind, key), payload)
    return payload


def fetch_a_share_index(symbol: str, *, max_age_seconds: int = 18 * 3600) -> dict[str, Any]:
    cached = _fresh_payload("a_share_indices", symbol, max_age_seconds)
    if cached:
        return cached
    from data_sources import fetch_a_share_index_daily
    import pandas as pd

    cache_path = _cache_path("a_share_indices", symbol)
    result = fetch_a_share_index_daily(symbol, include_rows=True)
    if result.ok:
        df = _normalize_columns(pd.DataFrame(result.data.get("rows_data") or []))
        payload = {
            "status": asdict(_status_from_result(result, symbol, cache_path)),
            "features": _series_features(df),
            "rows_tail": df.tail(260).to_dict(orient="records"),
            "failures_before_success": result.data.get("failures_before_success", []),
        }
    else:
        payload = {
            "status": asdict(_status_from_result(result, symbol, cache_path)),
            "features": {},
            "rows_tail": [],
            "failures": result.data.get("failures", []) if isinstance(result.data, dict) else [],
        }
    _write_json(cache_path, payload)
    return payload


def fetch_fred_series(alias: str, series_id: str, *, max_age_seconds: int = 18 * 3600) -> dict[str, Any]:
    cached = _fresh_payload("fred", alias, max_age_seconds)
    if cached:
        return cached
    from data_sources import fetch_fred_graph_csv
    import pandas as pd

    cache_path = _cache_path("fred", alias)
    result = fetch_fred_graph_csv(series_id, max_age_seconds=max_age_seconds)
    if result.ok:
        rows = result.data.get("rows_data") or []
        df = pd.DataFrame(rows).rename(columns={"observation_date": "date", series_id: "close"})
        df = _normalize_columns(df)
        payload = {
            "status": asdict(_status_from_result(result, alias, cache_path)),
            "series_id": series_id,
            "features": _series_features(df),
            "rows_tail": df.tail(260).to_dict(orient="records"),
        }
    else:
        payload = {
            "status": asdict(_status_from_result(result, alias, cache_path)),
            "series_id": series_id,
            "features": {},
            "rows_tail": [],
        }
    _write_json(cache_path, payload)
    return payload


def collect_macro_snapshots(*, max_age_seconds: int = 18 * 3600, admission: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = _fresh_aggregate("macro", "snapshots", max_age_seconds)
    if cached:
        return cached
    from data_sources import (
        get_cpi_ppi,
        get_cross_asset_fallback_sources,
        get_customs,
        get_fed_data_sources,
        get_international_oil_sources,
        get_official_source_registry,
        get_pmi,
        get_social_finance_m2,
    )

    source_admission = admission or _load_source_admission()
    funcs = {
        "cpi_ppi": get_cpi_ppi,
        "social_finance_m2": get_social_finance_m2,
        "fed_data_sources": get_fed_data_sources,
        "official_source_registry": get_official_source_registry,
    }
    supporting_funcs = {
        "cross_asset_fallback_sources": get_cross_asset_fallback_sources,
    }
    disclosure_funcs = {
        "pmi": get_pmi,
        "customs": get_customs,
        "international_oil_sources": get_international_oil_sources,
    }
    out: dict[str, Any] = {}
    for name, fn in funcs.items():
        try:
            result = fn()
            out[name] = _annotate_admission(name, {"status": asdict(_status_from_result(result, name)), "data": result.to_dict()}, source_admission)
        except Exception as exc:
            out[name] = _annotate_admission(name, {"status": asdict(DataStatus(name, False, "scripts.data_sources", "macro snapshot failed", error=repr(exc)[:220])), "data": {}}, source_admission)
    out["_source_admission"] = source_admission
    out["_supporting_context"] = {}
    for name, fn in supporting_funcs.items():
        try:
            result = fn()
            item = {"status": asdict(_status_from_result(result, name)), "data": result.to_dict()}
        except Exception as exc:
            item = {"status": asdict(DataStatus(name, False, "scripts.data_sources", "macro supporting context failed", error=repr(exc)[:220])), "data": {}}
        out["_supporting_context"][name] = _annotate_admission(name, item, source_admission)
    out["_disclosure_fallbacks"] = {}
    for name, fn in disclosure_funcs.items():
        try:
            result = fn()
            item = {"status": asdict(_status_from_result(result, name)), "data": result.to_dict()}
        except Exception as exc:
            item = {"status": asdict(DataStatus(name, False, "scripts.data_sources", "macro disclosure fallback failed", error=repr(exc)[:220])), "data": {}}
        out["_disclosure_fallbacks"][name] = _annotate_admission(name, item, source_admission)
    _write_json(_cache_path("macro", "snapshots"), out)
    return out


def collect_market_breadth(max_age_seconds: int, *, admission: dict[str, Any] | None = None) -> dict[str, Any]:
    cached = _fresh_aggregate("market", "breadth_and_dividend", max_age_seconds)
    if cached:
        return cached
    from data_sources import fetch_a_share_market_breadth, fetch_dividend_weekly_inputs, fetch_industry_board_snapshot

    source_admission = admission or _load_source_admission()
    out: dict[str, Any] = {}
    for name, fn in {
        "a_share_market_breadth": fetch_a_share_market_breadth,
        "dividend_weekly_inputs": fetch_dividend_weekly_inputs,
        "industry_board_snapshot": fetch_industry_board_snapshot,
    }.items():
        try:
            result = fn() if name != "a_share_market_breadth" else fn(max_cache_age_seconds=max_age_seconds)
            out[name] = _annotate_admission(name, {"status": asdict(_status_from_result(result, name)), "data": result.to_dict()}, source_admission)
        except Exception as exc:
            out[name] = _annotate_admission(name, {"status": asdict(DataStatus(name, False, "scripts.data_sources", "market breadth helper failed", error=repr(exc)[:220])), "data": {}}, source_admission)
    _write_json(_cache_path("market", "breadth_and_dividend"), out)
    return out


def collect_report_data_base(*, update: bool = True, max_age_seconds: int = 18 * 3600, light: bool = False) -> dict[str, Any]:
    if not update:
        latest = _read_json(LATEST_JSON)
        if latest:
            return latest
        return {"generated_at": None, "ok": False, "error": "no latest report_data_base cache"}

    data: dict[str, Any] = {
        "source_admission": _load_source_admission(),
        "asset_universe": ASSET_UNIVERSE,
        "models": MODEL_LIBRARY,
        "a_share_indices": {},
        "a_share_stocks": {},
        "etfs": {},
        "hk_stocks": {},
        "yahoo_assets": {},
        "fred": {},
        "macro": {},
        "market": {},
    }
    light_skipped: dict[str, list[str]] = {"a_share_indices": [], "a_share_stocks": [], "etfs": [], "hk_stocks": [], "yahoo_assets": [], "fred": [], "macro": [], "market": []}

    # ── light mode: skip HK stocks entirely; check cache freshness for others ──
    for symbol, meta in ASSET_UNIVERSE["a_share_indices"].items():
        if light and _is_cache_fresh("a_share_indices", symbol, max_age_seconds):
            cached = _latest_payload("a_share_indices", symbol)
            data["a_share_indices"][symbol] = cached or fetch_a_share_index(meta["source_symbol"], max_age_seconds=max_age_seconds)
            if cached:
                light_skipped["a_share_indices"].append(symbol)
                continue
        data["a_share_indices"][symbol] = fetch_a_share_index(meta["source_symbol"], max_age_seconds=max_age_seconds)

    for bucket in ["a_share_stocks", "etfs", "hk_stocks"]:
        if light and bucket == "hk_stocks":
            continue
        for symbol, meta in ASSET_UNIVERSE[bucket].items():
            if light and _is_cache_fresh(bucket, symbol, max_age_seconds):
                cached = _latest_payload(bucket, symbol)
                data[bucket][symbol] = cached
                if cached:
                    light_skipped[bucket].append(symbol)
                    continue
            data[bucket][symbol] = fetch_akshare_daily(bucket, meta["source_symbol"], max_age_seconds=max_age_seconds)

    for symbol, meta in ASSET_UNIVERSE["yahoo_assets"].items():
        if light and _is_cache_fresh("yahoo_assets", symbol, max_age_seconds):
            cached = _latest_payload("yahoo_assets", symbol)
            data["yahoo_assets"][symbol] = cached or fetch_yahoo_chart(meta["source_symbol"], max_age_seconds=max_age_seconds)
            if cached:
                light_skipped["yahoo_assets"].append(symbol)
                continue
        data["yahoo_assets"][symbol] = fetch_yahoo_chart(meta["source_symbol"], max_age_seconds=max_age_seconds)

    for alias, series_id in FRED_SERIES.items():
        if light and _is_cache_fresh("fred", alias, max_age_seconds):
            cached = _latest_payload("fred", alias)
            data["fred"][alias] = cached or fetch_fred_series(alias, series_id, max_age_seconds=max_age_seconds)
            if cached:
                light_skipped["fred"].append(alias)
                continue
        data["fred"][alias] = fetch_fred_series(alias, series_id, max_age_seconds=max_age_seconds)

    if light and _is_cache_fresh("macro", "_all", max_age_seconds):
        cached = _read_json(_cache_path("macro", "_all"))
        if cached and _payload_ok(cached):
            data["macro"] = cached
            light_skipped["macro"].append("_all")
        else:
            data["macro"] = collect_macro_snapshots(max_age_seconds=max_age_seconds, admission=data["source_admission"])
    else:
        data["macro"] = collect_macro_snapshots(max_age_seconds=max_age_seconds, admission=data["source_admission"])

    if light and _is_cache_fresh("market", "_all", max_age_seconds):
        cached = _read_json(_cache_path("market", "_all"))
        if cached and _payload_ok(cached):
            data["market"] = cached
            light_skipped["market"].append("_all")
        else:
            data["market"] = collect_market_breadth(max_age_seconds, admission=data["source_admission"])
    else:
        data["market"] = collect_market_breadth(max_age_seconds, admission=data["source_admission"])

    statuses = list(iter_statuses(data))
    ok_count = sum(1 for s in statuses if s.ok)
    fail_count = sum(1 for s in statuses if not s.ok)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(BASE_DIR),
        "daily_dir": str(DAILY_DIR),
        "ok": ok_count > 0,
        "status_summary": {"ok": ok_count, "failed": fail_count},
        "light": {"enabled": light, "skipped": {k: v for k, v in light_skipped.items() if v}} if light else None,
        "data": data,
    }
    # Write macro & market cache for light-mode freshness checks
    if data.get("macro"):
        _write_json(_cache_path("macro", "_all"), data["macro"])
    if data.get("market"):
        _write_json(_cache_path("market", "_all"), data["market"])

    _write_json(DAILY_DIR / "report_data_base.json", payload)
    _write_json(LATEST_JSON, payload)
    LATEST_MD.write_text(render_markdown_summary(payload), encoding="utf-8")
    return payload


def iter_statuses(payload: dict[str, Any]):
    def walk(value: Any):
        if isinstance(value, dict):
            status = value.get("status")
            if isinstance(status, dict) and "ok" in status and "name" in status:
                yield DataStatus(**{k: status.get(k) for k in DataStatus.__dataclass_fields__})
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    yield from walk(payload.get("data", payload))


def compact_inputs_for_report(payload: dict[str, Any], *, task: str | None = None) -> dict[str, Any]:
    data = payload.get("data", payload)
    compact: dict[str, Any] = {
        "generated_at": payload.get("generated_at"),
        "status_summary": payload.get("status_summary"),
        "source_admission": data.get("source_admission", {}),
        "models": data.get("models", MODEL_LIBRARY),
        "model_views": model_views_for_task(task),
        "macro": data.get("macro", {}),
        "market": data.get("market", {}),
        "features": {},
        "disclosure_fallbacks": {},
        "supporting_context": {},
        "failures": [],
    }
    for bucket in ["a_share_indices", "a_share_stocks", "etfs", "hk_stocks", "yahoo_assets", "fred"]:
        compact["features"][bucket] = {}
        for symbol, item in (data.get(bucket) or {}).items():
            compact["features"][bucket][symbol] = {
                "status": item.get("status", {}),
                "features": item.get("features", {}),
            }
    for bucket in ["macro", "market"]:
        compact["features"][bucket] = {}
        for key, item in (data.get(bucket) or {}).items():
            if str(key).startswith("_"):
                continue
            status = item.get("status", {})
            data_content = item.get("data", {})
            summary = {}
            if isinstance(data_content, dict):
                summary = {k: v for k, v in data_content.items() if not isinstance(v, (dict, list)) or len(str(v)) < 120}
            compact["features"][bucket][key] = {
                "status": status,
                "summary": summary,
                "admission": item.get("admission", {}),
            }
    macro = data.get("macro") or {}
    for source_bucket, compact_key in [
        ("_disclosure_fallbacks", "disclosure_fallbacks"),
        ("_supporting_context", "supporting_context"),
    ]:
        compact[compact_key]["macro"] = {}
        for key, item in (macro.get(source_bucket) or {}).items():
            status = item.get("status", {}) if isinstance(item, dict) else {}
            data_content = item.get("data", {}) if isinstance(item, dict) else {}
            summary = {}
            if isinstance(data_content, dict):
                summary = {k: v for k, v in data_content.items() if not isinstance(v, (dict, list)) or len(str(v)) < 120}
            compact[compact_key]["macro"][key] = {
                "status": status,
                "summary": summary,
                "admission": item.get("admission", {}) if isinstance(item, dict) else {},
            }
    for status in iter_statuses(payload):
        if not status.ok:
            compact["failures"].append(asdict(status))
    if task:
        compact["task_hint"] = task
    return compact


def render_markdown_summary(payload: dict[str, Any]) -> str:
    compact = compact_inputs_for_report(payload)
    lines = [
        "# Cron Report Data Base",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- cache_dir: {payload.get('cache_dir')}",
        f"- status: ok={payload.get('status_summary', {}).get('ok')} failed={payload.get('status_summary', {}).get('failed')}",
        "",
        "## Source Admission",
    ]
    admission = compact.get("source_admission") or {}
    probe_summary = admission.get("probe_summary") or {}
    lines.append(
        f"- probe_generated_at: {admission.get('probe_generated_at')} total={probe_summary.get('total')} "
        f"ok={probe_summary.get('ok')} recommended_for_core={probe_summary.get('recommended_for_core')}"
    )
    core_names = ", ".join(sorted((admission.get("core_sources") or {}).keys())) or "none"
    fallback_names = ", ".join(sorted((admission.get("disclosure_fallback_sources") or {}).keys())) or "none"
    supporting_names = ", ".join(sorted((admission.get("supporting_context_sources") or {}).keys())) or "none"
    audit = admission.get("admission_audit") or {}
    lines.extend(
        [
            f"- core_sources: {core_names}",
            f"- disclosure_fallback_sources: {fallback_names}",
            f"- supporting_context_sources: {supporting_names}",
            f"- admission_audit: clean={audit.get('clean')} promoted_core_count={audit.get('promoted_core_count')} "
            f"probe_recommended_count={audit.get('probe_recommended_count')} allowlist_not_recommended={audit.get('allowlist_not_recommended')} "
            f"recommended_not_allowlisted={audit.get('recommended_not_allowlisted')}",
            "- rule: core_sources may drive core conclusions; disclosure_fallback_sources require stale/archive caveats until live parsers are added.",
            "",
        "## Core Feature Snapshot",
        ]
    )
    for bucket, rows in compact.get("features", {}).items():
        lines.append(f"\n### {bucket}")
        for symbol, item in rows.items():
            st = item.get("status", {})
            ft = item.get("features", {})
            lines.append(
                f"- {symbol}: ok={st.get('ok')} latest={ft.get('latest')} date={ft.get('latest_date')} "
                f"r20={ft.get('return_20d_pct')}% ma144={ft.get('ma144')} ma300={ft.get('ma300')} "
                f"dd252={ft.get('max_drawdown_252d_pct')}% source={st.get('source')} admission={item.get('admission', {}).get('usage', '')}"
            )
    lines.extend(["", "## Disclosure Fallbacks"])
    disclosure = compact.get("disclosure_fallbacks", {}).get("macro", {})
    if disclosure:
        for name, item in disclosure.items():
            st = item.get("status", {})
            adm = item.get("admission", {})
            lines.append(
                f"- {name}: ok={st.get('ok')} source={st.get('source')} usage={adm.get('usage')} "
                f"reason={adm.get('reason')} pending={adm.get('pending_live_parser')}"
            )
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Supporting Context"])
    supporting = compact.get("supporting_context", {}).get("macro", {})
    if supporting:
        for name, item in supporting.items():
            st = item.get("status", {})
            adm = item.get("admission", {})
            lines.append(f"- {name}: ok={st.get('ok')} source={st.get('source')} usage={adm.get('usage')} reason={adm.get('reason')}")
    else:
        lines.append("- None recorded.")
    failures = compact.get("failures") or []
    lines.extend(["", "## Model Views"])
    for task, view in compact.get("model_views", {}).items():
        models = ", ".join(view.get("models") or []) or "none"
        chains = view.get("causal_chains") or []
        lines.append(f"- {task}: models={models}")
        for chain in chains[:4]:
            lines.append(f"  - {chain}")
    lines.extend(["", "## Failed Or Fallback Items"])
    if failures:
        for item in failures[:80]:
            lines.append(f"- {item.get('name')}: {item.get('error') or item.get('note')} source={item.get('source')}")
    else:
        lines.append("- None recorded.")
    lines.extend([
        "",
        "## Report Usage",
        "- Reports should import `collect_report_data_base` or read `generated/report_data_base/latest.json` first.",
        "- If a live source fails, use the latest cache and disclose failure reason, fallback date, and conclusion impact.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update/read shared cron report data base")
    parser.add_argument("--update", action="store_true", help="fetch/update data and write cache")
    parser.add_argument("--offline", action="store_true", help="only read latest cache")
    parser.add_argument("--light", action="store_true", help="skip slower non-core buckets where possible")
    parser.add_argument("--max-age-hours", type=float, default=18.0)
    parser.add_argument("--compact", action="store_true", help="print compact report inputs instead of full payload")
    parser.add_argument("--task", help="include task-specific model view in compact output")
    args = parser.parse_args()

    if args.offline:
        payload = collect_report_data_base(update=False)
    else:
        payload = collect_report_data_base(update=True if args.update or not args.offline else False, max_age_seconds=int(args.max_age_hours * 3600), light=args.light)
    print(json.dumps(compact_inputs_for_report(payload, task=args.task) if args.compact else payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
