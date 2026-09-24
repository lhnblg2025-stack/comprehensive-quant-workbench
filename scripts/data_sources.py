#!/usr/bin/env python3
"""Stable data-source helpers and fallbacks for report cron jobs.

Cron reports should import this module first for official macro data. The goal is
not to replace live APIs, but to provide a verified fallback layer when AkShare or
web endpoints are unavailable.
"""
from __future__ import annotations
import logging

from dataclasses import dataclass, asdict
from datetime import date, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
import concurrent.futures
import functools


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_SOURCES = ROOT / "config" / "private_data_sources.json"
CACHE_DIR = ROOT / "generated" / "data_source_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 请求级缓存：同一 akshare 接口+参数在 TTL 内重复调用命中缓存（默认当日有效，6 小时），
# 避免日报/周报等多脚本同一天重复抓同一数据。导入失败时降级为直接调用。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from report_paths import report_path  # noqa: E402
try:
    from quant_platform.legacy import cached_fetch
except Exception:
    cached_fetch = None
try:
    from quant_system.utils import safe_float as _safe_float_impl
except ImportError:
    _safe_float_impl = None

FETCH_CACHE_TTL = 6 * 3600


def _cached_ak(fn: Callable, *args, **kwargs) -> Any:
    """akshare 调用 + 请求级缓存（cached_fetch 不可用时直接调用）。"""
    if cached_fetch is None:
        return fn(*args, **kwargs)
    return cached_fetch(fn, *args, ttl_seconds=FETCH_CACHE_TTL, **kwargs)


# Default timeout for blocking data-source operations (seconds).
# Many AkShare / yfinance calls can hang indefinitely without this.
FETCH_TIMEOUT = 20


def _timeout_call(fn: Callable, timeout: int = FETCH_TIMEOUT, default: Any = None) -> Any:
    """Run a blocking call with a hard timeout using ThreadPoolExecutor.

    Usage:
        df = _timeout_call(lambda: ak.macro_china_pmi(), timeout=15)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            if default is _SENTINEL:
                raise TimeoutError(f"data source timed out after {timeout}s")
            return default


_SENTINEL = object()


@dataclass(frozen=True)
class SourceResult:
    name: str
    ok: bool
    data: dict[str, Any]
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Verified official/public fallback snapshots.
# Source: 国家统计局, 2026-07-09, 2026年6月份CPI/PPI数据解读
CPI_PPI_FALLBACKS: dict[str, dict[str, Any]] = {
    "2026-06": {
        "period": "2026-06",
        "published_date": "2026-07-09",
        "source_url": "https://www.stats.gov.cn/sj/sjjd/202607/t20260709_1964082.html",
        "cpi": {
            "yoy": 1.0,
            "mom": -0.3,
            "core_yoy": 1.0,
            "food_yoy": -1.6,
            "food_mom": -0.4,
            "non_food_yoy_note": "工业消费品价格同比上涨2.9%，服务价格同比上涨0.8%",
            "service_yoy": 0.8,
            "industrial_goods_yoy": 2.9,
            "pork_yoy": -15.9,
            "egg_yoy": 20.0,
            "gold_jewelry_yoy": 28.1,
            "gasoline_yoy": 17.0,
            "gold_jewelry_mom": -8.7,
            "gasoline_mom": -4.9,
        },
        "ppi": {
            "yoy": 4.1,
            "mom": -0.3,
            "coal_mining_yoy": 20.6,
            "electrical_machinery_yoy": 5.1,
            "computer_communication_electronics_yoy": 3.3,
            "ferrous_smelting_yoy": 3.1,
            "nonmetal_mineral_products_yoy": -4.4,
            "power_heat_supply_yoy": -4.4,
            "alcohol_beverage_tea_yoy": -5.3,
            "auto_manufacturing_yoy": -2.1,
            "oil_extraction_mom": -16.0,
            "refined_petroleum_mom": -3.1,
            "coal_mining_mom": 5.6,
        },
        "interpretation_points": [
            "CPI同比1.0%、核心CPI同比1.0%，消费端仍是温和通胀而非全面通胀。",
            "CPI环比-0.3%主要受黄金饰品、汽油和食品价格回落拖累。",
            "PPI同比4.1%且较上月扩大0.2个百分点，工业品价格仍强于居民消费端。",
            "PPI环比-0.3%显示上游价格动能边际降温，尤其受国际原油下行影响。",
        ],
    }
}


# Source: 中国人民银行, 2026-07-15, 2026年上半年金融统计数据报告
SOCIAL_FINANCE_M2_FALLBACKS: dict[str, dict[str, Any]] = {
    "2026-06": {
        "period": "2026-06",
        "published_date": "2026-07-15",
        "source_url": "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/2026071515025183948/index.html",
        "social_finance_stock": {
            "total_trillion_yuan": 462.06,
            "yoy": 7.4,
            "rmb_loans_to_real_economy_trillion_yuan": 279.16,
            "rmb_loans_to_real_economy_yoy": 5.3,
            "foreign_currency_loans_rmb_trillion_yuan": 1.18,
            "foreign_currency_loans_yoy": -2.9,
            "entrusted_loans_trillion_yuan": 11.24,
            "entrusted_loans_yoy": 0.5,
            "trust_loans_trillion_yuan": 4.62,
            "trust_loans_yoy": 4.0,
            "undiscounted_bank_acceptance_trillion_yuan": 2.02,
            "undiscounted_bank_acceptance_yoy": -2.8,
            "enterprise_bonds_trillion_yuan": 36.08,
            "enterprise_bonds_yoy": 8.9,
            "government_bonds_trillion_yuan": 101.36,
            "government_bonds_yoy": 14.2,
            "domestic_equity_financing_trillion_yuan": 12.49,
            "domestic_equity_financing_yoy": 5.0,
        },
        "social_finance_stock_share": {
            "rmb_loans_to_real_economy_pct": 60.4,
            "rmb_loans_to_real_economy_yoy_change_pct_points": -1.2,
            "foreign_currency_loans_pct": 0.3,
            "entrusted_loans_pct": 2.4,
            "trust_loans_pct": 1.0,
            "undiscounted_bank_acceptance_pct": 0.4,
            "enterprise_bonds_pct": 7.8,
            "government_bonds_pct": 21.9,
            "government_bonds_yoy_change_pct_points": 1.3,
            "domestic_equity_financing_pct": 2.7,
        },
        "social_finance_flow_h1": {
            "total_trillion_yuan": 20.84,
            "yoy_less_trillion_yuan": 2.02,
            "rmb_loans_to_real_economy_trillion_yuan": 10.76,
            "rmb_loans_yoy_less_trillion_yuan": 1.98,
            "foreign_currency_loans_rmb_billion_yuan": 160.9,
            "foreign_currency_loans_yoy_more_billion_yuan": 224.7,
            "entrusted_loans_billion_yuan": -78.8,
            "trust_loans_billion_yuan": -44.6,
            "undiscounted_bank_acceptance_billion_yuan": -125.6,
            "enterprise_bonds_trillion_yuan": 2.07,
            "enterprise_bonds_yoy_more_billion_yuan": 916.7,
            "government_bonds_trillion_yuan": 6.44,
            "government_bonds_yoy_less_trillion_yuan": 1.22,
            "domestic_equity_financing_billion_yuan": 293.3,
        },
        "money": {
            "m2_balance_trillion_yuan": 356.71,
            "m2_yoy": 8.0,
            "m1_balance_trillion_yuan": 118.48,
            "m1_yoy": 4.0,
            "m0_balance_trillion_yuan": 14.74,
            "m0_yoy": 11.8,
            "m2_m1_gap_pct_points": 4.0,
            "cash_injection_h1_billion_yuan": 641.7,
        },
        "deposits": {
            "rmb_deposit_balance_trillion_yuan": 346.44,
            "rmb_deposit_yoy": 8.2,
            "rmb_deposit_increase_h1_trillion_yuan": 17.76,
            "household_deposit_increase_h1_trillion_yuan": 7.58,
            "nonfinancial_corporate_deposit_increase_h1_trillion_yuan": 3.20,
            "fiscal_deposit_increase_h1_billion_yuan": 971.5,
            "nonbank_financial_deposit_increase_h1_trillion_yuan": 4.65,
        },
        "loans": {
            "rmb_loan_balance_trillion_yuan": 282.63,
            "rmb_loan_yoy": 5.2,
            "rmb_loan_increase_h1_trillion_yuan": 10.72,
            "household_loans_h1_billion_yuan": -366.8,
            "household_short_term_loans_h1_billion_yuan": -588.1,
            "household_medium_long_term_loans_h1_billion_yuan": 221.2,
            "corporate_loans_h1_trillion_yuan": 11.13,
            "corporate_short_term_loans_h1_trillion_yuan": 4.59,
            "corporate_medium_long_term_loans_h1_trillion_yuan": 5.55,
            "bill_financing_h1_billion_yuan": 814.3,
            "nonbank_financial_loans_h1_billion_yuan": -422.3,
        },
        "rates_fx": {
            "interbank_call_rate_june_pct": 1.41,
            "pledged_repo_rate_june_pct": 1.43,
            "fx_reserves_trillion_usd": 3.42,
            "usd_cny_month_end": 6.8109,
        },
        "interpretation_points": [
            "社融存量同比7.4%，上半年新增20.84万亿元且同比少2.02万亿元，信用扩张边际放慢。",
            "人民币贷款对社融新增的拖累最大，上半年对实体人民币贷款同比少增1.98万亿元。",
            "企业债同比多增、政府债同比少增，直接融资修复但财政融资节奏不再单向抬升。",
            "M2同比8.0%、M1同比4.0%，剪刀差4.0个百分点，货币活化弱于总量扩张。",
            "居民贷款上半年净减少3668亿元，居民部门仍在去杠杆或低杠杆扩张。",
        ],
    }
}


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _load_private_data_sources() -> dict[str, Any]:
    if not PRIVATE_DATA_SOURCES.exists():
        return {}
    try:
        # V12.3 密钥治理: env:NAME 占位符 → secret_loader 解析(环境变量/.env.secrets)
        from secret_loader import load_config_with_secrets  # noqa: PLC0415
        return load_config_with_secrets(PRIVATE_DATA_SOURCES)
    except Exception:
        return {}


def _cache_path(name: str, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
    return CACHE_DIR / f"{name}-{safe}.json"


def _read_json_cache(name: str, key: str, max_age_seconds: int) -> dict[str, Any] | None:
    p = _cache_path(name, key)
    if not p.exists() or time.time() - p.stat().st_mtime > max_age_seconds:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_cache(name: str, key: str, data: dict[str, Any]) -> None:
    p = _cache_path(name, key)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


def get_tushare_pro() -> Any | None:
    cfg = _load_private_data_sources().get("tushare_pro") or {}
    token = str(cfg.get("api_key") or cfg.get("token") or "")
    if not token:
        return None
    try:
        import tushare as ts
        return ts.pro_api(token)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    """D1 收敛: 复用 quant_system.utils.safe_float（import 失败时保留原实现）。

    utils 参数映射: default=None（None/空/失败→None）、clean_percent/clean_commas=False
    （原实现不清洗）、finite=True（nan/inf→None，对齐原 pd.isna 的 NaN→None 语义并
    纳入 utils 非有限值规范）、allow_bool=True（原 pd.isna(True)=False，bool 按数值
    参与转换，对齐 float(True)=1.0）。
    """
    if _safe_float_impl is not None:
        return _safe_float_impl(
            value,
            default=None,
            clean_percent=False,
            clean_commas=False,
            finite=True,
            allow_bool=True,
        )
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception as e:
        logging.getLogger(__name__).error(f"[data_sources] 操作失败: {e}", exc_info=True)
    try:
        return float(value)
    except Exception:
        return None


def _pick_col(columns: Any, names: list[str]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _call_akshare(label: str, fn: Any, cache_name: str | None = None) -> tuple[Any | None, str | None]:
    def _wrapped() -> Any:
        if cached_fetch is not None and cache_name:
            return cached_fetch(fn, ttl_seconds=FETCH_CACHE_TTL, cache_name=cache_name)
        return fn()

    try:
        return _timeout_call(_wrapped, timeout=FETCH_TIMEOUT), None
    except TimeoutError:
        return None, f"{label}: timed out after {FETCH_TIMEOUT}s"
    except Exception as exc:
        return None, f"{label}: {exc!r}"


def normalize_index_symbol(symbol: str) -> str:
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz")):
        return s
    if s.startswith(("0", "3")):
        return f"sz{s}" if s.startswith("399") else f"sh{s}"
    return s


def index_symbol_to_ts_code(symbol: str) -> str | None:
    normalized = normalize_index_symbol(symbol)
    if normalized.startswith("sh"):
        return normalized[2:] + ".SH"
    if normalized.startswith("sz"):
        return normalized[2:] + ".SZ"
    return None


def index_symbol_to_yahoo(symbol: str) -> str | None:
    normalized = normalize_index_symbol(symbol)
    if normalized.startswith("sh"):
        return normalized[2:] + ".SS"
    if normalized.startswith("sz"):
        return normalized[2:] + ".SZ"
    return None


def index_symbol_to_alpha(symbol: str) -> str | None:
    yahoo = index_symbol_to_yahoo(symbol)
    return yahoo


def index_symbol_to_finnhub(symbol: str) -> str | None:
    # Finnhub's free plan has uneven China index coverage. Use the common Yahoo-style
    # exchange suffix first and let the API return a structured empty result if absent.
    return index_symbol_to_yahoo(symbol)


def _normalize_daily_frame(df: Any) -> Any:
    import pandas as pd

    date_col = _pick_col(df.columns, ["date", "日期"])
    open_col = _pick_col(df.columns, ["open", "开盘"])
    high_col = _pick_col(df.columns, ["high", "最高"])
    low_col = _pick_col(df.columns, ["low", "最低"])
    close_col = _pick_col(df.columns, ["close", "收盘"])
    volume_col = _pick_col(df.columns, ["volume", "成交量"])
    amount_col = _pick_col(df.columns, ["amount", "成交额"])
    if not all([date_col, open_col, high_col, low_col, close_col]):
        raise ValueError(f"daily index columns not recognized: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    if volume_col:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    if amount_col:
        out["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date")


def _fetch_yahoo_index_daily(symbol: str) -> tuple[Any | None, str | None]:
    import pandas as pd
    import requests

    yahoo = index_symbol_to_yahoo(symbol)
    if not yahoo:
        return None, "Yahoo: unsupported symbol"
    cache_key = f"{yahoo}-{date.today():%Y%m%d}"
    cached = _read_json_cache("yahoo_index_daily", cache_key, 12 * 3600)
    if cached and cached.get("rows"):
        return pd.DataFrame(cached["rows"]), None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}"
    params = {"period1": 631123200, "period2": int(time.time()) + 86400, "interval": "1d", "events": "history"}
    try:
        resp = requests.get(url, params=params, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        result = ((resp.json().get("chart") or {}).get("result") or [None])[0]
        if not result:
            return None, "Yahoo: empty result"
        ts = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        rows = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        })
        rows = rows.dropna(subset=["date", "close"]).sort_values("date")
        if rows.empty:
            return None, "Yahoo: no valid rows"
        _write_json_cache("yahoo_index_daily", cache_key, {"rows": rows.to_dict(orient="records")})
        return rows, None
    except Exception as exc:
        return None, f"Yahoo chart({yahoo}): {repr(exc)[:180]}"


def _fetch_alpha_vantage_index_daily(symbol: str) -> tuple[Any | None, str | None]:
    import pandas as pd
    import requests

    cfg = _load_private_data_sources().get("alpha_vantage") or {}
    token = str(cfg.get("api_key") or cfg.get("token") or "")
    av_symbol = index_symbol_to_alpha(symbol)
    if not token or not av_symbol:
        return None, "Alpha Vantage: missing token or unsupported symbol"
    cache_key = f"{av_symbol}-{date.today():%Y%m%d}"
    cached = _read_json_cache("alpha_vantage_index_daily", cache_key, 24 * 3600)
    if cached and cached.get("rows"):
        return pd.DataFrame(cached["rows"]), None
    params = {"function": "TIME_SERIES_DAILY", "symbol": av_symbol, "outputsize": "compact", "apikey": token}
    try:
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        series = payload.get("Time Series (Daily)") or {}
        if not series:
            msg = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "empty series"
            return None, f"Alpha Vantage({av_symbol}): {msg}"
        rows = []
        for day, values in series.items():
            rows.append({
                "date": day,
                "open": values.get("1. open"),
                "high": values.get("2. high"),
                "low": values.get("3. low"),
                "close": values.get("4. close"),
                "volume": values.get("5. volume"),
            })
        df = _normalize_daily_frame(pd.DataFrame(rows))
        if df.empty:
            return None, f"Alpha Vantage({av_symbol}): no valid rows"
        _write_json_cache("alpha_vantage_index_daily", cache_key, {"rows": df.to_dict(orient="records")})
        return df, None
    except Exception as exc:
        return None, f"Alpha Vantage({av_symbol}): {repr(exc)[:180]}"


def _fetch_finnhub_index_daily(symbol: str) -> tuple[Any | None, str | None]:
    import pandas as pd
    import requests

    cfg = _load_private_data_sources().get("finnhub") or {}
    token = str(cfg.get("api_key") or cfg.get("token") or "")
    fh_symbol = index_symbol_to_finnhub(symbol)
    if not token or not fh_symbol:
        return None, "Finnhub: missing token or unsupported symbol"
    cache_key = f"{fh_symbol}-{date.today():%Y%m%d}"
    cached = _read_json_cache("finnhub_index_daily", cache_key, 24 * 3600)
    if cached and cached.get("rows"):
        return pd.DataFrame(cached["rows"]), None
    params = {"symbol": fh_symbol, "resolution": "D", "from": 631123200, "to": int(time.time()), "token": token}
    try:
        resp = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("s") != "ok":
            return None, f"Finnhub candle({fh_symbol}): status={payload.get('s')}"
        rows = pd.DataFrame({
            "date": pd.to_datetime(payload.get("t") or [], unit="s"),
            "open": payload.get("o"),
            "high": payload.get("h"),
            "low": payload.get("l"),
            "close": payload.get("c"),
            "volume": payload.get("v"),
        })
        rows = rows.dropna(subset=["date", "close"]).sort_values("date")
        if rows.empty:
            return None, f"Finnhub candle({fh_symbol}): no valid rows"
        _write_json_cache("finnhub_index_daily", cache_key, {"rows": rows.to_dict(orient="records")})
        return rows, None
    except Exception as exc:
        return None, f"Finnhub candle({fh_symbol}): {repr(exc)[:180]}"


def _fetch_efinance_index_daily(symbol: str) -> tuple[Any | None, str | None]:
    try:
        import efinance as ef
    except Exception:
        return None, "Efinance: package not installed"
    code = normalize_index_symbol(symbol)[2:] if normalize_index_symbol(symbol).startswith(("sh", "sz")) else str(symbol)
    try:
        df = ef.stock.get_quote_history(code)
        if df is None or df.empty:
            return None, f"Efinance({code}): empty"
        rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
        daily = _normalize_daily_frame(df.rename(columns=rename))
        if daily.empty:
            return None, f"Efinance({code}): no valid rows"
        return daily, None
    except Exception as exc:
        return None, f"Efinance({code}): {repr(exc)[:180]}"


def _weekly_from_daily(daily: Any) -> Any:
    d = daily.set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    if "amount" in d.columns:
        agg["amount"] = "sum"
    weekly = d.resample("W-FRI").agg(agg).dropna(subset=["close"])
    weekly["ret"] = weekly["close"].pct_change()
    return weekly


def fetch_a_share_index_daily(symbol: str, *, include_rows: bool = False) -> SourceResult:
    """Fetch A-share index daily bars with Eastmoney -> Tencent fallback.

    Eastmoney AkShare endpoints often fail with RemoteDisconnected in this
    environment. Tencent's index daily endpoint is slower for long histories but
    has been materially more stable, so reports should use this helper instead
    of calling stock_zh_index_daily_em directly.
    """
    import akshare as ak

    normalized = normalize_index_symbol(symbol)
    failures: list[str] = []
    for source_name, fn, cache_name in [
        (f"AkShare stock_zh_index_daily_em({normalized})", lambda: ak.stock_zh_index_daily_em(symbol=normalized), f"stock_zh_index_daily_em:{normalized}"),
        (f"AkShare stock_zh_index_daily_tx({normalized})", lambda: ak.stock_zh_index_daily_tx(symbol=normalized), f"stock_zh_index_daily_tx:{normalized}"),
    ]:
        raw, err = _call_akshare(source_name, fn, cache_name=cache_name)
        if raw is None:
            failures.append(err or source_name)
            continue
        try:
            daily = _normalize_daily_frame(raw)
        except Exception as exc:
            failures.append(f"{source_name} normalize: {exc!r}")
            continue
        data: dict[str, Any] = {
            "symbol": normalized,
            "start": str(daily["date"].min().date()),
            "end": str(daily["date"].max().date()),
            "rows": int(len(daily)),
            "latest": daily.tail(1).to_dict(orient="records")[0],
            "failures_before_success": failures,
        }
        if include_rows:
            data["rows_data"] = daily.to_dict(orient="records")
        return SourceResult("a_share_index_daily", True, data, source_name, "Eastmoney primary; Tencent fallback")
    pro = get_tushare_pro()
    if pro is not None:
        ts_code = index_symbol_to_ts_code(normalized)
        if ts_code:
            try:
                raw = pro.index_daily(ts_code=ts_code, start_date="19900101", end_date=date.today().strftime("%Y%m%d"))
                if raw is None or raw.empty:
                    failures.append(f"Tushare index_daily({ts_code}): empty")
                else:
                    raw = raw.rename(columns={"trade_date": "date", "vol": "volume"})
                    daily = _normalize_daily_frame(raw)
                    data = {
                        "symbol": normalized,
                        "start": str(daily["date"].min().date()),
                        "end": str(daily["date"].max().date()),
                        "rows": int(len(daily)),
                        "latest": daily.tail(1).to_dict(orient="records")[0],
                        "failures_before_success": failures,
                    }
                    if include_rows:
                        data["rows_data"] = daily.to_dict(orient="records")
                    return SourceResult("a_share_index_daily", True, data, f"Tushare index_daily({ts_code})", "Tushare fallback for index daily bars")
            except Exception as exc:
                failures.append(f"Tushare index_daily({ts_code}): {repr(exc)[:180]}")
    for source_name, fn in [
        ("Yahoo chart API", lambda: _fetch_yahoo_index_daily(normalized)),
        ("Alpha Vantage daily", lambda: _fetch_alpha_vantage_index_daily(normalized)),
        ("Finnhub candle", lambda: _fetch_finnhub_index_daily(normalized)),
        ("Efinance quote history", lambda: _fetch_efinance_index_daily(normalized)),
    ]:
        try:
            daily, err = fn()
        except Exception as exc:
            failures.append(f"{source_name}: {repr(exc)[:180]}")
            continue
        if daily is None:
            failures.append(err or f"{source_name}: failed")
            continue
        try:
            daily = _normalize_daily_frame(daily)
        except Exception as exc:
            failures.append(f"{source_name} normalize: {exc!r}")
            continue
        data = {
            "symbol": normalized,
            "start": str(daily["date"].min().date()),
            "end": str(daily["date"].max().date()),
            "rows": int(len(daily)),
            "latest": daily.tail(1).to_dict(orient="records")[0],
            "failures_before_success": failures,
        }
        if include_rows:
            data["rows_data"] = daily.to_dict(orient="records")
        return SourceResult("a_share_index_daily", True, data, source_name, "external market-data fallback")
    return SourceResult("a_share_index_daily", False, {"symbol": normalized, "failures": failures}, "multi-source", "all index daily sources failed")


def fetch_a_share_index_weekly(symbol: str, *, ma_windows: list[int] | None = None, include_rows: bool = False) -> SourceResult:
    daily_res = fetch_a_share_index_daily(symbol, include_rows=True)
    if not daily_res.ok:
        return SourceResult("a_share_index_weekly", False, daily_res.data, daily_res.source, daily_res.note)
    import pandas as pd

    daily = pd.DataFrame(daily_res.data["rows_data"])
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = _weekly_from_daily(daily)
    for window in ma_windows or []:
        weekly[f"ma{window}"] = weekly["close"].rolling(window, min_periods=window).mean()
    latest = weekly.tail(1).to_dict(orient="records")[0]
    data = {
        "symbol": daily_res.data["symbol"],
        "daily_source": daily_res.source,
        "daily_start": daily_res.data["start"],
        "daily_end": daily_res.data["end"],
        "daily_rows": daily_res.data["rows"],
        "weekly_rows": int(len(weekly)),
        "week_end": str(weekly.index[-1].date()),
        "latest": latest,
        "last4_week_returns_pct": [_safe_float(x * 100) for x in weekly["ret"].tail(4).tolist()],
        "last12_return_pct": _safe_float((weekly["close"].iloc[-1] / weekly["close"].iloc[-12] - 1) * 100) if len(weekly) >= 12 else None,
        "failures_before_success": daily_res.data.get("failures_before_success", []),
    }
    if include_rows:
        data["rows_data"] = weekly.reset_index().to_dict(orient="records")
    return SourceResult("a_share_index_weekly", True, data, daily_res.source, "daily bars resampled to W-FRI")


def fetch_tencent_index_spot(symbol: str) -> SourceResult:
    """Fetch one index spot quote from Tencent as a fallback for index spot EM."""
    import re
    import requests

    normalized = normalize_index_symbol(symbol)
    url = f"https://qt.gtimg.cn/q={normalized}"
    try:
        resp = requests.get(url, timeout=12)
        resp.encoding = "gbk"
        match = re.search(r'"([^"]+)"', resp.text.strip())
        if not match:
            raise ValueError(resp.text[:160])
        parts = match.group(1).split("~")
        prev_close = _safe_float(parts[4] if len(parts) > 4 else None)
        close = _safe_float(parts[3] if len(parts) > 3 else None)
        pct = (close / prev_close - 1) * 100 if close is not None and prev_close else None
        return SourceResult(
            "tencent_index_spot",
            True,
            {
                "symbol": normalized,
                "name": parts[1] if len(parts) > 1 else normalized,
                "close": close,
                "prev_close": prev_close,
                "pct_change": _safe_float(pct),
                "high": _safe_float(parts[33] if len(parts) > 33 else None),
                "low": _safe_float(parts[34] if len(parts) > 34 else None),
                "volume": _safe_float(parts[6] if len(parts) > 6 else None),
            },
            url,
            "Tencent quote fallback for one index",
        )
    except Exception as exc:
        return SourceResult("tencent_index_spot", False, {"symbol": normalized}, url, repr(exc))


# D7收敛: 与 a_share_daily_report.fetch_spot_tencent 同源（qt.gtimg.cn+baostock）但返回结构不同
# (SourceResult vs (DataFrame, SourceStatus))、重试/超时参数不同，签名不兼容，保留双实现。
def fetch_tencent_a_share_spot(trade_date: str | None = None, *, batch_size: int = 80, include_rows: bool = False) -> SourceResult:
    """Fetch broad A-share spot quotes through Tencent with baostock stock list.

    This is the main overseas-network fallback for AkShare Eastmoney full-market
    quotes. It is sufficient for breadth, equal-weight return and turnover, and
    avoids relying on stock_zh_a_spot_em as the only source.
    """
    import time
    import urllib.request

    import baostock as bs
    import pandas as pd

    t0 = time.time()
    day = trade_date or date.today().strftime("%Y%m%d")
    day_dash = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    rows: list[dict[str, Any]] = []
    lg = bs.login()
    try:
        if lg.error_code != "0":
            return SourceResult("tencent_a_share_spot", False, {"date": day}, "baostock.query_all_stock", lg.error_msg)
        rs = bs.query_all_stock(day=day_dash)
        codes: list[str] = []
        while rs.next():
            rec = dict(zip(rs.fields, rs.get_row_data()))
            code = str(rec.get("code") or "")
            if not (code.startswith("sh.6") or code.startswith("sz.00") or code.startswith("sz.30") or code.startswith("bj.")):
                continue
            prefix, num = code.split(".", 1)
            codes.append(prefix + num)
    finally:
        try:
            bs.logout()
        except Exception as e:
            logging.getLogger(__name__).error(f"[data_sources] 操作失败: {e}", exc_info=True)
    if not codes:
        return SourceResult("tencent_a_share_spot", False, {"date": day}, "baostock.query_all_stock", "empty stock list")

    errors: list[str] = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        url = "https://qt.gtimg.cn/q=" + ",".join("s_" + c for c in batch)
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("gbk", "ignore")
        except Exception as exc:
            errors.append(repr(exc)[:160])
            continue
        for item in text.split(";\n"):
            item = item.strip().rstrip(";")
            if '=\"' not in item:
                continue
            quote = item.split('="', 1)[1].rstrip('"')
            parts = quote.split("~")
            if len(parts) < 8:
                continue
            market = item.split("=", 1)[0].split("s_", 1)[-1][:2].lower()
            code_num = parts[2]
            pct = _safe_float(parts[5])
            if pct is None:
                continue
            rows.append(
                {
                    "代码": f"{market}{code_num}",
                    "名称": parts[1],
                    "最新价": _safe_float(parts[3]),
                    "涨跌幅": pct,
                    "成交量": (_safe_float(parts[6]) or 0.0) * 100,
                    "成交额": (_safe_float(parts[7]) or 0.0) * 10000,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return SourceResult(
            "tencent_a_share_spot",
            False,
            {"date": day, "batch_errors": errors[:5]},
            "https://qt.gtimg.cn + baostock.query_all_stock",
            "empty quotes",
        )
    data: dict[str, Any] = {
        "date": day,
        "rows": int(len(df)),
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "batch_errors": errors[:5],
    }
    if include_rows:
        data["rows_data"] = df.to_dict(orient="records")
    return SourceResult(
        "tencent_a_share_spot",
        True,
        data,
        "https://qt.gtimg.cn + baostock.query_all_stock",
        "Tencent batch quote fallback for overseas networks",
    )


def fetch_index_valuation_csindex(symbol: str = "000015", keywords: list[str] | None = None) -> SourceResult:
    """Find target index valuation rows in China Securities Index tables."""
    import akshare as ak
    import pandas as pd

    keywords = keywords or [symbol, "上证红利", "红利"]
    data, err = _call_akshare("AkShare stock_zh_index_value_csindex", lambda: ak.stock_zh_index_value_csindex(), cache_name="stock_zh_index_value_csindex")
    if data is None:
        return SourceResult("index_valuation_csindex", False, {"symbol": symbol}, "AkShare stock_zh_index_value_csindex", err or "call failed")
    mask = pd.Series(False, index=data.index)
    for col in ["指数代码", "指数中文全称", "指数中文简称", "指数英文简称"]:
        if col in data.columns:
            s = data[col].astype(str)
            for kw in keywords:
                mask = mask | s.str.contains(str(kw), case=False, na=False)
    rows = data[mask].copy()
    if rows.empty:
        return SourceResult(
            "index_valuation_csindex",
            False,
            {"symbol": symbol, "columns": list(map(str, data.columns)), "matched_rows": []},
            "AkShare stock_zh_index_value_csindex",
            "valuation table returned but target index was not matched",
        )
    if "日期" in rows.columns:
        rows["日期"] = pd.to_datetime(rows["日期"], errors="coerce")
        rows = rows.sort_values("日期", ascending=False)
    return SourceResult(
        "index_valuation_csindex",
        True,
        {"symbol": symbol, "rows": rows.head(10).astype(str).to_dict(orient="records"), "columns": list(map(str, data.columns))},
        "AkShare stock_zh_index_value_csindex",
        "matched by code/name keywords",
    )


# D7收敛: 与 a_share_daily_report.breadth_metrics 同名异口径（此处不去ST、无全A等权/涨停占比等硬规则字段），
# a_share_daily_report 为真源，保留两套口径，禁止把本函数当作硬规则替代。
def fetch_a_share_market_breadth(max_cache_age_seconds: int = 6 * 3600) -> SourceResult:
    import akshare as ak
    import pandas as pd

    failures: list[str] = []
    cache_key = date.today().strftime("%Y%m%d")
    cached = _read_json_cache("a_share_market_breadth", cache_key, max_cache_age_seconds)
    if cached:
        cached["note"] = str(cached.get("note") or "") + "; loaded from local cache"
        return SourceResult(**cached)
    source = ""
    df = None
    for source_name, fn, cache_name in [
        ("AkShare stock_zh_a_spot_sina", lambda: ak.stock_zh_a_spot(), "stock_zh_a_spot"),
        ("AkShare stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em(), "stock_zh_a_spot_em"),
    ]:
        candidate, err = _call_akshare(source_name, fn, cache_name=cache_name)
        if candidate is None:
            failures.append(err or f"{source_name} failed")
            continue
        df = candidate
        source = source_name
        break
    if df is None:
        fallback = fetch_tencent_a_share_spot(include_rows=True)
        if not fallback.ok:
            return SourceResult(
                "a_share_market_breadth",
                False,
                {"failures": failures, "fallback": fallback.to_dict()},
                "AkShare Sina/EM; Tencent fallback",
                "all full-market quote sources failed",
            )
        df = pd.DataFrame(fallback.data["rows_data"])
        source = fallback.source
    rename = {
        "涨跌百分比": "涨跌幅",
        "今收盘价": "最新价",
        "成交金额": "成交额",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    pct_col = _pick_col(df.columns, ["涨跌幅"])
    amount_col = _pick_col(df.columns, ["成交额", "成交金额"])
    code_col = _pick_col(df.columns, ["代码"])
    if not pct_col:
        return SourceResult("a_share_market_breadth", False, {"columns": list(map(str, df.columns)), "failures": failures}, source, "涨跌幅 column missing")
    pct = pd.to_numeric(df[pct_col], errors="coerce")
    valid = pct.notna()
    data = {
        "date": str(date.today()),
        "count": int(valid.sum()),
        "advancing_count": int((pct > 0).sum()),
        "flat_count": int((pct == 0).sum()),
        "declining_count": int((pct < 0).sum()),
        "advancing_pct": _safe_float((pct > 0).mean() * 100),
        "flat_pct": _safe_float((pct == 0).mean() * 100),
        "declining_pct": _safe_float((pct < 0).mean() * 100),
        "median_pct_change": _safe_float(pct.median()),
        "failures_before_success": failures,
    }
    if amount_col:
        amount = pd.to_numeric(df[amount_col], errors="coerce")
        data["turnover_billion"] = _safe_float(amount.sum() / 1e8)
    if code_col:
        code = df[code_col].astype(str)
        filtered = df[~code.str.startswith(("8", "4"), na=False)]
        fpct = pd.to_numeric(filtered[pct_col], errors="coerce")
        data["non_bse_count"] = int(fpct.notna().sum())
        data["non_bse_advancing_pct"] = _safe_float((fpct > 0).mean() * 100)
    result = SourceResult("a_share_market_breadth", True, data, source, "real-time A-share breadth with Sina/EM/Tencent fallback and local cache")
    _write_json_cache("a_share_market_breadth", cache_key, result.to_dict())
    return result


def fetch_industry_board_snapshot() -> SourceResult:
    import akshare as ak

    funcs = [
        ("AkShare stock_board_industry_name_em", lambda: ak.stock_board_industry_name_em(), "stock_board_industry_name_em"),
        ("AkShare stock_board_concept_name_em", lambda: ak.stock_board_concept_name_em(), "stock_board_concept_name_em"),
        ("AkShare stock_board_industry_summary_ths", lambda: ak.stock_board_industry_summary_ths(), "stock_board_industry_summary_ths"),
    ]
    failures: list[str] = []
    for source_name, fn, cache_name in funcs:
        df, err = _call_akshare(source_name, fn, cache_name=cache_name)
        if df is None:
            failures.append(err or source_name)
            continue
        name_col = _pick_col(df.columns, ["板块名称", "名称", "板块"])
        pct_col = _pick_col(df.columns, ["涨跌幅"])
        amount_col = _pick_col(df.columns, ["成交额", "总成交额"])
        rows = []
        wanted = ["银行", "煤炭", "石油", "石化", "交通", "运输", "公用", "电力", "港口", "高速"]
        if name_col:
            sub = df[df[name_col].astype(str).apply(lambda s: any(w in s for w in wanted))]
            for _, row in sub.head(30).iterrows():
                rows.append(
                    {
                        "name": str(row.get(name_col)),
                        "pct_change": _safe_float(row.get(pct_col)) if pct_col else None,
                        "amount_billion": _safe_float(_safe_float(row.get(amount_col)) / 1e8) if amount_col and _safe_float(row.get(amount_col)) is not None else None,
                    }
                )
        return SourceResult(
            "industry_board_snapshot",
            True,
            {"columns": list(map(str, df.columns)), "dividend_related_rows": rows, "failures_before_success": failures},
            source_name,
            "industry/concept board snapshot; rows filtered for dividend-related sectors",
        )
    return SourceResult("industry_board_snapshot", False, {"failures": failures}, "akshare", "all industry board sources failed")


def fetch_dividend_weekly_inputs(symbol: str = "sh000015") -> SourceResult:
    """Composite input bundle for the dividend-index weekly cron."""
    weekly = fetch_a_share_index_weekly(symbol, ma_windows=[750], include_rows=False)
    spot = fetch_tencent_index_spot(symbol)
    valuation = fetch_index_valuation_csindex("000015", ["000015", "上证红利"])
    breadth = fetch_a_share_market_breadth()
    industry = fetch_industry_board_snapshot()
    components = {
        "weekly": weekly.to_dict(),
        "spot": spot.to_dict(),
        "valuation": valuation.to_dict(),
        "market_breadth": breadth.to_dict(),
        "industry": industry.to_dict(),
    }
    ok = weekly.ok or spot.ok
    return SourceResult(
        "dividend_weekly_inputs",
        ok,
        components,
        "scripts.data_sources",
        "use this helper in the dividend weekly cron; do not call fragile AkShare endpoints directly in cron prompts",
    )


def get_private_data_source(name: str) -> SourceResult:
    """Return private data-source availability without exposing the raw secret."""
    cfg = _load_private_data_sources().get(name) or {}
    api_key = str(cfg.get("api_key") or "")
    data: dict[str, Any] = {}
    for k, v in cfg.items():
        if k == "api_key":
            continue
        if isinstance(v, str) and ("token=" in v.lower() or "apikey=" in v.lower() or "api_key=" in v.lower()):
            data[k] = v.split("?")[0] + "?***"
        else:
            data[k] = v
    if api_key:
        data["api_key_masked"] = _mask_secret(api_key)
    return SourceResult(
        name=name,
        ok=bool(api_key),
        data=data,
        source=str(PRIVATE_DATA_SOURCES),
        note="private data-source config loaded; raw secret is not returned",
    )


def get_nasdaq_api() -> SourceResult:
    return get_private_data_source("nasdaq_api")


def get_cpi_ppi_live() -> SourceResult:
    """Live CPI/PPI data from AkShare (国家统计局官方发布).

    Uses:
      - macro_china_cpi() → 全国CPI当月/同比/环比/累计
      - macro_china_ppi() → PPI当月/同比
    Falls back to the hardcoded verification snapshot if API fails.
    """
    import pandas as pd
    try:
        import akshare as ak
        cpi_df = _cached_ak(ak.macro_china_cpi)
        ppi_df = _cached_ak(ak.macro_china_ppi)

        if cpi_df is None or cpi_df.empty:
            raise ValueError("CPI AkShare returned empty")
        if ppi_df is None or ppi_df.empty:
            raise ValueError("PPI AkShare returned empty")

        # Parse "2026年06月份" → "2026-06"
        def _period(v: str) -> str:
            import re
            m = re.search(r"(\d{4})年(\d{2})月份?", str(v))
            if m:
                return f"{m.group(1)}-{m.group(2)}"
            return str(v)[:7]

        def _sf(v) -> float | None:
            try:
                return round(float(v), 2)
            except Exception:
                return None

        latest_cpi = cpi_df.iloc[0]
        latest_ppi = ppi_df.iloc[0]

        period_live = _period(latest_cpi.get("月份", ""))
        # pick the most recent with data match
        ppi_latest = ppi_df[ppi_df["月份"] == latest_cpi["月份"]]
        if ppi_latest.empty:
            ppi_latest = ppi_df.iloc[:1]
        ppi_row = ppi_latest.iloc[0]

        data = {
            "period": period_live,
            "source": "akshare macro_china_cpi / macro_china_ppi",
            "cpi": {
                "index": _sf(latest_cpi.get("全国-当月")),
                "yoy": _sf(latest_cpi.get("全国-同比增长", latest_cpi.get("全国-同比"))),
                "mom": _sf(latest_cpi.get("全国-环比增长", latest_cpi.get("全国-环比"))),
                "ytd": _sf(latest_cpi.get("全国-累计")),
                "urban_yoy": _sf(latest_cpi.get("城市-同比增长")),
                "rural_yoy": _sf(latest_cpi.get("农村-同比增长")),
            },
            "ppi": {
                "index": _sf(ppi_row.get("当月")),
                "yoy": _sf(ppi_row.get("当月同比增长", ppi_row.get("同比增长"))),
                "ytd": _sf(ppi_row.get("累计")),
            },
        }
        return SourceResult(
            name="cpi_ppi",
            ok=True,
            data=data,
            source="akshare",
            note=f"live CPI/PPI from NBS via AkShare for {period_live}",
        )
    except Exception as exc:
        return SourceResult(
            name="cpi_ppi",
            ok=False,
            data={"live_error": repr(exc)[:500]},
            source="akshare",
            note=f"live fetch failed: {exc!r}; falling back to archive snapshot",
        )


def get_cpi_ppi(period: str | None = None) -> SourceResult:
    """Return CPI/PPI data — try live AkShare first, then verified fallback."""
    live = get_cpi_ppi_live()
    if live.ok:
        return live
    selected_period = period or sorted(CPI_PPI_FALLBACKS)[-1]
    data = CPI_PPI_FALLBACKS.get(selected_period)
    if data:
        return SourceResult(
            name="cpi_ppi",
            ok=True,
            data=data,
            source=data["source_url"],
            note=f"fallback after live failed: {live.note}",
        )
    return SourceResult(
        name="cpi_ppi",
        ok=False,
        data={"live_error": live.note},
        source="",
        note=f"live + fallback both failed for period={selected_period}",
    )


def get_pmi_live() -> SourceResult:
    """Live PMI data from AkShare (国家统计局官方发布).

    Uses:
      - macro_china_pmi() → 制造业/非制造业 PMI 指数及同比
      - macro_china_non_man_pmi() → 补充非制造业 PMI 历史序列
    Falls back to desktop archive if API fails.
    """
    try:
        import akshare as ak
        pmi_df = _cached_ak(ak.macro_china_pmi)
        if pmi_df is None or pmi_df.empty:
            raise ValueError("PMI AkShare returned empty")

        def _period(v: str) -> str:
            import re
            m = re.search(r"(\d{4})年(\d{2})月份?", str(v))
            if m:
                return f"{m.group(1)}-{m.group(2)}"
            return str(v)[:7]

        def _sf(v) -> float | None:
            try:
                return round(float(v), 2)
            except Exception:
                return None

        latest = pmi_df.iloc[0]
        period_live = _period(latest.get("月份", ""))

        mfg = _sf(latest.get("制造业-指数"))
        mfg_yoy = _sf(latest.get("制造业-同比增长"))
        non_mfg = _sf(latest.get("非制造业-指数"))
        non_mfg_yoy = _sf(latest.get("非制造业-同比增长"))

        data = {
            "period": period_live,
            "source": "akshare macro_china_pmi",
            "manufacturing_pmi": mfg,
            "manufacturing_pmi_yoy": mfg_yoy,
            "non_manufacturing_pmi": non_mfg,
            "non_manufacturing_pmi_yoy": non_mfg_yoy,
            "interpretation_points": [
                f"制造业PMI {mfg}（{'扩张' if mfg and mfg >= 50 else '收缩'}），"
                f"同比 {mfg_yoy:+.1f}%" if mfg_yoy else "",
                f"非制造业PMI {non_mfg}（{'扩张' if non_mfg and non_mfg >= 50 else '收缩'}），"
                f"同比 {non_mfg_yoy:+.1f}%" if non_mfg_yoy else "",
            ],
        }
        return SourceResult(
            name="pmi",
            ok=True,
            data=data,
            source="akshare",
            note=f"live PMI from NBS via AkShare for {period_live}",
        )
    except Exception as exc:
        return SourceResult(
            name="pmi",
            ok=False,
            data={"live_error": repr(exc)[:500]},
            source="akshare",
            note=f"live fetch failed: {exc!r}; falling back to archive",
        )


def get_pmi(period: str | None = None) -> SourceResult:
    """Return PMI data — try live AkShare first, then desktop archive."""
    live = get_pmi_live()
    if live.ok:
        return live
    return _latest_markdown_report(
        "pmi",
        report_path("pmi", ensure_dir=False).parent,
        f"{period}_PMI.md" if period else "*_PMI.md",
    )


def _latest_markdown_report(name: str, directory: Path, pattern: str) -> SourceResult:
    """Return the latest desktop archived report as a last-resort fallback."""
    if not directory.exists():
        return SourceResult(name, False, {}, "", f"archive directory missing: {directory}")
    files = sorted(directory.glob(pattern), key=lambda p: p.name, reverse=True)
    if not files:
        return SourceResult(name, False, {}, "", f"no archived report matching {pattern} in {directory}")
    p = files[0]
    return SourceResult(
        name=name,
        ok=True,
        data={
            "archive_path": str(p),
            "content": p.read_text(encoding="utf-8", errors="replace"),
        },
        source=str(p),
        note="desktop archive markdown fallback; refresh live data before publishing if official release changed",
    )


def get_customs(period: str | None = None) -> SourceResult:
    return _latest_markdown_report(
        "customs",
        report_path("customs", ensure_dir=False).parent,
        f"{period}_海关进出口.md" if period else "*_海关进出口.md",
    )


def get_social_finance_m2(period: str | None = None) -> SourceResult:
    selected_period = period or sorted(SOCIAL_FINANCE_M2_FALLBACKS)[-1]
    data = SOCIAL_FINANCE_M2_FALLBACKS.get(selected_period)
    if data:
        return SourceResult(
            name="social_finance_m2",
            ok=False,  # W2.5 修复：硬编码快照非 live，不得 ok=True（否则 check_macro_data_sources 永远假绿）
            data=data,
            source=data["source_url"],
            note="hardcoded fallback snapshot (not live); refresh live before treating as current",
        )
    return _latest_markdown_report(
        "social_finance_m2",
        report_path("social_finance_m2", ensure_dir=False).parent,
        f"{period}_社融M2.md" if period else "*_社融M2.md",
    )


def get_international_oil(day: str | None = None) -> SourceResult:
    return _latest_markdown_report(
        "international_oil",
        report_path("international_oil", ensure_dir=False).parent,
        f"{day}_国际原油.md" if day else "*_国际原油.md",
    )


# Backward-compatible names already referenced by cron payloads.
fetch_cpi_ppi = get_cpi_ppi
fetch_pmi = get_pmi
fetch_customs = get_customs
fetch_social_finance_m2 = get_social_finance_m2
fetch_international_oil = get_international_oil


if __name__ == "__main__":
    import json
    checks = [
        get_cpi_ppi(),
        get_pmi(),
        get_customs(),
        get_social_finance_m2(),
        get_international_oil(),
    ]
    print(json.dumps([c.to_dict() for c in checks], ensure_ascii=False, indent=2))

# Official source registry for oil and US macro reports used by cron jobs.
# These functions return stable official endpoints plus archive fallbacks; report
# generators can fetch/parse the URLs they need without hard-coding sources in cron payloads.
OFFICIAL_SOURCE_URLS = {
    "opec_momr_index": "https://www.opec.org/monthly-oil-market-report.html",
    "opec_momr_latest_pdf": "https://momr.opec.org/pdf-download/",
    "eia_api_docs": "https://www.eia.gov/opendata/documentation.php",
    "eia_wpsr": "https://www.eia.gov/petroleum/supply/weekly/",
    "eia_wpsr_pdf": "https://www.eia.gov/petroleum/supply/pdf/wpsrall.pdf",
    "eia_wpsr_table1_csv": "https://ir.eia.gov/wpsr/table1.csv",
    "eia_wpsr_table4_csv": "https://ir.eia.gov/wpsr/table4.csv",
    "eia_wpsr_table9_csv": "https://ir.eia.gov/wpsr/table9.csv",
    "fred_api_docs": "https://fred.stlouisfed.org/docs/api/fred/",
    "fred_observations": "https://api.stlouisfed.org/fred/series/observations",
    "federal_reserve_releases": "https://www.federalreserve.gov/data.htm",
    "federal_reserve_h15": "https://www.federalreserve.gov/releases/h15/",
    "federal_reserve_h41": "https://www.federalreserve.gov/releases/h41/",
    "nasdaq_data_link_api": "https://data.nasdaq.com/api/v3/datasets/{dataset}.json",
    "nasdaq_data_link_docs": "https://docs.data.nasdaq.com/docs/in-depth-usage",
    "yahoo_chart_api": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "hkex_connect_shareholding": "https://www.hkexnews.hk/sdw/search/searchsdw.aspx",
    "hkex_connect_market_statistics": "https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Statistics/Hong-Kong-and-Mainland-Market-Highlights",
    "lme_market_data": "https://www.lme.com/Market-data",
    "shfe_market_data": "https://www.shfe.com.cn/statements/dataview.html",
}

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "policy_rate_upper": "DFEDTARU",
    "policy_rate_lower": "DFEDTARL",
    "us10y": "DGS10",
    "us2y": "DGS2",
    "real_10y": "DFII10",
    "breakeven_10y": "T10YIE",
    "dxy_proxy_trade_weighted_dollar": "DTWEXBGS",
    "wti_spot": "DCOILWTICO",
    "brent_spot": "DCOILBRENTEU",
}


def get_official_source_registry() -> SourceResult:
    return SourceResult(
        name="official_source_registry",
        ok=True,
        source="official-source-registry",
        data={
            "urls": OFFICIAL_SOURCE_URLS,
            "fred_series": FRED_SERIES,
            "private_sources": {
                "nasdaq_api": get_private_data_source("nasdaq_api").to_dict(),
                "tushare_pro": get_private_data_source("tushare_pro").to_dict(),
                "alpha_vantage": get_private_data_source("alpha_vantage").to_dict(),
                "finnhub": get_private_data_source("finnhub").to_dict(),
            },
            "reporting_rule": "数据源失败时要列出失败项、fallback 和可用替代口径；不能为了不报错而减少报告内容。",
        },
        note="stable official/public/private fallback registry for macro, oil, A-share, Hong Kong and cross-asset reports",
    )


def fetch_fred_graph_csv(series_id: str, *, max_age_seconds: int = 6 * 3600) -> SourceResult:
    """Fetch a FRED series through the public CSV graph endpoint.

    This endpoint is useful in cron because it does not need a FRED API key and
    provides enough history for rates/oil/dollar cross-asset context.
    """
    import pandas as pd

    series = str(series_id).strip().upper()
    if not series:
        return SourceResult("fred_graph_csv", False, {}, "FRED", "missing series id")
    cache_key = f"{series}-{date.today():%Y%m%d}"
    cached = _read_json_cache("fred_graph_csv", cache_key, max_age_seconds)
    if cached:
        return SourceResult("fred_graph_csv", True, cached, f"FRED {series}", "loaded from local cache")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        df = pd.read_csv(url, timeout=25)  # type: ignore[call-arg]
    except TypeError:
        try:
            df = pd.read_csv(url)
        except Exception as exc:
            return SourceResult("fred_graph_csv", False, {"series_id": series}, url, repr(exc)[:180])
    except Exception as exc:
        return SourceResult("fred_graph_csv", False, {"series_id": series}, url, repr(exc)[:180])
    if df.empty or series not in df.columns:
        return SourceResult("fred_graph_csv", False, {"series_id": series, "columns": list(df.columns)}, url, "empty or missing value column")
    df[series] = pd.to_numeric(df[series], errors="coerce")
    df = df.dropna(subset=[series]).tail(260)
    if df.empty:
        return SourceResult("fred_graph_csv", False, {"series_id": series}, url, "no numeric observations")
    latest = df.tail(1).to_dict(orient="records")[0]
    prev = df.tail(2).head(1).to_dict(orient="records")[0] if len(df) >= 2 else {}
    data = {
        "series_id": series,
        "latest": latest,
        "previous": prev,
        "rows": int(len(df)),
        "start": str(df.iloc[0]["observation_date"]),
        "end": str(df.iloc[-1]["observation_date"]),
        "rows_data": df.to_dict(orient="records"),
    }
    _write_json_cache("fred_graph_csv", cache_key, data)
    return SourceResult("fred_graph_csv", True, data, url, "public FRED CSV fallback")


def fetch_nasdaq_data_link_dataset(dataset: str, *, params: dict[str, Any] | None = None, max_age_seconds: int = 6 * 3600) -> SourceResult:
    """Fetch a Nasdaq Data Link dataset using the saved private key.

    Example datasets depend on the user's subscription. Keep this generic so
    reports can try a dataset and record a clear failure instead of crashing.
    """
    import requests

    cfg = _load_private_data_sources().get("nasdaq_api") or {}
    token = str(cfg.get("api_key") or "")
    ds = str(dataset).strip().strip("/")
    if not token:
        return SourceResult("nasdaq_data_link", False, {"dataset": ds}, "nasdaq_data_link", "missing nasdaq_api key")
    if not ds or "/" not in ds:
        return SourceResult("nasdaq_data_link", False, {"dataset": ds}, "nasdaq_data_link", "dataset must look like DATABASE/CODE")
    cache_key = f"{ds}-{json.dumps(params or {}, sort_keys=True, ensure_ascii=False)}-{date.today():%Y%m%d}"
    cached = _read_json_cache("nasdaq_data_link", cache_key, max_age_seconds)
    if cached:
        return SourceResult("nasdaq_data_link", True, cached, f"Nasdaq Data Link {ds}", "loaded from local cache")
    url = OFFICIAL_SOURCE_URLS["nasdaq_data_link_api"].format(dataset=ds)
    q = dict(params or {})
    q["api_key"] = token
    try:
        resp = requests.get(url, params=q, timeout=30)
        resp.raise_for_status()
        payload = resp.json().get("dataset") or resp.json()
    except Exception as exc:
        return SourceResult("nasdaq_data_link", False, {"dataset": ds, "params": params or {}}, url, repr(exc)[:220])
    data = {
        "dataset": ds,
        "name": payload.get("name") if isinstance(payload, dict) else None,
        "newest_available_date": payload.get("newest_available_date") if isinstance(payload, dict) else None,
        "oldest_available_date": payload.get("oldest_available_date") if isinstance(payload, dict) else None,
        "column_names": payload.get("column_names") if isinstance(payload, dict) else None,
        "data": (payload.get("data") if isinstance(payload, dict) else payload)[:30] if isinstance(payload.get("data") if isinstance(payload, dict) else payload, list) else payload,
    }
    _write_json_cache("nasdaq_data_link", cache_key, data)
    return SourceResult("nasdaq_data_link", True, data, url, "Nasdaq Data Link private-key fallback")


def get_cross_asset_fallback_sources() -> SourceResult:
    """Return usable cross-asset fallback map for richer reports."""
    return SourceResult(
        name="cross_asset_fallback_sources",
        ok=True,
        source="scripts.data_sources",
        data={
            "rates_and_dollar": {"primary": "FRED public CSV", "series": FRED_SERIES},
            "oil": get_international_oil_sources().to_dict(),
            "nasdaq_data_link": get_private_data_source("nasdaq_api").to_dict(),
            "global_equity_fallbacks": ["Yahoo Chart API", "Alpha Vantage", "Finnhub", "Nasdaq Data Link where subscribed"],
            "hk_h_share_fallbacks": ["HKEX SDW/Connect public pages", "Yahoo HK quote pages/API", "company exchange announcements", "local archives"],
            "metals_fallbacks": ["LME market data pages", "SHFE market data pages", "Nasdaq Data Link subscribed datasets", "local archives"],
            "quality_rule": "优先填充可验证数据；拿不到日度序列时用公开网页/本地归档/方向判断，并在报告中说明缺失项和下一步自动化源。",
        },
        note="Use this in cron reports before declaring cross-asset data unavailable.",
    )


def get_opec_momr(period: str | None = None) -> SourceResult:
    data = {
        "period": period,
        "index_url": OFFICIAL_SOURCE_URLS["opec_momr_index"],
        "latest_pdf_url": OFFICIAL_SOURCE_URLS["opec_momr_latest_pdf"],
        "appendix_hint": "Use the index page to resolve the latest monthly appendix XLSX link, e.g. momr-appendix-<month>-<year>.xlsx.",
    }
    return SourceResult(
        name="opec_momr",
        ok=True,
        source=OFFICIAL_SOURCE_URLS["opec_momr_index"],
        data=data,
        note="official OPEC Monthly Oil Market Report index; latest PDF endpoint is stable but month-specific appendix URL should be resolved from index",
    )


def get_eia_petroleum_weekly() -> SourceResult:
    data = {
        "report_url": OFFICIAL_SOURCE_URLS["eia_wpsr"],
        "pdf_url": OFFICIAL_SOURCE_URLS["eia_wpsr_pdf"],
        "tables": {
            "balance_sheet_csv": OFFICIAL_SOURCE_URLS["eia_wpsr_table1_csv"],
            "stocks_csv": OFFICIAL_SOURCE_URLS["eia_wpsr_table4_csv"],
            "prices_csv": OFFICIAL_SOURCE_URLS["eia_wpsr_table9_csv"],
        },
        "api_docs": OFFICIAL_SOURCE_URLS["eia_api_docs"],
    }
    return SourceResult(
        name="eia_petroleum_weekly",
        ok=True,
        source=OFFICIAL_SOURCE_URLS["eia_wpsr"],
        data=data,
        note="official EIA Weekly Petroleum Status Report with stable CSV table endpoints; prefer CSV tables before unofficial market APIs",
    )


def get_fed_data_sources() -> SourceResult:
    data = {
        "federal_reserve_data": OFFICIAL_SOURCE_URLS["federal_reserve_releases"],
        "h15_rates": OFFICIAL_SOURCE_URLS["federal_reserve_h15"],
        "h41_balance_sheet": OFFICIAL_SOURCE_URLS["federal_reserve_h41"],
        "fred_api_docs": OFFICIAL_SOURCE_URLS["fred_api_docs"],
        "fred_observations_endpoint": OFFICIAL_SOURCE_URLS["fred_observations"],
        "fred_series": FRED_SERIES,
    }
    return SourceResult(
        name="fed_data_sources",
        ok=True,
        source=OFFICIAL_SOURCE_URLS["federal_reserve_releases"],
        data=data,
        note="official Federal Reserve release pages plus FRED series IDs for cron macro/gold reports",
    )


def get_international_oil_sources(day: str | None = None) -> SourceResult:
    archive = get_international_oil(day)
    cross_asset_signals = {
        "framework": "intermarket-analysis + Greenspan framework",
        "oil_role": "原油同时是需求温度计、通胀输入项和成本冲击源，不能孤立解读。",
        "linked_reports": [
            "5-宏观数据周报：把原油库存、油价、美元和实际利率纳入通胀/增长/政策组合判断。",
            "4-黄金宏观周报：用油价-通胀预期-实际利率链条判断黄金弹性。",
            "A股量化日报：跟踪上游资源、油服、化工下游、航空交运、人民币和北向风险偏好映射。",
            "宏观-CPI-PPI数据解读：把原油和成品油变化映射到输入型通胀、PPI、交通燃料和利润分配。",
            "宏观-海关进出口数据解读：跟踪原油进口量价、贸易条件和能源账单变化。",
            "宏观-社融M2数据解读：结合油价冲击判断宽松空间、实体融资需求与金融条件。",
        ],
        "must_check": [
            "WTI/Brent 方向与美元指数是否背离或共振。",
            "美国名义利率、实际利率、通胀预期对油价与黄金的传导。",
            "商业原油库存、SPR、Cushing、汽油/馏分油库存是否同向验证。",
            "OPEC 供需平衡与 EIA 周度数据是否一致。",
            "A股资源链与成本端行业是否出现相反表现。",
        ],
    }
    data = {
        "day": day,
        "opec": get_opec_momr().data,
        "eia": get_eia_petroleum_weekly().data,
        "fred_series": {k: FRED_SERIES[k] for k in ["wti_spot", "brent_spot"]},
        "cross_asset_signals": cross_asset_signals,
        "archive": archive.data,
    }
    ok = bool(archive.ok)
    note = "official OPEC/EIA/FRED sources fixed; desktop archive available" if ok else "official OPEC/EIA/FRED sources fixed; desktop archive missing or empty"
    return SourceResult(name="international_oil_sources", ok=ok, source="official oil source registry", data=data, note=note)


def get_metals_api() -> SourceResult:
    """Fetch gold/silver/platinum/palladium spot prices from metals-api.com.

    Returns rates in USD per troy ounce (via inverse of API response).
    Example response:
      {"success": true, "timestamp": ..., "date": "2026-07-27",
       "rates": {"XAU": 0.000245, "XAG": 0.0171, "XPD": 0.000787, "XPT": 0.000618}}
    The rates are "per USD" (1 USD = X ounces), so we invert to get USD/oz.
    """
    cfg = _load_private_data_sources().get("metals_api") or {}
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://metals-api.com/api/latest")

    if not api_key or api_key == "YOUR_ACCESS_KEY":
        return SourceResult(
            name="metals_api", ok=False, data={},
            source="metals-api.com",
            note="请先在 private_data_sources.json 填入 metals_api.api_key",
        )

    try:
        url = f"{base_url}?access_key={api_key}&symbols=XAU,XAG,XPD,XPT"
        import requests as _req
        resp = _req.get(url, timeout=15)
        resp.raise_for_status()
        j = resp.json()
    except Exception as exc:
        return SourceResult(
            name="metals_api", ok=False, data={},
            source="metals-api.com",
            note=f"请求失败: {exc!r}",
        )

    if not j.get("success"):
        return SourceResult(
            name="metals_api", ok=False, data={},
            source="metals-api.com",
            note=f"API 返回错误: {j}",
        )

    rates = j.get("rates", {})
    prices: dict[str, float] = {}
    for metal, per_usd in rates.items():
        try:
            prices[metal] = round(1.0 / float(per_usd), 2) if float(per_usd) > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            prices[metal] = 0.0

    return SourceResult(
        name="metals_api",
        ok=True,
        data={
            "date": j.get("date", ""),
            "timestamp": j.get("timestamp", 0),
            "rates_raw": rates,
            "prices_usd_per_oz": prices,
            "XAU_USD": prices.get("XAU", 0),    # 黄金 USD/oz
            "XAG_USD": prices.get("XAG", 0),    # 白银 USD/oz
            "XPD_USD": prices.get("XPD", 0),    # 钯金 USD/oz
            "XPT_USD": prices.get("XPT", 0),    # 铂金 USD/oz
        },
        source="metals-api.com",
        note="金银铂钯现货价格（USD/盎司）",
    )


def get_oilpriceapi() -> SourceResult:
    """Fetch WTI/Brent spot prices from oilpriceapi.com.

    Uses bearer token auth via Authorization header.
    Supports codes: BRENT_CRUDE_USD, WTI_CRUDE_USD
    """
    cfg = _load_private_data_sources().get("oilpriceapi") or {}
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.oilpriceapi.com/v1")

    if not api_key:
        return SourceResult(
            name="oilpriceapi", ok=False, data={},
            source="oilpriceapi.com",
            note="API key 未配置",
        )

    import requests as _req
    headers = {"Authorization": f"Token {api_key}"}
    results = {}
    all_ok = True
    codes = ["BRENT_CRUDE_USD", "WTI_CRUDE_USD"]

    for code in codes:
        try:
            url = f"{base_url}/prices/latest?by_code={code}"
            resp = _req.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            j = resp.json()
            if j.get("status") == "success":
                results[code] = j["data"]
            else:
                results[code] = {"error": f"API status: {j.get('status')}"}
                all_ok = False
        except Exception as exc:
            results[code] = {"error": repr(exc)}
            all_ok = False

    data = {
        "prices": {},
        "changes_24h": {},
        "formatted": {},
    }
    for code, info in results.items():
        if isinstance(info, dict) and "price" in info:
            data["prices"][code] = info["price"]
            data["changes_24h"][code] = info.get("changes", {}).get("24h", {})
            data["formatted"][code] = info.get("formatted", "")
            data[f"{code}_price"] = info["price"]
            data[f"{code}_change_pct"] = info.get("changes", {}).get("24h", {}).get("percent", 0)
            data[f"{code}_collected_at"] = info.get("collected_at", "")

    return SourceResult(
        name="oilpriceapi",
        ok=all_ok and bool(data["prices"]),
        data=data,
        source="oilpriceapi.com",
        note="WTI/Brent 原油现货价格（USD/桶）",
    )


def get_yfinance_commodities() -> SourceResult:
    """Fetch commodity futures prices via yfinance (Yahoo Finance).

    Yahoo Finance futures symbols:
      GC=F 黄金, SI=F 白银, HG=F 铜, PL=F 铂金, PA=F 钯金
      CL=F WTI原油, RB=F 汽油, HO=F 柴油/燃油, NG=F 天然气

    Returns prices in USD (per troy oz for metals, per barrel for oil).
    """
    try:
        import yfinance as yf
    except ImportError:
        return SourceResult(
            name="yfinance_commodities", ok=False, data={},
            source="yfinance",
            note="yfinance 未安装",
        )

    symbols = {
        "GC=F": "黄金",
        "SI=F": "白银",
        "HG=F": "铜",
        "PL=F": "铂金",
        "PA=F": "钯金",
        "CL=F": "WTI原油",
        "NG=F": "天然气",
    }
    result: dict[str, Any] = {"prices": {}, "names": {}, "changes_24h_pct": {}}
    all_ok = True

    try:
        tickers = _timeout_call(lambda: yf.Tickers(list(symbols.keys())), timeout=FETCH_TIMEOUT*2)
        for sym, label in symbols.items():
            t = tickers.tickers.get(sym)
            if not t:
                all_ok = False
                continue
            info = t.info or {}
            price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("bid")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
            change_pct = ((price - prev_close) / prev_close * 100) if (price and prev_close and prev_close > 0) else None
            name = info.get("shortName") or label
            result["prices"][sym] = round(float(price), 4) if price else None
            result["names"][sym] = name
            result["changes_24h_pct"][sym] = round(change_pct, 2) if change_pct is not None else None
            result[f"{sym}_price"] = result["prices"][sym]
            result[f"{sym}_change_pct"] = result["changes_24h_pct"][sym]
            result[f"{sym}_name"] = name
    except Exception as exc:
        return SourceResult(
            name="yfinance_commodities", ok=False, data={},
            source="yfinance",
            note=f"请求失败: {exc!r}",
        )

    has_data = any(v is not None for v in result["prices"].values())
    return SourceResult(
        name="yfinance_commodities",
        ok=has_data,
        data=result,
        source="yfinance (Yahoo Finance)",
        note="商品期货价格 via yfinance",
    )


def get_fcsapi_forex_commodities() -> SourceResult:
    """Fetch spot gold/silver/forex via FCS API.

    Free tier: 3 requests/minute.
    Useful for XAU/USD spot price (different from GC=F futures).
    """
    cfg = _load_private_data_sources().get("fcsapi") or {}
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://fcsapi.com/api-v3")

    if not api_key:
        return SourceResult(
            name="fcsapi", ok=False, data={},
            source="fcsapi.com", note="API key 未配置",
        )

    import requests as _req
    symbols = [
        ("XAU/USD", "黄金现货"),
        ("XAG/USD", "白银现货"),
    ]
    results = {}
    all_ok = True

    for sym, label in symbols:
        url = f"{base_url}/forex/latest?symbol={sym}&access_key={api_key}"
        try:
            resp = _req.get(url, timeout=15)
            j = resp.json()
        except Exception as exc:
            results[sym] = {"error": repr(exc)}
            all_ok = False
            continue

        if not j.get("status"):
            results[sym] = {"error": str(j)
            }
            all_ok = False
            continue

        data = (j.get("response") or [None])[0]
        if data:
            results[sym] = {
                "price": float(data.get("c", 0)),
                "open": float(data.get("o", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "change": float(data.get("ch", 0)),
                "change_pct": float(data.get("cp", "0").replace("%", "") or 0),
                "time": data.get("t", ""),
                "label": label,
            }

    data = {}
    for sym, info in results.items():
        if "price" in info:
            key = sym.replace("/", "_")
            data[f"{key}_price"] = info["price"]
            data[f"{key}_change_pct"] = info.get("change_pct", 0)
            data[f"{key}_label"] = info["label"]
            data["prices"] = data.get("prices", {})
            data["prices"][sym] = info["price"]

    return SourceResult(
        name="fcsapi",
        ok=all_ok and bool(data.get("prices", {})),
        data=data,
        source="fcsapi.com (FCS API)",
        note="黄金/白银现货价格（XAU/USD, XAG/USD）免费版3次/分钟",
    )


def get_metals_dev() -> SourceResult:
    """Fetch comprehensive metal prices via api.metals.dev.

    Returns: gold/silver/platinum/palladium spot + LME base metals
    (copper, aluminum, lead, nickel, zinc) + LBMA benchmarks.
    Key: DPZWSOW4ZSHLQXGS7DYZ235GS7DYZ
    """
    cfg = _load_private_data_sources().get("metals_dev") or {}
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.metals.dev/v1/latest")

    if not api_key:
        return SourceResult(
            name="metals_dev", ok=False, data={},
            source="api.metals.dev", note="API key 未配置",
        )

    import requests as _req
    try:
        url = f"{base_url}?api_key={api_key}&currency=USD&unit=toz"
        resp = _req.get(url, timeout=15)
        resp.raise_for_status()
        j = resp.json()
    except Exception as exc:
        return SourceResult(
            name="metals_dev", ok=False, data={},
            source="api.metals.dev", note=f"请求失败: {exc!r}",
        )

    if j.get("status") != "success":
        return SourceResult(
            name="metals_dev", ok=False, data={},
            source="api.metals.dev", note=f"API 返回异常: {j}",
        )

    metals = j.get("metals", {}) or {}
    timestamps = j.get("timestamps", {})

    # Map to our structure
    data: dict[str, Any] = {
        "gold": metals.get("gold"),
        "silver": metals.get("silver"),
        "platinum": metals.get("platinum"),
        "palladium": metals.get("palladium"),
        "copper": metals.get("copper"),
        "aluminum": metals.get("aluminum"),
        "lead": metals.get("lead"),
        "nickel": metals.get("nickel"),
        "zinc": metals.get("zinc"),
        "lbma_gold_am": metals.get("lbma_gold_am"),
        "lbma_gold_pm": metals.get("lbma_gold_pm"),
        "lbma_silver": metals.get("lbma_silver"),
        "lme_copper": metals.get("lme_copper"),
        "lme_aluminum": metals.get("lme_aluminum"),
        "lme_lead": metals.get("lme_lead"),
        "lme_nickel": metals.get("lme_nickel"),
        "lme_zinc": metals.get("lme_zinc"),
        "timestamp": timestamps.get("metal", ""),
        "currency": j.get("currency", "USD"),
        "unit": j.get("unit", "toz"),
    }
    # Also populate convenience keys (in USD)
    for k in ["gold", "silver", "platinum", "palladium"]:
        v = metals.get(k)
        if v:
            data[f"{k}_price"] = v
            data[f"{k}_name"] = k.title()
            if k == "gold":
                data["XAU_USD"] = v
            elif k == "silver":
                data["XAG_USD"] = v

    return SourceResult(
        name="metals_dev",
        ok=bool(metals.get("gold")),
        data=data,
        source="api.metals.dev",
        note="贵金属+基本金属+LME现货价格",
    )


def get_finnhub_quote(symbol: str = "GC=F") -> SourceResult:
    """Fetch US stock/ETF quote via Finnhub.

    Useful for US stocks not easily available via yfinance.
    Note: A-share stocks are NOT supported on free plan.
    """
    cfg = _load_private_data_sources().get("finnhub") or {}
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://finnhub.io/api/v1")

    if not api_key:
        return SourceResult(
            name="finnhub", ok=False, data={},
            source="finnhub.io", note="API key 未配置",
        )

    import requests as _req
    try:
        url = f"{base_url}/quote?symbol={symbol}&token={api_key}"
        resp = _req.get(url, timeout=15)
        j = resp.json()
    except Exception as exc:
        return SourceResult(
            name="finnhub", ok=False, data={},
            source="finnhub.io", note=f"请求失败: {exc!r}",
        )

    if "c" not in j or j["c"] is None:
        return SourceResult(
            name="finnhub", ok=False, data={},
            source="finnhub.io", note=f"{symbol} 不可用: {j.get('error','?')}",
        )

    return SourceResult(
        name="finnhub", ok=True,
        data={
            "symbol": symbol,
            "price": j["c"],
            "change": j.get("d"),
            "change_pct": j.get("dp"),
            "high": j.get("h"),
            "low": j.get("l"),
            "open": j.get("o"),
            "prev_close": j.get("pc"),
            "timestamp": j.get("t"),
        },
        source="finnhub.io",
        note=f"{symbol} 实时行情",
    )


fetch_opec_momr = get_opec_momr
fetch_eia_petroleum_weekly = get_eia_petroleum_weekly
fetch_fed_data_sources = get_fed_data_sources
fetch_international_oil_sources = get_international_oil_sources
fetch_metals_api = get_metals_api
fetch_oilpriceapi = get_oilpriceapi
fetch_yfinance_commodities = get_yfinance_commodities
fetch_fcsapi = get_fcsapi_forex_commodities
fetch_metals_dev = get_metals_dev
fetch_finnhub = get_finnhub_quote
