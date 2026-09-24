"""
data.py — 数据层（旧 SQLite/CSV 缓存层）(D1 数据域收敛登记, 2026-08-11)

与 data_store.py 的关系：
  - data_store.py 是本仓数据访问的【唯一真源入口】（缓存 + 质量校验 + fill_gaps），
    它复用了本模块的 SQLite 存储原语（_get_conn/_query_daily_db/_write_daily_db/
    init_db/normalize_symbol 等）。
  - 本模块保留为存储底层，不向 data_store 转发公共函数：data_store 依赖本模块
    低层实现，若反向转发将产生循环依赖；且 fetch_daily/fetch_many 等公共函数的
    缓存/异常语义与 DataStore.get/get_many 不同，转发会改变调用方行为。
  - 新代码请优先使用 data_store.get_store().get()，旧公共函数保持兼容不动。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "generated" / "quant_system" / "cache"
DB_PATH = ROOT / "generated" / "quant_system" / "quant.db"
# Frozen local daily-price provider used before SQLite/API fallback.
WAREHOUSE_KLINE_DIR = ROOT / "data_warehouse" / "kline"

logger = logging.getLogger("quant_data")

# Lazy SQLite connection — inited by init_db() or first use
_conn: sqlite3.Connection | None = None
_DB_LOCK = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _DB_LOCK:
            if _conn is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_klines (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            PRIMARY KEY (symbol, date)
        );
        CREATE TABLE IF NOT EXISTS sector_stocks (
            symbol TEXT NOT NULL,
            sector_name TEXT NOT NULL,
            PRIMARY KEY (symbol, sector_name)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Region detection — decide whether this server is inside mainland China
# ---------------------------------------------------------------------------

_IS_ABROAD: bool | None = None


def _detect_region() -> bool:
    """Return True if this server appears to be abroad (outside mainland China).

    Heuristic: check external IP country via ip-api.com (free, no key).
    Cached after first call.
    """
    global _IS_ABROAD
    if _IS_ABROAD is not None:
        return _IS_ABROAD
    try:
        import requests as _req
        r = _req.get("http://ip-api.com/json/", timeout=5)
        data = r.json()
        country_code = data.get("countryCode", "")
        _IS_ABROAD = country_code != "CN"
    except Exception:
        # Fallback: assume abroad if IP check fails
        _IS_ABROAD = True
    return _IS_ABROAD


def is_abroad() -> bool:
    """Public accessor for region detection result."""
    return _detect_region()


# ---------------------------------------------------------------------------
# Symbol utilities
# ---------------------------------------------------------------------------

# Lazy-loaded stock name → code mapping
_name_map: dict[str, str] | None = None


def _load_name_map() -> dict[str, str]:
    """Load stock name → code mapping from local JSON cache."""
    global _name_map
    if _name_map is not None:
        return _name_map
    map_path = Path(__file__).resolve().parent / "stock_name_map.json"
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            _name_map = json.load(f)
    else:
        _name_map = {}
    return _name_map


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    # P2-Q2-fix: L196 港股 5 位代码 (00700.HK / HK00700 / 00700) 走独立分支,
    # 避免被补零成 6 位 A 股代码 (00700 → 000700)
    if s.endswith(".HK") or s.startswith("HK"):
        body = s[:-3] if s.endswith(".HK") else s[2:]
        digits = "".join(ch for ch in body if ch.isdigit())
        return digits[-5:].zfill(5) if digits else s
    if s.endswith((".SZ", ".SH")):
        return s[:6]
    if s.startswith(("SZ", "SH")) and len(s) >= 8:
        return s[2:8]
    if s.isdigit() and len(s) == 5:
        return s  # 5 位纯数字 → 港股代码, 不补零
    return s.zfill(6) if s.isdigit() and len(s) <= 6 else s


def resolve_symbol(symbol_or_name: str) -> str:
    """Resolve a stock code or full name to a normalized 6-digit code.

    Supports three input forms:
      1. 6-digit code (e.g. "002714", "601899") → normalised code
      2. Full Chinese name (e.g. "牧原股份") → code via local mapping
      3. Code with suffix (e.g. "002714.SZ", "SH601899") → normalised code

    Returns the 6-digit code if found, or the original input if unresolvable.
    """
    s = str(symbol_or_name).strip()
    # If it looks like a code (digits, possibly with suffix), try normalizing first
    if s.replace(".", "").replace("SZ", "").replace("SH", "").replace("sz", "").replace("sh", "").strip().isdigit():
        return normalize_symbol(s)
    # Otherwise try name lookup
    name_map = _load_name_map()
    code = name_map.get(s)
    if code:
        return normalize_symbol(code)
    # Partial match (user may not enter exact full name)
    for name, code in name_map.items():
        if s in name or name in s:
            return normalize_symbol(code)
    # Not found — return as-is
    return normalize_symbol(s)


def search_stock_name(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search stock names containing the query string.

    Returns a list of {"name": ..., "code": ...} sorted alphabetically by code.
    """
    name_map = _load_name_map()
    q = query.strip()
    if not q:
        return []
    # Direct match first, then contains match
    results = []
    for name, code in name_map.items():
        if name == q:
            results.insert(0, {"name": name, "code": code})
        elif q in name:
            results.append({"name": name, "code": code})
    results.sort(key=lambda x: x["code"])
    return results[:limit]


def market_symbol(symbol: str) -> str:
    # P2-Q2-fix: M191 统一市场前缀 (4/8/920xxx→bj), 北交所不再被当作 sz/无前缀
    from .sources_all import market_prefix

    symbol = normalize_symbol(symbol)
    return f"{market_prefix(symbol)}{symbol}"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_COLS_DB = ["symbol", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover", "pct_chg"]


def _ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SQLite-read columns back to proper dtypes."""
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    num_cols = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _read_warehouse_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Read a local frozen Kline file without contacting a remote source."""
    path = WAREHOUSE_KLINE_DIR / f"{normalize_symbol(symbol)}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("warehouse_read_failed symbol=%s error=%s", symbol, exc)
        return pd.DataFrame()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"):
        if col not in df.columns:
            df[col] = 0.0 if col in {"volume", "amount", "turnover"} else pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["pct_chg"].isna().all():
        df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100.0
    out = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def _write_daily_db(symbol: str, df: pd.DataFrame) -> None:
    """Upsert daily_klines rows into SQLite."""
    conn = _get_conn()
    df = df.copy()
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    # Keep only columns that exist in both the dataframe and the schema
    avail = [c for c in _COLS_DB if c in df.columns]
    data = df[avail].values.tolist()
    if not data:
        return
    placeholders = ",".join(["?"] * len(avail))
    sql = f"INSERT OR REPLACE INTO daily_klines ({','.join(avail)}) VALUES ({placeholders})"
    conn.executemany(sql, data)
    conn.commit()


def _query_daily_db(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Query daily_klines from SQLite for the given date range."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM daily_klines WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
        conn,
        params=(symbol, start, end),
    )
    return _ensure_dtypes(df)


def _latest_possible_trade_day(end_date: str) -> str:
    """返回 <= end_date 的最近一个可能的交易日 (周末回退到周五).

    P2-Q2-fix: M193 用于缓存新鲜度校验, 非交易日不算缺口.
    """
    d = datetime.strptime(end_date, "%Y%m%d")
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def _cache_is_stale(df: pd.DataFrame, end_date: str) -> bool:
    """缓存最新一根 K 线日期是否未覆盖到 end_date.

    P2-Q2-fix: M193 行数达标仍可能用过期数据; 周末等非交易日不算缺口.
    """
    if df is None or df.empty:
        return True
    max_date = df["date"].max()
    if isinstance(max_date, (pd.Timestamp, datetime)):
        max_str = max_date.strftime("%Y%m%d")
    else:
        max_str = str(max_date)[:10].replace("-", "")
    return max_str < _latest_possible_trade_day(end_date)


# ---------------------------------------------------------------------------
# Public: fetch_daily_db  (SQLite + API fallback)
# ---------------------------------------------------------------------------

def fetch_daily_db(
    symbol: str,
    start: str = "20200101",
    end: str | None = None,
    use_cache: bool = True,
    days_back: int = 480,
) -> pd.DataFrame:
    """Fetch daily klines via SQLite cache, supplementing from API if needed.

    Parameters
    ----------
    symbol : str        — 6-digit code.
    start : str         — earliest date YYYYMMDD.
    end : str | None    — latest date (default today).
    use_cache : bool    — if True, read from SQLite then supplement;
                          if False, pull from API unconditionally.
    days_back : int     — minimum number of trading days required.

    Returns
    -------
    pd.DataFrame with columns [date, open, high, low, close, volume, amount,
                                turnover, pct_chg], sorted by date.
    """
    symbol = normalize_symbol(symbol)
    end_date = end or date.today().strftime("%Y%m%d")

    if use_cache:
        init_db()
        df = _query_daily_db(symbol, start, end_date)

        if len(df) >= days_back:
            # P2-Q2-fix: M193 行数达标还不够, 校验最新日期是否覆盖 end_date;
            # 过期则增量补拉 (非交易日不算缺口), 失败降级返回缓存并告警可见
            if not _cache_is_stale(df, end_date):
                return df
            logger.warning(
                "Daily cache stale for %s: max=%s < expected=%s; refreshing tail",
                symbol,
                df["date"].max(),
                _latest_possible_trade_day(end_date),
            )
            try:
                new_df = _fetch_daily_with_fallbacks(symbol, start, end_date, "qfq", 2)
                if new_df is not None and not new_df.empty:
                    _write_daily_db(symbol, _normalize_akshare_daily(new_df))
                df = _query_daily_db(symbol, start, end_date)
            except Exception as exc:
                logger.warning("Incremental refresh failed for %s: %s", symbol, exc)
            return df

        # Not enough rows — extend search window backwards
        # ~1.4 calendar days per trading day
        extra_days = int((days_back - len(df)) * 1.4) + 30
        extended_start = (
            datetime.strptime(start, "%Y%m%d") - timedelta(days=extra_days)
        ).strftime("%Y%m%d")

        new_df = _fetch_daily_with_fallbacks(symbol, extended_start, end_date, "qfq", 3)
        if new_df is not None and not new_df.empty:
            new_df = _normalize_akshare_daily(new_df)
            _write_daily_db(symbol, new_df)

        # Re-read from DB after the upsert
        df = _query_daily_db(symbol, start, end_date)
        if len(df) >= days_back:
            return df
        # Return whatever we have if still insufficient
        return df

    # use_cache=False — fetch from API directly
    df = _fetch_daily_with_fallbacks(symbol, start, end_date, "qfq", 3)
    if df is None or df.empty:
        raise ValueError(f"No daily data for {symbol}")
    df = _normalize_akshare_daily(df)
    # Write to SQLite for future use
    init_db()
    _write_daily_db(symbol, df)
    return df


# ---------------------------------------------------------------------------
# Public: fetch_daily  (modified — now SQLite-backed, CSV fallback retained)
# ---------------------------------------------------------------------------

def fetch_daily(
    symbol: str,
    start: str = "20200101",
    end: str | None = None,
    adjust: str = "qfq",
    use_cache: bool = True,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily klines.  SQLite-backed when use_cache=True.

    Preserved signature for backward compatibility.
    CSV cache kept as secondary fallback.
    """
    symbol = normalize_symbol(symbol)
    end_date = end or date.today().strftime("%Y%m%d")

    if use_cache:
        # Prefer the immutable local warehouse for all cached reads. This keeps
        # GUI history and research backtests on the same reproducible snapshot.
        warehouse = _read_warehouse_daily(symbol, start, end_date)
        if not warehouse.empty:
            warehouse.attrs["data_source"] = "local_warehouse"
            warehouse.attrs["snapshot_path"] = str(WAREHOUSE_KLINE_DIR / f"{symbol}.parquet")
            return warehouse
        # --- secondary: SQLite ---
        init_db()
        df = _query_daily_db(symbol, start, end_date)
        if not df.empty:
            # Check we have enough data (approximate: at least half of days_back)
            expected = int((datetime.strptime(end_date, "%Y%m%d")
                            - datetime.strptime(start, "%Y%m%d")).days * 0.5)
            # P2-Q2-fix: M193 行数达标还不够, 校验最新日期覆盖 end_date
            if len(df) >= max(expected, 20) and not _cache_is_stale(df, end_date):
                return df

        # --- fallback: CSV cache ---
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"daily-{symbol}-{start}-{end_date}-{adjust}.csv"
        csv_stale: pd.DataFrame | None = None
        if cache_path.exists():
            df_csv = _read_daily_csv(cache_path)
            # P2-P2-Q10-M033-fix: CSV 回退同样校验新鲜度(与 SQLite 路径 _cache_is_stale 一致),
            # 过期缓存不得静默返回 —— 避免 decide_stock 用陈旧日线(BOLL/MA)与实时价混算止损位。
            if not df_csv.empty:
                if not _cache_is_stale(df_csv, end_date):
                    _write_daily_db(symbol, df_csv)  # Also write CSV data into SQLite
                    return df_csv
                csv_stale = df_csv
                logger.warning("CSV cache stale for %s; max=%s, refreshing from API",
                               symbol, df_csv["date"].max())

        # --- neither SQLite nor CSV has fresh data → fetch from API ---
        try:
            df_api = _fetch_daily_with_fallbacks(symbol, start, end_date, adjust, retries)
            if df_api is None or df_api.empty:
                raise ValueError(f"No daily data for {symbol}")
            out = _normalize_akshare_daily(df_api)
            out.to_csv(cache_path, index=False)
            _write_daily_db(symbol, out)
            return out
        except Exception as exc:
            # P2-P2-Q10-M033-fix: 刷新失败时降级必须可见 —— 有过期缓存则告警后返回, 否则上抛
            if csv_stale is not None:
                logger.warning("API refresh failed for %s: %s; serving stale CSV", symbol, exc)
                return csv_stale
            raise

    # use_cache=False — fetch from API unconditionally
    df_api = _fetch_daily_with_fallbacks(symbol, start, end_date, adjust, retries)
    if df_api is None or df_api.empty:
        raise ValueError(f"No daily data for {symbol}")
    out = _normalize_akshare_daily(df_api)
    # Still write to SQLite so future calls benefit
    init_db()
    _write_daily_db(symbol, out)
    # CSV cache as backup
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"daily-{symbol}-{start}-{end_date}-{adjust}.csv"
    out.to_csv(cache_path, index=False)
    return out


# ---------------------------------------------------------------------------
# Public: load_market_state  (unchanged)
# ---------------------------------------------------------------------------

def load_market_state() -> dict:
    data_dir = ROOT / "generated" / "a_share_data"
    files = sorted(data_dir.glob("*-meta.json"))
    if not files:
        return {"trade_date": None, "risk": 5, "risk_level": "unknown", "breadth": {}, "hot_metrics": {}, "source": "missing"}
    # 文件名排序不等于数据日排序，按 JSON 内 trade_date 选择最新记录。
    def _trade_key(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("trade_date")
            return str(value or "")
        except Exception:
            return ""
    latest = max(files, key=_trade_key)
    with latest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "trade_date": data.get("trade_date"),
        "risk": int(data.get("risk", 5)),
        "risk_level": data.get("risk_level", "unknown"),
        "breadth": data.get("breadth", {}),
        "hot_metrics": data.get("hot_metrics", {}),
        "source": str(latest),
    }


# ---------------------------------------------------------------------------
# Public: fetch_sector_map
# ---------------------------------------------------------------------------

def fetch_sector_map(refresh: bool = False) -> dict[str, str]:
    """Return {symbol: sector_name} for A-share stocks.

    Cached in the ``sector_stocks`` SQLite table.
    Pass ``refresh=True`` to re-pull from akshare.
    """
    init_db()
    conn = _get_conn()

    if not refresh:
        rows = conn.execute(
            "SELECT symbol, sector_name FROM sector_stocks ORDER BY symbol"
        ).fetchall()
        if rows:
            return dict(rows)

    # --- fetch from akshare ---
    import time
    import requests as _req

    # Try East Money batch API for industry info (faster, works per stock)
    # Only use this as fallback since it hits one stock at a time
    result: dict[str, str] = {}
    rows_to_insert: list[tuple[str, str]] = []

    try:
        import akshare as ak
        sectors_df = ak.stock_board_industry_name_em()
        sector_names = sectors_df["板块名称"].tolist()

        for sector_name in sector_names:
            try:
                cons_df = ak.stock_board_industry_cons_em(sector_name)
            except Exception as e:
                logger.error(f"[data] 操作失败: {e}", exc_info=True)
                continue
            for raw_sym in cons_df["代码"].tolist():
                sym = normalize_symbol(str(raw_sym))
                if sym not in result:
                    result[sym] = sector_name
                    rows_to_insert.append((sym, sector_name))
    except Exception as e:
        logger.error(f"[data] 操作失败: {e}", exc_info=True)

    # Batch write to SQLite if we got data
    if rows_to_insert:
        conn.executemany(
            "INSERT OR REPLACE INTO sector_stocks (symbol, sector_name) VALUES (?, ?)",
            rows_to_insert,
        )
        conn.commit()
        return result

    # Last resort: try single-stock EM API for a few known symbols
    try:
        url_tpl = "https://push2delay.eastmoney.com/api/qt/stock/get?secid={}.{}&fields=f57,f58,f127,f100"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        for sym in ["002714", "601899", "000858", "600519", "002594", "300750"]:
            exchange = 1 if sym.startswith("6") else 0
            try:
                r = _req.get(url_tpl.format(exchange, sym), headers=headers, timeout=3)
                d = r.json().get("data", {}) or {}
                ind = d.get("f100") or d.get("f127")
                if ind:
                    result[sym] = str(ind)
            except Exception as e:
                logger.error(f"[data] 操作失败: {e}", exc_info=True)
        if result:
            rows_to_insert = [(k, v) for k, v in result.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO sector_stocks (symbol, sector_name) VALUES (?, ?)",
                rows_to_insert,
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[data] 操作失败: {e}", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Public: fetch_many_db
# ---------------------------------------------------------------------------

def fetch_many_db(
    symbols: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Batch-load daily data for multiple symbols via SQLite.

    Returns {symbol: DataFrame}.  Failed symbols get an empty DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for raw in symbols:
        sym = normalize_symbol(raw)
        try:
            result[sym] = fetch_daily_db(sym, start=start, end=end, use_cache=use_cache)
        except Exception:
            result[sym] = pd.DataFrame()
    return result


# ---------------------------------------------------------------------------
# Public: fetch_many  (compatibility wrapper)
# ---------------------------------------------------------------------------

def fetch_many(
    symbols: Iterable[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Batch-load daily data.  Internally delegates to ``fetch_many_db``.

    Preserved signature for backward compatibility.

    P2-Q2-fix: M192 返回 dict 只含成功加载的股票; 失败股票不再以空 DataFrame
    占位混入数据字典, 统一记录到 ``__failures__`` 键 (DataFrame: symbol/error),
    供需要报告失败的调用方 (如 cli._extract_failures) 消费.
    """
    sym_list = list(symbols)
    result = fetch_many_db(sym_list, start=start, end=end, use_cache=use_cache)
    failures: dict[str, str] = {}
    ok: dict[str, pd.DataFrame] = {}
    for sym, df in result.items():
        if df.empty:
            failures[sym] = "empty"
        else:
            ok[sym] = df
    if not ok and failures:
        raise RuntimeError(f"All symbols failed: {failures}")
    if failures:
        logger.warning(
            "fetch_many: %d/%d symbols failed: %s",
            len(failures), len(sym_list), list(failures),
        )
        ok["__failures__"] = pd.DataFrame(
            [{"symbol": k, "error": v} for k, v in failures.items()]
        )
    return ok


# ---------------------------------------------------------------------------
# Internal helpers  (unchanged)
# ---------------------------------------------------------------------------

def _fetch_daily_with_fallbacks(
    symbol: str, start: str, end: str, adjust: str, retries: int = 1
) -> pd.DataFrame:
    """Fetch daily K-line using multi-source unified layer.

    优先级: baostock → Tencent原生 → Tushare Pro → akShare 腾讯fallback
    海外(UK)已验证 baostock 和 Tencent 原生可用。
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Try unified multi-source layer
    from .sources_all import fetch_daily_unified

    try:
        df = fetch_daily_unified(symbol, start=start, end=end, adjust=adjust, timeout=25)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.error(f"[data] 操作失败: {exc}", exc_info=True)

    # Fallback to Eastmoney AkShare daily. Avoid stock_zh_a_hist_tx here because
    # its 6-column fallback labels turnover lots as amount and can pollute cache.
    import akshare as ak
    import os as _os
    _os.environ["AKSHARE_PROGRESS"] = "0"
    errors: list[str] = []

    for attempt in range(1, retries + 1):
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
        except Exception as exc:
            errors.append(f"akshare_em={exc!r}")
            if attempt < retries:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Daily fetch failed for {symbol}: {'; '.join(errors)}"
    )


def _normalize_akshare_daily(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    out = df.rename(columns=mapping).copy()
    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in out.columns]

    # If data is already in English column names (from unified sources),
    # just clean it up
    if missing and all(c in df.columns for c in required):
        out = df.copy()
    elif missing:
        raise ValueError(
            f"Missing daily columns: {missing}; got {list(df.columns)}"
        )

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    for col in ["open", "high", "low", "close", "volume", "amount",
                "pct_chg", "turnover"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # P2-Q2-fix: M194 英文列输入 (统一层) 也产出 pct_chg, 与 _ensure_dtypes 对称
    if "pct_chg" not in out.columns and "close" in out.columns:
        out["pct_chg"] = out["close"].pct_change() * 100.0
    keep = [
        c for c in [
            "date", "open", "high", "low", "close",
            "volume", "amount", "pct_chg", "turnover",
        ] if c in out.columns
    ]
    return (
        out[keep]
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _read_daily_csv(path: Path) -> pd.DataFrame:
    # V11 审计修复（Medium）: 原实现无保护 read_csv，缓存文件损坏/列缺失时
    # 直接抛异常中断 fetch_daily 的降级链（无法回退到 API 刷新）。
    # 修正: 读取失败返回空 DataFrame，调用方判定为空后走 API 重拉。
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public: aggregate_timeframe  (unchanged)
# ---------------------------------------------------------------------------

def aggregate_timeframe(
    df: pd.DataFrame, period: str = "weekly"
) -> pd.DataFrame:
    """将日线聚合为周线或月线。
    period: "weekly"|"monthly"
    """
    if period == "daily" or period not in ("weekly", "monthly"):
        return df
    df = df.copy().sort_values("date")
    if period == "weekly":
        df["_period"] = (
            df["date"].dt.isocalendar().year.astype(str)
            + "-W"
            + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        freq = "W"
    else:
        df["_period"] = df["date"].dt.strftime("%Y-%m")
        freq = "M"
    agg = df.groupby("_period", sort=True).agg(
        date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in df.columns
               else ("close", "first"),
        amount=("amount", "sum") if "amount" in df.columns
               else ("close", "first"),
        pct_chg=(
            "close",
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100,
        ),
    )
    return agg


# ---------------------------------------------------------------------------
# Public: fetch_intraday  (minute-level K-lines)
# ---------------------------------------------------------------------------

def _fetch_intraday_sina(
    symbol: str, period: str, datalen: int = 480
) -> pd.DataFrame | None:
    """Fetch minute K-line from Sina API (works both in China and abroad).

    API endpoint: https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData
    Params: symbol=sh600519, scale=5, ma=no, datalen=480
    """
    import requests as _req
    sym = market_symbol(normalize_symbol(symbol))
    scale_map = {"1": "1", "5": "5", "15": "15", "30": "30", "60": "60"}
    scale = scale_map.get(period, "5")
    try:
        r = _req.get(
            "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData",
            params={"symbol": sym, "scale": scale, "ma": "no", "datalen": str(datalen)},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
            timeout=20,
        )
        if r.status_code != 200 or not r.text or '"__ERROR"' in r.text:
            return None
        import json
        data = json.loads(r.text)
        if not data:
            return None
        rows = []
        for item in data:
            rows.append({
                "date": item.get("day", ""),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "close": float(item.get("close", 0)),
                "volume": float(item.get("volume", 0)),
                "amount": float(item.get("amount", 0)),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def fetch_intraday(
    symbol: str,
    period: str = "5",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch minute-level K-line data with region-aware fallback.

    Abroad → Sina first (fast abroad), East Money fallback.
    China  → East Money first, Sina fallback.

    Parameters
    ----------
    symbol : str          — 6-digit A-share code.
    period : str          — "1" | "5" | "15" | "30" | "60" (minutes)
    start_date : str|None — "YYYY-MM-DD", default 14 days ago.
    end_date : str|None   — "YYYY-MM-DD", default today.

    Returns
    -------
    pd.DataFrame with columns [date, open, high, low, close, volume, amount].

    Note (P2-Q2-fix: M187): 分钟线统一返回【不复权】价格, 与日线 qfq 基期不同;
    跨复权基期拼接 (日内×日线) 需在策略层自行对齐, 除权日附近注意价差.
    """
    import akshare as ak

    sym = normalize_symbol(symbol)
    if not start_date:
        start_date = (date.today() - timedelta(days=14)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    abroad = _detect_region()
    import os as _os
    _os.environ["AKSHARE_PROGRESS"] = "0"

    def _try_em() -> pd.DataFrame | None:
        """Try East Money minute K-line."""
        try:
            # P2-Q2-fix: M187 分钟线统一用不复权 (adjust=""),
            # 与 _try_sina (新浪) 返回的基期一致, 避免同 symbol 换网络环境换基期
            df = ak.stock_zh_a_hist_min_em(
                symbol=sym,
                start_date=f"{start_date} 09:30:00",
                end_date=f"{end_date} 15:00:00",
                period=period, adjust="",
            )
            if df is not None and not df.empty:
                return _normalize_intraday_df(df)
        except Exception as e:
            logger.error(f"[data] 操作失败: {e}", exc_info=True)
        return None

    def _try_sina() -> pd.DataFrame | None:
        return _fetch_intraday_sina(sym, period, datalen=480)

    # ── try in region-optimised order ──
    result = None
    if abroad:
        result = _try_sina()
        if result is None or result.empty:
            result = _try_em()
    else:
        result = _try_em()
        if result is None or result.empty:
            result = _try_sina()

    if result is not None and not result.empty:
        # Filter by requested date range
        if start_date:
            result = result[result["date"] >= start_date]
        if end_date:
            result = result[result["date"] <= end_date + " 23:59:59"]
        return result.reset_index(drop=True)

    return pd.DataFrame()


def _normalize_intraday_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize East Money minute K-line DataFrame."""
    col_map = {"时间": "date", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume",
               "成交额": "amount", "最新价": "close"}
    out = df.rename(columns=col_map).copy()
    required = ["date", "open", "high", "low", "close"]
    if not all(c in out.columns for c in required):
        if "最新价" in out.columns and "close" not in out.columns:
            out["close"] = out["最新价"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep = [c for c in ["date", "open", "high", "low", "close",
                         "volume", "amount"] if c in out.columns]
    return out[keep].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# __main__: one-shot init / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print("✓ init_db()  — tables created (if not existed)")

    # quick fetch test
    _start_2y = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")
    df = fetch_daily_db("000001", start=_start_2y, use_cache=True, days_back=20)
    print(f"✓ fetch_daily_db 000001  →  {len(df)} rows, columns={list(df.columns)}")

    sectors = fetch_sector_map(refresh=False)
    print(f"✓ fetch_sector_map  →  {len(sectors)} stocks mapped")
    if sectors:
        sample = list(sectors.items())[:3]
        for sym, sec in sample:
            print(f"   {sym}  →  {sec}")

    many = fetch_many_db(["000001", "002714"], start=_start_2y, use_cache=True)
    print(f"✓ fetch_many_db  →  {list(many.keys())} symbols loaded")
    for sym, mdf in many.items():
        print(f"   {sym}: {len(mdf)} rows")
