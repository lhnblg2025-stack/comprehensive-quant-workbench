"""
量化交易系统 — 多层数据管道 (Data Pipeline)

自动化的多层数据获取与管理管道。
支持缓存到 SQLite, TTL 过期自动刷新, 批量财务获取。

用法:
  from quant_system.data_pipeline import fetch_all_market_data, update_data_if_stale
"""

from __future__ import annotations

import json
import sqlite3
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import threading
from threading import Thread

from quant_system.utils import to_float as _to_float

import requests

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
DB_PATH = ROOT / "market_data.db"

TENCENT_HEADERS = {"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"}
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
EASTMONEY_H = {"User-Agent": "Mozilla/5.0", "Referer": "https://emweb.securities.eastmoney.com/"}

# ── TTL 配置 ──
TTL = {
    "quotes": 300,          # 5 min
    "indices": 300,         # 5 min
    "north_flow": 1800,     # 30 min
    "margin": 3600,         # 1 h
    "futures": 600,         # 10 min
    "etf": 600,             # 10 min
    "financial": 86400,     # 1 d
}


# ════════════════════════════════════════════════════════════════
#  内部: SQLite 数据库连接
# ════════════════════════════════════════════════════════════════

import logging

logger = logging.getLogger("quant_data_pipeline")

# P2-Q2-fix: M189 每线程独立连接 (threading.local), 避免跨线程共享连接并发写
# 触发 "database is locked" / 游标冲突导致缓存静默丢失
_DB_LOCAL = threading.local()


def _get_db() -> sqlite3.Connection:
    """连接 market_data.db, 自动建表, 每线程独立连接."""
    conn = getattr(_DB_LOCAL, "conn", None)
    if conn is not None:
        return conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS data_cache (
            source TEXT PRIMARY KEY,
            data TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS data_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            status TEXT,
            rows_affected INTEGER,
            elapsed_ms INTEGER,
            error TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    _DB_LOCAL.conn = conn
    return conn


def _get_cache_mtime(source: str) -> float | None:
    """返回缓存条目的 unix 时间戳, 不存在返回 None."""
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT updated_at FROM data_cache WHERE source = ?", (source,)
        ).fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0]).timestamp()
    except Exception as e:
        logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    return None


def _set_cache(source: str, data: Any) -> None:
    """写入缓存."""
    timestamp = datetime.now(CST).isoformat()
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO data_cache (source, data, updated_at) VALUES (?, ?, ?)",
            (source, json.dumps(data, ensure_ascii=False, default=str), timestamp),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", source, exc)


def _get_cache(source: str, ttl: float | None = None) -> Any | None:
    """读取缓存；ttl 非空时按 updated_at 判断是否过期。

    审计 2026-08-16：原实现不检查 TTL，财务等数据可能永久返回旧值。
    """
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT data, updated_at FROM data_cache WHERE source = ?", (source,)
        ).fetchone()
        if row and row[0]:
            if ttl is not None:
                try:
                    age = (datetime.now(CST) - datetime.fromisoformat(row[1])).total_seconds()
                    if age > ttl:
                        return None  # 过期
                except Exception:
                    # 无法解析时间戳 → 视为过期强制刷新，绝不把过期数据当新鲜返回（假新鲜）
                    logger.warning(f"[data_pipeline] source={source} updated_at 无法解析({row[1]!r})，按过期处理")
                    return None
            return json.loads(row[0])
    except Exception as e:
        logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    return None


# ════════════════════════════════════════════════════════════════
#  日志记录
# ════════════════════════════════════════════════════════════════

def log_data_fetch(
    source: str,
    status: str,
    rows_affected: int = 0,
    elapsed_ms: int = 0,
    error: str = "",
) -> None:
    """记录数据获取日志到 data_log 表."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO data_log (source, status, rows_affected, elapsed_ms, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source, status, rows_affected, elapsed_ms, error, datetime.now(CST).isoformat()),
        )
        conn.commit()
    except Exception as exc:
        # P2-Q2-fix: M189 日志写失败不再静默吞掉, 降级必须可见
        logger.warning("data_log write failed (%s): %s", source, exc)


# ════════════════════════════════════════════════════════════════
#  数据源获取函数
# ════════════════════════════════════════════════════════════════

def _fetch_quotes() -> dict[str, Any]:
    """全市场行情."""
    from quant_system.watchlist import fetch_quotes as wl_quotes
    t0 = _time.time() * 1000
    try:
        quotes = wl_quotes(force=True)
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("quotes", "ok", len(quotes), elapsed)
        return {
            "ok": True,
            "count": len(quotes),
            "data": quotes,
            "updated_at": datetime.now(CST).isoformat(),
        }
    except Exception as e:
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("quotes", "error", 0, elapsed, str(e))
        return {"ok": False, "count": 0, "data": [], "error": str(e)}


def _fetch_sina_index(code: str) -> dict[str, Any]:
    """拉取单个新浪指数."""
    url = f"http://hq.sinajs.cn/list=s_{code}"
    try:
        r = requests.get(url, headers=SINA_HEADERS, timeout=6)
        text = r.text.strip().split('"')[1] if '"' in r.text else ""
        parts = text.split(",")
        if len(parts) >= 6:
            return {
                "index": _to_float(parts[1]),
                "change": _to_float(parts[2]),
                "change_pct": _to_float(parts[3]),
                "volume_wan": _to_float(parts[4]),
                "amount_yi": _to_float(parts[5]) / 10000,
            }
    except Exception as e:
        logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    return {}


def _fetch_indices() -> dict[str, Any]:
    """指数数据 (上证/深证/创业板/科创50/沪深300)."""
    t0 = _time.time() * 1000
    try:
        indices = {}
        codes = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000688": "科创50",
            "sh000300": "沪深300",
        }
        for code, name in codes.items():
            d = _fetch_sina_index(code)
            if d:
                indices[name] = d
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("indices", "ok", len(indices), elapsed)
        return {"ok": bool(indices), "data": indices, "count": len(indices), "updated_at": datetime.now(CST).isoformat()}
    except Exception as e:
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("indices", "error", 0, elapsed, str(e))
        return {"ok": False, "data": {}, "count": 0, "error": str(e)}


def _fetch_north_flow() -> dict[str, Any]:
    """北向资金 (沪深港通)."""
    t0 = _time.time() * 1000
    try:
        from quant_system.north_flow import fetch_north_summary
        data = fetch_north_summary()
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("north_flow", "ok", 1, elapsed)
        return {"ok": True, "data": data, "updated_at": datetime.now(CST).isoformat()}
    except Exception as e:
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("north_flow", "error", 0, elapsed, str(e))
        return {"ok": False, "data": {}, "error": str(e)}


def _fetch_margin() -> dict[str, Any]:
    """两融余额 (上交所+深交所)."""
    t0 = _time.time() * 1000
    try:
        from quant_system.margin import fetch_margin_summary
        data = fetch_margin_summary()
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("margin", "ok", 1, elapsed)
        return {"ok": data.get("ok", False), "data": data, "updated_at": datetime.now(CST).isoformat()}
    except Exception as e:
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch("margin", "error", 0, elapsed, str(e))
        return {"ok": False, "data": {}, "error": str(e)}


def _fetch_futures() -> dict[str, Any]:
    """股指期货 (IF/IC/IH 当月连续)."""
    t0 = _time.time() * 1000
    futures_map = {"IF": "1.IF", "IC": "1.IC", "IH": "1.IH"}
    result: dict[str, Any] = {}
    for name, em_code in futures_map.items():
        try:
            url = f"https://push2delay.eastmoney.com/api/qt/stock/get?secid={em_code}&fields=f43,f44,f45,f46,f47,f48,f170,f171,f57,f58,f60,f86"
            r = requests.get(url, headers=EASTMONEY_H, timeout=5)
            if r.status_code == 200:
                d = r.json().get("data", {})
                result[name] = {
                    "price": _to_float(d.get("f43", 0)) / 100,
                    "open": _to_float(d.get("f46", 0)) / 100,
                    "high": _to_float(d.get("f44", 0)) / 100,
                    "low": _to_float(d.get("f45", 0)) / 100,
                    "volume": _to_float(d.get("f47", 0)),
                    "amount": _to_float(d.get("f48", 0)),
                    "change": _to_float(d.get("f170", 0)) / 100,
                    "change_pct": _to_float(d.get("f171", 0)) / 100,
                    "pre_settle": _to_float(d.get("f60", 0)) / 100,
                    "hold": _to_float(d.get("f86", 0)),
                }
        except Exception as e:
            logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    elapsed = int(_time.time() * 1000 - t0)
    log_data_fetch("futures", "ok", len(result), elapsed)
    return {"ok": bool(result), "data": result, "count": len(result), "updated_at": datetime.now(CST).isoformat()}


def _fetch_etf() -> dict[str, Any]:
    """ETF数据 (510050/510300/510500 份额+净值)."""
    t0 = _time.time() * 1000
    etf_symbols = {
        "510050": "上证50ETF",
        "510300": "沪深300ETF",
        "510500": "中证500ETF",
    }
    result: dict[str, Any] = {}
    for sym, name in etf_symbols.items():
        try:
            codes = f"sh{sym}"
            url = f"http://qt.gtimg.cn/q={codes}"
            r = requests.get(url, headers=TENCENT_HEADERS, timeout=6)
            for line in r.text.strip().split(";\n"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                parts = [part.strip().strip('";') for part in line.split("~")]
                if len(parts) < 50:
                    continue
                try:
                    result[name] = {
                        "symbol": sym,
                        "name": parts[1],
                        "price": _to_float(parts[3]),
                        "prev_close": _to_float(parts[4]),
                        "change_pct": _to_float(parts[32]),
                        "volume": _to_float(parts[6]),
                        "amount": (_to_float(parts[37]) * 10000) if len(parts) > 37 else 0,
                        "turnover": _to_float(parts[38]) if len(parts) > 38 else 0,
                        "volume_ratio": _to_float(parts[49]) if len(parts) > 49 else 0,
                    }
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    elapsed = int(_time.time() * 1000 - t0)
    log_data_fetch("etf", "ok", len(result), elapsed)
    return {"ok": bool(result), "data": result, "count": len(result), "updated_at": datetime.now(CST).isoformat()}


# ════════════════════════════════════════════════════════════════
#  公共函数
# ════════════════════════════════════════════════════════════════

def fetch_all_market_data(date: str = "") -> dict[str, Any]:
    """获取当日完整的市场数据.

    Args:
        date: 日期字符串 (YYYY-MM-DD), 留空自动使用当日.

    Returns:
        dict 包含以下顶层键:
          timestamp, stocks_count, indices{...}, north_flow{...},
          margin{...}, futures{...}, etf{...}
    """
    ts = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
    result: dict[str, Any] = {
        "timestamp": ts,
        "date": date or datetime.now(CST).strftime("%Y-%m-%d"),
    }

    # 并行获取各数据源（审计 2026-08-16：改用 ThreadPoolExecutor，
    # 取代裸 Thread+join(timeout) 不取消线程、并发写 sources 无锁的问题）
    sources: dict[str, dict[str, Any]] = {}
    _src_lock = threading.Lock()
    tasks = [
        ("quotes", _fetch_quotes),
        ("indices", _fetch_indices),
        ("north_flow", _fetch_north_flow),
        ("margin", _fetch_margin),
        ("futures", _fetch_futures),
        ("etf", _fetch_etf),
    ]

    def _run_pair(pair):
        name, fn = pair
        try:
            data = fn()
            with _src_lock:
                sources[name] = data
            _set_cache(name, data)
        except Exception as e:
            logger.error(f"[data_pipeline] {name} 拉取失败: {e}", exc_info=True)

    if HAS_CONCURRENT and ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as pool:
            list(pool.map(_run_pair, tasks))
    else:
        for pair in tasks:
            _run_pair(pair)

    # 组装结果
    quotes_data = sources.get("quotes", {})
    result["stocks_count"] = quotes_data.get("count", 0)
    result["indices"] = sources.get("indices", {}).get("data", {})
    result["north_flow"] = sources.get("north_flow", {}).get("data", {})
    result["margin"] = sources.get("margin", {}).get("data", {})
    result["futures"] = sources.get("futures", {}).get("data", {})
    result["etf"] = sources.get("etf", {}).get("data", {})

    return result


def fetch_financial_batch(symbols: list[str], force: bool = False) -> dict[str, Any]:
    """批量获取财务数据.

    Args:
        symbols: 股票代码列表 (6位数字).
        force: 是否强制刷新缓存.

    Returns:
        {symbol: {roe, profit_growth, revenue_growth, ...}}
    """
    from quant_system.fundamental import fetch_financials

    result: dict[str, Any] = {}
    for sym in symbols:
        if not force:
            cached = _get_cache(f"financial_{sym}", ttl=TTL.get("financial"))
            if cached:
                result[sym] = cached
                continue

        t0 = _time.time() * 1000
        try:
            data = fetch_financials(sym)
            elapsed = int(_time.time() * 1000 - t0)
            if "error" not in data:
                summary = {
                    "symbol": sym,
                    "roe": data.get("roe"),
                    "profit_growth": data.get("profit_yoy"),
                    "revenue_growth": data.get("revenue_yoy"),
                    "gross_margin": data.get("gross_margin"),
                    "debt_ratio": data.get("debt_ratio"),
                    "eps": data.get("eps"),
                    "source": data.get("source", ""),
                }
                result[sym] = summary
                _set_cache(f"financial_{sym}", summary)
                log_data_fetch(f"financial_{sym}", "ok", 1, elapsed)
            else:
                log_data_fetch(f"financial_{sym}", "error", 0, elapsed, str(data.get("error", "")))
        except Exception as e:
            elapsed = int(_time.time() * 1000 - t0)
            log_data_fetch(f"financial_{sym}", "error", 0, elapsed, str(e))

    return result


def update_data_if_stale(force: bool = False) -> dict[str, Any]:
    """检查所有数据源新鲜度, 超过TTL的自动更新.

    Args:
        force: 是否强制全部更新.

    Returns:
        {updated_sources: [...], skipped_sources: [...]}
    """
    updated: list[str] = []
    skipped: list[str] = []
    now = _time.time()

    checks = [
        ("quotes", TTL["quotes"], _fetch_quotes),
        ("indices", TTL["indices"], _fetch_indices),
        ("north_flow", TTL["north_flow"], _fetch_north_flow),
        ("margin", TTL["margin"], _fetch_margin),
        ("futures", TTL["futures"], _fetch_futures),
        ("etf", TTL["etf"], _fetch_etf),
    ]

    for name, ttl, fn in checks:
        if force:
            data = fn()
            if isinstance(data, dict) and data.get("ok") is False:
                # 审计 2026-08-16：失败结果不得当作成功更新缓存
                skipped.append(name)
                continue
            _set_cache(name, data)
            updated.append(name)
            continue

        mtime = _get_cache_mtime(name)
        if mtime is None or (now - mtime) > ttl:
            data = fn()
            if isinstance(data, dict) and data.get("ok") is False:
                skipped.append(name)
                continue
            _set_cache(name, data)
            updated.append(name)
        else:
            skipped.append(name)

    return {
        "updated_sources": updated,
        "skipped_sources": skipped,
        "timestamp": datetime.now(CST).isoformat(),
    }


def get_cached(source: str) -> Any | None:
    """读取指定数据源的缓存."""
    return _get_cache(source)


# ════════════════════════════════════════════════════════════════
#  数据层升级: 增量更新 / 分钟K线 / 资金流 / 重试回退
# ════════════════════════════════════════════════════════════════

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is expected in runtime envs.
    pd = None  # type: ignore[assignment]

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HAS_CONCURRENT = True
except ImportError:
    ThreadPoolExecutor = None  # type: ignore[assignment]
    as_completed = None  # type: ignore[assignment]
    HAS_CONCURRENT = False


QUANT_HOME = Path.home() / ".quant_system"
KLINE_DB_PATH = QUANT_HOME / "klines.db"
MINUTE_DIR = QUANT_HOME / "minute_bars"
MINUTE_DB_PATH = QUANT_HOME / "minute_bars.db"
MONEY_FLOW_DIR = QUANT_HOME / "money_flow"
MONEY_FLOW_DB_PATH = QUANT_HOME / "money_flow.db"

DATA_SOURCE_PRIORITY = ["akshare", "tencent", "sina"]
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 5]


def _ensure_quant_dirs() -> None:
    """确保 ~/.quant_system 数据目录存在."""
    try:
        QUANT_HOME.mkdir(parents=True, exist_ok=True)
        MINUTE_DIR.mkdir(parents=True, exist_ok=True)
        MONEY_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)


def _normalize_symbol(symbol: str) -> str:
    """归一化股票代码为6位数字."""
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else str(symbol).strip()


def _market_prefix(symbol: str) -> str:
    """返回 A 股行情前缀: bj(北交所) / sh / sz.

    P2-Q2-fix: M191 统一前缀规则 (4/8/920xxx→bj), 北交所不再被归为 sz.
    """
    from quant_system.sources_all import market_prefix

    return market_prefix(symbol)


def _eastmoney_secid(symbol: str) -> str:
    """返回东方财富 secid 前缀代码 (sh→1, sz/bj→0)."""
    code = _normalize_symbol(symbol)
    market = "1" if _market_prefix(code) == "sh" else "0"
    return f"{market}.{code}"


# P2-Q2-fix: M189 每线程独立连接, 并按 db_path 建立连接缓存 dict,
# 修复 "_get_kline_db 只认第一次 path" 的问题
_KLINE_DB_LOCAL = threading.local()


def _get_kline_db(db_path: str | None = None) -> sqlite3.Connection:
    """连接K线缓存库并建表. 每线程按 db_path 独立连接."""
    path = str((Path(db_path).expanduser() if db_path else KLINE_DB_PATH).resolve())
    conns = getattr(_KLINE_DB_LOCAL, "conns", None)
    if conns is None:
        conns = {}
        _KLINE_DB_LOCAL.conns = conns
    conn = conns.get(path)
    if conn is not None:
        return conn
    _ensure_quant_dirs()
    if Path(path).parent:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_kline (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            amount REAL,
            amplitude REAL,
            pct_change REAL,
            pct_chg REAL,
            change REAL,
            turnover REAL,
            updated_at TEXT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(daily_kline)").fetchall()}
    if "pct_chg" not in cols:
        conn.execute("ALTER TABLE daily_kline ADD COLUMN pct_chg REAL")
    conn.commit()
    conns[path] = conn
    return conn


def _standardize_daily_kline(df: Any, symbol: str) -> Any:
    """标准化 AkShare 日K字段，同时产出 pct_chg 供下游统一消费."""
    if pd is None or df is None or df.empty:
        return None

    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    out = df.rename(columns=rename).copy()
    required = ["date", "open", "close", "high", "low", "volume", "amount"]
    for col in required:
        if col not in out.columns:
            return None

    keep = [
        "date", "open", "close", "high", "low", "volume", "amount",
        "amplitude", "pct_change", "pct_chg", "change", "turnover",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    out = out[keep]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in keep[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.insert(0, "symbol", _normalize_symbol(symbol))
    out["updated_at"] = datetime.now(CST).isoformat()
    out = out.dropna(subset=["date"]).drop_duplicates(["symbol", "date"], keep="last")
    # 审计 2026-08-16：先按日期升序，再计算 pct_chg，避免乱序导致前视/错误涨跌幅
    out = out.sort_values("date").reset_index(drop=True)
    # W2.5 修复：pct_chg 已在 keep 兜底为 None，须按「全空」判断而非「列不存在」
    # （否则上方补列后 "pct_chg not in columns" 恒为 False，pct_chg 永远不产出 → None 写库）
    if "pct_chg" in out.columns and out["pct_chg"].isna().all():
        if "pct_change" in out.columns and out["pct_change"].notna().any():
            out["pct_chg"] = out["pct_change"]
        elif "close" in out.columns:
            out["pct_chg"] = out["close"].pct_change() * 100.0
            out["pct_chg"] = out["pct_chg"].fillna(0.0)
    return out


def _write_daily_kline(df: Any, db_path: str | None = None) -> int:
    """写入日K缓存, 返回写入行数."""
    if pd is None or df is None or df.empty:
        return 0
    try:
        conn = _get_kline_db(db_path)
        rows = list(df.itertuples(index=False, name=None))
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_kline (
                symbol, date, open, close, high, low, volume, amount,
                amplitude, pct_change, pct_chg, change, turnover, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    except Exception as exc:
        # 审计 2026-08-16：写库失败不得静默吞掉（上层会误报成功）
        logger.error(f"[data_pipeline] _write_daily_kline 写库失败: {exc}", exc_info=True)
        raise


def _fetch_akshare_daily(symbol: str, start_date: str | None = None) -> Any:
    """通过 AkShare 获取前复权日K."""
    if pd is None:
        return None
    try:
        import akshare as ak

        start = start_date or "19900101"
        start = start.replace("-", "")
        end = datetime.now(CST).strftime("%Y%m%d")
        raw = ak.stock_zh_a_hist(
            symbol=_normalize_symbol(symbol),
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        # P2-Q2-fix: M188 akshare EM "成交量" 单位为手, 统一 ×100 转股
        if raw is not None and not raw.empty and "成交量" in raw.columns:
            raw = raw.copy()
            raw["成交量"] = pd.to_numeric(raw["成交量"], errors="coerce") * 100.0
        return _standardize_daily_kline(raw, symbol)
    except Exception as exc:
        logger.warning(f"[data_pipeline] _fetch_akshare_daily({symbol}) 失败: {exc}")
        return None


def _fetch_tencent_daily(symbol: str) -> Any:
    """通过腾讯日K接口获取近期数据作为回退源."""
    if pd is None:
        return None
    try:
        code = f"{_market_prefix(symbol)}{_normalize_symbol(symbol)}"
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{code},day,,,800,qfq"}
        r = requests.get(url, params=params, headers=TENCENT_HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json().get("data", {}).get(code, {})
        rows = data.get("qfqday") or data.get("day") or []
        if not rows:
            return None
        parsed = []
        for row in rows:
            if len(row) >= 6:
                parsed.append(
                    {
                        "date": row[0],
                        "open": row[1],
                        "close": row[2],
                        "high": row[3],
                        "low": row[4],
                        # P2-Q2-fix: M188 腾讯 fqkline 成交量单位为手, ×100 转股
                        "volume": _to_float(row[5]) * 100.0,
                        "amount": row[6] if len(row) > 6 else None,
                    }
                )
        return _standardize_daily_kline(pd.DataFrame(parsed), symbol)
    except Exception as exc:
        logger.warning(f"[data_pipeline] _fetch_tencent_daily({symbol}) 失败: {exc}")
        return None


def _fetch_sina_daily(symbol: str) -> Any:
    """通过新浪日K接口获取近期数据作为回退源."""
    if pd is None:
        return None
    try:
        code = f"{_market_prefix(symbol)}{_normalize_symbol(symbol)}"
        url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketData.getKLineData"
        params = {"symbol": code, "scale": 240, "ma": "no", "datalen": 800}
        r = requests.get(url, params=params, headers=SINA_HEADERS, timeout=8)
        r.raise_for_status()
        text = r.text
        # P2-Q2-fix: L204 先检测 __ERROR/服务名错误再解析 JSONP, 避免 json.loads 后
        # 遍历 dict 键触发 AttributeError (同 data.py 的 '"__ERROR"' 检查)
        if not text or "__ERROR" in text:
            return None
        start = text.find("(")
        end = text.rfind(")")
        if start >= 0 and end > start:
            text = text[start + 1:end]
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, list):
            return None
        parsed = []
        for row in payload:
            parsed.append(
                {
                    "date": row.get("day"),
                    "open": row.get("open"),
                    "close": row.get("close"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    # P2-Q2-fix: M197 新浪 volume 单位为股, 与库内统一单位一致无需换算;
                    # amount 接口不返回, 保持 None → 标准化后为 NaN (文档化降级)
                    "volume": row.get("volume"),
                    "amount": None,
                }
            )
        return _standardize_daily_kline(pd.DataFrame(parsed), symbol)
    except Exception as exc:
        logger.warning(f"[data_pipeline] _fetch_sina_daily({symbol}) 失败: {exc}")
        return None


def get_last_kline_date(symbol: str, db_path: str = None) -> str | None:
    """从 SQLite 查询已缓存的最新K线日期."""
    try:
        conn = _get_kline_db(db_path)
        row = conn.execute(
            "SELECT MAX(date) FROM daily_kline WHERE symbol = ?",
            (_normalize_symbol(symbol),),
        ).fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def fetch_incremental(symbol: str, force_full: bool = False) -> pd.DataFrame | None:
    """更新前复权K线。

    前复权价格会在新的除权除息后重述历史价，不能只追加 last_date 之后的数据；
    默认全量重拉并 INSERT OR REPLACE，避免库内混合不同复权基期。
    """
    if pd is None:
        return None

    code = _normalize_symbol(symbol)
    last_date = None
    start_date = None

    t0 = _time.time() * 1000
    df = _fetch_akshare_daily(code, start_date=start_date)
    if df is None:
        elapsed = int(_time.time() * 1000 - t0)
        log_data_fetch(f"kline_{code}", "error", 0, elapsed, "akshare failed")
        return None
    written = _write_daily_kline(df)
    elapsed = int(_time.time() * 1000 - t0)
    log_data_fetch(f"kline_{code}", "ok", written, elapsed)
    return df


def update_all_incrementally(universe: list[str] = None, max_workers: int = 4) -> dict:
    """增量更新股票池（线程池并行，返回统计）."""
    symbols = universe
    if symbols is None:
        symbols = []
        try:
            from quant_system.watchlist import DEFAULT_WATCHLIST

            symbols = list(DEFAULT_WATCHLIST)
        except Exception:
            try:
                cached = _get_cache("quotes") or {}
                rows = cached.get("data", []) if isinstance(cached, dict) else []
                symbols = [str(r.get("symbol") or r.get("code")) for r in rows if isinstance(r, dict)]
            except Exception:
                symbols = []

    result = {
        "total": len(symbols),
        "success": 0,
        "failed": 0,
        "rows": 0,
        "errors": {},
        "timestamp": datetime.now(CST).isoformat(),
    }

    def _job(sym: str) -> tuple[str, int, str]:
        try:
            df = fetch_incremental(sym)
            if df is None:
                return sym, 0, "fetch failed"
            return sym, len(df), ""
        except Exception as exc:
            return sym, 0, str(exc)

    if HAS_CONCURRENT and ThreadPoolExecutor is not None and as_completed is not None and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_job, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym, rows, err = fut.result()
                if err:
                    result["failed"] += 1
                    result["errors"][sym] = err
                else:
                    result["success"] += 1
                    result["rows"] += rows
    else:
        for sym in symbols:
            sym, rows, err = _job(sym)
            if err:
                result["failed"] += 1
                result["errors"][sym] = err
            else:
                result["success"] += 1
                result["rows"] += rows

    return result


def auto_refresh_cache(force_hour: int = 16, force_minute: int = 30) -> dict:
    """自动刷新：收盘后全量，交易时间增量，其余不变."""
    now = datetime.now(CST)
    cutoff = now.replace(hour=force_hour, minute=force_minute, second=0, microsecond=0)
    # 审计 2026-08-16：改用真实交易日历，节假日不再误触发、调休工作日不再漏检
    try:
        from quant_system.market_clock import is_trading_day
        is_trading = is_trading_day(now)
    except Exception:
        is_trading = now.weekday() < 5  # 降级：周末粗筛
    result = {"mode": "skip", "timestamp": now.isoformat(), "detail": {}}
    if not is_trading:
        return result

    if now >= cutoff:
        result["mode"] = "full"
        result["detail"] = update_data_if_stale(force=True)
        result["incremental"] = update_all_incrementally()
        return result

    in_trading = (
        (now.hour == 9 and now.minute >= 30)
        or (10 <= now.hour < 11)
        or (now.hour == 11 and now.minute <= 30)
        or (13 <= now.hour < 15)
    )
    if in_trading:
        result["mode"] = "incremental"
        result["detail"] = update_all_incrementally()
    return result


# P2-Q2-fix: M189 每线程独立连接, 避免跨线程共享连接并发写冲突
_MINUTE_DB_LOCAL = threading.local()


def _get_minute_db() -> sqlite3.Connection:
    """连接分钟K线回退 SQLite 缓存，每线程独立连接."""
    conn = getattr(_MINUTE_DB_LOCAL, "conn", None)
    if conn is not None:
        return conn
    _ensure_quant_dirs()
    conn = sqlite3.connect(str(MINUTE_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS minute_bars (
            symbol TEXT NOT NULL,
            period TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            amount REAL,
            latest_price REAL,
            avg_price REAL,
            updated_at TEXT,
            PRIMARY KEY (symbol, period, datetime)
        )
        """
    )
    conn.commit()
    _MINUTE_DB_LOCAL.conn = conn
    return conn


def _standardize_minute_bars(df: Any, symbol: str, period: str) -> Any:
    """标准化 AkShare 分钟K线字段."""
    if pd is None or df is None or df.empty:
        return None
    rename = {
        "时间": "datetime",
        "日期时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "最新价": "latest_price",
        "均价": "avg_price",
    }
    out = df.rename(columns=rename).copy()
    if "datetime" not in out.columns:
        first = out.columns[0]
        out = out.rename(columns={first: "datetime"})
    keep = ["datetime", "open", "close", "high", "low", "volume", "amount", "latest_price", "avg_price"]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    out = out[keep]
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in keep[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.insert(0, "period", str(period))
    out.insert(0, "symbol", _normalize_symbol(symbol))
    out["updated_at"] = datetime.now(CST).isoformat()
    return out.dropna(subset=["datetime"]).drop_duplicates(["symbol", "period", "datetime"], keep="last")


def _minute_cache_path(symbol: str, period: str) -> Path:
    """返回分钟K线 parquet 缓存路径."""
    return MINUTE_DIR / f"{_normalize_symbol(symbol)}_{period}min.parquet"


def _write_minute_bars(df: Any, symbol: str, period: str) -> str | None:
    """写入分钟K线缓存, parquet 不可用时回退 SQLite.

    P2-Q2-fix: L198 返回实际写入介质路径 (str), 失败返回 None.
    调用方以真值判断即可兼容旧的 bool 用法.
    """
    if pd is None or df is None or df.empty:
        return None
    _ensure_quant_dirs()
    if HAS_PARQUET:
        path = _minute_cache_path(symbol, period)
        try:
            df.to_parquet(path, index=False)
            return str(path)
        except Exception as e:
            logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    try:
        conn = _get_minute_db()
        rows = list(df.itertuples(index=False, name=None))
        conn.executemany(
            """
            INSERT OR REPLACE INTO minute_bars (
                symbol, period, datetime, open, close, high, low, volume,
                amount, latest_price, avg_price, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return str(MINUTE_DB_PATH)
    except Exception:
        return None


def _fetch_akshare_minute(symbol: str, period: str) -> Any:
    """调用 AkShare 获取分钟K线."""
    if pd is None:
        return None
    try:
        import akshare as ak

        # P2-Q2-fix: M195 分钟线统一 qfq, 与日线 qfq (fetch_incremental) 基期一致,
        # 避免日内/日线拼接策略在除权日附近价差失真
        raw = ak.stock_zh_a_hist_min_em(symbol=_normalize_symbol(symbol), period=str(period), adjust="qfq")
        return _standardize_minute_bars(raw, symbol, period)
    except Exception:
        return None


def download_minute_bars(symbols: list[str], periods: list[str] = None) -> dict:
    """下载分钟K线并缓存到 ~/.quant_system/minute_bars/{symbol}_{p}min.parquet.

    periods: ['1','5','15','30','60']; 用 AkShare stock_zh_a_hist_min_em;
    如果 parquet 不可用降级到 SQLite.
    """
    periods = periods or ["1", "5", "15", "30", "60"]
    stats = {"success": 0, "failed": 0, "rows": 0, "files": [], "errors": {}}
    for sym in symbols:
        for period in periods:
            df = _fetch_akshare_minute(sym, str(period))
            if df is None:
                stats["failed"] += 1
                stats["errors"][f"{sym}_{period}"] = "akshare failed"
                continue
            # P2-Q2-fix: L198 按实际写入介质记录路径 (parquet 失败回退 SQLite 时
            # 记录 SQLite 路径, 而非仍记 parquet 路径)
            written = _write_minute_bars(df, sym, str(period))
            if written:
                stats["success"] += 1
                stats["rows"] += len(df)
                stats["files"].append(written)
            else:
                stats["failed"] += 1
                stats["errors"][f"{sym}_{period}"] = "cache write failed"
    stats["timestamp"] = datetime.now(CST).isoformat()
    return stats


def get_minute_bars(symbol: str, period: str = "5", start: str = None, end: str = None) -> pd.DataFrame:
    """读取缓存的分钟K线."""
    if pd is None:
        return None
    code = _normalize_symbol(symbol)
    p = str(period)
    df = None
    try:
        path = _minute_cache_path(code, p)
        if HAS_PARQUET and path.exists():
            df = pd.read_parquet(path)
    except Exception:
        df = None
    if df is None:
        try:
            conn = _get_minute_db()
            df = pd.read_sql_query(
                "SELECT * FROM minute_bars WHERE symbol = ? AND period = ? ORDER BY datetime",
                conn,
                params=(code, p),
            )
        except Exception:
            return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if start:
            df = df[df["datetime"] >= pd.to_datetime(start)]
        if end:
            df = df[df["datetime"] <= pd.to_datetime(end)]
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return df


def update_minute_bars_incrementally(symbols: list[str], period: str = "5") -> int:
    """增量更新分钟K线."""
    if pd is None:
        return 0
    updated = 0
    for sym in symbols:
        code = _normalize_symbol(sym)
        old = get_minute_bars(code, period)
        new = _fetch_akshare_minute(code, str(period))
        if new is None:
            continue
        try:
            if old is not None and not old.empty:
                old = old.copy()
                old["datetime"] = pd.to_datetime(old["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                combined = pd.concat([old, new], ignore_index=True)
            else:
                combined = new
            combined = combined.drop_duplicates(["symbol", "period", "datetime"], keep="last")
            before = 0 if old is None or old.empty else len(old)
            if _write_minute_bars(combined, code, str(period)):
                updated += max(len(combined) - before, 0)
        except Exception as e:
            logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
            continue
    return updated


# P2-Q2-fix: M189 每线程独立连接, 避免跨线程共享连接并发写冲突
_MONEY_FLOW_DB_LOCAL = threading.local()


def _get_money_flow_db() -> sqlite3.Connection:
    """连接资金流 SQLite 缓存，每线程独立连接."""
    conn = getattr(_MONEY_FLOW_DB_LOCAL, "conn", None)
    if conn is not None:
        return conn
    _ensure_quant_dirs()
    conn = sqlite3.connect(str(MONEY_FLOW_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stock_money_flow (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            main_buy REAL,
            main_sell REAL,
            main_net REAL,
            super_large_net REAL,
            large_net REAL,
            medium_net REAL,
            retail_buy REAL,
            retail_sell REAL,
            retail_net REAL,
            updated_at TEXT,
            PRIMARY KEY (symbol, date)
        );
        CREATE TABLE IF NOT EXISTS market_money_flow (
            date TEXT PRIMARY KEY,
            sh_main_net REAL,
            sz_main_net REAL,
            total_main_net REAL,
            updated_at TEXT
        );
        """
    )
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(stock_money_flow)").fetchall()
    }
    for col in ["super_large_net", "large_net", "medium_net"]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE stock_money_flow ADD COLUMN {col} REAL")
    conn.commit()
    _MONEY_FLOW_DB_LOCAL.conn = conn
    return conn


def _first_existing(row: Any, names: list[str]) -> Any:
    """返回行中第一个存在的字段值."""
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _first_existing_col(df: Any, names: list[str]) -> str | None:
    """返回 DataFrame 中第一个存在的列名."""
    if df is None:
        return None
    cols = set(getattr(df, "columns", []))
    for name in names:
        if name in cols:
            return name
    return None


def _money_flow_abs_base(df: Any) -> float:
    """资金流占比分母：优先用买卖分量，其次用各档净流入绝对值，避免主力占比退化为 +/-100%."""
    if pd is None or df is None or df.empty:
        return 0.0
    base = 0.0
    component_cols = ["main_buy", "main_sell", "retail_buy", "retail_sell"]
    for col in component_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)
            base += float(values.sum())
    if base > 0:
        return base

    detail_net_cols = ["super_large_net", "large_net", "medium_net", "retail_net"]
    for col in detail_net_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)
            base += float(values.sum())
    if base > 0:
        return base

    net_cols = ["main_net", "retail_net"]
    for col in net_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)
            base += float(values.sum())
    return base


def _standardize_stock_money_flow(df: Any, symbol: str, days: int) -> Any:
    """标准化个股资金流字段."""
    if pd is None or df is None or df.empty:
        return None
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "symbol": _normalize_symbol(symbol),
                "date": _first_existing(row, ["日期", "date"]),
                "main_buy": _first_existing(row, ["主力流入-净额", "主力流入", "超大单流入", "main_buy"]),
                "main_sell": _first_existing(row, ["主力流出", "超大单流出", "main_sell"]),
                "main_net": _first_existing(row, ["主力净流入-净额", "主力净流入", "主力净额", "main_net"]),
                "super_large_net": _first_existing(row, ["超大单净流入-净额", "超大单净流入", "super_large_net"]),
                "large_net": _first_existing(row, ["大单净流入-净额", "大单净流入", "large_net"]),
                "medium_net": _first_existing(row, ["中单净流入-净额", "中单净流入", "medium_net"]),
                "retail_buy": _first_existing(row, ["小单流入", "散户流入", "retail_buy"]),
                "retail_sell": _first_existing(row, ["小单流出", "散户流出", "retail_sell"]),
                "retail_net": _first_existing(row, ["小单净流入-净额", "小单净流入", "散户净流入", "retail_net"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty or "date" not in out.columns:
        return None
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in [
        "main_buy", "main_sell", "main_net", "super_large_net", "large_net",
        "medium_net", "retail_buy", "retail_sell", "retail_net",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["updated_at"] = datetime.now(CST).isoformat()
    out = out.dropna(subset=["date"]).drop_duplicates(["symbol", "date"], keep="last")
    # 日期存在但资金列全空时不能作为 available，否则前端会画出空图并污染证据质量。
    flow_cols = ["main_net", "super_large_net", "large_net", "medium_net", "retail_net"]
    if not any(col in out.columns and out[col].notna().any() for col in flow_cols):
        return None
    return out.sort_values("date").tail(days)


def _money_flow_status(code: str, status: str, *, source: str, rows: int = 0,
                       as_of: str | None = None, message: str = "", error: str | None = None) -> dict:
    result = {
        "status": status, "symbol": code, "source": source,
        "rows": int(rows), "as_of": as_of, "message": message or status,
    }
    if error:
        result["error"] = str(error)[:220]
    return result


def get_stock_money_flow_with_status(symbol: str, days: int = 60, refresh: bool = False) -> tuple[Any, dict]:
    """Return (standardized frame, provenance status) for an individual stock flow.

    The legacy ``get_stock_money_flow`` API remains DataFrame-only. This richer
    method makes the UI distinguish cache absence, provider emptiness, provider
    errors and schema failures instead of collapsing all cases into ``missing``.
    """
    if pd is None:
        code = _normalize_symbol(symbol)
        return None, _money_flow_status(code, "error", source="pandas", message="pandas 不可用")
    code = _normalize_symbol(symbol)
    limit = max(1, int(days))
    cache_status = _money_flow_status(code, "cache_missing", source="local_cache", message="本地无个股资金流缓存")
    if not refresh:
        try:
            conn = _get_money_flow_db()
            cached = pd.read_sql_query(
                "SELECT * FROM (SELECT * FROM stock_money_flow WHERE symbol = ? ORDER BY date DESC LIMIT ?) ORDER BY date ASC",
                conn, params=(code, limit),
            )
            if cached is not None and not cached.empty:
                flow_cols = ["main_net", "super_large_net", "large_net", "medium_net", "retail_net"]
                if any(col in cached.columns and pd.to_numeric(cached[col], errors="coerce").notna().any() for col in flow_cols):
                    as_of = str(cached["date"].iloc[-1])[:10] if "date" in cached.columns else None
                    return cached, _money_flow_status(code, "available", source="local_sqlite", rows=len(cached), as_of=as_of, message="读取本地个股资金流缓存")
        except Exception as exc:
            logger.warning("stock money flow sqlite cache read failed for %s: %s", code, exc)
            cache_status = _money_flow_status(code, "cache_error", source="local_sqlite", message="本地SQLite读取失败，尝试其他来源", error=exc)
        try:
            csv_path = MONEY_FLOW_DIR / f"{code}.csv"
            if csv_path.exists():
                cached = pd.read_csv(csv_path)
                if cached is not None and not cached.empty:
                    flow_cols = ["main_net", "super_large_net", "large_net", "medium_net", "retail_net"]
                    if any(col in cached.columns and pd.to_numeric(cached[col], errors="coerce").notna().any() for col in flow_cols):
                        cached = cached.sort_values("date").tail(limit).reset_index(drop=True)
                        as_of = str(cached["date"].iloc[-1])[:10] if "date" in cached.columns else None
                        return cached, _money_flow_status(code, "available", source="local_csv", rows=len(cached), as_of=as_of, message="读取本地个股资金流CSV缓存")
        except Exception as exc:
            logger.warning("stock money flow csv cache read failed for %s: %s", code, exc)
            cache_status = _money_flow_status(code, "cache_error", source="local_csv", message="本地CSV读取失败，尝试远端来源", error=exc)

    frame, provider_status = download_money_flow_with_status(code, days=limit)
    if frame is not None and not frame.empty:
        return frame, provider_status
    provider_status.setdefault("cache_status", cache_status["status"])
    return None, provider_status


def get_stock_money_flow(symbol: str, days: int = 60, refresh: bool = False) -> Any:
    """兼容旧调用方：读取或下载个股资金流，失败返回 None。"""
    return get_stock_money_flow_with_status(symbol, days=days, refresh=refresh)[0]


def download_money_flow_with_status(symbol: str, days: int = 60) -> tuple[pd.DataFrame | None, dict]:
    """Download individual flow and return its explicit provider status."""
    if pd is None:
        code = _normalize_symbol(symbol)
        return None, _money_flow_status(code, "error", source="pandas", message="pandas 不可用")
    code = _normalize_symbol(symbol)
    try:
        import akshare as ak
    except Exception as exc:
        return None, _money_flow_status(code, "provider_error", source="akshare", message="AkShare 导入失败", error=exc)
    try:
        raw = ak.stock_individual_fund_flow(stock=code, market=_market_prefix(code))
    except Exception as exc:
        logger.warning("stock money flow provider failed for %s: %s", code, exc)
        return None, _money_flow_status(code, "provider_error", source="akshare.stock_individual_fund_flow", message="东方财富个股资金接口请求失败", error=exc)
    if raw is None or getattr(raw, "empty", True):
        return None, _money_flow_status(code, "provider_empty", source="akshare.stock_individual_fund_flow", message="东方财富个股资金接口返回空表")
    try:
        frame = _standardize_stock_money_flow(raw, code, days)
    except Exception as exc:
        return None, _money_flow_status(code, "schema_error", source="akshare.stock_individual_fund_flow", message="个股资金字段标准化失败", error=exc)
    if frame is None or frame.empty:
        return None, _money_flow_status(code, "schema_error", source="akshare.stock_individual_fund_flow", message="返回数据缺少可用日期或主力资金字段")

    _ensure_quant_dirs()
    csv_path = MONEY_FLOW_DIR / f"{code}.csv"
    try:
        frame.to_csv(csv_path, index=False)
    except Exception as exc:
        logger.error(f"[data_pipeline] 操作失败: {exc}", exc_info=True)
    try:
        conn = _get_money_flow_db()
        rows = list(frame.itertuples(index=False, name=None))
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_money_flow (
                symbol, date, main_buy, main_sell, main_net,
                super_large_net, large_net, medium_net,
                retail_buy, retail_sell, retail_net, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    except Exception as exc:
        logger.error(f"[data_pipeline] 操作失败: {exc}", exc_info=True)
    as_of = str(frame["date"].iloc[-1])[:10] if "date" in frame.columns else None
    return frame.reset_index(drop=True), _money_flow_status(code, "available", source="akshare.stock_individual_fund_flow", rows=len(frame), as_of=as_of, message="东方财富个股资金接口返回并已缓存")


def download_money_flow(symbol: str, days: int = 60) -> pd.DataFrame | None:
    """兼容旧调用方的下载接口；详细状态使用 download_money_flow_with_status。"""
    return download_money_flow_with_status(symbol, days=days)[0]


def download_market_money_flow(days: int = 60) -> pd.DataFrame | None:
    """全市场资金流（沪/深/总主力净流入）."""
    if pd is None:
        return None
    try:
        import akshare as ak

        raw = ak.stock_market_fund_flow()
        if raw is None or raw.empty:
            return None
        date_col = "日期" if "日期" in raw.columns else raw.columns[0]

        sh_col = _first_existing_col(raw, [
            "上证-主力净流入-净额", "沪市-主力净流入-净额", "上证主力净流入-净额",
            "上证-主力净流入", "沪市-主力净流入",
        ])
        sz_col = _first_existing_col(raw, [
            "深证-主力净流入-净额", "深市-主力净流入-净额", "深证主力净流入-净额",
            "深证-主力净流入", "深市-主力净流入",
        ])
        total_col = _first_existing_col(raw, [
            "主力净流入-净额", "主力净流入", "净流入", "主力净额", "全市场-主力净流入-净额",
        ])

        keep = [date_col]
        rename = {date_col: "date"}
        if sh_col:
            keep.append(sh_col)
            rename[sh_col] = "sh_main_net"
        if sz_col:
            keep.append(sz_col)
            rename[sz_col] = "sz_main_net"
        if total_col:
            keep.append(total_col)
            rename[total_col] = "total_main_net"
        if len(keep) == 1:
            return None

        out = raw[list(dict.fromkeys(keep))].rename(columns=rename).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ["sh_main_net", "sz_main_net"]:
            if col not in out.columns:
                out[col] = None
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "total_main_net" in out.columns:
            out["total_main_net"] = pd.to_numeric(out["total_main_net"], errors="coerce")
        else:
            out["total_main_net"] = out["sh_main_net"].fillna(0.0) + out["sz_main_net"].fillna(0.0)
        out["updated_at"] = datetime.now(CST).isoformat()
        out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").tail(days)
        try:
            conn = _get_money_flow_db()
            rows = list(out[["date", "sh_main_net", "sz_main_net", "total_main_net", "updated_at"]].itertuples(index=False, name=None))
            conn.executemany(
                """
                INSERT OR REPLACE INTO market_money_flow (
                    date, sh_main_net, sz_main_net, total_main_net, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        except Exception as e:
            logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
        return out.reset_index(drop=True)
    except Exception:
        return None


def get_main_force_ratio(symbol: str, days: int = 20) -> float:
    """近N日主力资金净流入占比 [-100%, 100%]."""
    if pd is None:
        return 0.0
    code = _normalize_symbol(symbol)
    df = None
    try:
        conn = _get_money_flow_db()
        df = pd.read_sql_query(
            "SELECT * FROM stock_money_flow WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            conn,
            params=(code, days),
        )
    except Exception:
        df = None
    if df is None or df.empty:
        df = download_money_flow(code, days)
    if df is None or df.empty:
        return 0.0
    try:
        recent = df.sort_values("date").tail(days)
        main_net = pd.to_numeric(recent["main_net"], errors="coerce").fillna(0.0).sum()
        base = _money_flow_abs_base(recent)
        if base <= 0:
            return 0.0
        ratio = main_net / base * 100.0
        return max(min(float(ratio), 100.0), -100.0)
    except Exception:
        return 0.0


def fetch_with_retry(symbol: str, sources: list[str] = None) -> pd.DataFrame | None:
    """带优先级和重试的数据获取."""
    selected = sources or DATA_SOURCE_PRIORITY
    fetchers = {
        "akshare": lambda sym: _fetch_akshare_daily(sym),
        "tencent": lambda sym: _fetch_tencent_daily(sym),
        "sina": lambda sym: _fetch_sina_daily(sym),
    }
    for source in selected:
        fetcher = fetchers.get(source)
        if fetcher is None:
            continue
        for attempt in range(MAX_RETRIES):
            try:
                df = fetcher(symbol)
                if df is not None and not df.empty:
                    return df
                # 审计 2026-08-16：fetcher 内部可能把异常吞成 None，
                # 空/None 结果也视为失败计入重试，否则 MAX_RETRIES 形同虚设
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"[data_pipeline] {source} {symbol} 第{attempt + 1}次返回空/None，重试")
            except Exception as e:
                logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                try:
                    _time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                except Exception as e:
                    logger.error(f"[data_pipeline] 操作失败: {e}", exc_info=True)
    return None
