"""
data_store.py — V8 DataStore 数据存储层

统一数据存取入口，替代各模块满天飞的 fetch_daily() 直接调用。

职责:
  1. 统一入口  — DataStore.get(symbol) / .get_many(symbols)
  2. 数据质量  — schema 校验、价格合理性、日期连续性、NaN 检测
  3. 新鲜度跟踪 — 每个 symbol 记录最后拉取时间，超过阈值自动刷新
  4. 缓存管理  — 检查/刷新/清理缓存
  5. 前复权保证 — 统一为 qfq，不依赖调用方传入
  6. 缺失填补  — 交易日历对齐 + 前向填充

用法:
  store = DataStore()
  df = store.get("002714", days=480)      # 自动缓存 + 质量校验
  dfs = store.get_many(["600519","000858"])  # 批量读取
  store.refresh("002714")                  # 强制刷新缓存
  store.status()                           # 打印缓存健康状态
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_data_store")

# Allow running as script
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── 复用现有 data.py 的底层实现 ──
from quant_system.data import (
    fetch_daily as _fetch_daily_legacy,
    normalize_symbol as _normalize,
    init_db as _init_db,
    _query_daily_db as _query_db,
    _write_daily_db as _write_db,
    _get_conn,
    DB_PATH,
    CACHE_DIR,
)

# V6 fix (Q1 CRITICAL: data_store.py:224-243):
#   _fill_missing_dates 改用 A股真实交易日历（market_clock 为全局唯一交易日历源）
#   做 reindex/ffill，不再用自然日 freq="D" 注入周末/节假日合成K线。
#   market_clock 内部裸导入 market_rules，需要 quant_system 目录在 sys.path。
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
try:
    from quant_system.market_clock import get_trade_calendar, is_trading_session
except Exception:
    try:
        from market_clock import get_trade_calendar, is_trading_session  # type: ignore[no-redef]
    except Exception:
        get_trade_calendar = None
        is_trading_session = None
        logger.warning(
            "[DataStore] market_clock 导入失败，_fill_missing_dates 将降级为工作日粗筛"
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FRESHNESS_HOURS = 48.0         # 缓存有效时长（小时，个人系统放宽）
MAX_DAYS_DEFAULT = 480         # 默认拉取天数
TRADING_DAYS_PER_YEAR = 242

# V4.1: 分钟级K线支持
MINUTE_PERIODS = ["1", "5", "15", "30", "60"]  # 支持的分钟周期
MINUTE_KLINE_CACHE = Path.home() / ".quant_system" / "minute_klines"  # 分钟K线缓存目录
TZ = "Asia/Shanghai"
CST = timezone(timedelta(hours=8))  # 北京时间（分钟K线新鲜度判断用）

# 数据质量阈值
MIN_PRICE = 0.01               # 最低合理价格
MAX_PRICE_CHG_PCT = 40.0       # 单日涨跌幅上限（%
MAX_VOLUME_RATIO = 100.0       # 量比上限（相对前日）
MIN_TRADING_DAYS = 20          # 最少需要交易日数

# Required columns after normalization
REQUIRED_COLS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "pct_chg",
]


# ---------------------------------------------------------------------------
# Freshness store (SQLite)
# ---------------------------------------------------------------------------

def _init_freshness_table() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_freshness (
            symbol TEXT PRIMARY KEY,
            last_fetch TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            last_date TEXT,
            source TEXT DEFAULT 'api',
            quality_score REAL DEFAULT 1.0
        )
    """)
    conn.commit()


def _record_fetch(symbol: str, df: pd.DataFrame, source: str = "api") -> None:
    _init_freshness_table()
    conn = _get_conn()
    # P2-Q1-fix (Q1-M001): date 统一为 datetime64 后，格式化 last_date 为
    #   "YYYY-MM-DD"，避免 status() 显示 "2024-08-02 00:00:00"。
    _last = df["date"].iloc[-1] if not df.empty else None
    last_date = None
    if _last is not None:
        last_date = (
            _last.strftime("%Y-%m-%d")
            if isinstance(_last, (pd.Timestamp, datetime))
            else str(_last)[:10]
        )
    conn.execute(
        """INSERT OR REPLACE INTO data_freshness
           (symbol, last_fetch, row_count, last_date, source, quality_score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            symbol,
            datetime.now(CST).isoformat(),
            len(df),
            last_date,
            source,
            _quality_score(df),
        ),
    )
    conn.commit()


def _freshness(symbol: str) -> dict[str, Any] | None:
    _init_freshness_table()
    row = _get_conn().execute(
        "SELECT * FROM data_freshness WHERE symbol = ?", (symbol,)
    ).fetchone()
    if row is None:
        return None
    keys = ["symbol", "last_fetch", "row_count", "last_date", "source", "quality_score"]
    return dict(zip(keys, row))


def _is_stale(symbol: str) -> bool:
    """True if symbol's cache is older than FRESHNESS_HOURS or missing.

    Timestamps are compared in UTC. Stored values may be timezone-aware or
    naive (legacy rows and tests); naive values are interpreted as local time
    and normalised, so staleness does not depend on the host timezone.
    """
    info = _freshness(symbol)
    if info is None:
        return True
    try:
        last = datetime.fromisoformat(info["last_fetch"])
        if last.tzinfo is None:
            last = last.astimezone()  # naive -> local tz, then normalise below
        age_h = (datetime.now(timezone.utc) - last.astimezone(timezone.utc)).total_seconds() / 3600
        return age_h > FRESHNESS_HOURS
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def _quality_score(df: pd.DataFrame) -> float:
    """Return a score 0–1 based on data quality heuristics."""
    if df.empty:
        return 0.0
    issues = 0.0
    checks = 0.0

    # NaN check
    for col in REQUIRED_COLS:
        if col in df.columns:
            checks += 1
            na_frac = df[col].isna().mean()
            issues += na_frac

    # Zero/negative prices
    for pcol in ("close", "open", "high", "low"):
        if pcol in df.columns:
            checks += 1
            bad = (df[pcol] <= 0).mean()
            issues += bad

    # Extreme pct_chg
    if "pct_chg" in df.columns:
        checks += 1
        extreme = (df["pct_chg"].abs() > 30).mean()
        issues += extreme

    # Date gap check
    if len(df) >= 2 and "date" in df.columns:
        checks += 0.5
        dates = pd.to_datetime(df["date"])
        gaps = (dates.diff().dt.days > 5).sum()
        issues += gaps / max(len(df), 1) * 2  # penalty for large gaps

    return max(0.0, 1.0 - issues / max(checks, 1))


def _validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Run quality checks, log warnings, return cleaned DataFrame."""
    if df.empty:
        raise ValueError(f"[DataStore] {symbol}: empty DataFrame")

    df = df.copy()

    # P2-Q1-fix (Q1-M001): 统一内部日期契约为 datetime64
    #   fill_gaps=True/False 两条返回路径的 date 列 dtype 一致，
    #   下游不再需要兼容 string 与 datetime64 两种格式。
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        # 审计 2026-08-16：先按日期升序，再计算 pct_chg，避免乱序导致前视/错误涨跌幅
        df = df.sort_values("date").reset_index(drop=True)

    # Ensure required columns; compute pct_chg if missing
    if 'pct_chg' not in df.columns and 'close' in df.columns:
        df['pct_chg'] = df['close'].pct_change() * 100.0
        df['pct_chg'] = df['pct_chg'].fillna(0.0)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[DataStore] {symbol}: missing columns {missing}")

    # Check for critical NaNs
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            logger.warning("[DataStore] %s: %d NaN(s) in '%s', ffill only (bfill would cause look-ahead bias)", symbol, n_nan, col)
            # V4.1 re-audit fix: ffill only — bfill from future data introduces look-ahead bias
            df[col] = df[col].ffill()

    # P2-Q1-fix (Q1-M002): 前导 NaN 行在 ffill 后仍为 NaN——
    #   `NaN <= MIN_PRICE` 恒为 False，价格过滤删不掉 → 原逻辑 fill_gaps=False
    #   会返回含 NaN 价格的帧。对价格列 dropna 兜底。
    df = df.dropna(subset=price_cols)

    # Remove rows with zero/negative close
    bad = df["close"] <= MIN_PRICE
    if bad.any():
        n_bad = bad.sum()
        df = df[~bad].copy()

    if df.empty:
        raise ValueError(f"[DataStore] {symbol}: all rows invalid after price filter")

    # Ensure dates are sorted ascending
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def _normalize_date_arg(d: str | None) -> str | None:
    """把 YYYY-MM-DD / YYYYMMDD / datetime / date 统一为 YYYYMMDD 字符串。"""
    if d is None:
        return None
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y%m%d")
    s = str(d).strip().replace("-", "").replace("/", "")
    if len(s) == 8 and s.isdigit():
        return s
    try:
        return pd.to_datetime(s).strftime("%Y%m%d")
    except Exception:
        return None


def _truncate_to_end(df: pd.DataFrame, end_str: str | None) -> pd.DataFrame:
    """按 end (YYYYMMDD) 截断数据到 ≤ end；end 为空时原样返回。"""
    if df is None or df.empty or not end_str or "date" not in df.columns:
        return df
    end_ts = pd.to_datetime(end_str, format="%Y%m%d", errors="coerce")
    if pd.isna(end_ts):
        return df
    dates = pd.to_datetime(df["date"])
    return df[dates <= end_ts].copy()


def _fill_missing_dates(df: pd.DataFrame, max_gap_days: int = 10) -> pd.DataFrame:
    """Forward-fill for trading-day gaps, limiting ffill to avoid over-extending during holidays.

    V6 fix (Q1 CRITICAL): V4.1 版用自然日 `freq="D"` 重建索引，会把周末/节假日
    注入合成K线（价格/成交量 ffill 自前日）——实测 2 个交易日被扩成 4 行，
    污染收益率、量能、交易日计数。
    V6 改为：
      - 优先用 A股真实交易日历（market_clock.get_trade_calendar()，全局唯一源）
        在数据区间内 reindex/ffill，周末/节假日行不进入结果；
      - 交易日历不可用或未覆盖区间时，降级为「工作日粗筛」（周一~周五）并告警
        （可见降级，不静默）；
      - 保留 ffill(limit=max_gap_days) 语义：长假缺口不跨期填充。
    """
    if df.empty or "date" not in df.columns:
        return df

    dates = pd.to_datetime(df["date"])
    dmin, dmax = dates.min(), dates.max()

    existing = set(dates.dt.strftime("%Y-%m-%d"))
    cal_days: set[str] = set()
    if get_trade_calendar is not None:
        try:
            cal = get_trade_calendar()
            if cal:
                cal_days = {d for d in cal if dmin <= pd.to_datetime(d) <= dmax}
        except Exception as exc:
            logger.warning("[DataStore] 交易日历获取异常(%s)，_fill_missing_dates 降级为工作日粗筛", exc)
    if not cal_days:
        wd = pd.date_range(dmin, dmax, freq="D")
        cal_days = set(wd[wd.dayofweek < 5].strftime("%Y-%m-%d"))
        logger.warning(
            "[DataStore] 交易日历不可用或未覆盖 %s~%s，_fill_missing_dates 降级为工作日粗筛",
            dmin.date(), dmax.date(),
        )

    # 并集：已有真实行 ∪ 真实交易日（既不丢数据，也不注入周末/节假日）
    full_range = pd.to_datetime(sorted(existing | cal_days))
    orig_idx = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.reindex(full_range)
    # V4.1 audit fix (fill_gaps 停牌合成K线): 交易日历中缺失的行 = 停牌日，
    # 价格列 ffill 保持连续（技术指标可用），但 volume/amount 必须置 0 ——
    # 停牌无成交。此前 ffill 把前日成交量也带入，回测引擎(bar.volume<=1e-8 不成交)
    # 的保护失效，停牌日会被误当可交易日成交。
    synthetic_mask = ~df.index.isin(orig_idx)
    if synthetic_mask.any():
        for c in ("volume", "amount"):
            if c in df.columns:
                df.loc[synthetic_mask, c] = 0.0
        if "turnover" in df.columns:
            df.loc[synthetic_mask, "turnover"] = 0.0
    # Only forward-fill small gaps; leave long gaps (holidays) as NaN
    # V4.1 re-audit fix: ffill only (bfill from future data introduces look-ahead bias)
    df = df.ffill(limit=max_gap_days)
    # V6 fix: 只按必需价格列去 NaN。
    #   实测 quant.db 中部分股票的 turnover 列为 NULL（写库时无该列），
    #   全列 dropna() 会把整表清空（实测 182 行 → 0 行）；
    #   turnover/amount/pct_chg 属可选列，缺失时保留行（降级可见，不伪造数据）。
    if "pct_chg" in df.columns:
        df["pct_chg"] = df["pct_chg"].fillna(0.0)
    price_req = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df.dropna(subset=price_req)
    df.index.name = "date"
    df = df.reset_index()
    # P2-Q1-fix (Q1-M001): 与 fill_gaps=False（缓存/API 路径）一致，date 保持 datetime64。
    #   V6 之前这里 .dt.strftime("%Y-%m-%d") 转字符串，导致同一 API 按 fill_gaps
    #   输出两种 dtype；datetime64 是统一内部契约，序列化层再转字符串。
    return df


# ---------------------------------------------------------------------------
# Core DataStore
# ---------------------------------------------------------------------------

class DataStore:
    """Unified data access layer.

    Usage:
        ds = DataStore()
        df = ds.get("002714")
        dfs = ds.get_many(["600519", "000858"])
        ds.refresh("002714")
        ds.refresh_all()
        ds.prune(days=90)
        print(ds.status())
    """

    def __init__(
        self,
        freshness_hours: float = FRESHNESS_HOURS,
        default_days: int = MAX_DAYS_DEFAULT,
    ) -> None:
        self.freshness_hours = freshness_hours
        self.default_days = default_days
        self.klines_db_path = str(Path.home() / '.quant_system' / 'klines.db')
        _init_db()
        _init_freshness_table()

    # ── V4.1 arch fix: data bridge between klines.db and quant.db ──

    def _bridge_from_klines_db(self, symbol: str, latest_date_in_quant: str) -> int:
        """尝试从 data_pipeline 的 klines.db 同步更新数据到 quant.db.

        Args:
            symbol: 归一化后的股票代码.
            latest_date_in_quant: 当前 quant.db 中该股票的最新日期 (YYYYMMDD).

        Returns:
            int: 桥接的行数 (0 表示无新数据或桥接失败).
        """
        if not Path(self.klines_db_path).exists():
            return 0
        try:
            # klines.db 存储日期格式为 YYYY-MM-DD, quant.db 为 YYYYMMDD
            latest_k = latest_date_in_quant[:4] + "-" + latest_date_in_quant[4:6] + "-" + latest_date_in_quant[6:8]
            conn_k = sqlite3.connect(self.klines_db_path)
            rows = conn_k.execute(
                "SELECT symbol, date, open, close, high, low, volume, amount, "
                "pct_change, turnover FROM daily_kline "
                "WHERE symbol = ? AND date > ? "
                "ORDER BY date ASC",
                (symbol, latest_k),
            ).fetchall()
            conn_k.close()
            if not rows:
                return 0
            # Map columns: klines.db -> quant.db (date: YYYY-MM-DD -> YYYYMMDD)
            upsert_rows = []
            for r in rows:
                d = r[1].replace("-", "")
                upsert_rows.append((
                    r[0],    # symbol
                    d,       # date (YYYYMMDD)
                    r[2],    # open
                    r[3],    # close (klines.db col order: open, close, high, low)
                    r[4],    # high
                    r[5],    # low
                    r[6],    # volume
                    r[7],    # amount
                    r[9],    # turnover (index 9)
                    r[8],    # pct_change -> pct_chg (index 8)
                ))
            conn_q = _get_conn()
            conn_q.executemany(
                "INSERT OR REPLACE INTO daily_klines "
                "(symbol, date, open, close, high, low, volume, amount, turnover, pct_chg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                upsert_rows,
            )
            conn_q.commit()
            logger.info("[bridge] %s: synced %d rows from klines.db (%s+)",
                        symbol, len(upsert_rows), latest_date_in_quant)
            return len(upsert_rows)
        except Exception as exc:
            logger.warning("[bridge] %s: klines.db sync failed: %s", symbol, exc)
            return 0

    # ── Public API ──────────────────────────────────────────────────────

    def get(
        self,
        symbol: str,
        days: int | None = None,
        start: str | None = None,
        force_refresh: bool = False,
        fill_gaps: bool = True,
        end: str | None = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """Fetch validated daily data for *symbol*.

        Args:
            symbol:  A-share stock code.
            days:    Minimum number of trading days.  Defaults to self.default_days.
            start:   Explicit start date YYYYMMDD.  Overrides *days*.
            force_refresh:  Skip cache, re-fetch from API.
            fill_gaps:     Forward-fill missing trading days.
            end:     Explicit end date YYYYMMDD (or YYYY-MM-DD).  Default: today.
                     V6 fix: 因子/回测调用方传历史 end 以严格截取 ≤ end 的数据，
                     杜绝前视偏差（Q5 CRITICAL: factor_model.py:203,210）。
            adjust: 复权方式 ("qfq" 前复权 / "" 不复权，默认 qfq)。
                     P2-Q1-fix (Q1-M004): 透传给 _fetch_daily_legacy；
                     缓存恒为 qfq，请求非 qfq 时强制走 API 重拉。

        Returns:
            DataFrame with columns [date, open, high, low, close, volume,
                                     amount, pct_chg, turnover].
        """
        symbol = _normalize(symbol)
        adjust = adjust or "qfq"

        # Determine start date
        if start is None:
            ndays = days or self.default_days
            # ~1.4 calendar days per trading day to be safe
            cal_days = int(ndays * 1.4) + 20
            start_dt = datetime.now(CST).replace(tzinfo=None) - timedelta(days=cal_days)
            start = start_dt.strftime("%Y%m%d")
        end_str = _normalize_date_arg(end) or datetime.now(CST).replace(tzinfo=None).strftime("%Y%m%d")
        today_str = datetime.now(CST).replace(tzinfo=None).strftime("%Y%m%d")

        # ── V10 fix (2026-08-07): data_warehouse parquet 全量行情优先 ──
        #   SQLite daily_klines 仅缓存过历史分析用过的 ~41 只股票；
        #   data_warehouse/kline/ 有 5204 只全量（2020~今），本机虚拟机网络受限
        #   （东财/新浪接口不通），走 API 刷新会失败或卡住。
        #   优先读 parquet，仅当文件缺失/日期不足时才回退 SQLite+API。
        pq_stale_fallback = None
        if not force_refresh:
            try:
                pq = self._read_parquet_klines(symbol, start, end_str)
                # V10-fix2: 只要非空即返回（260 天窗口对 2020 起的 parquet 通常
                #   只有 ~230 行——交易日密度 + 节假日；窗口不足由调用方处理）
                if pq is not None and not pq.empty:
                    pq_max = pd.to_datetime(pq["date"]).max()
                    # 2026-08-10 审计(Critical): 原逻辑完全绕过新鲜度检查，
                    # parquet 停更后永远返回旧数据。改为：末端距今天 >7 天视为陈旧，
                    # 陈旧时继续走 SQLite/API 刷新；刷新失败由 pq_stale_fallback 兜底。
                    fresh = (pd.Timestamp.now() - pq_max).days <= 7
                    if fresh or end_str < today_str:
                        return pq
                    pq_stale_fallback = pq
                    logger.warning("[DataStore] %s parquet 末端 %s 陈旧(>7天)，尝试刷新",
                                   symbol, pq_max.date())
            except Exception as _pq_exc:  # noqa: BLE001
                logger.debug("[DataStore] parquet 读取失败 %s: %s", symbol, _pq_exc)

        today_str = datetime.now(CST).replace(tzinfo=None).strftime("%Y%m%d")
        # 请求区间完全位于历史（end < 今天）时，数据不可变，无需因 stale 而刷新
        range_is_historical = end_str < today_str

        # Freshness check
        stale = _is_stale(symbol) if not force_refresh else True

        # Cache lookup (non-stale or forced-refresh fallback)
        cached = _query_db(symbol, start, end_str)
        has_cached = not cached.empty and len(cached) >= (days or 20)
        # P2-Q1-fix (Q1-M004): 缓存恒为 qfq，请求非 qfq 时缓存不可用，强制 API 重拉
        if adjust != "qfq":
            has_cached = False
        # V6 fix (Q1 HIGH: data_store.py:374): 仅传 start 时校验缓存最早日期 ≤ start，
        # 不足则视为覆盖不完整，走 API 补齐，避免静默返回部分历史。
        # 用交易日历判定「start 之后第一个交易日」作精确比较：start 若为周末/节假日
        # （如 2026-01-01），缓存从下一个交易日开始即视为覆盖完整。
        if has_cached:
            cached_min = cached["date"].min()
            cached_min_s = (
                cached_min.strftime("%Y%m%d")
                if isinstance(cached_min, (pd.Timestamp, datetime))
                else str(cached_min)[:8]
            )
            expected_first: str | None = None
            if get_trade_calendar is not None:
                try:
                    _start_dt = pd.to_datetime(start, format="%Y%m%d")
                    _cands = sorted(
                        d for d in get_trade_calendar() if pd.to_datetime(d) >= _start_dt
                    )
                    if _cands:
                        expected_first = _cands[0].replace("-", "")
                except Exception:
                    expected_first = None
            if expected_first is not None:
                if cached_min_s > expected_first:
                    has_cached = False
            elif cached_min_s > start:
                has_cached = False

        # ── 新鲜缓存（或历史区间）：直接返回 ──
        if not force_refresh and has_cached and (not stale or range_is_historical):
            cached = _validate(cached, symbol)
            cached = _truncate_to_end(cached, end_str)
            if fill_gaps:
                cached = _fill_missing_dates(cached)
            return cached

        # ── 过期缓存且区间延伸至今天：先尝试 klines.db 桥接，桥接成功则短路 ──
        if not force_refresh and has_cached:
            latest_q = cached["date"].max()
            # V6 fix (Q1 HIGH: data_store.py:380-386):
            #   V4.1 版 str(max()) 得 "2024-08-02 00:00:00"，切片得垃圾串 "2024--0-8-"
            #   → 桥接每次执行且 WHERE date > 垃圾串 匹配当月起全部行。
            latest_q_s = (
                latest_q.strftime("%Y%m%d")
                if isinstance(latest_q, (pd.Timestamp, datetime))
                else str(latest_q)[:8]
            )
            if latest_q_s < today_str:
                bridged = self._bridge_from_klines_db(symbol, latest_q_s)
                if bridged > 0:
                    # 重新读取 quant.db 以获取完整最新数据
                    cached = _query_db(symbol, start, end_str)
                    _record_fetch(symbol, cached, source="klines_bridge")
                    cached = _validate(cached, symbol)
                    cached = _truncate_to_end(cached, end_str)
                    if fill_gaps:
                        cached = _fill_missing_dates(cached)
                    return cached

        # ── 无缓存 / 强制刷新 / 过期且桥接无新数据 → API 刷新 ──
        # V6 fix (Q1 HIGH: data_store.py:370-421): stale 缓存不再永远返回旧数据，
        # 桥接无新数据时落入 API 刷新分支（原逻辑只在 force/无缓存时刷新）
        if force_refresh or not has_cached or stale:
            for attempt in range(2):
                try:
                    df = _fetch_daily_legacy(
                        symbol, start=start, use_cache=False, adjust=adjust, retries=3
                    )
                    if df is not None and not df.empty:
                        df = _validate(df, symbol)
                        # 落库完整数据（不截断），返回时再按 end 截断 → 缓存不丢新数据
                        _write_db(symbol, df)
                        _record_fetch(symbol, df)
                        df = _truncate_to_end(df, end_str)
                        if fill_gaps:
                            df = _fill_missing_dates(df)
                        return df
                    break
                except Exception:
                    if attempt == 1:
                        break
                    time.sleep(1)

        # 全失败
        if pq_stale_fallback is not None:
            # parquet 陈旧但 API 刷新失败 → 降级返回旧数据（带日志，比崩溃好）
            logger.warning("[DataStore] %s 刷新失败，降级返回陈旧 parquet(末端 %s)",
                           symbol, pd.to_datetime(pq_stale_fallback["date"]).max().date())
            pq_stale_fallback = _truncate_to_end(pq_stale_fallback, end_str)
            # 审计 2026-08-16：降级数据附加 stale 标记，下游可感知正在使用旧数据
            pq_stale_fallback.attrs["stale"] = True
            pq_stale_fallback.attrs["stale_reason"] = "parquet_陈旧且API刷新失败"
            return pq_stale_fallback
        if has_cached:
            cached = _validate(cached, symbol)
            cached = _truncate_to_end(cached, end_str)
            if fill_gaps:
                cached = _fill_missing_dates(cached)
            return cached
        raise RuntimeError(f"API和缓存均不可用: {symbol}")

    # ── V10: data_warehouse parquet 全量行情读取（本机无网环境主数据源）──
    _PARQUET_KLINE_DIR = ROOT / "data_warehouse" / "kline"

    def _read_parquet_klines(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """从 data_warehouse/kline/{symbol}.parquet 读取行情，裁剪到 [start, end]。

        返回列: date/open/high/low/close/volume/amount/pct_chg/turnover（与 get() 契约一致）；
        文件不存在或读取失败返回空 DataFrame。
        """
        try:
            p = self._PARQUET_KLINE_DIR / f"{symbol}.parquet"
            if not p.exists():
                return pd.DataFrame()
            df = pd.read_parquet(p)
            if df.empty:
                return df
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            # 审计 2026-08-16：先排序再补 pct_chg，避免 parquet 乱序造成前视
            df = df.sort_values("date").reset_index(drop=True)
            # 裁剪区间
            if end is not None:
                end_dt = pd.to_datetime(end, format="%Y%m%d")
                df = df[df["date"] <= end_dt]
            if start is not None:
                start_dt = pd.to_datetime(start, format="%Y%m%d")
                df = df[df["date"] >= start_dt]
            if df.empty:
                return df
            # 列对齐：parquet 无 pct_chg，按收盘价计算
            if "pct_chg" not in df.columns:
                df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100.0
            keep = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
            if "turnover" in df.columns:
                keep.append("turnover")
            for col in keep:
                if col not in df.columns:
                    df[col] = 0.0
            out = df[keep].sort_values("date").reset_index(drop=True)
            # 审计 2026-08-16：parquet 主路径读取也更新 freshness 表（source=parquet），
            # 避免停更时 _is_stale 仍按上次拉取时间误判新鲜
            try:
                _record_fetch(symbol, out, source="parquet")  # 模块级函数
            except Exception:
                pass
            return out
        except Exception as _e:  # noqa: BLE001
            logger.debug("[DataStore] parquet 读取异常 %s: %s", symbol, _e)
            return pd.DataFrame()

    def get_many(
        self,
        symbols: list[str],
        days: int | None = None,
        start: str | None = None,
        force_refresh: bool = False,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch multiple symbols in parallel.  Returns {symbol: DataFrame}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # V6 fix (Q1 MEDIUM: data_store.py:437): 空列表直接返回空 dict
        # （原 max_workers=min(8, 0)=0 → ThreadPoolExecutor 抛 ValueError）
        if not symbols:
            return {}

        result: dict[str, pd.DataFrame] = {}
        errors: list[str] = []

        def _fetch_one(sym: str) -> tuple[str, pd.DataFrame | None]:
            try:
                df = self.get(sym, days=days, start=start, force_refresh=force_refresh, end=end)
                return sym, df
            except Exception as exc:
                return sym, exc

        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym, data = fut.result()
                if isinstance(data, Exception):
                    errors.append(f"{sym}: {data}")
                else:
                    result[sym] = data

        if errors and not result:
            raise RuntimeError(f"[DataStore] get_many all failed: {'; '.join(errors)}")

        # V6 fix (Q1 CRITICAL: data_store.py:450-479): 跨股票日期对齐
        #   V4.1 版 `dates = set(pd.to_datetime(df["date"]))` 生成 Timestamp 集合，
        #   但 `df["date"].isin(common_dates)` 的 date 列是 _fill_missing_dates 输出的
        #   字符串 "YYYY-MM-DD"，字符串与 Timestamp 永不相等 → 所有行被过滤成空帧。
        #   V6 两侧统一为 "YYYY-MM-DD" 字符串再求交集。
        if len(result) > 1:
            common_dates = None
            for sym, df in result.items():
                if df is None or df.empty:
                    continue
                if isinstance(df.index, pd.DatetimeIndex):
                    dates = set(df.index.strftime("%Y-%m-%d"))
                elif "date" in df.columns:
                    dates = set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
                else:
                    continue
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates &= dates
            if common_dates:
                for sym in list(result.keys()):
                    df = result[sym]
                    if df is None or df.empty:
                        continue
                    if isinstance(df.index, pd.DatetimeIndex):
                        result[sym] = df[df.index.strftime("%Y-%m-%d").isin(common_dates)].copy()
                    elif "date" in df.columns:
                        result[sym] = df[
                            pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").isin(common_dates)
                        ].copy()
        return result

    def get_kline(
        self,
        symbol: str,
        days: int = 60,
    ) -> list[dict]:
        """Convenience: return kline as list-of-dicts (for JSON API)."""
        df = self.get(symbol, days=days)
        return df.to_dict("records")

    def refresh(self, symbol: str) -> dict[str, Any]:
        """Force re-fetch for one symbol.  Returns freshness info."""
        df = self.get(symbol, force_refresh=True)
        return self._info(symbol)

    def refresh_all(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Refresh all symbols in the freshness table (or given list)."""
        if symbols is None:
            rows = _get_conn().execute(
                "SELECT symbol FROM data_freshness ORDER BY last_fetch ASC"
            ).fetchall()
            symbols = [r[0] for r in rows]
        results: dict[str, Any] = {"refreshed": 0, "failed": 0, "errors": []}
        for sym in symbols:
            try:
                self.get(sym, force_refresh=True)
                results["refreshed"] += 1
            except Exception as exc:
                results["failed"] += 1
                results["errors"].append(f"{sym}: {exc}")
        return results

    def prune(self, days: int = 180) -> dict[str, Any]:
        """Remove symbols not fetched in *days* from freshness table."""
        cutoff = (datetime.now(CST).replace(tzinfo=None) - timedelta(days=days)).isoformat()
        conn = _get_conn()
        deleted = conn.execute(
            "DELETE FROM data_freshness WHERE last_fetch < ?", (cutoff,)
        ).rowcount
        conn.commit()
        return {"pruned": deleted}

    def status(self) -> dict[str, Any]:
        """Full cache health report."""
        _init_freshness_table()
        rows = _get_conn().execute(
            "SELECT * FROM data_freshness ORDER BY last_fetch DESC"
        ).fetchall()
        now = datetime.now(CST).replace(tzinfo=None)
        entries = []
        for r in rows:
            keys = ["symbol", "last_fetch", "row_count", "last_date", "source", "quality_score"]
            info = dict(zip(keys, r))
            try:
                age_h = (now - datetime.fromisoformat(info["last_fetch"])).total_seconds() / 3600
                info["age_h"] = round(age_h, 1)
                info["stale"] = age_h > self.freshness_hours
            except Exception:
                info["age_h"] = -1
                info["stale"] = True
            entries.append(info)

        db_size = DB_PATH.stat().st_size / 1024 if DB_PATH.exists() else 0
        return {
            "total_symbols": len(entries),
            "stale_count": sum(1 for e in entries if e["stale"]),
            "fresh_count": sum(1 for e in entries if not e["stale"]),
            "avg_quality": round(
                sum(e["quality_score"] for e in entries) / max(len(entries), 1), 3
            ),
            "db_size_kb": round(db_size, 1),
            "cache_dir_size_kb": round(
                sum(f.stat().st_size for f in CACHE_DIR.glob("*.csv")) / 1024, 1
            ) if CACHE_DIR.exists() else 0,
            "entries": entries[:50],  # top 50
        }

    # ── V11 (2026-08-07): 数据集元数据门面（骨架收口 P0-2）──
    #   所有数据集统一登记，schema_of 可查列定义，禁止散落直读 parquet。

    _DATASETS: dict[str, dict] = {
        # ── 个股维度（按 code 分文件）──
        "kline": {
            "path": "kline/{code}.parquet",
            "freq": "日",
            "columns": ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share", "pct_chg"],
            "source": "tencent/eastmoney",
        },
        "valuation": {
            "path": "valuation/{code}.parquet",
            "freq": "日",
            "columns": ["date", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"],
            "source": "baostock",
        },
        "financial": {
            "path": "financial/{code}.parquet",
            "freq": "季",
            "columns": ["date", "roe", "gross_margin", "net_margin", "revenue_yoy", "profit_yoy", "debt_ratio"],
            "source": "eastmoney/akshare",
        },
        # ── 截面/日频（按日期分文件）──
        "feature_store": {
            "path": "feature_store/*.parquet", "freq": "日",
            "columns": ["code", "date", "close", "pct_chg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "mom20", "mom60", "mom120"],
            "source": "warehouse",
        },
        "zt_pool": {"path": "market/zt_pool/*.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "zt_pool_subnew": {"path": "market/zt_pool_subnew/*.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "zt_pool_strong": {"path": "market/zt_pool_strong/*.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "zt_pool_dtgc": {"path": "market/zt_pool_dtgc/*.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "margin": {"path": "market/margin_detail_{sh,sz}/", "freq": "日", "columns": [], "source": "sse/szse"},
        "margin_summary": {"path": "market/market_margin_{sh,sz}.parquet", "freq": "日", "columns": [], "source": "sse/szse"},
        "lhb": {"path": "market/lhb_20*.parquet", "freq": "日", "columns": [], "source": "eastmoney"},  # 仅主表季度文件；细分表单独登记
        "lhb_inst": {"path": "market/lhb_jgmmtj_em.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "lhb_institution": {"path": "market/lhb_jgmmtj_em.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "lhb_rank": {"path": "market/lhb_ggtj_sina.parquet", "freq": "日", "columns": [], "source": "sina"},
        "lhb_broker": {"path": "market/lhb_hyyyb_em.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "gdhs": {"path": "market/gdhs_all.parquet", "freq": "季", "columns": [], "source": "eastmoney"},
        "fund_hold": {"path": "market/fund_portfolio_hold.parquet", "freq": "季", "columns": [], "source": "eastmoney"},
        "account_stat": {"path": "market/account_stat.parquet", "freq": "周", "columns": [], "source": "eastmoney"},
        "market_fund_flow": {"path": "market/stock_market_fund_flow.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "zlkp_jgcyd": {"path": "market/zlkp_jgcyd.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "news_sentiment": {"path": "market/news_sentiment.parquet", "freq": "日", "columns": [], "source": "eastmoney", "tol": 7},  # V11: 公告情绪（总纲#9 非量化指标）
        "bond_cov_daily": {"path": "market/bond_cov_daily/*.parquet", "freq": "日", "columns": [], "source": "eastmoney", "tol": 4000},  # akshare 该接口历史止于 2019，属数据源限制
        # ── 宏观/市场全景（月/日频；低频数据集实际按周/月批次更新，容忍更长间隔）──
        "macro": {"path": "macro/*.parquet", "freq": "月/日", "columns": [], "source": "akshare"},
        "index_pe": {"path": "market/index_pe.parquet", "freq": "日", "columns": [], "source": "legulegu", "tol": 7},  # legulegu 源滞后 2-4 天属正常
        "industry_pe": {"path": "market/industry_pe_cninfo.parquet", "freq": "日", "columns": [], "source": "cninfo"},
        "a_high_low": {"path": "oneoff/a_high_low.parquet", "freq": "日", "columns": [], "source": "legulegu", "tol": 7},
        "a_below_net": {"path": "oneoff/a_below_net.parquet", "freq": "日", "columns": [], "source": "legulegu", "tol": 7},
        "market_heat": {"path": "market/market_heat__*.parquet", "freq": "日", "columns": [], "source": "legulegu", "tol": 7},
        "futures": {"path": "market/futures_basis.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "commodity": {"path": "market/commodity__*.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "rates": {"path": "market/rates__*.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "shibor": {"path": "market/shibor.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "repo_rate": {"path": "market/repo_rate.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "bond_futures": {"path": "market/bond_futures.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "cb_spot": {"path": "market/cb_spot.parquet", "freq": "周", "columns": [], "source": "eastmoney"},
        "qvix": {"path": "market/qvix.parquet", "freq": "周", "columns": [], "source": "akshare"},
        "esg_rating": {"path": "market/esg_rating.parquet", "freq": "周", "columns": [], "source": "sina/hz"},  # V3 融合: quant_v6 ESG 数据
        # ── 行业/分类层（用户 #10：申万行业 + 概念 + 题材）──
        "sw_industry_map": {"path": "market/sw_industry_map.parquet", "freq": "静态", "columns": ["code", "name", "industry", "industry_code"], "source": "sw"},
        "sw_industry": {"path": "market/sw_industry.parquet", "freq": "静态", "columns": [], "source": "sw"},  # 截面快照无日期列，按静态处理
        "concept_board": {"path": "classification/concept_board.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "concept_member": {"path": "classification/concept_member.parquet", "freq": "日", "columns": [], "source": "eastmoney"},
        "theme_board": {"path": "classification/theme_board.parquet", "freq": "日", "columns": [], "source": "ths"},
        # ── 实时 ──
        "realtime": {"path": "realtime_snapshot/", "freq": "5分钟", "columns": ["code", "name", "price", "pre_close", "open", "volume_lot", "amount_wan", "high", "low", "pct_chg", "turnover", "pe_ttm", "pb", "ts"], "source": "tencent"},
    }

    def list_datasets(self) -> list[str]:
        """返回全部已登记数据集名。"""
        return sorted(self._DATASETS.keys())

    def schema_of(self, dataset: str) -> dict:
        """返回数据集 schema（列/频率/来源）。未登记则抛 KeyError。"""
        if dataset not in self._DATASETS:
            raise KeyError(f"未知数据集: {dataset}（可用: {sorted(self._DATASETS.keys())}）")
        return dict(self._DATASETS[dataset])

    def warehouse_root(self) -> Path:
        """数据仓库根目录。"""
        return ROOT / "data_warehouse"

    # ── V2 (2026-08-07): 数据集级统一读取门面（数据资产化 V2-1）──
    #   所有数据集按名读取，内部解析 _DATASETS 路径模式；
    #   支持按日期/前缀过滤，供日报/特征库/研究统一消费仓库。

    def get_dataset(
        self,
        dataset: str,
        date: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        files: int | None = None,
    ) -> pd.DataFrame:
        """按数据集名读取仓库数据（统一 Data Access Layer 入口）。

        Args:
            dataset: 数据集名（list_datasets() 可查）。
            date:    精确日期过滤（YYYYMMDD / YYYY-MM-DD），仅对含日期的数据集生效。
            start/end: 日期区间过滤（含）。
            limit:   返回行数上限（取最新）。
            files:   只读取最近 N 个文件（按文件名逆序），大幅提速；
                     适用于按日期分文件的截面数据（lhb/zt_pool/margin_detail 等）。

        Returns:
            合并后的 DataFrame；数据集为空返回空 DataFrame（保留列）。

        Raises:
            KeyError: 未登记的数据集。
        """
        if dataset not in self._DATASETS:
            raise KeyError(f"未知数据集: {dataset}（可用: {sorted(self._DATASETS.keys())}）")
        spec = self._DATASETS[dataset]
        pattern = spec["path"]
        ws = self.warehouse_root()

        # 解析路径模式：{code}.parquet → 全量 glob；目录/前缀模式 → 展开
        if "{" in pattern and "}" in pattern:
            # 如 kline/{code}.parquet / margin_detail_{sh,sz}/ → 展开为 glob
            import re as _re

            base, _, _tail = pattern.partition("{")
            _tail_head, _, _tail_rest = _tail.partition("}")
            opts = _tail_head.split(",")
            if "{code}" in pattern or "{code}" in _tail:
                # 个股维度数据集：支持单 code 过滤（date 参数对个股维度无意义）
                # 仅当调用方传了额外 filter 才需要；此处直接返回空列结构提示用法
                return pd.DataFrame()
            paths = []
            for opt in opts:
                p = ws / (base + opt + _tail_rest)
                if p.is_dir():
                    paths.extend(sorted(p.glob("*.parquet")))
                else:
                    paths.extend(sorted(ws.glob(base + opt + _tail_rest)))
        else:
            paths = sorted(ws.glob(pattern))

        # V3: files 参数——只保留最近 N 个文件（按文件名/时间逆序），避免全量读大截面
        if files is not None and len(paths) > files:
            paths = paths[-files:]

        frames: list[pd.DataFrame] = []
        for p in paths:
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
                continue
            # 按文件名/内容日期过滤
            if date or start or end:
                # V11 审计修复: 原实现只认英文日期列（date/trade_date/ts），
                # 中文日期列（"日期"/"交易日"）的数据集过滤静默失效。
                # 修正: 支持中文列名 + 时间戳列。
                date_cols = [c for c in df.columns if str(c) in (
                    "date", "trade_date", "ts", "日期", "交易日", "时间"
                ) or "日期" in str(c) or "date" in str(c).lower()]
                if date_cols:
                    c = date_cols[0]
                    dvals = pd.to_datetime(df[c], errors="coerce")
                    if date:
                        d = pd.to_datetime(date)
                        df = df[dvals.dt.date == d.date()]
                    if start:
                        df = df[dvals >= pd.to_datetime(start)]
                    if end:
                        df = df[dvals <= pd.to_datetime(end)]
            frames.append(df)
        if not frames:
            cols = spec.get("columns") or []
            return pd.DataFrame(columns=cols)
        out = pd.concat(frames, ignore_index=True)
        if limit:
            out = out.tail(limit)
        return out

    # ── V2 (2026-08-07): 全量数据集体检表（数据资产化 V2-1 / 前端数据基座视图）──
    #   扫描每个数据集：文件数 / 最新日期 / 预期频率 / 状态，供 freshness 面板与告警。

    def freshness(self, trade_date: str | None = None) -> list[dict]:
        """全量数据集新鲜度体检。

        Returns:
            [{
              "dataset": 数据集名,
              "files": 文件数,
              "latest": 最新日期(YYYY-MM-DD),
              "expected": 预期最新日期(YYYY-MM-DD),
              "freq": 频率,
              "status": "ok"|"stale"|"missing"|"n/a",
              "note": 说明,
            }, ...]
        """
        from datetime import date as _date

        ws = self.warehouse_root()
        rows = []
        # 预期最新交易日（默认最近一个自然日；交易日历可由调用方传入）
        expected = trade_date or _date.today().strftime("%Y-%m-%d")

        for name, spec in sorted(self._DATASETS.items()):
            pattern = spec["path"]
            freq = spec.get("freq", "-")
            entry = {"dataset": name, "files": 0, "latest": None, "expected": expected,
                     "freq": freq, "status": "missing", "note": ""}
            try:
                # 展开路径模式（{code} 为个股维度通配符 → 直接 glob 全目录）
                import re as _re
                paths: list = []
                if "{" in pattern:
                    if "{code}" in pattern:
                        # kline/{code}.parquet 等个股维度：目录下全部文件
                        base = pattern.split("{")[0]
                        p = ws / base
                        paths = sorted(p.glob("*.parquet")) if p.is_dir() else []
                    else:
                        base, _, _tail = pattern.partition("{")
                        _th, _, _tr = _tail.partition("}")
                        for opt in _th.split(","):
                            p = ws / (base + opt + _tr)
                            if p.is_dir():
                                paths.extend(sorted(p.glob("*.parquet")))
                            else:
                                paths.extend(sorted(ws.glob(base + opt + _tr)))
                else:
                    paths = sorted(ws.glob(pattern))
                entry["files"] = len(paths)
                if not paths:
                    entry["status"] = "missing"
                    rows.append(entry)
                    continue

                # 采样最新文件提取日期（文件名优先，其次内容首列，最后修改时间兜底）
                latest_dt = None
                for p in reversed(paths[-3:]):
                    stem = p.stem
                    # 文件名可能是区间格式 lhb_20260701_20260806 → 取最后一个日期（数据覆盖到该日）
                    m = _re.findall(r"(\d{8})", stem)
                    if m:
                        latest_dt = datetime.strptime(m[-1], "%Y%m%d").date()
                        break
                    try:
                        df = pd.read_parquet(p)
                        date_cols = [c for c in ("date", "trade_date", "ts") if c in df.columns]
                        if date_cols:
                            v = pd.to_datetime(df[date_cols[0]], errors="coerce").max()
                            if pd.notna(v):
                                latest_dt = v.date()
                                break
                    except Exception as e:
                        logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
                        continue
                    # 内容无日期列 → 用修改时间（反映最近更新）
                    latest_dt = datetime.fromtimestamp(p.stat().st_mtime).date()
                    break
                if latest_dt is None:
                    entry["status"] = "n/a"
                    rows.append(entry)
                    continue
                entry["latest"] = latest_dt.isoformat()
                # 状态判定：个股维度（K线/估值/财务）看文件数是否覆盖全市场；时间维度看最新日期
                if name in ("kline", "valuation", "financial"):
                    entry["status"] = "ok" if entry["files"] >= 5000 else "partial"
                    entry["note"] = f"覆盖 {entry['files']} 只"
                else:
                    gap = (_date.fromisoformat(expected) - latest_dt).days if expected else 0
                    # V3: 频率感知过期判定——月频/季频/静态数据允许更长间隔；tol 字段可逐数据集覆盖
                    freq_tol = {"月/日": 7, "月": 35, "季": 100, "周": 14, "静态": 9999}.get(freq, 2)
                    freq_tol = spec.get("tol", freq_tol)
                    if gap <= max(2, freq_tol):
                        entry["status"] = "ok"
                    elif gap <= 7:
                        entry["status"] = "stale"
                        entry["note"] = f"滞后 {gap} 天"
                    else:
                        entry["status"] = "stale"
                        entry["note"] = f"滞后 {gap} 天"
            except Exception as e:  # noqa: BLE001
                entry["status"] = "error"
                entry["note"] = str(e)[:80]
            rows.append(entry)
        return rows

    # ── V4.1: 分钟级K线支持 ────────────────────────────────────────────

    def get_minute_klines(
        self,
        symbol: str,
        period: str = "1",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取分钟级K线数据，支持 1/5/15/30/60 分钟。

        数据来源：ak.stock_zh_a_hist_min_em()
        缓存策略：
          - 缓存文件 ~/.quant_system/minute_klines/{symbol}_{period}.parquet
          - 今天之前的数据缓存后不再重复请求
          - 先读缓存，如果缺少当天数据则从 akshare 拉取合并

        Args:
            symbol:  A-share 股票代码。
            period:  分钟周期，可选 "1" / "5" / "15" / "30" / "60"。
            start_date:  起始日期 YYYY-MM-DD，默认 30 天前。
            end_date:    结束日期 YYYY-MM-DD，默认今天。

        Returns:
            DataFrame with columns:
                date, open, high, low, close, volume, amount

        V4.1 audit fix: 任何返回路径都按 [start_date, end_date] 过滤。
        此前缓存命中直接返回全量缓存，start_date/end_date 仅影响拉取范围，
        调用方传历史区间拿到的却是含未来数据的全量——前视偏差。
        """

        def _filter_range(df: pd.DataFrame, s: str | None, e: str | None) -> pd.DataFrame:
            """按请求区间过滤（日期字符串比较安全，缓存/新数据统一走这里）。"""
            if df is None or df.empty:
                return df
            out = df.copy()
            if "date" in out.columns:
                d = pd.to_datetime(out["date"])
                if s:
                    out = out[d >= pd.Timestamp(s)]
                    d = pd.to_datetime(out["date"])
                if e:
                    out = out[d <= pd.Timestamp(e) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
            return out.reset_index(drop=True)
        symbol = _normalize(symbol)
        period = str(period)
        if period not in MINUTE_PERIODS:
            raise ValueError(f"不支持的分钟周期: {period}，可选: {MINUTE_PERIODS}")

        # 默认时间范围：最近 30 天
        today = date.today()
        if end_date is None:
            end_date = today.isoformat()
        if start_date is None:
            start_date = (today - timedelta(days=30)).isoformat()

        cache_path = MINUTE_KLINE_CACHE / f"{symbol}_{period}.parquet"
        MINUTE_KLINE_CACHE.mkdir(parents=True, exist_ok=True)

        # ── 1) 尝试从 parquet 缓存加载 ──
        cached_df = pd.DataFrame()
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                cached_df["date"] = pd.to_datetime(cached_df["date"])
                logger.debug("[minute_kline] %s(%smin): 缓存加载 %d 行",
                             symbol, period, len(cached_df))
            except Exception as exc:
                logger.warning("[minute_kline] %s(%smin): 缓存损坏，重新拉取: %s",
                              symbol, period, exc)
                cached_df = pd.DataFrame()

        # ── 2) 判断是否需要从 API 补充当天数据 ──
        need_fetch = False
        today_str = today.isoformat()
        if cached_df.empty:
            need_fetch = True
        else:
            latest_cached = cached_df["date"].max()
            # 如果缓存最新日期 < 今天，需要补充
            if latest_cached < pd.Timestamp(today_str):
                need_fetch = True
            else:
                # P2-Q1-fix (Q1-M003): 盘中新鲜度 — 今日最新 bar 距今超过 N 分钟则重拉。
                #   原逻辑仅当缓存最新日期 < 今天才重拉，当日首次拉取后盘中不再更新
                #   （最新 bar 停留在首拉时刻）。仅在交易时段内生效，避免收盘后空刷。
                try:
                    now_bj = datetime.now(CST).replace(tzinfo=None)
                    age_min = (now_bj - latest_cached.to_pydatetime()).total_seconds() / 60
                    stale_after_min = max(int(period), 5)
                    in_session = is_trading_session() if is_trading_session is not None else True
                    if age_min > stale_after_min and in_session:
                        need_fetch = True
                except Exception as e:
                    logger.error(f"[data_store] 操作失败: {e}", exc_info=True)

        if need_fetch:
            try:
                # 延迟导入 akshare（避免不必要的依赖加载）
                import akshare as ak  # type: ignore[import-untyped]

                # 确定要拉取的时间范围
                fetch_start = start_date
                if not cached_df.empty:
                    # 只拉取缓存缺失的部分（从最新缓存日期次日开始）
                    latest_cached = cached_df["date"].max()
                    fetch_start = (latest_cached + timedelta(days=1)).strftime("%Y-%m-%d")

                logger.info("[minute_kline] %s(%smin): 从 akshare 拉取 %s ~ %s",
                            symbol, period, fetch_start, end_date)

                new_df = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period,
                    start_date=fetch_start,
                    end_date=end_date,
                )

                if new_df is not None and not new_df.empty:
                    # 标准化列名
                    new_df = new_df.rename(columns={
                        "时间": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "成交额": "amount",
                    })
                    new_df["date"] = pd.to_datetime(new_df["date"])
                    # 保留需要的列
                    cols_needed = ["date", "open", "high", "low", "close", "volume", "amount"]
                    new_df = new_df[[c for c in cols_needed if c in new_df.columns]]

                    # ── 3) 合并新数据到缓存 ──
                    if not cached_df.empty:
                        # 去重：以新数据为准
                        combined = pd.concat([cached_df, new_df], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["date"], keep="last")
                        combined = combined.sort_values("date").reset_index(drop=True)
                    else:
                        combined = new_df.sort_values("date").reset_index(drop=True)

                    # 写入缓存
                    combined.to_parquet(cache_path, index=False)
                    logger.info("[minute_kline] %s(%smin): 缓存已更新，共 %d 行",
                                symbol, period, len(combined))
                    return _filter_range(combined, start_date, end_date)

            except Exception as exc:
                logger.error("[minute_kline] %s(%smin): API 拉取失败: %s",
                             symbol, period, exc)
                # API 失败时，如果有缓存数据则返回缓存
                if not cached_df.empty:
                    logger.warning("[minute_kline] %s(%smin): 降级返回缓存数据",
                                  symbol, period)
                    return _filter_range(cached_df, start_date, end_date)
                raise

        return _filter_range(cached_df, start_date, end_date)

    def get_minute_factor(
        self,
        symbol: str,
        period: str = "1",
        factor_name: str = "ma5",
    ) -> pd.Series:
        """基于分钟K线计算技术因子，方便日内策略快速调用。

        支持的因子：
          - "ma5" / "ma20" / "ma60": 简单移动平均线(5/20/60周期)
          - "ema5" / "ema12" / "ema26": 指数移动平均线
          - "rsi" / "rsi6" / "rsi14": 相对强弱指标
          - "boll_upper" / "boll_mid" / "boll_lower": 布林带
          - "volume_ma5": 成交量移动平均
          - "pct_chg": 周期涨跌幅(%)

        Args:
            symbol:      A-share 股票代码。
            period:      分钟周期 "1" / "5" / "15" / "30" / "60"。
            factor_name: 因子名称，如上所列。

        Returns:
            pd.Series，index 为 datetime，值为因子计算结果。
        """
        df = self.get_minute_klines(symbol, period=period)
        if df.empty:
            return pd.Series(dtype=float)

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else None
        high = df["high"] if "high" in df.columns else None
        low = df["low"] if "low" in df.columns else None

        factor_name = factor_name.lower().strip()

        # 移动平均线
        if factor_name == "ma5":
            result = close.rolling(window=5).mean()
        elif factor_name == "ma20":
            result = close.rolling(window=20).mean()
        elif factor_name == "ma60":
            result = close.rolling(window=60).mean()
        elif factor_name == "ema5":
            result = close.ewm(span=5, adjust=False).mean()
        elif factor_name == "ema12":
            result = close.ewm(span=12, adjust=False).mean()
        elif factor_name == "ema26":
            result = close.ewm(span=26, adjust=False).mean()

        # RSI
        elif factor_name in ("rsi", "rsi6"):
            window = 6 if factor_name == "rsi6" else 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window=window).mean()
            loss = (-delta).clip(lower=0).rolling(window=window).mean()
            rs = gain / loss.replace(0, np.nan)
            result = 100 - (100 / (1 + rs))
        elif factor_name == "rsi14":
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta).clip(lower=0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            result = 100 - (100 / (1 + rs))

        # 布林带
        elif factor_name == "boll_mid":
            result = close.rolling(window=20).mean()
        elif factor_name == "boll_upper":
            mid = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            result = mid + 2 * std
        elif factor_name == "boll_lower":
            mid = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            result = mid - 2 * std

        # 成交量移动平均
        elif factor_name == "volume_ma5":
            if volume is None:
                return pd.Series(dtype=float, index=df.index)
            result = volume.rolling(window=5).mean()

        # 周期涨跌幅
        elif factor_name == "pct_chg":
            result = close.pct_change() * 100

        else:
            raise ValueError(f"不支持的分钟因子: {factor_name}")

        result.name = f"{factor_name}_{period}min"
        result.index = df["date"]
        return result

    def list_minute_symbols(self) -> list[str]:
        """列出缓存中有分钟数据的股票代码（去重）。

        Returns:
            list[str]: 归一化后的股票代码列表。
        """
        if not MINUTE_KLINE_CACHE.exists():
            return []

        symbols: set[str] = set()
        for fpath in MINUTE_KLINE_CACHE.glob("*.parquet"):
            # 文件名格式: {symbol}_{period}.parquet
            # 例如: 002714_1.parquet -> symbol=002714
            parts = fpath.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in MINUTE_PERIODS:
                symbols.add(parts[0])

        return sorted(symbols)

    # ── Helper ──────────────────────────────────────────────────────────

    def _info(self, symbol: str) -> dict[str, Any]:
        return _freshness(symbol) or {"symbol": symbol, "error": "not cached"}


# ---------------------------------------------------------------------------
# Singleton convenience
# ---------------------------------------------------------------------------

_store: DataStore | None = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store


# ---------------------------------------------------------------------------
# Drop-in replacement for fetch_daily (backward compatible)
# ---------------------------------------------------------------------------

def fetch_daily(
    symbol: str,
    start: str = "20200101",
    end: str | None = None,
    adjust: str = "qfq",
    use_cache: bool = True,
    retries: int = 3,
) -> pd.DataFrame:
    """Backward-compatible wrapper.  Prefer DataStore.get() for new code."""
    # P2-Q1-fix (Q1-M004): 直接透传 start/end/adjust 给 get()。
    #   V5.4 版把 start 折算成 days 后由 get() 重算（窗口约提前 20 天），
    #   end 被忽略、adjust 被忽略（内部恒 qfq）。
    ds = get_store()
    return ds.get(
        symbol,
        start=start,
        end=end,
        adjust=adjust,
        force_refresh=not use_cache,
        fill_gaps=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    ds = get_store()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        s = ds.status()
        print(f"Symbols: {s['total_symbols']}  "
              f"Fresh: {s['fresh_count']}  "
              f"Stale: {s['stale_count']}")
        print(f"Avg quality: {s['avg_quality']}  "
              f"DB: {s['db_size_kb']}KB  "
              f"CSV cache: {s['cache_dir_size_kb']}KB")
        for e in s["entries"][:10]:
            bar = "🟢" if not e["stale"] else "🟡"
            print(f"  {bar} {e['symbol']:>8}  "
                  f"rows={e['row_count']:>4}  "
                  f"q={e['quality_score']:.2f}  "
                  f"age={e['age_h']:>5.1f}h  "
                  f"→ {e['last_date']}")

    elif cmd == "refresh":
        symbols = sys.argv[2:]
        r = ds.refresh_all(symbols if symbols else None)
        print(f"Refreshed {r['refreshed']}, failed {r['failed']}")
        for err in r["errors"]:
            print(f"  ⚠ {err}")

    elif cmd == "prune":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
        r = ds.prune(days)
        print(f"Pruned {r['pruned']} stale entries")

    elif cmd == "get":
        sym = sys.argv[2] if len(sys.argv) > 2 else "002714"
        df = ds.get(sym)
        print(f"{sym}: {len(df)} rows, "
              f"from {df['date'].iloc[0]} to {df['date'].iloc[-1]}, "
              f"quality={_quality_score(df):.3f}")
        print(df[["date", "close", "volume", "pct_chg"]].tail(5).to_string(index=False))

    else:
        print(f"Usage: {sys.argv[0]} [status|refresh|prune|get]")


# ---------------------------------------------------------------------------
# Analytics store (DuckDB first, SQLite fallback)
# ---------------------------------------------------------------------------

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


_DUCKDB_PATH = os.path.join(str(Path.home()), ".quant_system", "analytics.duckdb")
_duckdb_conn = None
_duckdb_checked = False
_duckdb_unavailable_reason: str | None = None
_duckdb_lock = threading.RLock()
_sqlite_analytics_lock = threading.RLock()

_ANALYTICS_TABLES = {
    "factor_data",
    "ic_history",
    "regime_history",
    "portfolio_return",
}
_ANALYTICS_VIEWS = {
    "vw_factor_summary",
    "vw_latest_factor",
    "vw_ic_summary",
    "vw_regime_latest",
    "vw_portfolio_performance",
}
_SQLITE_CORE_TABLES = {"daily_klines", "sector_stocks", "data_freshness"}


def _quote_identifier(name: str) -> str:
    """Quote a simple SQL identifier after validation."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name)):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def _normalize_analytics_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        text = str(value).strip()
        if not text:
            return None
        if re.match(r"^\d{8}$", text):
            return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
        return pd.to_datetime(text).strftime("%Y-%m-%d")
    except Exception:
        return str(value).strip()


def _normalize_analytics_symbol(value: Any) -> str:
    try:
        return _normalize(str(value))
    except Exception:
        return str(value).strip()


def _get_duckdb():
    """DuckDB 连接（单例懒加载），不可用时返回 None"""
    global _duckdb_conn, _duckdb_checked, _duckdb_unavailable_reason
    if _duckdb_conn is not None:
        return _duckdb_conn
    if _duckdb_checked and _duckdb_conn is None:
        return None

    with _duckdb_lock:
        if _duckdb_conn is not None:
            return _duckdb_conn
        if _duckdb_checked and _duckdb_conn is None:
            return None
        _duckdb_checked = True
        try:
            Path(_DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
            import duckdb  # type: ignore

            _duckdb_conn = duckdb.connect(_DUCKDB_PATH)
            _duckdb_unavailable_reason = None
            return _duckdb_conn
        except Exception as exc:
            _duckdb_conn = None
            _duckdb_unavailable_reason = repr(exc)
            return None


def has_duckdb() -> bool:
    """检查 DuckDB 是否可用"""
    return _get_duckdb() is not None


def _analytics_schema_sql(dialect: str) -> str:
    return """
        CREATE TABLE IF NOT EXISTS factor_data (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            factor_value REAL,
            PRIMARY KEY (symbol, date, factor_name)
        );
        CREATE TABLE IF NOT EXISTS ic_history (
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            rank_ic REAL,
            icir REAL,
            n_stocks INTEGER,
            PRIMARY KEY (date, factor_name)
        );
        CREATE TABLE IF NOT EXISTS regime_history (
            date TEXT PRIMARY KEY,
            trend TEXT,
            volatility TEXT,
            liquidity TEXT,
            sentiment TEXT
        );
        CREATE TABLE IF NOT EXISTS portfolio_return (
            date TEXT NOT NULL,
            strategy TEXT NOT NULL,
            "return" REAL,
            benchmark_return REAL,
            PRIMARY KEY (date, strategy)
        );
        CREATE INDEX IF NOT EXISTS idx_factor_data_name_date
            ON factor_data (factor_name, date);
        CREATE INDEX IF NOT EXISTS idx_factor_data_symbol_date
            ON factor_data (symbol, date);
        CREATE INDEX IF NOT EXISTS idx_ic_history_factor_date
            ON ic_history (factor_name, date);
    """


def _duckdb_executescript(conn, script: str) -> None:
    for statement in [s.strip() for s in script.split(";") if s.strip()]:
        conn.execute(statement)


def _init_duckdb_analytics_db(conn) -> None:
    with _duckdb_lock:
        _duckdb_executescript(conn, _analytics_schema_sql("duckdb"))


def _init_sqlite_analytics_db() -> None:
    _init_db()
    with _sqlite_analytics_lock:
        conn = _get_conn()
        conn.executescript(_analytics_schema_sql("sqlite"))
        conn.commit()


def init_analytics_db():
    """初始化分析数据库
    - factor_data(symbol, date, factor_name, factor_value)
    - ic_history(date, factor_name, rank_ic, icir, n_stocks)
    - regime_history(date, trend, volatility, liquidity, sentiment)
    - portfolio_return(date, strategy, return, benchmark_return)
    """
    conn = _get_duckdb()
    if conn is not None:
        try:
            _init_duckdb_analytics_db(conn)
            return "duckdb"
        except Exception as e:
            logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
    try:
        _init_sqlite_analytics_db()
        return "sqlite"
    except Exception:
        return None


def _looks_like_query(sql: str) -> bool:
    head = sql.lstrip().split(None, 1)[0].lower() if sql.strip() else ""
    return head in {"select", "with", "show", "describe", "pragma", "explain"}


def _duckdb_query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    conn = _get_duckdb()
    if conn is None:
        raise RuntimeError(_duckdb_unavailable_reason or "duckdb unavailable")
    with _duckdb_lock:
        cur = conn.execute(sql, params or ())
        if _looks_like_query(sql):
            try:
                return cur.fetchdf()
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()


def _sqlite_query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    _init_db()
    with _sqlite_analytics_lock:
        conn = _get_conn()
        if _looks_like_query(sql):
            return pd.read_sql_query(sql, conn, params=params or ())
        if params:
            conn.execute(sql, params)
        else:
            conn.executescript(sql)
        conn.commit()
        return pd.DataFrame()


def _sql_mentions_any(sql: str, names: set[str]) -> bool:
    lowered = sql.lower()
    return any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in names)


def _empty_with_error(exc: Exception) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = repr(exc)
    return df


def smart_query(sql: str, params: tuple = None, prefer: str = None) -> pd.DataFrame:
    """自动判断查询类型路由到最合适的数据库"""
    params = params or ()
    preferred = (prefer or "").lower().strip()

    if preferred in {"duckdb", "fast", "analytics"}:
        return query_fast(sql, params)
    if preferred in {"sqlite", "legacy"}:
        try:
            return _sqlite_query(sql, params)
        except Exception as exc:
            return _empty_with_error(exc)

    analytics_names = _ANALYTICS_TABLES | _ANALYTICS_VIEWS
    if _sql_mentions_any(sql, analytics_names):
        if _sql_mentions_any(sql, _ANALYTICS_VIEWS):
            create_analytics_views()
        else:
            init_analytics_db()
        return query_fast(sql, params)

    if _sql_mentions_any(sql, _SQLITE_CORE_TABLES):
        try:
            return _sqlite_query(sql, params)
        except Exception:
            return query_fast(sql, params)

    if has_duckdb():
        try:
            return _duckdb_query(sql, params)
        except Exception as e:
            logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
    try:
        return _sqlite_query(sql, params)
    except Exception as exc:
        return _empty_with_error(exc)


def query_fast(sql: str, params: tuple = None) -> pd.DataFrame:
    """强制走 DuckDB（不可用时降级 SQLite）"""
    params = params or ()
    if _sql_mentions_any(sql, _ANALYTICS_VIEWS):
        create_analytics_views()
    elif _sql_mentions_any(sql, _ANALYTICS_TABLES):
        init_analytics_db()
    if has_duckdb():
        try:
            return _duckdb_query(sql, params)
        except Exception as e:
            logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
    try:
        return _sqlite_query(sql, params)
    except Exception as exc:
        return _empty_with_error(exc)


def _normalize_factor_frame(factor_name: str, data: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    columns = ["symbol", "date", "factor_name", "factor_value"]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)

    df = data.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
        if "index" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"index": "date"})

    rename_map: dict[Any, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"date", "trade_date", "datetime", "time", "day"}:
            rename_map[col] = "date"
        elif key in {"symbol", "code", "ticker", "ts_code", "stock", "secid"}:
            rename_map[col] = "symbol"
        elif key in {"factor_value", "value", "score", factor_name.lower()}:
            rename_map[col] = "factor_value"
    if rename_map:
        df = df.rename(columns=rename_map)

    if {"symbol", "date"}.issubset(df.columns):
        value_col = "factor_value" if "factor_value" in df.columns else None
        if value_col is None:
            candidates = [c for c in df.columns if c not in {"symbol", "date", "factor_name"}]
            value_col = candidates[0] if candidates else None
        if value_col is None:
            return pd.DataFrame(columns=columns)
        out = df[["symbol", "date", value_col]].rename(columns={value_col: "factor_value"})
    elif "date" in df.columns:
        value_cols = [c for c in df.columns if c != "date"]
        out = df.melt(id_vars=["date"], value_vars=value_cols,
                      var_name="symbol", value_name="factor_value")
    elif "symbol" in df.columns:
        value_candidates = [c for c in df.columns if c not in {"symbol", "factor_name"}]
        if len(value_candidates) > 1 and "factor_value" not in df.columns:
            out = df.melt(id_vars=["symbol"], value_vars=value_candidates,
                          var_name="date", value_name="factor_value")
        else:
            value_col = "factor_value" if "factor_value" in df.columns else None
            if value_col is None:
                value_col = value_candidates[0] if value_candidates else None
            if value_col is None:
                return pd.DataFrame(columns=columns)
            out = df[["symbol", value_col]].rename(columns={value_col: "factor_value"})
            out["date"] = pd.Timestamp.today().date().strftime("%Y-%m-%d")
    else:
        # P2-Q1-fix (Q1-M006): 无 date/symbol 列的帧 → 列名即股票代码。
        #   V5.4 版对 RangeIndex 纯股票列帧直接返回空 → store_factor_data 静默不落库。
        tmp = df.copy()
        if isinstance(tmp.index, pd.RangeIndex):
            # 多列 = 多股票：按列 melt 为 symbol；无日期时用调用方传入的 date（而非丢弃）
            if not len(tmp.columns):
                return pd.DataFrame(columns=columns)
            out = tmp.melt(var_name="symbol", value_name="factor_value")
            out["date"] = _normalize_analytics_date(date or pd.Timestamp.today().date().strftime("%Y-%m-%d"))
        else:
            idx = tmp.index
            reset_col = tmp.index.name or "index"
            tmp = tmp.reset_index()
            if isinstance(idx, pd.DatetimeIndex):
                # 索引是日期 → 作为 date 列，各列即各股票
                tmp = tmp.rename(columns={reset_col: "date"})
                out = tmp.melt(id_vars=["date"],
                               value_vars=[c for c in tmp.columns if c != "date"],
                               var_name="symbol", value_name="factor_value")
            else:
                tmp = tmp.rename(columns={reset_col: "symbol"})
                value_cols = [c for c in tmp.columns if c != "symbol"]
                out = tmp.melt(id_vars=["symbol"], value_vars=value_cols,
                               var_name="date", value_name="factor_value")
                if date:
                    out["date"] = _normalize_analytics_date(date)

    out = out.copy()
    out["factor_name"] = factor_name
    out["symbol"] = out["symbol"].map(_normalize_analytics_symbol)
    out["date"] = out["date"].map(_normalize_analytics_date)
    out["factor_value"] = pd.to_numeric(out["factor_value"], errors="coerce")
    out = out.dropna(subset=["symbol", "date", "factor_value"])
    out = out[columns].drop_duplicates(["symbol", "date", "factor_name"], keep="last")
    return out.reset_index(drop=True)


def _upsert_dataframe(table: str, df: pd.DataFrame, key_cols: list[str]) -> bool:
    if df.empty:
        return True
    init_analytics_db()
    table_q = _quote_identifier(table)
    cols = list(df.columns)
    col_q = ", ".join(_quote_identifier(c) for c in cols)

    conn = _get_duckdb()
    if conn is not None:
        try:
            temp_name = "_analytics_upsert_df"
            with _duckdb_lock:
                conn.register(temp_name, df)
                key_match = " AND ".join(
                    f"{table_q}.{_quote_identifier(c)} = {temp_name}.{_quote_identifier(c)}"
                    for c in key_cols
                )
                conn.execute(f"DELETE FROM {table_q} USING {temp_name} WHERE {key_match}")
                conn.execute(
                    f"INSERT INTO {table_q} ({col_q}) SELECT {col_q} FROM {temp_name}"
                )
                try:
                    conn.unregister(temp_name)
                except Exception as e:
                    logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
            return True
        except Exception as e:
            logger.error(f"[data_store] 操作失败: {e}", exc_info=True)

    try:
        _init_sqlite_analytics_db()
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO {table_q} ({col_q}) VALUES ({placeholders})"
        rows = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
        with _sqlite_analytics_lock:
            conn = _get_conn()
            conn.executemany(sql, rows)
            conn.commit()
        return True
    except Exception:
        return False


def store_factor_data(factor_name: str, data: pd.DataFrame, date: str | None = None) -> None:
    """存储因子数据到 analytics 库。

    Args:
        factor_name: 因子名。
        data: 因子DataFrame，支持宽表/窄表/纯股票列（无 date 列）等多种格式。
        date: 无 date 列时的回填日期（YYYY-MM-DD），缺省用今天。
              P2-Q1-fix (Q1-M006): 纯股票列因子帧无日期时不再丢弃，用调用方传入的 date。
    """
    df = _normalize_factor_frame(factor_name, data, date=date)
    _upsert_dataframe("factor_data", df, ["symbol", "date", "factor_name"])


def query_factor_data(factor_name, start_date=None, end_date=None, symbols=None) -> pd.DataFrame:
    init_analytics_db()
    sql = [
        "SELECT symbol, date, factor_name, factor_value",
        "FROM factor_data WHERE factor_name = ?",
    ]
    params: list[Any] = [factor_name]
    if start_date is not None:
        sql.append("AND date >= ?")
        params.append(_normalize_analytics_date(start_date))
    if end_date is not None:
        sql.append("AND date <= ?")
        params.append(_normalize_analytics_date(end_date))
    if symbols is not None:
        if isinstance(symbols, (str, int)):
            symbols = [symbols]
        norm_symbols = [_normalize_analytics_symbol(s) for s in symbols]
        if norm_symbols:
            sql.append("AND symbol IN (" + ", ".join(["?"] * len(norm_symbols)) + ")")
            params.extend(norm_symbols)
    sql.append("ORDER BY date, symbol")
    return smart_query(" ".join(sql), tuple(params), prefer="analytics")


def analyze_factor(factor_name: str) -> dict:
    df = query_factor_data(factor_name)
    if df.empty:
        return {"factor_name": factor_name, "count": 0, "available": False}

    values = pd.to_numeric(df["factor_value"], errors="coerce")
    out = {
        "factor_name": factor_name,
        "available": True,
        "count": int(values.count()),
        "n_symbols": int(df["symbol"].nunique()),
        "start_date": str(df["date"].min()),
        "end_date": str(df["date"].max()),
        "mean": float(values.mean()) if values.count() else None,
        "std": float(values.std()) if values.count() > 1 else None,
        "min": float(values.min()) if values.count() else None,
        "max": float(values.max()) if values.count() else None,
    }

    ic = smart_query(
        "SELECT date, rank_ic, icir, n_stocks FROM ic_history "
        "WHERE factor_name = ? ORDER BY date",
        (factor_name,),
        prefer="analytics",
    )
    if not ic.empty:
        rank_ic = pd.to_numeric(ic.get("rank_ic"), errors="coerce")
        icir = pd.to_numeric(ic.get("icir"), errors="coerce")
        out.update({
            "ic_count": int(rank_ic.count()),
            "avg_rank_ic": float(rank_ic.mean()) if rank_ic.count() else None,
            "avg_icir": float(icir.mean()) if icir.count() else None,
            "positive_ic_rate": float((rank_ic > 0).mean()) if rank_ic.count() else None,
            "latest_ic_date": str(ic["date"].iloc[-1]) if "date" in ic.columns else None,
        })
    return out


def store_ic_history(date, factor_name, rank_ic, icir, n_stocks):
    df = pd.DataFrame([{
        "date": _normalize_analytics_date(date),
        "factor_name": factor_name,
        "rank_ic": None if rank_ic is None else float(rank_ic),
        "icir": None if icir is None else float(icir),
        "n_stocks": None if n_stocks is None else int(n_stocks),
    }])
    _upsert_dataframe("ic_history", df, ["date", "factor_name"])


def store_regime(date, trend, volatility, liquidity, sentiment):
    df = pd.DataFrame([{
        "date": _normalize_analytics_date(date),
        "trend": None if trend is None else str(trend),
        "volatility": None if volatility is None else str(volatility),
        "liquidity": None if liquidity is None else str(liquidity),
        "sentiment": None if sentiment is None else str(sentiment),
    }])
    _upsert_dataframe("regime_history", df, ["date"])


def parallel_factor_query(factor_names, start, end, symbols=None, n_threads=4):
    """并行查多个因子"""
    if isinstance(factor_names, str):
        factor_names = [factor_names]
    names = list(factor_names or [])
    if not names:
        return {}

    workers = max(1, min(int(n_threads or 1), len(names)))
    result: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(query_factor_data, name, start, end, symbols): name
            for name in names
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result[name] = fut.result()
            except Exception as exc:
                result[name] = _empty_with_error(exc)
    return result


def migrate_sqlite_to_duckdb(tables: list[str] = None):
    conn = _get_duckdb()
    if conn is None:
        return {"duckdb": False, "migrated": {}, "errors": ["duckdb unavailable"]}

    _init_db()
    sqlite_conn = _get_conn()
    if tables is None:
        rows = sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        tables = [r[0] for r in rows]

    migrated: dict[str, int] = {}
    errors: list[str] = []
    for table in tables:
        try:
            table_q = _quote_identifier(table)
            df = pd.read_sql_query(f"SELECT * FROM {table_q}", sqlite_conn)
            with _duckdb_lock:
                temp_name = "_sqlite_migrate_df"
                conn.register(temp_name, df)
                conn.execute(f"CREATE OR REPLACE TABLE {table_q} AS SELECT * FROM {temp_name}")
                try:
                    conn.unregister(temp_name)
                except Exception as e:
                    logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
            migrated[table] = int(len(df))
        except Exception as exc:
            errors.append(f"{table}: {exc!r}")
    return {"duckdb": True, "path": _DUCKDB_PATH, "migrated": migrated, "errors": errors}


def _sqlite_db_stats() -> dict:
    _init_db()
    conn = _get_conn()
    stats = {
        "available": True,
        "path": str(DB_PATH),
        "exists": DB_PATH.exists(),
        "size_kb": round(DB_PATH.stat().st_size / 1024, 1) if DB_PATH.exists() else 0,
        "tables": {},
    }
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (table,) in rows:
            table_q = _quote_identifier(table)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_q}").fetchone()[0]
            info = {"rows": int(count)}
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table_q})").fetchall()]
            if "date" in cols:
                bounds = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table_q}").fetchone()
                info.update({"min_date": bounds[0], "max_date": bounds[1]})
            stats["tables"][table] = info
    except Exception as exc:
        stats["error"] = repr(exc)
    return stats


def _duckdb_db_stats() -> dict:
    conn = _get_duckdb()
    stats = {
        "available": conn is not None,
        "path": _DUCKDB_PATH,
        "exists": Path(_DUCKDB_PATH).exists(),
        "size_kb": round(Path(_DUCKDB_PATH).stat().st_size / 1024, 1)
        if Path(_DUCKDB_PATH).exists() else 0,
        "tables": {},
    }
    if conn is None:
        stats["error"] = _duckdb_unavailable_reason
        return stats
    try:
        with _duckdb_lock:
            rows = conn.execute(
                "SELECT table_name, table_type FROM information_schema.tables "
                "WHERE table_schema='main' ORDER BY table_name"
            ).fetchall()
            for table, table_type in rows:
                table_q = _quote_identifier(table)
                info = {"type": table_type}
                try:
                    info["rows"] = int(conn.execute(f"SELECT COUNT(*) FROM {table_q}").fetchone()[0])
                except Exception:
                    info["rows"] = None
                try:
                    cols = [r[0] for r in conn.execute(f"DESCRIBE {table_q}").fetchall()]
                    if "date" in cols:
                        bounds = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table_q}").fetchone()
                        info.update({"min_date": bounds[0], "max_date": bounds[1]})
                except Exception as e:
                    logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
                stats["tables"][table] = info
    except Exception as exc:
        stats["error"] = repr(exc)
    return stats


def check_db_stats(db_type: str = "all") -> dict:
    kind = (db_type or "all").lower()
    if kind == "sqlite":
        return {"sqlite": _sqlite_db_stats()}
    if kind == "duckdb":
        return {"duckdb": _duckdb_db_stats()}
    return {"sqlite": _sqlite_db_stats(), "duckdb": _duckdb_db_stats()}


def optimize_analytics_db():
    result = {"duckdb": False, "sqlite": False, "errors": []}
    conn = _get_duckdb()
    if conn is not None:
        try:
            with _duckdb_lock:
                conn.execute("ANALYZE")
                conn.execute("CHECKPOINT")
            result["duckdb"] = True
        except Exception as exc:
            result["errors"].append(f"duckdb: {exc!r}")
    try:
        _init_sqlite_analytics_db()
        with _sqlite_analytics_lock:
            sqlite_conn = _get_conn()
            sqlite_conn.execute("ANALYZE")
            sqlite_conn.commit()
            sqlite_conn.execute("VACUUM")
            sqlite_conn.commit()
        result["sqlite"] = True
    except Exception as exc:
        result["errors"].append(f"sqlite: {exc!r}")
    return result


def _analytics_views_sql(dialect: str) -> str:
    std_expr = "stddev_samp(factor_value)" if dialect == "duckdb" else "NULL"
    # DuckDB: CREATE OR REPLACE VIEW; SQLite: use IF NOT EXISTS + drop
    if dialect == "duckdb":
        create = "CREATE OR REPLACE VIEW"
    else:
        create = "CREATE VIEW IF NOT EXISTS"
    return f"""
        {create} vw_factor_summary AS
        SELECT
            factor_name,
            COUNT(*) AS n_obs,
            COUNT(DISTINCT symbol) AS n_symbols,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            AVG(factor_value) AS avg_value,
            {std_expr} AS std_value,
            MIN(factor_value) AS min_value,
            MAX(factor_value) AS max_value
        FROM factor_data
        GROUP BY factor_name;

        {create} vw_latest_factor AS
        SELECT f.symbol, f.date, f.factor_name, f.factor_value
        FROM factor_data f
        JOIN (
            SELECT factor_name, MAX(date) AS max_date
            FROM factor_data
            GROUP BY factor_name
        ) m ON f.factor_name = m.factor_name AND f.date = m.max_date;

        {create} vw_ic_summary AS
        SELECT
            factor_name,
            COUNT(*) AS n_periods,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            AVG(rank_ic) AS avg_rank_ic,
            AVG(icir) AS avg_icir,
            AVG(n_stocks) AS avg_n_stocks
        FROM ic_history
        GROUP BY factor_name;

        {create} vw_regime_latest AS
        SELECT date, trend, volatility, liquidity, sentiment
        FROM regime_history
        WHERE date = (SELECT MAX(date) FROM regime_history);

        {create} vw_portfolio_performance AS
        SELECT
            strategy,
            COUNT(*) AS n_periods,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            AVG("return") AS avg_return,
            SUM("return") AS cumulative_return,
            AVG(benchmark_return) AS avg_benchmark_return,
            SUM(benchmark_return) AS cumulative_benchmark_return,
            AVG("return" - benchmark_return) AS avg_alpha,
            SUM("return" - benchmark_return) AS cumulative_alpha
        FROM portfolio_return
        GROUP BY strategy;
    """


def create_analytics_views():
    backend = init_analytics_db()
    conn = _get_duckdb()
    if backend == "duckdb" and conn is not None:
        try:
            with _duckdb_lock:
                _duckdb_executescript(conn, _analytics_views_sql("duckdb"))
            return "duckdb"
        except Exception as e:
            logger.error(f"[data_store] 操作失败: {e}", exc_info=True)
    try:
        _init_sqlite_analytics_db()
        with _sqlite_analytics_lock:
            sqlite_conn = _get_conn()
            sqlite_conn.executescript(_analytics_views_sql("sqlite"))
            sqlite_conn.commit()
        return "sqlite"
    except Exception:
        return None


def query_view(view_name: str, filters: dict = None) -> pd.DataFrame:
    if view_name not in _ANALYTICS_VIEWS:
        return _empty_with_error(ValueError(f"Unknown analytics view: {view_name}"))
    create_analytics_views()

    clauses: list[str] = []
    params: list[Any] = []
    for key, value in (filters or {}).items():
        op = "="
        col = key
        if key.endswith("__gte"):
            col, op = key[:-5], ">="
        elif key.endswith("__lte"):
            col, op = key[:-5], "<="
        elif key.endswith("__gt"):
            col, op = key[:-4], ">"
        elif key.endswith("__lt"):
            col, op = key[:-4], "<"
        elif key.endswith("__between"):
            col, op = key[:-9], "BETWEEN"

        col_q = _quote_identifier(col)
        # P2-Q1-fix (Q1-L021): BETWEEN 必须传二元组 [lo, hi]。
        #   V5.4 版 __between 传标量/非二元组时会拼出 "col BETWEEN ?"（缺第二个占位符）
        #   → SQL 语法错误；len!=2 的列表还会被静默降级为 IN，语义错误。这里显式校验。
        if value is None:
            clauses.append(f"{col_q} IS NULL")
        elif op == "BETWEEN":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"__between 需要二元组 [lo, hi]，收到: {value!r}"
                )
            clauses.append(f"{col_q} BETWEEN ? AND ?")
            params.extend(value)
        elif isinstance(value, (list, tuple, set)) and not isinstance(value, str):
            values = list(value)
            if values:
                clauses.append(f"{col_q} IN (" + ", ".join(["?"] * len(values)) + ")")
                params.extend(values)
        else:
            clauses.append(f"{col_q} {op} ?")
            params.append(value)

    sql = f"SELECT * FROM {_quote_identifier(view_name)}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return smart_query(sql, tuple(params), prefer="analytics")
