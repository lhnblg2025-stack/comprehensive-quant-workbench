"""
cache.py — V7.0 统一本地持久化缓存层（长期稳定数据源方案 L3）
=============================================================
- SQLite 元数据（key → 更新时间/TTL/来源）+ Parquet 数据文件
- 命中且未过期 → 0 次外部调用；过期才刷新
- 线程安全（sqlite WAL）；自动清理过期/超大 key
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.cache")

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


class DataCache:
    """统一缓存。目录结构:
    data_cache/
      meta.sqlite3          # key, updated_at, ttl_hours, source, shape
      parquet/<key>.parquet # 数据
    """

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.cache_dir / "parquet"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._db = self.cache_dir / "meta.sqlite3"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    ttl_hours REAL NOT NULL,
                    source TEXT DEFAULT '',
                    shape TEXT DEFAULT ''
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")

    # ── 工具 ──────────────────────────────────────────────

    @staticmethod
    def make_key(prefix: str, *parts) -> str:
        """稳定 key：prefix + 各段 sha1。"""
        raw = "|".join(str(p) for p in parts)
        h = hashlib.sha1(raw.encode()).hexdigest()[:16]
        return f"{prefix}:{h}"

    def _meta(self, key: str) -> tuple | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT updated_at, ttl_hours, source, shape FROM cache_meta WHERE key=?",
                (key,)).fetchone()
        return row

    def _path(self, key: str) -> Path:
        safe = hashlib.sha1(key.encode()).hexdigest()
        return self.parquet_dir / f"{safe}.parquet"

    # ── 主接口 ────────────────────────────────────────────

    def get(self, key: str, ttl_hours: float | None = None) -> pd.DataFrame | None:
        """读取缓存。未命中/过期返回 None。"""
        row = self._meta(key)
        if row is None:
            return None
        updated, ttl, source, shape = row
        eff_ttl = ttl_hours if ttl_hours is not None else ttl
        if time.time() - updated > eff_ttl * 3600:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            log.warning(f"缓存读取失败 {key}: {e}")
            return None

    def set(self, key: str, df: pd.DataFrame, ttl_hours: float,
            source: str = "") -> None:
        """写入缓存。"""
        if df is None or df.empty:
            return
        p = self._path(key)
        try:
            df.to_parquet(p, index=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"缓存写入失败 {key}: {e}")
            return
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (key, updated_at, ttl_hours, source, shape)"
                " VALUES (?,?,?,?,?)",
                (key, time.time(), ttl_hours, source,
                 json.dumps(list(df.columns))[:200]))

    def cached_fetch(self, key: str, ttl_hours: float, fetcher, *args,
                     source: str = "", refresh: bool = False, **kwargs):
        """缓存优先抓取：命中返回缓存；否则调用 fetcher 并写缓存。

        fetcher(*args, **kwargs) -> pd.DataFrame
        """
        if not refresh:
            hit = self.get(key, ttl_hours)
            if hit is not None:
                return hit
        df = fetcher(*args, **kwargs)
        self.set(key, df, ttl_hours, source=source)
        return df

    # ── 维护 ──────────────────────────────────────────────

    def stats(self) -> dict:
        """缓存统计：条目数/占用/过期数。"""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("SELECT key, updated_at, ttl_hours FROM cache_meta").fetchall()
        total_bytes = sum(p.stat().st_size for p in self.parquet_dir.glob("*.parquet"))
        now = time.time()
        expired = sum(1 for _, u, t in rows if now - u > t * 3600)
        return {"entries": len(rows), "bytes": total_bytes, "expired": expired}

    def cleanup(self, max_bytes: int = 5 * 1024**3) -> int:
        """清理过期 + 超出体积上限的最旧条目。返回删除数。"""
        removed = 0
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("SELECT key, updated_at, ttl_hours FROM cache_meta").fetchall()
        now = time.time()
        for key, u, t in rows:
            if now - u > t * 3600:
                p = self._path(key)
                if p.exists():
                    p.unlink()
                with sqlite3.connect(self._db) as conn:
                    conn.execute("DELETE FROM cache_meta WHERE key=?", (key,))
                removed += 1
        # 体积超限清理最旧
        total = sum(p.stat().st_size for p in self.parquet_dir.glob("*.parquet"))
        if total > max_bytes:
            with sqlite3.connect(self._db) as conn:
                rows = conn.execute(
                    "SELECT key, updated_at FROM cache_meta ORDER BY updated_at ASC").fetchall()
            for key, _ in rows:
                if total <= max_bytes:
                    break
                p = self._path(key)
                if p.exists():
                    sz = p.stat().st_size
                    p.unlink()
                    total -= sz
                with sqlite3.connect(self._db) as conn:
                    conn.execute("DELETE FROM cache_meta WHERE key=?", (key,))
                removed += 1
        if removed:
            log.info(f"缓存清理 {removed} 项")
        return removed


_default: DataCache | None = None


def get_cache() -> DataCache:
    global _default
    if _default is None:
        _default = DataCache()
    return _default
