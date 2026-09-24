"""
量化系统 3.0 — 统一缓存层

提供装饰器风格的缓存，支持 TTL、统计、降级。

用法:
    from quant_system.cache import cached, cache_stats

    @cached(ttl=300, key_prefix="fm")
    def compute_fama_macbeth(symbols: list[str], step: int = 5) -> dict:
        ...

    stats = cache_stats()  # → {hits, misses, entries, ...}
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("quant_cache")


# ── 内存缓存 ──────────────────────────────────────────────

# 条目: key -> (ts, ttl, value)
#   P2-Q1-fix (Q1-L018): 记录每条目的 TTL，供惰性过期清理
_MEMO: dict[str, tuple[float, int, Any]] = {}
# P2-Q1-fix (Q1-M012): prefix -> {key 集合}，cache_clear(prefix=...) 按真实 key 清
#   （_MEMO 的 key 是 sha256 摘要，startswith(prefix) 永远匹配不到）
_MEMO_PREFIX: dict[str, set[str]] = {}
_MEMO_LOCK = threading.Lock()
_HITS = 0
_MISSES = 0
_CACHE_DB: Optional[Path] = None


def _connect() -> contextlib.AbstractContextManager[sqlite3.Connection]:
    """安全打开缓存 DB：with 退出时先 commit 再 close。

    `with sqlite3.connect(...) as conn:` 只提交事务不关闭连接，高频调用会
    泄漏 fd（quant-web 曾因 task_queue 同类问题耗尽 1024 fd 上限）。
    """
    @contextlib.contextmanager
    def _manager() -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(_CACHE_DB))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    return _manager()


def _evict_expired_locked(now: float | None = None) -> int:
    """惰性清理过期条目（调用方需持有 _MEMO_LOCK）。"""
    if now is None:
        now = time.time()
    expired = [k for k, (ts, ttl, _) in _MEMO.items() if now - ts > ttl]
    for k in expired:
        _MEMO.pop(k, None)
        for ks in _MEMO_PREFIX.values():
            ks.discard(k)
    return len(expired)


def set_db_path(path: Path) -> None:
    global _CACHE_DB
    _CACHE_DB = path


# ── 数据新鲜度 ──────────────────────────────────────────────

def data_freshness(entry_age: float, ttl: int) -> str:
    """Return a label for data freshness.

    Parameters
    ----------
    entry_age : float
        Age in seconds since the data was fetched.
    ttl : int
        Expected TTL in seconds.

    Returns
    -------
    "live" | "cached" | "stale"
    """
    if entry_age < 0:
        return "live"
    if entry_age < ttl * 0.5:
        return "live"
    if entry_age < ttl * 2:
        return "cached"
    return "stale"


# ── 装饰器 ────────────────────────────────────────────────

def cached(
    ttl: int = 600,
    key_prefix: str = "cache",
    serialize: bool = True,
    persist: bool = False,
) -> Callable:
    """Decorator: cache function return value with TTL.

    Parameters
    ----------
    ttl : int
        Time-to-live in seconds (default 300).
    key_prefix : str
        Prefix for cache key (default "cache").
    serialize : bool
        Whether to JSON-serialize when persisting (default True).
        P2-Q1-fix (Q1-M011): 仅影响持久化写入；内存缓存与函数返回值始终是
        原始对象（不再把 DataFrame/Timestamp 转成 list/str）。
    persist : bool
        Whether to persist to SQLite (default False).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            global _HITS, _MISSES
            # Build cache key from args
            key_parts = [key_prefix, func.__name__]
            for a in args:
                key_parts.append(str(a))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            cache_key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]
            now = time.time()

            # P2-Q1-fix (Q1-M012): 登记 key -> prefix，供 cache_clear(prefix=...) 精准清除
            with _MEMO_LOCK:
                _MEMO_PREFIX.setdefault(key_prefix, set()).add(cache_key)

            # 1. Check memory cache
            with _MEMO_LOCK:
                hit = _MEMO.get(cache_key)
                if hit and now - hit[0] <= ttl:
                    _HITS += 1
                    return hit[2]

            # 2. Check persistent cache (if enabled)
            if persist and _CACHE_DB and _CACHE_DB.exists():
                try:
                    with _connect() as conn:
                        conn.execute(
                            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)"
                        )
                        row = conn.execute(
                            "SELECT ts, data FROM cache WHERE key=?", (cache_key,)
                        ).fetchone()
                    if row and now - float(row[0]) <= ttl:
                        data = json.loads(row[1])
                        with _MEMO_LOCK:
                            _MEMO[cache_key] = (float(row[0]), ttl, data)
                            _HITS += 1
                        return data
                except Exception as exc:
                    logger.warning("Persistent cache read failed: %s", exc)

            # 3. Compute
            with _MEMO_LOCK:  # 审计 2026-08-16：计数加锁保证多线程下原子
                _MISSES += 1
            try:
                result = func(*args, **kwargs)
            except Exception:
                # On failure, return stale cache if available
                with _MEMO_LOCK:
                    stale = _MEMO.get(cache_key)
                if stale:
                    _HITS += 1
                    return stale[2]
                raise

            # 4. Store
            # P2-Q1-fix (Q1-M011): 内存缓存存「原始对象」，返回原始对象——
            #   V5.4 版 serialize=True(默认) 把返回值先 _json_safe：DataFrame→list、
            #   Timestamp→str、未知对象→str()，首次调用即改变返回类型，属潜伏的
            #   API 契约破坏。serialize 现在仅影响持久化写入。
            with _MEMO_LOCK:
                if len(_MEMO) > 1024:  # 惰性淘汰：条目过多时先清过期
                    _evict_expired_locked(now)
                _MEMO[cache_key] = (now, ttl, result)

            if persist and _CACHE_DB:
                try:
                    safe = result
                    if serialize:
                        safe = _json_safe(result)
                    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
                    with _connect() as conn:
                        conn.execute(
                            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)"
                        )
                        conn.execute(
                            "REPLACE INTO cache (key, ts, data) VALUES (?, ?, ?)",
                            (cache_key, now, json.dumps(safe, ensure_ascii=False, default=str)),
                        )
                except Exception as exc:
                    logger.warning("Persistent cache write failed: %s", exc)

            return result

        return wrapper

    return decorator


# ── 统计 ──────────────────────────────────────────────────

def cache_stats() -> dict:
    """Return cache statistics."""
    global _HITS, _MISSES
    with _MEMO_LOCK:
        # P2-Q1-fix (Q1-L018): 惰性清理过期条目，避免长期运行内存只增不减
        _evict_expired_locked()
        entries = len(_MEMO)
        now = time.time()
        ages = [int(now - ts) for ts, _, _ in _MEMO.values()] if entries else []
        oldest = max(ages) if ages else 0
    total = _HITS + _MISSES
    hit_rate = round(_HITS / total * 100, 1) if total else 0.0
    return {
        "entries": entries,
        "hits": _HITS,
        "misses": _MISSES,
        "hit_rate_pct": hit_rate,
        "total_calls": total,
        "oldest_age_seconds": oldest,
        "memory_mb": round(entries * 2048 / 1024 / 1024, 3),  # rough estimate
    }


def cache_clear(prefix: str | None = None) -> int:
    """Clear cache entries, optionally by prefix match."""
    global _HITS, _MISSES
    with _MEMO_LOCK:
        # P2-Q1-fix (Q1-M012): 用 prefix→keys 映射精确清除。
        #   V5.4 版用 `k.startswith(prefix)` 匹配 sha256 摘要，永远清不到（0 条）。
        if prefix:
            keys = list(_MEMO_PREFIX.get(prefix, set()))
            for k in keys:
                _MEMO.pop(k, None)
            _MEMO_PREFIX.pop(prefix, None)
        else:
            keys = list(_MEMO.keys())
            _MEMO.clear()
            _MEMO_PREFIX.clear()
    # 审计 2026-08-16：同步清理持久化 SQLite，避免旧数据重新进入内存
    if _CACHE_DB and _CACHE_DB.exists():
        try:
            with _connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts REAL, data TEXT)")
                if prefix:
                    # 无法按摘要回溯 prefix，只能全清（保守、不误删数据语义）
                    conn.execute("DELETE FROM cache")
                else:
                    conn.execute("DELETE FROM cache")
        except Exception as exc:
            logger.warning("Persistent cache clear failed: %s", exc)
    _HITS = 0
    _MISSES = 0
    return len(keys)


# ── JSON safe helper ──────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert non-serializable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)[:19]
    elif isinstance(obj, (pd.Series,)):
        return _json_safe(obj.to_dict())
    elif isinstance(obj, (pd.DataFrame,)):
        return _json_safe(obj.to_dict(orient="records"))
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except Exception:
            return None


# 运行时导入 pd/np 避免循环依赖
import numpy as np
import pandas as pd
