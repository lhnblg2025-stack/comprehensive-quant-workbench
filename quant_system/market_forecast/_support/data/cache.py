"""
cache.py — QuantV6 双层缓存
内存 TTL 缓存 + 磁盘 JSON 缓存。key = 接口名 + 参数 hash。
TTL 分级：盘中 30s / 日频 1天 / 低频 7天。
"""
from __future__ import annotations
import logging

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from quant_system.market_forecast._support.common.constants import TTL_DAILY, TTL_INTRADAY

_mem: dict[str, tuple[float, Any]] = {}
_mem_api: dict[str, str] = {}  # P2-Q12-fix: 记录 hash key 所属接口，clear_prefix 不再对 md5 做无效前缀匹配
_mem_lock = threading.Lock()

_DISK_DIR = Path(os.environ.get("QV6_CACHE_DIR", "/root/quant/state/cache"))


def _key(api_name: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps({"a": api_name, "args": args, "kwargs": kwargs}, default=str, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def make_key(api_name: str, *args, **kwargs) -> str:
    return _key(api_name, args, kwargs)


def get(api_name: str, ttl: int = TTL_INTRADAY, *args, **kwargs) -> Any:
    """内存缓存读取（命中且未过期返回，否则 None）。"""
    k = _key(api_name, args, kwargs)
    with _mem_lock:
        item = _mem.get(k)
        if item and time.monotonic() - item[0] < ttl:
            return item[1]
    return None


def put(api_name: str, value: Any, ttl: int = TTL_INTRADAY, *args, **kwargs) -> Any:
    """写入内存缓存。"""
    k = _key(api_name, args, kwargs)
    with _mem_lock:
        _mem[k] = (time.monotonic(), value)
        _mem_api[k] = api_name
    return value


def disk_get(api_name: str, ttl: int = TTL_DAILY, *args, **kwargs) -> Any:
    """磁盘 JSON 缓存读取（可序列化数据用）。"""
    p = _disk_path(api_name, args, kwargs)
    if not p.exists():
        return None
    try:
        if time.time() - p.stat().st_mtime > ttl:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def disk_put(api_name: str, value: Any, *args, **kwargs) -> Any:
    """写入磁盘缓存。"""
    p = _disk_path(api_name, args, kwargs)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, default=str)
    except Exception as e:
        logging.getLogger(__name__).error(f"[cache] 操作失败: {e}", exc_info=True)
    return value


def _disk_path(api_name: str, args: tuple, kwargs: dict) -> Path:
    return _DISK_DIR / f"{api_name}_{_key(api_name, args, kwargs)[:16]}.json"


def get_or_fetch(api_name: str, fetch_fn: Any, ttl: int = TTL_INTRADAY,
                 *args, **kwargs) -> Any:
    """
    组合 API：内存缓存 → fetch_fn() → 回填缓存。
    fetch_fn: 无参可调用对象（闭包捕获真实参数）。
    """
    v = get(api_name, ttl, *args, **kwargs)
    if v is not None:
        return v
    v = fetch_fn()
    return put(api_name, v, ttl, *args, **kwargs)


def clear_all() -> None:
    with _mem_lock:
        _mem.clear()
        _mem_api.clear()


def clear_prefix(api_name: str) -> None:
    """清除某接口全部缓存。"""
    with _mem_lock:
        for k in list(_mem.keys()):
            if _mem_api.get(k, "").startswith(api_name):  # P2-Q12-fix: 按记录的原始 api_name 匹配，而不是 md5 key
                del _mem[k]
                _mem_api.pop(k, None)
