"""
quant_platform.legacy — quant_v6 稳定能力薄转发层（单栈化收口，2026-08-08 已完全脱离 quant_v6）

单栈化原则（V3 方案）：
- quant_system 为主分析栈；quant_v6 已退役（移入 垃圾文件夹/quant_v6）
- 所有脚本/服务只准 import quant_platform.* / quant_system.*
- 本模块保留 quant_v6 时代的稳定函数签名，实现全部指向 quant_system 自包含实现

能力清单：
- cached_fetch: 请求级缓存（akshare 接口 TTL 缓存，内存实现，日报等批量脚本防重复请求）
- nice_process: 进程降优先级（Unix nice / Windows BelowNormal）
- scan_intraday / compute_intraday_signals: 盘中分时扫描与信号（quant_system.intraday_*）
- is_trading_time: 交易时段判断（quant_system.realtime）
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

# ── 请求级缓存（原 quant_v6.data.fetcher.cached_fetch 的内存版，签名兼容）──
_CACHE_LOCK = threading.Lock()
_CACHE_MEMO: dict[tuple, tuple] = {}  # key -> (expire_ts, value)


def cached_fetch(fn: Callable, *args: Any, ttl_seconds: int = 6 * 3600, **kwargs: Any) -> Any:
    """akshare 函数调用 + 请求级缓存（进程内 TTL 内存缓存）。

    同接口+参数在 TTL 内不重复发起网络请求（日报等批量脚本的防重复关键）。
    语义与旧 quant_v6 实现一致，纯内存实现，无外部依赖。
    """
    key = (fn.__module__, fn.__name__, args, tuple(sorted(kwargs.items())))
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE_MEMO.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = fn(*args, **kwargs)
    with _CACHE_LOCK:
        _CACHE_MEMO[key] = (now + ttl_seconds, value)
        # 简单防膨胀：超过 4096 条清掉最旧的 1/4
        if len(_CACHE_MEMO) > 4096:
            for k in sorted(_CACHE_MEMO, key=lambda k: _CACHE_MEMO[k][0])[: len(_CACHE_MEMO) // 4]:
                _CACHE_MEMO.pop(k, None)
    return value


# ── 进程降优先级 ────────────────────────────────────────────
def nice_process(priority: int = 10) -> None:
    """降低当前进程优先级（Unix nice；Windows 降为 BelowNormal）。"""
    try:
        if os.name == "nt":
            try:
                import psutil
                p = psutil.Process()
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                return
            except Exception:
                pass
        os.nice(priority)  # type: ignore[attr-defined]
    except Exception:
        pass


# ── 盘中引擎（quant_system 自包含实现）──────────────────────
def scan_intraday(*args: Any, **kwargs: Any) -> Any:
    """盘中分时扫描（新浪分钟源，quant_system.intraday_engine）。"""
    from quant_system.intraday_engine import scan_intraday as _f
    return _f(*args, **kwargs)


def compute_intraday_signals(*args: Any, **kwargs: Any) -> Any:
    """盘中信号计算（quant_system.intraday_signal）。"""
    from quant_system.intraday_signal import compute_intraday_signals as _f
    return _f(*args, **kwargs)


def is_trading_time(*args: Any, **kwargs: Any) -> Any:
    """是否交易时段（quant_system.realtime）。"""
    from quant_system.realtime import is_trading_time as _f
    return _f(*args, **kwargs)
