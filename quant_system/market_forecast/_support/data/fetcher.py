"""
fetcher.py — QuantV6 akshare 统一数据封装
延迟导入、socket 超时、重试、限流、异常包装。所有网络函数走这里，绝不裸调 akshare。
"""
from __future__ import annotations

import socket
import time
from typing import Any

from quant_system.market_forecast._support.common.exceptions import DataFetchError
from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.common.retry import retry

log = get_logger("qv6.fetcher")

_ak = None
_last_call_ts: dict[str, float] = {}
_MIN_INTERVAL = 0.3  # 同一接口最小调用间隔（秒），防限流


def ak():
    """延迟导入 akshare 单例。"""
    global _ak
    if _ak is None:
        import akshare
        _ak = akshare
        socket.setdefaulttimeout(10)
    return _ak


def _throttle(name: str) -> None:
    now = time.monotonic()
    last = _last_call_ts.get(name, 0)
    wait = _MIN_INTERVAL - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[name] = time.monotonic()


@retry(max_attempts=2, base_delay=1.0, exceptions=(Exception,))
def call(api_name: str, *args, **kwargs) -> Any:
    """
    调用 akshare 接口（带重试/限流）。
    用法: fetcher.call("stock_zh_a_hist", symbol="600001", ...)
    """
    a = ak()
    fn = getattr(a, api_name, None)
    if fn is None:
        raise DataFetchError(f"akshare 无此接口: {api_name}")
    _throttle(api_name)
    try:
        result = fn(*args, **kwargs)
        return result
    except Exception as e:
        raise DataFetchError(f"{api_name} 调用失败: {type(e).__name__}: {str(e)[:120]}") from e


def safe_call(api_name: str, *args, default=None, **kwargs) -> Any:
    """
    安全调用：任何失败返回 default（None/空DataFrame），绝不抛异常。
    用于非关键数据（缺失可容忍的组件）。
    """
    try:
        r = call(api_name, *args, **kwargs)
        return r if r is not None else default
    except Exception as e:
        log.warning(f"{api_name} 降级: {str(e)[:100]}")
        return default


def timed_call(api_name: str, timeout: float = 20.0, *args, **kwargs) -> Any:
    """带墙钟超时的调用（超时抛 WallTimeoutError）。"""
    import threading

    result: dict[str, Any] = {"value": None, "error": None}

    def _run():
        try:
            result["value"] = call(api_name, *args, **kwargs)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        from quant_system.market_forecast._support.common.exceptions import WallTimeoutError
        raise WallTimeoutError(f"{api_name} 超时 {timeout}s")
    if result["error"]:
        raise result["error"]  # type: ignore[misc]
    return result["value"]
