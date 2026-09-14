"""rate_limiter.py — 统一请求节流层（P2）

设计:
  - RateLimiter: 按 host 分桶的 token bucket（threading.Lock 保护，多线程幂等安全）
  - wait(): 阻塞直到拿到令牌；单次等待时间带随机抖动（默认 ±20%）
  - retry(): 失败指数退避，只在网络类异常重试
             （requests.RequestException / TimeoutError / ConnectionError），
             业务异常一律不重试、直接抛出
  - get_limiter(host): 按 host 缓存并返回全局 RateLimiter 实例

纯标准库、单文件自包含；import 本模块不触发任何网络请求。
"""

from __future__ import annotations

import os
import random
import threading
import time
from functools import wraps

__all__ = ["RateLimiter", "get_limiter", "retry"]


def _normalize_host(host: str) -> str:
    """把 URL/域名归一为 host key（去掉 scheme/端口/路径，统一小写）。"""
    h = str(host or "").strip().lower()
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/", 1)[0]
    h = h.split(":", 1)[0]
    return h or "default"


class RateLimiter:
    """按 host 分桶的令牌桶限流器。

    桶容量 = rps（初始满桶，允许 1 秒突发），令牌按 rps 速率补充；
    缺令牌时 wait() 阻塞，实际等待时长乘以随机抖动系数。
    """

    def __init__(self, rps: float = 3.0, jitter: float = 0.2,
                 burst: float | None = None) -> None:
        self.rps = float(rps)
        self.jitter = float(jitter)
        self.capacity = float(burst) if burst is not None else self.rps
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> float:
        """阻塞直到拿到一个令牌；返回实际等待的秒数。"""
        waited = 0.0
        while True:
            delay = self._try_take()
            if delay <= 0.0:
                return waited
            time.sleep(delay)
            waited += delay

    def _try_take(self) -> float:
        """取令牌：成功返回 0，缺令牌返回需等待秒数（含抖动）。"""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity,
                               self._tokens + (now - self._last) * self.rps)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            need = (1.0 - self._tokens) / self.rps
            return need * random.uniform(1.0 - self.jitter, 1.0 + self.jitter)


_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] | None = None
_NETWORK_EXCEPTIONS_LOCK = threading.Lock()


def _network_exceptions() -> tuple[type[BaseException], ...]:
    """返回应重试的网络类异常元组（延迟 import requests，缺席时用内建类兜底）。"""
    global _NETWORK_EXCEPTIONS
    if _NETWORK_EXCEPTIONS is None:
        with _NETWORK_EXCEPTIONS_LOCK:
            if _NETWORK_EXCEPTIONS is None:
                excs: list[type[BaseException]] = [TimeoutError, ConnectionError]
                try:
                    import requests as _requests
                    excs.append(_requests.RequestException)
                except ImportError:
                    pass
                _NETWORK_EXCEPTIONS = tuple(excs)
    return _NETWORK_EXCEPTIONS


def retry(fn=None, *, tries: int = 3, base: float = 1.0, backoff: float = 2.0):
    """失败指数退避重试（装饰器或直接调用），只在网络类异常时重试。

    用法:
        @retry
        def fetch(): ...

        @retry(tries=5, base=0.5, backoff=2.0)
        def fetch(): ...

        retry(fetch, tries=3)(args)
    """
    if fn is None:
        def _decorator(func):
            return retry(func, tries=tries, base=base, backoff=backoff)
        return _decorator

    attempts = max(1, int(tries))

    @wraps(fn)
    def wrapper(*args, **kwargs):
        delay = float(base)
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except _network_exceptions() as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    break
                time.sleep(delay)
                delay *= float(backoff)
        assert last_exc is not None
        raise last_exc
    return wrapper


_DEFAULT_RPS = float(os.environ.get("RATE_LIMITER_RPS", "3.0"))
_REGISTRY: dict[str, RateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_limiter(host: str, rps: float | None = None) -> RateLimiter:
    """按 host 返回（缓存）全局 RateLimiter 实例。

    rps 缺省用默认 3.0，可用环境变量 RATE_LIMITER_RPS 全局覆盖，
    或按调用方显式传入覆盖。
    """
    key = _normalize_host(host)
    rate = _DEFAULT_RPS if rps is None else float(rps)
    with _REGISTRY_LOCK:
        limiter = _REGISTRY.get(key)
        if limiter is None:
            limiter = RateLimiter(rps=rate)
            _REGISTRY[key] = limiter
        return limiter
