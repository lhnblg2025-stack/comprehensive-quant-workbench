"""
retry.py — QuantV6 重试装饰器
指数退避 + 抖动，按异常类型重试，最大次数与总超时控制。
"""
from __future__ import annotations
import logging

import random
import time
from functools import wraps
from typing import Callable, Type

from quant_system.market_forecast._support.common.exceptions import QuantV6Error


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: float = 0.3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """
    重试装饰器。
    - max_attempts: 最大尝试次数（含首次）
    - base_delay/max_delay: 指数退避 2^n * base，封顶 max_delay
    - jitter: 随机抖动比例
    - exceptions: 仅这些异常触发重试
    - on_retry: 每次重试前的回调 (exc, attempt)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt >= max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay *= 1 + random.uniform(-jitter, jitter)
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as e:
                            logging.getLogger(__name__).error(f"[retry] 操作失败: {e}", exc_info=True)
                    time.sleep(max(0.0, delay))
            raise last_exc  # type: ignore[misc]
        return wrapper

    return decorator


def retry_silent(func: Callable, max_attempts: int = 2, **kwargs):
    """单次调用版：失败返回 None（用于网络拉数，绝不抛异常）。"""
    for attempt in range(max_attempts):
        try:
            return func(**kwargs)
        except Exception:
            if attempt >= max_attempts - 1:
                return None
            time.sleep(1.0)
    return None


class CircuitBreaker:
    """熔断器：连续失败 n 次后熔断 open_duration 秒，期间直接抛 QuantV6Error。"""

    def __init__(self, fail_threshold: int = 5, open_duration: float = 60.0):
        self.fail_threshold = fail_threshold
        self.open_duration = open_duration
        self._fails = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return time.monotonic() - self._opened_at < self.open_duration

    def call(self, func: Callable, *args, **kwargs):
        if self.is_open:
            raise QuantV6Error(f"熔断器开启，拒绝调用 {getattr(func, '__name__', 'func')}")
        try:
            result = func(*args, **kwargs)
            self._fails = 0
            self._opened_at = None
            return result
        except Exception:
            self._fails += 1
            if self._fails >= self.fail_threshold:
                self._opened_at = time.monotonic()
            raise

    def reset(self) -> None:
        self._fails = 0
        self._opened_at = None
