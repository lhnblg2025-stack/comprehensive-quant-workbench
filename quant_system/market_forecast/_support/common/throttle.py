"""
throttle.py — QuantV6 CPU 节流
根据 CPU 负载动态调整轮询/任务间隔，低优先级运行。
"""
from __future__ import annotations
import logging

import os
import time
from typing import Callable


def get_cpu_load() -> float:
    """读取 1 分钟平均负载（Linux /proc/loadavg），返回 0~核数 的负载值。"""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().split()
            return float(parts[0])
    except Exception:
        return 0.0


def get_cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


class CpuThrottle:
    """
    动态节流器：负载越高，间隔越长。
    - base_interval: 低负载时的基础间隔（秒）
    - max_interval: 高负载时的最大间隔
    - threshold: 负载超过该倍数（相对核数）开始放大间隔
    """

    def __init__(self, base_interval: float = 5.0, max_interval: float = 60.0,
                 threshold: float = 0.8):
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.threshold = threshold
        self._last: float | None = None

    def current_interval(self) -> float:
        load = get_cpu_load()
        cores = get_cpu_count()
        ratio = load / max(cores, 1)
        if ratio <= self.threshold:
            return self.base_interval
        # 负载超阈值：线性放大到 max_interval
        excess = (ratio - self.threshold) / max(1.0 - self.threshold, 0.01)
        return min(self.base_interval + excess * (self.max_interval - self.base_interval),
                   self.max_interval)

    def wait(self) -> None:
        """按当前负载等待。"""
        interval = self.current_interval()
        time.sleep(interval)
        self._last = interval

    def wait_until(self, target_monotonic: float) -> None:
        """等到目标时间（分段 sleep，中途可响应）。"""
        while True:
            remain = target_monotonic - time.monotonic()
            if remain <= 0:
                return
            time.sleep(min(remain, self.current_interval()))


def nice_process(priority: int = 10) -> None:
    """降低当前进程优先级（Unix）。"""
    try:
        os.nice(priority)
    except Exception as e:
        logging.getLogger(__name__).error(f"[throttle] 操作失败: {e}", exc_info=True)


def with_cpu_throttle(func: Callable, base_interval: float = 5.0):
    """装饰器：每次调用前先按负载等待。"""
    throttle = CpuThrottle(base_interval=base_interval)

    def wrapper(*args, **kwargs):
        throttle.wait()
        return func(*args, **kwargs)

    return wrapper
