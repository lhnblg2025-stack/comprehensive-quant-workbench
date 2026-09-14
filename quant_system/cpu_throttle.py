"""
CPU 资源节流工具 — 应用于所有后台任务/轮询的底层。

P2-Q28-fix(L370): 本模块当前全项目无任何导入方（死代码），保留作为独立工具；
文档已与实现统一（阈值按每核负载 load/CPU_COUNT 归一化判定，而非绝对负载）。

用法:
    from quant_system.cpu_throttle import CpuThrottle

    throttle = CpuThrottle()

    # 在轮询循环中：
    while True:
        do_work()
        throttle.sleep(default_ms=60000)  # 自动根据CPU负载动态调整

    # 或在函数前加装饰器：
    @throttle.wrap(interval_ms=30000)
    def my_poll():
        ...

功能:
    - 读取 /proc/loadavg 判断 CPU 负载（1 分钟平均）
    - 负载按 CPU 核心数归一化: load_per_core = load / CPU_COUNT
    - load_per_core > 2.0 → 增大间隔 1.5x
    - load_per_core > 4.0 → 增大间隔 3x
    - load_per_core > 6.0 → 跳过本次轮询
    - 每 10 秒检查一次负载（不每轮都读/proc）
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("cpu_throttle")

CPU_COUNT = os.cpu_count() or 1
LOAD_THRESHOLDS = [
    (2.0, 1.5, "轻度"),    # load > 2*core → 1.5x delay
    (4.0, 3.0, "中度"),    # load > 4*core → 3x delay
    (6.0, 0, "重度"),      # load > 6*core → skip
]

# 全局单例，支持跨模块共享
_global_throttle: "CpuThrottle | None" = None


def get_global() -> "CpuThrottle":
    global _global_throttle
    if _global_throttle is None:
        _global_throttle = CpuThrottle()
    return _global_throttle


class CpuThrottle:
    """CPU 感知的节流器，动态调整轮询间隔。"""

    def __init__(self, check_interval: float = 10.0):
        self._check_interval = check_interval
        self._last_check = 0.0
        self._current_load = 0.0
        self._factor = 1.0
        self._skip_next = False
        self._lock = threading.Lock()

    def _read_load(self) -> float:
        """读取 /proc/loadavg 1 分钟平均负载。"""
        try:
            raw = Path("/proc/loadavg").read_text().strip().split()
            return float(raw[0]) if raw else 0.0
        except Exception:
            return 0.0

    def _update_if_stale(self) -> None:
        # Fast path: read _last_check without lock (GIL-safe for simple floats)
        if time.monotonic() - self._last_check < self._check_interval:
            return
        with self._lock:
            # Re-check with fresh timestamp inside lock
            if time.monotonic() - self._last_check < self._check_interval:
                return
            self._last_check = time.monotonic()
            load = self._read_load()
            self._current_load = load
            # 按 CPU 核心数归一化
            load_per_core = load / CPU_COUNT
            factor = 1.0
            skip = False
            for threshold, mult, _ in LOAD_THRESHOLDS:
                if load_per_core >= threshold:
                    factor = mult
                    skip = (mult == 0)
            self._factor = factor
            self._skip_next = skip
            if factor > 1.5 or skip:
                logger.info(
                    f"CPU 负载 {load:.1f}({load_per_core:.1f}/核), "
                    f"节流因子 {factor}x{' 跳过' if skip else ''}"
                )

    @property
    def load_factor(self) -> float:
        self._update_if_stale()
        with self._lock:
            return self._factor

    @property
    def should_skip(self) -> bool:
        self._update_if_stale()
        with self._lock:
            return self._skip_next

    def sleep(self, default_ms: float = 60000) -> None:
        """动态延迟。高负载时延长，超高负载时直接返回（跳过本次）。"""
        self._update_if_stale()
        with self._lock:
            skip = self._skip_next
            factor = self._factor
        if skip:
            logger.debug("CPU 负载过高，跳过本次轮询")
            return
        delay = default_ms * factor
        time.sleep(delay / 1000.0)

    def wrap(self, interval_ms: float = 30000):
        """装饰器：使函数在轮询循环中受 CPU 节流控制。"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self._update_if_stale()
                with self._lock:
                    skip = self._skip_next
                    factor = self._factor
                if skip:
                    logger.debug(f"CPU 负载过高，跳过 {func.__name__}")
                    time.sleep(interval_ms / 1000.0 / CPU_COUNT)
                    return
                result = func(*args, **kwargs)
                delay = interval_ms * factor
                time.sleep(delay / 1000.0)
                return result
            return wrapper
        return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = CpuThrottle(check_interval=3)
    for i in range(5):
        print(f"Iteration {i}: factor={t.load_factor:.1f}, skip={t.should_skip}")
        t.sleep(10000)
