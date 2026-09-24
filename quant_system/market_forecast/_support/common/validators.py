"""
validators.py — QuantV6 参数校验
symbol/日期/权重/正数/范围 校验装饰器与函数。
"""
from __future__ import annotations

import re
from functools import wraps
from typing import Any, Callable

from quant_system.market_forecast._support.common.exceptions import ConfigError, StrategyError

_SYMBOL_RE = re.compile(r"^(6\d{5}|0\d{5}|3\d{5}|4\d{5}|8\d{5}|9\d{5})$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_valid_symbol(code: str) -> bool:
    """A股 6 位代码校验（60/00/30/43/83/92 等）。"""
    return bool(_SYMBOL_RE.match(str(code)))


def normalize_symbol(code: str) -> str:
    """规范化代码：去 sh/sz/bj 前缀，补零。"""
    code = str(code).strip().lower()
    for p in ("sh", "sz", "bj"):
        if code.startswith(p):
            code = code[len(p):]
    return code.zfill(6) if code.isdigit() else code


def is_valid_date(s: str) -> bool:
    return bool(_DATE_RE.match(str(s)))


def validate_symbol(func: Callable) -> Callable:
    """装饰器：symbol 参数必须是合法代码。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        for a in args:
            if isinstance(a, str) and re.fullmatch(r"[a-zA-Z]*\d{4,6}", a):
                if not is_valid_symbol(normalize_symbol(a)):
                    raise ConfigError(f"非法股票代码: {a}")
        return func(*args, **kwargs)

    return wrapper


def validate_weights(weights: dict[str, float], tolerance: float = 1e-6) -> dict[str, float]:
    """权重校验：非负、和为1（容差内自动归一）。"""
    if not weights:
        return {}
    total = sum(v for v in weights.values() if isinstance(v, (int, float)))
    if total <= 0:
        raise StrategyError("权重总和必须为正")
    out = {k: float(v) / total for k, v in weights.items() if v > 0}
    if abs(sum(out.values()) - 1.0) > tolerance:
        raise StrategyError("权重无法归一化")
    return out


def require_positive(name: str, value: float) -> float:
    if value is None or value <= 0:
        raise ConfigError(f"{name} 必须为正数: {value}")
    return float(value)


def require_range(name: str, value: float, lo: float, hi: float) -> float:
    v = float(value)
    if not (lo <= v <= hi):
        raise ConfigError(f"{name} 超出范围 [{lo}, {hi}]: {v}")
    return v


def ensure_dict(value: Any, name: str = "参数") -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} 必须是 dict")
    return value
