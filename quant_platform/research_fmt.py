# -*- coding: utf-8 -*-
"""
_format.py — research 层共享格式化工具
=====================================
- colorize_ret: 涨红跌绿（A股习惯）终端着色；markdown 输出一律用纯文本符号。
- fmt_pct / fmt_num: 数字格式化。
- bar / bar_row: ASCII 条形图。
- star_rating: 星级（★☆）。
- zscore_to_100: z-score → 0-100 logistic 映射（与 sentiment_thermometer 同口径）。

纯只读：不交易、不写仓库、不改参数。
"""
from __future__ import annotations

import sys

# ── ANSI 颜色（涨红跌绿，A股习惯）──────────────────────────
_RED = "\033[31m"      # 涨 → 红
_GREEN = "\033[32m"    # 跌 → 绿
_RESET = "\033[0m"


def _use_color(use_color: bool | None) -> bool:
    """自动检测终端是否支持颜色；显式传入则覆盖。"""
    if use_color is not None:
        return use_color
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def colorize_ret(value: float, use_color: bool | None = None) -> str:
    """涨红跌绿：正数红色、负数绿色、零无着色（带符号）。"""
    s = f"{value:+.2f}"
    if not _use_color(use_color) or abs(value) < 1e-12:
        return s
    color = _RED if value > 0 else _GREEN
    return f"{color}{s}{_RESET}"


def fmt_pct(value: float, signed: bool = True, nd: int = 2) -> str:
    """百分比格式化，如 +3.25% / -1.10% / 0.00%。"""
    sign = "+" if (signed and value > 0) else ""
    return f"{sign}{value:.{nd}f}%"


def fmt_num(value: float, nd: int = 2) -> str:
    """普通数字格式化。"""
    return f"{value:.{nd}f}"


def bar(value: float, max_value: float, width: int = 24, char: str = "█") -> str:
    """单条 ASCII 条形图（相对 max_value 归一化）。"""
    if max_value <= 0:
        return ""
    ratio = max(0.0, min(1.0, abs(value) / max_value))
    n = int(round(ratio * width))
    return char * n


def bar_row(label: str, value: float, max_value: float, width: int = 24,
            unit: str = "", signed: bool = True) -> str:
    """一行条形图：label  value  ████  （用于资金/涨跌排序）。"""
    v = f"{value:+.2f}{unit}" if signed else f"{value:.2f}{unit}"
    return f"{label:<10s} {v:>12s} |{bar(value, max_value, width)}"


def star_rating(score: float, max_stars: int = 5) -> str:
    """0-100 分 → ★ 星级（四舍五入到 0.5 星）。"""
    n = max(0.0, min(float(max_stars), score / 100.0 * max_stars))
    full = int(n)
    half = 1 if (n - full) >= 0.5 else 0
    return "★" * full + "☆" * half + "☆" * (max_stars - full - half)


def zscore_to_100(z: float) -> float:
    """z-score → 0-100 概率分（logistic 映射，与情绪温度计同口径）。"""
    import math

    return float(100.0 / (1.0 + math.exp(-z)))


def temp_label(score: float) -> str:
    """市场温度标签。"""
    if score >= 80:
        return "过热"
    if score >= 65:
        return "偏热"
    if score >= 45:
        return "中性"
    if score >= 30:
        return "偏冷"
    return "冰冷"
