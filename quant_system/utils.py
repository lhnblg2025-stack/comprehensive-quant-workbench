"""
quant_system/utils.py — 数据域公共工具层 (D1 收敛, 2026-08-11)

全系统唯一真源，收敛此前散落在各模块的私有实现：

  safe_float(x, default=None, *, clean_percent=True, clean_commas=True,
             units=None, finite=True)
    语义：把任意取值安全转 float。
      - None/bool → 返回 default
      - 数值/可解析字符串 → 有限值返回 float(x)；非有限(nan/inf) 返回 default
        （finite=True，默认）
      - 字符串可清洗逗号/百分号，支持中文单位（亿/万）倍数
      - 任何解析失败 → 返回 default
    （对应原 portfolio_diagnostics/exposure.py、fundamental.py、
    risk_management_pro.py、financial_data.py 的 _safe_float 语义。）

  to_float(x, default=0.0)
    语义：最小化安全转换。None/空串/'-' 返回 default；其余原样 float()，
    解析失败返回 default；不过滤 nan/inf、不清洗逗号/百分号。
    （对应原 market_context.py、trading_system.py、market_attribution.py 的
    _safe_float，以及 fundamental_analysis.py、cross_section.py、north_flow.py、
    sector_rotation.py、data_quality.py、market_regime.py、data_pipeline.py 的
    _to_float 语义。需要 None/NaN 语义时传 default=None/float("nan")。）

  now_cst() / today_cst()
    语义：CST (UTC+8) 时区的当前时间 / 当前日期。
    （合并自 intraday_decision.py、intraday_monitor.py、
    market_forecast/_support/common/time_utils.py 的 now_cst。）
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping

__all__ = ["safe_float", "to_float", "now_cst", "today_cst", "CST"]

CST = timezone(timedelta(hours=8))  # 北京时间 (UTC+8)


def safe_float(
    x: Any,
    default: float | None = None,
    *,
    clean_percent: bool = True,
    clean_commas: bool = True,
    units: Mapping[str, float] | None = None,
    finite: bool = True,
    allow_bool: bool = False,
) -> float | None:
    """把任意取值安全转 float（富语义：清洗 + 有限性过滤）。

    Args:
        x: 待转换取值。
        default: 转换失败/不可得时的返回值。
        clean_percent: 字符串是否去除尾部 '%'。
        clean_commas: 字符串是否去除千分位逗号。
        units: 中文单位 → 倍数字典；命中时解析值乘倍数。
        finite: True 时非有限结果(nan/inf)按转换失败处理返回 default。
        allow_bool: True 时 bool 按数值参与转换（float(True)=1.0），
                    对齐 financial_data/risk_management_pro/exposure 原语义。

    Returns:
        float 或 default。
    """
    # 2026-08-14 修复: pandas 取值常为 numpy.bool_（np.bool_ 不是 Python bool,
    # isinstance 拦截失效 → float(False)=0.0 污染财务字段, 测试实证 profit:0.0）。
    # 统一按 bool 语义拦截（np.bool_ 也拦截, 除非 allow_bool）。
    # 2026-08-14 v2: 版本无关——numpy 1.x 类型名 bool_ / 2.x 类型名 bool,
    # 按"模块=numpy + 类型名含 bool"判断（np.bytes_/np.str_ 不含 bool 不误伤）。
    _is_bool = isinstance(x, bool) or (
        type(x).__module__ == "numpy" and "bool" in type(x).__name__)
    if x is None or (_is_bool and not allow_bool):
        return default
    try:
        if isinstance(x, str):
            text = x.strip()
            if clean_commas:
                text = text.replace(",", "")
            if clean_percent:
                text = text.replace("%", "")
            multiplier = 1.0
            if units:
                for unit, factor in units.items():
                    if text.endswith(unit):
                        text = text[: -len(unit)]
                        multiplier = factor
                        break
            # 'nan'（小写）回落 default；'NaN' 交由 float() 解析（nan_fallback 决定去向），
            # 对齐 risk_management_pro 原语义对两种写法的不同处理。
            if not text or text in {"-", "--", "None", "nan"}:
                return default
            out = float(text) * multiplier
        else:
            out = float(x)
        if finite:
            if not math.isfinite(out):
                return default
        return out
    except (TypeError, ValueError):
        return default


def to_float(x: Any, default: float = 0.0, *, finite: bool = False) -> float:
    """把任意取值安全转 float（最小语义）。

    与各模块原 _safe_float(简单版)/_to_float 一致：
    None/空串/'-' 返回 default；其余原样 float()（"nan"/"inf" 原样通过），
    解析失败返回 default；不进行逗号/百分号/单位清洗。
    """
    if x in (None, "", "-"):
        return default
    try:
        out = float(x)
    except (TypeError, ValueError):
        return default
    if finite and not math.isfinite(out):
        return default
    return out


def now_cst() -> datetime:
    """返回 CST (UTC+8) 时区的当前时间。"""
    return datetime.now(CST)


def today_cst() -> date:
    """返回 CST (UTC+8) 时区的当前日期。"""
    return datetime.now(CST).date()
