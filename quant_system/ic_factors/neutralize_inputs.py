"""
neutralize_inputs.py — P1-1 中性化输入构建（数据层 → industry_map / market_cap）
=================================================================================
审计 P1-1（审计_因子层.md）要求：`neutralize()`/`orthogonalize()` 是死代码，
IC 在原始（未剥离市值/行业）面板上计算。本模块负责从数据层**诚实**地构建
中性化所需的两个输入，并明确报告哪些输入真实可用、哪些缺失。

- industry_map:  Series(股票代码 → SW 一级行业)，来源 data_warehouse/market/sw_industry_map.parquet
- market_cap:    DataFrame(date × 股票代码) 流通市值（元），来源 kline 的 close × outstanding_share
                 （无独立 float_mv 列时按此推导；若 kline 缺 outstanding_share 则该字段缺失）
- 诚实原则：缺失输入时返回空/None，绝不伪造数据；调用方据此"真实中性化"或
  "明确降级为仅接通框架"，并在报告/日志中如实声明。

D4收敛登记: 中性化/正交化独特保留；本模块为其输入适配层，不强迁。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor.neutralize_inputs")

_ROOT = Path(__file__).resolve().parent.parent.parent


def build_industry_map(
    codes: list[str] | None = None,
    path: Path | None = None,
    onerror: str = "warn",
) -> pd.Series:
    """从 SW 一级行业映射表构建 股票代码→行业 Series。

    数据源: data_warehouse/market/sw_industry_map.parquet（code/industry 两列）。
    codes 非 None 时按给定代码子集过滤（缺失行业的代码保留为 NaN→中性化时归"未知"）。

    返回 Series(index=股票代码, values=行业名)。表缺失/损坏 → 空 Series（调用方自行降级）。
    """
    p = path or (_ROOT / "data_warehouse" / "market" / "sw_industry_map.parquet")
    if not p.exists():
        log.warning("行业映射表不存在，行业中性化输入缺失: %s", p)
        return pd.Series(dtype=object)
    try:
        df = pd.read_parquet(p)
    except Exception as e:  # noqa: BLE001
        log.warning("行业映射表读取失败: %s", e)
        return pd.Series(dtype=object)
    if "code" not in df.columns or "industry" not in df.columns:
        log.warning("行业映射表缺 code/industry 列，行业中性化输入缺失")
        return pd.Series(dtype=object)
    ser = df.set_index("code")["industry"].dropna()
    ser = ser[~ser.index.duplicated(keep="first")]
    if codes is not None:
        ser = ser.reindex([str(c) for c in codes]).dropna()
    log.info("行业中性化输入可用: %d 只股票映射到行业", int(ser.shape[0]))
    return ser


def build_market_cap(
    kl: dict[str, pd.DataFrame],
    common: pd.DatetimeIndex | None = None,
    codes: list[str] | None = None,
    onerror: str = "warn",
) -> pd.DataFrame:
    """从日线 kline 构建流通市值面板 DataFrame(date × 股票代码)。

    流通市值 = close × outstanding_share（close 元/股 × 股本）。
    单只股票缺 close/outstanding_share 列 → 该股票跳过（列缺失）。
    全部股票都缺 → 返回空 DataFrame（调用方降级为仅行业中性化/仅框架）。

    kl: {code: DataFrame(date/close/outstanding_share/...)}
    common: 统一交易日索引（可选，未给则用各股票日期的并集）
    """
    if not kl:
        return pd.DataFrame()
    col_map = {}
    for code, df in kl.items():
        if df is None or df.empty:
            continue
        if not {"date", "close", "outstanding_share"} <= set(df.columns):
            continue
        sub = df[["date", "close", "outstanding_share"]].copy()
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["date", "close", "outstanding_share"])
        if sub.empty:
            continue
        sub["mc"] = pd.to_numeric(sub["close"], errors="coerce") * pd.to_numeric(
            sub["outstanding_share"], errors="coerce")
        col_map[str(code)] = sub.set_index("date")["mc"]
    if not col_map:
        log.warning("kline 无 close×outstanding_share，流通市值中性化输入缺失（仅接通行业中性化或框架）")
        return pd.DataFrame()
    mc = pd.DataFrame(col_map).sort_index()
    if common is not None:
        mc = mc.reindex(common)
    log.info("市值中性化输入可用: %d 只 × %d 日", int(mc.shape[1]), int(mc.shape[0]))
    return mc


def neutralize_input_summary(
    industry_map: pd.Series,
    market_cap: pd.DataFrame,
) -> dict:
    """中性化输入可用性总览（供报告/日志诚实声明）。"""
    return {
        "industry_available": bool(industry_map.shape[0] > 0),
        "industry_count": int(industry_map.shape[0]),
        "market_cap_available": bool(not market_cap.empty),
        "market_cap_days": int(market_cap.shape[0]) if not market_cap.empty else 0,
        "market_cap_stocks": int(market_cap.shape[1]) if not market_cap.empty else 0,
    }
