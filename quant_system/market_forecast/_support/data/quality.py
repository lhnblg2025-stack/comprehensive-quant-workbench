# -*- coding: utf-8 -*-
"""
quality.py — 数据质量检测模块

对拉取到的 DataFrame 进行质量体检，提供:
    - missing_rate_report  缺失率统计
    - detect_outliers      异常值检测（IQR / Z-Score）
    - detect_jumps         跳变检测（相邻值突变，常用于价格/资金流序列）
    - report_data_quality  一键生成完整质量报告（dict）

本模块为纯本地计算，不涉及网络，始终返回结构化报告结果。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 空报告各字段的标准列名
_MISSING_COLS = ["column", "missing_count", "missing_rate"]


def missing_rate_report(df: pd.DataFrame) -> pd.DataFrame:
    """统计各列缺失率，按缺失率降序；输入为空返回空 DataFrame。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=_MISSING_COLS)
    try:
        missing_count = df.isna().sum()
        report = pd.DataFrame({
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_rate": (missing_count.values / len(df)).round(4),
        })
        return report.sort_values("missing_rate", ascending=False).reset_index(drop=True)
    except Exception as exc:
        logger.warning("缺失率统计失败: %s", exc)
        return pd.DataFrame(columns=_MISSING_COLS)


def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """异常值检测，返回布尔掩码（True 表示异常）。

    Args:
        series: 待检测数值序列（NaN 自动忽略）。
        method: iqr=四分位距法; zscore=Z-Score 法。
        threshold: IQR 法取倍数（默认 3）；Z-Score 法取阈值（默认 3）。

    Returns:
        bool Series，与输入索引对齐；非数值/全空序列返回全 False。
    """
    empty_mask = pd.Series(False, index=series.index if series is not None else [])
    if series is None or series.empty:
        return empty_mask
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return empty_mask
        if method == "zscore":
            std = s.std(ddof=0)
            flag = (s - s.mean()).abs() > threshold * std if std > 0 else pd.Series(False, index=s.index)
        else:  # iqr
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return empty_mask
            flag = (s < q1 - threshold * iqr) | (s > q3 + threshold * iqr)
        mask = empty_mask.copy()
        mask.loc[flag.index] = flag
        return mask
    except Exception as exc:
        logger.warning("异常值检测失败: %s", exc)
        return empty_mask


def detect_jumps(series: pd.Series, pct_threshold: float = 0.1) -> pd.Series:
    """跳变检测：相邻值变动幅度超过阈值即视为跳变；首位置恒为 False。"""
    empty_mask = pd.Series(False, index=series.index if series is not None else [])
    if series is None or series.empty:
        return empty_mask
    try:
        s = pd.to_numeric(series, errors="coerce")
        denom = s.shift(1).abs().replace(0, np.nan)
        mask = ((s - s.shift(1)) / denom).abs() > pct_threshold
        mask.iloc[0] = False  # 首行无前值，不判跳变
        return mask.fillna(False)
    except Exception as exc:
        logger.warning("跳变检测失败: %s", exc)
        return empty_mask


def report_data_quality(
    df: pd.DataFrame,
    numeric_cols: list | None = None,
    jump_cols: list | None = None,
    outlier_method: str = "iqr",
) -> dict:
    """生成数据质量总报告（缺失率 + 异常值 + 跳变 + 基础统计）。

    Args:
        df: 待检测 DataFrame。
        numeric_cols: 参与异常值检测的数值列，默认自动选取。
        jump_cols: 参与跳变检测的列，默认取 numeric_cols。
        outlier_method: 异常值方法 iqr/zscore。

    Returns:
        dict: {"shape", "missing", "outliers", "jumps", "basic_stats"}；
        空输入返回仅含 shape 的字典。
    """
    empty_report = {"shape": (0, 0), "missing": pd.DataFrame(), "outliers": {}, "jumps": {}, "basic_stats": pd.DataFrame()}
    if df is None or df.empty:
        return empty_report
    try:
        num_df = df.select_dtypes(include=[np.number])
        numeric_cols = numeric_cols or list(num_df.columns)
        jump_cols = jump_cols or numeric_cols
        outliers = {c: int(detect_outliers(df[c], method=outlier_method).sum()) for c in numeric_cols if c in df.columns}
        jumps = {c: int(detect_jumps(df[c]).sum()) for c in jump_cols if c in df.columns}
        return {
            "shape": df.shape,
            "missing": missing_rate_report(df),
            "outliers": outliers,
            "jumps": jumps,
            "basic_stats": num_df.describe() if not num_df.empty else pd.DataFrame(),
        }
    except Exception as exc:
        logger.warning("生成数据质量报告失败: %s", exc)
        return {"shape": df.shape, **empty_report}
