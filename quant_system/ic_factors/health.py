"""
health.py — 因子健康度断言与熔断
================================
对单因子 / 因子面板做缺失率、值域、常数性检查，产出健康报告；
status=fail 的因子在 IC 计算或因子输出时被标记 / 过滤（熔断），
避免缺失率过高、值域越界（RSI>100、收益率>100% 等）、常数因子
进入截面污染信号。

用法:
    from quant_system.ic_factors.health import check_factor_health, filter_unhealthy
    report = check_factor_health(series, name="tech_rsi14",
                                 rules={"min": 0, "max": 100})
    frame = filter_unhealthy(factor_frame, health_df)   # 熔断：剔除 FAIL 列
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 因子健康检查独特保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor.health")

# 默认健康断言规则（与任务口径一致）
DEFAULT_MAX_MISSING_RATIO = 0.30   # 缺失率 > 30% → FAIL
DEFAULT_RULES: dict = {"max_missing_ratio": DEFAULT_MAX_MISSING_RATIO}


def _as_float_series(series) -> pd.Series:
    """容忍 list / np.ndarray / Series 输入，统一 float Series。"""
    if isinstance(series, pd.Series):
        return series.astype("float64")
    return pd.Series(series, dtype="float64")


def check_factor_health(series, name: str = "", rules: dict | None = None) -> dict:
    """单因子健康断言。

    rules 支持:
      - max_missing_ratio: 缺失率上限（默认 0.30，超过 → FAIL）
      - min / max: 值域边界（越界 → FAIL）
      - constant_ok: 是否允许常数因子（默认 False，std==0 且非全 NaN → FAIL）

    返回报告 dict: {name, missing_ratio, min, max, std, status, issues[]}
    status: "ok" / "fail"
    """
    rules = {**DEFAULT_RULES, **(rules or {})}
    s = _as_float_series(series)
    valid = s.dropna()
    n_total = int(len(s))
    n_valid = int(len(valid))
    missing_ratio = 1.0 - (n_valid / n_total) if n_total else 1.0
    issues: list[str] = []

    if n_total == 0 or n_valid == 0:
        issues.append("全 NaN / 空")
    else:
        min_v = float(valid.min())
        max_v = float(valid.max())
        std_v = float(valid.std())
        if missing_ratio > rules.get("max_missing_ratio", DEFAULT_MAX_MISSING_RATIO):
            issues.append(f"缺失率 {missing_ratio:.1%} > {rules['max_missing_ratio']:.0%}")
        if "min" in rules and min_v < float(rules["min"]) - 1e-12:
            issues.append(f"值域越界: min {min_v:.6g} < {rules['min']}")
        if "max" in rules and max_v > float(rules["max"]) + 1e-12:
            issues.append(f"值域越界: max {max_v:.6g} > {rules['max']}")
        if not rules.get("constant_ok", False) and std_v == 0:
            issues.append("常数因子 (std == 0)")
        if np.isinf(valid).any():
            issues.append("含 inf 非有限值")

    status = "fail" if issues else "ok"
    return {
        "name": name,
        "missing_ratio": round(missing_ratio, 4),
        "min": round(float(valid.min()), 6) if n_valid else np.nan,
        "max": round(float(valid.max()), 6) if n_valid else np.nan,
        "std": round(float(valid.std()), 6) if n_valid else np.nan,
        "status": status,
        "issues": issues,
    }


def registry_rules(name: str) -> dict:
    """取注册表为该因子配置的 health 规则（未配置 → 空 dict，用默认规则）。"""
    try:
        from quant_system.ic_factors.registry import get_factor
        meta = get_factor(name)
        return dict(meta.health_rules) if meta is not None else {}
    except Exception:  # noqa: BLE001 - 注册表未就绪时静默回退
        return {}


def check_frame_health(frame: pd.DataFrame,
                       rules_map: dict[str, dict] | None = None) -> pd.DataFrame:
    """因子面板逐列健康断言。

    frame: DataFrame（列=因子，index=日期或股票代码）
    rules_map: {因子名: rules}；未提供的因子自动取 registry.health_rules，
    仍无则用默认规则（缺失率 > 30%）。
    返回: DataFrame(index=因子名, columns=[name, missing_ratio, min, max, std,
                                         status, issues])
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["name", "missing_ratio", "min", "max",
                                     "std", "status", "issues"])
    rows = []
    for col in frame.columns:
        rules = (rules_map or {}).get(col) or registry_rules(col) or None
        rows.append(check_factor_health(frame[col], name=str(col), rules=rules))
    out = pd.DataFrame(rows).set_index("name")
    out.index.name = "factor"
    return out


def filter_unhealthy(frame: pd.DataFrame,
                     health_df: pd.DataFrame | None = None,
                     rules_map: dict[str, dict] | None = None) -> pd.DataFrame:
    """熔断过滤：剔除健康检查 FAIL 的因子列。

    health_df 缺省时内部先跑 check_frame_health。
    返回只含 status=ok 列的因子面板（IC 计算 / 因子输出前调用）。
    """
    if frame is None or frame.empty:
        return frame
    if health_df is None:
        health_df = check_frame_health(frame, rules_map)
    if health_df.empty:
        return frame
    ok_cols = health_df.index[health_df["status"] == "ok"].tolist()
    drop = [c for c in frame.columns if c not in ok_cols]
    if drop:
        log.info(f"熔断过滤 FAIL 因子 {len(drop)} 个: {drop[:20]}")
    return frame.drop(columns=drop)
