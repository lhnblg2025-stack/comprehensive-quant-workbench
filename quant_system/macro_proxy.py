"""宏观高频代理模型 P3c。

低频宏观因子（PPI/PMI/CPI）发布滞后 30-40 天。本模块用日频可得的代理
序列，通过滚动线性回归把月度宏观因子投影成“每日估计值”，并输出样本内
R² 作为可靠性分数，供下游因子系统显式识别 is_proxy=True。

默认数据仓库:
  {repo_root}/data_warehouse/macro/{target}.parquet
  {repo_root}/data_warehouse/market/{proxy_name}.parquet
测试或沙箱可用环境变量 ``MACRO_DATA_DIR`` 指向一个包含 ``data_warehouse``
的仓库根目录覆盖。
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .policy import get_policy

logger = logging.getLogger("quant_macro_proxy")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 启发式映射：日度代理 -> 月度宏观目标。
# 可被 config/policy.yaml 中的 macro_proxy.proxy_map 覆盖；后续可按
# 因子 IC/样本外表现调优。
PROXY_MAP: dict[str, list[str]] = {
    # 商品价格同比 -> 工业品出厂价格同比
    "ppi_yearly": ["commodity__crude", "commodity__coal"],
    # 资金面 -> 财新制造业 PMI 景气
    "cx_pmi_yearly": ["repo_rate", "bond_futures"],
}

# 目标值列优先顺序；未命中时退化为常见的同比/今值/值列。
_TARGET_VALUE_PRIORITY = [
    "当月同比增长",
    "全国-同比增长",
    "制造业-同比增长",
    "同比增长",
    "同比",
    "今值",
    "值",
    "value",
    "target",
    "close",
    "收盘价",
]

_DATE_COLUMNS = ("日期", "date", "月份")


def _warehouse() -> Path:
    """返回 data_warehouse 路径，支持 MACRO_DATA_DIR 覆盖。"""
    env_root = os.environ.get("MACRO_DATA_DIR")
    root = Path(env_root) if env_root else _REPO_ROOT
    if root.name == "data_warehouse":
        return root
    return root / "data_warehouse"


def _get_proxy_map() -> dict[str, list[str]]:
    """合并默认 PROXY_MAP 与 policy.yaml 覆盖项。"""
    policy_map = get_policy("macro_proxy.proxy_map", None)
    if not isinstance(policy_map, dict):
        return dict(PROXY_MAP)
    merged = dict(PROXY_MAP)
    for key, value in policy_map.items():
        if isinstance(value, (list, tuple)):
            merged[str(key)] = [str(item) for item in value]
    return merged


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("missing parquet: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - 损坏文件防御
        logger.warning("failed to read parquet %s: %s", path, exc)
        return pd.DataFrame()


def _find_date_column(df: pd.DataFrame) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for key in ("date", "日期", "月份"):
        if key in lowered:
            return lowered[key]
    for key in lowered:
        if "date" in key or key in {"日期", "月份"}:
            return lowered[key]
    return None


def _find_value_column(df: pd.DataFrame, date_col: str | None) -> str | None:
    """选择代理序列的价格/值列，排除日期和常见非价格列。"""
    preferred = ["close", "收盘价", "值", "value", "今值", "FR007"]
    for col in preferred:
        if col in df.columns:
            return col
    excluded = {date_col, "date", "日期", "月份", "volume", "成交量", "持仓量",
                "chg", "ret"}
    numeric_cols = [c for c in df.columns if c not in excluded
                    and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def _find_target_value_column(df: pd.DataFrame, target: str) -> str | None:
    """选择宏观目标序列值列，特殊目标优先官方发布口径。"""
    explicit: dict[str, list[str]] = {
        "ppi_yearly": ["当月同比增长", "全国-同比增长", "同比增长", "同比"],
        "cx_pmi_yearly": ["今值", "值", "value", "close"],
    }
    candidates = list(explicit.get(target, [])) + list(_TARGET_VALUE_PRIORITY)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in df.columns:
            return candidate
        for col in df.columns:
            if candidate.lower() in str(col).lower():
                return col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None


def _parse_target_dates(values: pd.Series, column_name: str | None) -> pd.Series:
    """解析宏观日期；中文月份列统一转为月末。"""
    original = values.astype(str)
    if column_name == "月份" or original.str.contains("月份", na=False).any():
        parsed = pd.to_datetime(original, format="%Y年%m月份", errors="coerce")
        return parsed + pd.offsets.MonthEnd(0)
    return pd.to_datetime(values, errors="coerce")


def load_proxy_series(proxy_name: str) -> pd.Series:
    """读取日频代理序列并计算年同比近似值。

    Returns:
        以日期为索引、值为日度同比（百分数）的 pd.Series；文件缺失/无数据
        时返回空 Series。
    """
    path = _warehouse() / "market" / f"{proxy_name}.parquet"
    df = _read_parquet(path)
    if df.empty:
        return pd.Series(dtype=float, name=proxy_name)

    date_col = _find_date_column(df)
    value_col = _find_value_column(df, date_col)
    if date_col is None or value_col is None:
        logger.warning("cannot identify date/value columns for proxy %s", proxy_name)
        return pd.Series(dtype=float, name=proxy_name)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna(subset=["date", "value"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        return pd.Series(dtype=float, name=proxy_name)

    series = out.set_index("date")["value"].astype(float).sort_index()
    yoy = series.pct_change(periods=252, fill_method=None) * 100.0
    yoy = yoy.replace([np.inf, -np.inf], np.nan).dropna()
    yoy.name = proxy_name
    yoy.index.name = "date"
    return yoy


def _load_target_series(target: str) -> pd.Series:
    """读取月度宏观目标序列，返回月末日期 -> 值。"""
    path = _warehouse() / "macro" / f"{target}.parquet"
    df = _read_parquet(path)
    if df.empty:
        return pd.Series(dtype=float, name=target)

    date_col = _find_date_column(df)
    value_col = _find_target_value_column(df, target)
    if date_col is None or value_col is None:
        logger.warning("cannot identify date/value columns for target %s", target)
        return pd.Series(dtype=float, name=target)

    dates = _parse_target_dates(df[date_col], date_col)
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.DataFrame({"date": dates, "value": values}).dropna(
        subset=["date", "value"]
    )
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        return pd.Series(dtype=float, name=target)

    series = out.set_index("date")["value"].astype(float).sort_index()
    series.name = target
    series.index.name = "date"
    return series


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """线性回归；优先 sklearn，不可用时退化 numpy.linalg.lstsq。"""
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
    except Exception:
        x_design = np.column_stack([x, np.ones(len(x))])
        coef_all, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        coef, intercept = coef_all[:-1], float(coef_all[-1])
        y_hat = x_design @ coef_all
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return coef, intercept, r2

    model = LinearRegression().fit(x, y)
    return model.coef_, float(model.intercept_), float(model.score(x, y))


def build_proxy_model(target: str, lookback_years: int = 3) -> dict[str, Any] | None:
    """用最近 lookback_years 年对齐数据拟合目标 ~ 日频代理。

    数据不足 12 个对齐月、代理缺失或目标不存在时返回 None，不抛异常。
    """
    proxy_map = _get_proxy_map()
    proxies = proxy_map.get(target)
    if not proxies:
        logger.warning("no proxy mapping for target=%s", target)
        return None

    target_series = _load_target_series(target)
    if len(target_series) < 12:
        logger.warning("insufficient target months for %s: n=%s", target, len(target_series))
        return None

    aligned = target_series.to_frame("target")
    for proxy_name in proxies:
        proxy_series = load_proxy_series(proxy_name)
        if proxy_series.empty:
            logger.warning("insufficient proxy data for %s", proxy_name)
            return None
        monthly = proxy_series.resample("ME").last()
        aligned[proxy_name] = monthly

    aligned = aligned.dropna()
    if len(aligned) < 12:
        logger.warning("insufficient aligned months for %s: n=%s", target, len(aligned))
        return None

    last_date = aligned.index.max()
    start = last_date - pd.DateOffset(years=lookback_years) + pd.Timedelta(days=1)
    window = aligned.loc[aligned.index >= start]
    if len(window) < 12:
        logger.warning(
            "insufficient in-window months for %s: n=%s (need >=12)",
            target,
            len(window),
        )
        return None

    x = window[proxies].to_numpy(dtype=float)
    y = window["target"].to_numpy(dtype=float)
    coef, intercept, r2 = _fit_linear_regression(x, y)
    coef_map = dict(zip(proxies, [float(v) for v in coef]))

    return {
        "target": target,
        "proxies": list(proxies),
        "r2": float(r2),
        "reliability_score": float(r2),
        "coef": coef_map,
        "intercept": float(intercept),
        "n": int(len(window)),
        "last_date": pd.Timestamp(last_date),
        "is_proxy": True,
        "insufficient": False,
    }


def _empty_daily_estimate(target: str, reason: str = "insufficient") -> pd.Series:
    series = pd.Series(dtype=float, name=f"{target}_daily_proxy")
    series.attrs = {
        "is_proxy": True,
        "target": target,
        "insufficient": True,
        "reason": reason,
    }
    return series


def daily_estimate(target: str, as_of: str | pd.Timestamp | None = None) -> pd.Series:
    """用最近一次回归系数把日度代理投影为每日宏观估计。

    返回以日期为索引的 Series；as_of 之后的数据会被截断。样本不足或文件
    缺失返回空 Series，attrs 标记 is_proxy=True 与 insufficient=True。
    """
    model = build_proxy_model(target)
    if model is None:
        return _empty_daily_estimate(target)

    proxies = model["proxies"]
    daily_frames = []
    for proxy_name in proxies:
        series = load_proxy_series(proxy_name)
        if series.empty:
            return _empty_daily_estimate(target, reason=f"missing_{proxy_name}")
        daily_frames.append(series)

    x = pd.concat(daily_frames, axis=1, join="inner").dropna()
    if x.empty:
        return _empty_daily_estimate(target, reason="no_aligned_daily_data")

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        x = x.loc[x.index <= cutoff]
        if x.empty:
            return _empty_daily_estimate(target, reason="as_of_before_data")

    coefs = np.array([model["coef"][name] for name in proxies], dtype=float)
    estimate = model["intercept"] + x.to_numpy(dtype=float) @ coefs
    out = pd.Series(estimate, index=x.index, name=f"{target}_daily_proxy")
    out.index.name = "date"
    out.attrs = {
        "is_proxy": True,
        "target": target,
        "reliability_score": model["reliability_score"],
        "r2": model["r2"],
        "as_of": str(pd.Timestamp(as_of)) if as_of is not None else str(x.index.max()),
        "insufficient": False,
    }
    return out


def _print_daily_estimate(target: str, as_of: str | pd.Timestamp | None = None,
                          lookback_years: int = 3) -> int:
    model = build_proxy_model(target, lookback_years=lookback_years)
    if model is None:
        print(f"target={target} insufficient_data=True")
        return 0

    print(
        f"target={target} r2={model['r2']:.4f} "
        f"reliability_score={model['reliability_score']:.4f} "
        f"n={model['n']} last_date={pd.Timestamp(model['last_date']).date()}"
    )
    estimate = daily_estimate(target, as_of=as_of)
    if estimate.empty:
        print("target={} insufficient_data=True".format(target))
        return 0
    tail = estimate.tail(5)
    print(tail.to_string(float_format=lambda value: f"{value:.4f}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="宏观高频代理估计")
    parser.add_argument("--target", default="ppi_yearly", help="宏观目标名")
    parser.add_argument("--print", action="store_true", help="打印最近 5 日估计")
    parser.add_argument("--as-of", default=None, help="估计截止日期")
    parser.add_argument("--lookback-years", type=int, default=3)
    args = parser.parse_args(argv)

    if args.print:
        return _print_daily_estimate(
            args.target,
            as_of=args.as_of,
            lookback_years=args.lookback_years,
        )
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
