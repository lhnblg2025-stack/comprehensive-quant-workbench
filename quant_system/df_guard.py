"""DataFrame 形状守卫工具。

在关键数据入口处统一校验 DataFrame，避免空表或缺失列向下游扩散成
难以定位的形状错误。默认严格开启，可用环境变量 DF_GUARD_STRICT=0
临时关闭全部校验（生产应急降级用，正常不建议关闭）。

None 语义说明：
  guard_df(None, allow_empty=False) -> None
  guard_df(None, allow_empty=True)  -> EmptyDataError
  即 None 与“空 DataFrame”是两种不同状态：默认返回 None 表示“无数据”，
  而显式允许空数据时仍传入 None 视为调用方错误。
"""
from __future__ import annotations

import os

import pandas as pd


class EmptyDataError(Exception):
    """DataFrame 为空或缺少必需列。"""


_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _strict_enabled() -> bool:
    raw = os.environ.get("DF_GUARD_STRICT", "1").strip().lower()
    return raw not in _FALSE_VALUES


def guard_df(
    df: pd.DataFrame | None,
    name: str = "dataframe",
    require_cols: tuple[str, ...] | list[str] | str = (),
    allow_empty: bool = False,
) -> pd.DataFrame:
    """校验并返回原 DataFrame（不做 copy）。

    参数:
        df: 待校验的 DataFrame；可为 None。
        name: 报错时使用的数据名称。
        require_cols: 必需的列名。
        allow_empty: 是否允许空 DataFrame。

    返回:
        原对象；仅 None + allow_empty=False 时返回 None。

    抛出:
        TypeError: 非 None 且非 pandas.DataFrame。
        EmptyDataError: 空数据、缺列，或 None + allow_empty=True。
    """
    if not _strict_enabled():
        return df

    if df is None:
        if allow_empty:
            raise EmptyDataError(f"{name} 为 None（allow_empty=True 时不允许传 None）")
        return None

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} 必须是 pandas.DataFrame，实际类型为 {type(df).__name__}")

    if len(df) == 0 and not allow_empty:
        raise EmptyDataError(f"{name} 为空")

    if isinstance(require_cols, str):
        require_cols = (require_cols,)
    else:
        require_cols = tuple(require_cols or ())
    missing = [col for col in require_cols if col not in df.columns]
    if missing:
        missing_text = ", ".join(str(col) for col in missing)
        raise EmptyDataError(f"{name} 缺少列: {missing_text}")

    return df


__all__ = ["EmptyDataError", "guard_df"]
