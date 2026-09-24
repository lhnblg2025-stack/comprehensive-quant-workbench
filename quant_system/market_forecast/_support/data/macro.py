# -*- coding: utf-8 -*-
"""
macro.py — 宏观数据模块

拉取中国主要宏观经济指标（akshare 宏观接口）:
    - macro_china_pmi    制造业/非制造业 PMI
    - macro_china_cpi    CPI 同比/环比
    - macro_china_ppi    PPI
    - macro_china_m2     M2 货币供应量
    - macro_china_gdp    GDP 当季值/累计值

所有网络函数捕获异常并返回空 DataFrame，绝不崩溃。
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)

#: 各宏观接口对应的 akshare 函数名（统一通过 get_macro 分发）
MACRO_API = {
    "pmi": "macro_china_pmi",
    "cpi": "macro_china_cpi",
    "ppi": "macro_china_ppi",
    "m2": "macro_china_m2",
    "gdp": "macro_china_gdp",
    "money_supply": "macro_china_money_supply",
}


def get_macro(name: str = "pmi") -> pd.DataFrame:
    """按名称拉取宏观指标。

    Args:
        name: 指标名，支持 pmi/cpi/ppi/m2/gdp/money_supply。

    Returns:
        宏观指标 DataFrame；失败或未知名称返回空 DataFrame。
    """
    if ak is None:
        return pd.DataFrame()
    func_name = MACRO_API.get(name)
    if func_name is None:
        logger.warning("未知宏观指标: %s", name)
        return pd.DataFrame()
    func = getattr(ak, func_name, None)
    if func is None:
        logger.warning("akshare 无此接口: %s", func_name)
        return pd.DataFrame()
    try:
        df = func()
        if df is None or df.empty:
            return pd.DataFrame()
        return df.copy()
    except Exception as exc:
        logger.warning("拉取宏观数据失败 %s: %s", func_name, exc)
        return pd.DataFrame()


def get_pmi() -> pd.DataFrame:
    """拉取中国制造业 PMI 数据。失败返回空 DataFrame。"""
    return get_macro("pmi")


def get_cpi() -> pd.DataFrame:
    """拉取中国 CPI 数据。失败返回空 DataFrame。"""
    return get_macro("cpi")


def get_ppi() -> pd.DataFrame:
    """拉取中国 PPI 数据。失败返回空 DataFrame。"""
    return get_macro("ppi")


def get_m2() -> pd.DataFrame:
    """拉取中国 M2 货币供应量数据。失败返回空 DataFrame。"""
    return get_macro("m2")


def get_gdp() -> pd.DataFrame:
    """拉取中国 GDP 数据。失败返回空 DataFrame。"""
    return get_macro("gdp")


def get_macro_snapshot(names: list | None = None) -> dict:
    """批量拉取多个宏观指标并打包为快照 dict。

    Args:
        names: 指标名列表，默认拉取全部支持指标。

    Returns:
        dict: {name: DataFrame}，各指标独立容错，单指标失败不影响其他指标。
    """
    names = names or list(MACRO_API.keys())
    snapshot: dict = {}
    for name in names:
        df = get_macro(name)
        snapshot[name] = df
        if df.empty:
            logger.warning("宏观指标 %s 拉取为空", name)
    return snapshot


def get_latest_value(name: str = "cpi") -> dict | None:
    """获取宏观指标最新一行的快照（日期+数值列）。

    Args:
        name: 指标名。

    Returns:
        dict: {"indicator":..., "latest": {...}}；失败返回 None。
    """
    df = get_macro(name)
    if df.empty:
        return None
    try:
        latest = df.iloc[-1].to_dict()
        return {"indicator": name, "latest": latest}
    except Exception as exc:
        logger.warning("生成宏观快照失败 %s: %s", name, exc)
        return None


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    print("PMI 行数:", len(get_pmi()))
    print("CPI 最新:", get_latest_value("cpi"))
