# -*- coding: utf-8 -*-
"""
universe.py — 股票池构建模块

提供三类股票池:
    1. 全 A 股票池: ak.stock_zh_a_spot_em（失败降级 ak.stock_zh_a_spot）
    2. 指数成分池:  ak.index_stock_cons_weight_csindex 构建
                    沪深300(000300)/中证500(000905)/中证1000(000852)
    3. 过滤规则:   剔除 ST/*ST/退市/北交所(代码以 4/8/92 开头) 股票

所有网络函数捕获异常并返回空列表/空 DataFrame，绝不崩溃。
"""

from __future__ import annotations

import logging
import re

import pandas as pd

try:
    import akshare as ak
except Exception as exc:  # pragma: no cover
    ak = None
    logging.getLogger(__name__).warning("akshare 导入失败: %s", exc)

logger = logging.getLogger(__name__)

#: 指数代码 -> 中文名
INDEX_MAP = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000"}
#: 北交所代码前缀
_BJ_PREFIX = ("4", "8", "92")
#: 风险警示/退市名称模式
_RISK_PATTERN = re.compile(r"(ST|退)", re.IGNORECASE)


def _filter_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤 ST/*ST/退市/北交所股票，保留正常 A 股。异常时返回空 DataFrame。"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        out = df.copy()
        code_col = "代码" if "代码" in out.columns else "code"
        if code_col not in out.columns:
            logger.warning("全A快照缺少代码列，无法过滤")
            return pd.DataFrame()
        out = out[~out[code_col].astype(str).str.startswith(_BJ_PREFIX)]
        name_col = "名称" if "名称" in out.columns else "name"
        if name_col in out.columns:
            out = out[~out[name_col].astype(str).str.contains(_RISK_PATTERN, regex=True)]
        return out.reset_index(drop=True)
    except Exception as exc:
        logger.warning("过滤股票池失败: %s", exc)
        return pd.DataFrame()


def get_all_a_spot() -> pd.DataFrame:
    """获取全 A 股实时快照；主接口失败时降级到旧接口，都失败返回空 DataFrame。"""
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return df.copy()
        logger.warning("stock_zh_a_spot_em 返回空，尝试降级接口")
    except Exception as exc:
        logger.warning("stock_zh_a_spot_em 失败: %s，尝试降级接口", exc)
    try:
        df = ak.stock_zh_a_spot()  # 新浪旧接口，列名为英文 code/name
        if df is not None and not df.empty:
            return df.copy()
    except Exception as exc:
        logger.warning("stock_zh_a_spot 降级失败: %s", exc)
    return pd.DataFrame()


def get_all_a_universe() -> list:
    """构建过滤后的全 A 股票代码池（剔除 ST/退市/北交所）。失败返回空列表。"""
    df = _filter_stocks(get_all_a_spot())
    if df.empty:
        return []
    code_col = "代码" if "代码" in df.columns else "code"
    try:
        # 兼容新浪接口的 sh600000 格式，统一取后 6 位
        codes = df[code_col].astype(str).str[-6:].tolist()
        return [c for c in codes if c.isdigit()]
    except Exception as exc:
        logger.warning("生成全A代码列表失败: %s", exc)
        return []


def get_index_cons(index_code: str = "000300") -> list:
    """拉取指数成分股权重表并返回成分股代码列表。失败返回空列表。"""
    if ak is None:
        return []
    try:
        df = ak.index_stock_cons_weight_csindex(symbol=index_code)
        if df is None or df.empty or "成分券代码" not in df.columns:
            return []
        codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
        return list(dict.fromkeys(codes))  # 去重保序
    except Exception as exc:
        logger.warning("拉取指数成分失败 %s: %s", index_code, exc)
        return []


def get_index_universe(index_code: str = "000300") -> dict:
    """构建指数成分股票池（过滤后代码列表 + 原始权重表）。失败返回 codes=[] 的空 dict。"""
    result = {"index": index_code, "name": INDEX_MAP.get(index_code, index_code), "codes": [], "weight_df": pd.DataFrame()}
    if ak is None:
        return result
    try:
        df = ak.index_stock_cons_weight_csindex(symbol=index_code)
        if df is None or df.empty:
            return result
        codes = [c for c in get_index_cons(index_code) if not c.startswith(_BJ_PREFIX)]
        result["codes"] = codes
        result["weight_df"] = df.copy()
        return result
    except Exception as exc:
        logger.warning("构建指数池失败 %s: %s", index_code, exc)
        return result


def get_multi_index_universe(index_codes: list | None = None) -> dict:
    """批量构建多个指数股票池；单指数失败不影响其他指数。"""
    pools: dict = {}
    for code in (index_codes or list(INDEX_MAP.keys())):
        pools[code] = get_index_universe(code)
    return pools
