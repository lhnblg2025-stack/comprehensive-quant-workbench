"""
processor.py — QuantV6 因子预处理流水线
去极值(MAD)/标准化(zscore)/市值中性化/行业中性化/相关性去重/缺失策略。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 预处理流水线独特保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.math_utils import mad_winsorize, zscore


def mad_clean(df: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """MAD 去极值（逐列）。"""
    return df.apply(lambda s: mad_winsorize(s, n) if s.notna().sum() > 5 else s)


def standardize(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """z-score 标准化 + clip。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    审计 2026-08-16：若输入为含日期的时序面板（MultiIndex 或 date 列），
    全样本 mean/std 会引入未来信息——显式告警，由调用方保证传入单日截面。
    """
    if isinstance(df.index, pd.MultiIndex) and any(
        n in ("date", "datetime", "trade_date") for n in df.index.names
    ):
        import logging as _log
        _log.getLogger("quant_ic_factors").warning(
            "standardize: 输入为含日期的时序面板(MultiIndex level=%s)，全样本标准化会产生前视泄漏；请按单日截面逐日调用",
            [n for n in df.index.names if n in ("date", "datetime", "trade_date")],
        )
    elif "date" in df.columns:
        import logging as _log
        _log.getLogger("quant_ic_factors").warning(
            "standardize: 输入含 date 列，疑似多日期面板，全样本标准化会产生前视泄漏；请按单日截面调用"
        )
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].astype(float)
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            out[col] = 0.0
        else:
            out[col] = ((s - s.mean()) / std).clip(-clip, clip)
    return out


def neutralize(df: pd.DataFrame, exposures: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    中性化：对行业哑变量回归取残差（简化版：分组去均值）。
    exposures: {code: {industry: str, market_cap: float}} 或 None。
    返回中性化后的因子矩阵（与原索引一致）。
    D4收敛登记: 跨模块同名异签名-不强迁保留
    """
    if exposures is None:
        return df
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    ind_map = {}
    cap_map = {}
    for code, meta in exposures.items():
        if isinstance(meta, dict):
            ind_map[code] = meta.get("industry", "")
            cap_map[code] = meta.get("market_cap", np.nan)
    ind_s = pd.Series(ind_map)
    cap_s = pd.Series(cap_map).astype(float)

    for col in df.columns:
        s = df[col]
        # 行业中性化：组内去均值
        if len(ind_s):
            grp_mean = s.groupby(ind_s.reindex(s.index)).transform("mean")
            s = s - grp_mean
        # 市值中性化：对市值对数回归取残差（简化：rank 残差）
        cap = cap_s.reindex(s.index)
        valid = cap.notna() & s.notna()
        if valid.sum() > 20:
            x = np.log1p(cap[valid]).values
            y = s[valid].values
            beta = np.polyfit(x, y, 1)
            resid = y - np.polyval(beta, x)
            s.loc[valid] = resid
        out[col] = s
    return out


def dedup_correlated(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """相关性去重：>threshold 的列保留先出现的。"""
    if df.shape[1] <= 2:
        return df
    corr = df.corr().abs()
    drop_cols = set()
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[j] not in drop_cols and not pd.isna(corr.iloc[i, j]) and corr.iloc[i, j] > threshold:
                drop_cols.add(cols[j])
    return df.drop(columns=list(drop_cols))


def fill_missing(df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """缺失填充：median/zero/ffill。"""
    if method == "zero":
        return df.fillna(0.0)
    if method == "ffill":
        return df.ffill().fillna(0.0)
    return df.fillna(df.median()).fillna(0.0)


def process_factor_frame(df: pd.DataFrame, exposures: pd.DataFrame | None = None,
                         dedup: bool = True) -> pd.DataFrame:
    """
    完整预处理流水线：去极值 → 中性化 → 标准化 → 去重 → 填充。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = mad_clean(df)
    df = neutralize(df, exposures)
    df = standardize(df)
    if dedup:
        df = dedup_correlated(df)
    df = fill_missing(df)
    return df


# ── 涨跌停 / 停牌状态掩码 ─────────────────────────────────
# 封板股/停牌股的因子值（缩量企稳、收益≈0 等）会被误读为有效信号，
# 污染截面。下列函数按日粒度给出状态掩码，配合 apply_state_mask 将
# 对应因子值置 NaN（不参与截面）。


def _pct_chg(df: pd.DataFrame) -> pd.Series:
    """单只股票日涨跌幅（%）：优先取现成涨跌幅列，否则由 close 计算。"""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    for col in ("pct_chg", "涨跌幅"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    if "close" in df.columns:
        c = pd.to_numeric(df["close"], errors="coerce")
        return c.pct_change() * 100.0
    return pd.Series(np.nan, index=df.index)


def _board_limit_pct(code: str | None) -> float:
    """按代码前缀判断板块涨跌幅阈值（%，近似）：主板10 / 创业板科创板20 / 北交所30。"""
    if code:
        if code.startswith(("688", "689", "30")):
            return 19.9
        if code.startswith(("4", "8", "92")):
            return 29.9
    return 9.9


def _limit_threshold_pct(df: pd.DataFrame, code: str | None = None) -> float:
    """ST（±4.9%）优先于板块阈值。数据含 name 列或 isST 标志时识别 ST。"""
    if df is not None:
        if "name" in df.columns and df["name"].astype(str).str.contains("ST", case=False, na=False).any():
            return 4.9
        if "isST" in df.columns:
            flag = pd.to_numeric(df["isST"], errors="coerce")
            if (flag == 1).any():
                return 4.9
    return _board_limit_pct(code)


def limit_up_mask(df: pd.DataFrame, code: str | None = None) -> pd.Series:
    """涨停日掩码（布尔 Series，index 与 df 对齐）。"""
    pct = _pct_chg(df)
    return pct >= _limit_threshold_pct(df, code)


def limit_down_mask(df: pd.DataFrame, code: str | None = None) -> pd.Series:
    """跌停日掩码（布尔 Series，index 与 df 对齐）。"""
    pct = _pct_chg(df)
    return pct <= -_limit_threshold_pct(df, code)


def suspension_mask(df: pd.DataFrame) -> pd.Series:
    """停牌日掩码：成交量为 0 或 NaN 视为停牌（布尔 Series）。"""
    if df is None or len(df) == 0:
        return pd.Series(dtype=bool)
    if "volume" not in df.columns:
        return pd.Series(False, index=df.index)
    v = pd.to_numeric(df["volume"], errors="coerce")
    return v.isna() | (v == 0)


def state_mask(df: pd.DataFrame, code: str | None = None) -> pd.Series:
    """综合状态掩码：涨停/跌停/停牌任一成立 → True。"""
    return (limit_up_mask(df, code) | limit_down_mask(df, code)
            | suspension_mask(df))


def apply_state_mask(factor_df, mask) -> pd.DataFrame | pd.Series:
    """停牌/涨跌停状态日因子值置 NaN（不参与截面）。

    mask 两种形态：
    - pd.Series：与 factor_df 按 index 对齐（时间序列逐日掩码）；
    - dict {code: bool | Series}：跨截面掩码，取每只股票最新交易日的状态
      （Series 时取最后一个有效值），True 的股票整行置 NaN。
    """
    if factor_df is None or len(factor_df) == 0:
        return factor_df
    out = factor_df.copy()
    if isinstance(mask, pd.Series):
        bad = mask.reindex(out.index).fillna(False).astype(bool)
        out.loc[bad] = np.nan
    elif isinstance(mask, dict):
        for code, m in mask.items():
            if code not in out.index:
                continue
            flag = m
            if isinstance(m, pd.Series):
                m = m.dropna()
                flag = bool(m.iloc[-1]) if len(m) else False
            if flag:
                out.loc[code] = np.nan
    return out
