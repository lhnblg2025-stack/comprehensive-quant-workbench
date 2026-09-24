"""
quant_system.market_factors_v7 — 市场级衍生品/另类因子（V3 融合层）

来源：quant_v6/factors/derivatives_v7.py + alternative_v7.py（quant_v6 退役后融合）
设计：纯函数式（无 quant_v6 依赖），数据走 DataStore 仓库（futures_basis/shibor/
      repo_rate/bond_futures/qvix/cb_spot），输出市场级指标 {"MARKET": value}。

用途：regime 判断 / 择时 / 市场健康监控（非横截面个股因子，不污染因子池）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.data_store import get_store


def _mk(value) -> pd.Series:
    """市场级单值 → 横截面 Series。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return pd.Series(dtype=float)
    return pd.Series({"MARKET": float(value)})


def _load(dataset: str) -> pd.DataFrame:
    """从 DataStore 仓库读取市场数据集（失败返回空 DataFrame）。"""
    try:
        return get_store().get_dataset(dataset)
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# 1. 股指期货基差（IF/IC/IM 主力合约）
# ══════════════════════════════════════════════════════════

def if_basis_rate() -> pd.Series:
    """沪深300股指期货主力基差率（升水>0乐观 / 贴水=恐慌）。"""
    df = _load("futures")
    if df.empty or "contract" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[df["contract"] == "IF"]
    if sub.empty or "basis_rate" not in sub.columns:
        return pd.Series(dtype=float)
    return _mk(sub["basis_rate"].iloc[-1])


def ic_im_basis_diff() -> pd.Series:
    """IC 与 IM 基差之差（小盘风格资金预期）。"""
    df = _load("futures")
    if df.empty or "contract" not in df.columns:
        return pd.Series(dtype=float)
    ic = df[df["contract"] == "IC"]
    im = df[df["contract"] == "IM"]
    if ic.empty or im.empty or "basis_rate" not in df.columns:
        return pd.Series(dtype=float)
    return _mk(ic["basis_rate"].iloc[-1] - im["basis_rate"].iloc[-1])


def if_basis_5d() -> pd.Series:
    """IF 基差率 5 日变化（贴水加深=情绪恶化）。"""
    df = _load("futures")
    if df.empty or "contract" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[df["contract"] == "IF"]
    if len(sub) < 6 or "basis_rate" not in sub.columns:
        return pd.Series(dtype=float)
    return _mk(sub["basis_rate"].iloc[-1] - sub["basis_rate"].iloc[-6])


# ══════════════════════════════════════════════════════════
# 2. 国债期货（利率预期）
# ══════════════════════════════════════════════════════════

def t_futures_ret20() -> pd.Series:
    """国债期货主力 20 日动量（利率下行利好债券）。"""
    df = _load("bond_futures")
    if df.empty or len(df) < 21 or "收盘价" not in df.columns:
        return pd.Series(dtype=float)
    px = df["收盘价"].astype(float)
    return _mk(px.iloc[-1] / px.iloc[-21] - 1)


def t_futures_ret5() -> pd.Series:
    """国债期货主力 5 日动量。"""
    df = _load("bond_futures")
    if df.empty or len(df) < 6 or "收盘价" not in df.columns:
        return pd.Series(dtype=float)
    px = df["收盘价"].astype(float)
    return _mk(px.iloc[-1] / px.iloc[-6] - 1)


# ══════════════════════════════════════════════════════════
# 3. 期权波动率 QVIX
# ══════════════════════════════════════════════════════════

def qvix50_level() -> pd.Series:
    """50ETF 期权中国波指 QVIX 水平（高=恐慌）。"""
    df = _load("qvix")
    if df.empty or "symbol" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "50etf"]
    if sub.empty or "close" not in sub.columns:
        return pd.Series(dtype=float)
    return _mk(sub["close"].astype(float).iloc[-1])


def qvix50_change5() -> pd.Series:
    """QVIX 5 日变化（上升=恐慌加剧）。"""
    df = _load("qvix")
    if df.empty or "symbol" not in df.columns:
        return pd.Series(dtype=float)
    sub = df[df["symbol"] == "50etf"]
    if len(sub) < 6 or "close" not in sub.columns:
        return pd.Series(dtype=float)
    v = sub["close"].astype(float)
    return _mk(v.iloc[-1] - v.iloc[-6])


def qvix300_minus_50() -> pd.Series:
    """300ETF 与 50ETF QVIX 之差（大盘股相对紧张度）。"""
    df = _load("qvix")
    if df.empty or "symbol" not in df.columns:
        return pd.Series(dtype=float)
    d50 = df[df["symbol"] == "50etf"]
    d300 = df[df["symbol"] == "300etf"]
    if d50.empty or d300.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    return _mk(float(d300["close"].iloc[-1]) - float(d50["close"].iloc[-1]))


# ══════════════════════════════════════════════════════════
# 4. 可转债（风险偏好）
# ══════════════════════════════════════════════════════════

def cb_double_low_median() -> pd.Series:
    """全市场转债双低中位数（低=市场便宜）。"""
    df = _load("cb_spot")
    if df.empty or "double_low" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["double_low"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.median())


def cb_premium_avg() -> pd.Series:
    """转债平均转股溢价率（高=投机情绪热）。"""
    df = _load("cb_spot")
    if df.empty or "premium_rt" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["premium_rt"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.mean())


def cb_avg_price() -> pd.Series:
    """转债平均价格（100 以下占比高=熊市底部区）。"""
    df = _load("cb_spot")
    if df.empty or "trade" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["trade"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.mean())


# ══════════════════════════════════════════════════════════
# 5. 资金面（回购/Shibor）
# ══════════════════════════════════════════════════════════

def dr007_level() -> pd.Series:
    """银行间 7 天回购利率水平（高=流动性紧）。"""
    df = _load("repo_rate")
    if df.empty or "FR007" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["FR007"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


def dr007_change5() -> pd.Series:
    """FR007 5 日变化（上升=资金面收紧）。"""
    df = _load("repo_rate")
    if df.empty or "FR007" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["FR007"], errors="coerce").dropna()
    if len(v) < 6:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] - v.iloc[-6])


def shibor3m_level() -> pd.Series:
    """Shibor 3M 水平（中长期资金价格）。"""
    df = _load("shibor")
    if df.empty or "rate" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["rate"], errors="coerce").dropna()
    if v.empty:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1])


def shibor3m_change10() -> pd.Series:
    """Shibor 3M 10 日变化。"""
    df = _load("shibor")
    if df.empty or "rate" not in df.columns:
        return pd.Series(dtype=float)
    v = pd.to_numeric(df["rate"], errors="coerce").dropna()
    if len(v) < 11:
        return pd.Series(dtype=float)
    return _mk(v.iloc[-1] - v.iloc[-11])


# ══════════════════════════════════════════════════════════
# 6. 另类域（票房——猫眼实时票房）
# ══════════════════════════════════════════════════════════

def boxoffice_total() -> pd.Series:
    """实时票房总额（节假日/档期热度）。

    V3 审计修复: 原实现读 market_fund_flow（资金流数据，无票房列）恒返回空
    → 改用 movie_boxoffice_realtime 直连；失败返回空。
    """
    try:
        import akshare as ak
        df = ak.movie_boxoffice_realtime()
        if df is None or df.empty:
            return pd.Series(dtype=float)
        col = [c for c in df.columns if "综合票房" in str(c) or "实时票房" in str(c)]
        if not col:
            return pd.Series(dtype=float)
        v = pd.to_numeric(df[col[0]], errors="coerce").dropna()
        if v.empty:
            return pd.Series(dtype=float)
        return _mk(v.sum())
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════
# 统一入口
# ══════════════════════════════════════════════════════════

#: 全部市场级因子: 名称 → 计算函数（供 regime/择时/健康监控统一调用）
MARKET_FACTORS: dict[str, callable] = {
    "if_basis_rate": if_basis_rate,
    "ic_im_basis_diff": ic_im_basis_diff,
    "if_basis_5d": if_basis_5d,
    "t_futures_ret20": t_futures_ret20,
    "t_futures_ret5": t_futures_ret5,
    "qvix50_level": qvix50_level,
    "qvix50_change5": qvix50_change5,
    "qvix300_minus_50": qvix300_minus_50,
    "cb_double_low_median": cb_double_low_median,
    "cb_premium_avg": cb_premium_avg,
    "cb_avg_price": cb_avg_price,
    "dr007_level": dr007_level,
    "dr007_change5": dr007_change5,
    "shibor3m_level": shibor3m_level,
    "shibor3m_change10": shibor3m_change10,
    "boxoffice_total": boxoffice_total,
}


def compute_market_factors() -> dict[str, float]:
    """计算全部市场级因子，返回 {因子名: 值}（缺失值丢弃）。"""
    import logging
    log = logging.getLogger("quant_system.market_factors_v7")
    out: dict[str, float] = {}
    for name, fn in MARKET_FACTORS.items():
        try:
            s = fn()
            if not s.empty:
                out[name] = float(s.iloc[0])
            else:
                log.debug("市场因子 %s: 无数据（数据集缺失或列不匹配）", name)
        except Exception as exc:
            log.debug("市场因子 %s 计算异常: %s", name, exc)
            continue
    return out


def market_factor_snapshot() -> dict:
    """市场因子快照（含方向注释，供报告/接口输出）。"""
    vals = compute_market_factors()
    return {"date": pd.Timestamp.now().strftime("%Y-%m-%d"), "factors": vals,
            "count": len(vals), "note": "市场级衍生品/另类因子（regime/择时用）"}


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    snap = market_factor_snapshot()
    print("=== 市场级因子快照 ===")
    print(f"日期: {snap['date']} | 因子数: {snap['count']}")
    for k, v in sorted(snap["factors"].items()):
        print(f"  {k}: {v}")
