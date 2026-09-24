"""
alt_factors.py — 另类数据因子（V10 数据榨干：把存着没用的数据变成因子）

数据源（全部本地 data_warehouse，零网络）：
  - esg_rating.parquet   : ESG 评分 6250 只 → esg_score（横截面因子）
  - zlkp_jgcyd.parquet   : 机构参与度日序列 → inst_participation（市场级因子）
  - fund_hold.parquet    : 基金持仓 → fund_hold_ratio（横截面因子）
  - a_below_net.parquet  : 破净股数 → below_net_count（市场级因子）

注册机制：@register_factor 装饰器，registry.autodiscover() 自动发现。
这些因子是"截面型/市场型"，输入 data dict（含 alt 域），
由 ic_vectorized 的 alt 面板构建函数喂数据。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.zoo import register_factor

log = get_logger("qv6.alt_factors")

WAREHOUSE = Path(__file__).resolve().parent.parent.parent / "data_warehouse" / "market"


# ── ESG 因子 ──────────────────────────────────────────────
def _load_esg() -> pd.DataFrame:
    """读 ESG 评分 parquet → {code: score}。"""
    p = WAREHOUSE / "esg_rating.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    code_col = next((c for c in df.columns if "代码" in str(c)), None)
    score_col = next((c for c in df.columns if "ESG" in str(c).upper() and "评分" in str(c)), None)
    if code_col is None or score_col is None:
        return pd.DataFrame()
    out = df[[code_col, score_col]].copy()
    out.columns = ["code", "esg"]
    out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["esg"] = pd.to_numeric(out["esg"], errors="coerce")
    return out.dropna()


@register_factor("esg_score", direction=1, active=False,
                 description="ESG评分（横截面快照，非时间序列，仅用于横截面分析）")
def esg_score(data: dict, **kw) -> pd.Series:
    """横截面 ESG 评分（t 日横截面，全市场）。
    D4收敛登记: 独特因子保留
    """
    alt = data.get("alt") or {}
    esg = alt.get("esg")
    if esg is None or len(esg) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(esg)
    s.index = [str(x).replace(".0", "") for x in s.index]
    # 标准化（Z-score）
    mu, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(dtype=float)
    return ((s - mu) / sd).dropna()


# ── 机构参与度因子 ────────────────────────────────────────
def _load_zlkp() -> pd.DataFrame:
    """读机构参与度 parquet → date × value。"""
    p = WAREHOUSE / "zlkp_jgcyd.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    dcol = next((c for c in df.columns if "交易日" in str(c)), None)
    vcol = next((c for c in df.columns if "参与度" in str(c)), None)
    if dcol is None or vcol is None:
        return pd.DataFrame()
    out = df[[dcol, vcol]].copy()
    out.columns = ["date", "val"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    return out.dropna().set_index("date").sort_index()


@register_factor("inst_participation", direction=1,
                 description="机构参与度（市场级：机构活跃→方向+1）")
def inst_participation(data: dict, **kw) -> pd.Series:
    """市场级机构参与度（日序列，全市场同一值）。
    D4收敛登记: 独特因子保留
    """
    alt = data.get("alt") or {}
    zlkp = alt.get("zlkp")
    if zlkp is None or len(zlkp) == 0:
        return pd.Series(dtype=float)
    s = zlkp["val"].copy()
    # 5 日均值平滑
    sm = s.rolling(5, min_periods=2).mean()
    return sm.dropna()


# ── 基金持仓因子 ──────────────────────────────────────────
def _load_fund_hold() -> pd.DataFrame:
    """读基金持仓 parquet → {code: 持有基金家数}。"""
    p = WAREHOUSE / "fund_portfolio_hold.parquet"
    if not p.exists():
        cands = sorted(WAREHOUSE.glob("fund*hold*.parquet"))
        p = cands[0] if cands else p
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    code_col = next((c for c in df.columns if "股票代码" in str(c) or "代码" in str(c)), None)
    ratio_col = next((c for c in df.columns if "持有基金家数" in str(c) or "家数" in str(c)), None)
    if code_col is None or ratio_col is None:
        return pd.DataFrame()
    out = df[[code_col, ratio_col]].copy()
    out.columns = ["code", "ratio"]
    out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    return out.dropna()


@register_factor("fund_hold_ratio", direction=1, active=False,
                 description="持有基金家数（横截面快照，非时间序列，仅用于横截面分析）")
def fund_hold_ratio(data: dict, **kw) -> pd.Series:
    """横截面基金持仓比例（Z-score 标准化）。
    D4收敛登记: 独特因子保留
    """
    alt = data.get("alt") or {}
    fh = alt.get("fund_hold")
    if fh is None or len(fh) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(fh)
    s.index = [str(x).replace(".0", "") for x in s.index]
    mu, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(dtype=float)
    return ((s - mu) / sd).dropna()


# ── 破净股占比因子（市场级，保留但不进 IC 面板，供日报使用）──
@register_factor("below_net_ratio", direction=1, active=False,
                 description="破净股占比（市场级：破净多→底部区域，方向+1）")
def below_net_ratio(data: dict, **kw) -> pd.Series:
    """市场破净股占比（日序列）。
    D4收敛登记: 独特因子保留
    """
    alt = data.get("alt") or {}
    bn = alt.get("below_net")
    if bn is None or len(bn) == 0:
        return pd.Series(dtype=float)
    s = bn.copy()
    return s.dropna()
