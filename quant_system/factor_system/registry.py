"""
factor_system/registry.py — 因子注册表
V4.1 深度改造

管理所有因子的定义、分类、元数据。
V4.1 feature: 因子注册表
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。
"""

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd


class FactorCategory(Enum):
    """因子大类，用于组合、监控和流水线调度。"""

    MOMENTUM = "momentum"        # 动量因子
    VALUE = "value"              # 价值因子
    QUALITY = "quality"          # 质量因子
    GROWTH = "growth"            # 增长因子
    VOLATILITY = "volatility"    # 波动率因子
    LIQUIDITY = "liquidity"      # 流动性因子
    SIZE = "size"                # 规模因子
    TECHNICAL = "technical"      # 技术因子
    SENTIMENT = "sentiment"      # 情绪因子
    MACRO = "macro"              # 宏观因子
    COMPOSITE = "composite"      # 合成因子


class FactorHorizon(Enum):
    """因子预期有效周期。"""

    INTRADAY = "intraday"        # 日内
    SHORT = "short"              # 1-5天
    MEDIUM = "medium"            # 5-20天
    LONG = "long"                # 20-60天
    MACRO = "macro"              # >60天


@dataclass
class FactorDef:
    """因子定义。"""

    name: str                        # 因子名
    category: FactorCategory         # 类别
    horizon: FactorHorizon           # 有效周期
    formula: str                     # 计算公式说明
    description: str                 # 描述
    computation_fn: Optional[Callable[..., pd.Series]] = None  # 计算函数
    requires: list[str] = field(default_factory=list)  # 依赖的原始数据
    parameters: dict = field(default_factory=dict)  # 参数默认值
    is_active: bool = True           # 是否活跃
    version: str = "1.0"             # 版本
    min_stocks: int = 10              # 截面最少股票数（V6 fix: 100→10，小股票池/测试场景可用）


class FactorRegistry:
    """因子注册表——管理所有因子的生命周期。"""

    _instance: Optional["FactorRegistry"] = None

    def __new__(cls) -> "FactorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}
            cls._instance._categories = {}
            cls._instance._defaults_registered = False
        return cls._instance

    def register(self, factor: FactorDef) -> None:
        """注册一个因子；同名注册会覆盖定义并维护分类索引。"""
        old = self._factors.get(factor.name)
        if old is not None and old.category in self._categories:
            self._categories[old.category] = [n for n in self._categories[old.category] if n != factor.name]

        self._factors[factor.name] = factor
        if factor.category not in self._categories:
            self._categories[factor.category] = []
        if factor.name not in self._categories[factor.category]:
            self._categories[factor.category].append(factor.name)

    def unregister(self, name: str) -> None:
        """注销一个因子，通常用于实验因子下线。"""
        factor = self._factors.pop(name, None)
        if factor is not None and factor.category in self._categories:
            self._categories[factor.category] = [n for n in self._categories[factor.category] if n != name]

    def clear(self) -> None:
        """清空注册表，主要用于测试或重新加载默认因子。"""
        self._factors.clear()
        self._categories.clear()
        self._defaults_registered = False

    def get(self, name: str) -> Optional[FactorDef]:
        """按名称获取因子定义。"""
        return self._factors.get(name)

    def exists(self, name: str) -> bool:
        """检查因子是否已注册。"""
        return name in self._factors

    def list_by_category(self, category: FactorCategory) -> list[FactorDef]:
        """列出某一分类下的因子。"""
        return [self._factors[n] for n in self._categories.get(category, []) if n in self._factors]

    def list_all(self) -> list[FactorDef]:
        """列出全部因子。"""
        return list(self._factors.values())

    def list_active(self) -> list[FactorDef]:
        """列出当前活跃因子。"""
        return [f for f in self._factors.values() if f.is_active]

    def list_names(self, active_only: bool = False) -> list[str]:
        """列出因子名。"""
        factors = self.list_active() if active_only else self.list_all()
        return [f.name for f in factors]

    def categories(self) -> list[FactorCategory]:
        """列出已覆盖分类。"""
        return list(self._categories.keys())

    def to_frame(self) -> pd.DataFrame:
        """导出注册表元数据，便于审计和报表展示。"""
        rows = []
        for factor in self.list_all():
            rows.append({
                "name": factor.name,
                "category": factor.category.value,
                "horizon": factor.horizon.value,
                "formula": factor.formula,
                "description": factor.description,
                "requires": ",".join(factor.requires),
                "parameters": factor.parameters,
                "is_active": factor.is_active,
                "version": factor.version,
                "min_stocks": factor.min_stocks,
            })
        return pd.DataFrame(rows)


def _fd(
    name: str,
    category: FactorCategory,
    horizon: FactorHorizon,
    formula: str,
    description: str,
    requires: list[str],
    parameters: Optional[dict] = None,
    version: str = "1.0",
    min_stocks: int = 10,
) -> FactorDef:
    """创建因子定义的小工具，减少默认注册表样板代码。

    V6 fix: min_stocks 默认 100→10（Q5 HIGH: registry.py:58,155），
    避免 <100 行输入时全部因子返回 NaN。
    """
    return FactorDef(
        name=name,
        category=category,
        horizon=horizon,
        formula=formula,
        description=description,
        requires=requires,
        parameters=parameters or {},
        version=version,
        min_stocks=min_stocks,
    )


# ===========================================================================
# V6 fix (Q5 CRITICAL: registry.py:53,180-250): 因子真实计算函数库
#   V5.4 注册的 74 个因子 computation_fn 全部为 None → engine.compute 必然抛
#   TypeError: 'NoneType' object is not callable；compute_many 静默返回 NaN 列。
#   V6 为每个因子绑定真实实现，支持两种输入布局：
#     - 面板/时序布局：rows=日期（DatetimeIndex 或 date 列），按滚动窗口计算，
#       返回与输入行对齐的逐期因子值（前 N 行为暖机 NaN，符合惯例）；
#     - 截面布局：rows=股票，从已有特征列（close/market_cap/returns/...）按行计算。
#   所有实现均防御式：必需列缺失 → 返回 NaN（降级可见），绝不抛异常。
# ===========================================================================

def _nan_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index)


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    return df[name] if name in df.columns else None


def _fcol(df: pd.DataFrame, name: str) -> pd.Series | None:
    c = _col(df, name)
    return c.astype(float) if c is not None else None


def _xs_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """截面布局下查找预计算列（如 returns_126d / momentum_6m）。"""
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)
    return None


def _is_timeseries(df: pd.DataFrame) -> bool:
    """True = 行是按时间排序的单一标的时序（DatetimeIndex 或 date 列）。"""
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    if "date" in df.columns and len(df) > 1 and df["date"].nunique() == len(df):
        return True
    return False


def _returns_series(df: pd.DataFrame) -> pd.Series | None:
    r = _fcol(df, "returns")
    if r is not None:
        return r
    c = _fcol(df, "close")
    if c is not None:
        return c.pct_change()
    return None


def _rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
    """滚动窗口计算；min_periods 取 max(2, window//2) 以容忍小样本。"""
    mp = max(2, window // 2)
    if fn == "mean":
        return series.rolling(window, min_periods=mp).mean()
    if fn == "std":
        return series.rolling(window, min_periods=mp).std()
    if fn == "sum":
        return series.rolling(window, min_periods=mp).sum()
    if fn == "skew":
        return series.rolling(window, min_periods=mp).skew()
    if fn == "kurt":
        return series.rolling(window, min_periods=mp).kurt()
    if fn == "prod":
        return (1.0 + series.fillna(0.0)).rolling(window, min_periods=mp).apply(np.prod, raw=True) - 1.0
    if fn == "min":
        return series.rolling(window, min_periods=mp).min()
    if fn == "max":
        return series.rolling(window, min_periods=mp).max()
    raise ValueError(f"未知滚动算子: {fn}")


def _z(s: pd.Series) -> pd.Series:
    """时间序列 z-score：expanding 均值/标准差（只用当前及过去，杜绝前视）。

    W2.5 P2 修复：原全样本 (s-mean)/std 含未来分布，合成因子接回测会前视。
    """
    s = s.astype(float)
    mean = s.expanding(min_periods=20).mean()
    std = s.expanding(min_periods=20).std()
    z = (s - mean) / std
    return z.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)


# ── 动量 ────────────────────────────────────────────────────────────────
def _f_momentum_12m1m(df: pd.DataFrame, total_days: int = 252, skip_days: int = 21) -> pd.Series:
    """12-1 动量：剔除最近 1 个月的 12 个月累计收益（Barra 标准）。
    D4收敛登记: 与factor_zoo同名异口径-不强迁保留
    """
    r = _returns_series(df)
    if r is None:
        return _nan_series(df)
    all_r = 1.0 + r.fillna(0.0)
    prod_all = all_r.rolling(total_days + skip_days, min_periods=2).apply(np.prod, raw=True)
    prod_skip = all_r.rolling(skip_days, min_periods=2).apply(np.prod, raw=True)
    return prod_all / prod_skip - 1.0


def _f_momentum_days(df: pd.DataFrame, days: int = 126, alias: str = "momentum_6m") -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    r = _returns_series(df)
    if r is None:
        c = _xs_col(df, [f"returns_{days}d", alias])
        return c if c is not None else _nan_series(df)
    return _rolling(r, days, "prod")


def _f_short_term_reversal(df: pd.DataFrame, days: int = 5) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    if r is None:
        c = _xs_col(df, [f"returns_{days}d"])
        return c if c is not None else _nan_series(df)
    return -_rolling(r, days, "prod")


def _f_industry_momentum(df: pd.DataFrame, days: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    if r is None:
        return _nan_series(df)
    ret20 = _rolling(r, days, "sum")
    ind = _col(df, "industry_code")
    if ind is not None:
        return ret20.groupby(ind).transform("mean")
    return ret20


def _f_residual_momentum(df: pd.DataFrame, days: int = 60) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    m = _fcol(df, "market_returns")
    if r is None or m is None:
        return _nan_series(df)
    mp = max(2, days // 2)
    beta = r.rolling(days, min_periods=mp).cov(m) / m.rolling(days, min_periods=mp).var().replace(0.0, np.nan)
    resid = (r - beta * m).fillna(0.0)
    return resid.rolling(days, min_periods=2).sum()


# ── 价值 / 质量（行内公式） ────────────────────────────────────────────
def _f_ratio(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    a, b = _fcol(df, num), _fcol(df, den)
    if a is None or b is None:
        return _nan_series(df)
    return a / b.replace(0.0, np.nan)


def _f_book_to_price(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "book_value", "market_cap")
def _f_earnings_to_price(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "earnings", "market_cap")
def _f_cash_flow_to_price(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "operating_cf", "market_cap")
def _f_sales_to_price(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "revenue", "market_cap")
def _f_dividend_yield(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "dividend_per_share", "close")
def _f_fcf_yield(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "free_cash_flow", "market_cap")
def _f_pe_ttm(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "close", "eps_ttm")
def _f_pb_lf(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "close", "book_value_per_share")
def _f_ev_to_ebitda(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "enterprise_value", "ebitda")
def _f_roe_ttm(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "net_profit_ttm", "equity")
def _f_roa_ttm(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "net_profit_ttm", "total_assets")
def _f_net_margin_ttm(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "net_profit_ttm", "revenue")
def _f_debt_to_equity(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "total_liabilities", "equity")
def _f_current_ratio(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "current_assets", "current_liabilities")
def _f_asset_turnover(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "revenue", "total_assets")
def _f_interest_coverage(df):
    """D4收敛登记: 独特因子保留"""
    return _f_ratio(df, "ebit", "interest_expense")


def _f_gross_margin_ttm(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    rev, cost = _fcol(df, "revenue"), _fcol(df, "cost")
    if rev is None or cost is None:
        return _nan_series(df)
    return (rev - cost) / rev.replace(0.0, np.nan)


def _f_accruals_ratio(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    np_, ocf, ta = _fcol(df, "net_profit"), _fcol(df, "operating_cf"), _fcol(df, "total_assets")
    if np_ is None or ocf is None or ta is None:
        return _nan_series(df)
    return (np_ - ocf) / ta.replace(0.0, np.nan)


def _f_earnings_stability(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    roe = _fcol(df, "roe")
    if roe is None:
        return _nan_series(df)
    return -roe.rolling(window, min_periods=max(2, window // 2)).std()


# ── 增长 ────────────────────────────────────────────────────────────────
def _f_shift_growth(df: pd.DataFrame, col: str, periods: int = 4) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    s = _fcol(df, col)
    if s is None or len(df) < periods + 1:
        return _nan_series(df)
    return s / s.shift(periods) - 1.0


def _f_revenue_growth_yoy(df):
    """D4收敛登记: 独特因子保留"""
    return _f_shift_growth(df, "revenue", 4)
def _f_profit_growth_yoy(df):
    """D4收敛登记: 独特因子保留"""
    return _f_shift_growth(df, "net_profit", 4)
def _f_roe_growth_yoy(df):
    """D4收敛登记: 独特因子保留"""
    return _f_shift_growth(df, "roe", 4)
def _f_eps_revision_20d(df, window=20):
    """D4收敛登记: 独特因子保留"""
    return _f_shift_growth(df, "expected_eps", window)


def _f_earnings_surprise(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    a, e, c = _fcol(df, "actual_eps"), _fcol(df, "expected_eps"), _fcol(df, "close")
    if a is None or e is None or c is None:
        return _nan_series(df)
    return (a - e) / c.replace(0.0, np.nan)


def _f_revenue_surprise(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    a, e = _fcol(df, "actual_revenue"), _fcol(df, "expected_revenue")
    if a is None or e is None:
        return _nan_series(df)
    return (a - e) / e.replace(0.0, np.nan)


# ── 波动率 ──────────────────────────────────────────────────────────────
def _f_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    if r is None:
        c = _xs_col(df, [f"vol_{window}d"])
        return c if c is not None else _nan_series(df)
    return _rolling(r, window, "std")


def _f_beta(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    r, m = _returns_series(df), _fcol(df, "market_returns")
    if r is None or m is None:
        return _nan_series(df)
    mp = max(2, window // 2)
    cov = r.rolling(window, min_periods=mp).cov(m)
    var = m.rolling(window, min_periods=mp).var().replace(0.0, np.nan)
    return cov / var


def _f_max_drawdown(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    c = _fcol(df, "close")
    if c is None:
        return _nan_series(df)
    mp = max(2, window // 2)
    dd = c / c.rolling(window, min_periods=mp).max() - 1.0
    return dd.rolling(window, min_periods=mp).min()


def _f_downside_risk(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    r = _returns_series(df)
    if r is None:
        return _nan_series(df)
    return _rolling(r.clip(upper=0.0), window, "std")


def _f_idio_vol(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    r, m = _returns_series(df), _fcol(df, "market_returns")
    if r is None or m is None:
        return _nan_series(df)
    mp = max(2, window // 2)
    beta = r.rolling(window, min_periods=mp).cov(m) / m.rolling(window, min_periods=mp).var().replace(0.0, np.nan)
    resid = (r - beta * m).fillna(0.0)
    return resid.rolling(window, min_periods=mp).std()


def _f_skewness(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    if r is None:
        return _nan_series(df)
    return _rolling(r, window, "skew")


def _f_kurtosis(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r = _returns_series(df)
    if r is None:
        return _nan_series(df)
    return _rolling(r, window, "kurt")


# ── 流动性 ──────────────────────────────────────────────────────────────
def _f_turnover_avg(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    t = _fcol(df, "turnover")
    return _rolling(t, window, "mean") if t is not None else _nan_series(df)


def _f_turnover_std(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    t = _fcol(df, "turnover")
    return _rolling(t, window, "std") if t is not None else _nan_series(df)


def _f_amihud(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    r, amt = _returns_series(df), _fcol(df, "amount")
    if r is None or amt is None:
        return _nan_series(df)
    illiq = (r.abs() / amt.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return _rolling(illiq, window, "mean")


def _f_dollar_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    amt = _fcol(df, "amount")
    if amt is None:
        return _nan_series(df)
    return np.log(_rolling(amt, window, "mean").clip(lower=1.0))


def _f_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    v = _fcol(df, "volume")
    if v is None:
        return _nan_series(df)
    return v / _rolling(v, window, "mean").replace(0.0, np.nan)


def _f_zero_volume_days(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    v = _fcol(df, "volume")
    if v is None:
        return _nan_series(df)
    zero = (v == 0).astype(float)
    return _rolling(zero, window, "sum") / window


# ── 规模 ────────────────────────────────────────────────────────────────
def _f_ln_cap(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    mc = _fcol(df, "market_cap")
    if mc is None:
        c = _fcol(df, "ln_cap")
        return c if c is not None else _nan_series(df)
    return np.log(mc.clip(lower=1.0))


def _f_ln_float_cap(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    fmc = _fcol(df, "float_market_cap")
    if fmc is None:
        c = _fcol(df, "ln_float_cap")
        return c if c is not None else _nan_series(df)
    return np.log(fmc.clip(lower=1.0))


def _f_non_linear_size(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    base = _f_ln_cap(df)
    return base ** 3


# ── 技术 ────────────────────────────────────────────────────────────────
def _f_ma_pct(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    c = _fcol(df, "close")
    if c is None:
        return _nan_series(df)
    ma = c.rolling(window, min_periods=max(2, window // 2)).mean()
    return c / ma - 1.0


def _f_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    c = _fcol(df, "close")
    if c is None:
        return _nan_series(df)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    mp = max(2, window // 2)
    avg_gain = gain.rolling(window, min_periods=mp).mean()
    avg_loss = loss.rolling(window, min_periods=mp).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _f_macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    c = _fcol(df, "close")
    if c is None:
        return _nan_series(df)
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def _f_bollinger(df: pd.DataFrame, window: int = 20, n_std: int = 2) -> pd.Series:
    """D4收敛登记: 与factor_zoo同名异口径-不强迁保留"""
    c = _fcol(df, "close")
    if c is None:
        return _nan_series(df)
    mp = max(2, window // 2)
    ma = c.rolling(window, min_periods=mp).mean()
    sd = c.rolling(window, min_periods=mp).std()
    return (c - ma) / (2.0 * n_std * sd).replace(0.0, np.nan)


def _f_obv_ma_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    c, v = _fcol(df, "close"), _fcol(df, "volume")
    if c is None or v is None:
        return _nan_series(df)
    sign = np.sign(c.diff().fillna(0.0))
    obv = (sign * v).cumsum()
    ma = obv.rolling(window, min_periods=max(2, window // 2)).mean()
    return obv / ma.replace(0.0, np.nan)


def _f_vpt(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    c, v = _fcol(df, "close"), _fcol(df, "volume")
    if c is None or v is None:
        return _nan_series(df)
    return (c.pct_change().fillna(0.0) * v).cumsum()


def _f_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    hi, lo, c = _fcol(df, "high"), _fcol(df, "low"), _fcol(df, "close")
    if hi is None or lo is None or c is None:
        return _nan_series(df)
    mp = max(2, window // 2)
    hh = hi.rolling(window, min_periods=mp).max()
    ll = lo.rolling(window, min_periods=mp).min()
    return -100.0 * (hh - c) / (hh - ll).replace(0.0, np.nan)


def _f_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    hi, lo, c = _fcol(df, "high"), _fcol(df, "low"), _fcol(df, "close")
    if hi is None or lo is None or c is None:
        return _nan_series(df)
    pc = c.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return _rolling(tr, window, "mean")


# ── 情绪 / 宏观（行内公式） ────────────────────────────────────────────
def _f_margin_change(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    mb = _fcol(df, "margin_balance")
    if mb is None or len(df) < window + 1:
        return _nan_series(df)
    return mb / mb.shift(window) - 1.0


def _f_short_interest(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    sv, tv = _fcol(df, "short_volume"), _fcol(df, "total_volume")
    if sv is None or tv is None:
        return _nan_series(df)
    return sv / tv.replace(0.0, np.nan)


def _f_fund_flow(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    nf, fmc = _fcol(df, "net_fund_flow"), _fcol(df, "float_market_cap")
    if nf is None or fmc is None:
        return _nan_series(df)
    return _rolling(nf, window, "sum") / fmc.replace(0.0, np.nan)


def _f_northbound_flow(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    nb, fmc = _fcol(df, "northbound_net_buy"), _fcol(df, "float_market_cap")
    if nb is None or fmc is None:
        return _nan_series(df)
    return _rolling(nb, window, "sum") / fmc.replace(0.0, np.nan)


def _f_yield_diff(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    a, b = _fcol(df, "10y_yield"), _fcol(df, "2y_yield")
    if a is None or b is None:
        return _nan_series(df)
    return a - b


def _f_vix(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    v = _fcol(df, "volatility_index")
    return v if v is not None else _nan_series(df)


def _f_credit_spread(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    a, b = _fcol(df, "corp_bond_yield"), _fcol(df, "treasury_yield")
    if a is None or b is None:
        return _nan_series(df)
    return a - b


def _f_m2_growth(df: pd.DataFrame) -> pd.Series:
    """D4收敛登记: 独特因子保留"""
    m = _fcol(df, "m2_yoy")
    return m if m is not None else _nan_series(df)


# ── 合成因子 ────────────────────────────────────────────────────────────
def _f_composite(df: pd.DataFrame, components: list[str] | None = None,
                 weights: list[float] | None = None) -> pd.Series:
    """合成因子：对子因子 z-score 后加权求和（每行按可用子因子取均值，NaN 安全）。
    D4收敛登记: 独特因子保留
    """
    if not components:
        return _nan_series(df)
    reg = FactorRegistry()
    parts = []
    for i, comp in enumerate(components):
        fd = reg.get(comp)
        if fd is not None and fd.computation_fn is not None:
            impl_fn, impl_kw = fd.computation_fn, fd.parameters
        else:
            entry = _FACTOR_IMPLS.get(comp)
            if entry is None:
                continue
            impl_fn, impl_kw = entry
        try:
            s = impl_fn(df, **impl_kw)
        except Exception as e:
            logging.getLogger(__name__).error(f"[registry] 操作失败: {e}", exc_info=True)
            continue
        if s is None or s.isna().all():
            continue
        w = weights[i] if weights and i < len(weights) else 1.0
        parts.append(_z(s) * w)
    if not parts:
        return _nan_series(df)
    combo = pd.concat(parts, axis=1)
    return combo.sum(axis=1) / combo.notna().sum(axis=1).clip(lower=1.0)


# 因子名 → (实现函数, 默认参数)。engine 实际调用时以 FactorDef.parameters
# 为准，此表仅作兜底与合成因子内部调用。
_FACTOR_IMPLS: dict[str, tuple[Callable, dict]] = {
    # 动量
    "momentum_12m1m": (_f_momentum_12m1m, {"total_days": 252, "skip_days": 21}),
    "momentum_6m": (_f_momentum_days, {"days": 126, "alias": "momentum_6m"}),
    "momentum_3m": (_f_momentum_days, {"days": 63, "alias": "momentum_3m"}),
    "momentum_1m": (_f_momentum_days, {"days": 21, "alias": "momentum_1m"}),
    "short_term_reversal": (_f_short_term_reversal, {"days": 5}),
    "industry_momentum": (_f_industry_momentum, {"days": 20}),
    "residual_momentum_60d": (_f_residual_momentum, {"days": 60}),
    # 价值
    "book_to_price": (_f_book_to_price, {}),
    "earnings_to_price": (_f_earnings_to_price, {}),
    "cash_flow_to_price": (_f_cash_flow_to_price, {}),
    "sales_to_price": (_f_sales_to_price, {}),
    "dividend_yield": (_f_dividend_yield, {}),
    "fcf_yield": (_f_fcf_yield, {}),
    "pe_ttm": (_f_pe_ttm, {}),
    "pb_lf": (_f_pb_lf, {}),
    "ev_to_ebitda": (_f_ev_to_ebitda, {}),
    # 质量
    "roe_ttm": (_f_roe_ttm, {}),
    "roa_ttm": (_f_roa_ttm, {}),
    "gross_margin_ttm": (_f_gross_margin_ttm, {}),
    "net_margin_ttm": (_f_net_margin_ttm, {}),
    "debt_to_equity": (_f_debt_to_equity, {}),
    "current_ratio": (_f_current_ratio, {}),
    "accruals_ratio": (_f_accruals_ratio, {}),
    "earnings_stability": (_f_earnings_stability, {"window": 20}),
    "asset_turnover": (_f_asset_turnover, {}),
    "interest_coverage": (_f_interest_coverage, {}),
    # 增长
    "revenue_growth_yoy": (_f_revenue_growth_yoy, {}),
    "profit_growth_yoy": (_f_profit_growth_yoy, {}),
    "earnings_surprise": (_f_earnings_surprise, {}),
    "revenue_surprise": (_f_revenue_surprise, {}),
    "roe_growth_yoy": (_f_roe_growth_yoy, {}),
    "eps_revision_20d": (_f_eps_revision_20d, {"window": 20}),
    # 波动率
    "beta_60d": (_f_beta, {"window": 60}),
    "volatility_20d": (_f_volatility, {"window": 20}),
    "volatility_60d": (_f_volatility, {"window": 60}),
    "max_drawdown_60d": (_f_max_drawdown, {"window": 60}),
    "downside_risk_60d": (_f_downside_risk, {"window": 60}),
    "idiosyncratic_vol_60d": (_f_idio_vol, {"window": 60}),
    "skewness_20d": (_f_skewness, {"window": 20}),
    "kurtosis_20d": (_f_kurtosis, {"window": 20}),
    # 流动性
    "turnover_20d_avg": (_f_turnover_avg, {"window": 20}),
    "turnover_std_20d": (_f_turnover_std, {"window": 20}),
    "amihud_illiquidity_20d": (_f_amihud, {"window": 20}),
    "dollar_volume_20d": (_f_dollar_volume, {"window": 20}),
    "volume_ratio": (_f_volume_ratio, {"window": 20}),
    "zero_volume_days_20d": (_f_zero_volume_days, {"window": 20}),
    # 规模
    "ln_cap": (_f_ln_cap, {}),
    "ln_float_cap": (_f_ln_float_cap, {}),
    "non_linear_size": (_f_non_linear_size, {}),
    # 技术
    "ma5_pct": (_f_ma_pct, {"window": 5}),
    "ma20_pct": (_f_ma_pct, {"window": 20}),
    "ma60_pct": (_f_ma_pct, {"window": 60}),
    "rsi_14": (_f_rsi, {"window": 14}),
    "rsi_6": (_f_rsi, {"window": 6}),
    "macd_signal": (_f_macd_signal, {"fast": 12, "slow": 26, "signal": 9}),
    "bollinger_position": (_f_bollinger, {"window": 20, "n_std": 2}),
    "obv_ma_ratio": (_f_obv_ma_ratio, {"window": 20}),
    "volume_price_trend": (_f_vpt, {}),
    "williams_r_14": (_f_williams_r, {"window": 14}),
    "atr_14": (_f_atr, {"window": 14}),
    # 情绪
    "margin_change_5d": (_f_margin_change, {"window": 5}),
    "short_interest_ratio": (_f_short_interest, {}),
    "fund_flow_5d": (_f_fund_flow, {"window": 5}),
    "northbound_flow_5d": (_f_northbound_flow, {"window": 5}),
    # 宏观
    "treasury_yield_diff": (_f_yield_diff, {}),
    "vix_similar": (_f_vix, {}),
    "credit_spread": (_f_credit_spread, {}),
    "m2_growth": (_f_m2_growth, {}),
    # 合成
    "value_quality_score": (_f_composite, {"components": ["book_to_price", "roe_ttm", "fcf_yield"]}),
    "quality_growth_score": (_f_composite, {"components": ["roe_ttm", "revenue_growth_yoy", "profit_growth_yoy"]}),
    "low_vol_momentum_score": (_f_composite, {"components": ["momentum_3m", "volatility_60d"], "weights": [1.0, -1.0]}),
    "liquidity_adjusted_momentum": (_f_composite, {"components": ["momentum_6m", "dollar_volume_20d"]}),
    "defensive_quality_score": (_f_composite, {"components": ["roe_ttm", "debt_to_equity", "downside_risk_60d"], "weights": [1.0, -1.0, -1.0]}),
    "all_weather_score": (_f_composite, {"components": ["book_to_price", "roe_ttm", "momentum_6m", "volatility_60d", "dollar_volume_20d"], "weights": [1.0, 1.0, 1.0, -1.0, 0.5]}),
}


def register_default_factors(reset: bool = False) -> FactorRegistry:
    """注册首批 60+ 因子。"""
    registry = FactorRegistry()
    if reset:
        registry.clear()
    if registry._defaults_registered and not reset:
        return registry

    factors = [
        # === 动量因子 ===
        _fd("momentum_12m1m", FactorCategory.MOMENTUM, FactorHorizon.LONG, "returns_252d - returns_21d", "12个月动量剔除最近1个月（Barra标准）", ["close"], {"total_days": 252, "skip_days": 21}),
        _fd("momentum_6m", FactorCategory.MOMENTUM, FactorHorizon.MEDIUM, "returns_126d", "6个月动量", ["close"], {"days": 126}),
        _fd("momentum_3m", FactorCategory.MOMENTUM, FactorHorizon.MEDIUM, "returns_63d", "3个月动量", ["close"], {"days": 63}),
        _fd("short_term_reversal", FactorCategory.MOMENTUM, FactorHorizon.SHORT, "-returns_5d", "短期反转（最近5日收益取负）", ["close"], {"days": 5}),
        _fd("industry_momentum", FactorCategory.MOMENTUM, FactorHorizon.MEDIUM, "industry_returns_20d", "行业动量（行业指数20日收益）", ["industry_code", "close"], {"days": 20}),
        _fd("momentum_1m", FactorCategory.MOMENTUM, FactorHorizon.SHORT, "returns_21d", "1个月价格动量", ["close"], {"days": 21}),
        _fd("residual_momentum_60d", FactorCategory.MOMENTUM, FactorHorizon.MEDIUM, "returns_60d - beta * market_returns_60d", "Beta调整后的残差动量", ["close", "market_returns"], {"days": 60}),

        # === 价值因子 ===
        _fd("book_to_price", FactorCategory.VALUE, FactorHorizon.LONG, "book_value / market_cap", "账面市值比（BP）", ["book_value", "market_cap"]),
        _fd("earnings_to_price", FactorCategory.VALUE, FactorHorizon.LONG, "earnings / market_cap", "盈利收益率（EP）", ["earnings", "market_cap"]),
        _fd("cash_flow_to_price", FactorCategory.VALUE, FactorHorizon.LONG, "operating_cf / market_cap", "现金流收益率（CFP）", ["operating_cf", "market_cap"]),
        _fd("sales_to_price", FactorCategory.VALUE, FactorHorizon.LONG, "revenue / market_cap", "市销率倒数（SP）", ["revenue", "market_cap"]),
        _fd("dividend_yield", FactorCategory.VALUE, FactorHorizon.LONG, "dividend_per_share / close", "股息率", ["dividend_per_share", "close"]),
        _fd("fcf_yield", FactorCategory.VALUE, FactorHorizon.LONG, "free_cash_flow / market_cap", "自由现金流收益率", ["free_cash_flow", "market_cap"]),
        _fd("pe_ttm", FactorCategory.VALUE, FactorHorizon.LONG, "close / eps_ttm", "市盈率TTM", ["close", "eps_ttm"]),
        _fd("pb_lf", FactorCategory.VALUE, FactorHorizon.LONG, "close / book_value_per_share", "市净率", ["close", "book_value_per_share"]),
        _fd("ev_to_ebitda", FactorCategory.VALUE, FactorHorizon.LONG, "enterprise_value / ebitda", "企业价值倍数（低值更便宜）", ["enterprise_value", "ebitda"]),

        # === 质量因子 ===
        _fd("roe_ttm", FactorCategory.QUALITY, FactorHorizon.LONG, "net_profit_ttm / equity", "净资产收益率TTM", ["net_profit_ttm", "equity"]),
        _fd("roa_ttm", FactorCategory.QUALITY, FactorHorizon.LONG, "net_profit_ttm / total_assets", "总资产收益率", ["net_profit_ttm", "total_assets"]),
        _fd("gross_margin_ttm", FactorCategory.QUALITY, FactorHorizon.LONG, "(revenue - cost) / revenue", "毛利率TTM", ["revenue", "cost"]),
        _fd("net_margin_ttm", FactorCategory.QUALITY, FactorHorizon.LONG, "net_profit_ttm / revenue", "净利率TTM", ["net_profit_ttm", "revenue"]),
        _fd("debt_to_equity", FactorCategory.QUALITY, FactorHorizon.LONG, "total_liabilities / equity", "资产负债率", ["total_liabilities", "equity"]),
        _fd("current_ratio", FactorCategory.QUALITY, FactorHorizon.LONG, "current_assets / current_liabilities", "流动比率", ["current_assets", "current_liabilities"]),
        _fd("accruals_ratio", FactorCategory.QUALITY, FactorHorizon.MEDIUM, "(net_profit - operating_cf) / total_assets", "应计比率（Sloan 1996）", ["net_profit", "operating_cf", "total_assets"]),
        _fd("earnings_stability", FactorCategory.QUALITY, FactorHorizon.LONG, "-std(roe, 20)", "盈利稳定性（ROE波动率取负）", ["roe"], {"window": 20}),
        _fd("asset_turnover", FactorCategory.QUALITY, FactorHorizon.LONG, "revenue / total_assets", "总资产周转率", ["revenue", "total_assets"]),
        _fd("interest_coverage", FactorCategory.QUALITY, FactorHorizon.LONG, "ebit / interest_expense", "利息保障倍数", ["ebit", "interest_expense"]),

        # === 增长因子 ===
        _fd("revenue_growth_yoy", FactorCategory.GROWTH, FactorHorizon.LONG, "revenue / revenue_4q_ago - 1", "营收同比增长", ["revenue"]),
        _fd("profit_growth_yoy", FactorCategory.GROWTH, FactorHorizon.LONG, "net_profit / net_profit_4q_ago - 1", "净利润同比增长", ["net_profit"]),
        _fd("earnings_surprise", FactorCategory.GROWTH, FactorHorizon.SHORT, "(actual_eps - expected_eps) / close", "盈利超预期", ["actual_eps", "expected_eps", "close"]),
        _fd("revenue_surprise", FactorCategory.GROWTH, FactorHorizon.SHORT, "(actual_revenue - expected_revenue) / expected_revenue", "营收超预期", ["actual_revenue", "expected_revenue"]),
        _fd("roe_growth_yoy", FactorCategory.GROWTH, FactorHorizon.LONG, "roe / roe_4q_ago - 1", "ROE同比改善", ["roe"]),
        _fd("eps_revision_20d", FactorCategory.GROWTH, FactorHorizon.SHORT, "expected_eps / expected_eps_20d_ago - 1", "一致预期EPS 20日上修幅度", ["expected_eps"], {"window": 20}),

        # === 波动率因子 ===
        _fd("beta_60d", FactorCategory.VOLATILITY, FactorHorizon.MEDIUM, "cov(returns, market_returns, 60) / var(market_returns, 60)", "60日Beta（相对沪深300）", ["returns", "market_returns"], {"window": 60}),
        _fd("volatility_20d", FactorCategory.VOLATILITY, FactorHorizon.SHORT, "std(returns, 20)", "20日收益波动率", ["returns"], {"window": 20}),
        _fd("volatility_60d", FactorCategory.VOLATILITY, FactorHorizon.MEDIUM, "std(returns, 60)", "60日收益波动率", ["returns"], {"window": 60}),
        _fd("max_drawdown_60d", FactorCategory.VOLATILITY, FactorHorizon.MEDIUM, "max_drawdown(close, 60)", "60日最大回撤", ["close"], {"window": 60}),
        _fd("downside_risk_60d", FactorCategory.VOLATILITY, FactorHorizon.MEDIUM, "std(min(returns, 0), 60)", "下行风险（60日负收益波动率）", ["returns"], {"window": 60}),
        _fd("idiosyncratic_vol_60d", FactorCategory.VOLATILITY, FactorHorizon.MEDIUM, "std(residual_from_beta_regression, 60)", "特异波动率（Beta回归残差波动率）", ["returns", "market_returns"], {"window": 60}),
        _fd("skewness_20d", FactorCategory.VOLATILITY, FactorHorizon.SHORT, "skew(returns, 20)", "20日收益偏度", ["returns"], {"window": 20}),
        _fd("kurtosis_20d", FactorCategory.VOLATILITY, FactorHorizon.SHORT, "kurtosis(returns, 20)", "20日收益峰度", ["returns"], {"window": 20}),

        # === 流动性因子 ===
        _fd("turnover_20d_avg", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "mean(turnover, 20)", "20日平均换手率", ["turnover"], {"window": 20}),
        _fd("turnover_std_20d", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "std(turnover, 20)", "换手率波动率", ["turnover"], {"window": 20}),
        _fd("amihud_illiquidity_20d", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "mean(abs(returns) / amount, 20)", "Amihud非流动性指标（20日均）", ["returns", "amount"], {"window": 20}),
        _fd("dollar_volume_20d", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "log(mean(amount, 20))", "对数日均成交额", ["amount"], {"window": 20}),
        _fd("volume_ratio", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "volume / mean(volume, 20)", "量比（当日成交量/20日均量）", ["volume"], {"window": 20}),
        _fd("zero_volume_days_20d", FactorCategory.LIQUIDITY, FactorHorizon.SHORT, "count(volume == 0, 20) / 20", "20日零成交占比", ["volume"], {"window": 20}),

        # === 规模因子 ===
        _fd("ln_cap", FactorCategory.SIZE, FactorHorizon.LONG, "log(market_cap)", "对数总市值", ["market_cap"]),
        _fd("ln_float_cap", FactorCategory.SIZE, FactorHorizon.LONG, "log(float_market_cap)", "对数流通市值", ["float_market_cap"]),
        _fd("non_linear_size", FactorCategory.SIZE, FactorHorizon.LONG, "ln_cap ** 3", "非线性规模（Barra CNE6）", ["ln_cap"]),

        # === 技术因子 ===
        _fd("ma5_pct", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "close / ma(close, 5) - 1", "收盘价相对5日均线偏离度", ["close"], {"window": 5}),
        _fd("ma20_pct", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "close / ma(close, 20) - 1", "收盘价相对20日均线偏离度", ["close"], {"window": 20}),
        _fd("ma60_pct", FactorCategory.TECHNICAL, FactorHorizon.MEDIUM, "close / ma(close, 60) - 1", "收盘价相对60日均线偏离度", ["close"], {"window": 60}),
        _fd("rsi_14", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "rsi(close, 14)", "14日RSI", ["close"], {"window": 14}),
        _fd("rsi_6", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "rsi(close, 6)", "6日RSI", ["close"], {"window": 6}),
        _fd("macd_signal", FactorCategory.TECHNICAL, FactorHorizon.MEDIUM, "macd(close, 12, 26) - signal(close, 12, 26, 9)", "MACD信号线差值", ["close"], {"fast": 12, "slow": 26, "signal": 9}),
        _fd("bollinger_position", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "(close - ma(close,20)) / (2*std(close,20))", "布林带位置（当前价在带中的z-score）", ["close"], {"window": 20, "n_std": 2}),
        _fd("obv_ma_ratio", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "obv(close, volume) / ma(obv, 20)", "OBV相对20日均值比", ["close", "volume"], {"window": 20}),
        _fd("volume_price_trend", FactorCategory.TECHNICAL, FactorHorizon.MEDIUM, "vpt(close, volume)", "量价趋势指标", ["close", "volume"]),
        _fd("williams_r_14", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "williams_r(high, low, close, 14)", "Williams %R 14日", ["high", "low", "close"], {"window": 14}),
        _fd("atr_14", FactorCategory.TECHNICAL, FactorHorizon.SHORT, "atr(high, low, close, 14)", "14日平均真实波幅", ["high", "low", "close"], {"window": 14}),

        # === 情绪因子 ===
        _fd("margin_change_5d", FactorCategory.SENTIMENT, FactorHorizon.SHORT, "margin_balance / margin_balance_5d_ago - 1", "融资余额5日变化率", ["margin_balance"], {"window": 5}),
        _fd("short_interest_ratio", FactorCategory.SENTIMENT, FactorHorizon.SHORT, "short_volume / total_volume", "融券卖出占比", ["short_volume", "total_volume"]),
        _fd("fund_flow_5d", FactorCategory.SENTIMENT, FactorHorizon.SHORT, "net_fund_flow_5d / float_market_cap", "主力资金5日净流入占比", ["net_fund_flow", "float_market_cap"], {"window": 5}),
        _fd("northbound_flow_5d", FactorCategory.SENTIMENT, FactorHorizon.SHORT, "northbound_net_buy_5d / float_market_cap", "北向资金5日净买入占比", ["northbound_net_buy", "float_market_cap"], {"window": 5}),

        # === 宏观因子 ===
        _fd("treasury_yield_diff", FactorCategory.MACRO, FactorHorizon.MACRO, "10y_yield - 2y_yield", "期限利差（10年-2年国债收益率差）", ["10y_yield", "2y_yield"]),
        _fd("vix_similar", FactorCategory.MACRO, FactorHorizon.SHORT, "volatility_index_value", "市场隐含波动率水平", ["volatility_index"]),
        _fd("credit_spread", FactorCategory.MACRO, FactorHorizon.MACRO, "corp_bond_yield - treasury_yield", "信用利差", ["corp_bond_yield", "treasury_yield"]),
        _fd("m2_growth", FactorCategory.MACRO, FactorHorizon.MACRO, "m2_yoy", "M2同比增速", ["m2_yoy"]),

        # === 合成因子 ===
        _fd("value_quality_score", FactorCategory.COMPOSITE, FactorHorizon.LONG, "z(book_to_price) + z(roe_ttm) + z(fcf_yield)", "价值质量合成分", ["book_to_price", "roe_ttm", "fcf_yield"], {"components": ["book_to_price", "roe_ttm", "fcf_yield"]}),
        _fd("quality_growth_score", FactorCategory.COMPOSITE, FactorHorizon.LONG, "z(roe_ttm) + z(revenue_growth_yoy) + z(profit_growth_yoy)", "质量增长合成分", ["roe_ttm", "revenue_growth_yoy", "profit_growth_yoy"], {"components": ["roe_ttm", "revenue_growth_yoy", "profit_growth_yoy"]}),
        _fd("low_vol_momentum_score", FactorCategory.COMPOSITE, FactorHorizon.MEDIUM, "z(momentum_3m) - z(volatility_60d)", "低波动量合成分", ["momentum_3m", "volatility_60d"], {"components": ["momentum_3m", "volatility_60d"], "weights": [1.0, -1.0]}),
        _fd("liquidity_adjusted_momentum", FactorCategory.COMPOSITE, FactorHorizon.MEDIUM, "z(momentum_6m) + z(dollar_volume_20d)", "流动性调整动量", ["momentum_6m", "dollar_volume_20d"], {"components": ["momentum_6m", "dollar_volume_20d"]}),
        _fd("defensive_quality_score", FactorCategory.COMPOSITE, FactorHorizon.LONG, "z(roe_ttm) - z(debt_to_equity) - z(downside_risk_60d)", "防御型质量分", ["roe_ttm", "debt_to_equity", "downside_risk_60d"], {"components": ["roe_ttm", "debt_to_equity", "downside_risk_60d"], "weights": [1.0, -1.0, -1.0]}),
        _fd("all_weather_score", FactorCategory.COMPOSITE, FactorHorizon.LONG, "value + quality + momentum + low_vol + liquidity", "全周期多因子综合分", ["book_to_price", "roe_ttm", "momentum_6m", "volatility_60d", "dollar_volume_20d"], {"components": ["book_to_price", "roe_ttm", "momentum_6m", "volatility_60d", "dollar_volume_20d"], "weights": [1.0, 1.0, 1.0, -1.0, 0.5]}),
    ]

    for factor in factors:
        # V6 fix (Q5 CRITICAL: registry.py:53,180-250): 绑定真实计算函数。
        #   V5.4 注册的因子 computation_fn 全为 None → engine.compute 抛
        #   TypeError；V6 改为：缺失实现即 fail-fast（启动即报错，绝不静默）。
        impl = _FACTOR_IMPLS.get(factor.name)
        if impl is None:
            raise ValueError(
                f"[FactorRegistry] 因子 {factor.name} 缺少 computation_fn 实现，"
                f"请在 _FACTOR_IMPLS 中登记（V6 要求全部 74 个因子绑定真实实现）"
            )
        factor.computation_fn = impl[0]
        registry.register(factor)
    registry._defaults_registered = True
    return registry


def get_default_registry() -> FactorRegistry:
    """获取已预注册默认因子的全局注册表。"""
    return register_default_factors()


DEFAULT_REGISTRY = register_default_factors()


__all__ = [
    "DEFAULT_REGISTRY",
    "FactorCategory",
    "FactorDef",
    "FactorHorizon",
    "FactorRegistry",
    "get_default_registry",
    "register_default_factors",
]
