"""
composite.py — V7.0 因子合成器（多因子七步法第 5-7 步）
=====================================================
- 输入: panels: dict[因子名 -> DataFrame(date × stocks) 原始横截面值]
- 合成: 逐日横截面 z-score → 加权求和 → (date × stocks) 合成得分
- 方法: EW / ICW / ICIR / MVO
- DynamicFactorSelector: regime + factor_health 动态调权
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 因子合成层(EW/ICW/ICIR/MVO)独特保留；common_dates 已收敛至 base.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.ic_factors.base import common_dates

log = get_logger("qv6.factor.composite")

# ── 因子白名单（V8：IC 实证驱动的噪声过滤）─────────────────────────────
# 依据 generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv 的 grade：
#   A/B → 保留；C → 降权 0.5；D → 剔除（权重 0）
# 白名单文件可外部覆盖：config/factor_whitelist.csv（列 factor,grade）
IC_REPORT_DEFAULT = Path(__file__).resolve().parents[2] / "generated" / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"


def _load_grade_map(csv_path: str | Path | None = None) -> dict[str, str]:
    """读取 IC 报告，返回 {因子名: grade}。文件缺失时返回空 dict（不拦截任何因子）。"""
    p = Path(csv_path) if csv_path else IC_REPORT_DEFAULT
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if "factor" not in df.columns or "grade" not in df.columns:
            return {}
        return dict(zip(df["factor"].astype(str), df["grade"].astype(str)))
    except Exception as e:
        log.warning(f"factor whitelist 加载失败: {e}")
        return {}


def grade_factor_weights(factors: list[str], grade_map: dict[str, str],
                         c_weight: float = 0.5, d_weight: float = 0.0) -> pd.Series:
    """依据 IC grade 生成因子权重系数：A/B=1.0, C=c_weight, D=d_weight。"""
    w = pd.Series(1.0, index=factors)
    for f in factors:
        g = grade_map.get(str(f), "")
        if g == "C":
            w[f] = c_weight
        elif g == "D":
            w[f] = d_weight
    return w


def _zscore_cross(X: pd.DataFrame) -> pd.DataFrame:
    """逐行(股票)横截面 z-score（MAD 截尾）。X: index=股票, columns=因子。"""
    from quant_system.market_forecast._support.common.math_utils import mad_winsorize
    out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        v = X[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 30:
            continue
        w = mad_winsorize(v)
        mu, sd = w.mean(), w.std()
        if sd > 1e-12:
            out.loc[w.index, c] = (w - mu) / sd
    return out


def composite_scores(panels: dict[str, pd.DataFrame],
                     weights: pd.Series) -> pd.DataFrame:
    """加权合成（逐日横截面 z-score 后加权求和）。

    panels: {因子名: DataFrame(date × stocks)}
    weights: Series(index=因子名, 值=权重, 无需归一)
    返回: DataFrame(date × stocks) 合成得分
    D4收敛登记: 独特因子保留
    """
    cols = [f for f in weights.index if f in panels]
    if not cols:
        return pd.DataFrame()
    dates = common_dates({c: panels[c] for c in cols})
    if len(dates) == 0:
        return pd.DataFrame()
    stocks = sorted(set().union(*[set(panels[c].columns) for c in cols]))
    out = pd.DataFrame(index=dates, columns=stocks, dtype=float)
    w = weights.reindex(cols).fillna(0.0)
    w = w / (w.abs().sum() + 1e-12)
    for d in dates:
        X = pd.DataFrame({c: panels[c].loc[d] for c in cols})
        Z = _zscore_cross(X)
        if Z.empty:
            continue
        score = Z.mul(w, axis=1).sum(axis=1)
        out.loc[d, score.index] = score
    return out


def composite_ew(panels: dict[str, pd.DataFrame],
                 factors: list[str] | None = None) -> pd.DataFrame:
    """等权合成。
    D4收敛登记: 独特因子保留
    """
    cols = factors or list(panels.keys())
    if not cols:
        return pd.DataFrame()
    w = pd.Series(1.0, index=cols)
    return composite_scores(panels, w)


def composite_icw(panels: dict[str, pd.DataFrame], ics: pd.Series,
                  factors: list[str] | None = None) -> pd.DataFrame:
    """信息系数加权。ics: Series(index=因子名 → IC 均值)。
    D4收敛登记: 独特因子保留
    """
    cols = factors or list(panels.keys())
    w = ics.reindex(cols).abs().fillna(0.0)
    if w.sum() <= 0:
        return composite_ew(panels, cols)
    return composite_scores(panels, w)


def composite_icir(panels: dict[str, pd.DataFrame], icirs: pd.Series,
                   factors: list[str] | None = None) -> pd.DataFrame:
    """ICIR 加权（默认）。
    D4收敛登记: 独特因子保留
    """
    cols = factors or list(panels.keys())
    w = icirs.reindex(cols).abs().fillna(0.0)
    if w.sum() <= 0:
        return composite_ew(panels, cols)
    return composite_scores(panels, w)


def composite_mvo(panels: dict[str, pd.DataFrame],
                  ret_panels: dict[str, pd.DataFrame],
                  factors: list[str] | None = None,
                  l2: float = 0.5) -> pd.DataFrame:
    """均值-方差优化合成（因子收益协方差，正则化防过拟合）。
    D4收敛登记: 独特因子保留

    W2.5 P2 警告：权重用全样本因子收益 mean/cov 一次性估出再套全部日期，
    逐日使用会前视（未来收益分布进入历史权重）。逐日场景请改 rolling/expanding。
    """
    cols = factors or list(panels.keys())
    if len(cols) < 2:
        return composite_ew(panels, cols)
    # 逐日因子"收益"代理: 因子值 × 个股次日收益 的横截面协方差
    dates = common_dates({c: panels[c] for c in cols})
    ret_dates = common_dates(ret_panels) if ret_panels else pd.Index([])
    dates = dates.intersection(ret_dates)
    if len(dates) < 60:
        return composite_icir(panels, pd.Series(1.0, index=cols), cols)
    f_ret = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for d in dates:
        for c in cols:
            fv = panels[c].loc[d].astype(float)
            rv = next(iter(ret_panels.values())).loc[d].astype(float) \
                if len(ret_panels) == 1 else _avg_ret(ret_panels, d)
            m = fv.notna() & rv.notna() & np.isfinite(fv) & np.isfinite(rv)
            if m.sum() >= 30:
                f_ret.loc[d, c] = fv[m].cov(rv[m]) * 100
    f_ret = f_ret.dropna(how="all")
    if f_ret.shape[0] < 60 or f_ret.shape[1] < 2:
        return composite_icir(panels, pd.Series(1.0, index=cols), cols)
    mu = f_ret.mean().values
    cov = f_ret.cov().values + np.eye(len(cols)) * l2
    try:
        inv = np.linalg.inv(cov)
        w = inv @ mu
    except np.linalg.LinAlgError:
        w = np.ones(len(cols)) / len(cols)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        w = np.ones(len(cols)) / len(cols)
    ws = pd.Series(w / w.sum(), index=cols)
    return composite_scores(panels, ws)


def _avg_ret(ret_panels: dict[str, pd.DataFrame], d) -> pd.Series:
    parts = [p.loc[d].astype(float) for p in ret_panels.values() if d in p.index]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1)


# ── 动态选择器 ──────────────────────────────────────────────

REGIME_CATEGORY_MULT = {
    "bull":   {"momentum": 1.3, "flow": 1.3, "sentiment": 1.3, "growth": 1.2,
               "quality": 1.0, "value": 0.8, "reversal": 0.7, "volatility": 0.8},
    "oscill": {"quality": 1.3, "value": 1.3, "reversal": 1.3, "volatility": 1.1,
               "momentum": 0.9, "flow": 1.0, "sentiment": 0.9},
    "bear":   {"reversal": 1.5, "volatility": 1.4, "value": 1.3, "quality": 1.2,
               "momentum": 0.5, "flow": 0.6, "sentiment": 0.5, "growth": 0.6},
    "panic":  {"reversal": 1.5, "volatility": 1.4, "value": 1.3, "quality": 1.2,
               "momentum": 0.4, "flow": 0.5, "sentiment": 0.4},
}
REGIMES = ("bull", "oscill", "bear", "panic")


def detect_regime(index_ret: pd.Series, n_ma: int = 60) -> str:
    """简易 regime 识别：价格在 60 日均线上/下 + 近 20 日波动率分位。"""
    if index_ret.empty or len(index_ret) < n_ma + 20:
        return "oscill"
    price = (1 + index_ret).cumprod()
    ma = price.rolling(n_ma).mean()
    above = float(price.iloc[-1] > ma.iloc[-1])
    vol20 = float(index_ret.tail(20).std())
    vol_hist = index_ret.rolling(120).std().dropna()
    vol_pct = float((vol_hist <= vol20).mean()) if len(vol_hist) > 20 else 0.5
    if above and vol_pct < 0.7:
        return "bull"
    if above:
        return "oscill"
    if vol_pct > 0.8:
        return "panic"
    return "bear"


@dataclass
class DynamicFactorSelector:
    """动态选择器：regime + factor_health → 权重向量。"""

    regime: str = "oscill"
    icir_weight: float = 0.6      # 基础权重中 ICIR 占比
    ew_weight: float = 0.4        # 等权兜底占比
    name_to_category: dict = field(default_factory=dict)  # 因子→大类

    def _cat(self, name: str) -> str:
        if name in self.name_to_category:
            return self.name_to_category[name]
        # 前缀猜测
        for k, v in {
            "mom": "momentum", "rev": "reversal", "vol_": "volatility",
            "realized": "volatility", "beta": "volatility", "atr": "volatility",
            "amount": "liquidity", "amihud": "liquidity", "turnover": "liquidity",
            "spread": "liquidity", "kdj": "reversal", "rsi": "reversal",
            "bias": "reversal", "new_high": "momentum", "ma_": "momentum",
            "roe": "quality", "gross": "quality", "quality": "quality",
            "growth": "growth", "rev_g": "growth", "pe_": "value", "pb_": "value",
            "val": "value", "north": "flow", "margin": "flow", "flow": "flow",
            "holder": "position", "pledge": "position", "unlock": "position",
            "news": "sentiment", "hot": "sentiment", "sent": "sentiment",
        }.items():
            if name.startswith(k):
                return v
        return "momentum"

    def weights(self, icirs: pd.Series, health: pd.DataFrame | None = None,
                factors: list[str] | None = None, grade_map: dict[str, str] | None = None,
                oos_rules: dict | None = None) -> pd.Series:
        """计算因子权重（归一）。

        grade_map: {因子名: grade}，D 级剔除、C 级降权。
        oos_rules: oos_pool_rule.load_oos_report() 返回的 {因子名: {...}}；
            W2.4 硬规则：OOS 符号翻转 → 降权/退池（乘 weight_multiplier）。
        """
        cols = factors or list(icirs.index)
        if not cols:
            return pd.Series(dtype=float)
        if grade_map:
            gw = grade_factor_weights(cols, grade_map)
            cols = [c for c in cols if gw.get(c, 1.0) > 0]  # 剔除 D 级（权重 0）
            if not cols:
                return pd.Series(dtype=float)
        else:
            gw = None
        ic = icirs.reindex(cols).abs().fillna(0.0)
        w_ic = ic / (ic.sum() + 1e-12)
        w_ew = pd.Series(1.0 / len(cols), index=cols)
        w = self.ew_weight * w_ew + self.icir_weight * w_ic
        if gw is not None:
            w = w * gw.reindex(cols).fillna(1.0)  # C 级降权
        mult = REGIME_CATEGORY_MULT.get(self.regime, {})
        for c in cols:
            cat = self._cat(c)
            if cat in mult:
                w[c] *= mult[cat]
        # W2.4：OOS 入池硬规则（翻转降权/退池），归一化前施加系数
        if oos_rules:
            from quant_system.ic_factors.oos_pool_rule import apply_oos_rule_to_weights
            w = apply_oos_rule_to_weights(w, oos_rules)
            w = w[w > 0]  # retire → 0 直接剔出池
        s = w.sum()
        if s > 0:
            w = w / s
        return w

    def update_regime(self, index_ret: pd.Series) -> str:
        self.regime = detect_regime(index_ret)
        log.info(f"regime → {self.regime}")
        return self.regime


@dataclass
class CompositeEngine:
    """统一合成入口。"""

    method: str = "icir"          # ew / icw / icir / mvo
    selector: DynamicFactorSelector = field(default_factory=DynamicFactorSelector)

    def compute(self, panels: dict[str, pd.DataFrame],
                *, icirs: pd.Series | None = None,
                ret_panels: dict[str, pd.DataFrame] | None = None,
                health: pd.DataFrame | None = None,
                factors: list[str] | None = None,
                use_whitelist: bool = True,
                use_oos_rule: bool = True,
                oos_report_path: str | Path | None = None) -> pd.DataFrame:
        """输出 composite_scores: DataFrame(date × stocks)。

        use_whitelist=True 时按 IC grade 剔除 D 级、降权 C 级（V8 噪声过滤）。
        use_oos_rule=True 时按 W2.4 OOS 符号翻转硬规则降权/退池（报告缺失则规则不生效）。
        """
        cols = factors or list(panels.keys())
        if not cols:
            return pd.DataFrame()
        grade_map = _load_grade_map() if use_whitelist else {}
        oos_rules = {}
        if use_oos_rule:
            from quant_system.ic_factors.oos_pool_rule import load_oos_report
            oos_rules = load_oos_report(oos_report_path)
        if self.method == "ew":
            w = pd.Series(1.0, index=cols)
            if grade_map:
                gw = grade_factor_weights(cols, grade_map)
                cols = [c for c in cols if gw.get(c, 1.0) > 0]
                w = w.reindex(cols).fillna(0.0)
            if oos_rules:
                from quant_system.ic_factors.oos_pool_rule import apply_oos_rule_to_weights
                w = apply_oos_rule_to_weights(w, oos_rules)
                cols = [c for c in cols if w.get(c, 0.0) > 0]
                return composite_scores(panels, w) if cols else pd.DataFrame()
            return composite_ew(panels, cols)
        if self.method == "icw" and icirs is not None:
            return composite_icw(panels, icirs, cols)
        if self.method == "mvo" and ret_panels is not None:
            return composite_mvo(panels, ret_panels, cols)
        if icirs is not None:
            w = self.selector.weights(icirs, health, cols, grade_map, oos_rules)
            return composite_scores(panels, w)
        return composite_ew(panels, cols)
