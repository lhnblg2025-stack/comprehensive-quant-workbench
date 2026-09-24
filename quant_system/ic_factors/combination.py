"""
combination.py — QuantV6 因子合成
方向感知加权（动量40%/低波30%/技术30%），按可用因子动态加权归一化。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 因子组合(方向/权重)独特保留。
"""
from __future__ import annotations

import pandas as pd

from quant_system.market_forecast._support.common.math_utils import normalize_weights
from quant_system.ic_factors.processor import process_factor_frame

# 默认分组权重（可配置）。P1-7：新增 volume/liquidity 两组后五组均分。
DEFAULT_GROUP_WEIGHTS = {"momentum": 0.20, "lowvol": 0.20, "technical": 0.20,
                         "volume": 0.20, "liquidity": 0.20}

# 因子 → 分组映射（P1-7 补全：覆盖 zoo.py 全部 38 个注册因子）
# 量能类 → volume 组；流动性类 → liquidity 组；趋势/突破类 → momentum；
# 反转/超买超卖/形态类 → technical。
FACTOR_GROUP = {
    # 动量/趋势
    "mom20": "momentum", "mom1m": "momentum", "mom3m": "momentum", "mom6m": "momentum",
    "mom12m": "momentum", "ma_cross": "momentum", "macd_hist": "momentum",
    "macd_cross": "momentum", "new_high_dist": "momentum", "new_low_dist": "momentum",
    "ma_trend": "momentum",
    # 量能（新建组）
    "volume_trend": "volume", "volume_surge": "volume", "vol_20d": "volume",
    "volume_ratio": "volume", "obv_slope": "volume", "volume_price_fit": "volume",
    # 低波/风险
    "realized_vol": "lowvol", "downside_vol": "lowvol", "vol_change": "lowvol",
    "atr_ratio": "lowvol", "max_dd_12m": "lowvol", "beta_60d": "lowvol",
    "downside_beta": "lowvol", "boll_width": "lowvol",
    # 流动性（新建组）
    "amount_liquidity": "liquidity", "amihud_illiq": "liquidity",
    "turnover_stability": "liquidity", "spread_approx": "liquidity",
    # 技术/反转/超买超卖
    "rsi14": "technical", "bias12": "technical", "cci_20": "technical",
    "boll_pos": "technical", "kdj": "technical", "williams": "technical",
    "dmi": "technical", "donchian": "technical", "high_low_pos": "technical",
    "macd_div": "technical", "rev5": "technical",
}


def _apply_direction(frame: pd.DataFrame) -> pd.DataFrame:
    """P0-2：按 zoo 元数据对原始因子列做方向校正（越大越看多）。

    仅用于调用方传入“原始值”帧的场景（apply_direction=True）。
    经 zoo.compute_factor_frame（默认 apply_direction=True）产出的帧
    已方向校正，不得再次调用（避免双重翻转）。
    """
    from quant_system.ic_factors.zoo import get_factor
    out = frame.copy()
    for col in out.columns:
        fac = get_factor(col)
        if fac is not None and fac.direction != 1:
            out[col] = out[col] * fac.direction
    return out


def combine_factors(frame: pd.DataFrame, weights: dict[str, float] | None = None,
                    group_weights: dict[str, float] | None = None,
                    exposures: pd.DataFrame | None = None,
                    apply_direction: bool = False) -> pd.Series:
    """
    因子合成综合分（越大越好）。
    - weights: 因子级权重覆盖（None 时用分组权重自动分配）
    - group_weights: 分组权重（默认五组均分 20% 每组）
    - exposures: 中性化暴露（可选）
    - apply_direction: 帧是否为“原始值”？True 时按 zoo 元数据乘 direction
      （compute_factor_frame 输出已方向校正，默认 False 避免双重翻转）
    """
    if frame is None or len(frame) == 0:
        return pd.Series(dtype=float)

    if apply_direction:
        frame = _apply_direction(frame)

    z = process_factor_frame(frame, exposures=exposures)
    if z is None or len(z) == 0:
        return pd.Series(dtype=float)

    gw = {**DEFAULT_GROUP_WEIGHTS, **(group_weights or {})}

    if weights is None:
        # 分组内等权，组间按 group_weights
        per_factor: dict[str, float] = {}
        groups_used: dict[str, int] = {}
        for col in z.columns:
            g = FACTOR_GROUP.get(col, "technical")
            groups_used[g] = groups_used.get(g, 0) + 1
        for col in z.columns:
            g = FACTOR_GROUP.get(col, "technical")
            n = max(groups_used.get(g, 1), 1)
            per_factor[col] = gw.get(g, 0.3) / n
        weights = per_factor

    weights = normalize_weights({k: v for k, v in weights.items() if k in z.columns})
    score = pd.Series(0.0, index=z.index)
    for col in z.columns:
        w = weights.get(col, 0.0)
        if w > 0:
            score += z[col].fillna(0.0) * w
    return score.sort_values(ascending=False)


def top_n(score: pd.Series, n: int = 20) -> list[str]:
    """取综合分前 N 股票代码。"""
    if score is None or len(score) == 0:
        return []
    return [str(x) for x in score.dropna().sort_values(ascending=False).head(n).index]


def factor_contribution(score: pd.Series, frame: pd.DataFrame,
                        weights: dict[str, float]) -> pd.DataFrame:
    """因子贡献分解（每只股票的因子得分×权重）。"""
    out = pd.DataFrame(index=score.index)
    for col, w in weights.items():
        if col in frame.columns:
            out[col] = frame[col].fillna(0.0) * w
    out["total"] = out.sum(axis=1)
    return out
