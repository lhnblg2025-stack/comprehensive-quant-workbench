"""
filter.py — V7.0 因子四重过滤（多因子七步法第 3 步）
=====================================================
有效性 / 稳定性 / 冗余性 / 时效性 四重过滤 → 存活因子池。
产出 factor_health 表（供 warehouse 持久化 + 合成器消费）。
D4收敛登记 (2026-08-11): 因子域收敛（保守策略）——因子定义唯一真源 = factor_zoo.py；本模块因子与 factor_zoo 同名异口径/异名异口径的均保留独立实现，不强迁。 因子筛选/IC过滤独特保留。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor.filter")


def compute_ic(factor_wide: pd.DataFrame, ret_wide: pd.DataFrame,
               method: str = "spearman", min_n: int = 30) -> pd.Series:
    """逐日横截面 IC。

    factor_wide / ret_wide: index=date, columns=股票代码
    min_n: 当日最少有效股票数（默认 30）
    返回: Series(index=date, name='ic')
    """
    dates = factor_wide.index.intersection(ret_wide.index)
    ics = []
    for d in dates:
        f = factor_wide.loc[d].astype(float)
        r = ret_wide.loc[d].astype(float)
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_n:   # 最少 min_n 只股票
            continue
        if method == "spearman":
            ic = f[mask].rank().corr(r[mask].rank())
        else:
            ic = f[mask].corr(r[mask])
        if pd.notna(ic):
            ics.append((d, ic))
    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(dict(ics)).sort_index()


def ic_stats(ic: pd.Series) -> dict:
    """IC 统计：均值/ICIR/胜率/标准差。"""
    if ic.empty:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_winrate": np.nan,
                "ic_std": np.nan, "n": 0}
    mean = float(ic.mean())
    std = float(ic.std())
    ir = mean / std if std > 1e-12 else 0.0
    win = float((ic > 0).mean())
    return {"ic_mean": round(mean, 5), "icir": round(ir, 4),
            "ic_winrate": round(win, 4), "ic_std": round(std, 5),
            "n": int(len(ic))}


def factor_corr_matrix(factor_wide: pd.DataFrame) -> pd.DataFrame:
    """因子间秩相关矩阵（横截面均值）。

    支持两种布局：
    - 宽表：index=唯一日期, columns=因子（每日期 1 行）→ 无横截面，返回 NaN 占位
    - 长表：index=重复日期（每日期 N 只股票行）, columns=因子 → 按日期分组计算
      横截面 Spearman 相关，再对日期取均值；组内行数 < 30 的日期跳过。

    向量化实现（按日期 groupby + DataFrame.corr），避免逐 (行×列对)
    pandas loc 嵌套循环的 O(n²) 性能问题。
    """
    cols = list(factor_wide.columns)
    if len(cols) < 2:
        return pd.DataFrame(index=cols, columns=cols)
    recent = factor_wide.tail(120)
    if recent.index.is_unique:
        # 宽表：每日期仅 1 行，无横截面 → NaN 占位
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    mats = []
    for _, g in recent.groupby(level=0):
        if len(g) < 30:
            continue
        mats.append(g.rank().corr(method="spearman"))
    if not mats:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    out = sum(mats) / len(mats)
    return out.reindex(index=cols, columns=cols)


def dedup_clusters(corr: pd.DataFrame, threshold: float = 0.7,
                   keep: dict | None = None) -> list[list[str]]:
    """按相关矩阵聚类去冗余。返回簇列表，每簇保留 keep 指定的因子（默认 None=全部标注）。"""
    cols = list(corr.columns)
    visited = set()
    clusters = []
    for i, a in enumerate(cols):
        if a in visited:
            continue
        group = [a]
        for b in cols[i + 1:]:
            if b in visited:
                continue
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                group.append(b)
        for g in group:
            visited.add(g)
        clusters.append(group)
    return clusters


@dataclass
class FilterResult:
    """四重过滤结果。"""
    factor_health: pd.DataFrame          # 每因子 IC/ICIR/PSI/status
    active_factors: list[str]           # 存活因子池
    dropped: dict                        # {因子: 淘汰原因}
    clusters: list[list[str]] = field(default_factory=list)  # 冗余簇


def filter_factors(
    ic_frame: pd.DataFrame,               # index=因子名, columns=[ic_mean, icir, ic_winrate, ic_std, n]
    psi_frame: pd.Series | None = None,   # index=因子名 → PSI 值
    corr_matrix: pd.DataFrame | None = None,
    *,
    ic_min: float = 0.02,                 # 有效性: |IC均值| 下限
    icir_min: float = 0.3,                # 有效性: ICIR 下限
    winrate_min: float = 0.55,            # 稳定性: 月度胜率下限
    corr_threshold: float = 0.7,          # 冗余性: 秩相关阈值
    psi_watch: float = 0.25,              # 时效性: PSI 观察阈值
    psi_drop: float = 0.4,                # 时效性: PSI 停用阈值
    min_samples: int = 30,                # 最少样本期数
) -> FilterResult:
    """四重过滤主入口。

    返回 factor_health DataFrame:
      factor | ic_mean | icir | ic_winrate | ic_std | n | psi | status | reason
    status: active / watch / inactive
    """
    if ic_frame.empty:
        return FilterResult(pd.DataFrame(), [], {}, [])
    health = ic_frame.copy()
    # 时效性
    if psi_frame is not None:
        health["psi"] = psi_frame.reindex(health.index).fillna(0.0)
    else:
        health["psi"] = 0.0

    dropped: dict[str, str] = {}
    reasons = {}
    for name, row in health.iterrows():
        r = ""
        n = row.get("n", 0)
        ic_mean = row.get("ic_mean", np.nan)
        icir = row.get("icir", np.nan)
        if n < min_samples:
            r = f"样本不足(n={n}<{min_samples})"
        # NaN IC 视为"IC 数据不足"：float('nan') or 0 返回 nan（truthy），
        # abs(nan) < ic_min 恒为 False，导致 NaN IC 的因子永远逃过弱 IC 检查
        elif pd.isna(ic_mean) or pd.isna(icir):
            r = "IC数据不足(NaN)"
        elif abs(ic_mean) < ic_min and icir < icir_min:
            r = f"IC弱(|IC|={abs(ic_mean):.3f}<{ic_min}, ICIR={icir:.2f}<{icir_min})"
        if r:
            dropped[name] = r
            reasons[name] = r

    health["status"] = "active"
    health["reason"] = ""
    for name, r in dropped.items():
        health.loc[name, "status"] = "inactive"
        health.loc[name, "reason"] = r

    # 稳定性：胜率低于阈值的降为 watch（NaN 胜率不触发：NaN < x 为 False）
    mask = (health["status"] == "active") & (health.get("ic_winrate", pd.Series(dtype=float)) < winrate_min)
    for name in health.index[mask]:
        health.loc[name, "status"] = "watch"
        health.loc[name, "reason"] = f"胜率{health.loc[name, 'ic_winrate']:.2f}<{winrate_min}"

    # 时效性：PSI 降权/停用
    mask_watch = (health["status"] == "active") & (health["psi"] > psi_watch)
    for name in health.index[mask_watch]:
        health.loc[name, "status"] = "watch"
        health.loc[name, "reason"] = f"PSI={health.loc[name, 'psi']:.2f}漂移"
    mask_drop = health["psi"] > psi_drop
    for name in health.index[mask_drop]:
        health.loc[name, "status"] = "inactive"
        health.loc[name, "reason"] = f"PSI={health.loc[name, 'psi']:.2f}严重漂移"
        dropped[name] = health.loc[name, "reason"]

    # 冗余性：聚类（不直接淘汰，交合成器在簇内择优）
    clusters = []
    if corr_matrix is not None and not corr_matrix.empty:
        clusters = dedup_clusters(corr_matrix, threshold=corr_threshold)

    active = [n for n in health.index if health.loc[n, "status"] == "active"]
    log.info(f"因子过滤完成: 全池{len(health)} → active {len(active)}, watch "
             f"{(health['status'] == 'watch').sum()}, inactive {len(dropped)}")
    return FilterResult(health, active, dropped, clusters)


def factor_health_to_frame(health: pd.DataFrame) -> pd.DataFrame:
    """标准化 factor_health 输出（列顺序固定，供 warehouse 持久化）。"""
    cols = ["factor", "ic_mean", "icir", "ic_winrate", "ic_std", "n", "psi",
            "status", "reason"]
    out = health.reset_index().rename(columns={"index": "factor"})
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


# ── 新股/停牌复牌冷静期（V7.0 P2 修复）────────────────────
# 说明: 股票池/选股层的新股<60日排除在 strategy/multifactor_v7.py
#       （min_age_days=60，list_days 过滤）已存在并保留；这里补充
#       统一的新股掩码辅助 + 停牌>30交易日复牌后动量因子冷静期。

DEFAULT_MOMENTUM_PREFIXES = ("tech_mom", "tech_ret", "tech_cum_ret")
DEFAULT_MOMENTUM_NAMES = {"mom20", "mom1m", "mom3m", "mom6m", "mom12m",
                          "ma_cross", "macd_cross", "volume_trend"}


def new_stock_mask(list_days: pd.Series, min_age_days: int = 60) -> pd.Series:
    """新股排除掩码：True = 已上市 >= min_age_days 个交易日（可入池）。

    list_days: index=股票代码 → 上市以来交易日数（缺失按 1e9 视为老股）。
    与 multifactor_v7 的 list_days 过滤语义一致，供过滤/合成层复用。
    """
    ld = list_days.astype(float).fillna(10 ** 9)
    return ld >= min_age_days


def detect_suspension_periods(volume: pd.Series,
                              min_suspension_days: int = 30) -> pd.DataFrame:
    """识别停牌区间：成交量 <= 0 的连续交易日段。

    volume: index=交易日 → 当日成交量（0/NaN 表示无成交=停牌）。
    返回 DataFrame[start, end, days]，仅含连续停牌 >= min_suspension_days 的段。
    """
    if volume is None or len(volume) == 0:
        return pd.DataFrame(columns=["start", "end", "days"])
    # NaN 成交量视为无成交=停牌（NaN <= 0 为 False，需先 fillna(0)）
    z = (pd.to_numeric(volume, errors="coerce").fillna(0) <= 0).to_numpy()
    dates = pd.DatetimeIndex(volume.index)
    rows = []
    i, n = 0, len(z)
    while i < n:
        if not z[i]:
            i += 1
            continue
        j = i
        while j < n and z[j]:
            j += 1
        if j - i >= min_suspension_days:
            rows.append({"start": dates[i], "end": dates[j - 1],
                         "days": j - i})
        i = j
    if not rows:
        return pd.DataFrame(columns=["start", "end", "days"])
    return pd.DataFrame(rows)


def suspension_cooldown_mask(dates: pd.DatetimeIndex,
                             volume: pd.Series,
                             cooldown_days: int = 10,
                             min_suspension_days: int = 30) -> pd.Series:
    """停牌复牌冷静期掩码：True = 处于复牌后 cooldown_days 个交易日内。

    - 停牌定义: 成交量 <= 0 的连续交易日段（>= min_suspension_days 才算）
    - 复牌后从停牌段结束的下一个交易日起，连续 cooldown_days 个交易日为冷静期
    返回 bool Series(index=dates)。
    """
    n = len(dates)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or volume is None or len(volume) == 0:
        return pd.Series(mask, index=dates)
    vol = pd.to_numeric(volume, errors="coerce").reindex(dates)
    z = (vol <= 0).to_numpy()
    i, j = 0, 0
    while i < n:
        if not z[i]:
            i += 1
            continue
        j = i
        while j < n and z[j]:
            j += 1
        if j - i >= min_suspension_days:
            end = min(j + cooldown_days, n)
            mask[j:end] = True
        i = j
    return pd.Series(mask, index=dates)


def apply_suspension_cooldown(factor_wide: pd.DataFrame,
                              volume_wide: pd.DataFrame,
                              cooldown_days: int = 10,
                              min_suspension_days: int = 30) -> pd.DataFrame:
    """单因子面板冷静期处理：复牌后冷静期内因子值置 NaN（等价权重归零）。

    factor_wide / volume_wide: index=date, columns=股票代码。
    动量因子在停牌>30交易日复牌后的 10 个交易日内失去意义
    （动量被停牌扭曲），置 NaN 使其不参与横截面 IC 与合成。
    """
    if factor_wide is None or factor_wide.empty:
        return factor_wide
    out = factor_wide.copy()
    vw = volume_wide.reindex(index=out.index, columns=out.columns)
    for code in out.columns:
        cd = suspension_cooldown_mask(out.index, vw[code],
                                      cooldown_days=cooldown_days,
                                      min_suspension_days=min_suspension_days)
        if cd.any():
            out.loc[cd, code] = np.nan
    return out


def zero_momentum_after_suspension(factor_panels: dict,
                                   volume_wide: pd.DataFrame,
                                   momentum_factors: list | None = None,
                                   cooldown_days: int = 10,
                                   min_suspension_days: int = 30) -> dict:
    """多因子面板批量冷静期：仅对动量类因子生效（其余因子不动）。

    factor_panels: {因子名: DataFrame(index=date, columns=股票)}
    momentum_factors: 显式指定动量因子名单；None 时自动识别
      （前缀 tech_mom/tech_ret/tech_cum_ret 或默认动量因子集合）。
    返回处理后的面板 dict（未命中动量名单的面板原样返回）。
    """
    if not factor_panels:
        return factor_panels

    def _is_momentum(name: str) -> bool:
        return (name.startswith(DEFAULT_MOMENTUM_PREFIXES)
                or name in DEFAULT_MOMENTUM_NAMES)

    targets = momentum_factors
    if targets is None:
        targets = [n for n in factor_panels if _is_momentum(n)]
    out = dict(factor_panels)
    for name in targets:
        panel = factor_panels.get(name)
        if panel is None:
            continue
        out[name] = apply_suspension_cooldown(
            panel, volume_wide, cooldown_days=cooldown_days,
            min_suspension_days=min_suspension_days)
    return out
