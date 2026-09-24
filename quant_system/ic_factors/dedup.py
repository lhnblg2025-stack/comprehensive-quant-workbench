"""
dedup.py — P1-4 因子去冗余（相关聚类分族 + 代表因子 + 正交化副本）
==================================================================
审计_因子层.md P1-4：281 注册因子中大量同源变体（mom20/1m/3m/6m/12m、
tech_mom_*、gtja_014/018/020 全是动量、多 KDJ/CCI/RSI/boll 变体）天然高共线。
`filter.dedup_clusters` 只"标注"不淘汰，`composite.dedup_correlated` 仅合成末步
>0.95 去重。本模块提供**去冗余/缩并**报告工具：

  - 对因子面板矩阵算相关矩阵 → 层次聚类分族；
  - 每族内按 ICIR 降序选出**代表因子**（另可按族内排名产出代表清单）；
  - 族内因子可做 `orthogonalize`（Gram-Schmidt）产出正交化副本供复合层选用。

重要：这是**报告/分析工具**，不改变既有因子集兼容性、不破坏下游——
不删注册表里的因子，只产出缩并建议 / 正交化面板。

D4收敛登记: 因子去冗余/正交化独特保留；本模块为其相关聚类与代表因子分析补充。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from quant_system.market_forecast._support.common.logger import get_logger

log = get_logger("qv6.factor.dedup")


def factor_corr_matrix(panels: dict[str, pd.DataFrame],
                       min_common_dates: int = 20) -> pd.DataFrame:
    """因子间平均截面 Spearman 相关矩阵。

    对每对因子，在共同有效日期上逐日做横截面秩相关（FactorModel 语义），
    再对日期取均值。日期交集 < min_common_dates 时该对记为 NaN。

    向量化：对每个因子面板只在共同日期上一次性 rank（秩标度一致），
    再按日期遍历一次向量化算所有因子对的秩相关，避免逐日期逐对的
    pandas `.rank().corr()` 开销。

    panels: {因子名: DataFrame(index=date, columns=股票代码)}
    返回 DataFrame(index/columns=因子名)。
    """
    names = [n for n, p in panels.items() if p is not None and not p.empty]
    if not names:
        return pd.DataFrame(index=[], columns=[])
    # 全体共同日期（宽裕交集；个别因子缺整日在此容忍）
    common = panels[names[0]].index
    for n in names[1:]:
        common = common.intersection(panels[n].index)
        if len(common) < min_common_dates:
            break
    # 逐因子算横截面 rank（按行 rank）
    ranks = {}
    for n in names:
        sub = panels[n].reindex(common)
        # 全 NaN 行 rank 会得到 NaN → 用有效股票掩码在 rank 后置 NaN
        ranks[n] = sub.rank(axis=1)
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        corr.loc[a, a] = 1.0
        Ra = ranks[a].to_numpy(dtype=float)          # (n_dates, n_stocks)
        validA = ~np.isnan(Ra)
        for j in range(i + 1, len(names)):
            b = names[j]
            Rb = ranks[b].to_numpy(dtype=float)
            m = validA & ~np.isnan(Rb)               # 两因子同日均有效
            n_valid = m.sum(axis=1)
            if (n_valid < min_common_dates).all():
                corr.loc[a, b] = corr.loc[b, a] = np.nan
                continue
            # 有日期的掩码按行置 NaN，再按行中心化（row-wise 有效子集）
            Fa = np.where(m, Ra, np.nan)
            Fb = np.where(m, Rb, np.nan)
            with np.errstate(invalid="ignore"):
                Fa_c = Fa - np.nanmean(Fa, axis=1, keepdims=True)
                Fb_c = Fb - np.nanmean(Fb, axis=1, keepdims=True)
                num = np.nansum(Fa_c * Fb_c, axis=1)
                den = np.sqrt(np.nansum(Fa_c * Fa_c, axis=1)
                              * np.nansum(Fb_c * Fb_c, axis=1))
                corr_d = np.where((n_valid >= 20) & (den > 1e-12),
                                  num / np.where(den > 1e-12, den, 1),
                                  np.nan)
            good = corr_d[np.isfinite(corr_d)]
            corr.loc[a, b] = corr.loc[b, a] = float(np.mean(good)) if good.size else np.nan
    return corr


def hierarchical_clusters(corr: pd.DataFrame,
                          metric: str = "distance",
                          threshold: float = 1.0) -> dict[str, int]:
    """层次聚类分族，返回 {因子: 族ID}。

    距离 = 1 - |corr|，做 ward 层次聚类，再按 `threshold` 截断分族。
    corr 为 NaN 的边按距离 1（完全不相关）处理。
    返回 dict 因子名 → 整数族 ID（0..n-1）。
    threshold: 归一化距离阈值（1 - |corr|），越小分族越细。
    """
    names = list(corr.index)
    n = len(names)
    if n == 0:
        return {}
    dist = np.array(corr, dtype=float)
    dist = 1.0 - np.abs(dist)
    np.fill_diagonal(dist, 0.0)
    dist = np.nan_to_num(dist, nan=1.0)
    sym = (dist + dist.T) / 2.0
    # 距离阵需对称、下三角为主；转入紧凑距离
    condensed = squareform(sym)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=threshold, criterion="distance")
    # labels 从 1 起，重排为 0 起连续
    order = {}
    counter = 0
    for lab in labels:
        if lab not in order:
            order[lab] = counter
            counter += 1
    return {name: order[int(lab)] for name, lab in zip(names, labels)}


def families_from_clusters(clusters: dict[str, int]) -> dict[int, list[str]]:
    """把 {因子:族ID} 转为 {族ID: [因子...]}，天然邻近排序。"""
    fam: dict[int, list[str]] = {}
    for name, fid in sorted(clusters.items()):
        fam.setdefault(fid, []).append(name)
    return fam


def select_representatives(families: dict[int, list[str]],
                           icir: dict[str, float] | None = None,
                           prefer: list[str] | None = None,
                           max_per_family: int = 1,
                           use_abs=True) -> dict[str, str]:
    """每族选代表因子。返回 {代表因子: 所属族因子列表的键}。

    选法：族内按 ICIR 降序（NaN 放后），选前 max_per_family 个为代表；
    同 ICIR 时 prefer 顺序优先（否则按字典序稳定）。
    返回 {代表因子名: 族首因子名}（族首=该族字典序最小，用于分组显示）。
    """
    repre: dict[str, str] = {}
    for fid, members in sorted(families.items()):
        keyfn = lambda m: (  # noqa: E731
            -abs(icir.get(m, np.nan)) if (icir and icir.get(m) is not None and np.isfinite(icir.get(m)))
            else 0.0)
        ranked = sorted(members, key=lambda m: (keyfn(m), prefer.index(m) if prefer and m in prefer else 999, m))
        for rep in ranked[:max_per_family]:
            repre[rep] = members[0]
    return repre


def decorrelate_families(panels: dict[str, pd.DataFrame],
                         families: dict[int, list[str]],
                         prefer_order: list[str] | None = None,
                         max_per_family: int = 1,
                         icir: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    """对每个多因子族做 Gram-Schmidt 正交化，产出正交化副本面板。

    返回 {因子名: DataFrame} 正交化后的面板集（仅含被正交的因子）。
    单元素族不平滑，直接保留原面板。
    """
    from quant_system.ic_factors.neutralize import orthogonalize
    out: dict[str, pd.DataFrame] = {}
    for fid, members in families.items():
        if len(members) <= 1:
            continue
        # 按 ICIR 降序排 order（代表优先），保证正交化顺序有意义
        icir = icir or {}
        order = sorted(members, key=lambda m: (
            -abs(icir.get(m, np.nan)) if icir.get(m) is not None and np.isfinite(icir.get(m)) else 0.0,
            prefer_order.index(m) if prefer_order and m in prefer_order else 999, m))
        sub = {m: panels[m] for m in members if m in panels}
        orth = orthogonalize(sub, order=order)
        out.update(orth)
    return out


def dedup_report(panels: dict[str, pd.DataFrame],
                 icir: dict[str, float] | None = None,
                 threshold: float = 1.0,
                 prefer_order: list[str] | None = None,
                 max_per_family: int = 1) -> dict:
    """完整去冗余报告。返回 dict:
        - corr: 相关矩阵
        - families: {族ID: [因子]}
        - representatives: {代表因子: 族首因子名}
        - summary: 各族统计
    """
    corr = factor_corr_matrix(panels)
    clusters = hierarchical_clusters(corr, threshold=threshold)
    families = families_from_clusters(clusters)
    representatives = select_representatives(families, icir, prefer_order, max_per_family)
    summary = []
    for fid, members in sorted(families.items()):
        summary.append({
            "family_id": fid, "size": len(members),
            "members": members,
            "representative": next((m for m in members if m in representatives), None),
        })
    return {
        "corr": corr,
        "clusters": clusters,
        "families": families,
        "representatives": representatives,
        "summary": summary,
    }
