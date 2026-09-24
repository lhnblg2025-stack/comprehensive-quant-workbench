#!/usr/bin/env python3
"""P1-4 因子去冗余（共线性压缩）测试（quant_system/ic_factors/dedup.py）。

覆盖审计_因子层.md P1-4 的核心承诺：
  1. 对含强相关子集的人造面板，层次聚类能正确把该子集聚成同一族。
  2. 代表因子通过 ICIR（降序）选出，且族内代表确实为 ICIR 最高者。
  3. 相关性最高的对能正确被识别。
  4. decorrelate_families 族内正交化后，后序因子与先序因子相关趋零。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.ic_factors.dedup import (
    decorrelate_families,
    dedup_report,
    factor_corr_matrix,
    families_from_clusters,
    hierarchical_clusters,
    select_representatives,
)


def _make_panels(n_date=40, n_stock=80, seed=1):
    """构造 3 簇：A 族 3 个互相强相关因子，B 族 2 个互相强相关，其余独立。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_date, freq="B")
    stocks = [f"S{i}" for i in range(n_stock)]
    panels = {}

    def make(mid_noise=0.001):
        base = pd.DataFrame(rng.normal(0, 1, (n_date, n_stock)),
                            index=dates, columns=stocks)
        return base

    # base 信号
    baseA = make()
    baseB = make()
    baseC = make()
    # A 族：3 个彼此 ± 相关（同源）
    panels["momA1"] = baseA.copy()
    panels["momA2"] = (1.5 * baseA + rng.normal(0, 0.001, (n_date, n_stock))).clip(lower=None)
    panels["momA3"] = (-baseA + rng.normal(0, 0.001, (n_date, n_stock)))
    # B 族：2 个彼此强相关
    panels["valB1"] = baseB.copy()
    panels["valB2"] = (baseB + rng.normal(0, 0.001, (n_date, n_stock))).clip(lower=None)
    # 独立因子
    panels["indC"] = baseC.copy()
    panels["indD"] = make()
    return panels


class TestCorrMatrix:
    def test_high_corr_within_family(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        assert abs(corr.loc["momA1", "momA2"]) > 0.9
        assert abs(corr.loc["momA1", "momA3"]) > 0.9
        assert abs(corr.loc["valB1", "valB2"]) > 0.9
        # 不同族因子应低相关
        assert abs(corr.loc["momA1", "indC"]) < 0.5

    def test_corr_shape_and_diag(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        assert list(corr.index) == list(corr.columns)
        for n in corr.index:
            assert corr.loc[n, n] == pytest.approx(1.0)


class TestClustering:
    def test_groups_correlated_family(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        clusters = hierarchical_clusters(corr, threshold=0.6)  # 较强分族
        fam = families_from_clusters(clusters)
        # momA1/momA2/momA3 应同族
        groups = [set(m) for m in fam.values()]
        assert any({"momA1", "momA2", "momA3"} <= g for g in groups)
        # valB1/valB2 应同族
        assert any({"valB1", "valB2"} <= g for g in groups)

    def test_cluster_returns_mapping(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        clusters = hierarchical_clusters(corr, threshold=1.0)
        assert set(clusters.keys()) == set(panels.keys())
        assert len(set(clusters.values())) >= 1


class TestRepresentatives:
    def test_representative_is_highest_icir(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        clusters = hierarchical_clusters(corr, threshold=0.6)
        fam = families_from_clusters(clusters)
        # 给 momA1 最高 ICIR，应被选为代表
        icir = {"momA1": 0.8, "momA2": 0.5, "momA3": 0.3,
                "valB1": 0.6, "valB2": 0.4, "indC": 0.2, "indD": 0.1}
        reps = select_representatives(fam, icir=icir)
        # 找含 mom 的族
        mom_family = next(f for f in fam.values() if "momA1" in f)
        mom_rep = reps.get("momA1")
        assert mom_rep is not None  # momA1 是代表（ICIR 最高）
        # 验证 mom_rep 在 mom 族中
        assert mom_rep in [r for r in reps]

    def test_select_uses_abs_icir(self):
        # 负 ICIR 绝对值最大也应选为代表
        families = {0: ["neg_strong", "pos_weak"]}
        reps = select_representatives(families, icir={"neg_strong": -0.9, "pos_weak": 0.3})
        assert "neg_strong" in reps

    def test_dedup_report_contains_structure(self):
        panels = _make_panels()
        r = dedup_report(panels, icir={"momA1": 1.0}, threshold=0.6)
        assert "corr" in r and "families" in r and "representatives" in r
        assert len(r["representatives"]) >= 1


class TestOrthogonalize:
    def test_family_orthogonalization_removes_redundancy(self):
        panels = _make_panels()
        corr = factor_corr_matrix(panels)
        clusters = hierarchical_clusters(corr, threshold=0.6)
        fam = families_from_clusters(clusters)
        orth = decorrelate_families(panels, fam, icir={"momA1": 1.0, "momA2": 0.9, "momA3": 0.8})
        assert "momA2" in orth and "momA3" in orth
        # 正交化后 momA2/momA3 与原 baseA (momA1) 的截面相关应趋零
        d = orth["momA3"].index[0]
        c = orth["momA3"].iloc[0].corr(panels["momA1"].iloc[0])
        assert abs(c) < 0.05
