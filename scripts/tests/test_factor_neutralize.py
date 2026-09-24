#!/usr/bin/env python3
"""P1-1 中性化/正交化测试（quant_system/ic_factors/neutralize + neutralize_inputs）。

覆盖审计_因子层.md P1-1 的核心承诺：
  1. neutralize() 逐日横截面回归取残差后，残差与已知行业暴露的相关性显著下降。
  2. neutralize 支持 市值 + 行业 双输入。
  3. orthogonalize() 按 order Gram-Schmidt 正交后，后序因子与先序因子相关性趋零。
  4. neutralize_inputs 诚实构建输入：可用性声明准确、缺失时如实降级（不伪造）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.ic_factors.neutralize import neutralize, orthogonalize
from quant_system.ic_factors.neutralize_inputs import (
    build_industry_map,
    build_market_cap,
    neutralize_input_summary,
)


def _make_industry_panel(n_date: int = 20, n_stock: int = 90,
                         seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """构造 2 行业强暴露因子面板 + industry_map。

    股票 S000..S089；S[00,30,60]=行业A，S[01,31,61]=行业B，其余行业C/D/E。
    因子值 = 5*indA + 10*indB + 噪声，使行业暴露极强（IC 被行业混淆）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_date, freq="B")
    stocks = [f"S{i:03d}" for i in range(n_stock)]
    industry = []
    for i in range(n_stock):
        if i % 30 == 0:
            industry.append("A")
        elif i % 31 == 0:
            industry.append("B")
        else:
            industry.append({0: "C", 1: "D", 2: "E"}[i % 3])
    industry_map = pd.Series(industry, index=stocks)

    rows = np.zeros((n_date, n_stock))
    for d in range(n_date):
        noise = rng.normal(0, 0.5, n_stock)
        f = np.array([5.0 if industry_map.iloc[i] == "A"
                      else 10.0 if industry_map.iloc[i] == "B" else 0.0
                      for i in range(n_stock)]) + noise
        rows[d] = f
    panel = pd.DataFrame(rows, index=dates, columns=stocks)
    return panel, industry_map


class TestNeutralizeIndustry:
    def test_residual_loses_industry_exposure(self):
        panel, industry_map = _make_industry_panel(n_stock=90)
        dummies = pd.get_dummies(industry_map)  # A/B/C/D/E 哑变量

        residual = neutralize(panel, industry_map=industry_map)

        assert residual.shape == panel.shape
        assert residual.notna().any().any()

        # 残差面板与行业哑变量的相关性，应对应显著下降
        raw_corrs = []
        neut_corrs = []
        for d in panel.index[:5]:
            raw = panel.loc[d].rank()
            res = residual.loc[d].rank()
            for col in dummies.columns:
                ind = dummies[col].astype(float)
                raw_corrs.append(abs(raw.corr(ind)))
                neut_corrs.append(abs(res.corr(ind)))
        assert np.mean(neut_corrs) < np.mean(raw_corrs)
        # 中性化后对行业暴露应基本剥离（相关均值显著 < 原值，且不大）
        assert np.mean(neut_corrs) < 0.3

    def test_residual_not_dominated_by_original(self):
        # 残差不等于原始（确有回归作用），且与原因子仍保有一定信息
        panel, industry_map = _make_industry_panel(seed=11)
        residual = neutralize(panel, industry_map=industry_map)
        assert not np.allclose(residual.values, panel.values)
        mean_corr = np.mean([panel.loc[d].corr(residual.loc[d])
                             for d in panel.index])
        assert 0.2 < mean_corr <= 1.0

    def test_industry_and_market_cap_combined(self):
        panel, industry_map = _make_industry_panel()
        # 构造 log 市值列，加入市值暴露
        rng = np.random.default_rng(3)
        mc = pd.DataFrame(
            np.abs(rng.normal(1e10, 1e9, (panel.shape[0], panel.shape[1]))),
            index=panel.index, columns=panel.columns, dtype=float)
        residual = neutralize(panel, industry_map=industry_map, market_cap=mc)
        assert residual.shape == panel.shape
        assert residual.notna().any().any()

    def test_missing_inputs_fall_back_gracefully(self):
        # 无行业/市值输入时，neutralize 不应崩溃（等价于不中性化或仅去截距）
        panel, _ = _make_industry_panel()
        out = neutralize(panel)  # industry_map/market_cap 均 None
        assert out.shape == panel.shape


class TestOrthogonalize:
    def test_orthogonalize_removes_redundancy(self):
        rng = np.random.default_rng(5)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        stocks = [f"S{i:03d}" for i in range(60)]
        base = pd.DataFrame(
            rng.normal(0, 1, (30, 60)), index=dates, columns=stocks)
        # 三个高度相关因子（同源冗余）
        f2 = base + rng.normal(0, 0.01, base.shape)  # 几乎 = base
        f3 = 2.0 * base + rng.normal(0, 0.01, base.shape)
        panels = {"f1": base, "f2": f2, "f3": f3}

        raw_corr = panels["f1"].iloc[0].corr(panels["f2"].iloc[0])
        assert raw_corr > 0.99  # 构造确认高度共线

        out = orthogonalize(panels, order=["f1", "f2", "f3"])

        # 正交化后，f2/f3 与先序因子的截面相关应趋零
        for later in ["f2", "f3"]:
            for prior in ["f1"] if later == "f2" else ["f1", "f2"]:
                c = out[later].iloc[0].corr(out[prior].iloc[0])
                assert abs(c) < 1e-6

    def test_orthogonalize_shape_preserved(self):
        panels, _ = self._stock_panels()
        out = orthogonalize(panels)
        for k in panels:
            assert out[k].shape == panels[k].shape

    def _stock_panels(self):
        rng = np.random.default_rng(9)
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        stocks = [f"S{i:03d}" for i in range(50)]
        return ({
            "a": pd.DataFrame(rng.normal(0, 1, (40, 50)), index=dates, columns=stocks),
            "b": pd.DataFrame(rng.normal(0, 1, (40, 50)), index=dates, columns=stocks),
        }, None)


class TestNeutralizeInputs:
    def test_summary_reflects_availability_honestly(self, tmp_path):
        # 行业表缺失 → 空 series，summary 明确标 industry_available=False
        fake = tmp_path / "nope.parquet"
        im = build_industry_map(["000001"], path=fake)
        assert im.empty
        assert neutralize_input_summary(im, pd.DataFrame()) == {
            "industry_available": False, "industry_count": 0,
            "market_cap_available": False, "market_cap_days": 0, "market_cap_stocks": 0,
        }

    def test_build_market_cap_uses_close_times_shares(self):
        kl = {"600000": pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "close": [10.0, 12.0],
            "outstanding_share": [1e8, 1e8],
        })}
        mc = build_market_cap(kl)
        assert mc.loc[pd.Timestamp("2024-01-01"), "600000"] == pytest.approx(1e9)
        assert mc.loc[pd.Timestamp("2024-01-02"), "600000"] == pytest.approx(1.2e9)

    def test_build_market_cap_missing_cols_returns_empty(self):
        # 股票缺 outstanding_share → 该股票跳过，最终为空（诚实降级）
        kl = {"600000": pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]})}
        mc = build_market_cap(kl)
        assert mc.empty
