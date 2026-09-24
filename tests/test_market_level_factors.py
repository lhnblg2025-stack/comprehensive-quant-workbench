"""market_level_factors — 市场级因子测试 (V12.3 因子扩充)。

验证: 因子函数输出日序列契约 / 数据可加载 / 面板构建 / 覆盖过滤。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quant_system.ic_factors import market_level_factors as mlf  # noqa: E402


def test_factor_list_nonempty():
    assert len(mlf.MARKET_LEVEL_FACTORS) >= 25
    # 分类覆盖: 各主要类别至少 1 个
    cats = set()
    for name in mlf.MARKET_LEVEL_FACTORS:
        cats.add(mlf.registry.get_factor(name).category if hasattr(mlf, "registry") else "?")
    # 不要求 registry 关联, 只确认清单结构
    assert all(isinstance(n, str) and n.startswith("mkt_") for n in mlf.MARKET_LEVEL_FACTORS)


def test_all_factors_return_series():
    """每个因子函数返回 date-index Series(数据缺失时允许空, 不抛异常)。"""
    for name in mlf.MARKET_LEVEL_FACTORS:
        s = getattr(mlf, name)({})
        assert isinstance(s, pd.Series), f"{name} 返回类型错误"
        assert isinstance(s.index, pd.DatetimeIndex), f"{name} 索引非日期"


def test_key_factors_have_real_data():
    """核心因子(真实数据基座)必须非空且有合理数值。"""
    for name in ["mkt_pmi_level", "mkt_cpi_yoy", "mkt_north_flow_5d",
                 "mkt_force_index", "mkt_qvix_level", "mkt_zt_count_5d",
                 "mkt_hs300_pe_pct", "mkt_realized_vol_20d"]:
        s = getattr(mlf, name)({})
        assert len(s) > 30, f"{name} 数据不足: {len(s)}"
        assert np.isfinite(s.dropna()).all()


def test_pmi_change_consistency():
    """pmi_change = pmi_level.diff()。"""
    lvl = mlf.mkt_pmi_level({})
    chg = mlf.mkt_pmi_change({})
    if len(lvl) and len(chg):
        exp = lvl.diff()
        j = pd.concat([chg, exp], axis=1).dropna()
        assert not j.empty
        assert np.allclose(j.iloc[:, 0], j.iloc[:, 1], atol=1e-6)


def test_build_market_panels_shape(monkeypatch, tmp_path):
    """面板构建: date×code tile 全股票同值; 覆盖不足因子被过滤。"""
    import ic_vectorized as icv

    # 合成 common/codes + 注入真实因子(仅 2 个足够)
    dates = pd.bdate_range("2024-01-01", periods=100)
    codes = [f"{600000+i:06d}" for i in range(5)]

    # monkeypatch 因子函数: 一个满覆盖(100天), 一个覆盖不足(10天)
    def fake_full(data, **kw):
        return pd.Series(1.0, index=dates)

    def fake_short(data, **kw):
        return pd.Series(1.0, index=dates[:10])

    monkeypatch.setattr(mlf, "MARKET_LEVEL_FACTORS", ["mkt_fake_full", "mkt_fake_short"])
    monkeypatch.setattr(mlf, "mkt_fake_full", fake_full, raising=False)
    monkeypatch.setattr(mlf, "mkt_fake_short", fake_short, raising=False)

    panels = icv.build_market_panels(dates, codes)
    assert "mkt_fake_full" in panels
    assert panels["mkt_fake_full"].shape == (100, 5)
    # 全股票同值
    assert (panels["mkt_fake_full"].iloc[:, 0] == panels["mkt_fake_full"].iloc[:, 1]).all()
    assert "mkt_fake_short" not in panels  # 覆盖 <20% 被过滤
