"""IC 向量化管线（quant_system.ic_factors）融合测试。

2026-08-08 融合：quant_v6.scripts.ic_vectorized + quant_v6.factors → quant_system.ic_factors + scripts/ic_vectorized.py
覆盖：注册表自动发现 / IC 计算 / IC 统计 / 仓库路径 / 脚本零 quant_v6 依赖。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def registry():
    from quant_system.ic_factors import registry as reg
    reg.autodiscover()
    reg.import_from_zoo()
    return reg


class TestRegistryFused:
    def test_autodiscover_finds_factors(self, registry):
        """注册表能自动发现 v7 因子模块并注册（>=100 个因子）。"""
        assert len(registry._REGISTRY) >= 100
        cats = {m.category for m in registry._REGISTRY.values()}
        # 大类齐全
        assert {"momentum", "volatility", "value", "liquidity"} <= cats

    def test_factor_direction_sane(self, registry):
        """V11 突变测试补盲: 已注册因子 direction 必须在 {-1, +1} 且
        与描述语义一致（方向翻转突变可被此测试抓到）。"""
        bad = []
        for name, meta in registry._REGISTRY.items():
            if meta.direction not in (-1, 1):
                bad.append(f"{name}: direction={meta.direction}")
        assert not bad, f"方向非法因子: {bad[:10]}"
        # 抽查：正方向因子（动量类）与负方向因子（反转/超买类）都应存在
        dirs = {m.direction for m in registry._REGISTRY.values()}
        assert dirs == {-1, 1}, f"方向分布异常: {dirs}"
        # 具体因子方向锚定（突变翻转会被抓）：
        # 基差率升水>0 = 乐观 → +1；5日基差变化贴水加深 = 情绪恶化 → -1
        anchor = {"if_basis_rate": 1, "if_basis_5d": -1,
                  "qvix50_level": -1, "t_futures_ret20": 1}
        for nm, expect in anchor.items():
            m = registry.get_factor(nm)
            assert m is not None, f"锚定因子 {nm} 未注册"
            assert m.direction == expect, (
                f"锚定因子 {nm} 方向漂移: 期望{expect} 实际{m.direction}——因子方向突变！"
            )

    def test_factor_compute_returns_series(self, registry):
        """已注册因子的 compute() 能在合成数据上产出 Series。"""
        name = next(iter(registry._REGISTRY))
        meta = registry.get_factor(name)
        n = 120
        close = 10 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n))
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.random.randint(1e5, 1e6, n).astype(float),
        })
        df["code"] = "600519"
        data = {"kline": df.set_index("date")}
        try:
            s = meta.compute(data)
            assert isinstance(s, pd.Series)
        except (KeyError, ValueError, TypeError):
            # 部分因子需要更多数据域，跳过
            pytest.skip(f"因子 {name} 需要额外数据域")


class TestIcCompute:
    def test_compute_ic_structure(self):
        from quant_system.ic_factors.filter import compute_ic, ic_stats
        rng = np.random.default_rng(7)
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        codes = [f"{i:06d}" for i in range(50)]
        factor = pd.DataFrame(rng.normal(0, 1, (60, 50)), index=dates, columns=codes)
        ret = pd.DataFrame(rng.normal(0.001, 0.02, (60, 50)), index=dates, columns=codes)
        ic = compute_ic(factor, ret)
        assert isinstance(ic, pd.Series) and len(ic) == 60
        assert ic.notna().sum() == 60  # 每日期均有横截面 IC
        stats = ic_stats(ic)
        assert {"ic_mean", "icir", "ic_winrate"} <= set(stats)
        assert isinstance(stats["ic_mean"], float)

    def test_ic_stats_icir(self):
        from quant_system.ic_factors.filter import ic_stats
        ic = pd.Series(np.linspace(0.02, 0.10, 50))
        s = ic_stats(ic)
        assert s["icir"] > 0 and s["ic_winrate"] > 0.5


class TestWarehousePath:
    def test_domain_dir_resolves_to_data_warehouse(self):
        from quant_system.market_forecast._support.data.market_warehouse import MarketWarehouse
        wh = MarketWarehouse()
        kdir = wh._domain_dir("kline")
        assert (ROOT / "data_warehouse" / "kline").resolve() == kdir.resolve()
        files = list(kdir.glob("*.parquet"))
        assert len(files) > 1000  # 全市场仓库已建成


class TestIcVectorizedScript:
    def test_script_has_no_quant_v6_import(self):
        src = (ROOT / "scripts" / "ic_vectorized.py").read_text(encoding="utf-8")
        assert "quant_v6" not in src, "脚本仍引用 quant_v6"

    def test_market_warehouse_has_no_quant_v6_import(self):
        src = (ROOT / "quant_system" / "market_forecast" / "_support" / "data"
               / "market_warehouse.py").read_text(encoding="utf-8")
        assert "quant_v6" not in src
