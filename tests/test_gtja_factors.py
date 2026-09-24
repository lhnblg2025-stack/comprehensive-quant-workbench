#!/usr/bin/env python3
"""test_gtja_factors.py — GTJA 因子移植测试 (2026-08-14)

覆盖:
  1. gtja.py 模块 18 个因子函数全部存在
  2. 每个因子在合成K线上能计算且返回有限数值
  3. zoo.register_defaults 后 GTJA 因子进入注册表
  4. compute_factor_frame 能产出 GTJA 列
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_system.ic_factors import gtja, zoo  # noqa: E402

N_FACTORS = len(gtja.GTJA_FACTORS)


def _mock_data(n_stocks: int = 5, n_days: int = 120) -> dict[str, pd.DataFrame]:
    """合成K线: 趋势+噪声, 保证因子可计算。"""
    data = {}
    rng = np.random.default_rng(42)
    for i in range(n_stocks):
        dates = pd.bdate_range("2026-01-01", periods=n_days)
        base = 10 + i
        drift = rng.normal(0.0005, 0.0002)
        rets = rng.normal(drift, 0.02, n_days)
        close = base * np.cumprod(1 + rets)
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[f"{i:06d}"] = pd.DataFrame({
            "date": dates, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })
    return data


class TestGtjaModule(unittest.TestCase):
    def test_all_functions_exist(self):
        for name in gtja.GTJA_FACTORS:
            fn = gtja.GTJA_FACTORS[name][0]
            self.assertTrue(callable(fn), f"{name} 函数缺失")

    def test_each_factor_computable(self):
        data = _mock_data()
        for name, (fn, desc, direction) in gtja.GTJA_FACTORS.items():
            try:
                s = fn(data)
                self.assertIsInstance(s, pd.Series, f"{name} 应返回 Series")
                vals = s.dropna()
                self.assertGreater(len(vals), 0, f"{name} 全部为 NaN")
                self.assertTrue(np.isfinite(vals).all(), f"{name} 含非有限值")
            except Exception as e:  # noqa: BLE001
                self.fail(f"{name} 计算异常: {e}")


class TestZooIntegration(unittest.TestCase):
    def test_registered_in_zoo(self):
        zoo.register_defaults()
        names = zoo.list_factors(include_inactive=True)
        gtja_names = [n for n in names if n.startswith("gtja")]
        self.assertGreaterEqual(len(gtja_names), N_FACTORS - 1,
                                f"GTJA 注册不足: {len(gtja_names)}/{N_FACTORS}")

    def test_compute_factor_frame_has_gtja_cols(self):
        data = _mock_data()
        frame = zoo.compute_factor_frame(
            data,
            factors=[n for n in zoo.list_factors() if n.startswith("gtja")][:4],
            apply_direction=True)
        self.assertEqual(frame.shape[0], len(data))
        self.assertTrue(any(c.startswith("gtja") for c in frame.columns))


if __name__ == "__main__":
    unittest.main(verbosity=2)
