"""integrations 适配器单元测试。

P2-Q28-fix(L369): 本文件用例均基于合成 DataFrame（不依赖实时行情），
补固定随机种子保证可复现。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


class TestAlphalensAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        np.random.seed(42)  # P2-Q28-fix(L369): 确定性
        from quant_system.integrations.alphalens_adapter import AlphalensFactorAnalysis
        cls.AlphalensFactorAnalysis = AlphalensFactorAnalysis

    def test_init(self):
        afa = self.AlphalensFactorAnalysis()
        self.assertIsNotNone(afa)

    def test_available_flag(self):
        afa = self.AlphalensFactorAnalysis()
        self.assertIn(afa.available, (True, False))

    def test_compute_ic_with_simulated_data(self):
        afa = self.AlphalensFactorAnalysis()
        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        codes = ["000001", "000002"]
        factor = pd.DataFrame(np.random.randn(100, 2), index=dates, columns=codes)
        prices = pd.DataFrame(100 + np.cumsum(np.random.randn(100, 2), axis=0),
                              index=dates, columns=codes)
        result = afa.compute_ic(factor, prices)
        self.assertIsInstance(result, dict)

    def test_compute_all(self):
        afa = self.AlphalensFactorAnalysis()
        dates = pd.date_range("2025-01-01", periods=100, freq="B")
        codes = ["000001", "000002", "000003"]
        factor = pd.DataFrame(np.random.randn(100, 3), index=dates, columns=codes)
        prices = pd.DataFrame(100 + np.cumsum(np.random.randn(100, 3), axis=0),
                              index=dates, columns=codes)
        result = afa.compute_all(factor, prices)
        self.assertIn("ic", result)
        self.assertIn("quantile", result)
        self.assertIn("turnover", result)

    def test_compute_factor_turnover(self):
        afa = self.AlphalensFactorAnalysis()
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        codes = ["000001", "000002", "000003", "000004", "000005"]
        factor = pd.DataFrame(np.random.randn(50, 5), index=dates, columns=codes)
        result = afa.compute_factor_turnover(factor)
        self.assertIsInstance(result, dict)


class TestRiskfolioAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        np.random.seed(42)  # P2-Q28-fix(L369): 确定性
        from quant_system.integrations.riskfolio_adapter import RiskfolioOptimizer
        cls.RiskfolioOptimizer = RiskfolioOptimizer

    def test_init(self):
        ro = self.RiskfolioOptimizer()
        self.assertIsNotNone(ro)

    def test_hrp_with_simulated_data(self):
        ro = self.RiskfolioOptimizer()
        dates = pd.date_range("2025-01-01", periods=500, freq="B")
        codes = ["600519", "000858", "300750"]
        returns = pd.DataFrame(np.random.randn(500, 3) * 0.02,
                               index=dates, columns=codes)
        result = ro.hrp(returns)
        self.assertIsInstance(result, dict)
        self.assertIn("weights", result)

    def test_mean_variance(self):
        ro = self.RiskfolioOptimizer()
        dates = pd.date_range("2025-01-01", periods=500, freq="B")
        codes = ["600519", "000858"]
        returns = pd.DataFrame(np.random.randn(500, 2) * 0.02,
                               index=dates, columns=codes)
        result = ro.mean_variance(returns)
        self.assertIn("weights", result)

    def test_risk_parity(self):
        ro = self.RiskfolioOptimizer()
        dates = pd.date_range("2025-01-01", periods=500, freq="B")
        codes = ["600519", "000858", "300750", "601318"]
        returns = pd.DataFrame(np.random.randn(500, 4) * 0.02,
                               index=dates, columns=codes)
        result = ro.risk_parity(returns)
        self.assertIn("weights", result)


class TestPyfolioAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        np.random.seed(42)  # P2-Q28-fix(L369): 确定性
        from quant_system.integrations.pyfolio_adapter import PyfolioAnalyzer
        cls.PyfolioAnalyzer = PyfolioAnalyzer

    def test_init(self):
        pa = self.PyfolioAnalyzer()
        self.assertIsNotNone(pa)

    def test_tear_sheet(self):
        pa = self.PyfolioAnalyzer()
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.015 + 0.0005, index=dates)
        result = pa.tear_sheet(returns)
        self.assertIsInstance(result, dict)
        self.assertIn("sharpe", result)
        self.assertIn("max_drawdown", result)

    def test_drawdown_analysis(self):
        pa = self.PyfolioAnalyzer()
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.015, index=dates)
        result = pa.drawdown_analysis(returns)
        self.assertIn("max_drawdown", result)
        self.assertIn("peak_date", result)

    def test_benchmark_comparison(self):
        pa = self.PyfolioAnalyzer()
        dates = pd.date_range("2025-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.015, index=dates)
        bm = pd.Series(np.random.randn(252) * 0.012, index=dates)
        result = pa.benchmark_comparison(returns, bm)
        self.assertIn("tracking_error", result)
        self.assertIn("information_ratio", result)

    def test_rolling_stats(self):
        pa = self.PyfolioAnalyzer()
        dates = pd.date_range("2025-01-01", periods=500, freq="B")
        returns = pd.Series(np.random.randn(500) * 0.015, index=dates)
        result = pa.rolling_stats(returns, window=60)
        self.assertIn("rolling_sharpe_last", result)


if __name__ == "__main__":
    unittest.main()
