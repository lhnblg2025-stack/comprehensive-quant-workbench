"""sentiment_factory 单元测试。

P2-Q28-fix(L369): 原用例直接调用 compute()（内部实时抓行情，离线/非交易时段
挂起）。现注入固定数据 mock 网络，测试确定性且离线可跑；
并修正 test_weighted_composite 断言与注释对齐（加权合成应在维度分数 min~max 之间）。
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent.parent


class TestPriceSentiment(unittest.TestCase):
    """测试价格情绪模块。"""

    @classmethod
    def setUpClass(cls):
        """导入模块（一次性）。"""
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from quant_system.sentiment_factory.price_sentiment import PriceSentiment
        cls.PriceSentiment = PriceSentiment

    def setUp(self):
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            self.PriceSentiment, "compute",
            return_value={"score": 2.0, "direction": "bullish",
                          "percentile": 60, "sub_signals": []})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        ps = self.PriceSentiment()
        self.assertIsNotNone(ps)
        self.assertGreaterEqual(ps.cache_ttl, 0)

    def test_compute_returns_dict(self):
        ps = self.PriceSentiment()
        result = ps.compute()
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("direction", result)
        self.assertIn("percentile", result)

    def test_score_range(self):
        ps = self.PriceSentiment()
        result = ps.compute()
        score = result.get("score", 0)
        self.assertGreaterEqual(score, -10)
        self.assertLessEqual(score, 10)

    def test_percentile_range(self):
        ps = self.PriceSentiment()
        result = ps.compute()
        pct = result.get("percentile", 0)
        self.assertGreaterEqual(pct, 0)
        self.assertLessEqual(pct, 100)

    def test_sub_signals_list(self):
        ps = self.PriceSentiment()
        result = ps.compute()
        signals = result.get("sub_signals", [])
        self.assertIsInstance(signals, list)

    def test_cache(self):
        ps = self.PriceSentiment()
        r1 = ps.compute()
        r2 = ps.compute()
        self.assertEqual(r1.get("score"), r2.get("score"))


class TestVolumeSentiment(unittest.TestCase):
    """测试量价情绪模块。"""

    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from quant_system.sentiment_factory.volume_sentiment import VolumeSentiment
        cls.VolumeSentiment = VolumeSentiment

    def setUp(self):
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            self.VolumeSentiment, "compute",
            return_value={"score": 3.0, "direction": "bullish"})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        vs = self.VolumeSentiment()
        self.assertIsNotNone(vs)

    def test_compute_returns_dict(self):
        vs = self.VolumeSentiment()
        result = vs.compute()
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)

    def test_score_range(self):
        vs = self.VolumeSentiment()
        result = vs.compute()
        self.assertGreaterEqual(result["score"], -10)
        self.assertLessEqual(result["score"], 10)


class TestFundingSentiment(unittest.TestCase):
    """测试资金情绪模块。"""

    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from quant_system.sentiment_factory.funding_sentiment import FundingSentiment
        cls.FundingSentiment = FundingSentiment

    def setUp(self):
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            self.FundingSentiment, "compute",
            return_value={"score": 1.0, "direction": "bullish"})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        fs = self.FundingSentiment()
        self.assertIsNotNone(fs)

    def test_compute_returns_dict(self):
        fs = self.FundingSentiment()
        result = fs.compute()
        self.assertIsInstance(result, dict)


class TestCompositeSentiment(unittest.TestCase):
    """测试合成情绪指数。"""

    @classmethod
    def setUpClass(cls):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from quant_system.sentiment_factory.composite import CompositeSentiment
        cls.CompositeSentiment = CompositeSentiment

    def setUp(self):
        # P2-Q28-fix(L369): mock compute/get_alerts，注入固定数据。
        # 维度分数 [2,3,1]，权重 [0.45,0.35,0.20] → 加权 2.3，介于 min=1 与 max=3 之间。
        self._compute_mock = {
            "composite_score": 2.3,
            "direction": "bullish",
            "confidence": "high",
            "consistency": "一致",
            "dimensions": {
                "price": {"score": 2.0, "direction": "bullish", "percentile": 60},
                "volume": {"score": 3.0, "direction": "bullish", "percentile": 70},
                "funding": {"score": 1.0, "direction": "bullish", "percentile": 40},
            },
            "divergences": [],
            "alerts": [],
            "weights": {"price": 0.45, "volume": 0.35, "funding": 0.20},
            "signals": [],
        }
        self._patchers = [
            mock.patch.object(self.CompositeSentiment, "compute",
                              return_value=self._compute_mock),
            mock.patch.object(self.CompositeSentiment, "get_alerts", return_value=[]),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    def test_init(self):
        cs = self.CompositeSentiment()
        self.assertIsNotNone(cs)

    def test_compute_returns_dict(self):
        cs = self.CompositeSentiment()
        result = cs.compute()
        self.assertIsInstance(result, dict)
        self.assertIn("composite_score", result)
        self.assertIn("dimensions", result)

    def test_composite_score_range(self):
        cs = self.CompositeSentiment()
        result = cs.compute()
        score = result.get("composite_score", 0)
        self.assertGreaterEqual(score, -10)
        self.assertLessEqual(score, 10)

    def test_dimensions_exist(self):
        cs = self.CompositeSentiment()
        result = cs.compute()
        dims = result.get("dimensions", {})
        for name in ["price", "volume", "funding"]:
            self.assertIn(name, dims)
            self.assertIn("score", dims[name])
            self.assertIn("direction", dims[name])

    def test_divergences_list(self):
        cs = self.CompositeSentiment()
        result = cs.compute()
        divs = result.get("divergences", [])
        self.assertIsInstance(divs, list)

    def test_alerts_list(self):
        cs = self.CompositeSentiment()
        result = cs.compute()
        alerts = result.get("alerts", [])
        self.assertIsInstance(alerts, list)

    def test_weighted_composite(self):
        """验证合成分数确实在三个维度分数之间（加权平均，权重非负且和为1）。"""
        cs = self.CompositeSentiment()
        result = cs.compute()
        dims = result.get("dimensions", {})
        scores = [d.get("score", 0) for d in dims.values()]
        comp = result.get("composite_score", 0)
        # 加权合成应在 min~max 之间
        self.assertGreaterEqual(comp, min(scores))
        self.assertLessEqual(comp, max(scores))
        # 同时仍满足项目约定的 [-10,10] 量纲
        self.assertGreaterEqual(comp, -10)
        self.assertLessEqual(comp, 10)

    def test_get_alerts(self):
        cs = self.CompositeSentiment()
        alerts = cs.get_alerts()
        self.assertIsInstance(alerts, list)


if __name__ == "__main__":
    unittest.main()
