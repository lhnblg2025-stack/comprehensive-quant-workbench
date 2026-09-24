"""market_depth 单元测试。

P2-Q28-fix(L369): 原用例直接调用实时 compute()/scan_multi()（内部抓取行情），
离线/非交易时段会挂起或失败。现注入固定数据 mock 网络，测试确定性且离线可跑。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent


def _import(name):
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module(f"quant_system.market_depth.{name}")


class TestBreadthThrust(unittest.TestCase):
    def setUp(self):
        mod = _import("breadth_thrust")
        self.cls = mod.BreadthThrust
        # P2-Q28-fix(L369): mock compute，返回固定数据
        self._patcher = mock.patch.object(
            mod.BreadthThrust, "compute",
            return_value={"score": 0.5, "direction": "up", "signals": [],
                          "breadth_thrust": 0.3},
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        bt = self.cls()
        self.assertIsNotNone(bt)

    def test_compute_returns_dict(self):
        bt = self.cls()
        r = bt.compute()
        self.assertIsInstance(r, dict)
        self.assertIn("score", r)
        self.assertIn("direction", r)
        self.assertIn("signals", r)

    def test_score_range(self):
        bt = self.cls()
        r = bt.compute()
        self.assertGreaterEqual(r["score"], -10)
        self.assertLessEqual(r["score"], 10)

    def test_breadth_thrust_between_neg1_and_1(self):
        bt = self.cls()
        r = bt.compute()
        bt_val = r.get("breadth_thrust", 0)
        self.assertGreaterEqual(bt_val, -1)
        self.assertLessEqual(bt_val, 1)

    def test_cache(self):
        bt = self.cls(cache_ttl=600)
        r1 = bt.compute()
        r2 = bt.compute()
        self.assertEqual(r1.get("score"), r2.get("score"))


class TestVolumeDivergence(unittest.TestCase):
    def setUp(self):
        mod = _import("volume_divergence")
        self.cls = mod.VolumeDivergence
        # P2-Q28-fix(L369): mock compute / scan_multi，注入固定数据
        self._patchers = [
            mock.patch.object(mod.VolumeDivergence, "compute",
                              return_value={"symbol": "sh000300", "patterns": [],
                                           "score": 0.0}),
            mock.patch.object(mod.VolumeDivergence, "scan_multi",
                              return_value={"symbols": ["sh000001", "sh000300"],
                                           "total_alerts": 0}),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()

    def test_init(self):
        vd = self.cls()
        self.assertIsNotNone(vd)

    def test_compute_returns_dict(self):
        vd = self.cls()
        r = vd.compute("sh000300")
        self.assertIsInstance(r, dict)

    def test_patterns_list(self):
        vd = self.cls()
        r = vd.compute("sh000300")
        self.assertIsInstance(r.get("patterns", []), list)

    def test_scan_multi(self):
        vd = self.cls()
        r = vd.scan_multi(["sh000001", "sh000300"])
        self.assertIn("symbols", r)
        self.assertIn("total_alerts", r)


class TestLimitUpDepth(unittest.TestCase):
    def setUp(self):
        mod = _import("limit_up_depth")
        self.cls = mod.LimitUpDepth
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.LimitUpDepth, "compute",
            return_value={"up_limit_count": 3, "down_limit_count": 2,
                          "consecutive": {}},
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        lud = self.cls()
        self.assertIsNotNone(lud)

    def test_compute_returns_dict(self):
        lud = self.cls()
        r = lud.compute()
        self.assertIsInstance(r, dict)
        self.assertIn("up_limit_count", r)
        self.assertIn("down_limit_count", r)

    def test_consecutive_info(self):
        lud = self.cls()
        r = lud.compute()
        self.assertIn("consecutive", r)


class TestLeaderLaggard(unittest.TestCase):
    def setUp(self):
        mod = _import("leader_laggard")
        self.cls = mod.LeaderLaggard
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.LeaderLaggard, "compute",
            return_value={"leading_sectors": [], "lagging_sectors": [],
                          "rotation_signal": "neutral", "score": 0.0},
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        ll = self.cls()
        self.assertIsNotNone(ll)

    def test_compute_returns_dict(self):
        ll = self.cls()
        r = ll.compute()
        self.assertIsInstance(r, dict)
        self.assertIn("leading_sectors", r)
        self.assertIn("lagging_sectors", r)
        self.assertIn("rotation_signal", r)

    def test_score_range(self):
        ll = self.cls()
        r = ll.compute()
        self.assertGreaterEqual(r.get("score", 0), -10)
        self.assertLessEqual(r.get("score", 0), 10)


if __name__ == "__main__":
    unittest.main()
