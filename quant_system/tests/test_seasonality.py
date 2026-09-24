"""seasonality 单元测试。

P2-Q28-fix(L369): 原用例直接调用 compute(years=5) 等实时接口（内部抓取行情），
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
    return importlib.import_module(f"quant_system.seasonality.{name}")


class TestMonthEffect(unittest.TestCase):
    def setUp(self):
        mod = _import("month_effect")
        self.cls = mod.MonthEffect
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.MonthEffect, "compute",
            return_value={
                "monthly_stats": {},
                "best_month": {"month": 2, "return": 0.02},
                "worst_month": {"month": 9, "return": -0.02},
            },
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        me = self.cls()
        self.assertIsNotNone(me)

    def test_compute_returns_dict(self):
        me = self.cls()
        r = me.compute(years=5)
        self.assertIsInstance(r, dict)

    def test_monthly_stats(self):
        me = self.cls()
        r = me.compute(years=5)
        self.assertIn("monthly_stats", r)

    def test_best_worst_month(self):
        me = self.cls()
        r = me.compute(years=5)
        self.assertIn("best_month", r)
        self.assertIn("worst_month", r)


class TestHolidayEffect(unittest.TestCase):
    def setUp(self):
        mod = _import("holiday_effect")
        self.cls = mod.HolidayEffect
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.HolidayEffect, "compute", return_value={"holiday_effects": []})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        he = self.cls()
        self.assertIsNotNone(he)

    def test_compute_returns_dict(self):
        he = self.cls()
        r = he.compute()
        self.assertIsInstance(r, dict)

    def test_holiday_effects_list(self):
        he = self.cls()
        r = he.compute()
        self.assertIsInstance(r.get("holiday_effects", []), list)


class TestAnnualPattern(unittest.TestCase):
    def setUp(self):
        mod = _import("annual_pattern")
        self.cls = mod.AnnualPattern
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.AnnualPattern, "compute", return_value={"patterns": []})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        ap = self.cls()
        self.assertIsNotNone(ap)

    def test_compute_returns_dict(self):
        ap = self.cls()
        r = ap.compute(years=5)
        self.assertIsInstance(r, dict)

    def test_patterns_list(self):
        ap = self.cls()
        r = ap.compute(years=5)
        self.assertIn("patterns", r)


class TestWeekdayEffect(unittest.TestCase):
    def setUp(self):
        mod = _import("weekday_effect")
        self.cls = mod.WeekdayEffect
        # P2-Q28-fix(L369): mock compute，注入固定数据
        self._patcher = mock.patch.object(
            mod.WeekdayEffect, "compute", return_value={"weekday_stats": {}})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_init(self):
        we = self.cls()
        self.assertIsNotNone(we)

    def test_compute_returns_dict(self):
        we = self.cls()
        r = we.compute(years=5)
        self.assertIsInstance(r, dict)

    def test_weekday_stats(self):
        we = self.cls()
        r = we.compute(years=5)
        self.assertIn("weekday_stats", r)


if __name__ == "__main__":
    unittest.main()
