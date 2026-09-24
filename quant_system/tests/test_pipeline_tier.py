"""analysis_core/pipeline_tier 冷热分离注册表/查询工具单元测试。

覆盖:
  - 注册/覆盖/归一化
  - tier_of / is_hot 查询
  - 未知模块默认值
  - hot/warm/cold 清单输出
  - cold_schedule 调度建议结构
  - report 汇总结构
  - 非法 tier 拒绝

无网络，纯注册表查询，全部离线。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.pipeline_tier as tier
    return tier


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tier = _import()

    def test_register_module_normalizes_name(self):
        tier = self.tier
        registry = {}
        key = tier.register_module("Battle-Map", "hot", registry=registry)
        self.assertEqual(key, "battle_map")
        self.assertEqual(registry[key], "hot")

    def test_register_module_updates_default_registry(self):
        tier = self.tier
        old = tier.MODULE_TIERS.copy()
        self.addCleanup(lambda: (tier.MODULE_TIERS.clear(), tier.MODULE_TIERS.update(old)))
        key = tier.register_module("test_cold_dummy", "cold")
        self.assertEqual(tier.tier_of(key), "cold")

    def test_invalid_tier_rejected(self):
        tier = self.tier
        with self.assertRaises(ValueError):
            tier.register_module("x", "urgent")


class TestQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tier = _import()

    def test_tier_of_known_module(self):
        self.assertEqual(self.tier.tier_of("fusion"), "hot")
        self.assertEqual(self.tier.tier_of("macro_system"), "warm")
        self.assertEqual(self.tier.tier_of("announcement_arbitrage"), "cold")

    def test_unknown_module_defaults_to_cold(self):
        self.assertEqual(self.tier.tier_of("not_implemented_yet"), "cold")

    def test_is_hot_known_and_unknown(self):
        self.assertTrue(self.tier.is_hot("battle_map"))
        self.assertFalse(self.tier.is_hot("macro_system"))
        self.assertFalse(self.tier.is_hot("unknown_cold_dummy"))


class TestListsAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tier = _import()

    def test_hot_list_contains_core_and_excludes_cold(self):
        tier = self.tier
        hot = tier.hot_modules()
        self.assertIn("fusion", hot)
        self.assertIn("battle_map", hot)
        self.assertNotIn("announcement_arbitrage", hot)

    def test_cold_list_contains_on_demand_modules_and_is_sorted(self):
        tier = self.tier
        cold = tier.cold_modules()
        self.assertIn("announcement_arbitrage", cold)
        self.assertIn("concept_lifecycle", cold)
        self.assertIn("seasonality", cold)
        self.assertNotIn("fusion", cold)
        self.assertEqual(cold, sorted(cold))

    def test_cold_schedule_structure(self):
        schedule = self.tier.cold_schedule()
        self.assertIn("cold_modules", schedule)
        self.assertEqual(schedule["trigger_policy"], "on_demand")
        self.assertEqual(schedule["count"], len(schedule["cold_modules"]))

    def test_report_counts_sum_to_registry_size(self):
        tier = self.tier
        report = tier.report()
        self.assertEqual(set(report["counts"]), {"hot", "warm", "cold"})
        self.assertEqual(
            sum(report["counts"].values()),
            len(tier.MODULE_TIERS),
        )
        self.assertIn("modules_by_tier", report)


if __name__ == "__main__":
    unittest.main()
