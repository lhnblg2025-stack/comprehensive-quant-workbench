"""analysis_core/context_router 场景路由单元测试。

覆盖:
  - data_sources 注册表 scope 字段
  - 风格象限→数据源集合正确、scope 过滤、非 global 源只在命中场景加载
  - 降级：无象限→默认全 global
  - api_enabled=False 的源不参与加载
  - load_for_context 返回结构、日期、确定性排序

无网络：使用默认注册表或自定义 registry，全部离线。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.context_router as cr
    return cr


def _import_ds():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.data_sources as ds
    return ds


class TestRegistryScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import_ds()

    def test_all_source_status_have_scope_field(self):
        for name, info in self.ds.SOURCE_STATUS.items():
            self.assertIn("scope", info, f"{name} 缺 scope 字段")

    def test_registry_scopes_are_lists_or_strings(self):
        for name, info in self.ds.SOURCE_STATUS.items():
            scope = info["scope"]
            self.assertTrue(
                isinstance(scope, (str, list, tuple, set)),
                f"{name} scope 类型异常: {type(scope)}",
            )

    def test_industry_car_has_chain_auto_scope(self):
        scopes = self.ds.SOURCE_STATUS["industry_car"]["scope"]
        normalized = scopes if isinstance(scopes, list) else [scopes]
        self.assertIn("chain_auto", normalized)


class TestContextRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cr = _import()

    def test_no_context_defaults_to_global_only(self):
        out = self.cr.load_for_context(context=None)
        self.assertEqual(out["context"], "default_global")
        self.assertTrue(all(row["scope"] == ["global"] for row in out["sources"]))
        self.assertIn("global", out["scopes"])
        self.assertEqual(out["source_names"], sorted(out["source_names"]))

    def test_context_output_contains_date(self):
        out = self.cr.load_for_context(context="消费", date="2026-08-13")
        self.assertEqual(out["date"], "2026-08-13")
        self.assertIn("source_names", out)
        self.assertIn("sources", out)

    def test_small_cap_loads_industry_car(self):
        names = self.cr.route_sources(context="小盘")
        self.assertIn("industry_car", names)
        self.assertIn("eastmoney", names)

    def test_large_cap_does_not_load_chain_auto_special_source(self):
        names = self.cr.route_sources(context="大盘")
        self.assertNotIn("industry_car", names)
        self.assertIn("eastmoney", names)

    def test_disabled_api_sources_are_excluded(self):
        names = self.cr.route_sources(context="小盘")
        self.assertNotIn("tushare", names)
        self.assertNotIn("ifind", names)

    def test_custom_registry_scope_filter(self):
        cr = self.cr
        registry = {
            "g1": {"desc": "global 1", "status": "active", "scope": ["global"]},
            "g2": {"desc": "global 2", "status": "active", "scope": ["global"]},
            "auto": {"desc": "汽车", "status": "active", "scope": ["chain_auto"]},
            "estate": {"desc": "地产", "status": "active", "scope": ["chain_realestate"]},
            "off": {"desc": "禁用", "status": "partial", "api_enabled": False,
                     "scope": ["global", "chain_auto"]},
        }
        names = cr.route_sources(context="小盘", registry=registry)
        self.assertEqual(names, ["auto", "g1", "g2"])
        names_large = cr.route_sources(context="大盘", registry=registry)
        self.assertEqual(names_large, ["estate", "g1", "g2"])
        names_default = cr.route_sources(context=None, registry=registry)
        self.assertEqual(names_default, ["g1", "g2"])

    def test_load_for_context_matches_reason(self):
        cr = self.cr
        registry = {
            "g1": {"desc": "global 1", "status": "active", "scope": ["global"]},
            "auto": {"desc": "汽车", "status": "active", "scope": ["chain_auto"]},
        }
        out = cr.load_for_context(context="消费", date="2026-08-13", registry=registry)
        self.assertEqual(out["source_names"], ["auto", "g1"])
        self.assertIn("auto", out["matched"])
        self.assertEqual(out["global"], ["g1"])
        auto = next(row for row in out["sources"] if row["source"] == "auto")
        self.assertEqual(auto["reason"], "context")
        g1 = next(row for row in out["sources"] if row["source"] == "g1")
        self.assertEqual(g1["reason"], "global")

    def test_context_router_deterministic_order(self):
        cr = self.cr
        first = cr.load_for_context(context="消费", date="2026-08-13")
        second = cr.load_for_context(context="消费", date="2026-08-13")
        self.assertEqual(first["source_names"], second["source_names"])


if __name__ == "__main__":
    unittest.main()
