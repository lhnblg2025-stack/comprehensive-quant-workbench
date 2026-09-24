"""analysis_core/macro_system 体系5 宏观周期系统单元测试。

覆盖:
  - 美林时钟简化 base_mode：复苏/过热/滞胀/衰退/中性过渡
  - 海外交叉 _cross_check：滞胀避险模式 → 强化滞胀判定
  - _indicator 新鲜度：>45 天陈旧不参与
  - MacroSystem.view() 返回 {agent:'宏观周期', signal, confidence, evidence}

全部纯逻辑测试，不依赖 data_warehouse / 网络 / RAG。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.macro_system as ms
    return ms


def _drivers(growth="中性", inflation="中性", liquidity="中性", risk="中性"):
    return {
        "growth": {"state": growth, "indicators": []},
        "inflation": {"state": inflation, "indicators": []},
        "liquidity": {"state": liquidity, "indicators": []},
        "risk_appetite": {"state": risk, "indicators": []},
    }


def _overseas(mode):
    return {"mode": mode, "flags": {"gold_up": True, "oil_up": True,
                                    "yield_high": True, "gold_oil_up": True}}


class TestBaseMode(unittest.TestCase):
    def test_recovery(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="强", inflation="弱", liquidity="宽"))
        self.assertEqual(mode, "复苏")

    def test_overheat(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="强", inflation="强"))
        self.assertEqual(mode, "过热")

    def test_stagflation(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="弱", inflation="强"))
        self.assertEqual(mode, "滞胀")

    def test_recession(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="弱", inflation="弱", liquidity="宽"))
        self.assertEqual(mode, "衰退")

    def test_transition(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="弱", inflation="中性", liquidity="宽"))
        self.assertEqual(mode, "中性过渡")

    def test_missing_driver_treated_neutral(self):
        ms = _import()
        mode, _ = ms._base_mode(_drivers(growth="数据缺失", inflation="强"))
        self.assertEqual(mode, "中性过渡")


class TestCrossCheck(unittest.TestCase):
    def test_stagflation_boost_from_transition(self):
        ms = _import()
        drivers = _drivers(growth="弱", inflation="中性", liquidity="宽")
        mode, notes, boost = ms._cross_check("中性过渡", drivers, _overseas("滞胀避险模式"))
        self.assertEqual(mode, "滞胀（避险强化）")
        self.assertTrue(boost)

    def test_stagflation_confirmed(self):
        ms = _import()
        drivers = _drivers(growth="弱", inflation="强")
        mode, _, boost = ms._cross_check("滞胀", drivers, _overseas("滞胀避险模式"))
        self.assertEqual(mode, "滞胀（海外交叉确认）")
        self.assertTrue(boost)

    def test_no_boost_when_overseas_neutral(self):
        ms = _import()
        drivers = _drivers(growth="弱", inflation="中性")
        mode, _, boost = ms._cross_check("中性过渡", drivers, _overseas("中性"))
        self.assertEqual(mode, "中性过渡")
        self.assertFalse(boost)

    def test_recession_not_boosted_when_inflation_weak(self):
        ms = _import()
        drivers = _drivers(growth="弱", inflation="弱", liquidity="宽")
        mode, _, boost = ms._cross_check("衰退", drivers, _overseas("滞胀避险模式"))
        self.assertEqual(mode, "衰退")
        self.assertFalse(boost)


class TestIndicatorFreshness(unittest.TestCase):
    def test_stale_excluded(self):
        ms = _import()
        ref = pd.Timestamp("2026-08-11")
        ind = ms._indicator("测试指标", 8.0, pd.Timestamp("2026-06-01"), ref, 0.5, ms._state_m2)
        self.assertFalse(ind["fresh"])
        self.assertIsNone(ind["state"])
        self.assertIn("陈旧", ind["note"])

    def test_fresh_included(self):
        ms = _import()
        ref = pd.Timestamp("2026-08-11")
        ind = ms._indicator("测试指标", 1.43, pd.Timestamp("2026-08-06"), ref, 0.3, ms._state_shibor)
        self.assertTrue(ind["fresh"])
        self.assertEqual(ind["state"], "宽")

    def test_missing_labeled(self):
        ms = _import()
        ref = pd.Timestamp("2026-08-11")
        ind = ms._indicator("测试指标", None, None, ref, 0.2, ms._state_cpi)
        self.assertFalse(ind["fresh"])
        self.assertEqual(ind["note"], "数据缺失")


class TestView(unittest.TestCase):
    def test_view_contract(self):
        ms_mod = _import()
        msys = ms_mod.MacroSystem()
        fake = {
            "date": "2026-08-11",
            "status": "ok",
            "mode": "滞胀（避险强化）",
            "base_mode": "中性过渡",
            "confidence": 0.81,
            "evidence": ["增长弱", "海外交叉滞胀"],
            "drivers": {"growth": {"state": "弱"}, "inflation": {"state": "中性"},
                        "liquidity": {"state": "宽"}, "risk_appetite": {"state": "中性"}},
            "asset_mapping": {"prefer": ["黄金"], "avoid": ["成长"]},
        }
        with mock.patch.object(msys, "detect", return_value=fake):
            v = msys.view("2026-08-11")
        self.assertEqual(v["agent"], "宏观周期")
        self.assertEqual(v["signal"], "防守")
        self.assertEqual(v["confidence"], 0.81)
        self.assertIn("增长弱", v["evidence"])
        self.assertEqual(v["detail"]["mode"], "滞胀（避险强化）")

    def test_view_signal_mapping(self):
        ms_mod = _import()
        msys = ms_mod.MacroSystem()
        cases = {"复苏": "多", "过热": "多", "衰退": "空", "中性过渡": "震荡"}
        for mode, want in cases.items():
            fake = {"date": "2026-08-11", "status": "ok", "mode": mode,
                    "base_mode": mode, "confidence": 0.5, "evidence": [],
                    "drivers": {"growth": {"state": "中性"}}, "asset_mapping": {}}
            with mock.patch.object(msys, "detect", return_value=fake):
                self.assertEqual(msys.view().get("signal"), want)


if __name__ == "__main__":
    unittest.main()
