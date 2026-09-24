"""battle_map 行动卡（build_action_tree）单元测试。

只测新增行动卡逻辑与 render_md / build_map 的接入：
- 高置信度 → 方向含仓位、价格区间取龙头、风险股止损建议
- 低置信度 → 观望 + 原因，不输出新增标的
- 无核心攻击 / 无风险股 / 空 bm 的兜底
- render_md 插入行动卡区块
- build_map 返回值新增 action_tree 键（兼容不破坏现有调用方）
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.battle_map as bm
    return bm


class TestActionTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def _high_bm(self):
        return {
            "confidence": 0.8,
            "recommended": "进攻",
            "position_range": "5-14%",
            "macro_veto": "none",
            "attack_groups": {
                "core": [{"name": "主线A", "score": 3.2,
                          "leader": {"name": "万邦医药", "boards": 3}}],
                "observe": [],
            },
            "risk_watch": [
                {"name": "万邦医药", "code": "301520", "risk": "高换手", "action": "竞价弱则走"},
                {"name": "分歧B", "code": "000002", "risk": "炸板", "action": "不接力"},
            ],
        }

    def test_high_confidence_full_cards(self):
        tree = self.bm.build_action_tree(self._high_bm())
        self.assertEqual(len(tree), 3)
        self.assertEqual(tree[0]["line"], "① 操作方向")
        self.assertIn("进攻", tree[0]["text"])
        self.assertIn("5-14%", tree[0]["text"])
        self.assertIn("万邦医药", tree[1]["text"])
        self.assertIn("昨收±2%", tree[1]["text"])
        self.assertIn("万邦医药 竞价弱则走", tree[2]["text"])

    def test_low_confidence_hold(self):
        bm = {
            "confidence": 0.4,
            "recommended": "防守",
            "position_range": "5-14%",
            "macro_veto": "soft",
            "attack_groups": {
                "core": [{"name": "主线A", "leader": {"name": "万邦医药"}}],
            },
            "risk_watch": [],
        }
        tree = self.bm.build_action_tree(bm)
        self.assertIn("观望/不操作", tree[0]["text"])
        self.assertIn("soft", tree[0]["text"])
        self.assertEqual(tree[1]["text"], "无新增标的")
        self.assertIn("不追高，破位止损", tree[2]["text"])

    def test_no_core_attack(self):
        bm = self._high_bm()
        bm["attack_groups"] = {"core": [], "observe": []}
        tree = self.bm.build_action_tree(bm)
        self.assertEqual(tree[1]["text"], "无新增标的")

    def test_no_risk_stocks_uses_discipline(self):
        bm = self._high_bm()
        bm["risk_watch"] = []
        tree = self.bm.build_action_tree(bm)
        self.assertIn("破位止损", tree[2]["text"])

    def test_empty_bm_safe(self):
        tree = self.bm.build_action_tree({})
        self.assertEqual(len(tree), 3)
        self.assertIn("观望/不操作", tree[0]["text"])
        self.assertEqual(tree[1]["text"], "无新增标的")

    def test_render_md_has_action_tree_section(self):
        bm = {
            "date": "2026-08-10",
            "emotion_stage": "发酵", "regime": "亢奋", "macro_veto": "none",
            "confidence": 0.8, "recommended": "进攻",
            "position_range": "5-14%", "position_basis": "决策卡5-14%",
            "bid_watch": [],
            "attack_groups": {"core": [{"name": "主线A", "score": 3.2,
                                        "strategy": "打板(换手充分)",
                                        "leader": {"name": "万邦医药", "boards": 3}}],
                              "observe": []},
            "risk_watch": [{"name": "万邦医药", "code": "301520",
                            "risk": "高换手", "action": "竞价弱则走"}],
            "linkage": [],
            "rag_explanations": {"types": []},
            "degraded": [],
        }
        with mock.patch("quant_system.analysis_core.hypothesis_generator.generate",
                        return_value=[]), \
                mock.patch("quant_system.analysis_core.multi_agent.arbitrate",
                           return_value={}), \
                mock.patch("quant_system.analysis_core.game_theory_model.current_game",
                           return_value={}):
            md = self.bm.render_md(bm)
        self.assertIn("## ⚡ 行动卡", md)
        self.assertIn("**① 操作方向**", md)

    def test_build_map_returns_action_tree_key(self):
        from quant_system.tests.test_battle_map_v11 import _TmpMixin, _mock_env

        class Case(_TmpMixin, unittest.TestCase):
            pass

        case = Case(methodName="runTest")
        tmp = case._make_tmp()
        try:
            for patcher in _mock_env(self.bm, tmp):
                patcher.start()
            try:
                result = self.bm.build_map("2026-08-10")
            finally:
                mock.patch.stopall()
        finally:
            case.doCleanups()
        self.assertIn("action_tree", result)
        self.assertEqual(len(result["action_tree"]), 3)


if __name__ == "__main__":
    unittest.main()
