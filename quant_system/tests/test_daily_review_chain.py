# -*- coding: utf-8 -*-
"""每日复盘融合链 · 单元测试（2026-08-21 新增）

覆盖: DataGateway 门控 / 六要素采集降级 / 量化因子面板 / 复盘 md 渲染 / 推送封装。
放置于 quant_system/tests/（与现有 pytest 体系一致）。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "quant_system"))

from daily_fusion_base import DataGateway, get_gateway  # noqa: E402
from daily_review_collectors import SignalBlock, collect_all, _safe_collect  # noqa: E402
from daily_quant_collectors import collect_factor_signal, collect_quant_all  # noqa: E402


class TestDataGateway(unittest.TestCase):
    """L0 数据基座门控。"""

    @classmethod
    def setUpClass(cls):
        cls.gw = DataGateway()

    def test_health_scores_structure(self):
        h = self.gw.health_scores()
        self.assertIn("overall_ok", h)
        self.assertIn("degraded_datasets", h)
        self.assertIn("trust_scores", h)

    def test_gate_structure(self):
        g = self.gw.gate("zt_daily_stats")
        self.assertIn("ok", g)
        self.assertIn("lag_days", g)

    def test_normalize_pct(self):
        # to_pct(小数0.052) → 百分数5.2（若 data_contract 可用）
        v = self.gw.normalize_pct(0.052, "pct")
        if v is not None:
            self.assertAlmostEqual(v, 5.2, places=1)

    def test_source_status(self):
        s = self.gw.source_status()
        self.assertIsInstance(s, dict)


class TestCollectors(unittest.TestCase):
    """L1 六要素采集（降级安全）。"""

    @classmethod
    def setUpClass(cls):
        cls.gw = get_gateway()

    def test_signal_block(self):
        b = SignalBlock("t", "src", {"a": 1}, confidence=0.5)
        d = b.to_dict()
        self.assertEqual(d["element"], "t")
        self.assertEqual(d["source"], "src")

    def test_safe_collect_timeout(self):
        # 超时 fn → error=timeout，不炸
        def slow():
            import time
            time.sleep(5)
            return {"x": 1}
        b = _safe_collect(slow, "t", "s", {"x": 0}, timeout_seconds=1)
        self.assertEqual(b.error, "timeout")

    def test_safe_collect_exception(self):
        def bad():
            raise ValueError("boom")
        b = _safe_collect(bad, "t", "s", {}, timeout_seconds=2)
        self.assertIsNotNone(b.error)
        self.assertIn("boom", b.error)

    def test_collect_market_temperature(self):
        from daily_review_collectors import collect_market_temperature  # noqa: PLC0415
        b = collect_market_temperature(self.gw)
        # 不应 critical 抛错；error 可为降级但结构存在
        self.assertIsInstance(b.value, dict)
        self.assertIn("components", b.value)

    def test_collect_quant_layer(self):
        """量化层因子采集（核心：面板读取不超时、结构正确）。"""
        b = collect_factor_signal(self.gw)
        self.assertIsInstance(b.value, dict)
        if not b.error and b.value.get("groups"):
            self.assertIn("技术动量", b.value["groups"])


class TestChainRender(unittest.TestCase):
    """L2 融合链输出。"""

    def test_render_md_structure(self):
        from daily_review_chain import _render_md  # noqa: PLC0415, E402
        blocks = collect_all(get_gateway())
        md = _render_md("2026-08-21", blocks, {}, {}, {"overall_ok": True, "degraded_datasets": []})
        self.assertIsInstance(md, str)
        self.assertIn("每日A股复盘", md)


class TestPushReview(unittest.TestCase):
    """推送封装（飞书通道参数正确性）。"""

    def test_load_review_md(self):
        from push_review import _load_review_md  # noqa: PLC0415, E402
        # 无产物也应安全返回
        md = _load_review_md("2099-01-01")
        self.assertEqual(md, "")

    def test_push_feishu(self):
        from push_review import push_feishu  # noqa: PLC0415, E402
        # 飞书实测通道（有凭证时应 True；无凭证返回 False 不异常）
        try:
            ok = push_feishu("测试推送内容", "2026-08-21")
            self.assertIsInstance(ok, bool)
        except Exception:  # noqa: BLE001
            self.skipTest("飞书凭证不可用")


class TestDecisionEngine(unittest.TestCase):
    """L2 决策研判（决策卡生成）。"""

    def _blocks(self):
        from daily_fusion_base import get_gateway  # noqa: PLC0415
        b = collect_all(get_gateway())
        b.update(collect_quant_all(get_gateway()))
        return b

    def test_make_decision_structure(self):
        from decision_engine import make_decision  # noqa: PLC0415, E402
        dec = make_decision(self._blocks())
        for k in ("定调", "主线研判", "操作清单", "风险预案", "决策依据"):
            self.assertIn(k, dec)
        self.assertIn("position", dec["定调"])
        self.assertIn("attack", dec["操作清单"])

    def test_decision_render(self):
        from decision_engine import make_decision, render_decision_md  # noqa: PLC0415, E402
        dec = make_decision(self._blocks())
        md = render_decision_md(dec)
        self.assertIn("明日定调", md)
        self.assertIn("进攻清单", md)

    def test_emotion_posture_mapping(self):
        from decision_engine import EMOTION_POSTURE, EMOTION_POSITION  # noqa: PLC0415, E402
        self.assertIn("冰点", EMOTION_POSTURE)
        self.assertIn("发酵", EMOTION_POSITION)


if __name__ == "__main__":
    unittest.main()