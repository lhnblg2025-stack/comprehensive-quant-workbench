"""analysis_core/cycle_system 市场周期系统单元测试。

覆盖:
  - 周期阶段识别: 上涨/顶部/下跌/底部/筑底 五态（合成趋势序列）+ 样本不足防御
  - T形底部: 底部横盘(<5%) + 放量突破(量比>1.5) / 无量不构成信号 / 样本不足
  - VIX 买入区: VIX>30 恐慌买入 / 温度<30 替代 / 温度>85+VIX<15 风险区
  - 周期共振: 一致(强) / 结构分化 / 无数据
  - 综合: 阶段×共振×VIX → 布局/持有/减仓/防守 + T形底/风险区修正
  - CycleSystem.view: multi_agent 兼容结构 / detect 异常降级
  - RAG: 检索失败 → '检索不可用'
  - report: 写 md 文件（mock detect，不依赖真实 data_warehouse）
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module("quant_system.analysis_core.cycle_system")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="cycle_system_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


def _synth_df(segs, seed=2) -> pd.DataFrame:
    """合成日K: segs = [(天数, 日趋势, 日波动)]。"""
    rng = np.random.default_rng(seed)
    closes, vols = [], []
    for days, tr, vol in segs:
        c = 100.0 if not closes else closes[-1]
        chunk = c * np.cumprod(1 + tr + rng.normal(0, vol, days))
        closes.extend(chunk.tolist())
        vols.extend([1e6] * days)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=len(closes)),
        "open": closes * 0.99, "high": closes * 1.02,
        "low": closes * 0.98, "close": closes, "volume": vols,
    })


class TestStageDetection(unittest.TestCase):
    """周期阶段识别（MA20/MA60 排列 + MA斜率 + ROC 动量 → 五态）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_uptrend(self):
        r = self.m._detect_stage(_synth_df([(200, 0.002, 0.004)]))
        self.assertEqual(r["stage"], "上涨")
        self.assertGreater(r["confidence"], 0.7)

    def test_downtrend(self):
        r = self.m._detect_stage(_synth_df([(200, -0.002, 0.004)]))
        self.assertEqual(r["stage"], "下跌")
        self.assertGreater(r["confidence"], 0.7)

    def test_top(self):
        r = self.m._detect_stage(_synth_df([(140, 0.002, 0.004), (25, -0.001, 0.003)]))
        self.assertEqual(r["stage"], "顶部")

    def test_bottom(self):
        r = self.m._detect_stage(_synth_df([(140, -0.002, 0.004), (15, 0.0025, 0.004)]))
        self.assertEqual(r["stage"], "底部")

    def test_base_building(self):
        r = self.m._detect_stage(_synth_df([(140, -0.002, 0.004), (45, 0.0, 0.0006)]))
        self.assertEqual(r["stage"], "筑底")

    def test_insufficient_samples(self):
        r = self.m._detect_stage(_synth_df([(40, 0.002, 0.004)]))
        self.assertEqual(r["stage"], "未知")
        self.assertIn("样本不足", r["note"])


def _t_df(breakout_vol: float = 2.0e6, breakout_pct: float = 1.01) -> pd.DataFrame:
    """构造 T形底数据: 下跌60日 → 横盘15日(振幅<5%) → 放量突破日。"""
    rng = np.random.default_rng(5)
    pre = 100.0 * np.cumprod(1 - 0.002 + rng.normal(0, 0.004, 60))
    base_end = pre[-1]
    base = base_end * (1 + rng.normal(0, 0.0015, 15))
    closes = np.concatenate([pre, base, [base.max() * breakout_pct]])
    n = len(closes)
    vols = np.concatenate([np.full(60, 1e6), np.full(15, 1e6), [breakout_vol]])
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=n),
        "open": closes * 0.99, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes, "volume": vols,
    })


class TestTFormation(unittest.TestCase):
    """T形底部检测（底部横盘<5% + 量比>1.5 放量突破）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_flat_base_volume_breakout(self):
        r = self.m._detect_t_formation(_t_df())
        self.assertEqual(r["signal"], self.m.T_FORMATION_SIGNAL)
        self.assertLess(r["range_pct"], 5.0)
        self.assertGreater(r["volume_ratio"], 1.5)
        self.assertGreater(r["breakout_pct"], 0.0)

    def test_no_breakout_without_volume(self):
        r = self.m._detect_t_formation(_t_df(breakout_vol=1.0e6))
        self.assertIsNone(r["signal"])

    def test_no_signal_on_downtrend(self):
        r = self.m._detect_t_formation(_synth_df([(200, -0.002, 0.004)]))
        self.assertIsNone(r["signal"])

    def test_insufficient_samples(self):
        r = self.m._detect_t_formation(_synth_df([(30, -0.002, 0.004)]))
        self.assertIsNone(r["signal"])
        self.assertIn("样本不足", r["note"])


class TestVixZone(unittest.TestCase):
    """VIX 买入区（VIX>30 或 温度<30 替代 → 恐慌买入；温度>85+VIX<15 → 风险区）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_vix_panic_buy_zone(self):
        z = self.m._detect_vix_zone(35.0, 50.0)
        self.assertEqual(z["zone"], "恐慌买入区")
        self.assertEqual(z["level"], "buy")

    def test_temp_substitute_panic(self):
        z = self.m._detect_vix_zone(None, 25.0)
        self.assertEqual(z["zone"], "恐慌买入区")
        self.assertIn("温度替代", z["reason"])

    def test_high_temp_low_vix_risk(self):
        z = self.m._detect_vix_zone(12.0, 90.0)
        self.assertEqual(z["zone"], "风险区")
        self.assertEqual(z["level"], "risk")

    def test_temp_substitute_risk(self):
        z = self.m._detect_vix_zone(None, 90.0)
        self.assertEqual(z["zone"], "高温风险区")
        self.assertEqual(z["level"], "risk")

    def test_neutral(self):
        z = self.m._detect_vix_zone(20.0, 50.0)
        self.assertEqual(z["level"], "neutral")


class TestResonance(unittest.TestCase):
    """周期共振（多指数同阶段=一致；分歧=结构分化）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_consistent_strong(self):
        r = self.m._resonance(["底部", "底部", "底部", "下跌"])
        self.assertEqual(r["state"], "周期一致(强)")
        self.assertEqual(r["dominant"], "底部")

    def test_divergence(self):
        r = self.m._resonance(["上涨", "下跌", "顶部", "底部"])
        self.assertEqual(r["state"], "结构分化")

    def test_no_data(self):
        r = self.m._resonance([])
        self.assertEqual(r["state"], "无数据")
        self.assertIsNone(r["dominant"])


class TestComposite(unittest.TestCase):
    """综合: 阶段×共振×VIX → 状态 + 策略建议。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _comp(self, dominant="底部", reso_state="周期一致(强)", share=0.75,
              vix_level="neutral", vix_zone="中性区", t_hits=None, breadth=None):
        return self.m._composite({
            "resonance": {"state": reso_state, "dominant": dominant, "share": share,
                          "counts": {dominant: 3, "下跌": 1}, "note": ""},
            "vix_zone": {"zone": vix_zone, "level": vix_level, "reason": ""},
            "t_hits": t_hits or [],
            "breadth": breadth,
        })

    def test_bottom_layout(self):
        c = self._comp(dominant="底部")
        self.assertEqual(c["state"], "底部期")
        self.assertIn("布局", c["strategy"])
        self.assertEqual(c["signal"], "多")

    def test_uptrend_hold(self):
        c = self._comp(dominant="上涨")
        self.assertIn("持有", c["strategy"])
        self.assertEqual(c["signal"], "多")

    def test_top_reduce(self):
        c = self._comp(dominant="顶部")
        self.assertIn("减仓", c["strategy"])
        self.assertEqual(c["signal"], "空")

    def test_decline_defense(self):
        c = self._comp(dominant="下跌")
        self.assertIn("防守", c["strategy"])
        self.assertEqual(c["signal"], "空")

    def test_divergence_downgrade(self):
        c = self._comp(reso_state="结构分化", share=0.5)
        self.assertEqual(c["state"], "结构分化期")
        self.assertEqual(c["signal"], "震荡")
        self.assertLess(c["confidence"], 0.6)

    def test_vix_panic_boost_layout(self):
        c = self._comp(dominant="底部", vix_level="buy", vix_zone="恐慌买入区")
        self.assertIn("逆向布局", c["strategy"])
        self.assertEqual(c["signal"], "多")

    def test_vix_risk_zone(self):
        c = self._comp(dominant="上涨", vix_level="risk", vix_zone="风险区")
        self.assertIn("减仓防守", c["strategy"])
        self.assertEqual(c["signal"], "空")

    def test_t_formation_confirms(self):
        c = self._comp(dominant="底部", t_hits=[{
            "index": "沪深300", "date": "2026-08-01", "range_pct": 3.0,
            "volume_ratio": 1.8, "breakout_pct": 1.0}])
        self.assertIn("T形底突破确认", c["strategy"])


class TestView(unittest.TestCase):
    """CycleSystem.view: multi_agent 兼容结构 / 异常降级。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _canned(self):
        return {
            "date": "2026-08-10", "data_date": "2026-08-10",
            "signal": "多", "confidence": 0.72, "status": "ok",
            "evidence": ["周期共振: 周期一致(强)"],
            "resonance": {"state": "周期一致(强)", "dominant": "底部"},
            "vix_zone": {"zone": "中性区"},
            "t_hits": [], "temperature": 50.0, "vix": None,
            "composite": {"state": "底部期", "strategy": "布局"},
            "rag": {"basis": "检索可用"},
        }

    def test_view_multi_agent_compatible(self):
        cs = self.m.CycleSystem()
        with mock.patch.object(cs, "detect", return_value=self._canned()):
            v = cs.view()
        self.assertEqual(v["agent"], "市场周期")
        self.assertEqual(v["signal"], "多")
        self.assertEqual(v["view"], "多")
        self.assertIsInstance(v["evidence"], list)
        self.assertTrue(v["evidence"])
        self.assertEqual(v["status"], "ok")
        self.assertEqual(v["detail"]["cycle_state"], "底部期")
        self.assertEqual(v["detail"]["strategy"], "布局")

    def test_view_degraded_on_detect_error(self):
        cs = self.m.CycleSystem()
        with mock.patch.object(cs, "detect", side_effect=RuntimeError("boom")):
            v = cs.view()
        self.assertEqual(v["status"], "degraded")
        self.assertEqual(v["signal"], "震荡")
        self.assertEqual(v["confidence"], 0.0)


class TestRagFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_search_failure_marks_unavailable(self):
        from quant_system.analysis_core import knowledge_rag
        cs = self.m.CycleSystem()
        cs._rag_cache = None
        with mock.patch.object(knowledge_rag, "search", side_effect=RuntimeError("no model")):
            out = cs._rag()
        self.assertEqual(out["basis"], "检索不可用")
        self.assertFalse(out["available"])
        self.assertEqual(out["hits"], [])


class TestReportWrite(unittest.TestCase, _TmpDirMixin):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_report_writes_md(self):
        out_dir = self._make_tmp()
        cs = self.m.CycleSystem(out_dir=out_dir)
        with mock.patch.object(cs, "detect", return_value={
            "data_date": "2026-08-10", "date": "2026-08-10",
            "signal": "多", "confidence": 0.72, "status": "ok",
            "evidence": ["周期共振: 周期一致(强)", "RAG: 检索可用"],
            "indexes": [], "t_hits": [], "breadth": None,
            "resonance": {"state": "周期一致(强)", "dominant": "底部", "note": ""},
            "vix_zone": {"zone": "中性区", "reason": ""},
            "composite": {"state": "底部期", "strategy": "布局",
                          "suggestions": ["底部阶段：分批布局"]},
            "rag": {"query": "市场周期 筑底 T形底部 VIX", "basis": "检索可用",
                    "hits": [{"file": "skills/a.md", "cat": "c", "score": 0.5,
                              "summary": "摘要"}]},
            "data_status": {}, "temperature": 50.0, "vix": None,
        }):
            path = cs.report(date="2026-08-10")
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("市场周期报告 2026-08-10", text)
        self.assertIn("布局", text)


if __name__ == "__main__":
    unittest.main()
