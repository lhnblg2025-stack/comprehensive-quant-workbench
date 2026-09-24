"""analysis_core/multi_agent 专家委员会单元测试。

覆盖:
  - 11 个 _view_* 专家视图（全 mock 数据驱动，不依赖真实模块数据）
  - _system_view 归一化（词表映射 / 非法 view / status 映射）
  - agent_views 全链路 + 单专家降级（不阻塞）
  - _effective_weights 降级权重重分配
  - arbitrate 加权投票：多/空/震荡/硬否决 + 落盘（临时 OUT_DIR）
  - _build_disagreements 分歧检测 + debate 辩论输出
  - _is_conflict / _opposite / _implication 纯逻辑
  - 魔鬼代言人辩论（域L）: 触发条件(6:5触发/8:2不触发) / 3轮攻讦结构 / 答辩评分 / 终审 /
    audit_trail 落盘 / 旧接口兼容
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.multi_agent as ma
    return ma


def _make_views(**overrides) -> list[dict]:
    """构造 11 个健康 agent 的标准视图（默认全部'震荡'低置信，互不干扰）。"""
    names = ["情绪面", "资金面", "题材面", "技术面", "风险面", "规律面",
             "三屏趋势", "量价筹码", "情绪周期", "宏观周期", "风险纪律"]
    views = [{"agent": n, "view": "震荡", "confidence": 0.3, "evidence": [f"{n}-ev"],
              "weight": 0.0, "status": "ok", "detail": {"date": "2026-08-10"}}
             for n in names]
    for key, val in overrides.items():
        views[key] = val
    return views


class _TmpOutMixin:
    def _tmp_out(self):
        self._out = Path(tempfile.mkdtemp(prefix="multi_agent_test_"))
        self.addCleanup(shutil.rmtree, self._out, ignore_errors=True)
        return self._out


class TestViewEmotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _view(self, **raw):
        with mock.patch.object(self.ma, "emotion_run_today", return_value=raw):
            return self.ma._view_emotion("2026-08-10")

    def test_ice_gives_bearish_high_conf(self):
        v = self._view(stage="ice", stage_cn="冰点期", confidence=0.3,
                       zt_cnt=15, max_board=2, zb_rate=0.4,
                       transition_probs={"修复期(repair)": 0.8})
        self.assertEqual(v["agent"], "情绪面")
        self.assertEqual(v["view"], "空")
        self.assertGreaterEqual(v["confidence"], 0.6)
        self.assertEqual(v["status"], "ok")
        self.assertTrue(any("涨停15家" in e for e in v["evidence"]))
        self.assertTrue(any("最高2板" in e for e in v["evidence"]))
        self.assertTrue(any("炸板率40%" in e for e in v["evidence"]))
        self.assertTrue(any("明日倾向" in e for e in v["evidence"]))

    def test_ferment_gives_bullish(self):
        v = self._view(stage="ferment", stage_cn="发酵期", confidence=0.7)
        self.assertEqual(v["view"], "多")
        self.assertAlmostEqual(v["confidence"], 0.7)

    def test_climax_bullish_confidence_capped(self):
        v = self._view(stage="climax", stage_cn="高潮期", confidence=1.2)
        self.assertEqual(v["view"], "多")
        self.assertLessEqual(v["confidence"], 0.95)

    def test_divergence_gives_neutral(self):
        v = self._view(stage="divergence", stage_cn="分歧期", confidence=0.9)
        self.assertEqual(v["view"], "震荡")
        self.assertGreaterEqual(v["confidence"], 0.5)

    def test_unknown_stage_neutral(self):
        v = self._view(stage="weird", stage_cn="?", confidence=0.1,
                       zt_cnt=None, max_board=None, zb_rate=None,
                       transition_probs={})
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.5)


class TestViewFund(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _df(self, fi, signs="{}", youzi=1e8, jg=2e8, north=-0.5e8,
            margin=3.0, dates=("2026-08-06", "2026-08-07", "2026-08-10")):
        return pd.DataFrame({
            "date": pd.to_datetime(dates),
            "force_index": [50, 55, fi],
            "signs": [signs, signs, signs],
            "youzi_net": [1e8, 1e8, youzi],
            "jg_net": [1e8, 1e8, jg],
            "north_net": [1e8, 1e8, north],
            "margin_delta": [1.0, 1.0, margin],
        })

    def _view(self, df):
        with mock.patch.object(self.ma, "fund_latest", return_value=df):
            return self.ma._view_fund("2026-08-10")

    def test_strong_force_index_bullish(self):
        v = self._view(self._df(fi=88, signs='{"main": 1, "retail": -1}'))
        self.assertEqual(v["view"], "多")
        self.assertAlmostEqual(v["confidence"], 0.88)
        self.assertEqual(v["status"], "ok")
        self.assertEqual(v["detail"]["date"], "2026-08-10")
        self.assertTrue(any("合力指数88" in e for e in v["evidence"]))
        self.assertTrue(any("游资净买+1.0亿" in e for e in v["evidence"]))
        self.assertTrue(any("机构净买+2.0亿" in e for e in v["evidence"]))
        self.assertTrue(any("北向净买-0.5亿" in e for e in v["evidence"]))
        self.assertTrue(any("融资Δ+3.0亿" in e for e in v["evidence"]))
        self.assertTrue(any("main+" in e and "retail-" in e for e in v["evidence"]))

    def test_weak_force_index_bearish(self):
        v = self._view(self._df(fi=20))
        self.assertEqual(v["view"], "空")
        self.assertAlmostEqual(v["confidence"], 0.8)

    def test_mid_force_index_neutral(self):
        v = self._view(self._df(fi=50))
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.5)

    def test_bad_signs_json_does_not_raise(self):
        v = self._view(self._df(fi=50, signs="{not json"))
        self.assertEqual(v["status"], "ok")
        self.assertEqual(v["view"], "震荡")


class TestViewTheme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _view(self, themes=None, reso=None, date="2026-08-10"):
        raw = {"date": date, "themes": themes or []}
        with mock.patch.object(self.ma, "today_themes", return_value=raw), \
                mock.patch.object(self.ma, "resonance_score_day", return_value=reso):
            return self.ma._view_theme(date)

    def _reso(self, scores, levels):
        return pd.DataFrame({"score": scores, "level": levels})

    def test_hot_themes_bullish(self):
        themes = [{"role": "主线", "board": "固态电池", "zt_cnt": 15},
                  {"role": "主线", "board": "AI算力", "zt_cnt": 12},
                  {"role": "支线", "board": "低空经济", "zt_cnt": 5}]
        reso = self._reso([3.0, 2.6, 1.0], ["主线", "主线", "支线"])
        v = self._view(themes=themes, reso=reso)
        self.assertEqual(v["view"], "多")
        self.assertGreaterEqual(v["confidence"], 0.55)
        self.assertEqual(v["detail"]["main_count"], 2)
        self.assertTrue(any("最强=固态电池(涨停15)" in e for e in v["evidence"]))
        self.assertTrue(any("共振主线2条" in e for e in v["evidence"]))

    def test_no_themes_and_low_resonance_bearish(self):
        v = self._view(themes=[], reso=self._reso([0.5], ["支线"]))
        self.assertEqual(v["view"], "空")
        self.assertAlmostEqual(v["confidence"], 0.6)

    def test_tepid_neutral(self):
        themes = [{"role": "支线", "board": "x", "zt_cnt": 3}]
        v = self._view(themes=themes, reso=self._reso([1.5], ["支线"]))
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.5)

    def test_resonance_none_safe(self):
        v = self._view(themes=None, reso=None)
        self.assertEqual(v["view"], "空")  # 无题材无共振 → 空
        self.assertEqual(v["detail"]["top_score"], None)


class TestViewTechnical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _view(self, regime, bias=None, vol=None):
        raw = {"regime": regime, "bias_ma300": bias, "vol20_annual": vol}
        with mock.patch.object(self.ma, "get_regime", return_value=raw):
            return self.ma._view_technical("2026-08-10")

    def test_uptrend_bullish(self):
        v = self._view("趋势市", bias=2.5, vol=18.0)
        self.assertEqual(v["view"], "多")
        self.assertTrue(any("机制=趋势市" in e for e in v["evidence"]))
        self.assertTrue(any("MA300乖离+2.50%" in e for e in v["evidence"]))

    def test_downtrend_bearish(self):
        v = self._view("下跌趋势", bias=-3.0)
        self.assertEqual(v["view"], "空")

    def test_range_neutral(self):
        v = self._view("震荡市", bias=0.5)
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.55)

    def test_unknown_regime_neutral(self):
        v = self._view("?")
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.5)


class TestViewRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _view(self, level="none", reasons=None, issues=None, mode="normal"):
        veto = {"level": level, "reasons": reasons or [],
                "position_coef": 0.5, "pe_percentile": 60, "buffett_ratio": 0.8}
        heal = {"issues": issues or [], "decision_mode": mode}
        with mock.patch.object(self.ma, "get_veto", return_value=veto), \
                mock.patch.object(self.ma, "healer_check", return_value=heal):
            return self.ma._view_risk("2026-08-10")

    def test_hard_veto_defensive(self):
        v = self._view(level="hard", reasons=["宏观硬否决"])
        self.assertEqual(v["view"], "防守")
        self.assertAlmostEqual(v["confidence"], 0.95)
        self.assertEqual(v["detail"]["veto_level"], "hard")

    def test_soft_veto_bearish(self):
        v = self._view(level="soft")
        self.assertEqual(v["view"], "空")
        self.assertAlmostEqual(v["confidence"], 0.7)

    def test_conservative_mode_neutral(self):
        # 2026-08-21 审计: 数据健康降级(conservative)是"数据不可信"而非"看跌"，
        # 语义从"空"改为中性"震荡"并降置信，避免把数据缺失误当方向性空头。
        v = self._view(level="none", issues=[{"msg": "涨停数据缺失"}],
                       mode="conservative")
        self.assertEqual(v["view"], "震荡")
        self.assertLess(v["confidence"], 0.55)
        self.assertTrue(any("数据健康=conservative(1项异常)" in e for e in v["evidence"]))

    def test_normal_bullish(self):
        v = self._view(level="none", mode="normal")
        self.assertEqual(v["view"], "多")
        self.assertAlmostEqual(v["confidence"], 0.55)


class TestSystemViews(unittest.TestCase):
    """_system_view + 5 个体系模块包装视图（trend/vpa/emotion_system/macro/risk_discipline）。"""

    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_system_view_maps_signal_words(self):
        raw = {"signal": "看多", "confidence": 0.8, "evidence": ["a"],
               "status": "ok", "detail": {"x": 1}}
        v = self.ma._system_view("三屏趋势", raw)
        self.assertEqual(v["view"], "多")
        self.assertEqual(v["agent"], "三屏趋势")
        self.assertEqual(v["status"], "ok")

    def test_system_view_invalid_view_neutral(self):
        v = self.ma._system_view("三屏趋势", {"view": "看不懂", "confidence": 0.9})
        self.assertEqual(v["view"], "震荡")
        self.assertAlmostEqual(v["confidence"], 0.9)

    def test_system_view_non_ok_status_degraded(self):
        v = self.ma._system_view("量价筹码", {"view": "空", "confidence": 0.7,
                                            "status": "degraded"})
        self.assertEqual(v["status"], "degraded")

    def test_system_view_non_list_evidence(self):
        v = self.ma._system_view("情绪周期", {"view": "多", "confidence": 0.6,
                                            "evidence": "not-a-list", "status": "ok"})
        self.assertEqual(v["evidence"], [])

    def _view_with_class(self, fn_name, module, class_name, raw):
        with mock.patch(f"{module}.{class_name}") as cls:
            cls.return_value.view.return_value = raw
            return getattr(self.ma, fn_name)("2026-08-10")

    def test_view_trend(self):
        v = self._view_with_class("_view_trend", "quant_system.analysis_core.trend_system",
                                  "TrendSystem", {"view": "多", "confidence": 0.8,
                                                  "status": "ok", "detail": {}})
        self.assertEqual(v["agent"], "三屏趋势")
        self.assertEqual(v["view"], "多")

    def test_view_vpa(self):
        v = self._view_with_class("_view_vpa", "quant_system.analysis_core.vpa_system",
                                  "VpaSystem", {"view": "空", "confidence": 0.6,
                                                "status": "ok", "detail": {}})
        self.assertEqual(v["agent"], "量价筹码")
        self.assertEqual(v["view"], "空")

    def test_view_emotion_system(self):
        v = self._view_with_class("_view_emotion_system",
                                  "quant_system.analysis_core.emotion_system",
                                  "EmotionSystem", {"view": "震荡", "confidence": 0.5,
                                                    "status": "ok", "detail": {}})
        self.assertEqual(v["agent"], "情绪周期")

    def test_view_risk_discipline(self):
        v = self._view_with_class("_view_risk_discipline",
                                  "quant_system.analysis_core.risk_system",
                                  "RiskSystem", {"view": "空", "confidence": 0.8,
                                                 "status": "ok", "detail": {}})
        self.assertEqual(v["agent"], "风险纪律")

    def test_view_macro_appends_ai_rules(self):
        with mock.patch("quant_system.analysis_core.macro_system.MacroSystem") as cls, \
                mock.patch("quant_system.analysis_core.macro_learner.load_latest_learner_result",
                           return_value={"thresholds": [], "sector_adjustments": []}), \
                mock.patch("quant_system.analysis_core.macro_learner.format_ai_rules",
                           return_value=["AI规律: xxx"]):
            cls.return_value.view.return_value = {"view": "多", "confidence": 0.6,
                                                  "status": "ok", "detail": {},
                                                  "evidence": ["宏观证据"]}
            v = self.ma._view_macro("2026-08-10")
        self.assertEqual(v["agent"], "宏观周期")
        self.assertEqual(v["evidence"][-1], "AI规律: xxx")

    def test_view_macro_learner_failure_ignored(self):
        with mock.patch("quant_system.analysis_core.macro_system.MacroSystem") as cls, \
                mock.patch("quant_system.analysis_core.macro_learner.load_latest_learner_result",
                           side_effect=RuntimeError("boom")):
            cls.return_value.view.return_value = {"view": "多", "confidence": 0.6,
                                                  "status": "ok", "detail": {},
                                                  "evidence": ["宏观证据"]}
            v = self.ma._view_macro("2026-08-10")
        self.assertEqual(v["view"], "多")
        self.assertEqual(v["evidence"], ["宏观证据"])


class TestAgentViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _emotion(self, stage="ferment", conf=0.6):
        return {"date": "2026-08-10", "stage": stage, "stage_cn": "发酵期",
                "confidence": conf, "zt_cnt": 60, "max_board": 4, "zb_rate": 0.2,
                "transition_probs": {"修复期(repair)": 0.6}}

    def _all_ok(self):
        fund = pd.DataFrame({
            "date": pd.to_datetime(["2026-08-10"]),
            "force_index": [75], "signs": ["{}"],
            "youzi_net": [1e8], "jg_net": [1e8], "north_net": [1e8],
            "margin_delta": [1.0],
        })
        return [
            mock.patch.object(self.ma, "emotion_run_today", return_value=self._emotion()),
            mock.patch.object(self.ma, "fund_latest", return_value=fund),
            mock.patch.object(self.ma, "today_themes",
                              return_value={"date": "2026-08-10",
                                            "themes": [{"role": "主线", "board": "x", "zt_cnt": 10}]}),
            mock.patch.object(self.ma, "resonance_score_day",
                              return_value=pd.DataFrame({"score": [2.6], "level": ["主线"]})),
            mock.patch.object(self.ma, "get_regime", return_value={"regime": "震荡市",
                                                                   "bias_ma300": 0.5,
                                                                   "vol20_annual": 15}),
            mock.patch.object(self.ma, "get_veto", return_value={"level": "none",
                                                                 "reasons": [], "position_coef": 1.0}),
            mock.patch.object(self.ma, "healer_check", return_value={"issues": [],
                                                                     "decision_mode": "normal"}),
            mock.patch("quant_system.analysis_core.pattern_agent.view",
                       return_value={"agent": "规律面", "view": "多", "confidence": 0.6,
                                     "evidence": ["规律"], "weight": 0.09, "status": "ok",
                                     "detail": {}}),
            mock.patch("quant_system.analysis_core.trend_system.TrendSystem"),
            mock.patch("quant_system.analysis_core.vpa_system.VpaSystem"),
            mock.patch("quant_system.analysis_core.emotion_system.EmotionSystem"),
            mock.patch("quant_system.analysis_core.macro_system.MacroSystem"),
            mock.patch("quant_system.analysis_core.risk_system.RiskSystem"),
            mock.patch("quant_system.analysis_core.macro_learner.load_latest_learner_result",
                       return_value=None),
            mock.patch("quant_system.analysis_core.macro_learner.format_ai_rules",
                       return_value=[]),
        ]

    def _patch_all(self):
        patches = self._all_ok()
        for p in patches:
            p.start()
        # 5 个体系模块类视图统一配置为 ok（patch 已生效后再设 return_value）
        for cls, view in [("trend_system.TrendSystem", "多"),
                          ("vpa_system.VpaSystem", "空"),
                          ("emotion_system.EmotionSystem", "震荡"),
                          ("macro_system.MacroSystem", "多"),
                          ("risk_system.RiskSystem", "震荡")]:
            mod_path, class_name = cls.rsplit(".", 1)
            mod = __import__("quant_system.analysis_core." + mod_path, fromlist=[class_name])
            getattr(mod, class_name).return_value.view.return_value = {
                "view": view, "confidence": 0.5, "status": "ok", "detail": {},
                "evidence": []}
        self.addCleanup(mock.patch.stopall)

    def test_agent_views_all_eleven_ok(self):
        self._patch_all()
        views = self.ma.agent_views("2026-08-10")
        self.assertEqual(len(views), 11)
        self.assertTrue(all(v["status"] == "ok" for v in views))
        self.assertEqual([v["agent"] for v in views],
                         list(self.ma.BASE_WEIGHTS))

    def test_agent_views_emotion_failure_degraded(self):
        self._patch_all()
        with mock.patch.object(self.ma, "emotion_run_today",
                               side_effect=RuntimeError("emotion boom")):
            views = self.ma.agent_views("2026-08-10")
        self.assertEqual(views[0]["status"], "degraded")
        self.assertEqual(views[0]["agent"], "情绪面")
        self.assertEqual(len(views), 11)
        self.assertTrue(all(v["status"] == "ok" for v in views[1:]))

    def test_agent_views_fund_failure_degraded(self):
        self._patch_all()
        with mock.patch.object(self.ma, "fund_latest",
                               side_effect=RuntimeError("fund boom")):
            views = self.ma.agent_views("2026-08-10")
        self.assertEqual(views[1]["status"], "degraded")
        self.assertEqual(views[1]["agent"], "资金面")
        self.assertIn("模块异常: fund boom", views[1]["evidence"][0])

    def test_agent_views_pattern_failure_degraded(self):
        self._patch_all()
        with mock.patch("quant_system.analysis_core.pattern_agent.view",
                        side_effect=RuntimeError("pattern boom")):
            views = self.ma.agent_views("2026-08-10")
        self.assertEqual([v["agent"] for v in views if v["status"] == "degraded"],
                         ["规律面"])


class TestEffectiveWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_all_healthy_keeps_base(self):
        views = _make_views()
        w = self.ma._effective_weights(views)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertAlmostEqual(w["情绪面"], self.ma.BASE_WEIGHTS["情绪面"])

    def test_degraded_renormalized(self):
        views = _make_views()
        views[0]["status"] = "degraded"   # 情绪面
        views[5]["status"] = "degraded"   # 规律面
        w = self.ma._effective_weights(views)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertEqual(w["情绪面"], 0.0)
        self.assertEqual(w["规律面"], 0.0)
        self.assertGreater(w["资金面"], self.ma.BASE_WEIGHTS["资金面"])

    def test_all_degraded_returns_base(self):
        views = _make_views()
        for v in views:
            v["status"] = "degraded"
        w = self.ma._effective_weights(views)
        self.assertAlmostEqual(w["情绪面"], self.ma.BASE_WEIGHTS["情绪面"])
        self.assertGreater(w["情绪面"], 0.0)


class TestLogicHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_is_conflict_table(self):
        self.assertFalse(self.ma._is_conflict("多", "多"))
        self.assertTrue(self.ma._is_conflict("多", "空"))
        self.assertTrue(self.ma._is_conflict("防守", "多"))   # 防守视为空
        self.assertTrue(self.ma._is_conflict("多", "震荡"))
        self.assertFalse(self.ma._is_conflict("震荡", "震荡"))
        self.assertFalse(self.ma._is_conflict("震荡", "多"))

    def test_opposite_table(self):
        self.assertTrue(self.ma._opposite("多", "空"))
        self.assertTrue(self.ma._opposite("防守", "多"))
        self.assertFalse(self.ma._opposite("多", "震荡"))
        self.assertFalse(self.ma._opposite("震荡", "震荡"))

    def test_implication_table(self):
        self.assertIn("方向背离", self.ma._implication("多", "空"))
        self.assertIn("方向背离", self.ma._implication("空", "多"))
        self.assertIn("上行需新资金", self.ma._implication("多", "震荡"))
        self.assertIn("等待方向确认", self.ma._implication("震荡", "震荡"))

    def test_degraded_record(self):
        d = self.ma._degraded("资金面", ValueError("坏数据"))
        self.assertEqual(d["agent"], "资金面")
        self.assertEqual(d["status"], "degraded")
        self.assertEqual(d["view"], "震荡")
        self.assertEqual(d["confidence"], 0.0)


class TestBuildDisagreements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _views(self):
        views = _make_views()
        # 情绪面看多(高置信)、资金面看空(高置信) → 相互对立 + 与共识相反
        views[0] = {"agent": "情绪面", "view": "多", "confidence": 0.9,
                    "evidence": ["情绪强"], "weight": 0.12, "status": "ok", "detail": {}}
        views[1] = {"agent": "资金面", "view": "空", "confidence": 0.8,
                    "evidence": ["资金弱"], "weight": 0.12, "status": "ok", "detail": {}}
        views[2] = {"agent": "题材面", "view": "多", "confidence": 0.3,   # 低置信不参与
                    "evidence": ["题材热"], "weight": 0.10, "status": "ok", "detail": {}}
        return views

    def test_conflicting_agents_produce_debate_points(self):
        views = self._views()
        votes = [{"agent": v["agent"], "view": v["view"], "confidence": v["confidence"],
                  "value": self.ma.VIEW_SCORE[v["view"]], "weight": 0.1, "status": v["status"]}
                 for v in views]
        dis = self.ma._build_disagreements(views, "震荡", votes, 0.4)
        between = {d["between"] for d in dis}
        self.assertIn("情绪面 vs 共识(震荡)", between)
        self.assertIn("资金面 vs 共识(震荡)", between)
        self.assertIn("情绪面 vs 资金面", between)
        for d in dis:
            self.assertIn("side_a", d)
            self.assertIn("side_b", d)
            self.assertIn("implication", d)

    def test_no_conflict_when_consensus_matches(self):
        views = _make_views()
        for v in views:
            v["view"] = "多"
            v["confidence"] = 0.8
        votes = [{"agent": v["agent"], "view": v["view"], "confidence": v["confidence"],
                  "value": 1.0, "weight": 0.1, "status": v["status"]} for v in views]
        dis = self.ma._build_disagreements(views, "多", votes, 0.9)
        self.assertEqual(dis, [])

    def test_degraded_excluded_from_disagreement(self):
        views = _make_views()
        views[0] = {"agent": "情绪面", "view": "多", "confidence": 0.9,
                    "evidence": [], "weight": 0.12, "status": "degraded", "detail": {}}
        votes = [{"agent": v["agent"], "view": v["view"], "confidence": v["confidence"],
                  "value": self.ma.VIEW_SCORE[v["view"]], "weight": 0.0,
                  "status": v["status"]} for v in views]
        dis = self.ma._build_disagreements(views, "震荡", votes, 0.3)
        self.assertFalse(any(d["between"].startswith("情绪面") for d in dis))


class TestArbitrate(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _bullish_views(self):
        views = _make_views()
        for v in views:
            v["view"] = "多"
            v["confidence"] = 0.9
            v["weight"] = self.ma.BASE_WEIGHTS[v["agent"]]
        return views

    def _arbitrate(self, views, date="2026-08-10"):
        self._tmp_out()
        with mock.patch.object(self.ma, "OUT_DIR", self._out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            return self.ma.arbitrate(date)

    def test_bullish_consensus_and_file(self):
        res = self._arbitrate(self._bullish_views())
        self.assertEqual(res["consensus"], "多")
        self.assertEqual(res["date"], "2026-08-10")
        self.assertGreater(res["weighted_sum"], 0.3)
        self.assertEqual(len(res["votes"]), 11)
        out = self._out / "multi_agent_2026-08-10.json"
        self.assertTrue(out.exists())
        saved = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(saved["consensus"], "多")
        self.assertIn("weights_note", saved)

    def test_bearish_consensus(self):
        views = self._bullish_views()
        for v in views:
            v["view"] = "空"
        res = self._arbitrate(views)
        self.assertEqual(res["consensus"], "空")
        self.assertLess(res["weighted_sum"], -0.3)

    def test_neutral_consensus(self):
        views = self._bullish_views()
        for v in views:
            v["view"] = "震荡"
            v["confidence"] = 0.5
        res = self._arbitrate(views)
        self.assertEqual(res["consensus"], "震荡")
        self.assertAlmostEqual(res["weighted_sum"], 0.0, places=6)

    def test_hard_veto_defensive(self):
        views = self._bullish_views()
        views[4] = {"agent": "风险面", "view": "防守", "confidence": 1.0,
                    "evidence": ["硬否决"], "weight": 0.12, "status": "ok",
                    "detail": {"date": "2026-08-10"}}
        res = self._arbitrate(views)
        self.assertEqual(res["consensus"], "防守")
        self.assertAlmostEqual(res["confidence"], 0.85)

    def test_degraded_agent_renormalizes_weights(self):
        views = self._bullish_views()
        views[0] = {"agent": "情绪面", "view": "震荡", "confidence": 0.0,
                    "evidence": ["模块异常: x"], "weight": 0.12, "status": "degraded",
                    "detail": {"error": "x"}}
        res = self._arbitrate(views)
        vote_w = {v["agent"]: v["weight"] for v in res["votes"]}
        self.assertEqual(vote_w["情绪面"], 0.0)
        self.assertAlmostEqual(sum(vote_w.values()), 1.0, places=4)
        self.assertIn("降级重分配[情绪面 失效]", res["weights_note"])

    def test_all_degraded_still_returns_structure(self):
        views = _make_views()
        for v in views:
            v["status"] = "degraded"
            v["confidence"] = 0.0
        res = self._arbitrate(views)
        self.assertEqual(res["consensus"], "震荡")
        self.assertEqual(len(res["votes"]), 11)
        # 全部降级时退回基础权重（非零），共识由 0 权重加权和决定
        self.assertTrue(any(v["weight"] > 0.0 for v in res["votes"]))


class TestDebate(_TmpOutMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def test_debate_builds_two_sided_logic(self):
        views = _make_views()
        views[0] = {"agent": "情绪面", "view": "多", "confidence": 0.9,
                    "evidence": ["涨停家数新高"], "weight": 0.12, "status": "ok",
                    "detail": {}}
        views[1] = {"agent": "资金面", "view": "空", "confidence": 0.8,
                    "evidence": ["北向净流出"], "weight": 0.12, "status": "ok",
                    "detail": {}}
        self._tmp_out()
        with mock.patch.object(self.ma, "OUT_DIR", self._out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            recs = self.ma.debate("2026-08-10")
        self.assertTrue(recs)
        rec = recs[0]
        self.assertIn("正方逻辑", rec)
        self.assertIn("反方逻辑", rec)
        self.assertTrue(rec["正方逻辑"]["logic"])
        self.assertTrue(rec["反方逻辑"]["logic"])
        self.assertTrue(any("情绪面 vs 资金面" == r["between"] for r in recs))

    def test_debate_empty_when_no_disagreement(self):
        views = _make_views()
        self._tmp_out()
        with mock.patch.object(self.ma, "OUT_DIR", self._out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            recs = self.ma.debate("2026-08-10")
        self.assertEqual(recs, [])


class TestDevilAdvocate(_TmpOutMixin, unittest.TestCase):
    """魔鬼代言人辩论（域L 增强）: 触发/攻讦/答辩评分/终审/audit_trail/兼容。"""

    @classmethod
    def setUpClass(cls):
        cls.ma = _import()

    def _close_views(self, win_evidence=None):
        """6:5 接近投票: 前 6 位（重权重）看多 0.95，后 5 位看空 0.7 → 共识多、胜率 6/11。"""
        views = _make_views()
        for i, v in enumerate(views):
            if i < 6:
                v["view"] = "多"
                v["confidence"] = 0.95
                v["evidence"] = [win_evidence[i]] if win_evidence else [f"赢家证据{i}"]
            else:
                v["view"] = "空"
                v["confidence"] = 0.7
                v["evidence"] = [f"输家证据{i}"]
        return views

    def _lopsided_views(self):
        """8:2+1震荡 悬殊投票 → 方向性胜率 8/10=80%，不触发辩论。"""
        views = _make_views()
        for i, v in enumerate(views):
            if i in (3, 10):
                v["view"], v["confidence"] = "空", 0.7
            elif i == 9:
                v["view"], v["confidence"] = "震荡", 0.3
            else:
                v["view"], v["confidence"] = "多", 0.95
        return views

    def _arbitrate(self, views, date="2026-08-10", **kw):
        self._tmp_out()
        with mock.patch.object(self.ma, "OUT_DIR", self._out), \
                mock.patch.object(self.ma, "agent_views", return_value=views):
            return self.ma.arbitrate(date, **kw)

    # ── 触发条件 ──
    def test_close_vote_6_5_triggers_devil_advocate(self):
        res = self._arbitrate(self._close_views())
        self.assertIn("devil_advocate", res)
        dev = res["devil_advocate"]
        self.assertTrue(dev["triggered"])
        self.assertAlmostEqual(dev["win_rate"], 6 / 11, places=4)
        self.assertEqual(dev["directional_votes"], {"aligned": 6, "opposing": 5})

    def test_lopsided_vote_8_2_not_triggered(self):
        res = self._arbitrate(self._lopsided_views())
        self.assertNotIn("devil_advocate", res)
        self.assertEqual(res["consensus"], "多")

    def test_win_rate_trigger_boundary(self):
        def votes(a, b):
            return ([{"view": "多", "status": "ok", "confidence": 0.9}] * a +
                    [{"view": "空", "status": "ok", "confidence": 0.7}] * b)
        lo, hi = self.ma.DEVIL_ADVOCATE_MIN, self.ma.DEVIL_ADVOCATE_MAX
        self.assertEqual(self.ma._win_rate("多", votes(9, 11)), lo)
        self.assertEqual(self.ma._win_rate("多", votes(13, 7)), hi)
        self.assertLess(self.ma._win_rate("多", votes(8, 10)), lo)
        self.assertGreater(self.ma._win_rate("多", votes(14, 7)), hi)
        self.assertIsNone(self.ma._win_rate("震荡", votes(6, 5)))

    def test_neutral_consensus_never_triggers(self):
        views = self._close_views()
        for v in views:
            v["confidence"] = 0.4  # 加权和落入震荡区间
        res = self._arbitrate(views)
        self.assertEqual(res["consensus"], "震荡")
        self.assertNotIn("devil_advocate", res)

    # ── 攻讦结构 ──
    def test_devil_advocate_generates_three_attacks(self):
        res = self._arbitrate(self._close_views())
        rounds = res["devil_advocate"]["rounds"]
        self.assertEqual(len(rounds), 3)
        for r in rounds:
            self.assertTrue(r["针对的证据"])
            self.assertTrue(r["反驳逻辑"])
            self.assertTrue(r["可能的反向场景"])

    def test_attacks_target_winning_evidence_from_opposing_side(self):
        res = self._arbitrate(self._close_views())
        dev = res["devil_advocate"]
        self.assertIn(dev["rounds"][0]["针对的证据"], dev["winning_side"]["top_evidence"])
        self.assertTrue(dev["opposing_side"]["experts"])
        self.assertIn(dev["opposing_side"]["experts"][0], dev["rounds"][0]["反驳逻辑"])

    # ── 答辩评分 ──
    def _rebuttal(self, cited, target="目标证据", consistent="一致"):
        return {"针对的证据": target, "引用证据": cited, "逻辑一致性": consistent}

    def test_score_rebuttal_valid(self):
        sc = self.ma._score_rebuttal(self._rebuttal(["新证据A"]))
        self.assertTrue(sc["valid"])
        self.assertGreater(sc["score"], 0)

    def test_score_rebuttal_invalid_when_no_new_evidence(self):
        sc = self.ma._score_rebuttal(self._rebuttal(["目标证据"]))
        self.assertFalse(sc["valid"])

    def test_score_rebuttal_invalid_when_logic_inconsistent(self):
        sc = self.ma._score_rebuttal(self._rebuttal(["新证据A"], consistent="不一致"))
        self.assertFalse(sc["valid"])

    def test_score_rebuttal_more_evidence_higher_score(self):
        one = self.ma._score_rebuttal(self._rebuttal(["新证据A"]))
        two = self.ma._score_rebuttal(self._rebuttal(["新证据A", "新证据B"]))
        self.assertGreater(two["score"], one["score"])
        self.assertTrue(two["valid"])

    # ── 终审 ──
    def test_final_verdict_upholds_with_two_valid(self):
        self.assertEqual(self.ma._final_verdict([True, True, False]), "维持")

    def test_final_verdict_divergence_below_two(self):
        self.assertEqual(self.ma._final_verdict([True, False, False]), "分歧")
        self.assertEqual(self.ma._final_verdict([False, False, False]), "分歧")

    def test_survives_three_rounds_maintains_consensus(self):
        res = self._arbitrate(self._close_views())
        dev = res["devil_advocate"]
        self.assertEqual(dev["verdict"], "维持")
        self.assertEqual(res["consensus"], "多")
        self.assertEqual(dev["final_consensus"], "多")
        self.assertGreaterEqual(sum(1 for r in dev["rounds"] if r["valid"]), 2)

    def test_fails_downgrades_to_divergence(self):
        views = self._close_views(win_evidence=["同一证据"] * 6)
        res = self._arbitrate(views)
        dev = res["devil_advocate"]
        self.assertEqual(dev["verdict"], "分歧")
        self.assertEqual(res["consensus"], "分歧")
        self.assertEqual(dev["final_consensus"], "分歧")
        self.assertLessEqual(res["confidence"], 0.5)

    # ── audit_trail 落盘 ──
    def test_audit_trail_written_on_trigger(self):
        self._arbitrate(self._close_views())
        p = self._out / "audit_trail" / "20260810.json"
        self.assertTrue(p.exists())
        rec = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(rec["kind"], "devil_advocate")
        self.assertEqual(rec["date"], "2026-08-10")
        self.assertEqual(rec["verdict"], "维持")
        self.assertEqual(rec["vote"]["consensus_before"], "多")
        self.assertEqual(rec["vote"]["consensus_after"], "多")
        self.assertEqual(len(rec["rounds"]), 3)
        self.assertTrue(rec["rounds"][0]["反驳逻辑"])
        self.assertTrue(rec["rounds"][0]["回应"])
        self.assertIn("score", rec["rounds"][0])
        self.assertIn("valid", rec["rounds"][0])

    def test_audit_trail_not_written_without_trigger(self):
        self._arbitrate(self._lopsided_views())
        self.assertFalse((self._out / "audit_trail" / "20260810.json").exists())

    # ── 兼容性 ──
    def test_compat_output_unchanged_when_not_triggered(self):
        views = _make_views()
        for v in views:
            v["view"] = "多"
            v["confidence"] = 0.9
            v["weight"] = self.ma.BASE_WEIGHTS[v["agent"]]
        res = self._arbitrate(views)
        self.assertNotIn("devil_advocate", res)
        self.assertEqual(res["consensus"], "多")
        self.assertEqual(len(res["votes"]), 11)
        self.assertIn("disagreement", res)
        self.assertIn("weights_note", res)

    def test_compat_devil_advocate_switch_keeps_old_output(self):
        res = self._arbitrate(self._close_views(), devil_advocate=False)
        self.assertNotIn("devil_advocate", res)
        self.assertEqual(res["consensus"], "多")
        self.assertAlmostEqual(res["confidence"], 0.65, places=2)

    def test_compat_debate_interface(self):
        self._tmp_out()
        with mock.patch.object(self.ma, "OUT_DIR", self._out), \
                mock.patch.object(self.ma, "agent_views", return_value=self._close_views()):
            recs = self.ma.debate("2026-08-10")
        self.assertTrue(recs)
        for rec in recs:
            self.assertIn("正方逻辑", rec)
            self.assertIn("反方逻辑", rec)


if __name__ == "__main__":
    unittest.main()

