"""analysis_core/battle_map 作战地图（V11 决策大脑）单元测试。

覆盖:
  - _safe 容错包装（异常→__error__ / 默认值）
  - _bid_watch 竞价锚点: 高板梯队/主线/风险股/涨停溢价兜底/上限3条
  - _risk_stocks: 高换手涨停/炸板股/上限5条/空表降级
  - _attack_groups: 主线/支线分组、官方龙头优先、fallback 龙头、打板禁用规则、
    core/observe 截断、空表降级
  - _load_pattern_explanations: 缺失/损坏/无当日 → {}，正常 → 规整 types
  - build_map 全链路（每日指定日期）: 数据健康/情绪/资金合力/共振/机制/宏观/
    产业链联动/个股风险/仓位/置信度/RAG 解释/落盘 JSON+MD
  - build_map 全降级: 各模块失败 → degraded 标注 + 不抛错 + 仓位×健康系数
  - render_md 各章节渲染（假设/专家委员会/游资博弈/降级/链上标注）

无网络: 全部 mock；MARKET_DIR/ZT_EM_DAILY/ROOT 路径常量覆盖到 tmp。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.battle_map as bm
    return bm


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="battle_map_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


TODAY = datetime.now().date()


def _stats_df(n=5, max_board=5, premium=1.5, zt_cnt=60):
    dates = pd.date_range(TODAY - timedelta(days=n), periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "zt_cnt": [zt_cnt] * n,
        "max_board": [max_board] * n,
        "zb_rate": [0.2] * n,
        "premium": [premium] * n,
        "jr1": [0.5] * n,
    })


def _em_df(target="2026-08-10"):
    rows = [
        {"date": pd.Timestamp(target), "code": "000001", "name": "龙头A",
         "is_zt": True, "is_zb": False, "turnover": 25.0, "board_count": 4, "amount": 1e8},
        {"date": pd.Timestamp(target), "code": "000002", "name": "股B",
         "is_zt": True, "is_zb": False, "turnover": 5.0, "board_count": 2, "amount": 1e7},
        {"date": pd.Timestamp(target), "code": "000003", "name": "炸板C",
         "is_zt": False, "is_zb": True, "turnover": 30.0, "board_count": 0, "amount": 1e7},
    ]
    return pd.DataFrame(rows)


def _resonance_df(date_str="2026-08-10"):
    return pd.DataFrame({
        "date": [date_str, date_str],
        "board_name": ["主线概念A", "支线概念B"],
        "concept": ["BK1000", "BK2000"],
        "level": ["主线", "支线"],
        "score": [3.2, 1.5],
    })


class TestSafe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def test_error_marker(self):
        def boom():
            raise RuntimeError("x")
        r = self.bm._safe(boom)
        self.assertIn("__error__", r)
        self.assertIn("x", r["__error__"])

    def test_default(self):
        def boom():
            raise ValueError("y")
        r = self.bm._safe(boom, None)
        self.assertIn("__error__", r)          # default=None → 错误标记（非 None 本身）
        self.assertEqual(self.bm._safe(lambda: 42, None), 42)


class TestBidWatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def test_anchors_capped_at_3(self):
        stats = _stats_df(max_board=5)
        resonance = [{"level": "主线", "board_name": "主线概念A"}]
        risks = [{"name": "龙头A"}]
        out = self.bm._bid_watch({}, stats, resonance, {}, risks)
        self.assertLessEqual(len(out), 3)
        joined = " ".join(b["item"] for b in out)
        # 最高板(2条) + 主线(1条) 已占满 3 条上限，风险股被截断 → 只有 3 条
        self.assertIn("最高板", joined)
        self.assertIn("主线[主线概念A]", joined)
        self.assertNotIn("风险股[龙头A]", joined)

    def test_risk_stock_anchor(self):
        stats = _stats_df(max_board=2)
        resonance = [{"level": "主线", "board_name": "主线概念A"}]
        risks = [{"name": "龙头A"}]
        out = self.bm._bid_watch({}, stats, resonance, {}, risks)
        joined = " ".join(b["item"] for b in out)
        self.assertIn("主线[主线概念A]", joined)
        self.assertIn("风险股[龙头A]", joined)

    def test_premium_fallback(self):
        stats = _stats_df(max_board=1, premium=-2.0)
        out = self.bm._bid_watch({}, stats, [], {}, [])
        self.assertEqual(len(out), 1)
        self.assertIn("负溢价禁打板", out[0]["signal"])
        stats2 = _stats_df(max_board=1, premium=1.0)
        out2 = self.bm._bid_watch({}, stats2, [], {}, [])
        self.assertIn("打板", out2[0]["action"])

    def test_no_stats(self):
        self.assertEqual(self.bm._bid_watch({}, None, [], {}, []), [])


class TestRiskStocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def test_high_turnover_and_zb(self):
        em = _em_df()
        risks = self.bm._risk_stocks(_stats_df(), em)
        codes = {r["code"] for r in risks}
        self.assertIn("000001", codes)   # 高换手涨停
        self.assertIn("000003", codes)   # 炸板
        self.assertLessEqual(len(risks), 5)

    def test_empty_em(self):
        self.assertEqual(self.bm._risk_stocks(_stats_df(), None), [])
        self.assertEqual(self.bm._risk_stocks(_stats_df(), pd.DataFrame()), [])


class TestAttackGroups(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def _concept_board(self, tmp):
        p = tmp / "classification"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "board_code": ["BK1000"],
            "board_name": ["主线概念A"],
            "leader_name": ["龙头A"],
        }).to_parquet(p / "concept_board.parquet", index=False)

    def test_groups_with_official_leader(self):
        tmp = self._make_tmp()
        self._concept_board(tmp)
        resonance = [{"level": "主线", "board_name": "主线概念A", "concept": "BK1000", "score": 3.2},
                     {"level": "支线", "board_name": "支线概念B", "concept": "BK2000", "score": 1.5},
                     {"level": "支线", "board_name": "支线概念B", "concept": "BK2000", "score": 1.5}]  # 重复 → 去重
        em = _em_df()
        code2con = {}
        con2codes = {"BK1000": {"000001", "000002"}, "BK2000": {"000003"}}
        with mock.patch.object(self.bm, "MARKET_DIR", tmp / "market"):
            groups = self.bm._attack_groups(resonance, em, {"rules": {"打板": "允许"}}, code2con, con2codes)
        self.assertEqual(len(groups["core"]), 1)
        self.assertEqual(groups["core"][0]["leader"]["code"], "000001")
        self.assertEqual(groups["core"][0]["strategy"], "打板(换手充分)")
        self.assertEqual(len(groups["observe"]), 1)
        self.assertEqual(groups["core_total"], 1)

    def test_ban_zhuo_uses_trend(self):
        tmp = self._make_tmp()
        self._concept_board(tmp)
        resonance = [{"level": "主线", "board_name": "主线概念A", "concept": "BK1000", "score": 3.2}]
        em = _em_df()
        with mock.patch.object(self.bm, "MARKET_DIR", tmp / "market"):
            groups = self.bm._attack_groups(
                resonance, em, {"rules": {"打板": "禁用"}}, {}, {"BK1000": {"000001"}})
        self.assertEqual(groups["core"][0]["strategy"], "趋势低吸")

    def test_fallback_leader_without_board_file(self):
        tmp = self._make_tmp()
        resonance = [{"level": "主线", "board_name": "主线概念A", "concept": "BK1000", "score": 3.2}]
        em = _em_df()
        # 无 concept_board 文件 → fallback 成分∩涨停 排序取首
        with mock.patch.object(self.bm, "MARKET_DIR", tmp / "market"):
            groups = self.bm._attack_groups(
                resonance, em, {"rules": {}}, {}, {"BK1000": {"000001", "000002"}})
        self.assertIsNotNone(groups["core"][0]["leader"])

    def test_empty_em(self):
        groups = self.bm._attack_groups([], None, {}, {}, {})
        self.assertEqual(groups["core"], [])


class TestLoadPatternExplanations(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def _write(self, tmp, payload):
        p = tmp / "data_warehouse" / "patterns"
        p.mkdir(parents=True, exist_ok=True)
        (p / "pattern_explanations.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_and_corrupt(self):
        tmp = self._make_tmp()
        with mock.patch.object(self.bm, "ROOT", tmp):
            self.assertEqual(self.bm._load_pattern_explanations("2026-08-10"), {})
        self._write(tmp, "{broken")
        with mock.patch.object(self.bm, "ROOT", tmp):
            self.assertEqual(self.bm._load_pattern_explanations("2026-08-10"), {})
        self._write(tmp, [1, 2, 3])
        with mock.patch.object(self.bm, "ROOT", tmp):
            self.assertEqual(self.bm._load_pattern_explanations("2026-08-10"), {})

    def test_no_day_key(self):
        tmp = self._make_tmp()
        self._write(tmp, {"2026-08-09": {"x": {}}})
        with mock.patch.object(self.bm, "ROOT", tmp):
            self.assertEqual(self.bm._load_pattern_explanations("2026-08-10"), {})

    def test_valid_day(self):
        tmp = self._make_tmp()
        self._write(tmp, {"2026-08-10": {
            "head_shoulder": {"query": "q1", "pattern_count": 3,
                              "hits": [{"cat": "c", "file": "f.md", "score": 0.9,
                                        "summary": "s"}, "junk"]},
            "not_a_dict": "x",
        }})
        with mock.patch.object(self.bm, "ROOT", tmp):
            r = self.bm._load_pattern_explanations("2026-08-10")
        self.assertEqual(r["date"], "2026-08-10")
        self.assertEqual(r["total_types"], 1)
        t = r["types"][0]
        self.assertEqual(t["type"], "head_shoulder")
        self.assertEqual(t["pattern_count"], 3)
        self.assertEqual(len(t["hits"]), 1)
        self.assertTrue(t["cached"])


def _mock_env(bm_mod, tmp, degraded=False):
    """为 build_map 挂载全套 mock；degraded=True 时全部失败。"""
    today = datetime.now().date()
    target = "2026-08-10"

    def _stats():
        if degraded:
            raise RuntimeError("read fail")
        frame = _stats_df()
        frame["date"] = pd.date_range("2026-08-04", periods=len(frame), freq="B")
        return frame

    def _health(*a, **k):
        if degraded:
            return None
        return {"overall_ok": True, "degraded_datasets": []}

    def _emotion():
        if degraded:
            raise RuntimeError("no data")
        return {"stage": "ferment", "stage_cn": "发酵", "confidence": 0.7}

    def _forces():
        if degraded:
            raise RuntimeError("no data")
        return pd.DataFrame({"date": [pd.Timestamp(target)],
                             "force_index": [70.0]})

    def _score(date_str):
        if degraded:
            return pd.DataFrame()
        return _resonance_df(target)

    def _regime(*a, **k):
        return {} if degraded else {"regime": "亢奋", "rules": {"打板": "允许", "仓位系数": 1.0}}

    def _veto(*a, **k):
        return {} if degraded else {"level": "none", "position_coef": 1.0}

    def _links(board, top_k=3):
        if degraded:
            raise RuntimeError("no graph")
        return [{"dst": "传导板块X", "lift": 0.9}]

    def _chain(board, date_str):
        if degraded:
            raise RuntimeError("no chain")
        return [{"chain": "新能源车", "layer": "midstream", "sectors": ["电池", "整车"]}]

    def _card():
        return {} if degraded else {"position_range": "20-50%"}

    def _weights(*a, **k):
        return {} if degraded else {"emotion_cycle": 1.0, "scenario": 1.0}

    def _micro(date_str):
        if degraded:
            raise RuntimeError("no micro")
        return {"distribution_risks": [{"code": "000004", "name": "出货D",
                                        "judge": "出货", "zb_times": 2}]}

    def _signals(n=4):
        if degraded:
            raise RuntimeError("no signals")
        return [{"board_name": "主线概念A", "signal": "过热",
                 "reason": "连续多日拥挤"}]

    def _hyp(date_str):
        return [] if degraded else [{"hypothesis": "H1", "evidence": "E1",
                                     "action": "A1", "confidence": 0.5}]

    def _arb(date_str):
        return {} if degraded else {"consensus": "多", "confidence": 0.7,
                                    "votes": [{"agent": "a", "view": "多", "confidence": 0.6}],
                                    "disagreement": [{"between": "a/b", "issue": "x"}]}

    def _game(date_str):
        return {} if degraded else {"strategy": "S1", "ev": 1.2, "win_rate": 0.6, "reason": "R1"}

    def _social():
        # V12.3 情绪接入: battle_map 集成市场社媒情绪轴。测试环境返回"无数据"(不联网),
        # 使 conf_bump=0, 保持原有 confidence 打点; degraded 时也应优雅空。
        return {"market_sentiment": None, "coverage": 0.0,
                "per_source": {}, "_mock_": True} if not degraded else {"_mock_": True, "error": "no social"}

    return [
        mock.patch.object(bm_mod, "MARKET_DIR", tmp / "data_warehouse" / "market"),
        mock.patch.object(bm_mod, "ZT_EM_DAILY", tmp / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"),
        mock.patch.object(bm_mod, "ROOT", tmp),
        mock.patch.object(bm_mod, "_latest_stats", side_effect=_stats),
        mock.patch.object(bm_mod.resonance_scorer, "score_day", side_effect=_score),
        mock.patch.object(bm_mod.resonance_scorer, "_load_concept_map",
                          return_value=({}, {"BK1000": {"000001", "000002"},
                                             "BK2000": {"000003"}})),
        mock.patch.object(bm_mod, "get_regime", side_effect=_regime),
        mock.patch.object(bm_mod, "get_veto", side_effect=_veto),
        mock.patch.object(bm_mod.industry_graph, "links", side_effect=_links),
        mock.patch("quant_system.analysis_core.chain_map.propagate_from_concept", side_effect=_chain),
        mock.patch("quant_system.analysis_core.data_health_check.run", side_effect=_health),
        mock.patch("quant_system.analysis_core.emotion_cycle.run_today", side_effect=_emotion),
        mock.patch("quant_system.analysis_core.fund_forces.latest", side_effect=_forces),
        mock.patch("quant_system.analysis_core.decision_card.build_card", side_effect=_card),
        mock.patch("quant_system.analysis_core.calibration.get_module_weights", side_effect=_weights),
        mock.patch("quant_system.analysis_core.market_microstructure.run_today", side_effect=_micro),
        mock.patch("quant_system.analysis_core.leader_follower.top_signals", side_effect=_signals),
        mock.patch("quant_system.analysis_core.hypothesis_generator.generate", side_effect=_hyp),
        mock.patch("quant_system.analysis_core.multi_agent.arbitrate", side_effect=_arb),
        mock.patch("quant_system.analysis_core.game_theory_model.current_game", side_effect=_game),
        mock.patch("quant_system.analysis_core.social_sentiment.market_sentiment_composite", side_effect=_social),
    ]


class TestBuildMap(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def _write_em_and_pat(self, tmp, target="2026-08-10"):
        market = tmp / "data_warehouse" / "market"
        market.mkdir(parents=True, exist_ok=True)
        _em_df(target).to_parquet(market / "zt_pool_em_daily.parquet", index=False)
        cls = tmp / "data_warehouse" / "classification"
        cls.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "board_code": ["BK1000"], "board_name": ["主线概念A"], "leader_name": ["龙头A"],
        }).to_parquet(cls / "concept_board.parquet", index=False)
        pat = tmp / "data_warehouse" / "patterns"
        pat.mkdir(parents=True, exist_ok=True)
        (pat / "pattern_explanations.json").write_text(json.dumps({
            "2026-08-10": {"head_shoulder": {"query": "q", "pattern_count": 3,
                                             "hits": [{"cat": "c", "file": "f.md",
                                                       "score": 0.9, "summary": "s"}]}},
        }), encoding="utf-8")

    def test_full_build(self):
        tmp = self._make_tmp()
        self._write_em_and_pat(tmp)
        for p in _mock_env(self.bm, tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        bm = self.bm.build_map("2026-08-10")
        self.assertEqual(bm["date"], "2026-08-10")
        self.assertEqual(bm["emotion_stage"], "发酵")
        self.assertEqual(bm["regime"], "亢奋")
        self.assertEqual(bm["macro_veto"], "none")
        self.assertTrue(bm["bid_watch"])
        self.assertEqual(len(bm["attack_groups"]["core"]), 1)
        self.assertTrue(bm["risk_watch"])
        # 个股风险: 高换手 + 炸板 + 微观出货 + 跟风过热
        risks = " ".join(f"{r['name']}:{r['risk']}" for r in bm["risk_watch"])
        self.assertIn("龙头A", risks)
        self.assertIn("炸板C", risks)
        self.assertIn("出货D", risks)
        self.assertIn("过热", risks)
        # 产业链联动: industry_graph + chain_map 先验链
        self.assertTrue(bm["linkage"])
        lk = bm["linkage"][0]
        self.assertEqual(lk["from"], "主线概念A")
        self.assertEqual(lk["chain"], "新能源车")
        # RAG 解释
        self.assertEqual(bm["rag_explanations"]["total_types"], 1)
        # 仓位: 20-50% × 1.0 × 1.0 × 1.0
        self.assertEqual(bm["position_range"], "20-50%")
        # 历史日期不计入当前社交情绪，避免未来信息污染；不可回放的
        # 社交源触发降级扣分，最终置信度为 0.65。
        self.assertEqual(bm["confidence"], 0.65)
        self.assertEqual(bm["confidence_components"]["final"], bm["confidence"])
        self.assertAlmostEqual(bm["confidence_components"]["raw"], 0.65)
        self.assertEqual(bm["recommended"], "进攻")
        self.assertEqual(bm["degraded"], ["social_sentiment 不支持历史日期，已排除当前快照"])
        # 落盘 JSON + MD
        self.assertTrue((tmp / "generated" / "battle_map_2026-08-10.json").exists())
        self.assertTrue((tmp / "generated" / "battle_map_2026-08-10.md").exists())

    def test_degraded_build(self):
        tmp = self._make_tmp()
        for p in _mock_env(self.bm, tmp, degraded=True):
            p.start()
        self.addCleanup(mock.patch.stopall)
        bm = self.bm.build_map("2026-08-10")
        self.assertIn("date", bm)
        self.assertIn("degraded", bm)
        self.assertTrue(bm["degraded"])
        joined = " | ".join(bm["degraded"])
        self.assertIn("数据健康", joined)
        self.assertIn("resonance", joined)
        self.assertIn("pattern_explanations 缓存缺失", joined)
        self.assertEqual(bm["position_range"], "8-24%")  # 10-30% × 0.8 健康系数
        self.assertEqual(bm["recommended"], "防守")
        self.assertEqual(bm["risk_watch"], [])
        self.assertIn("未知", bm["emotion_stage"])

    def test_default_date_today(self):
        tmp = self._make_tmp()
        self._write_em_and_pat(tmp, target=TODAY.isoformat())
        for p in _mock_env(self.bm, tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        bm = self.bm.build_map(None)
        self.assertEqual(bm["date"], TODAY.isoformat())
        self.assertTrue((tmp / "generated" / f"battle_map_{TODAY.isoformat()}.json").exists())


class TestRenderMd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm = _import()

    def _bm(self):
        return {
            "date": "2026-08-10",
            "emotion_stage": "发酵", "regime": "亢奋", "macro_veto": "none",
            "confidence": 0.7, "recommended": "进攻",
            "position_range": "20-50%", "position_basis": "决策卡20-50% × 宏观1.0 × 机制1.0 × 健康1.0",
            "bid_watch": [{"item": "i1", "signal": "乐观", "action": "持有"}],
            "attack_groups": {
                "core": [{"name": "主线概念A", "score": 3.2, "strategy": "打板",
                          "leader": {"name": "龙头A", "boards": 4}},
                         {"name": "无龙头", "score": 1.0, "strategy": "打板", "leader": None}],
                "observe": [{"name": "支线概念B", "score": 1.5, "strategy": "轻仓试错"}],
            },
            "risk_watch": [{"name": "龙头A", "code": "000001", "risk": "高换手", "action": "走"}],
            "linkage": [
                {"from": "主线概念A", "to": ["传导板块X(lift0.9)"],
                 "chain": "新能源车", "layer": "midstream", "sectors": ["电池", "整车"]},
                {"from": "纯链", "to": [], "chain": "光伏", "layer": "upstream", "sectors": ["硅料"]},
            ],
            "rag_explanations": {"types": [{"type": "t1", "name": "头肩顶", "pattern_count": 3,
                                            "query": "q", "cached": True,
                                            "hits": [{"cat": "c", "file": "f.md",
                                                      "score": 0.9, "summary": "摘要"}]}]},
            "degraded": ["resonance: 无评分"],
        }

    def test_render_sections(self):
        bm = self._bm()
        with mock.patch("quant_system.analysis_core.hypothesis_generator.generate",
                        return_value=[{"hypothesis": "H1", "evidence": "E1",
                                       "action": "A1", "confidence": 0.5}]), \
                mock.patch("quant_system.analysis_core.multi_agent.arbitrate",
                           return_value={"consensus": "多", "confidence": 0.7,
                                         "votes": [{"agent": "a", "view": "多",
                                                    "confidence": 0.6}],
                                         "disagreement": [{"between": "a/b", "issue": "x"}]}), \
                mock.patch("quant_system.analysis_core.game_theory_model.current_game",
                           return_value={"strategy": "S1", "ev": 1.2, "win_rate": 0.6,
                                         "reason": "R1"}):
            md = self.bm.render_md(bm)
        for section in ("❶ 竞价观察锚点", "❷ 攻击方向", "❸ 个股风险清单",
                        "❹ 产业链联动", "💡 今日假设", "🧠 专家委员会",
                        "📚 规律逻辑依据", "⚔️ 游资博弈格局", "⚠️ 数据降级"):
            self.assertIn(section, md)
        self.assertIn("新能源车·中游 制造 电池/整车", md)
        self.assertIn("光伏·上游 原料 硅料", md)
        self.assertIn("共识: **多**", md)

    def test_render_degraded_edges(self):
        bm = self._bm()
        bm["risk_watch"] = []
        bm["linkage"] = []
        bm["rag_explanations"] = {"types": []}
        with mock.patch("quant_system.analysis_core.hypothesis_generator.generate",
                        side_effect=RuntimeError("no data")), \
                mock.patch("quant_system.analysis_core.multi_agent.arbitrate",
                           return_value={}), \
                mock.patch("quant_system.analysis_core.game_theory_model.current_game",
                           return_value={}):
            md = self.bm.render_md(bm)
        self.assertIn("无（数据未覆盖）", md)
        self.assertIn("无强传导", md)
        self.assertIn("当日无规律解释缓存", md)
        self.assertNotIn("💡 今日假设", md)


if __name__ == "__main__":
    unittest.main()
