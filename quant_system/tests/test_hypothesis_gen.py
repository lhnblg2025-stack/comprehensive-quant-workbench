"""analysis_core/hypothesis_generator 生成式策略假设引擎单元测试。

覆盖:
  - _classify_stage 全分支（高潮/分歧/退潮/冰点/发酵/修复/未知）
  - 数据装载 _load_stats/_load_fund/_load_theme: 缺失→空、≤target 截断
  - _load_battle_map: 目标日/请求日读取、缺失、损坏降级
  - _build_snapshot 防前视: 请求未来日期 → target 钳制到本地数据最大日
  - 模板 A/B/C/D 全逻辑 + _template_generate 编排
  - _llm_enabled/_llm_generate: 未配置→None、网络异常→None、非法JSON→None、
    正常→items、字段缺失过滤
  - generate(): 模板通道落盘、LLM 通道、LLM 失败回退模板、无数据空列表、k 钳制

无网络：urllib 全程 mock；路径常量 mock.patch.object 覆盖到 tmp。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.hypothesis_generator as hg
    return hg


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="hyp_gen_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


def _stats_df(n=12, zt=100, mb=5, zb=0.20, premium=1.5, jr1=0.5):
    dates = pd.date_range("2026-07-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "zt_cnt": [zt] * n,
        "max_board": [mb] * n,
        "zb_rate": [zb] * n,
        "premium": [premium] * n,
        "jr1": [jr1] * n,
    })


def _fund_df(n=4, youzi=-2.5, force=60.0, start="2026-07-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "force_index": [force] * n,
        "signs": [json.dumps({"游资": youzi})] * n,
    })


def _theme_df(n=3, zt_cnt=6, max_board=5):
    dates = pd.date_range("2026-07-01", periods=n, freq="B")
    return pd.DataFrame({
        "concept": ["BK1"] * n,
        "date": dates,
        "zt_cnt": [zt_cnt] * n,
        "max_board": [max_board] * n,
        "board_name": ["主线板块"] * n,
        "role": ["主线"] * n,
    })


class TestClassifyStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def _row(self, **kw):
        base = {"zt_cnt": 60, "max_board": 3, "zb_rate": 0.20, "premium": 0.5}
        base.update(kw)
        return pd.Series(base)

    def test_climax(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=100, max_board=6)), "高潮")

    def test_divergence(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=80, zb_rate=0.45)), "分歧")

    def test_retreat(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=40, max_board=2)), "退潮")

    def test_ice_point(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=30, max_board=4)), "冰点")

    def test_ferment(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=60, max_board=3)), "发酵")

    def test_repair(self):
        self.assertEqual(self.hg._classify_stage(self._row(zt_cnt=45, max_board=3, zb_rate=0.2)), "修复")

    def test_unknown(self):
        self.assertEqual(self.hg._classify_stage(pd.Series({"zt_cnt": None})), "未知")


class TestLoaders(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_load_stats_missing(self):
        tmp = self._make_tmp()
        with mock.patch.object(self.hg, "ZT_DAILY_STATS", tmp / "nope.parquet"):
            df, row = self.hg._load_stats(pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertIsNone(row)

    def test_load_stats_truncates_to_target(self):
        tmp = self._make_tmp()
        f = tmp / "zt.parquet"
        df = _stats_df(n=6)
        df["date"] = df["date"] + pd.Timedelta(days=0)
        df.to_parquet(f)
        target = df["date"].iloc[3]
        with mock.patch.object(self.hg, "ZT_DAILY_STATS", f):
            out, row = self.hg._load_stats(target)
        self.assertEqual(len(out), 4)  # <= target 仅 4 行
        self.assertEqual(pd.Timestamp(row["date"]), target)

    def test_load_stats_empty_returns_none(self):
        tmp = self._make_tmp()
        f = tmp / "zt.parquet"
        pd.DataFrame(columns=["date", "zt_cnt"]).to_parquet(f)
        with mock.patch.object(self.hg, "ZT_DAILY_STATS", f):
            df, row = self.hg._load_stats(pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)

    def test_load_fund_missing_and_truncate(self):
        tmp = self._make_tmp()
        f = tmp / "fund.parquet"
        with mock.patch.object(self.hg, "FUND_FORCES", tmp / "nope.parquet"):
            self.assertTrue(self.hg._load_fund(pd.Timestamp("2026-08-12")).empty)
        fund = _fund_df(n=4)
        fund.to_parquet(f)
        with mock.patch.object(self.hg, "FUND_FORCES", f):
            out = self.hg._load_fund(fund["date"].iloc[2])
        self.assertEqual(len(out), 3)

    def test_load_theme_missing_and_truncate(self):
        tmp = self._make_tmp()
        f = tmp / "theme.parquet"
        with mock.patch.object(self.hg, "THEME_CYCLE", tmp / "nope.parquet"):
            self.assertTrue(self.hg._load_theme(pd.Timestamp("2026-08-12")).empty)
        th = _theme_df(n=4)
        th.to_parquet(f)
        with mock.patch.object(self.hg, "THEME_CYCLE", f):
            out = self.hg._load_theme(th["date"].iloc[1])
        self.assertEqual(len(out), 2)

    def test_load_battle_map(self):
        tmp = self._make_tmp()
        gen = tmp / "generated"
        gen.mkdir()
        (gen / "battle_map_2026-08-10.json").write_text(
            json.dumps({"regime": "亢奋", "position_range": "10-30%"}), encoding="utf-8")
        with mock.patch.object(self.hg, "GENERATED_DIR", gen):
            bm = self.hg._load_battle_map(pd.Timestamp("2026-08-10"), "2026-08-10")
        self.assertEqual(bm["regime"], "亢奋")

    def test_load_battle_map_missing_and_corrupt(self):
        tmp = self._make_tmp()
        gen = tmp / "generated"
        gen.mkdir()
        with mock.patch.object(self.hg, "GENERATED_DIR", gen):
            self.assertEqual(self.hg._load_battle_map(pd.Timestamp("2026-08-10"), None), {})
        (gen / "battle_map_2026-08-10.json").write_text("{broken", encoding="utf-8")
        with mock.patch.object(self.hg, "GENERATED_DIR", gen):
            self.assertEqual(self.hg._load_battle_map(pd.Timestamp("2026-08-10"), "2026-08-10"), {})


class TestBuildSnapshot(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def _write_all(self, tmp):
        stats = _stats_df(n=8)
        stats["date"] = pd.date_range("2026-08-01", periods=8, freq="B")
        stats.to_parquet(tmp / "zt.parquet")
        _fund_df(n=4, start="2026-08-01").to_parquet(tmp / "fund.parquet")
        _theme_df(n=4, zt_cnt=6).to_parquet(tmp / "theme.parquet")
        return stats

    def _patch(self, tmp):
        return [
            mock.patch.object(self.hg, "ZT_DAILY_STATS", tmp / "zt.parquet"),
            mock.patch.object(self.hg, "FUND_FORCES", tmp / "fund.parquet"),
            mock.patch.object(self.hg, "THEME_CYCLE", tmp / "theme.parquet"),
            mock.patch.object(self.hg, "GENERATED_DIR", tmp / "generated"),
        ]

    def test_build_snapshot_clamps_future_date(self):
        """防前视硬断言：请求未来日期 → target 钳制到本地最大日，绝不使用未来数据。"""
        tmp = self._make_tmp()
        stats = self._write_all(tmp)
        for p in self._patch(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        max_date = stats["date"].max().date().isoformat()
        snapshot, target = self.hg._build_snapshot("2099-01-01")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["date"], max_date)
        self.assertEqual(target.date().isoformat(), max_date)
        self.assertEqual(snapshot["llm_snapshot"]["date"], max_date)

    def test_build_snapshot_default_uses_max(self):
        tmp = self._make_tmp()
        stats = self._write_all(tmp)
        for p in self._patch(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        snapshot, target = self.hg._build_snapshot(None)
        self.assertEqual(snapshot["date"], stats["date"].max().date().isoformat())

    def test_build_snapshot_missing_file(self):
        tmp = self._make_tmp()
        for p in self._patch(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(self.hg._build_snapshot(None), (None, None))

    def test_build_snapshot_empty_df(self):
        tmp = self._make_tmp()
        pd.DataFrame(columns=["date"]).to_parquet(tmp / "zt.parquet")
        for p in self._patch(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(self.hg._build_snapshot(None), (None, None))

    def test_llm_snapshot_contains_battle_map_fields(self):
        tmp = self._make_tmp()
        self._write_all(tmp)
        gen = tmp / "generated"
        gen.mkdir()
        (gen / "battle_map_2026-08-10.json").write_text(json.dumps({
            "regime": "亢奋", "position_range": "30-50%", "recommended": "进攻",
            "confidence": 0.8, "emotion_stage": "发酵"}), encoding="utf-8")
        for p in self._patch(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        snapshot, _ = self.hg._build_snapshot("2026-08-10")
        bm = snapshot["llm_snapshot"]["battle_map"]
        self.assertEqual(bm["regime"], "亢奋")
        self.assertEqual(snapshot["llm_snapshot"]["emotion_stage"], "发酵")
        self.assertIn("zt_cnt", snapshot["llm_snapshot"]["zt_stats"])


def _template_a_df():
    """构造: 末日单日崩塌 3→1 板 + 历史 ≥3 次同型回落。"""
    rows = [
        # date, zt_cnt, max_board, zb_rate, jr1
        ("2026-07-01", 60, 5, 0.30, 0.50),
        ("2026-07-02", 55, 3, 0.25, 0.40),   # diff -2
        ("2026-07-03", 52, 3, 0.20, 0.50),
        ("2026-07-06", 50, 2, 0.22, 0.40),   # diff -1
        ("2026-07-07", 48, 0, 0.18, 0.50),   # diff -2
        ("2026-07-08", 45, 0, 0.20, 0.40),
        ("2026-07-09", 40, 2, 0.15, 0.50),
        ("2026-07-10", 42, 0, 0.18, 0.40),   # diff -2
        ("2026-07-13", 38, 3, 0.12, 0.50),
        ("2026-07-14", 35, 1, 0.10, 0.40),   # 当日: 单日崩塌 -2
    ]
    df = pd.DataFrame(rows, columns=["date", "zt_cnt", "max_board", "zb_rate", "jr1"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestTemplateA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_single_day_collapse(self):
        item = self.hg._template_a(_template_a_df())
        self.assertIsNotNone(item)
        self.assertEqual(item["template"], "A")
        self.assertIn("炸板率虽<40%", item["hypothesis"])
        self.assertIn("3板崩塌至1板", item["hypothesis"])
        self.assertIn("打板胜率", item["evidence"])
        self.assertGreaterEqual(item["confidence"], 0.30)

    def test_high_zb_rate_returns_none(self):
        df = _template_a_df().copy()
        df.loc[df.index[-1], "zb_rate"] = 0.50
        self.assertIsNone(self.hg._template_a(df))

    def test_flat_boards_returns_none(self):
        df = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=10, freq="B"),
            "zt_cnt": [50] * 10, "max_board": [4] * 10,
            "zb_rate": [0.2] * 10, "jr1": [0.5] * 10,
        })
        self.assertIsNone(self.hg._template_a(df))

    def test_insufficient_history_returns_none(self):
        df = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=4, freq="B"),
            "zt_cnt": [50, 48, 46, 30], "max_board": [5, 4, 3, 1],
            "zb_rate": [0.2, 0.2, 0.2, 0.1], "jr1": [0.5] * 4,
        })
        self.assertIsNone(self.hg._template_a(df))


class TestTemplateB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_streak_detected(self):
        item = self.hg._template_b(_theme_df(n=3, zt_cnt=6))
        self.assertIsNotNone(item)
        self.assertEqual(item["template"], "B")
        self.assertIn("连续大涨3日", item["hypothesis"])
        self.assertEqual(item["confidence"], round(min(0.85, 0.45 + 0.02 * 3), 2))

    def test_empty_returns_none(self):
        self.assertIsNone(self.hg._template_b(pd.DataFrame()))

    def test_short_streak_returns_none(self):
        th = _theme_df(n=2, zt_cnt=3)  # zt_cnt < ZT_SURGE
        self.assertIsNone(self.hg._template_b(th))

    def test_best_streak_chosen(self):
        th = pd.DataFrame({
            "concept": ["BK1"] * 3 + ["BK2"] * 2,
            "date": list(pd.date_range("2026-07-01", periods=3, freq="B"))
                    + list(pd.date_range("2026-07-01", periods=2, freq="B")),
            "zt_cnt": [6, 6, 6, 5, 5],
            "max_board": [4, 5, 6, 3, 3],
            "board_name": ["长主线"] * 3 + ["短支线"] * 2,
            "role": ["主线"] * 5,
        })
        item = self.hg._template_b(th)
        self.assertIn("长主线", item["hypothesis"])


class TestTemplateC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def _setup(self, stage="高潮", youzi=-2.5):
        df = _stats_df(n=5)
        fund = _fund_df(n=5, youzi=youzi)
        bm = {"emotion_stage": stage}
        return df, fund, bm

    def test_positive_divergence(self):
        df, fund, bm = self._setup("高潮", -2.5)
        item = self.hg._template_c(df, fund, bm)
        self.assertIsNotNone(item)
        self.assertEqual(item["template"], "C")
        self.assertIn("背离", item["hypothesis"])
        self.assertIn("净流出", item["evidence"])

    def test_negative_divergence(self):
        df = _stats_df(n=5, zt=30, mb=4)  # 历史行分类为冰点
        fund = _fund_df(n=5, youzi=3.0)
        bm = {"emotion_stage": "冰点"}
        item = self.hg._template_c(df, fund, bm)
        self.assertIsNotNone(item)
        self.assertEqual(item["template"], "C")
        self.assertIn("情绪弱但资金进场", item["action"])

    def test_empty_fund_returns_none(self):
        df, _, bm = self._setup()
        self.assertIsNone(self.hg._template_c(df, pd.DataFrame(), bm))

    def test_no_divergence_returns_none(self):
        df, fund, bm = self._setup("高潮", 3.0)  # 正向情绪+游资流入 → 无背离
        self.assertIsNone(self.hg._template_c(df, fund, bm))

    def test_insufficient_history_returns_none(self):
        df = _stats_df(n=3)
        fund = _fund_df(n=3, youzi=-2.5)
        item = self.hg._template_c(df, fund, {"emotion_stage": "高潮"})
        self.assertIsNone(item)  # hist 去除当日后 <3

    def test_missing_signs_returns_none(self):
        df = _stats_df(n=5)
        fund = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=5, freq="B"),
            "force_index": [60] * 5,
            "signs": [None] * 5,
        })
        self.assertIsNone(self.hg._template_c(df, fund, {"emotion_stage": "高潮"}))

    def test_stage_from_stats_fallback(self):
        df = _stats_df(n=5, zt=100, mb=6)  # 高潮
        fund = _fund_df(n=5, youzi=-2.5)
        item = self.hg._template_c(df, fund, {})  # bm 无 emotion_stage → 用 _classify_stage
        self.assertIsNotNone(item)


def _template_d_df():
    rows = [
        ("2026-07-01", 50, 3, 0.18, 0.5),
        ("2026-07-02", 60, 3, 0.22, 0.5),
        ("2026-07-03", 70, 3, 0.20, 0.5),
        ("2026-07-06", 80, 5, 0.10, 0.5),
        ("2026-07-07", 90, 5, 0.10, 0.5),
        ("2026-07-08", 95, 6, 0.05, 0.5),
        ("2026-07-09", 55, 4, 0.30, 0.5),
        ("2026-07-10", 65, 3, 0.20, 0.5),   # 当日
    ]
    df = pd.DataFrame(rows, columns=["date", "zt_cnt", "max_board", "zb_rate", "jr1"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestTemplateD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_combo_hist(self):
        item = self.hg._template_d(_template_d_df())
        self.assertIsNotNone(item)
        self.assertEqual(item["template"], "D")
        self.assertIn("最高板3板+炸板率20.0%组合", item["hypothesis"])
        self.assertIn("次日总仓位系数", item["action"])

    def test_missing_values_returns_none(self):
        df = _template_d_df().copy()
        df.loc[df.index[-1], "max_board"] = None
        self.assertIsNone(self.hg._template_d(df))

    def test_insufficient_hist_returns_none(self):
        # 历史行均为最高板3板但炸板率远离(0.60) → 无同组合 → None
        df = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=4, freq="B"),
            "zt_cnt": [50, 60, 70, 65], "max_board": [3, 3, 3, 3],
            "zb_rate": [0.60, 0.60, 0.60, 0.20], "jr1": [0.5] * 4,
        })
        self.assertIsNone(self.hg._template_d(df))


class _HttpResp:
    """最小 http.client.HTTPResponse 替身（read 返回固定 bytes）。"""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


class TestTemplateGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_orchestration_caps_k(self):
        snapshot = {
            "stats_df": _template_a_df(),
            "theme_df": _theme_df(),
            "fund_df": _fund_df(),
            "battle_map": {"emotion_stage": "高潮"},
        }
        items = self.hg._template_generate(snapshot, k=2)
        self.assertLessEqual(len(items), 2)
        for it in items:
            self.assertEqual(it["source"], "template")
            self.assertIn("hypothesis", it)
        # k=1 → 只保留第一条（模板A）
        items1 = self.hg._template_generate(snapshot, k=1)
        self.assertEqual(len(items1), 1)
        self.assertEqual(items1[0]["template"], "A")

    def test_builder_exception_skipped(self):
        snapshot = {"stats_df": None, "theme_df": pd.DataFrame(),
                    "fund_df": pd.DataFrame(), "battle_map": {}}
        with mock.patch.object(self.hg, "_template_a", side_effect=RuntimeError("boom")):
            items = self.hg._template_generate(snapshot, k=3)
        self.assertEqual(items, [])


class TestLlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def test_enabled_flag(self):
        with mock.patch.dict("os.environ", {"HYPOTHESIS_LLM": "1"}):
            self.assertTrue(self.hg._llm_enabled())
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(self.hg._llm_enabled(), os.environ.get("HYPOTHESIS_LLM", "0") == "1")

    def test_no_config_returns_none(self):
        snapshot = {"llm_snapshot": {}}
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(self.hg._llm_generate(snapshot, 3))

    def test_network_error_returns_none(self):
        snapshot = {"llm_snapshot": {"date": "2026-08-10"}}
        with mock.patch.dict("os.environ", {
                "HYPOTHESIS_LLM_BASE": "http://x", "HYPOTHESIS_LLM_KEY": "k"}), \
                mock.patch("urllib.request.urlopen", side_effect=RuntimeError("no net")):
            self.assertIsNone(self.hg._llm_generate(snapshot, 3))

    def test_invalid_json_returns_none(self):
        snapshot = {"llm_snapshot": {}}
        with mock.patch.dict("os.environ", {
                "HYPOTHESIS_LLM_BASE": "http://x", "HYPOTHESIS_LLM_KEY": "k"}), \
                mock.patch("urllib.request.urlopen",
                           return_value=_HttpResp(b'{"choices": [{"message": {"content": "not json"}}]}')):
            self.assertIsNone(self.hg._llm_generate(snapshot, 3))

    def test_valid_response(self):
        snapshot = {"llm_snapshot": {}}
        body = json.dumps([{"hypothesis": "H", "evidence": "E", "action": "A", "confidence": 0.6}])
        content = json.dumps({"choices": [{"message": {"content": body}}]}).encode()
        with mock.patch.dict("os.environ", {
                "HYPOTHESIS_LLM_BASE": "http://x", "HYPOTHESIS_LLM_KEY": "k",
                "HYPOTHESIS_LLM_MODEL": "m"}), \
                mock.patch("urllib.request.urlopen",
                           return_value=_HttpResp(content)):
            items = self.hg._llm_generate(snapshot, 3)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "llm")

    def test_incomplete_items_filtered(self):
        snapshot = {"llm_snapshot": {}}
        body = json.dumps([{"hypothesis": "H", "evidence": "E", "action": "A"},
                           {"hypothesis": "H2"}])  # 缺 evidence/action → 过滤
        content = json.dumps({"choices": [{"message": {"content": body}}]}).encode()
        with mock.patch.dict("os.environ", {
                "HYPOTHESIS_LLM_BASE": "http://x", "HYPOTHESIS_LLM_KEY": "k"}), \
                mock.patch("urllib.request.urlopen",
                           return_value=_HttpResp(content)):
            items = self.hg._llm_generate(snapshot, 3)
        self.assertEqual(len(items), 1)


class TestGenerate(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hg = _import()

    def _setup(self, tmp, with_bm=False):
        stats = _stats_df(n=10, zt=100, mb=5)
        stats["date"] = pd.date_range("2026-08-01", periods=10, freq="B")
        stats.to_parquet(tmp / "zt.parquet")
        _fund_df(n=10, youzi=-2.5, start="2026-08-01").to_parquet(tmp / "fund.parquet")
        _theme_df(n=10, zt_cnt=6).to_parquet(tmp / "theme.parquet")
        gen = tmp / "generated"
        if with_bm:
            gen.mkdir()
            (gen / "battle_map_2026-08-12.json").write_text(json.dumps({
                "regime": "亢奋", "position_range": "10-30%",
                "recommended": "试错", "confidence": 0.5, "emotion_stage": "发酵"}),
                encoding="utf-8")
        return [
            mock.patch.object(self.hg, "ZT_DAILY_STATS", tmp / "zt.parquet"),
            mock.patch.object(self.hg, "FUND_FORCES", tmp / "fund.parquet"),
            mock.patch.object(self.hg, "THEME_CYCLE", tmp / "theme.parquet"),
            mock.patch.object(self.hg, "GENERATED_DIR", gen),
        ]

    def test_template_channel_writes_file(self):
        tmp = self._make_tmp()
        for p in self._setup(tmp, with_bm=True):
            p.start()
        self.addCleanup(mock.patch.stopall)
        out = self.hg.generate("2026-08-12", k=3)
        self.assertTrue(out)
        for it in out:
            self.assertEqual(it["source"], "template")
        path = tmp / "generated" / "hypotheses_2026-08-12.json"
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["channel"], "template")
        self.assertEqual(payload["k"], len(out))

    def test_llm_channel(self):
        tmp = self._make_tmp()
        for p in self._setup(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(self.hg, "_llm_enabled", return_value=True), \
                mock.patch.object(self.hg, "_llm_generate", return_value=[
                    {"hypothesis": "H", "evidence": "E", "action": "A",
                     "confidence": 0.5, "source": "llm"}]):
            out = self.hg.generate("2026-08-12", k=3)
        self.assertEqual(out[0]["source"], "llm")
        payload = json.loads((tmp / "generated" / "hypotheses_2026-08-12.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["channel"], "llm")

    def test_llm_failure_falls_back_to_template(self):
        tmp = self._make_tmp()
        for p in self._setup(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(self.hg, "_llm_enabled", return_value=True), \
                mock.patch.object(self.hg, "_llm_generate", return_value=None):
            out = self.hg.generate("2026-08-12", k=3)
        self.assertTrue(out)
        self.assertEqual(out[0]["source"], "template")

    def test_no_data_returns_empty(self):
        tmp = self._make_tmp()
        for p in self._setup(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        (tmp / "zt.parquet").unlink()
        self.assertEqual(self.hg.generate("2026-08-12"), [])

    def test_k_clamped(self):
        tmp = self._make_tmp()
        for p in self._setup(tmp):
            p.start()
        self.addCleanup(mock.patch.stopall)
        out = self.hg.generate("2026-08-12", k=0)  # k<1 → 3
        self.assertLessEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()
