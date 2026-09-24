"""analysis_core/trend_system 三屏趋势系统单元测试。

覆盖:
  - 三屏检测: 周线 MA20/MA60+斜率 / 日线 MACD柱+MA20 / 近5日 MA5+斜率（合成趋势序列）
  - 综合评级: 三屏同向=强趋势 / 两屏=中等 / 混乱=震荡 / 样本不足
  - 趋势线: pivot 线性回归 + Murphy 突破确认（>3% 或 2日收盘）
  - detect_symbol: 样本不足防御、正常输出
  - TrendSystem.view: multi_agent 兼容结构 / detect 异常降级
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
    return importlib.import_module("quant_system.analysis_core.trend_system")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="trend_system_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


def _synth_df(n=420, trend=0.002, vol=0.004, start=100.0, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + trend + rng.normal(0, vol, n))
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"date": dates, "open": close * 0.99, "high": close * 1.02,
                         "low": close * 0.98, "close": close, "volume": 1e6})


class TestWeeklyScreen(unittest.TestCase):
    """第一屏 大周期（周线 MA20/MA60 排列 + MA20 斜率）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_uptrend_bullish(self):
        s = self.m._screen_weekly(_synth_df(trend=0.002))
        self.assertEqual(s["trend"], "多")
        self.assertGreater(s["n_bars"], 60)
        self.assertGreater(s["ma20"], s["ma60"])

    def test_downtrend_bearish(self):
        s = self.m._screen_weekly(_synth_df(trend=-0.002))
        self.assertEqual(s["trend"], "空")
        self.assertLess(s["ma20"], s["ma60"])

    def test_insufficient_samples(self):
        s = self.m._screen_weekly(_synth_df(n=100))
        self.assertEqual(s["trend"], "样本不足")
        self.assertTrue(s["note"])


class TestDailyScreen(unittest.TestCase):
    """第二屏 中周期（日线 MACD 柱方向 + 收盘 vs MA20）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_uptrend_bullish(self):
        s = self.m._screen_daily(_synth_df(trend=0.002)["close"])
        self.assertEqual(s["trend"], "多")
        self.assertTrue(s["hist_up"])
        self.assertEqual(s["close_vs_ma20"], "上")

    def test_insufficient_samples(self):
        s = self.m._screen_daily(pd.Series(np.linspace(100, 120, 10)))
        self.assertEqual(s["trend"], "样本不足")


class TestShortScreen(unittest.TestCase):
    """第三屏 小周期（近5日 收盘 vs MA5 + 斜率）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_recent_rise_bullish(self):
        close = pd.Series(np.linspace(100, 130, 30))
        s = self.m._screen_short(close)
        self.assertEqual(s["trend"], "多")
        self.assertEqual(s["close_vs_ma5"], "上")

    def test_recent_fall_bearish(self):
        close = pd.Series(np.linspace(130, 100, 30))
        s = self.m._screen_short(close)
        self.assertEqual(s["trend"], "空")

    def test_insufficient_samples(self):
        s = self.m._screen_short(pd.Series([1.0, 2.0, 3.0]))
        self.assertEqual(s["trend"], "样本不足")


class TestComposite(unittest.TestCase):
    """综合评级规则。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _screen(self, trend):
        return {"trend": trend}

    def test_three_same_strong(self):
        c = self.m._composite(self._screen("多"), self._screen("多"), self._screen("多"))
        self.assertEqual(c["rating"], "强趋势多")
        self.assertEqual(c["confidence"], 0.85)
        c2 = self.m._composite(self._screen("空"), self._screen("空"), self._screen("空"))
        self.assertEqual(c2["rating"], "强趋势空")

    def test_two_same_medium(self):
        c = self.m._composite(self._screen("多"), self._screen("多"), self._screen("震荡"))
        self.assertEqual(c["rating"], "中等多")
        c2 = self.m._composite(self._screen("空"), self._screen("空"), self._screen("多"))
        self.assertEqual(c2["rating"], "中等空")

    def test_mixed_range(self):
        c = self.m._composite(self._screen("多"), self._screen("空"), self._screen("震荡"))
        self.assertEqual(c["rating"], "震荡")

    def test_insufficient_any_screen(self):
        c = self.m._composite(self._screen("多"), self._screen("多"),
                              self._screen("样本不足"))
        self.assertEqual(c["rating"], "样本不足")
        self.assertEqual(c["confidence"], 0.0)


class TestTrendline(unittest.TestCase):
    """pivot 线性回归趋势线 + Murphy 突破确认。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _decline_df(self, n=120):
        rng = np.random.default_rng(7)
        base = 100 * np.cumprod(1 - 0.002 + rng.normal(0, 0.004, n))
        dates = pd.bdate_range("2025-01-01", periods=n)
        return pd.DataFrame({"date": dates, "open": base, "high": base * 1.01,
                             "low": base * 0.99, "close": base})

    def test_breakout_plus_3pct_confirmed(self):
        df = self._decline_df()
        tl = self.m._trendline(df["high"], df["low"], df["close"])
        self.assertFalse(tl["break_up"])
        lvl = tl["down"]["level"]
        self.assertTrue(np.isfinite(lvl))
        extra = pd.DataFrame({"date": [df["date"].iloc[-1] + pd.Timedelta(days=1)],
                              "open": [lvl * 1.01], "high": [lvl * 1.05],
                              "low": [lvl * 0.99], "close": [lvl * 1.04]})
        df2 = pd.concat([df, extra], ignore_index=True)
        tl2 = self.m._trendline(df2["high"], df2["low"], df2["close"])
        self.assertTrue(tl2["break_up"])
        self.assertTrue(tl2["break_up_confirmed"])
        self.assertGreater(tl2["break_up_gap_pct"], 3.0)

    def test_breakout_below_3pct_unconfirmed(self):
        df = self._decline_df()
        tl = self.m._trendline(df["high"], df["low"], df["close"])
        lvl = tl["down"]["level"]
        extra = pd.DataFrame({"date": [df["date"].iloc[-1] + pd.Timedelta(days=1)],
                              "open": [lvl * 0.99], "high": [lvl * 1.02],
                              "low": [lvl * 0.98], "close": [lvl * 1.01]})
        df2 = pd.concat([df, extra], ignore_index=True)
        tl2 = self.m._trendline(df2["high"], df2["low"], df2["close"])
        self.assertTrue(tl2["break_up"])
        self.assertFalse(tl2["break_up_confirmed"])

    def test_two_day_close_confirmation(self):
        df = self._decline_df()
        tl = self.m._trendline(df["high"], df["low"], df["close"])
        lvl = tl["down"]["level"]
        day1 = pd.DataFrame({"date": [df["date"].iloc[-1] + pd.Timedelta(days=1)],
                             "open": [lvl * 0.99], "high": [lvl * 1.005],
                             "low": [lvl * 0.98], "close": [lvl * 1.002]})
        day2 = pd.DataFrame({"date": [df["date"].iloc[-1] + pd.Timedelta(days=2)],
                             "open": [lvl * 1.0], "high": [lvl * 1.01],
                             "low": [lvl * 0.99], "close": [lvl * 1.004]})
        df2 = pd.concat([df, day1, day2], ignore_index=True)
        tl2 = self.m._trendline(df2["high"], df2["low"], df2["close"])
        self.assertTrue(tl2["break_up"])
        self.assertTrue(tl2["break_up_confirmed"])  # 连续2日收盘在线位另一侧


class TestDetectSymbol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_normal_output(self):
        r = self.m.detect_symbol(_synth_df(trend=0.002), "601899", "紫金矿业")
        self.assertTrue(r["ok"])
        self.assertEqual(r["code"], "601899")
        self.assertIn(r["rating"], ("强趋势多", "中等多", "震荡", "样本不足"))
        self.assertTrue(r["weekly"] and r["daily"] and r["short"])
        self.assertTrue(r["trendline"] is not None)
        self.assertGreaterEqual(r["confidence"], 0.0)

    def test_insufficient_defensive(self):
        tiny = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=10),
                             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0})
        r = self.m.detect_symbol(tiny, "000001", "测试")
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "样本不足")
        self.assertTrue(r["sample_note"])

    def test_bad_data_defensive(self):
        df = pd.DataFrame({"date": ["2026-01-01", "bad"], "close": [1.0, None]})
        r = self.m.detect_symbol(df, "000002", "测试2")
        self.assertFalse(r["ok"])  # 不抛异常，标注失败


class TestTrendSystemView(unittest.TestCase, _TmpDirMixin):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _canned(self):
        row_base = {"weekly": {"trend": "多"}, "daily": {"trend": "多"},
                    "short": {"trend": "多"}, "rating": "强趋势多",
                    "confidence": 0.85, "composite_note": "", "signals": [], "ok": True}
        bench = {"code": "000300", "name": "沪深300", "is_index": True, "close": 4702.0,
                 **row_base}
        stock = {"code": "601899", "name": "紫金矿业", "is_index": False, "close": 35.45,
                 **row_base}
        return {"date": "2026-08-10",
                "meta": {"universe": "watch", "benchmark": "market/x.parquet"},
                "rag": {"basis": "检索可用", "query": "三屏趋势 突破 确认"},
                "rows": [bench, stock]}

    def test_view_multi_agent_compatible(self):
        ts = self.m.TrendSystem()
        with mock.patch.object(ts, "detect", return_value=self._canned()):
            v = ts.view(watch=["601899"])
        self.assertEqual(v["agent"], "三屏趋势")
        self.assertEqual(v["signal"], "多")
        self.assertIn("confidence", v)
        self.assertIsInstance(v["evidence"], list)
        self.assertTrue(v["evidence"])
        self.assertEqual(v["status"], "ok")

    def test_view_code_specific(self):
        ts = self.m.TrendSystem()
        with mock.patch.object(ts, "detect", return_value=self._canned()):
            v = ts.view(code="601899")
        self.assertEqual(v["object_code"], "601899")
        self.assertEqual(v["signal"], "多")

    def test_view_degraded_on_detect_error(self):
        ts = self.m.TrendSystem()
        with mock.patch.object(ts, "detect", side_effect=RuntimeError("boom")):
            v = ts.view()
        self.assertEqual(v["status"], "degraded")
        self.assertEqual(v["signal"], "震荡")
        self.assertEqual(v["confidence"], 0.0)


class TestRagFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_search_failure_marks_unavailable(self):
        from quant_system.analysis_core import knowledge_rag
        ts = self.m.TrendSystem()
        ts._rag_cache = None
        with mock.patch.object(knowledge_rag, "search", side_effect=RuntimeError("no model")):
            out = ts._rag()
        self.assertEqual(out["basis"], "检索不可用")
        self.assertFalse(out["available"])
        self.assertEqual(out["hits"], [])


class TestReportWrite(unittest.TestCase, _TmpDirMixin):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_report_writes_md(self):
        out_dir = self._make_tmp()
        ts = self.m.TrendSystem(out_dir=out_dir)
        canned = {"date": "2026-08-10",
                  "meta": {"universe": "watch", "files_total": 2, "parsed": 2,
                           "skip_read_error": 0, "skip_schema": 0, "skip_empty": 0,
                           "skip_no_data": 0, "insufficient": 0, "elapsed_sec": 0.1,
                           "benchmark": "market/index_daily_沪深300.parquet",
                           "benchmark_available": True},
                  "rag": {"query": "三屏趋势 突破 确认", "basis": "检索可用",
                          "hits": [{"file": "skills/a.md", "cat": "c", "score": 0.5,
                                    "summary": "摘要"}]},
                  "rows": [
                      {"code": "000300", "name": "沪深300", "is_index": True, "ok": True,
                       "close": 4702.0, "pct_chg": 0.5, "rating": "强趋势多",
                       "confidence": 0.85, "composite_note": "",
                       "weekly": {"trend": "多", "ma20": 4752.0, "ma60": 4566.0,
                                  "slope_pct": 0.05, "note": ""},
                       "daily": {"trend": "多", "hist_up": True, "hist": 15.3,
                                 "ma20": 4655.0, "close_vs_ma20": "上", "note": ""},
                       "short": {"trend": "多", "ma5": 4661.0, "slope_pct": 0.5,
                                 "close_vs_ma5": "上", "note": ""},
                       "trendline": {}, "signals": [], "votes": {"多": 3}, "sample_note": ""},
                      {"code": "601899", "name": "紫金矿业", "is_index": False, "ok": True,
                       "close": 35.45, "pct_chg": 0.85, "rating": "强趋势多",
                       "confidence": 0.85, "composite_note": "",
                       "weekly": {"trend": "多", "ma20": 33.0, "ma60": 31.0,
                                  "slope_pct": 0.3, "note": ""},
                       "daily": {"trend": "多", "hist_up": True, "hist": 0.5,
                                 "ma20": 34.5, "close_vs_ma20": "上", "note": ""},
                       "short": {"trend": "多", "ma5": 35.0, "slope_pct": 0.4,
                                 "close_vs_ma5": "上", "note": ""},
                       "trendline": {}, "signals": [], "votes": {"多": 3}, "sample_note": ""},
                  ]}
        with mock.patch.object(ts, "detect", return_value=canned):
            path = ts.report(watch=["601899"])
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("# 三屏趋势系统", text)
        self.assertIn("2026-08-10", text)
        self.assertIn("三屏趋势 突破 确认", text)
        self.assertIn("紫金矿业", text)


if __name__ == "__main__":
    unittest.main()
