"""analysis_core/valuation_system 防前视修复单元测试。

覆盖:
  - _indicator_history as_of 截断（报告期/快照 ≤date，未来行剔除）
  - _price_history as_of 截断（分析日当天可用，未来行情剔除）
  - earnings_forecast 公告日期列 ≤date 过滤 / 无公告日期列标注 as_of
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
    return importlib.import_module("quant_system.analysis_core.valuation_system")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="valuation_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


class TestIndicatorHistoryAsOf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _fake_db(self):
        class _C:
            def execute(self, *a, **k):
                return self
            def fetchall(self):
                # 报告期行: 2026-06-30（Q2报告期）晚于分析日 2026-06-01 → 应被截断
                return [("20260331", 10.0), ("20260630", 12.0), ("20251231", 8.0)]
            def close(self):
                pass
        return _C()

    def test_truncates_future_rows(self):
        with mock.patch.object(self.m, "_db", return_value=self._fake_db()):
            df = self.m._indicator_history("600519", "net_profit_parent",
                                           as_of=pd.Timestamp("2026-06-01"))
        dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertIn("2026-03-31", dates)
        self.assertIn("2025-12-31", dates)
        self.assertNotIn("2026-06-30", dates)  # 未来报告期剔除
        self.assertEqual(sorted(dates), dates)

    def test_no_as_of_keeps_all(self):
        with mock.patch.object(self.m, "_db", return_value=self._fake_db()):
            df = self.m._indicator_history("600519", "net_profit_parent")
        self.assertEqual(len(df), 3)

    def test_latest_report_respects_cutoff(self):
        with mock.patch.object(self.m, "_db", return_value=self._fake_db()):
            df = self.m._indicator_history("600519", "net_profit_parent",
                                           as_of=pd.Timestamp("2026-06-01"))
        as_of, value = self.m._latest_report(df)
        self.assertEqual(str(as_of.date()), "2026-03-31")
        self.assertEqual(value, 10.0)


class TestPriceHistoryAsOf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def test_truncates_future_rows(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-05-29", "2026-06-01", "2026-06-02", "2026-08-11"]),
            "close": [1.0, 2.0, 3.0, 4.0],
        })
        with mock.patch("quant_system.data_store.get_store",
                        side_effect=lambda: mock.Mock(get=lambda *a, **k: df)):
            out = self.m._price_history("600519", as_of=pd.Timestamp("2026-06-01"))
        self.assertEqual(out["date"].max(), pd.Timestamp("2026-06-01"))
        self.assertEqual(len(out), 2)  # 06-02/08-11 未来行情剔除


class TestEarningsForecastAsOf(_TmpDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import()

    def _write_parquet(self, path, rows, cols):
        pd.DataFrame(rows, columns=cols).to_parquet(path)

    def test_ann_date_col_filters_future(self):
        tmp = self._make_tmp()
        p = tmp / "forecast.parquet"
        self._write_parquet(p, [
            {"code": "600519", "预告类型": "预增", "公告日期": "2026-05-15"},
            {"code": "600519", "预告类型": "预减", "公告日期": "2026-07-10"},
        ], ["code", "预告类型", "公告日期"])
        with mock.patch.object(self.m, "_find_forecast_parquets", return_value=[p]):
            out = self.m.earnings_forecast("600519", None, as_of=pd.Timestamp("2026-06-01"))
        self.assertEqual(out["label"], "业绩支撑")  # 5-15 预增；7-10 预减被截断
        self.assertIn("公告≤2026-06-01", out["detail"])

    def test_no_ann_col_marks_as_of(self):
        tmp = self._make_tmp()
        p = tmp / "forecast.parquet"
        self._write_parquet(p, [
            {"code": "600519", "预告类型": "预减"},
        ], ["code", "预告类型"])
        with mock.patch.object(self.m, "_find_forecast_parquets", return_value=[p]):
            out = self.m.earnings_forecast("600519", None, as_of=pd.Timestamp("2026-06-01"))
        self.assertEqual(out["label"], "业绩雷")
        self.assertIn("as_of=2026-06-01", out["detail"])
        self.assertIn("无公告日期列", out["detail"])

    def test_all_after_cutoff_skipped_to_proxy(self):
        tmp = self._make_tmp()
        p = tmp / "forecast.parquet"
        self._write_parquet(p, [
            {"code": "600519", "预告类型": "预减", "公告日期": "2026-07-10"},
        ], ["code", "预告类型", "公告日期"])
        with mock.patch.object(self.m, "_find_forecast_parquets", return_value=[p]):
            out = self.m.earnings_forecast("600519", -5.0, as_of=pd.Timestamp("2026-06-01"))
        self.assertEqual(out["source"], "proxy")
        self.assertEqual(out["label"], "业绩雷")


if __name__ == "__main__":
    unittest.main()
