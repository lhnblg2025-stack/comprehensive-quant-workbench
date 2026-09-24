"""analysis_core/behavior_system 防前视 + 口径对齐单元测试。

覆盖:
  - _load_nextday_kline 按分析日截断K线：分析日涨停股的 T+1（未来）不引入
  - _proxy_temp 与 emotion_system 口径一致（premium_t3 兜底）
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


def _import(name):
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module(f"quant_system.analysis_core.{name}")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="behavior_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


class TestLoadNextdayKline(_TmpDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _import("behavior_system")

    def test_no_future_kline_for_analysis_day(self):
        m = self.m
        kdir = self._make_tmp()
        pd.DataFrame({
            "date": pd.to_datetime(["2026-05-28", "2026-05-29", "2026-06-01"]),
            "open": [10.0, 10.5, 11.0],
            "close": [10.2, 10.6, 10.8],
        }).to_parquet(kdir / "600519.parquet")
        pool = pd.DataFrame({
            "date": pd.to_datetime(["2026-05-29", "2026-06-01"]),
            "code": ["600519", "600519"],
            "close": [10.6, 10.8],
            "next_pct": [2.0, 1.0],
        })
        with mock.patch.object(m, "KLINE_DIR", kdir):
            out = m._load_nextday_kline(pool, "2026-06-01")
        self.assertIsNotNone(out)
        # 分析日 06-01 涨停股的 T+1 是未来 → 剔除；05-29 的次日 06-01 可用
        dates = out["date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(dates, ["2026-05-29"])
        self.assertNotIn("2026-06-01", dates)

    def test_without_truncation_would_include_future(self):
        m = self.m
        kdir = self._make_tmp()
        pd.DataFrame({
            "date": pd.to_datetime(["2026-05-29", "2026-06-01", "2026-06-02"]),
            "open": [10.0, 10.5, 11.0],
            "close": [10.2, 10.6, 10.8],
        }).to_parquet(kdir / "600519.parquet")
        pool = pd.DataFrame({
            "date": pd.to_datetime(["2026-06-01"]),
            "code": ["600519"],
            "close": [10.8],
            "next_pct": [1.0],
        })
        with mock.patch.object(m, "KLINE_DIR", kdir):
            out = m._load_nextday_kline(pool, "2026-06-01")
        self.assertIsNone(out)  # 06-01 涨停 → 次日 06-02 越界，无可用证据


class TestProxyTempAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beh = _import("behavior_system")
        cls.emo = _import("emotion_system")

    def test_premium_t3_fallback_matches_emotion(self):
        row = pd.Series({
            "zt_cnt": 60, "max_board": 5, "zb_rate": 0.2, "dt_cnt": 5,
            "jr1": np.nan, "jr1_t3": 0.3, "premium": np.nan, "premium_t3": 2.0,
        })
        self.assertEqual(self.beh._proxy_temp(row), self.emo._proxy_temp(row))

    def test_premium_present_uses_premium(self):
        row = pd.Series({
            "zt_cnt": 60, "max_board": 5, "zb_rate": 0.2, "dt_cnt": 5,
            "jr1": 0.3, "premium": -3.0, "premium_t3": 2.0,
        })
        self.assertEqual(self.beh._proxy_temp(row), self.emo._proxy_temp(row))


if __name__ == "__main__":
    unittest.main()
