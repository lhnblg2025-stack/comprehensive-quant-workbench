"""data_store 核心链路单元测试。

覆盖:
  - get_many 跨标的日期对齐（V6 fix, 原 V4.1 字符串/Timestamp 失配修复点）
  - _fill_missing_dates 停牌日 volume/amount=0（回测语义修复点）
  - FRESHNESS 缓存命中/过期判定
  - prune 保留窗口（默认 180 天）

全部用临时 SQLite/构造 DataFrame 隔离，不依赖真实 data_warehouse 与网络。
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.data_store as ds
    return ds


def _mk_df(dates, close=None, volume=None):
    """构造带 REQUIRED_COLS 的测试 DataFrame。"""
    n = len(dates)
    close = close if close is not None else np.linspace(10.0, 15.0, n)
    volume = volume if volume is not None else np.full(n, 1000.0)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
        "amount": close * volume,
        "pct_chg": np.concatenate([[0.0], np.diff(close) / close[:-1] * 100.0]),
    })


class _TempDbMixin:
    """把 data_store 的 SQLite 访问重定向到临时库。"""

    def _setup_temp_db(self):
        self.ds = _import()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="ds_test_"))
        self.addCleanup(self._cleanup_tmp)
        self._conn = sqlite3.connect(str(self._tmpdir / "freshness.db"),
                                     check_same_thread=False)
        self._patchers = [
            mock.patch.object(self.ds, "_get_conn", return_value=self._conn),
            mock.patch.object(self.ds, "_init_db", return_value=None),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)
        self.ds._init_freshness_table()

    def _cleanup_tmp(self):
        import shutil
        self._conn.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    def _insert_freshness(self, symbol, age_hours, row_count=100):
        ts = (datetime.now() - timedelta(hours=age_hours)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO data_freshness "
            "(symbol, last_fetch, row_count, last_date, source, quality_score) "
            "VALUES (?, ?, ?, ?, 'test', 1.0)",
            (symbol, ts, row_count, "2026-08-10"),
        )
        self._conn.commit()


class TestGetManyAlignment(unittest.TestCase):
    """get_many 跨标的日期对齐。"""

    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def _make_store(self):
        store = self.ds.DataStore()
        return store

    def test_aligns_common_dates_across_symbols(self):
        # 600519 有独有首日 08-03，000858 有独有末日 08-11 → 对齐后只留交集
        dates_a = ["2026-08-03", "2026-08-04", "2026-08-05",
                   "2026-08-06", "2026-08-07", "2026-08-10"]
        dates_b = ["2026-08-04", "2026-08-05", "2026-08-06",
                   "2026-08-07", "2026-08-10", "2026-08-11"]
        df_a = _mk_df(dates_a)
        df_b = _mk_df(dates_b)
        common = ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-10"]

        store = self._make_store()
        with mock.patch.object(
            self.ds.DataStore, "get",
            side_effect=lambda sym, **kw: df_a if sym == "600519" else df_b,
        ):
            result = store.get_many(["600519", "000858"])
        self.assertEqual(set(result.keys()), {"600519", "000858"})
        for sym, df in result.items():
            got = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
            self.assertEqual(got, common, f"{sym} 日期未对齐到公共交集")

    def test_single_symbol_passthrough(self):
        df_a = _mk_df(pd.bdate_range("2026-08-01", periods=10))
        store = self._make_store()
        with mock.patch.object(self.ds.DataStore, "get", return_value=df_a):
            result = store.get_many(["600519"])
        self.assertEqual(len(result["600519"]), 10)

    def test_empty_symbols_returns_empty_dict(self):
        store = self._make_store()
        self.assertEqual(store.get_many([]), {})

    def test_partial_failure_keeps_successful_symbols(self):
        df_a = _mk_df(pd.bdate_range("2026-08-01", periods=10))
        store = self._make_store()
        with mock.patch.object(
            self.ds.DataStore, "get",
            side_effect=lambda sym, **kw: df_a if sym == "600519" else (_ for _ in ()).throw(ValueError("boom")),
        ):
            result = store.get_many(["600519", "000858"])
        self.assertIn("600519", result)

    def test_all_failure_raises_runtime_error(self):
        # 全部标的拉取失败 → get_many 抛 RuntimeError（不返回空 dict 静默吞错）
        store = self._make_store()
        with mock.patch.object(
            self.ds.DataStore, "get",
            side_effect=lambda sym, **kw: (_ for _ in ()).throw(ValueError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "all failed"):
                store.get_many(["600519", "000858"])


class TestFillGaps(unittest.TestCase):
    """_fill_missing_dates 停牌日 volume=0 回测语义。"""

    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def _calendar(self):
        return {d.strftime("%Y-%m-%d")
                for d in pd.bdate_range("2026-08-03", "2026-08-07")}

    def test_suspension_day_volume_zero_price_ffill(self):
        df = _mk_df(["2026-08-03", "2026-08-05"],
                    close=np.array([10.2, 10.7]),
                    volume=np.array([1000.0, 1200.0]))
        with mock.patch.object(self.ds, "get_trade_calendar", return_value=self._calendar()):
            out = self.ds._fill_missing_dates(df)
        dates = out["date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(dates, ["2026-08-03", "2026-08-04", "2026-08-05"])
        # 停牌合成行：价格 ffill、成交量/成交额必须为 0
        syn = out[out["date"].dt.strftime("%Y-%m-%d") == "2026-08-04"].iloc[0]
        self.assertEqual(syn["close"], 10.2)
        self.assertEqual(syn["volume"], 0.0)
        self.assertEqual(syn["amount"], 0.0)
        # 真实行成交量保留
        real = out[out["date"].dt.strftime("%Y-%m-%d") == "2026-08-05"].iloc[0]
        self.assertEqual(real["volume"], 1200.0)
        # 周末不注入
        self.assertNotIn("2026-08-08", dates)
        self.assertNotIn("2026-08-09", dates)

    def test_full_calendar_rows_untouched(self):
        dates = pd.bdate_range("2026-08-03", "2026-08-07")
        df = _mk_df(dates)
        with mock.patch.object(self.ds, "get_trade_calendar", return_value=self._calendar()):
            out = self.ds._fill_missing_dates(df)
        self.assertEqual(len(out), 5)
        self.assertTrue((out["volume"] == 1000.0).all())

    def test_long_gap_not_overfilled(self):
        # 缺口远超 ffill limit=10 个交易日：只允许填充前 10 个合成交易日，
        # 缺口中部保持 NaN 被剔除，末尾真实交易日原始行保留（volume 不置 0）
        df = _mk_df(["2026-08-03", "2026-09-02"])
        cal = {d.strftime("%Y-%m-%d")
               for d in pd.bdate_range("2026-08-03", "2026-09-02")}
        with mock.patch.object(self.ds, "get_trade_calendar", return_value=cal):
            out = self.ds._fill_missing_dates(df)
        # 首日 + 10 个 ffill 合成交易日 + 末尾原始行 = 12 行（不绑定具体日期）
        self.assertEqual(len(out), 12)
        # 缺口内合成行全部 volume=0；真实行保留原 volume
        self.assertTrue((out["volume"].iloc[1:-1] == 0.0).all())
        self.assertEqual(out["volume"].iloc[-1], df["volume"].iloc[-1])


class TestFreshness(_TempDbMixin, unittest.TestCase):
    """FRESHNESS 缓存命中/过期判定。"""

    def setUp(self):
        self._setup_temp_db()

    def test_missing_freshness_is_stale(self):
        self.assertTrue(self.ds._is_stale("600519"))

    def test_fresh_cache_hit(self):
        self._insert_freshness("600519", age_hours=1)
        self.assertFalse(self.ds._is_stale("600519"))

    def test_boundary_47h_fresh_49h_expired(self):
        # 两个独立 symbol 分别验证 47h 新鲜 / 49h 过期，避免依赖 REPLACE 覆盖语义
        self._insert_freshness("fresh_47h", age_hours=47)
        self.assertFalse(self.ds._is_stale("fresh_47h"))
        self._insert_freshness("expired_49h", age_hours=49)
        self.assertTrue(self.ds._is_stale("expired_49h"))

    def test_record_fetch_writes_freshness(self):
        df = _mk_df(pd.bdate_range("2026-08-03", periods=5))
        self.ds._record_fetch("600519", df)
        info = self.ds._freshness("600519")
        self.assertIsNotNone(info)
        self.assertEqual(info["symbol"], "600519")
        self.assertEqual(info["row_count"], 5)
        self.assertFalse(self.ds._is_stale("600519"))

class TestPrune(_TempDbMixin, unittest.TestCase):
    """prune 保留窗口（默认 180d）。"""

    def setUp(self):
        self._setup_temp_db()

    def test_default_prune_keeps_180d_window(self):
        self._insert_freshness("old_200d", age_hours=200 * 24)
        self._insert_freshness("mid_100d", age_hours=100 * 24)
        self._insert_freshness("recent_10d", age_hours=10 * 24)
        store = self.ds.DataStore()
        res = store.prune()
        self.assertEqual(res, {"pruned": 1})
        remaining = {r[0] for r in self._conn.execute(
            "SELECT symbol FROM data_freshness").fetchall()}
        self.assertEqual(remaining, {"mid_100d", "recent_10d"})

    def test_custom_days_window(self):
        self._insert_freshness("old_200d", age_hours=200 * 24)
        self._insert_freshness("mid_100d", age_hours=100 * 24)
        self._insert_freshness("recent_10d", age_hours=10 * 24)
        store = self.ds.DataStore()
        res = store.prune(days=30)
        self.assertEqual(res, {"pruned": 2})
        remaining = {r[0] for r in self._conn.execute(
            "SELECT symbol FROM data_freshness").fetchall()}
        self.assertEqual(remaining, {"recent_10d"})


if __name__ == "__main__":
    unittest.main()
