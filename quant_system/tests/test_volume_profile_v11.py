"""analysis_core/volume_profile 交易密集区单元测试。

覆盖:
  - compute_volume_profile：空数据/单K线/样本不足/无成交(全0量)/无价格区间(贴POC)/
    正常 POC+上下沿+位置标注（上方/区间内/下方）
  - fmt_pos 输出文本 / render_row / calc_volume_profile 别名
  - load_kline_60d：文件缺失/读取失败/缺关键列/无数据/目标日无交易/60 日截断
  - main CLI：正常输出 / 非法代码 / 空 watch
"""

from __future__ import annotations

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
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.volume_profile as vp
    return vp


def _make_df(prices, volumes=None, low_pad=0.5, high_pad=0.5):
    n = len(prices)
    vols = [10.0] * n if volumes is None else volumes
    rows = []
    for i, c in enumerate(prices):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "open": c, "high": c + high_pad, "low": c - low_pad,
                     "close": c, "volume": vols[i]})
    return pd.DataFrame(rows)


class _TmpMixin:
    pass


class TestComputeProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vp = _import()

    def test_none_and_empty_insufficient(self):
        r = self.vp.compute_volume_profile(None)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "样本不足")
        self.assertIsNone(r["poc"])
        self.assertEqual(r["days"], 0)
        r2 = self.vp.compute_volume_profile(pd.DataFrame())
        self.assertFalse(r2["ok"])
        self.assertEqual(r2["n"], 0)

    def test_single_kline_insufficient(self):
        df = _make_df([100.0])
        r = self.vp.compute_volume_profile(df)
        self.assertFalse(r["ok"])
        self.assertEqual(r["note"], "样本不足")

    def test_under_min_samples_insufficient(self):
        df = _make_df([100.0 + i for i in range(19)])
        r = self.vp.compute_volume_profile(df)
        self.assertFalse(r["ok"])
        self.assertEqual(r["n"], 19)

    def test_all_nan_volume_insufficient(self):
        df = _make_df([100.0 + i for i in range(20)], volumes=[np.nan] * 20)
        r = self.vp.compute_volume_profile(df)
        self.assertFalse(r["ok"])

    def test_zero_volume_with_range(self):
        """全部无成交但有价格区间 → 不抛，密集带退化为全区间，峰占量 NaN。"""
        df = _make_df([100.0 + i * 0.2 for i in range(60)], volumes=[0.0] * 60)
        r = self.vp.compute_volume_profile(df)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["lower"], float(df["low"].min()), places=6)
        # 几何网格向上取整 → 上沿 ≥ 最高价（允许一个 0.5% 格宽余量）
        self.assertGreaterEqual(r["upper"], float(df["high"].max()))
        self.assertLess(r["upper"], float(df["high"].max()) * 1.01)
        self.assertTrue(np.isnan(r["peak_share"]))

    def test_flat_price_touches_poc(self):
        # 高低价完全相等 → 无价格区间 → 贴 POC
        df = _make_df([150.0] * 60, low_pad=0.0, high_pad=0.0)
        r = self.vp.compute_volume_profile(df)
        self.assertTrue(r["ok"])
        self.assertEqual(r["pos_label"], "贴POC")
        self.assertEqual(r["pos_pct"], 100.0)
        self.assertEqual(r["upper"], 150.0)
        self.assertEqual(r["lower"], 150.0)
        self.assertEqual(r["poc"], 150.0)
        self.assertEqual(r["peak_share"], 100.0)

    def test_normal_profile_with_poc_above_position(self):
        prices = np.linspace(100.0, 200.0, 60)
        vols = [10.0] * 60
        vols[30] = 1000.0  # 量能集中在中间价（~150）→ POC 应落在此处
        df = _make_df(list(prices), volumes=vols)
        r = self.vp.compute_volume_profile(df)
        self.assertTrue(r["ok"])
        self.assertEqual(r["days"], 60)
        self.assertGreaterEqual(r["poc"], 100.0)
        self.assertLessEqual(r["poc"], 200.0)
        self.assertLess(r["lower"], r["upper"])
        # 量能峰在 ~150，POC 应显著偏离当前价 200
        self.assertLess(r["poc"], 180.0)
        self.assertEqual(r["pos_label"], "上方")
        self.assertGreater(r["pos"], 1.0)
        self.assertGreater(r["peak_share"], 0.0)
        self.assertLessEqual(r["peak_share"], 100.0)

    def test_close_inside_band(self):
        df = _make_df([150.0] * 60, low_pad=1.0, high_pad=1.0)
        r = self.vp.compute_volume_profile(df)
        self.assertEqual(r["pos_label"], "区间内")
        self.assertGreaterEqual(r["pos_pct"], 0.0)
        self.assertLessEqual(r["pos_pct"], 100.0)
        self.assertAlmostEqual(r["pos"], r["pos_pct"] / 100.0, places=6)

    def test_close_below_band(self):
        prices = [150.0] * 59 + [120.0]  # 最后一根跌破密集区
        vols = [100.0] * 59 + [10.0]
        df = _make_df(prices, volumes=vols, low_pad=1.0, high_pad=1.0)
        r = self.vp.compute_volume_profile(df)
        self.assertEqual(r["pos_label"], "下方")
        self.assertLess(r["pos"], 0.0)
        self.assertLess(r["close"], r["lower"])

    def test_calc_volume_profile_alias(self):
        self.assertIs(self.vp.calc_volume_profile, self.vp.compute_volume_profile)


class TestFmtPos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vp = _import()

    def test_above_upper(self):
        vp = {"pos_label": "上方", "close": 110.0, "upper": 100.0}
        self.assertEqual(self.vp.fmt_pos(vp), "高于上沿 10.0%")

    def test_below_lower(self):
        vp = {"pos_label": "下方", "close": 90.0, "lower": 100.0}
        self.assertEqual(self.vp.fmt_pos(vp), "低于下沿 11.1%")

    def test_inside_band(self):
        vp = {"pos_label": "区间内", "pos_pct": 42.5}
        self.assertEqual(self.vp.fmt_pos(vp), "区间内 42%")

    def test_touch_poc(self):
        vp = {"pos_label": "贴POC", "pos_pct": 100.0}
        self.assertEqual(self.vp.fmt_pos(vp), "贴POC 100%")

    def test_missing_metrics_falls_back_to_label(self):
        self.assertEqual(self.vp.fmt_pos({"pos_label": "上方"}), "上方")
        self.assertEqual(self.vp.fmt_pos({}), "—")

    def test_render_row(self):
        ok = {"ok": True, "days": 60, "poc": 150.0, "upper": 155.0, "lower": 145.0,
              "close": 160.0, "pos_label": "上方", "pos_pct": 250.0,
              "peak_share": 30.5}
        line = self.vp.render_row("600519", "贵州茅台", ok)
        self.assertIn("600519", line)
        self.assertIn("60日", line)
        self.assertIn("POC", line)
        bad = self.vp.render_row("000001", "平安银行", {"ok": False, "error": "无数据"})
        self.assertIn("无数据", bad)


class TestLoadKline60d(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vp = _import()

    def _patch_kline_dir(self):
        mock.patch.object(self.vp, "KLINE_DIR", self._tmp).start()
        self.addCleanup(mock.patch.stopall)

    def _write(self, code, n=70, start="2026-05-01"):
        df = pd.DataFrame({
            "date": pd.date_range(start, periods=n, freq="D"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000.0,
        })
        df.to_parquet(self._tmp / f"{code}.parquet", index=False)
        return df

    def test_missing_file(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._patch_kline_dir()
        df, err = self.vp.load_kline_60d("600999", pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertEqual(err, "kline文件缺失")

    def test_normal_window_60_days(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write("000001")
        self._patch_kline_dir()
        df, err = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-12"))
        self.assertEqual(err, "")
        self.assertEqual(len(df), 60)

    def test_target_date_truncation(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write("000001", start="2026-07-01")  # 70 天跨到 9 月
        self._patch_kline_dir()
        df, _ = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-05"))
        self.assertLessEqual(df["date"].max(), pd.Timestamp("2026-08-05"))
        self.assertEqual(len(df), 36)  # 7-01..8-05

    def _write_placeholder(self):
        pd.DataFrame({"date": ["2026-08-01"], "close": [1.0]}).to_parquet(
            self._tmp / "000001.parquet", index=False)

    def test_read_failure(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write_placeholder()
        self._patch_kline_dir()
        with mock.patch.object(self.vp, "read_kline_window",
                               side_effect=RuntimeError("boom")):
            df, err = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertIn("读取失败", err)

    def test_missing_key_columns(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write_placeholder()
        self._patch_kline_dir()
        with mock.patch.object(self.vp, "read_kline_window",
                               return_value=pd.DataFrame({"date": ["2026-08-01"]})):
            df, err = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertEqual(err, "缺关键列")

    def test_no_trade_on_target(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write_placeholder()
        self._patch_kline_dir()
        future = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
            "close": [1.0, 1.0], "volume": [1.0, 1.0]})
        with mock.patch.object(self.vp, "read_kline_window", return_value=future):
            df, err = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertEqual(err, "目标日无交易")

    def test_empty_result(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._write_placeholder()
        self._patch_kline_dir()
        with mock.patch.object(self.vp, "read_kline_window",
                               return_value=pd.DataFrame()):
            df, err = self.vp.load_kline_60d("000001", pd.Timestamp("2026-08-12"))
        self.assertIsNone(df)
        self.assertEqual(err, "无数据")


class TestMain(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vp = _import()

    def _run(self, argv):
        import contextlib
        import io
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["volume_profile"] + argv), \
                mock.patch.object(self.vp, "load_names", return_value=({}, {})), \
                contextlib.redirect_stdout(buf):
            try:
                self.vp.main()
            except SystemExit as e:
                return buf.getvalue(), e.code
        return buf.getvalue(), None

    def test_valid_watch_output(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        pd.DataFrame({
            "date": pd.date_range("2026-05-01", periods=60, freq="D"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000.0}).to_parquet(self._tmp / "000001.parquet", index=False)
        mock.patch.object(self.vp, "KLINE_DIR", self._tmp).start()
        self.addCleanup(mock.patch.stopall)
        out, code = self._run(["--watch", "000001,600999"])
        self.assertIsNone(code)
        self.assertIn("[交易密集区]", out)
        self.assertIn("000001", out)
        self.assertIn("kline文件缺失", out)  # 600999 无文件

    def test_invalid_code_exits(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        out, code = self._run(["--watch", "abc"])
        self.assertEqual(code, 1)
        self.assertIn("非法代码", out)

    def test_empty_watch_exits(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vp_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        out, code = self._run(["--watch", " , "])
        self.assertEqual(code, 1)
        self.assertIn("不能为空", out)


if __name__ == "__main__":
    unittest.main()

