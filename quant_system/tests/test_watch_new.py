"""analysis_core 新功能模块单元测试（rs_strength / breakout_watch / volume_profile / fund_flow_divergence）。

覆盖:
  - compute_rs_metrics：日期对齐 / MA20 / 斜率 / 评级 / 60日新高边界 / 样本不足
  - breakout_watch.scan：60/120/250 新高边界、ST / 北交所 / 次新剔除
  - compute_volume_profile：POC / 上下沿 / 位置标签 / 样本不足
  - fund_flow_divergence：涨>3%主力流出→背离、跌>3%主力流入→承接、mock 网络解析

K线/名称/快照全部走临时目录构造 parquet；资金流网络请求全部 mock。
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
        self._tmp = Path(tempfile.mkdtemp(prefix="watch_new_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


def _rs_series(rs_values):
    idx = pd.bdate_range("2026-06-01", periods=len(rs_values))
    close = pd.Series(100.0 * np.asarray(rs_values, dtype=float), index=idx)
    bench = pd.Series(100.0, index=idx)
    return close, bench


class TestRSMetrics(unittest.TestCase):
    """compute_rs_metrics 纯计算。"""

    @classmethod
    def setUpClass(cls):
        cls.rs = _import("rs_strength")

    def _calc(self, rs_values):
        close, bench = _rs_series(rs_values)
        return self.rs.compute_rs_metrics(close, bench)

    def test_alignment_and_ma20(self):
        # 基准只有后 25 个交易日 → inner join 对齐到 25 点，rs=1.0
        idx = pd.bdate_range("2026-06-01", periods=40)
        close = pd.Series(100.0, index=idx)
        bench = pd.Series(100.0, index=idx[15:])
        res = self.rs.compute_rs_metrics(close, bench)
        self.assertEqual(res["n_points"], 25)
        self.assertEqual(res["rs"], 1.0)
        self.assertEqual(res["rs_ma20"], 1.0)

    def test_strong_rating(self):
        # 持续跑赢基准：斜率>0 且 rs 高于 MA20 → 强
        res = self._calc([1.0 + 0.02 * i for i in range(40)])
        self.assertEqual(res["rating"], "强")
        self.assertGreater(res["rs_slope"], 0)
        self.assertGreater(res["rs"], res["rs_ma20"])

    def test_mid_rating(self):
        # 末端急跌：斜率仍>0 但 rs 跌破 MA20 → 中
        res = self._calc([1.0 + 0.05 * i for i in range(24)] + [1.30])
        self.assertEqual(res["rating"], "中")
        self.assertGreater(res["rs_slope"], 0)
        self.assertLess(res["rs"], res["rs_ma20"])

    def test_weak_rating(self):
        res = self._calc([2.0 - 0.02 * i for i in range(40)])
        self.assertEqual(res["rating"], "弱")
        self.assertLess(res["rs_slope"], 0)
        self.assertLess(res["rs"], res["rs_ma20"])

    def test_60d_high_true(self):
        res = self._calc([1.0 + 0.001 * i for i in range(65)])
        self.assertTrue(res["rs_60d_high"])
        self.assertEqual(res["n_points"], 65)

    def test_60d_high_equal_boundary_false(self):
        # 新高要求严格大于此前 60 日窗口最高价：末值 = 前值 = 窗口最高 → 相等不算新高
        vals = [1.0 + 0.001 * i for i in range(65)]
        vals[-1] = vals[-2]
        res = self._calc(vals)
        self.assertFalse(res["rs_60d_high"])

    def test_insufficient_samples(self):
        res = self._calc([1.0] * 10)
        self.assertEqual(res["rating"], "样本不足")
        self.assertEqual(res["n_points"], 0)
        self.assertTrue(np.isnan(res["rs"]))


class TestBreakoutScan(_TmpDirMixin, unittest.TestCase):
    """breakout_watch.scan：新高边界与剔除规则。"""

    @classmethod
    def setUpClass(cls):
        cls.bw = _import("breakout_watch")

    def setUp(self):
        self._make_tmp()
        self._kdir = self._tmp / "kline"
        self._mdir = self._tmp / "market"
        self._sdir = self._tmp / "snapshot"
        for d in (self._kdir, self._mdir, self._sdir):
            d.mkdir(parents=True, exist_ok=True)
        self._patchers = [
            mock.patch.object(self.bw, "KLINE_DIR", self._kdir),
            mock.patch.object(self.bw, "MARKET_DIR", self._mdir),
            mock.patch.object(self.bw, "SNAPSHOT_DIR", self._sdir),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)
        self._write_names()

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    def _write_names(self):
        names = pd.DataFrame({
            "code": ["600001", "600002", "600003", "600004", "830001"],
            "name": ["正常一", "正常二", "*ST测试", "次新股", "北交股"],
            "is_st": [False, False, True, False, False],
        })
        names.to_parquet(self._mdir / "stock_names.parquet")

    def _write_kline(self, code, n_days, closes, volume=None):
        dates = pd.bdate_range("2025-07-07", "2026-08-10")[-n_days:]
        closes = np.asarray(closes, dtype=float)
        volume = volume if volume is not None else np.full(n_days, 100.0)
        df = pd.DataFrame({
            "date": dates,
            "open": closes * 0.99,
            "high": closes * 1.02,
            "low": closes * 0.98,
            "close": closes,
            "volume": volume,
            "amount": closes * volume,
        })
        df.to_parquet(self._kdir / f"{code}.parquet")

    def test_new_high_boundaries_60_120_250(self):
        n = 280
        # 600001: 末价 10 严格高于此前所有收盘 → 三档新高
        base = np.linspace(9.0, 9.99, n)
        c1 = base.copy()
        c1[-1] = 10.0
        self._write_kline("600001", n, c1)
        # 600002: 末价与前高相等（10 == 10）→ 三档均非新高
        c2 = base.copy()
        c2[-2] = 10.0
        c2[-1] = 10.0
        self._write_kline("600002", n, c2)
        result, meta = self.bw.scan([60, 120, 250],
                                    pd.Timestamp("2026-08-10"))
        for n_days in (60, 120, 250):
            codes = [r["code"] for r in result["new_high_lists"][n_days]]
            self.assertEqual(codes, ["600001"],
                             f"{n_days}日新高列表应只含 600001")
        # 600002 严格相等边界 → 不入任何新高列表
        all_codes = {c for rows in result["new_high_lists"].values()
                     for r in rows for c in [r["code"]]}
        self.assertNotIn("600002", all_codes)
        # 行结构带 new_high 字典
        row = result["new_high_lists"][60][0]
        self.assertEqual(row["new_high"], {60: True, 120: True, 250: True})

    def test_st_bj_new_stock_exclusion(self):
        n = 280
        base = np.linspace(9.0, 9.99, n)
        c = base.copy()
        c[-1] = 10.0
        self._write_kline("600001", n, c)      # 正常
        self._write_kline("600003", n, c)      # ST → skip_st
        self._write_kline("830001", n, c)      # 北交所(8 开头) → skip_bj
        self._write_kline("600004", 100, c[:100])  # 次新(<250 行) → skip_new
        result, meta = self.bw.scan([60, 120, 250],
                                    pd.Timestamp("2026-08-10"))
        self.assertEqual(meta["skip_st"], 1)
        self.assertEqual(meta["skip_bj"], 1)
        self.assertEqual(meta["skip_new"], 1)
        self.assertEqual(meta["universe"], 1)
        self.assertEqual(meta["new_high_cnt"][60], 1)


class TestVolumeProfile(_TmpDirMixin, unittest.TestCase):
    """compute_volume_profile：POC/上下沿/位置。"""

    @classmethod
    def setUpClass(cls):
        cls.vp = _import("volume_profile")

    def _profile_df(self, last_close):
        # 40 根重仓柱集中在 15.45~15.55，两侧少量低价/高价柱
        heavy = pd.DataFrame({
            "close": [15.5],
            "low": [15.45],
            "high": [15.55],
            "volume": [1000.0],
        })
        heavy = pd.concat([heavy] * 40, ignore_index=True)
        low_side = pd.DataFrame({
            "close": [11.0], "low": [10.0], "high": [12.0], "volume": [10.0],
        })
        low_side = pd.concat([low_side] * 10, ignore_index=True)
        high_side = pd.DataFrame({
            "close": [19.0], "low": [18.0], "high": [20.0], "volume": [10.0],
        })
        high_side = pd.concat([high_side] * 10, ignore_index=True)
        df = pd.concat([low_side, high_side, heavy], ignore_index=True)
        df.loc[df.index[-1], "close"] = last_close
        return df

    def test_poc_and_edges(self):
        res = self.vp.compute_volume_profile(self._profile_df(15.5))
        self.assertTrue(res["ok"])
        self.assertEqual(res["days"], 60)
        self.assertGreater(res["poc"], 15.4)
        self.assertLess(res["poc"], 15.6)
        self.assertLessEqual(res["lower"], res["poc"])
        self.assertLessEqual(res["poc"], res["upper"])
        self.assertEqual(res["pos_label"], "区间内")
        self.assertTrue(0 <= res["pos"] <= 1)
        # peak_share 单位 = 百分比（如 49.75 = 49.75%），重仓柱占总量过半 → >40
        self.assertGreater(res["peak_share"], 40)

    def test_above_upper(self):
        res = self.vp.compute_volume_profile(self._profile_df(16.5))
        self.assertEqual(res["pos_label"], "上方")
        self.assertGreater(res["pos"], 1)

    def test_below_lower(self):
        res = self.vp.compute_volume_profile(self._profile_df(14.5))
        self.assertEqual(res["pos_label"], "下方")
        self.assertLess(res["pos"], 0)

    def test_insufficient_samples(self):
        small = pd.DataFrame({
            "close": [10.0] * 10, "low": [9.9] * 10,
            "high": [10.1] * 10, "volume": [100.0] * 10,
        })
        res = self.vp.compute_volume_profile(small)
        self.assertFalse(res["ok"])
        self.assertIsNone(res["poc"])
        self.assertIn("样本不足", res["error"])

    def test_all_zero_volume(self):
        # 全零成交量：不崩溃、无除零；峰格占比未定义 → NaN（不能视为 0% 或 100%）
        df = pd.DataFrame({
            "close": np.linspace(10.0, 12.0, 25),
            "low": np.linspace(9.9, 11.9, 25),
            "high": np.linspace(10.1, 12.1, 25),
            "volume": np.zeros(25),
        })
        res = self.vp.compute_volume_profile(df)
        self.assertTrue(res["ok"])
        self.assertTrue(np.isnan(res["peak_share"]))


class TestFundFlowDivergence(unittest.TestCase):
    """fund_flow_divergence：背离/承接判定与 mock 网络。"""

    @classmethod
    def setUpClass(cls):
        cls.ffd = _import("fund_flow_divergence")

    def test_judge_divergence_rules(self):
        j = self.ffd.judge_divergence
        self.assertIn("量价背离", j(5.0, -1e7))
        self.assertIn("主力承接", j(-4.0, 2e7))
        self.assertEqual(j(1.0, 1e7), "正常")
        self.assertEqual(j(5.0, None), "数据不足")
        self.assertEqual(j(None, 1e7), "数据不足")

    def test_boundary_not_divergence(self):
        j = self.ffd.judge_divergence
        self.assertEqual(j(3.0, -1e7), "正常")   # 恰好 +3% 不触发
        self.assertEqual(j(-3.0, 2e7), "正常")   # 恰好 -3% 不触发

    def _fake_response(self, klines):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"data": {"klines": klines}}
        return resp

    def test_fetch_fund_flow_parses_and_picks_target_date(self):
        klines = [
            "2026-08-07,100,1,2,3,4,5",
            "2026-08-10,-50,1,2,3,4,5",
        ]
        with mock.patch.object(self.ffd.requests, "get",
                               return_value=self._fake_response(klines)):
            res = self.ffd.fetch_fund_flow("1.600519",
                                           pd.Timestamp("2026-08-10"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["date"], "2026-08-10")
        self.assertEqual(res["main_net"], -50.0)
        self.assertEqual(res["super_net"], 4.0)

    def test_fetch_fund_flow_defaults_to_last_row(self):
        klines = ["2026-08-07,100,1,2,3,4,5"]
        with mock.patch.object(self.ffd.requests, "get",
                               return_value=self._fake_response(klines)):
            res = self.ffd.fetch_fund_flow("1.600519")
        self.assertEqual(res["date"], "2026-08-07")
        self.assertEqual(res["main_net"], 100.0)

    def test_fetch_fund_flow_by_code_network_exception(self):
        # 网络异常被 fetch_fund_flow_by_code 兜底为防御性 dict，不向调用方抛异常
        with mock.patch.object(self.ffd.requests, "get",
                               side_effect=RuntimeError("网络超时")):
            res = self.ffd.fetch_fund_flow_by_code("600519")
        self.assertFalse(res["ok"])
        self.assertTrue(res["unavailable"])
        self.assertIn("RuntimeError", res["error"])
        self.assertEqual(res["code"], "600519")

    def test_check_divergence_up_with_outflow(self):
        with mock.patch.object(self.ffd, "local_pct_chg",
                               return_value=(5.5, "")), \
             mock.patch.object(self.ffd, "fetch_fund_flow_by_code",
                               return_value={"ok": True, "main_net": -1e7,
                                             "super_net": -5e6,
                                             "date": "2026-08-10",
                                             "code": "600519"}):
            res = self.ffd.check_divergence("600519")
        self.assertIn("量价背离", res["signal"])
        self.assertEqual(res["main_net"], -1e7)

    def test_check_divergence_down_with_inflow(self):
        with mock.patch.object(self.ffd, "local_pct_chg",
                               return_value=(-4.2, "")), \
             mock.patch.object(self.ffd, "fetch_fund_flow_by_code",
                               return_value={"ok": True, "main_net": 2e7,
                                             "super_net": 1e7,
                                             "date": "2026-08-10",
                                             "code": "600519"}):
            res = self.ffd.check_divergence("600519")
        self.assertIn("主力承接", res["signal"])

    def test_check_divergence_unavailable_and_skip(self):
        with mock.patch.object(self.ffd, "local_pct_chg",
                               return_value=(1.0, "")), \
             mock.patch.object(self.ffd, "fetch_fund_flow_by_code",
                               return_value={"ok": False,
                                             "error": "接口无数据"}):
            res = self.ffd.check_divergence("600519")
        self.assertEqual(res["signal"], "unavailable")

        with mock.patch.object(self.ffd, "local_pct_chg",
                               return_value=(1.0, "")), \
             mock.patch.object(self.ffd, "fetch_fund_flow_by_code",
                               return_value={"ok": False, "skipped": True,
                                             "note": "北交所跳过"}):
            res = self.ffd.check_divergence("600519")
        self.assertEqual(res["signal"], "北交所跳过")


if __name__ == "__main__":
    unittest.main()
