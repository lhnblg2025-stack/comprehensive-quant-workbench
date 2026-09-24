"""analysis_core/fusion 信号融合层单元测试（域E 补测）。

覆盖:
  - fuse_today 全链路：真实小 parquet（stats/forces/themes/social）+ mock 云源/本地代理/
    行业广度/宏观 → 温度修正链、交叉信号、双源校验、降级链闭环
  - 行业广度校验：申万 vs 中证 口径一致/口径分歧（industry.check 字段）
  - 降级链：云指标挂/本地代理挂/数据陈旧 → unavailable 闭环，不抛异常
  - read_fusion_latest 边界：无数据/未来日期/列选择/损坏文件/坏日期
  - _industry_breadth / _csi_breadth / _macro_latest_yoy 真实 parquet 边界
  - _data_lag / _as_of_lag / store 落盘去重
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CST = timezone(timedelta(hours=8))


def _import():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.fusion as fusion
    return fusion


def _patch_now(fusion):
    """固定 '当前时间' 为 2026-08-12，保证陈旧度判定与运行日期无关（幂等）。"""

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 8, 12, tzinfo=CST)

    p = mock.patch.object(fusion, "datetime", _FakeDateTime)
    p.start()
    return p


class _TmpMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="fusion_v11_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


EMO = {"date": "2026-08-10", "data_date": "2026-08-10", "lag_days": 2,
       "stage": "ferment", "stage_cn": "发酵期", "zt_cnt": 60, "max_board": 4,
       "zb_rate": 0.2, "premium": 1.0}

SW_OK = {"status": "available", "as_of": "2026-08-10", "breadth": 0.8,
         "top_up": [{"name": "a", "pct": 1.0}], "top_down": [{"name": "b", "pct": -1.0}],
         "reason": "sw"}
CSI_OK = {"status": "available", "as_of": "2026-08-10", "breadth": 0.9,
          "top_up": [{"name": "a", "pct": 1.0}], "top_down": [{"name": "b", "pct": -1.0}],
          "reason": "csi"}
CLOUD_OK = {"status": "available", "date": "2026-08-10", "signal": "高扰动",
            "total_announcements": 5,
            "hot_rank": {"status": "available", "as_of": "2026-08-10", "avg_pct": 1.0},
            "social": {"status": "available", "as_of": "2026-08-10"}}
PROXIES_OK = {"status": "available", "date": "2026-08-10", "signal": "升温",
              "proxies": {k: {"signal": "升温"} for k in (
                  "supply_chain_heat", "consumption_heat",
                  "new_stock_activity", "retail_focus")}}
MACRO_OK = {"status": "available", "label": "温和", "cpi": 2.0, "m2": 8.0,
            "as_of": "2026-07"}


class TestFuseTodayFullChain(_TmpMixin, unittest.TestCase):
    """全链路：所有输入可用 → 温度 95、共振交叉信号、双源口径一致。"""

    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _write_market(self, tmp, stats_date="2026-08-10"):
        pd.DataFrame({"date": [pd.Timestamp(stats_date)],
                      "zb_rate": [0.2], "premium": [1.0]}).to_parquet(
            tmp / "zt_daily_stats.parquet", index=False)
        pd.DataFrame({"date": [pd.Timestamp(stats_date)],
                      "force_index": [95.0]}).to_parquet(
            tmp / "fund_forces.parquet", index=False)
        pd.DataFrame({
            "date": [pd.Timestamp(stats_date)] * 4,
            "role": ["主线", "主线", "支线", "支线"],
            "stage": ["burst", "burst", "rise", "rise"],
            "zt_cnt": [20, 18, 5, 3]}).to_parquet(
            tmp / "theme_cycle.parquet", index=False)
        pd.DataFrame({
            "date": [pd.Timestamp(stats_date)] * 2,
            "platform": ["weibo", "xueqiu"],
            "ok": [True, True]}).to_parquet(
            tmp / "social_sentiment.parquet", index=False)

    def _fuse(self, tmp, *, cloud=CLOUD_OK, proxies=PROXIES_OK, sw=SW_OK,
              csi=CSI_OK, macro=MACRO_OK, emo=None, stats_date="2026-08-10"):
        self._write_market(tmp, stats_date)
        emo = emo or EMO

        def _pat(target, val):
            # callable → side_effect（模拟模块抛异常），否则 return_value
            return mock.patch.object(target, val,
                                     side_effect=cloud if callable(cloud) and val == "cloud_sources" else None,
                                     return_value=None if callable(cloud) and val == "cloud_sources" else cloud)

        patchers = [
            mock.patch.object(self.fusion, "run_today", return_value=emo),
            mock.patch.object(self.fusion, "MARKET_DIR", tmp),
        ]
        for target, attr, val in [
            (self.fusion.alternative_data, "cloud_sources", cloud),
            (self.fusion.alternative_data, "local_proxies", proxies),
            (self.fusion, "_industry_breadth", sw),
            (self.fusion, "_csi_breadth", csi),
            (self.fusion, "_macro_env", macro),
        ]:
            if callable(val):
                patchers.append(mock.patch.object(target, attr, side_effect=val))
            else:
                patchers.append(mock.patch.object(target, attr, return_value=val))
        _patch_now(self.fusion)  # 只 start 一次（不在 patchers 列表中）
        for p in patchers:
            p.start()
        self._fusion_patchers = patchers
        self.addCleanup(self._stop_fusion_patchers)
        return self.fusion.fuse_today()

    def _stop_fusion_patchers(self):
        for p in self._fusion_patchers:
            p.stop()

    def test_full_chain_temperature_and_cross(self):
        tmp = self._make_tmp()
        res = self._fuse(tmp)
        self.assertEqual(res["date"], "2026-08-10")
        # 65(发酵) → 资金95加权 74 → 题材爆发+8 82 → 本地代理+8 90 → 行业普涨+5 95
        self.assertEqual(res["temperature"], 95)
        self.assertIn("高温", res["tag"])
        self.assertTrue(any("四路资金合力95极强" in s for s in res["signals"]))
        self.assertTrue(any("行业广度双源一致" in s for s in res["signals"]))
        self.assertTrue(any("口径一致" in s for s in res["signals"]))
        self.assertTrue(any(c["type"] == "共振" for c in res["cross"]))
        # 行业双源结构
        self.assertEqual(res["industry"]["confidence"], "双源一致")
        self.assertEqual(res["industry"]["check"]["label"], "口径一致")
        self.assertEqual(res["industry"]["csi"]["status"], "available")
        # 另类数据闭环
        self.assertEqual(res["alt"]["status"], "available")
        self.assertEqual(res["alt"]["cninfo"]["status"], "available")
        self.assertEqual(res["alt"]["hot_rank"]["status"], "available")
        self.assertEqual(res["alt"]["local_proxies"]["status"], "available")
        self.assertEqual(res["macro_env"]["status"], "available")

    def test_breadth_divergence_label(self):
        tmp = self._make_tmp()
        csi_low = dict(CSI_OK, breadth=0.2)
        res = self._fuse(tmp, csi=csi_low)
        self.assertEqual(res["industry"]["confidence"], "分歧")
        self.assertEqual(res["industry"]["check"]["label"], "口径分歧")
        self.assertIn("差60.0pct", res["industry"]["check"]["note"])
        # 修正减半发生在温度修正信号里（industry.check 只展示口径说明）
        self.assertTrue(any("修正减半" in s for s in res["signals"]))
        self.assertTrue(any("+2.5" in s for s in res["signals"]))
        # 65→(资金95)74→题材+8 82→代理+8 90→行业分歧修正减半+2.5 92.5 → int 截断 92
        self.assertEqual(res["temperature"], 92)

    def test_cloud_and_proxy_down_closed_loop(self):
        tmp = self._make_tmp()

        def boom(*a, **k):
            raise RuntimeError("cloud down")

        res = self._fuse(tmp, cloud=boom, proxies=boom)
        self.assertEqual(res["alt"]["status"], "unavailable")
        self.assertEqual(res["alt"]["cninfo"]["status"], "unavailable")
        self.assertEqual(res["alt"]["hot_rank"]["status"], "unavailable")
        self.assertEqual(res["alt"]["social"]["status"], "unavailable")
        self.assertEqual(res["alt"]["local_proxies"]["status"], "unavailable")
        self.assertEqual(res["industry"]["status"], "available")  # 行业不受影响

    def test_stale_cloud_removed(self):
        tmp = self._make_tmp()
        stale_cloud = {"status": "available", "date": "2026-08-01",
                       "signal": "高扰动", "total_announcements": 1,
                       "hot_rank": {"status": "available", "as_of": "2026-08-01",
                                    "avg_pct": 1.0},
                       "social": {"status": "available", "as_of": "2026-08-01"}}
        res = self._fuse(tmp, cloud=stale_cloud)
        self.assertEqual(res["alt"]["cninfo"]["status"], "unavailable")
        self.assertEqual(res["alt"]["hot_rank"]["status"], "unavailable")
        self.assertTrue(any("巨潮公告云快照落后9日" in s for s in res["signals"]))

    def test_stale_market_data_removed_from_fusion(self):
        tmp = self._make_tmp()

        def _unav(*a, **k):
            return {"status": "unavailable", "reason": "test"}

        res = self._fuse(tmp, stats_date="2026-08-01",  # 落后 ref 9 日
                         cloud=_unav, proxies=_unav, sw=_unav, csi=_unav, macro=None)
        self.assertTrue(any("资金合力落后9日" in s for s in res["signals"]))
        self.assertTrue(any("涨停统计落后9日" in s for s in res["signals"]))
        # 旧数据不参与温度 → 保持情绪阶段基准
        self.assertEqual(res["temperature"], self.fusion.STAGE_TEMP["ferment"])
        self.assertEqual(res["signals"][0].startswith("⚠️ 数据降级"), True)

    def test_macro_env_exception_degrades(self):
        tmp = self._make_tmp()

        def boom(*a, **k):
            raise RuntimeError("macro down")

        res = self._fuse(tmp, macro=boom)
        self.assertEqual(res["macro_env"]["status"], "unavailable")
        self.assertIn("RuntimeError", res["macro_env"]["reason"])

    def test_macro_stagflation_correction(self):
        tmp = self._make_tmp()
        macro = {"status": "available", "label": "滞胀", "cpi": 4.2, "m2": 11.0,
                 "as_of": "2026-07"}
        res = self._fuse(tmp, macro=macro)
        # 65→(资金95)74→题材+8 82→代理+8 90→行业+5 95→滞胀-3 92
        self.assertEqual(res["temperature"], 92)
        self.assertTrue(any("滞胀风险-3" in s for s in res["signals"]))
        self.assertEqual(res["macro_env"]["label"], "滞胀")


class TestReadFusionLatest(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _patch_out(self, tmp):
        p = mock.patch.object(self.fusion, "MARKET_DIR", tmp)
        p.start()
        self.addCleanup(p.stop)

    def _write(self, tmp, rows, corrupt=False):
        df = pd.DataFrame(rows)
        if corrupt:
            (tmp / "fusion.parquet").write_bytes(b"not-parquet")
        else:
            df.to_parquet(tmp / "fusion.parquet", index=False)

    def test_latest_at_or_before_ref(self):
        tmp = self._make_tmp()
        self._write(tmp, {"date": ["2026-08-01", "2026-08-05"],
                          "temperature": [60, 75], "signals": ["a", "b"]})
        self._patch_out(tmp)
        res = self.fusion.read_fusion_latest("2026-08-06")
        self.assertEqual(res["date"], "2026-08-05")
        self.assertEqual(res["temperature"], 75)
        res2 = self.fusion.read_fusion_latest("2026-08-01")
        self.assertEqual(res2["date"], "2026-08-01")

    def test_column_selection(self):
        tmp = self._make_tmp()
        self._write(tmp, {"date": ["2026-08-05"], "temperature": [75],
                          "signals": ["b"]})
        self._patch_out(tmp)
        res = self.fusion.read_fusion_latest("2026-08-06", cols=["temperature"])
        self.assertEqual(set(res.keys()), {"date", "temperature"})

    def test_no_data_file(self):
        tmp = self._make_tmp()
        self._patch_out(tmp)
        self.assertIsNone(self.fusion.read_fusion_latest("2026-08-06"))

    def test_future_ref_returns_none(self):
        tmp = self._make_tmp()
        self._write(tmp, {"date": ["2026-08-05"], "temperature": [75]})
        self._patch_out(tmp)
        self.assertIsNone(self.fusion.read_fusion_latest("2026-08-01"))

    def test_corrupt_file_degrades_to_none(self):
        """域G 修复后：损坏 parquet → 降级返回 None 不抛（docstring 承诺的降级闭环）。"""
        tmp = self._make_tmp()
        self._write(tmp, {}, corrupt=True)
        self._patch_out(tmp)
        self.assertIsNone(self.fusion.read_fusion_latest("2026-08-06"))

    def test_invalid_dates_returns_none(self):
        tmp = self._make_tmp()
        self._write(tmp, {"date": ["garbage"], "temperature": [75]})
        self._patch_out(tmp)
        self.assertIsNone(self.fusion.read_fusion_latest("2026-08-06"))


class TestIndustryBreadthParquet(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _patch_paths(self, tmp, hist_exists=True, sw_exists=True):
        hist_path = tmp / "sw_first_hist.parquet" if hist_exists else tmp / "missing_hist.parquet"
        sw_path = tmp / "sw_first.parquet" if sw_exists else tmp / "missing_sw.parquet"
        p1 = mock.patch.object(self.fusion, "SW_FIRST_HIST", hist_path)
        p2 = mock.patch.object(self.fusion, "SW_FIRST", sw_path)
        p1.start(); p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_normal_breadth_and_name_mapping(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "代码": ["801010", "801020", "801030", "801040"] * 2,
            "日期": ["2026-08-07"] * 4 + ["2026-08-10"] * 4,
            "收盘": [100, 100, 100, 100, 101, 99, 102, 103],
        }).to_parquet(tmp / "sw_first_hist.parquet", index=False)
        pd.DataFrame({"行业代码": ["801010.SI", "801020.SI", "801030.SI", "801040.SI"],
                      "行业名称": ["有色金属", "电力设备", "汽车", "银行"]}).to_parquet(
            tmp / "sw_first.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["as_of"], "2026-08-10")
        self.assertEqual(res["n"], 4)
        self.assertEqual(res["breadth"], 0.75)
        self.assertEqual(res["top_up"][0]["name"], "银行")
        self.assertEqual(res["top_up"][0]["pct"], 3.0)
        self.assertEqual(res["top_down"][0]["name"], "电力设备")
        self.assertEqual(res["top_down"][0]["pct"], -1.0)

    def test_missing_hist_file(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp, hist_exists=False)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("缺失", res["reason"])

    def test_no_data_before_ref(self):
        tmp = self._make_tmp()
        pd.DataFrame({"代码": ["801010"], "日期": ["2026-09-01"], "收盘": [100]}).to_parquet(
            tmp / "sw_first_hist.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("无", res["reason"])

    def test_less_than_two_trading_days(self):
        tmp = self._make_tmp()
        pd.DataFrame({"代码": ["801010", "801020"], "日期": ["2026-08-10"] * 2,
                      "收盘": [100, 101]}).to_parquet(
            tmp / "sw_first_hist.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("交易日不足 2 日", res["reason"])

    def test_no_valid_close_pairs(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "代码": ["801010"] * 2,
            "日期": ["2026-08-07", "2026-08-10"],
            "收盘": [0, 101],  # 上一日收盘 0 → 无效对
        }).to_parquet(tmp / "sw_first_hist.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("无有效收盘价对", res["reason"])

    def test_sw_map_read_error_falls_back_to_code(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "代码": ["801010", "801020"] * 2,
            "日期": ["2026-08-07"] * 2 + ["2026-08-10"] * 2,
            "收盘": [100, 100, 101, 102],
        }).to_parquet(tmp / "sw_first_hist.parquet", index=False)
        (tmp / "sw_first.parquet").write_bytes(b"corrupt")
        self._patch_paths(tmp)
        res = self.fusion._industry_breadth("2026-08-10")
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["top_up"][0]["name"], "801020")  # 名称回退为代码


class TestCsiBreadthSingleDate(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _patch_paths(self, tmp):
        p1 = mock.patch.object(self.fusion, "CSI_INDUSTRY_HIST",
                               tmp / "csi_industry_hist.parquet")
        p2 = mock.patch.object(self.fusion, "CSI_INDUSTRY",
                               tmp / "csi_industry.parquet")
        p1.start(); p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_single_date_uses_pct_column(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "日期": ["2026-08-10"] * 3,
            "指数代码": ["A001", "A002", "A003"],
            "指数中文简称": ["一", "二", "三"],
            "收盘": [100, 100, 100],
            "涨跌幅": [1.0, -2.0, 0.5],
        }).to_parquet(tmp / "csi_industry_hist.parquet", index=False)
        pd.DataFrame({"指数代码": ["A001", "A002", "A003"],
                      "指数简称": ["甲", "乙", "丙"]}).to_parquet(
            tmp / "csi_industry.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._csi_breadth("2026-08-10")
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["breadth"], round(2 / 3, 4))
        self.assertEqual(res["top_up"][0]["name"], "甲")
        self.assertEqual(res["top_down"][0]["name"], "乙")

    def test_single_date_all_nan_pct_unavailable(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "日期": ["2026-08-10"] * 2,
            "指数代码": ["A001", "A002"],
            "指数中文简称": ["一", "二"],
            "收盘": [100, 100],
            "涨跌幅": [np.nan, np.nan],
        }).to_parquet(tmp / "csi_industry_hist.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._csi_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("无有效涨跌幅", res["reason"])

    def test_no_industry_in_list_unavailable(self):
        tmp = self._make_tmp()
        pd.DataFrame({
            "日期": ["2026-08-10"] * 2,
            "指数代码": ["X001", "X002"],
            "指数中文简称": ["一", "二"],
            "收盘": [100, 100],
            "涨跌幅": [1.0, -1.0],
        }).to_parquet(tmp / "csi_industry_hist.parquet", index=False)
        pd.DataFrame({"指数代码": ["A001"], "指数简称": ["甲"]}).to_parquet(
            tmp / "csi_industry.parquet", index=False)
        self._patch_paths(tmp)
        res = self.fusion._csi_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("最新交易日无行业指数", res["reason"])


class TestMacroLatestYoy(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _patch_cpi(self, tmp, rows=None, corrupt=False):
        path = tmp / "cpi_yearly.parquet"
        if corrupt:
            path.write_bytes(b"bad")
        elif rows is not None:
            pd.DataFrame(rows, columns=["月份", "全国-同比增长"]).to_parquet(path, index=False)
        p = mock.patch.object(self.fusion, "CPI_YEARLY", path)
        p.start()
        self.addCleanup(p.stop)

    def test_latest_yoy_within_ref(self):
        tmp = self._make_tmp()
        # 语义：月份按月初日期比较，只取 ≤ref 的最新一期（08 期月初 08-01 ≤ ref 时会被纳入）
        self._patch_cpi(tmp, [("2026年06月份", 2.0), ("2026年07月份", 2.5),
                              ("2026年08月份", 99.0)])
        v, as_of = self.fusion._macro_latest_yoy(
            tmp / "cpi_yearly.parquet", ["全国-同比增长"], "2026-07-31")
        self.assertEqual(v, 2.5)
        self.assertEqual(as_of, "2026-07")
        v2, as_of2 = self.fusion._macro_latest_yoy(
            tmp / "cpi_yearly.parquet", ["全国-同比增长"], "2026-08-11")
        self.assertEqual(v2, 99.0)  # 月初 08-01 ≤ ref → 08 期已"可见"
        self.assertEqual(as_of2, "2026-08")

    def test_missing_file(self):
        tmp = self._make_tmp()
        v, as_of = self.fusion._macro_latest_yoy(
            tmp / "nope.parquet", ["全国-同比增长"], "2026-08-11")
        self.assertIsNone(v)
        self.assertIsNone(as_of)

    def test_missing_column(self):
        tmp = self._make_tmp()
        pd.DataFrame({"月份": ["2026年07月份"], "别的列": [1.0]}).to_parquet(
            tmp / "cpi.parquet", index=False)
        v, as_of = self.fusion._macro_latest_yoy(
            tmp / "cpi.parquet", ["全国-同比增长"], "2026-08-11")
        self.assertIsNone(v)

    def test_bad_month_format_skipped(self):
        tmp = self._make_tmp()
        self._patch_cpi(tmp, [("bad-month", 1.0), ("2026年07月份", 2.0)])
        v, as_of = self.fusion._macro_latest_yoy(
            tmp / "cpi_yearly.parquet", ["全国-同比增长"], "2026-08-11")
        self.assertEqual(v, 2.0)


class TestLagsAndStore(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def test_data_lag(self):
        self.assertIsNone(self.fusion._data_lag(None, "2026-08-10"))
        self.assertIsNone(self.fusion._data_lag(pd.DataFrame(), "2026-08-10"))
        df = pd.DataFrame({"date": [pd.Timestamp("2026-08-07")]})
        self.assertEqual(self.fusion._data_lag(df, "2026-08-10"), 3)
        df2 = pd.DataFrame({"date": [pd.Timestamp("2026-08-11")]})
        self.assertEqual(self.fusion._data_lag(df2, "2026-08-10"), 0)

    def test_as_of_lag(self):
        self.assertEqual(self.fusion._as_of_lag("20260807", "2026-08-10"), 3)
        self.assertEqual(self.fusion._as_of_lag("2026-08-07", "2026-08-10"), 3)
        self.assertEqual(self.fusion._as_of_lag("2026-07", "2026-08-10"), 40)
        self.assertIsNone(self.fusion._as_of_lag("", "2026-08-10"))
        self.assertIsNone(self.fusion._as_of_lag(None, "2026-08-10"))
        self.assertIsNone(self.fusion._as_of_lag("garbage", "2026-08-10"))

    def test_store_dedup_by_date(self):
        tmp = self._make_tmp()
        p = mock.patch.object(self.fusion, "OUT", tmp / "fusion.parquet")
        p.start()
        self.addCleanup(p.stop)
        res1 = {"date": "2026-08-10", "temperature": 60, "signals": ["a"],
                "cross": [], "alt": {}, "industry": {}, "macro_env": {}}
        res2 = dict(res1, temperature=90)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.fusion.store(res1)
            self.fusion.store(res2)
        df = pd.read_parquet(tmp / "fusion.parquet")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["temperature"], 90)

    def test_load_latest_with_and_without_cols(self):
        tmp = self._make_tmp()
        pd.DataFrame({"a": [1], "b": [2]}).to_parquet(tmp / "x.parquet", index=False)
        p = mock.patch.object(self.fusion, "MARKET_DIR", tmp)
        p.start()
        self.addCleanup(p.stop)
        self.assertEqual(list(self.fusion._load_latest("x.parquet").columns), ["a", "b"])
        self.assertEqual(list(self.fusion._load_latest("x.parquet", ["a"]).columns), ["a"])
        self.assertIsNone(self.fusion._load_latest("missing.parquet"))


if __name__ == "__main__":
    unittest.main()

