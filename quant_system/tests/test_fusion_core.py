"""analysis_core/fusion 核心链路单元测试。

覆盖:
  - _csi_breadth：正常计算 / 数据缺失 / 陈旧>3日剔除（fuse_today 层）
  - 申万×中证双源校验：一致→confidence / 分歧→温度修正减半
  - _macro_env：温和 / 通缩 / 滞胀 / 45 天陈旧 stale 不修正 / 缺失
  - fuse_today：温度 [0,100] 钳制、返回结构（alt/industry/macro_env 字段）

文件读取全部用构造 parquet（临时目录）或 mock 隔离，不依赖真实 data_warehouse。
"""

from __future__ import annotations

import shutil
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
    import quant_system.analysis_core.fusion as fusion
    return fusion


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="fusion_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


_TODAY = datetime.now().strftime("%Y-%m-%d")
_STALE = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

EMO_FERMENT = {
    "date": _TODAY,
    "data_date": _TODAY,
    "lag_days": 1,
    "stage": "ferment",
    "stage_cn": "发酵期",
    "zt_cnt": 60,
    "max_board": 4,
    "zb_rate": 0.2,
    "premium": 1.0,
}
EMO_CLIMAX = dict(EMO_FERMENT, stage="climax", stage_cn="高潮期", zt_cnt=110)
EMO_ICE = dict(EMO_FERMENT, stage="ice", stage_cn="冰点期", zt_cnt=20, max_board=1)


def _unavailable_industry(reason="测试无数据"):
    return {"status": "unavailable", "as_of": None, "breadth": None,
            "top_up": [], "top_down": [], "reason": reason}


def _sw_available(breadth=0.8, as_of=None):
    as_of = as_of or _TODAY
    return {"status": "available", "as_of": as_of, "breadth": breadth,
            "top_up": [{"name": "x", "pct": 1.0}],
            "top_down": [{"name": "y", "pct": -1.0}],
            "reason": "test"}


def _csi_available(breadth=0.9, as_of=None):
    as_of = as_of or _TODAY
    return {"status": "available", "as_of": as_of, "breadth": breadth,
            "top_up": [{"name": "x", "pct": 1.0}],
            "top_down": [{"name": "y", "pct": -1.0}],
            "reason": "test"}


class _FuseTodayMixin:
    """fuse_today 的固定 mock 环境（全部输入隔离，无真实读文件/网络）。"""

    def _patch_fuse_today(self, *, emo=None, load_latest=None, cloud=None,
                         proxies=None, sw=None, csi=None, macro=None):
        self.fusion = _import()
        emo = emo or EMO_FERMENT
        load_latest = load_latest or (lambda name: None)
        cloud = cloud if cloud is not None else {"status": "unavailable"}
        proxies = proxies if proxies is not None else {"status": "unavailable"}
        sw = sw if sw is not None else _unavailable_industry()
        csi = csi if csi is not None else _unavailable_industry()
        self._patchers = [
            mock.patch.object(self.fusion, "run_today", return_value=emo),
            mock.patch.object(self.fusion, "_load_latest", side_effect=load_latest),
            mock.patch.object(self.fusion.alternative_data, "cloud_sources",
                              return_value=cloud),
            mock.patch.object(self.fusion.alternative_data, "local_proxies",
                              return_value=proxies),
            mock.patch.object(self.fusion, "_industry_breadth", return_value=sw),
            mock.patch.object(self.fusion, "_csi_breadth", return_value=csi),
            mock.patch.object(self.fusion, "_macro_env", return_value=macro),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    def _fuse(self):
        return self.fusion.fuse_today()


class TestCsiBreadth(_TmpDirMixin, unittest.TestCase):
    """_csi_breadth 正常/缺失。"""

    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _write_csi_files(self, hist, ind):
        self._make_tmp()
        hist.to_parquet(self._tmp / "csi_industry_hist.parquet")
        ind.to_parquet(self._tmp / "csi_industry.parquet")

    def _patch_paths(self):
        p = mock.patch.object(self.fusion, "CSI_INDUSTRY_HIST",
                              self._tmp / "csi_industry_hist.parquet")
        q = mock.patch.object(self.fusion, "CSI_INDUSTRY",
                              self._tmp / "csi_industry.parquet")
        p.start(); q.start()
        self.addCleanup(p.stop); self.addCleanup(q.stop)

    def test_normal_breadth_and_ranking(self):
        hist = pd.DataFrame({
            "日期": ["2026-08-07"] * 4 + ["2026-08-10"] * 4,
            "指数代码": ["A001", "A002", "A003", "A004"] * 2,
            "指数中文简称": ["一", "二", "三", "四"] * 2,
            "收盘": [100, 100, 100, 100, 101, 98, 103, 99],
            "涨跌幅": [np.nan] * 4 + [np.nan, -2.0, np.nan, 1.0],
        })
        ind = pd.DataFrame({
            "指数代码": ["A001", "A002", "A003"],
            "指数简称": ["名称一", "名称二", "名称三"],
        })
        self._write_csi_files(hist, ind)
        self._patch_paths()
        res = self.fusion._csi_breadth("2026-08-10")
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["as_of"], "2026-08-10")
        # A004 不在行业清单 → 剔除；A001/A003 用收盘自算，A002 用现成涨跌幅
        self.assertEqual(res["n"], 3)
        self.assertAlmostEqual(res["breadth"], round(2 / 3, 4))
        self.assertEqual(res["top_up"][0]["name"], "名称三")
        self.assertEqual(res["top_up"][0]["pct"], 3.0)
        self.assertEqual(res["top_down"][0]["name"], "名称二")
        self.assertEqual(res["top_down"][0]["pct"], -2.0)

    def test_missing_hist_file(self):
        self._make_tmp()
        self._patch_paths()  # hist 文件不存在
        res = self.fusion._csi_breadth("2026-08-10")
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("缺失", res["reason"])

    def test_latest_date_not_in_range(self):
        hist = pd.DataFrame({
            "日期": ["2026-07-01", "2026-07-02"],
            "指数代码": ["A001", "A001"],
            "指数中文简称": ["一", "一"],
            "收盘": [100, 101],
            "涨跌幅": [np.nan, 1.0],
        })
        ind = pd.DataFrame({"指数代码": ["A001"], "指数简称": ["名称一"]})
        self._write_csi_files(hist, ind)
        self._patch_paths()
        res = self.fusion._csi_breadth("2026-08-10")  # 数据落后 ref 一个多月
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["as_of"], "2026-07-02")


class TestCsiStaleRemoval(_FuseTodayMixin, unittest.TestCase):
    """中证行业数据落后 >3 日 → 剔除且标记降级。"""

    def test_csi_stale_removed_from_fusion(self):
        sw = _sw_available(0.8, as_of=_TODAY)
        csi = _csi_available(0.9, as_of=_STALE)  # 落后 5 日
        self._patch_fuse_today(emo=EMO_FERMENT, sw=sw, csi=csi, macro=None)
        res = self._fuse()
        self.assertEqual(res["industry"]["csi"]["status"], "unavailable")
        self.assertTrue(any("中证行业最新交易日落后5日" in s for s in res["signals"]))
        # 中证被剔除后不参与双源置信度：confidence 不可能取任一双源值
        self.assertNotEqual(res["industry"].get("confidence"), "双源一致")
        self.assertNotEqual(res["industry"].get("confidence"), "分歧")


class TestDualSourceValidation(_FuseTodayMixin, unittest.TestCase):
    """申万×中证双源校验：一致→confidence；分歧→修正减半。"""

    def test_consistent_breadth_confidence(self):
        sw = _sw_available(0.8, as_of=_TODAY)
        csi = _csi_available(0.9, as_of=_TODAY)
        self._patch_fuse_today(emo=EMO_FERMENT, sw=sw, csi=csi, macro=None)
        res = self._fuse()
        self.assertEqual(res["industry"]["confidence"], "双源一致")
        self.assertTrue(any("行业广度双源一致" in s for s in res["signals"]))
        # 双源一致时行业广度修正生效：温度应高于无修正基准（不绑定精确 +5 白盒值）
        self.assertGreater(res["temperature"], self.fusion.STAGE_TEMP["ferment"])
        self.assertLessEqual(res["temperature"], 100)

    def test_divergence_halves_correction(self):
        sw = _sw_available(0.8, as_of=_TODAY)
        csi = _csi_available(0.2, as_of=_TODAY)  # 方向分歧
        self._patch_fuse_today(emo=EMO_FERMENT, sw=sw, csi=csi, macro=None)
        res = self._fuse()
        self.assertEqual(res["industry"]["confidence"], "分歧")
        self.assertTrue(any("修正减半" in s for s in res["signals"]))
        self.assertTrue(any("+2.5" in s for s in res["signals"]))
        # 65 + 2.5 = 67.5：fusion 用 int() 截断 → 67（若改 round 则为 68），
        # 断言区间覆盖两种取整语义，避免绑定单一实现
        self.assertIn(res["temperature"], (67, 68))


class TestMacroEnv(_TmpDirMixin, unittest.TestCase):
    """_macro_env 标签与 45 天陈旧判定。"""

    @classmethod
    def setUpClass(cls):
        cls.fusion = _import()

    def _write_macro(self, cpi_rows, m2_rows):
        self._make_tmp()
        if cpi_rows is not None:
            pd.DataFrame(cpi_rows, columns=["月份", "全国-同比增长"]).to_parquet(
                self._tmp / "cpi_yearly.parquet")
        if m2_rows is not None:
            pd.DataFrame(m2_rows, columns=["月份", "货币和准货币(M2)-同比增长"]).to_parquet(
                self._tmp / "m2_yearly.parquet")
        self._patchers = [
            mock.patch.object(self.fusion, "CPI_YEARLY",
                              self._tmp / "cpi_yearly.parquet"),
            mock.patch.object(self.fusion, "M2_YEARLY",
                              self._tmp / "m2_yearly.parquet"),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    def _env(self, cpi, m2, months=("2026年07月份",)):
        rows = [(m, v) for m, v in zip(months, (cpi,) * len(months))]
        rows2 = [(m, v) for m, v in zip(months, (m2,) * len(months))]
        self._write_macro(rows, rows2)
        return self.fusion._macro_env("2026-08-10")

    def test_mild_env(self):
        res = self._env(2.0, 8.0)
        self.assertEqual(res["status"], "available")
        self.assertEqual(res["label"], "温和")
        self.assertEqual(res["as_of"], "2026-07")

    def test_deflation_env(self):
        res = self._env(-0.5, 8.0)
        self.assertEqual(res["label"], "通缩")

    def test_stagflation_env(self):
        res = self._env(4.2, 11.0)
        self.assertEqual(res["label"], "滞胀")

    def test_stale_over_45_days(self):
        res = self._env(2.0, 8.0, months=("2026年05月份",))
        self.assertEqual(res["status"], "stale")
        self.assertEqual(res["label"], "温和")

    def test_missing_files_return_none(self):
        self._make_tmp()
        self._patchers = [
            mock.patch.object(self.fusion, "CPI_YEARLY",
                              self._tmp / "nope.parquet"),
            mock.patch.object(self.fusion, "M2_YEARLY",
                              self._tmp / "nope2.parquet"),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)
        self.assertIsNone(self.fusion._macro_env("2026-08-10"))


class TestFuseToday(_FuseTodayMixin, unittest.TestCase):
    """fuse_today 温度范围与返回结构。"""

    def _assert_structure(self, res):
        for key in ("date", "temperature", "tag", "emotion_stage",
                    "signals", "cross", "alt", "industry", "macro_env"):
            self.assertIn(key, res)
        self.assertIsInstance(res["signals"], list)
        self.assertIsInstance(res["cross"], list)
        self.assertIsInstance(res["alt"], dict)
        self.assertIsInstance(res["industry"], dict)
        self.assertIsInstance(res["macro_env"], dict)
        for key in ("cninfo", "hot_rank", "social", "local_proxies"):
            self.assertIn(key, res["alt"])
        self.assertIn("csi", res["industry"])

    def test_baseline_structure_and_range(self):
        self._patch_fuse_today(emo=EMO_FERMENT, macro=None)
        res = self._fuse()
        self._assert_structure(res)
        self.assertEqual(res["date"], _TODAY)
        self.assertEqual(res["emotion_stage"], "发酵期")
        # 无修正输入 → 温度 = 情绪阶段基准（从实现提取常量，不硬编码 65）
        self.assertEqual(res["temperature"], self.fusion.STAGE_TEMP["ferment"])
        self.assertTrue(0 <= res["temperature"] <= 100)
        self.assertEqual(res["alt"]["status"], "unavailable")
        self.assertEqual(res["macro_env"]["status"], "unavailable")

    def test_temperature_clamped_to_100(self):
        # 高潮 85 + 题材爆发 8 + 行业普涨 5 + 本地代理升温 11 = 109 → 钳制 100
        themes = pd.DataFrame({
            "date": [_TODAY] * 4,
            "role": ["主线", "主线", "支线", "支线"],
            "stage": ["burst", "burst", "rise", "rise"],
            "zt_cnt": [20] * 4,
        })
        proxies = {"status": "available", "date": _TODAY, "signal": "升温",
                   "proxies": {k: {"signal": "升温"} for k in (
                       "supply_chain_heat", "consumption_heat",
                       "new_stock_activity", "retail_focus")}}
        load = {"theme_cycle.parquet": themes}
        self._patch_fuse_today(
            emo=EMO_CLIMAX, load_latest=lambda name: load.get(name),
            proxies=proxies, sw=_sw_available(0.8), csi=_csi_available(0.9),
            macro=None)
        res = self._fuse()
        self.assertEqual(res["temperature"], 100)
        self.assertLessEqual(res["temperature"], 100)

    def test_temperature_clamped_to_0(self):
        # 冰点 15 - 炸板率 15 - 溢价 10 = -10 → 钳制 0
        stats = pd.DataFrame({"date": [_TODAY],
                              "zb_rate": [0.5], "premium": [-1.0]})
        load = {"zt_daily_stats.parquet": stats}
        self._patch_fuse_today(emo=EMO_ICE, load_latest=lambda name: load.get(name))
        res = self._fuse()
        self.assertEqual(res["temperature"], 0)
        self.assertGreaterEqual(res["temperature"], 0)
        self.assertTrue(any("炸板率50%偏高→降温" in s for s in res["signals"]))

    def test_macro_stale_no_temperature_correction(self):
        macro = {"status": "stale", "label": "温和", "cpi": 2.0, "m2": 8.0,
                 "as_of": "2026-05-01"}
        self._patch_fuse_today(emo=EMO_FERMENT, macro=macro)
        res = self._fuse()
        # stale 仅展示，不参与温度修正 → 保持情绪阶段基准（常量）
        self.assertEqual(res["temperature"], self.fusion.STAGE_TEMP["ferment"])
        self.assertTrue(any("仅展示，不参与温度修正" in s for s in res["signals"]))
        self.assertEqual(res["macro_env"]["status"], "stale")


if __name__ == "__main__":
    unittest.main()
