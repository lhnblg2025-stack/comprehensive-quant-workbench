"""analysis_core/data_sources 统一数据源注册表单元测试。

覆盖:
  - _cloud_snapshot_status: 目录缺失/无匹配/文件名 YYYYMMDD/parquet date 列/mtime
    兜底/无法解析 as_of /新鲜度阈值（陈旧标注）
  - _cloud_snapshots_status: 全云快照自检 + social(weibo+baidu) 任一新鲜即 ok
  - check_availability: 全源 mock（akshare/pywencai/pytdx/baostock）+ 分类状态
    兜底（tushare/ifind api_enabled=False）+ 云快照合并 + verbose 输出
  - _ths_parse_members: 新版 stockpage 链接 / 旧版 /stock/code/ / JSON data
  - fetch_ths_concepts / fetch_ths_concept_members: 增量合并去重、断点续传、
    refresh、失败降级、空行、保存路径
  - ths_members_status: 进度统计（无文件/正常/损坏降级）

无网络：所有外部调用全部 mock，路径常量用 mock.patch.object 覆盖到 tmp。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.data_sources as ds
    return ds


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="ds_v11_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


class TestCloudSnapshotStatus(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def test_missing_dir(self):
        tmp = self._make_tmp()
        r = self.ds._cloud_snapshot_status(tmp / "nope", "*.parquet")
        self.assertFalse(r["ok"])
        self.assertIn("目录缺失", r["note"])
        self.assertEqual(r["n_files"], 0)
        self.assertIsNone(r["as_of"])

    def test_no_matching_files(self):
        tmp = self._make_tmp()
        (tmp / "sub").mkdir()
        r = self.ds._cloud_snapshot_status(tmp / "sub", "*.parquet")
        self.assertFalse(r["ok"])
        self.assertIn("无匹配文件", r["note"])

    def test_filename_date_fresh(self):
        tmp = self._make_tmp()
        d = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        (tmp / f"hot_rank_{d}.parquet").write_bytes(b"x")
        r = self.ds._cloud_snapshot_status(tmp, "hot_rank_*.parquet")
        self.assertTrue(r["ok"])
        self.assertEqual(r["as_of"], d)
        self.assertIn("新鲜", r["note"])

    def test_filename_date_stale(self):
        tmp = self._make_tmp()
        old = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        (tmp / f"hot_rank_{old}.parquet").write_bytes(b"x")
        r = self.ds._cloud_snapshot_status(tmp, "hot_rank_*.parquet")
        self.assertFalse(r["ok"])
        self.assertIn("陈旧", r["note"])
        self.assertIn("阈值3", r["note"])

    def test_parquet_date_column(self):
        tmp = self._make_tmp()
        f = tmp / "sw_first.parquet"
        pd.DataFrame({"date": [pd.Timestamp(datetime.now().date())]}).to_parquet(f)
        r = self.ds._cloud_snapshot_status(tmp, "sw_first.parquet")
        self.assertTrue(r["ok"])
        self.assertEqual(r["as_of"], datetime.now().strftime("%Y%m%d"))

    def test_parquet_cn_date_column(self):
        tmp = self._make_tmp()
        f = tmp / "car.parquet"
        pd.DataFrame({"日期": [pd.Timestamp(datetime.now().date())]}).to_parquet(f)
        r = self.ds._cloud_snapshot_status(tmp, "car.parquet")
        self.assertTrue(r["ok"])

    def test_mtime_fallback(self):
        tmp = self._make_tmp()
        f = tmp / "baidu_hot_nodate.json"
        f.write_text("{}")
        r = self.ds._cloud_snapshot_status(tmp, "baidu_hot_*.json")
        self.assertTrue(r["ok"])
        self.assertEqual(r["as_of"], datetime.now().strftime("%Y%m%d"))

    def test_unparseable_as_of(self):
        tmp = self._make_tmp()
        (tmp / "bad.parquet").write_bytes(b"x")

        class _FakeDt:
            @staticmethod
            def fromtimestamp(*a, **k):
                raise OSError("boom")

            @staticmethod
            def now(tz=None):
                return datetime.now(tz)

            @staticmethod
            def strptime(s, fmt):
                return datetime.strptime(s, fmt)

        with mock.patch.object(self.ds, "datetime", _FakeDt):
            r = self.ds._cloud_snapshot_status(tmp, "bad.parquet")
        self.assertFalse(r["ok"])
        self.assertIn("无法解析 as_of", r["note"])


class TestCloudSnapshotsStatus(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def _make_dw(self, tmp, **files):
        dw = tmp / "data_warehouse"
        for rel, payload in files.items():
            p = dw / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str):
                p.write_text(payload)
            else:
                p.write_bytes(payload)
        return dw

    def test_all_sources(self):
        tmp = self._make_tmp()
        today = datetime.now().strftime("%Y%m%d")
        self._make_dw(
            tmp,
            **{
                "hot_rank/hot_rank_%s.parquet" % today: b"x",
                "social/weibo_%s.parquet" % today: b"x",
                "social/baidu_hot_%s.json" % today: "{}",
                "cninfo/cninfo_%s.json" % today: "{}",
                "classification/concept_member_ths.parquet": b"x",
                "industry/sw_first.parquet": b"x",
                "industry/csi_industry.parquet": b"x",
                "industry/car_2026.parquet": b"x",
            },
        )
        with mock.patch.object(self.ds, "ROOT", tmp):
            checks = self.ds._cloud_snapshots_status()
        for k in ("hot_rank", "social", "cninfo_cloud", "ths_concept",
                  "industry_sw", "industry_csi", "industry_car"):
            self.assertIn(k, checks)
            self.assertTrue(checks[k]["ok"], k)
        self.assertIn("baidu", checks["social"]["note"])
        self.assertEqual(checks["social"]["n_files"], 2)

    def test_social_baidu_fallback(self):
        tmp = self._make_tmp()
        today = datetime.now().strftime("%Y%m%d")
        self._make_dw(tmp, **{"social/baidu_hot_%s.json" % today: "{}"})
        with mock.patch.object(self.ds, "ROOT", tmp):
            checks = self.ds._cloud_snapshots_status()
        self.assertTrue(checks["social"]["ok"])
        self.assertEqual(checks["social"]["as_of"], today)
        self.assertFalse(checks["hot_rank"]["ok"])


def _fake_ak(**overrides):
    ak = types.ModuleType("akshare")

    def _df(n=3):
        return pd.DataFrame({"code": [f"{i:06d}" for i in range(n)]})

    class _BoardName:
        def __call__(self, *a, **k):
            return pd.DataFrame({"name": ["概念A", "概念B"], "code": ["BK1", "BK2"]})

    base = {
        "stock_zt_pool_em": lambda *a, **k: _df(),
        "stock_board_concept_name_ths": _BoardName(),
        "stock_zh_a_disclosure_report_cninfo": lambda *a, **k: _df(2),
    }
    for name, fn in overrides.items():
        base[name] = fn
    for name, fn in base.items():
        setattr(ak, name, fn)
    return ak


class TestCheckAvailability(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def test_all_sources_ok(self):
        tmp = self._make_tmp()
        today = datetime.now().strftime("%Y%m%d")
        fake_ak = _fake_ak()
        fake_wencai = types.ModuleType("pywencai")
        fake_wencai.get = lambda *a, **k: pd.DataFrame({"x": [1]})
        fake_bs = types.ModuleType("baostock")
        fake_bs.login = lambda: types.SimpleNamespace(error_code="0", error_msg="ok")
        fake_bs.logout = lambda: None
        fake_tdx = types.ModuleType("pytdx.hq")
        fake_tdx.TdxHq_API = lambda: types.SimpleNamespace(
            connect=lambda *a, **k: True,
            get_security_count=lambda *a, **k: 5000,
            disconnect=lambda: None,
        )
        fake_pytdx = types.ModuleType("pytdx")
        fake_pytdx.hq = fake_tdx
        self._write_dw(tmp, today)
        patches = [
            mock.patch.dict(sys.modules, {
                "akshare": fake_ak, "pywencai": fake_wencai,
                "baostock": fake_bs, "pytdx": fake_pytdx, "pytdx.hq": fake_tdx,
            }),
            mock.patch.object(self.ds, "ROOT", tmp),
        ]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        res = self.ds.check_availability(verbose=False)
        self.assertEqual(res["eastmoney"]["ok"], True)
        self.assertEqual(res["ths"]["ok"], True)
        self.assertEqual(res["wencai"]["ok"], True)
        self.assertEqual(res["pytdx"]["ok"], True)
        self.assertEqual(res["cninfo"]["ok"], True)
        self.assertEqual(res["baostock"]["ok"], True)
        self.assertEqual(res["tushare"]["ok"], False)
        self.assertEqual(res["ifind"]["ok"], False)
        self.assertEqual(res["hot_rank"]["ok"], True)
        self.assertEqual(res["industry_car"]["ok"], True)
        # 所有注册源都有自检条目
        for k in self.ds.SOURCE_STATUS:
            self.assertIn(k, res)

    def test_source_failures_annotated(self):
        tmp = self._make_tmp()

        def boom(*a, **k):
            raise RuntimeError("网络错误")

        fake_ak = _fake_ak(
            stock_zt_pool_em=boom,
            stock_board_concept_name_ths=boom,
            stock_zh_a_disclosure_report_cninfo=boom,
        )
        fake_wencai = types.ModuleType("pywencai")
        fake_wencai.get = lambda *a, **k: None
        fake_bs = types.ModuleType("baostock")
        fake_bs.login = lambda: (_ for _ in ()).throw(RuntimeError("conn"))
        fake_bs.logout = lambda: None
        fake_tdx = types.ModuleType("pytdx.hq")
        fake_tdx.TdxHq_API = lambda: types.SimpleNamespace(
            connect=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("conn")))
        fake_pytdx = types.ModuleType("pytdx")
        fake_pytdx.hq = fake_tdx
        patches = [
            mock.patch.dict(sys.modules, {
                "akshare": fake_ak, "pywencai": fake_wencai,
                "baostock": fake_bs, "pytdx": fake_pytdx, "pytdx.hq": fake_tdx,
            }),
            mock.patch.object(self.ds, "ROOT", tmp),
        ]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        res = self.ds.check_availability(verbose=False)
        for k in ("eastmoney", "ths", "cninfo", "wencai", "baostock", "pytdx"):
            self.assertFalse(res[k]["ok"], k)
            self.assertTrue(res[k]["note"])

    def test_verbose_prints(self):
        tmp = self._make_tmp()
        today = datetime.now().strftime("%Y%m%d")
        self._write_dw(tmp, today)
        fake_ak = _fake_ak()
        fake_wencai = types.ModuleType("pywencai")
        fake_wencai.get = lambda *a, **k: pd.DataFrame({"x": [1]})
        fake_bs = types.ModuleType("baostock")
        fake_bs.login = lambda: types.SimpleNamespace(error_code="0", error_msg="ok")
        fake_bs.logout = lambda: None
        fake_tdx = types.ModuleType("pytdx.hq")
        fake_tdx.TdxHq_API = lambda: types.SimpleNamespace(
            connect=lambda *a, **k: True,
            get_security_count=lambda *a, **k: 10,
            disconnect=lambda: None)
        fake_pytdx = types.ModuleType("pytdx")
        fake_pytdx.hq = fake_tdx
        patches = [
            mock.patch.dict(sys.modules, {
                "akshare": fake_ak, "pywencai": fake_wencai,
                "baostock": fake_bs, "pytdx": fake_pytdx, "pytdx.hq": fake_tdx,
            }),
            mock.patch.object(self.ds, "ROOT", tmp),
        ]
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        from contextlib import redirect_stdout
        import io
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.ds.check_availability(verbose=True)
        self.assertIn("eastmoney", buf.getvalue())

    def _write_dw(self, tmp, today):
        dw = tmp / "data_warehouse"
        files = {
            "hot_rank/hot_rank_%s.parquet" % today: b"x",
            "social/weibo_%s.parquet" % today: b"x",
            "cninfo/cninfo_%s.json" % today: "{}",
            "classification/concept_member_ths.parquet": b"x",
            "industry/sw_first.parquet": b"x",
            "industry/csi_industry.parquet": b"x",
            "industry/car_2026.parquet": b"x",
        }
        for rel, payload in files.items():
            p = dw / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str):
                p.write_text(payload)
            else:
                p.write_bytes(payload)


class TestThsParseMembers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def test_new_structure(self):
        html = ('<a href="http://stockpage.10jqka.com.cn/002298/">'
                '<a href="http://stockpage.10jqka.com.cn/600519/">')
        out = self.ds._ths_parse_members(html)
        self.assertEqual(out, [("002298", ""), ("600519", "")])

    def test_old_structure_fallback(self):
        html = '<a href="/stock/code/000001.html">'
        out = self.ds._ths_parse_members(html)
        self.assertEqual(out, [("000001", "")])

    def test_json_payload(self):
        html = json.dumps({"data": '<a href="http://stockpage.10jqka.com.cn/300750/">'})
        out = self.ds._ths_parse_members(html)
        self.assertEqual(out, [("300750", "")])

    def test_json_invalid_falls_back_to_regex(self):
        html = '{"data": <broken>} <a href="http://stockpage.10jqka.com.cn/000002/">'
        out = self.ds._ths_parse_members(html)
        self.assertEqual(out, [("000002", "")])

    def test_no_match(self):
        self.assertEqual(self.ds._ths_parse_members("<html>no links</html>"), [])


class TestFetchThsConcepts(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def test_save(self):
        tmp = self._make_tmp()
        market = tmp / "market"
        market.mkdir(parents=True)
        fake_ak = _fake_ak()
        with mock.patch.dict(sys.modules, {"akshare": fake_ak}), \
                mock.patch.object(self.ds, "MARKET_DIR", market):
            out = self.ds.fetch_ths_concepts(save=True)
        self.assertEqual(list(out.columns), ["concept", "code"])
        self.assertEqual(len(out), 2)
        p = market / "concept_ths_boards.parquet"
        self.assertTrue(p.exists())
        saved = pd.read_parquet(p)
        self.assertEqual(list(saved.columns), ["concept", "code"])

    def test_no_save(self):
        tmp = self._make_tmp()
        market = tmp / "market"
        fake_ak = _fake_ak()
        with mock.patch.dict(sys.modules, {"akshare": fake_ak}), \
                mock.patch.object(self.ds, "MARKET_DIR", market):
            out = self.ds.fetch_ths_concepts(save=False)
        self.assertEqual(len(out), 2)
        self.assertFalse((market / "concept_ths_boards.parquet").exists())


class TestFetchThsConceptMembers(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def _patch(self, tmp):
        market = tmp / "market"
        market.mkdir(parents=True)
        alt = tmp / "dw" / "classification"
        alt.mkdir(parents=True)
        patches = [
            mock.patch.object(self.ds, "MARKET_DIR", market),
            mock.patch.object(self.ds, "THS_MEMBER_FILE", market / "concept_member_ths.parquet"),
            mock.patch.object(self.ds, "THS_MEMBER_ALT", alt / "concept_member_ths.parquet"),
            mock.patch.object(self.ds, "THS_MEMBER_DONE", market / "done.json"),
        ]
        return market, patches

    def _page_fn(self, page1_html, page2_html=""):
        def _fake_page(url, timeout=15):
            if "/page/1/" in url:
                return page1_html
            if page2_html and "/page/2/" in url:
                return page2_html
            return ""
        return _fake_page

    def test_incremental_fetch_merges_dedup(self):
        tmp = self._make_tmp()
        market, patches = self._patch(tmp)
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        page = self._page_fn(
            '<a href="http://stockpage.10jqka.com.cn/000001/">'
            '<a href="http://stockpage.10jqka.com.cn/000002/">',
            '<a href="http://stockpage.10jqka.com.cn/000003/">',
        )
        with mock.patch.dict(sys.modules, {"akshare": _fake_ak()}), \
                mock.patch.object(self.ds, "_ths_page_text", side_effect=page), \
                mock.patch("time.sleep"):
            df = self.ds.fetch_ths_concept_members()
        self.assertEqual(len(df), 6)  # 2 板块 × 3 代码
        self.assertTrue((market / "done.json").exists())
        done = json.loads((market / "done.json").read_text(encoding="utf-8"))
        self.assertIn("BK1", done)
        self.assertIn("BK2", done)
        # 两个输出路径都落盘
        self.assertTrue((market / "concept_member_ths.parquet").exists())
        self.assertTrue((tmp / "dw" / "classification" / "concept_member_ths.parquet").exists())

    def test_merge_with_existing_dedup(self):
        tmp = self._make_tmp()
        market, patches = self._patch(tmp)
        pd.DataFrame({"concept": ["概念A"], "code": ["000001"]}).to_parquet(
            market / "concept_member_ths.parquet")
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        page = self._page_fn('<a href="http://stockpage.10jqka.com.cn/000001/">'
                             '<a href="http://stockpage.10jqka.com.cn/000002/">')
        with mock.patch.dict(sys.modules, {"akshare": _fake_ak()}), \
                mock.patch.object(self.ds, "_ths_page_text", side_effect=page), \
                mock.patch("time.sleep"):
            df = self.ds.fetch_ths_concept_members(limit=1)
        self.assertEqual(len(df), 2)
        saved = pd.read_parquet(market / "concept_member_ths.parquet")
        self.assertEqual(len(saved), 2)  # 000001 与旧数据去重, 000002 新增

    def test_refresh_redoes_done_boards(self):
        tmp = self._make_tmp()
        market, patches = self._patch(tmp)
        (market / "done.json").write_text(json.dumps({"BK1": True, "BK2": True}))
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        page = self._page_fn('<a href="http://stockpage.10jqka.com.cn/000009/">')
        with mock.patch.dict(sys.modules, {"akshare": _fake_ak()}), \
                mock.patch.object(self.ds, "_ths_page_text", side_effect=page), \
                mock.patch("time.sleep"):
            df = self.ds.fetch_ths_concept_members(refresh=True)
        self.assertEqual(len(df), 2)  # 两个板块都重抓
        self.assertEqual(set(df["code"]), {"000009"})

    def test_page_failure_recorded(self):
        tmp = self._make_tmp()
        market, patches = self._patch(tmp)
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)

        def boom(url, timeout=15):
            raise RuntimeError("timeout")

        with mock.patch.dict(sys.modules, {"akshare": _fake_ak()}), \
                mock.patch.object(self.ds, "_ths_page_text", side_effect=boom), \
                mock.patch("time.sleep"):
            df = self.ds.fetch_ths_concept_members()
        self.assertTrue(df.empty)
        # got=0 → 不写 done 文件
        self.assertFalse((market / "done.json").exists())

    def test_boards_none_early_return(self):
        tmp = self._make_tmp()
        market, patches = self._patch(tmp)
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        fake_ak = _fake_ak()
        fake_ak.stock_board_concept_name_ths = lambda *a, **k: None
        with mock.patch.dict(sys.modules, {"akshare": fake_ak}):
            df = self.ds.fetch_ths_concept_members()
        self.assertTrue(df.empty)


class TestThsMembersStatus(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _import()

    def test_empty(self):
        tmp = self._make_tmp()
        market = tmp / "market"
        market.mkdir(parents=True)
        with mock.patch.object(self.ds, "MARKET_DIR", market), \
                mock.patch.object(self.ds, "THS_MEMBER_FILE", market / "m.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_ALT", market / "m2.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_DONE", market / "done.json"):
            st = self.ds.ths_members_status()
        self.assertEqual(st, {"boards_total": 0, "boards_done": 0, "member_rows": 0, "pct": 0.0})

    def test_normal(self):
        tmp = self._make_tmp()
        market = tmp / "market"
        market.mkdir(parents=True)
        pd.DataFrame({"concept": ["A"], "code": ["000001"]}).to_parquet(
            market / "concept_ths_boards.parquet")
        (market / "done.json").write_text(json.dumps({"BK1": True}))
        pd.DataFrame({"concept": ["A"], "code": ["000001"]}).to_parquet(market / "m.parquet")
        with mock.patch.object(self.ds, "MARKET_DIR", market), \
                mock.patch.object(self.ds, "THS_MEMBER_FILE", market / "m.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_ALT", market / "m2.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_DONE", market / "done.json"):
            st = self.ds.ths_members_status()
        self.assertEqual(st["boards_total"], 1)
        self.assertEqual(st["boards_done"], 1)
        self.assertEqual(st["member_rows"], 1)
        self.assertEqual(st["pct"], 100.0)

    def test_corrupt_done_json(self):
        tmp = self._make_tmp()
        market = tmp / "market"
        market.mkdir(parents=True)
        pd.DataFrame({"concept": ["A"], "code": ["000001"]}).to_parquet(
            market / "concept_ths_boards.parquet")
        (market / "done.json").write_text("{broken")
        with mock.patch.object(self.ds, "MARKET_DIR", market), \
                mock.patch.object(self.ds, "THS_MEMBER_FILE", market / "m.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_ALT", market / "m2.parquet"), \
                mock.patch.object(self.ds, "THS_MEMBER_DONE", market / "done.json"):
            st = self.ds.ths_members_status()
        self.assertEqual(st["boards_done"], 0)
        self.assertEqual(st["member_rows"], 0)


if __name__ == "__main__":
    unittest.main()
