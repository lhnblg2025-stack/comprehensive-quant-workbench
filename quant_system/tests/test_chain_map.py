"""analysis_core/chain_map 产业链先验因果映射引擎单元测试。

覆盖:
  - _read_parquet（缺失→None / 损坏→抛）/ _build 全文件构造
  - load()（模块级缓存 + 线程安全双检锁 + 缺失文件空映射不抛）
  - upstream_of / downstream_of / chains_of（去重保序）
  - chain_temperature（三层温度、≤date 防前视、边界空结果）
  - propagate_from_concept（题材→链映射、空结果不抛、BK 直传）
  - _signal 全分支 / _cli 各参数
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

CHAINS = {
    "新能源车": [
        ("锂矿", "upstream", "有色金属"),
        ("动力电池", "midstream", "电力设备"),
        ("整车", "downstream", "汽车"),
    ],
    "光伏": [
        ("硅料", "upstream", "基础化工"),
        ("组件", "downstream", "电力设备"),
    ],
}
RELATIONS = [
    ("有色金属", "电力设备", "资源→制造"),
    ("基础化工", "电力设备", "材料→制造"),
    ("电力设备", "汽车", "制造→整车"),
]
SW_NAME2CODE = {"有色金属": "801010", "电力设备": "801020",
                "汽车": "801030", "基础化工": "801040"}
SW_CODE2NAME = {v: k for k, v in SW_NAME2CODE.items()}


def _import():
    import quant_system.analysis_core.chain_map as cm
    return cm


def _hist(extra_future: bool = False) -> pd.DataFrame:
    """6 个交易日（08-01..08-06）收盘 99→104；extra_future 追加 08-07 起 +50% 跳涨。"""
    rows = []
    for code in ("801010", "801020", "801030"):
        for i, day in enumerate(range(1, 7)):
            rows.append({"代码": code, "日期": f"2026-08-{day:02d}",
                         "收盘": 99 + i})
    if extra_future:
        for code in ("801010", "801020", "801030"):
            for j, day in enumerate(range(7, 11)):
                rows.append({"代码": code, "日期": f"2026-08-{day:02d}",
                             "收盘": 105 * 1.5 ** (j + 1)})
    return pd.DataFrame(rows)


_UNSET = object()


def _chainmap(hist=_UNSET) -> "object":
    cm = _import()
    h = _hist() if hist is _UNSET else hist
    if h is not None and not h.empty and "日期" in h.columns:
        h = h.copy()
        h["日期"] = pd.to_datetime(h["日期"])  # 与 _build 内转换口径一致
    return cm.ChainMap(CHAINS, RELATIONS, SW_NAME2CODE, SW_CODE2NAME,
                       {"000001": "有色金属", "000002": "汽车"},
                       {"固态电池": "BK1000"}, h)


class _TmpMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="chain_map_test_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


class TestReadParquet(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def test_missing_file_returns_none(self):
        self._make_tmp()
        self.assertIsNone(self.cm._read_parquet(self._tmp / "nope.parquet"))

    def test_corrupt_file_raises(self):
        self._make_tmp()
        bad = self._tmp / "bad.parquet"
        bad.write_bytes(b"not-a-parquet")
        with self.assertRaises(Exception):
            self.cm._read_parquet(bad)

    def test_valid_file_reads(self):
        self._make_tmp()
        f = self._tmp / "ok.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(f)
        df = self.cm._read_parquet(f)
        self.assertEqual(len(df), 2)


class TestBuildAndLoad(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def tearDown(self):
        self.cm._CACHE = None

    def _write_all(self, tmp):
        pd.DataFrame({"行业代码": [f"{c}.SI" for c in SW_CODE2NAME],
                      "行业名称": list(SW_CODE2NAME.values())}).to_parquet(
            tmp / "sw_first.parquet", index=False)
        pd.DataFrame({"证券代码": ["000001", "000002"],
                      "行业代码": ["801010.SI", "801030.SI"]}).to_parquet(
            tmp / "sw_first_cons.parquet", index=False)
        _hist().to_parquet(tmp / "sw_first_hist.parquet", index=False)
        pd.DataFrame({"board_name": ["固态电池"], "board_code": ["BK1000"]}).to_parquet(
            tmp / "concept_board.parquet", index=False)

    def _patch_paths(self, tmp):
        for name in ("SW_FIRST", "SW_FIRST_CONS", "SW_FIRST_HIST", "CONCEPT_BOARD"):
            mock.patch.object(self.cm, name, tmp / f"{name}.parquet".lower()).start()
        self.addCleanup(mock.patch.stopall)

    def test_build_populates_all_maps(self):
        tmp = self._make_tmp()
        self._write_all(tmp)
        self._patch_paths(tmp)
        cm_obj = self.cm._build()
        self.assertEqual(cm_obj.sw_name2code["有色金属"], "801010")
        self.assertEqual(cm_obj.sw_code2name["801010"], "有色金属")
        self.assertEqual(cm_obj.stock_sector["000001"], "有色金属")
        self.assertEqual(cm_obj.board_name2code["固态电池"], "BK1000")
        self.assertIsNotNone(cm_obj.hist)
        self.assertEqual(len(cm_obj.hist), 18)

    def test_load_caches_same_object(self):
        tmp = self._make_tmp()
        self._write_all(tmp)
        self._patch_paths(tmp)
        a = self.cm.load()
        b = self.cm.load()
        self.assertIs(a, b)

    def test_load_thread_safety_builds_once(self):
        tmp = self._make_tmp()
        self._write_all(tmp)
        self._patch_paths(tmp)
        self.cm._CACHE = None
        counter = {"n": 0}
        real_build = self.cm._build

        def slow_build():
            counter["n"] += 1
            import time
            time.sleep(0.05)
            return real_build()

        results: list = [None] * 8
        with mock.patch.object(self.cm, "_build", side_effect=slow_build):
            threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, self.cm.load()))
                       for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(counter["n"], 1)
        self.assertTrue(all(r is results[0] and r is not None for r in results))

    def test_load_missing_files_empty_maps_no_raise(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        cm_obj = self.cm.load()
        self.assertEqual(cm_obj.sw_name2code, {})
        self.assertEqual(cm_obj.stock_sector, {})
        self.assertEqual(cm_obj.board_name2code, {})
        self.assertIsNone(cm_obj.hist)


class TestRelations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def setUp(self):
        self.cm_obj = _chainmap()

    def test_upstream_of_dedup_order(self):
        self.assertEqual(self.cm_obj.upstream_of("电力设备"),
                         ["有色金属", "基础化工"])
        self.assertEqual(self.cm_obj.upstream_of("汽车"), ["电力设备"])
        self.assertEqual(self.cm_obj.upstream_of("不存在"), [])

    def test_downstream_of(self):
        self.assertEqual(self.cm_obj.downstream_of("电力设备"), ["汽车"])
        self.assertEqual(self.cm_obj.downstream_of("有色金属"), ["电力设备"])
        self.assertEqual(self.cm_obj.downstream_of("不存在"), [])

    def test_chains_of(self):
        out = self.cm_obj.chains_of("电力设备")
        self.assertEqual(out, [
            {"chain": "新能源车", "layer": "midstream", "node": "动力电池"},
            {"chain": "光伏", "layer": "downstream", "node": "组件"},
        ])
        self.assertEqual(self.cm_obj.chains_of("食品饮料"), [])

    def test_module_level_convenience(self):
        with mock.patch.object(self.cm, "load", return_value=self.cm_obj):
            self.assertEqual(self.cm.upstream_of("汽车"), ["电力设备"])
            self.assertEqual(self.cm.downstream_of("有色金属"), ["电力设备"])
            self.assertEqual(self.cm.chains_of("有色金属"),
                             [{"chain": "新能源车", "layer": "upstream", "node": "锂矿"}])
            self.assertEqual(self.cm.chain_temperature("不存在", "2026-08-06"), {})


class TestChainTemperature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def test_three_layer_temperature_and_no_lookahead(self):
        base = _chainmap(hist=_hist())                       # 只到 08-06
        fut = _chainmap(hist=_hist(extra_future=True))       # 08-07 起 +50% 跳涨
        r1 = base.chain_temperature("新能源车", "2026-08-06")
        r2 = fut.chain_temperature("新能源车", "2026-08-06")
        self.assertEqual(r1, r2)  # 未来数据不得影响 ≤date 温度（防前视硬断言）
        self.assertEqual(r1["chain"], "新能源车")
        self.assertEqual(r1["as_of"], "2026-08-06")
        self.assertEqual(set(r1["layers"]), {"upstream", "midstream", "downstream"})
        for layer in ("upstream", "midstream", "downstream"):
            info = r1["layers"][layer]
            self.assertAlmostEqual(info["avg_ret"], 0.0101, places=3)
            self.assertGreater(info["temp"], 50.0)
            self.assertLessEqual(info["temp"], 100.0)
            self.assertIn("sectors", info)
        self.assertAlmostEqual(r1["layers"]["upstream"]["total_ret"], 104 / 99 - 1, places=4)

    def test_unknown_chain_returns_empty(self):
        self.assertEqual(_chainmap().chain_temperature("不存在", "2026-08-06"), {})

    def test_none_or_empty_hist(self):
        cm = _chainmap(hist=None)
        self.assertEqual(cm.chain_temperature("新能源车", "2026-08-06"), {})
        cm2 = _chainmap(hist=pd.DataFrame())
        self.assertEqual(cm2.chain_temperature("新能源车", "2026-08-06"), {})

    def test_date_before_all_data_returns_empty(self):
        cm = _chainmap()
        self.assertEqual(cm.chain_temperature("新能源车", "2026-07-31"), {})

    def test_invalid_date_falls_back_to_hist_max(self):
        cm = _chainmap()
        r = cm.chain_temperature("新能源车", "not-a-date")
        self.assertEqual(r["as_of"], "2026-08-06")

    def test_layer_without_codes_omitted(self):
        # 只映射 upstream（基础化工 801040），且 hist 只有该行业数据
        h = pd.DataFrame({
            "代码": ["801040"] * 6,
            "日期": pd.to_datetime([f"2026-08-{d:02d}" for d in range(1, 7)]),
            "收盘": [99, 100, 101, 102, 103, 104],
        })
        cm = _chainmap(hist=h)
        cm.chains = {"光伏": [("硅料", "upstream", "基础化工")]}
        cm.sw_name2code = {"基础化工": "801040"}
        cm.sw_code2name = {"801040": "基础化工"}
        r = cm.chain_temperature("光伏", "2026-08-06")
        self.assertEqual(set(r["layers"]), {"upstream"})
        self.assertEqual(r["layers"]["upstream"]["sectors"], ["基础化工"])


class TestPropagateFromConcept(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def setUp(self):
        self._tmp = self._make_tmp()
        self.cm_obj = _chainmap()
        self.member_file = self._tmp / "concept_member.parquet"
        mock.patch.object(self.cm, "CONCEPT_MEMBER", self.member_file).start()
        self.addCleanup(mock.patch.stopall)

    def _members(self, rows):
        pd.DataFrame(rows, columns=["concept", "code"]).to_parquet(self.member_file,
                                                                   index=False)

    def test_concept_maps_to_chain_layers(self):
        self._members([("BK1000", "000001"), ("BK1000", "000002"),
                       ("BK1000", "000003")])
        out = self.cm_obj.propagate_from_concept("固态电池", "2026-08-06")
        self.assertEqual(len(out), 2)
        by_layer = {e["layer"]: e for e in out}
        self.assertIn("upstream", by_layer)
        self.assertIn("downstream", by_layer)
        up = by_layer["upstream"]
        self.assertEqual(up["chain"], "新能源车")
        self.assertEqual(up["node"], ["锂矿"])
        self.assertEqual(up["sectors"], ["有色金属"])
        self.assertIsNotNone(up["up_rets"])
        self.assertIsNotNone(up["mid_rets"])   # 新能源车链含中游节点 → 有值
        self.assertIsNotNone(up["down_rets"])
        self.assertEqual(up["signal"], "升温（全链正向传导）")
        down = by_layer["downstream"]
        self.assertEqual(down["node"], ["整车"])
        self.assertEqual(down["sectors"], ["汽车"])

    def test_bk_code_passthrough(self):
        self._members([("BK1000", "000001")])
        out = self.cm_obj.propagate_from_concept("BK1000", "2026-08-06")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["sectors"], ["有色金属"])

    def test_empty_inputs_do_not_raise(self):
        self.assertEqual(self.cm_obj.propagate_from_concept("", "2026-08-06"), [])
        self.assertEqual(self.cm_obj.propagate_from_concept("不存在", "2026-08-06"), [])

    def test_missing_member_file_returns_empty(self):
        self.assertEqual(self.cm_obj.propagate_from_concept("固态电池", "2026-08-06"), [])

    def test_board_without_members_returns_empty(self):
        self._members([("BK9999", "000001")])
        self.assertEqual(self.cm_obj.propagate_from_concept("固态电池", "2026-08-06"), [])

    def test_members_without_sector_mapping_returns_empty(self):
        self._members([("BK1000", "600999")])  # 600999 不在 stock_sector
        self.assertEqual(self.cm_obj.propagate_from_concept("固态电池", "2026-08-06"), [])

    def test_resolve_board(self):
        self.assertEqual(self.cm_obj._resolve_board("BK1234"), "BK1234")
        self.assertEqual(self.cm_obj._resolve_board("固态电池"), "BK1000")
        self.assertIsNone(self.cm_obj._resolve_board("不存在的题材"))


class TestSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def test_signal_branches(self):
        self.assertEqual(self.cm._signal("upstream", None, 0, 0), "数据不足")
        self.assertEqual(self.cm._signal("upstream", 0.01, 0.01, 0.01),
                         "升温（全链正向传导）")
        self.assertEqual(self.cm._signal("upstream", 0.01, -0.01, -0.02),
                         "上游强·下游弱")
        self.assertEqual(self.cm._signal("upstream", -0.01, 0.01, 0.02),
                         "下游独立走强")
        self.assertEqual(self.cm._signal("midstream", 0.01, 0.02, None), "升温")
        self.assertEqual(self.cm._signal("midstream", 0.01, -0.01, None), "降温")
        self.assertEqual(self.cm._signal("midstream", 0.0, 0.0, 0.0), "走平")


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = _import()

    def _run(self, argv):
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["chain_map"] + argv), \
                redirect_stdout(buf):
            self.cm._cli()
        return buf.getvalue()

    def test_chains_listing(self):
        out = self._run(["--chains"])
        self.assertIn("新能源车", out)
        self.assertIn("光伏", out)

    def test_sector_query(self):
        cm_obj = _chainmap()
        with mock.patch.object(self.cm, "load", return_value=cm_obj):
            out = self._run(["--sector", "电力设备"])
        self.assertIn("上游: ['有色金属', '基础化工']", out)
        self.assertIn("下游: ['汽车']", out)
        self.assertIn("链归属", out)

    def test_temperature_query(self):
        cm_obj = _chainmap()
        with mock.patch.object(self.cm, "load", return_value=cm_obj):
            out = self._run(["--temp", "新能源车", "--date", "2026-08-06"])
        parsed = json.loads(out)
        self.assertEqual(parsed["as_of"], "2026-08-06")

    def test_concept_query(self):
        cm_obj = _chainmap()
        with mock.patch.object(self.cm, "load", return_value=cm_obj), \
                mock.patch.object(self.cm, "CONCEPT_MEMBER",
                                  Path(tempfile.mkdtemp()) / "nope.parquet"):
            out = self._run(["--concept", "固态电池", "--date", "2026-08-06"])
        self.assertEqual(json.loads(out), [])

    def test_help(self):
        out = self._run([])
        self.assertIn("usage", out)


if __name__ == "__main__":
    unittest.main()

