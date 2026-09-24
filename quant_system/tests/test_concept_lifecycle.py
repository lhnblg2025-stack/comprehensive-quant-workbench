"""analysis_core/concept_lifecycle 概念生命周期知识图谱单元测试。

覆盖:
  - _cache_fresh: 缺文件/坏 meta/schema 不匹配/mtime 变化 → False；完整 → True
  - _Cache._build 冷路径: 面板矩阵/段检测/龙头/退潮深度/特征矩阵/落盘
  - _Cache._load_disk 热路径: 读盘还原全部中间表
  - _resample 下采样/补零
  - concept_archive: 无历史→首次活跃、多段档案（间隔分布/龙头更替/最新段 ongoing）
  - similar_episodes: <2 段→[]、相似度计算、退潮深度读取、缺失特征→[]
  - concept_report 防前视硬断言: 指定 date 只用 ≤date 的 theme_cycle 行，
    未来日期的概念/阶段绝不混入；当日无活跃概念降级；JSON 落盘

无网络: 全部本地 mock；路径常量 mock.patch.object 到 tmp；缓存模块级 _CACHE 隔离。
"""

from __future__ import annotations

import json
import os
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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.concept_lifecycle as cl
    return cl


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="concept_lifecycle_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def _patch_paths(self, tmp):
        cl = _import()
        gen = tmp / "generated"
        cache = gen / "cache"
        paths = {
            "CONCEPT_MEMBER": tmp / "concept_member.parquet",
            "CONCEPT_BOARD": tmp / "concept_board.parquet",
            "ZT_HISTORY": tmp / "zt_pool_history.parquet",
            "THEME_CYCLE": tmp / "theme_cycle.parquet",
            "GEN_DIR": gen,
            "CACHE_DIR": cache,
            "PANEL_CACHE": cache / "panel.parquet",
            "LONG_CACHE": cache / "long.parquet",
            "EPISODES_CACHE": cache / "episodes.parquet",
            "LEADERS_CACHE": cache / "leaders.parquet",
            "RETREAT_CACHE": cache / "retreat.parquet",
            "FEATURES_CACHE": cache / "features.parquet",
            "NAMES_CACHE": cache / "names.parquet",
            "META_CACHE": cache / "meta.json",
        }
        for name, p in paths.items():
            mock.patch.object(cl, name, p).start()
        self.addCleanup(mock.patch.stopall)
        return paths


def _write_sources(tmp):
    """10 交易日源数据: 概念 BK1000 成员 000001..000004。

    day0-2 全涨停(zt_cnt=4, max_board=4) → 段1
    day3-7 仅 000002 涨停(zt_cnt=1, 非活跃) → 间隔日
    day8-9 全涨停(zt_cnt=4, max_board=5) → 段2（间隔 = 8-2-1 = 5 交易日 → 新炒作）
    """
    dates = pd.date_range("2026-07-01", periods=10, freq="B")
    zt_rows = []
    for d_idx in (0, 1, 2, 8, 9):
        bc_ldr = 5 if d_idx >= 8 else 4
        for code, name, bc in (("000001", "龙头A", bc_ldr), ("000002", "股B", 1),
                               ("000003", "股C", 1), ("000004", "股D", 1)):
            zt_rows.append({"date": dates[d_idx], "code": code, "is_zt": True,
                            "board_count": bc, "name": name})
    for d_idx in (3, 4, 5, 6, 7):
        zt_rows.append({"date": dates[d_idx], "code": "000002", "is_zt": True,
                        "board_count": 1, "name": "股B"})
    zt = pd.DataFrame(zt_rows)
    zt = zt.drop_duplicates(["date", "code"], keep="last")
    zt.to_parquet(tmp / "zt_pool_history.parquet", index=False)

    members = pd.DataFrame({
        "concept": ["BK1000"] * 4,
        "code": ["000001", "000002", "000003", "000004"],
        "name": ["龙头A", "股B", "股C", "股D"],
    })
    members.to_parquet(tmp / "concept_member.parquet", index=False)

    board = pd.DataFrame({"board_code": ["BK1000"], "board_name": ["固态电池"]})
    board.to_parquet(tmp / "concept_board.parquet", index=False)

    theme = pd.DataFrame({
        "concept": ["BK1000"] * 3 + ["BK2000"],
        "date": [dates[2], dates[8], dates[9], dates[8]],
        "zt_cnt": [4, 4, 4, 9],
        "max_board": [4, 5, 5, 6],
        "stage": ["发酵", "发酵", "发酵", "高潮"],
        "board_name": ["固态电池"] * 4,
    })
    theme.to_parquet(tmp / "theme_cycle.parquet", index=False)


class TestCacheFresh(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _import()

    def test_missing_files_false(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        self.assertFalse(self.cl._cache_fresh())

    def test_corrupt_meta_false(self):
        tmp = self._make_tmp()
        paths = self._patch_paths(tmp)
        for k in ("PANEL_CACHE", "LONG_CACHE", "EPISODES_CACHE", "LEADERS_CACHE",
                  "RETREAT_CACHE", "FEATURES_CACHE", "NAMES_CACHE", "META_CACHE"):
            paths[k].parent.mkdir(parents=True, exist_ok=True)
            paths[k].write_bytes(b"x")
        paths["META_CACHE"].write_text("{broken", encoding="utf-8")
        self.assertFalse(self.cl._cache_fresh())

    def test_schema_mismatch_false(self):
        tmp = self._make_tmp()
        paths = self._patch_paths(tmp)
        for k in ("PANEL_CACHE", "LONG_CACHE", "EPISODES_CACHE", "LEADERS_CACHE",
                  "RETREAT_CACHE", "FEATURES_CACHE", "NAMES_CACHE"):
            paths[k].parent.mkdir(parents=True, exist_ok=True)
            paths[k].write_bytes(b"x")
        paths["META_CACHE"].write_text(json.dumps(
            {"schema_version": self.cl.SCHEMA_VERSION - 1}), encoding="utf-8")
        self.assertFalse(self.cl._cache_fresh())

    def test_mtime_mismatch_false_and_full_true(self):
        tmp = self._make_tmp()
        paths = self._patch_paths(tmp)
        # 先完整构建（写入 meta 与 mtime 匹配）
        _write_sources(tmp)
        cache = self.cl._Cache(use_disk=False)
        self.assertTrue(self.cl._cache_fresh())
        # 源文件 mtime 变化 → 不新鲜
        zt = tmp / "zt_pool_history.parquet"
        os.utime(zt, (zt.stat().st_atime + 100, zt.stat().st_mtime + 100))
        self.assertFalse(self.cl._cache_fresh())
        # 恢复后重新构建 → 新鲜
        cache2 = self.cl._Cache(use_disk=False)
        self.assertTrue(self.cl._cache_fresh())
        self.assertEqual(len(cache.episodes), 2)


class TestBuildAndLoad(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _import()

    def test_build_full_pipeline(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        _write_sources(tmp)
        cache = self.cl._Cache(use_disk=False)
        # 面板: 1 概念 × 10 日
        self.assertEqual(len(cache.panel), 10)
        self.assertEqual(cache.n_days, 10)
        self.assertEqual(cache.concepts, ["BK1000"])
        self.assertEqual(cache.zt_latest, "2026-07-14")
        # 段检测: 2 段
        self.assertEqual(len(cache.episodes), 2)
        ep = cache.episodes.sort_values("start_d").reset_index(drop=True)
        self.assertEqual(list(ep["dur"]), [3, 2])
        self.assertEqual(list(ep["max_board"]), [4, 5])
        # 龙头
        self.assertEqual(len(cache.leaders), 2)
        self.assertEqual(cache.leaders.iloc[0]["code"], "000001")
        # 退潮深度
        self.assertEqual(len(cache.retreat), 2)
        # 特征矩阵 [n_ep, SIM_FEATURES+2]
        self.assertIn("BK1000", cache.features)
        self.assertEqual(cache.features["BK1000"].shape, (2, self.cl.SIM_FEATURES + 2))
        # 落盘文件
        self.assertTrue((tmp / "generated" / "cache" / "panel.parquet").exists())
        self.assertTrue((tmp / "generated" / "cache" / "meta.json").exists())

    def test_load_disk_roundtrip(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        _write_sources(tmp)
        self.cl._Cache(use_disk=False)
        cache = self.cl._Cache(use_disk=True)
        self.assertEqual(cache.concepts, ["BK1000"])
        self.assertEqual(cache.n_days, 10)
        self.assertEqual(cache.name_map["000001"], "龙头A")
        self.assertEqual(cache.board_names["BK1000"], "固态电池")
        self.assertEqual(cache.member_names["BK1000"]["000002"], "股B")
        self.assertEqual(len(cache.episodes), 2)
        self.assertEqual(len(cache.leaders), 2)
        self.assertEqual(cache.zt_cnt_mat.shape, (1, 10))
        # 特征读取
        self.assertIn("BK1000", cache.features)
        # _get_seq 惰性解析
        seq = cache._get_seq()
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], [4, 4, 4])

    def test_resample(self):
        self.assertEqual(list(self.cl._Cache._resample(np.array([1.0, 2.0, 3.0]), n=3)), [1.0, 2.0, 3.0])
        out = self.cl._Cache._resample(np.array([1.0, 2.0, 3.0, 4.0]), n=2)
        self.assertEqual(list(out), [1.0, 4.0])
        out2 = self.cl._Cache._resample(np.array([1.0, 2.0]), n=8)
        self.assertEqual(list(out2[:2]), [1.0, 2.0])
        self.assertTrue(np.all(out2[2:] == 0))


def _manual_cache():
    """手工构造 _Cache（绕过 __init__），属性与 _build 输出同构。"""
    cl = _import()
    cache = cl._Cache.__new__(cl._Cache)
    dates = pd.date_range("2026-07-01", periods=10, freq="B")
    cache.dates = np.sort(pd.to_datetime(dates).to_numpy())
    cache.day_index = {d: i for i, d in enumerate(cache.dates)}
    cache.n_days = 10
    cache.zt_latest = "2026-07-14"
    cache.concepts = ["BK1000"]
    cache.board_names = {"BK1000": "固态电池"}
    cache.name_map = {"000001": "龙头A", "000002": "股B"}
    cache.member_names = {"BK1000": {"000001": "龙头A", "000002": "股B"}}
    ep = pd.DataFrame([
        {"concept": "BK1000", "ep_idx": 0, "start_d": 0, "end_d": 2, "dur": 3,
         "seq": json.dumps([4, 4, 4]), "max_board": 4},
        {"concept": "BK1000", "ep_idx": 1, "start_d": 7, "end_d": 9, "dur": 3,
         "seq": json.dumps([4, 4, 4]), "max_board": 5},
    ])
    ep["start_date"] = ep["start_d"].map(cache.dates.__getitem__)
    ep["end_date"] = ep["end_d"].map(cache.dates.__getitem__)
    cache.episodes = ep
    cache.leaders = pd.DataFrame([
        {"concept": "BK1000", "ep_idx": 0, "code": "000001", "name": "龙头A",
         "max_board": 4, "cnt": 4},
        {"concept": "BK1000", "ep_idx": 1, "code": "000002", "name": "股B",
         "max_board": 5, "cnt": 3},
    ])
    cache.retreat = pd.DataFrame([
        {"concept": "BK1000", "ep_idx": 0, "retreat_5": 0.8, "retreat_10": 0.4},
        {"concept": "BK1000", "ep_idx": 1, "retreat_5": np.nan, "retreat_10": np.nan},
    ])
    rng = np.random.default_rng(42)
    cache.features = {"BK1000": rng.normal(size=(2, cl.SIM_FEATURES + 2))}
    cache.panel = None
    cache.zt_cnt_mat = np.zeros((1, 10), dtype=int)
    cache.max_board_mat = np.zeros((1, 10), dtype=int)
    cache._seq_parsed = None
    return cache


class TestConceptArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _import()

    def test_no_history_first_active(self):
        cache = _manual_cache()
        cache.episodes = cache.episodes[cache.episodes["concept"] == "NOPE"]
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            r = self.cl.concept_archive("BK9999")
        self.assertEqual(r["note"], "首次活跃")
        self.assertEqual(r["name"], "")

    def test_archive_multiple_episodes(self):
        cache = _manual_cache()
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            r = self.cl.concept_archive("BK1000")
        self.assertEqual(r["episode_count"], 2)
        self.assertEqual(r["avg_duration"], 3.0)
        self.assertEqual(r["max_duration"], 3)
        self.assertEqual(r["latest_episode"]["start"], "2026-07-10")
        self.assertEqual(r["latest_episode"]["end"], "2026-07-14")
        self.assertTrue(r["latest_episode"]["ongoing"])
        # 龙头更替（000001 → 000002）
        self.assertEqual(len(r["leader_changes"]), 1)
        self.assertEqual(r["leader_changes"][0]["from"], "龙头A")
        self.assertEqual(r["leader_changes"][0]["to"], "股B")
        self.assertEqual(r["leader"]["code"], "000002")
        # 间隔分布: 段0 结束 day2, 段1 开始 day7 → 间隔 4 个交易日
        dist = {d["interval"]: d["count"] for d in r["interval_dist"]}
        self.assertEqual(dist["1-4"], 1)
        self.assertEqual(r["avg_interval"], 4.0)

    def test_interval_dist_helper(self):
        ep = pd.DataFrame({
            "start_d": [0, 7], "end_d": [2, 9], "start_date": pd.to_datetime(["2026-07-01", "2026-07-10"]),
        })
        dist, gaps = self.cl._interval_dist(ep)
        self.assertEqual(gaps, [4])
        self.assertEqual(len(dist), 4)
        self.assertEqual(self.cl._interval_dist(ep.iloc[:1]), ([], []))


class TestSimilarEpisodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _import()

    def test_too_few_episodes(self):
        cache = _manual_cache()
        cache.episodes = cache.episodes.iloc[:1]
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            self.assertEqual(self.cl.similar_episodes("BK1000"), [])

    def test_missing_features(self):
        cache = _manual_cache()
        cache.features = {}
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            self.assertEqual(self.cl.similar_episodes("BK1000"), [])

    def test_similarity_and_retreat(self):
        cache = _manual_cache()
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            out = self.cl.similar_episodes("BK1000", k=2)
        self.assertEqual(len(out), 1)  # 只有 1 个历史段
        row = out[0]
        self.assertIn("similarity", row)
        self.assertEqual(row["zt_seq"], [4, 4, 4])
        self.assertEqual(row["retreat_5d"], 0.8)
        self.assertEqual(row["retreat_10d"], 0.4)
        self.assertIn("start", row)


class TestConceptReport(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _import()

    def test_report_no_active_concepts(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        dates = pd.date_range("2026-07-01", periods=3, freq="B")
        pd.DataFrame({
            "concept": ["BK1000"], "date": [dates[0]], "zt_cnt": [1],
            "max_board": [2], "stage": ["冰点"], "board_name": ["固态电池"],
        }).to_parquet(tmp / "theme_cycle.parquet")
        cache = _manual_cache()
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            r = self.cl.concept_report("2026-07-01")
        self.assertEqual(r["note"], "当日无活跃概念")
        self.assertEqual(r["concepts"], [])

    def test_report_anti_lookahead(self):
        """防前视硬断言: 指定 date 时只允许 ≤date 的 theme_cycle 行；
        未来日期的概念/阶段绝不混入报告。"""
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        dates = pd.date_range("2026-07-01", periods=3, freq="B")
        pd.DataFrame({
            "concept": ["BK1000", "BK1000", "BK2000"],
            "date": [dates[0], dates[1], dates[1]],
            "zt_cnt": [4, 4, 9],
            "max_board": [4, 5, 6],
            "stage": ["发酵", "发酵", "高潮"],
            "board_name": ["固态电池", "固态电池", "未来概念"],
        }).to_parquet(tmp / "theme_cycle.parquet")
        cache = _manual_cache()
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            r = self.cl.concept_report("2026-07-01")
        self.assertEqual(r["date"], "2026-07-01")
        # 只用 date=07-01 的行: BK1000（发酵），BK2000（未来 07-02）不得出现
        self.assertEqual([e["concept"] for e in r["concepts"]], ["BK1000"])
        self.assertEqual(r["concepts"][0]["stage"], "发酵")
        # 07-02 的未来行（BK1000 发酵 / BK2000 高潮）均不得混入
        self.assertNotIn("BK2000", [e["concept"] for e in r["concepts"]])
        # again 标记: episode_count>=2 → True
        self.assertTrue(r["concepts"][0]["again"])
        self.assertEqual(r["active_concept_count"], 1)
        self.assertEqual(r["again_count"], 1)
        # JSON 落盘
        out = tmp / "generated" / "concept_lifecycle_2026-07-01.json"
        self.assertTrue(out.exists())
        self.assertEqual(r["_saved"], str(out))

    def test_report_default_latest_date(self):
        tmp = self._make_tmp()
        self._patch_paths(tmp)
        dates = pd.date_range("2026-07-01", periods=3, freq="B")
        pd.DataFrame({
            "concept": ["BK1000", "BK2000"],
            "date": [dates[0], dates[2]],
            "zt_cnt": [4, 9],
            "max_board": [4, 6],
            "stage": ["发酵", "高潮"],
            "board_name": ["固态电池", "未来概念"],
        }).to_parquet(tmp / "theme_cycle.parquet")
        cache = _manual_cache()
        with mock.patch.object(self.cl, "_get_cache", return_value=cache):
            r = self.cl.concept_report(None)
        self.assertEqual(r["date"], str(dates[2].date()))
        codes = [e["concept"] for e in r["concepts"]]
        self.assertNotIn("BK1000", codes)  # 07-01 行非最新日
        self.assertIn("BK2000", codes)
        self.assertEqual(r["concepts"][0]["stage"], "高潮")


if __name__ == "__main__":
    unittest.main()
