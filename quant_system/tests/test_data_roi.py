"""analysis_core/data_roi_scorer Data ROI 评分器单元测试。

覆盖:
  - build_mock_dataset: 10 源 mock 数据结构、目标/场景目标对齐
  - score_sources: 按 ROI 降序、字段完整性、DataFrame 输入、正负与 tier
  - 场景特种兵: 全局 ROI 低但 chain_auto 场景 ROI 高
  - 输出: json + md 双格式落盘、日期目录、无数据降级

无网络：全部 mock 数据，输出路径覆盖到 tmp。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.data_roi_scorer as roi
    return roi


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="data_roi_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


class TestMockData(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roi = _import()

    def test_build_mock_dataset_has_10_sources_and_target(self):
        data = self.roi.build_mock_dataset(seed=7)
        self.assertEqual(len(data["dataset"]), 10)
        self.assertIsInstance(data["target"], pd.Series)
        for name, series in data["dataset"].items():
            self.assertIsInstance(series, pd.Series)
            self.assertEqual(len(series), len(data["target"]))

    def test_build_mock_dataset_scenario_targets_aligned(self):
        data = self.roi.build_mock_dataset(seed=7)
        self.assertIn("chain_auto", data["scenario_targets"])
        scenario = data["scenario_targets"]["chain_auto"]
        self.assertEqual(len(scenario), len(data["target"]))
        self.assertEqual(list(scenario.index), list(data["target"].index))

    def test_build_mock_dataset_is_reproducible(self):
        first = self.roi.build_mock_dataset(seed=7)
        second = self.roi.build_mock_dataset(seed=7)
        self.assertTrue(first["target"].equals(second["target"]))


class TestScoring(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roi = _import()

    def test_score_sources_sorted_by_roi_desc(self):
        roi = self.roi
        data = roi.build_mock_dataset(seed=7)
        result = roi.score_sources(
            data["dataset"], data["target"],
            source_scopes=data.get("source_scopes"),
            scenario_targets=data.get("scenario_targets"),
            date="2026-08-13",
        )
        scores = [row["roi"] for row in result["sources"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result["date"], "2026-08-13")

    def test_score_sources_required_fields(self):
        roi = self.roi
        data = roi.build_mock_dataset(seed=7)
        result = roi.score_sources(data["dataset"], data["target"], date="2026-08-13")
        self.assertIn("sources", result)
        self.assertGreaterEqual(len(result["sources"]), 1)
        row = result["sources"][0]
        for key in ("source", "roi", "sign", "tier", "scopes", "scenario_roi",
                    "scenario_tiers", "special_forces", "suggestion"):
            self.assertIn(key, row)
        self.assertIn(row["sign"], {"positive", "negative", "neutral"})
        self.assertIn(row["tier"], {"hot", "warm", "cold"})
        self.assertIn("global", row["scopes"])

    def test_score_sources_accepts_dataframe(self):
        roi = self.roi
        data = roi.build_mock_dataset(seed=7)
        df = pd.DataFrame(data["dataset"])
        result = roi.score_sources(df, data["target"], date="2026-08-13")
        self.assertEqual(len(result["sources"]), df.shape[1])

    def test_scenario_special_forces_recognized(self):
        roi = self.roi
        data = roi.build_mock_dataset(seed=7)
        result = roi.score_sources(
            data["dataset"], data["target"],
            source_scopes=data.get("source_scopes"),
            scenario_targets=data.get("scenario_targets"),
            date="2026-08-13",
        )
        rows = {row["source"]: row for row in result["sources"]}
        special = data["source_scopes"]["source_09"][-1] if data["source_scopes"] else None
        self.assertIsNotNone(special)
        source = rows["source_09"]
        self.assertIn(special, source["special_forces"])
        self.assertIn("场景特种兵", source["suggestion"])

    def test_tier_counts_sum_to_sources(self):
        roi = self.roi
        data = roi.build_mock_dataset(seed=7)
        result = roi.score_sources(data["dataset"], data["target"], date="2026-08-13")
        counts = result["summary"]["tier_counts"]
        self.assertEqual(sum(counts.values()), len(result["sources"]))


class TestOutput(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roi = _import()

    def test_write_report_creates_json_and_md(self):
        roi = self.roi
        tmp = self._make_tmp()
        data = roi.build_mock_dataset(seed=7)
        result = roi.score_sources(data["dataset"], data["target"], date="2026-08-13")
        out = roi.write_report(result, output_dir=tmp)
        self.assertTrue((out / "data_roi_20260813.json").exists())
        self.assertTrue((out / "data_roi_20260813.md").exists())
        payload = json.loads((out / "data_roi_20260813.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["date"], "2026-08-13")
        self.assertEqual(len(payload["sources"]), len(result["sources"]))

    def test_no_data_degrades_gracefully(self):
        roi = self.roi
        tmp = self._make_tmp()
        result = roi.score_sources({}, pd.Series(dtype=float), date="2026-08-13")
        self.assertTrue(result["summary"].get("degraded"))
        self.assertEqual(result["summary"]["n_sources"], 0)
        self.assertEqual(result["sources"], [])
        out = roi.write_report(result, output_dir=tmp)
        self.assertTrue((out / "data_roi_20260813.md").exists())
        text = (out / "data_roi_20260813.md").read_text(encoding="utf-8")
        self.assertIn("无可用数据", text)

    def test_run_writes_to_date_subdir(self):
        roi = self.roi
        tmp = self._make_tmp()
        data = roi.build_mock_dataset(seed=7)
        result = roi.run(
            date="2026-08-13",
            dataset=data["dataset"],
            target=data["target"],
            output_dir=tmp,
        )
        out = tmp / "20260813" / "data_roi_20260813.json"
        self.assertTrue(out.exists())
        self.assertEqual(result["summary"]["n_sources"], len(data["dataset"]))


if __name__ == "__main__":
    unittest.main()
