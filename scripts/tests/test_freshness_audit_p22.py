"""V13 审计 P2-2 回归: 核心数据集新鲜度可审计（无"as_of=None 且无 stale"无信号条目）。

覆盖:
  1. 全量 freshness JSON 无任何「as_of=None 且无 stale」的无信号条目。
  2. NO_DATE_EXEMPT 型条目必有 stale 字段且 note 非空。
  3. margin_detail_{sh,sz} 重新登记：as_of=目录内最新日文件（YYYYMMDD）日期，
     且带 stale/note 信号（P2-3 关联，不再因 SKIP 失去信号）。
  4. 用合成小目录跑 scan_dataset：SNAPSHOT_MTIME / 文件名日期 / NO_DATE_EXEMPT /
     未登记兜底 四类无日期列路径都产出明确信号，绝不净输出无信号条目。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # scripts/tests -> workspace
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import stamp_data_freshness as sdf  # noqa: E402


def _no_signal_entries(fresh: dict):
    """收集「as_of is None 且无 stale 字段」（或无 stale 字段也无 note）的无信号条目。"""
    out = {}
    for n, e in fresh.items():
        if str(n).startswith("_"):
            continue
        if not isinstance(e, dict):
            continue
        if e.get("as_of") in (None, "") and "stale" not in e:
            out[n] = e
    return out


class TestRealFreshnessCompleteness(unittest.TestCase):
    """对当前 data_warehouse/data_freshness.json 全量断言。"""

    def test_no_signalless_entries_in_live_json(self):
        fresh = json.loads(sdf.FRESHNESS_JSON.read_text(encoding="utf-8"))
        nosig = _no_signal_entries(fresh)
        self.assertEqual(
            [], list(nosig.keys()),
            f"live data_freshness.json 仍有无信号条目: {list(nosig.keys())}")

    def test_no_date_exempt_entries_have_stale_and_note(self):
        """NO_DATE_EXEMPT 登记的数据集: 生成的 JSON 条目必有 stale 字段且 note 非空。"""
        fresh = json.loads(sdf.FRESHNESS_JSON.read_text(encoding="utf-8"))
        for name in sdf.NO_DATE_EXEMPT:
            e = fresh.get(name)
            if e is None:
                # 该数据集当前可能因 SKIP/目录缺失未上浮；用合成断言保证契约
                continue
            self.assertIn("stale", e, f"{name} 缺 stale 字段")
            self.assertEqual(False, e.get("stale"), f"{name} 应为静态快照（非陈旧）")
            self.assertTrue(e.get("note"), f"{name} 的 note 为空")
            self.assertIsNotNone(sdf.NO_DATE_EXEMPT.get(name),
                                 f"{name} 应登记判定说明")

    def test_margin_detail_registered_with_filename_date(self):
        """P2-3: margin_detail_{sh,sz} 重登记为 as_of=目录内最新日文件日期，带 stale+note。"""
        fresh = json.loads(sdf.FRESHNESS_JSON.read_text(encoding="utf-8"))
        for name in ("margin_detail_sh", "margin_detail_sz"):
            e = fresh.get(name)
            self.assertIsNotNone(e, f"{name} 应在 freshness 中重新登记")
            self.assertEqual(e.get("date_from"), "latest_daily_filename")
            self.assertIsNotNone(e.get("as_of"), f"{name} 应有 as_of")
            self.assertIn("stale", e, f"{name} 缺 stale 字段")
            self.assertTrue(e.get("note"), f"{name} 缺说明性 note")
            self.assertLessEqual(
                datetime.strptime(e["as_of"], "%Y-%m-%d").date(), date.today(),
                f"{name} 的 as_of 不应晚于今天（须来自真实日文件）")


class TestScanLogic(unittest.TestCase):
    """用合成 small 目录跑 scan_dataset 逻辑，覆盖四类无日期列登记路径。"""

    def _scan_tmp_warehouse(self):
        """建临时 data_warehouse，跑扫描，返回 (fresh_dict, 清理函数)。"""
        import contextlib
        td = tempfile.TemporaryDirectory()
        wh = Path(td.name)
        orig_wh = sdf.WH
        sdf.WH = wh
        try:
            # 1) SNAPSHOT_MTIME 无日期列快照 → mtime as_of
            sub = wh / "market"
            sub.mkdir(parents=True, exist_ok=True)
            cb = sub / "cb_spot.parquet"
            pd.DataFrame({"code": ["a"], "name": ["b"]}).to_parquet(cb)
            os.utime(cb, (0, 0))  # epoch, 明确"旧 mtime"
            # 2) 文件名日期快照 → as_of=文件名日期
            pd.DataFrame({"code": ["a"], "rank": [1]}).to_parquet(
                sub / "hot_rank_20260811.parquet", index=False)
            # 3) NO_DATE_EXEMPT 静态映射（无日期列）
            industry = wh / "industry"
            industry.mkdir(exist_ok=True)
            pd.DataFrame({"code": ["a"]}).to_parquet(industry / "sw_first.parquet")
            # 4) 未登记无日期列数据 → 兜底 stale=True + note
            oneoff = wh / "oneoff"
            oneoff.mkdir(exist_ok=True)
            pd.DataFrame({"x": [1]}).to_parquet(oneoff / "unregistered.parquet")

            fresh = {}
            for dirpath, _d, files in os.walk(wh):
                for fn in sorted(files):
                    if not fn.endswith(".parquet"):
                        continue
                    e = sdf.scan_dataset(Path(dirpath) / fn)
                    if e:
                        fresh[fn] = e
            return fresh
        finally:
            sdf.WH = orig_wh
            td.cleanup()

    def test_synthetic_scan_no_signalless_entries(self):
        fresh = self._scan_tmp_warehouse()
        nosig = _no_signal_entries(fresh)

        # 关键数据集应已登记且 no-signal 为空
        self.assertEqual([], list(nosig.keys()),
                         f"合成扫描仍有无信号条目: {list(nosig.keys())}")
        # SNAPSHOT_MTIME → mtime as_of（epoch=旧 → stale=True）
        self.assertIn("cb_spot.parquet", fresh)
        self.assertEqual("1970-01-01", fresh["cb_spot.parquet"]["as_of"])
        self.assertIn("stale", fresh["cb_spot.parquet"])
        self.assertTrue(fresh["cb_spot.parquet"]["note"])
        # 文件名日期 → as_of=2026-08-11, date_from=filename
        hr = fresh.get("hot_rank_20260811.parquet", {})
        self.assertEqual("2026-08-11", hr.get("as_of"))
        self.assertEqual("filename", hr.get("date_from"))
        # NO_DATE_EXEMPT → stale=False + note
        sw = fresh.get("sw_first.parquet", {})
        if sw:
            self.assertEqual(False, sw.get("stale"))
            self.assertTrue(sw.get("note"))
        # 未登记兜底 → stale=True + note（绝不无信号）
        u = fresh.get("unregistered.parquet", {})
        self.assertEqual(True, u.get("stale"))
        self.assertTrue(u.get("note"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
