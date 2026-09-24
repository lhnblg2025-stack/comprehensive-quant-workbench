#!/usr/bin/env python3
"""test_after_close_extra.py — 盘后增强测试 (2026-08-14)

覆盖:
  1. lhb_short_term 返回结构(游资营业部/个股龙虎榜或 error)
  2. value_picks 低估池: 全部满足 PE/PB 阈值且打分排序
  3. render_extra_md 含两章标题(龙虎榜资金/中长线低估池)
  4. build_extra 落盘 json+md 且可被 battle_map 引用
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core import after_close_extra as ace  # noqa: E402


class TestLhbShortTerm(unittest.TestCase):
    def test_structure(self):
        out = ace.lhb_short_term("2026-08-13")
        self.assertIn("broker_top", out)
        self.assertIn("stock_top", out)
        # 有数据或无数据都必须优雅(不抛异常)
        if out["stock_top"]:
            s = out["stock_top"][0]
            for k in ("code", "name", "net", "pct", "reason"):
                self.assertIn(k, s)


class TestValuePicks(unittest.TestCase):
    def test_picks_meet_thresholds(self):
        picks = ace.value_picks("2026-08-13", top_n=5)
        for p in picks:
            self.assertLess(p["pe"], ace.VALUE_PE_MAX + 0.01)
            self.assertLess(p["pb"], ace.VALUE_PB_MAX + 0.01)
            self.assertGreater(p["score"], 0)
            self.assertIn("dist52w", p)
        # 打分降序
        scores = [p["score"] for p in picks]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hist_value_at_timestable(self):
        # V12.3 审计 P1-3: 历史估值/价格快照必须是 time-consistent（≤date）
        # 任选 data_warehouse 实际有 valuation+kline 的标的
        import glob
        vfiles = glob.glob(str(ace.VALUATION_DIR / "*.parquet"))[:3]
        for vf in vfiles:
            sym = Path(vf).stem
            h = ace._hist_value_at(sym, "2020-06-15")
            # K线早于股票上市/未覆盖时返回 None（合法降级）; 有值则校验字段
            if h is not None:
                self.assertGreater(h["price"], 0)
                # PE 可为负(亏损股), 但必须是非 NaN 数值——原始快照如实返回,
                # value_picks 层的 pe<=0 过滤负责剔除, 此处不误判
                self.assertFalse(h["pe"] != h["pe"])  # 非 NaN
                # 52周低不得大于当日价（位置约束成立）
                if h["low52"] and h["price"]:
                    self.assertLessEqual(round(h["low52"], 2),
                                         round(h["price"], 2) + 1e-6)


class TestRender(unittest.TestCase):
    def test_md_has_two_sections(self):
        extra = {"date": "2026-08-13", "lhb": {"broker_top": [], "stock_top": [],
                                               "date": "2026-08-13"},
                 "value_picks": [{"code": "600015", "name": "华夏银行", "pe": 3.9,
                                  "pb": 0.33, "roe": 19.0, "growth": -1.0,
                                  "dist52w": 10.0, "score": 76.1}]}
        md = ace.render_extra_md(extra)
        self.assertIn("龙虎榜资金", md)
        self.assertIn("中长线低估池", md)
        self.assertIn("华夏银行", md)


class TestBuildExtra(unittest.TestCase):
    def test_build_writes_files(self):
        extra = ace.build_extra("2026-08-13")
        j = ROOT / "generated" / "after_close_extra_2026-08-13.json"
        m = ROOT / "generated" / "after_close_extra_2026-08-13.md"
        self.assertTrue(j.exists())
        self.assertTrue(m.exists())
        d = json.loads(j.read_text(encoding="utf-8"))
        self.assertIn("lhb", d)
        self.assertIn("value_picks", d)
        self.assertIn("date", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPushReport(unittest.TestCase):
    def test_build_push_text_has_all_sections(self):
        from scripts import push_after_close_report as pcr  # noqa: PLC0415
        sys.path.insert(0, str(ROOT))
        text = pcr.build_push_text("2026-08-26")
        self.assertIn("次日交易决策", text)
        self.assertIn("资金与结构", text)
        self.assertIn("数据缺失", text)

    def test_build_push_text_empty_graceful(self):
        from scripts import push_after_close_report as pcr
        sys.path.insert(0, str(ROOT))
        with self.assertRaises(ValueError):
            pcr.build_push_text("2099-01-01")  # 无统一快照必须阻断推送
