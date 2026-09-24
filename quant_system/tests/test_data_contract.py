"""analysis_core/data_contract 单位统一层单元测试。

覆盖:
  - to_decimal / to_pct 双向正确（50% ↔ 0.5）
  - 边界 0 / 100 / -100
  - format_pct 展示层格式化
  - 单位审计：构造百分数列 → 标记疑似百分数
  - 审计输出结构（JSON/MD 落盘）
  - 无数据降级

无网络：全部 mock parquet，输出路径覆盖到 tmp。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import quant_system.analysis_core.data_contract as dc
    return dc


class _TmpMixin:
    def _make_tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="data_contract_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


class TestConverters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc = _import()

    def test_to_decimal_50_to_05(self):
        self.assertEqual(self.dc.to_decimal(50), 0.5)

    def test_to_pct_05_to_50(self):
        self.assertEqual(self.dc.to_pct(0.5), 50.0)

    def test_roundtrip_50_percent(self):
        self.assertAlmostEqual(self.dc.to_pct(self.dc.to_decimal(50)), 50.0)

    def test_boundaries_zero_100_negative_100(self):
        dc = self.dc
        self.assertEqual(dc.to_decimal(0), 0.0)
        self.assertEqual(dc.to_decimal(100), 1.0)
        self.assertEqual(dc.to_decimal(-100), -1.0)
        self.assertEqual(dc.to_pct(0.0), 0.0)
        self.assertEqual(dc.to_pct(1.0), 100.0)
        self.assertEqual(dc.to_pct(-1.0), -100.0)

    def test_format_pct(self):
        self.assertEqual(self.dc.format_pct(0.5), "50.00%")
        self.assertEqual(self.dc.format_pct(-0.015, decimals=2), "-1.50%")

    def test_invalid_input_returns_none(self):
        self.assertIsNone(self.dc.to_decimal(None))
        self.assertIsNone(self.dc.to_pct("not-a-number"))


class TestFormatPctValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc = _import()

    def test_pct_unit_passthrough(self):
        self.assertEqual(self.dc.format_pct_value(30.0, unit="pct"), "30.00%")
        self.assertEqual(self.dc.format_pct_value(1.23, unit="pct"), "1.23%")

    def test_decimal_unit_multiplies_by_100(self):
        self.assertEqual(self.dc.format_pct_value(0.5, unit="decimal"), "50.00%")
        self.assertEqual(self.dc.format_pct_value(-0.015, unit="decimal"), "-1.50%")

    def test_heuristic_fallback(self):
        self.assertEqual(self.dc.format_pct_value(0.5), "50.00%")
        self.assertEqual(self.dc.format_pct_value(30.0), "30.00%")

    def test_digits(self):
        self.assertEqual(self.dc.format_pct_value(30.0, digits=0, unit="pct"), "30%")
        self.assertEqual(self.dc.format_pct_value(0.5, digits=1, unit="decimal"), "50.0%")

    def test_invalid_values_return_na(self):
        self.assertEqual(self.dc.format_pct_value(None), "n/a")
        self.assertEqual(self.dc.format_pct_value(float("nan")), "n/a")
        self.assertEqual(self.dc.format_pct_value(float("inf")), "n/a")

    def test_zero_and_boundaries(self):
        self.assertEqual(self.dc.format_pct_value(0.0, unit="pct"), "0.00%")
        self.assertEqual(self.dc.format_pct_value(0.0, unit="decimal"), "0.00%")
        self.assertEqual(self.dc.format_pct_value(100.0, unit="pct"), "100.00%")
        self.assertEqual(self.dc.format_pct_value(-100.0, unit="pct"), "-100.00%")
        self.assertEqual(self.dc.format_pct_value(1.0, unit="decimal"), "100.00%")
        self.assertEqual(self.dc.format_pct_value(-1.0, unit="decimal"), "-100.00%")

    def test_display_modules_use_contract_formatter(self):
        from quant_system.analysis_core import rs_strength, watch_card

        self.assertEqual(watch_card._fmt_pct(2.34), "2.34%")
        self.assertEqual(watch_card._fmt_pct(float("nan")), "n/a")
        self.assertEqual(rs_strength.fmt_slope(2.345), "2.345%")


class TestUnitAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc = _import()

    def test_percent_column_flagged(self):
        result = self.dc.audit_column(
            pd.Series([0.01, 1.5, 2.0, -2.5]),
            column="pct_chg",
            dataset="mock/market.parquet",
        )
        self.assertEqual(result["suspected_unit"], "percent")
        self.assertEqual(result["min"], -2.5)
        self.assertEqual(result["max"], 2.0)
        self.assertIn("to_decimal", result["suggestion"])

    def test_decimal_column_not_flagged(self):
        result = self.dc.audit_column(
            pd.Series([0.01, -0.02, 0.005]),
            column="pct_chg",
            dataset="mock/market.parquet",
        )
        self.assertEqual(result["suspected_unit"], "decimal")

    def test_empty_column_skipped(self):
        result = self.dc.audit_column(pd.Series([], dtype=float), "pct_chg", "mock/x.parquet")
        self.assertEqual(result["suspected_unit"], "empty")
        self.assertEqual(result["value_count"], 0)

    def test_chinese_column_keyword(self):
        self.assertTrue(self.dc.is_contract_column("涨跌幅"))
        self.assertTrue(self.dc.is_contract_column("收益率"))

    def test_infer_column_unit_uses_mode(self):
        result = self.dc.infer_column_unit(
            pd.Series([0.01, 0.02, 0.03, 2.5]),
        )
        self.assertEqual(result["unit"], "decimal")
        self.assertEqual(result["decimal_votes"], 3)
        self.assertEqual(result["pct_votes"], 1)

    def test_audit_column_exposes_unit_and_votes(self):
        result = self.dc.audit_column(
            pd.Series([0.01, 0.02, 2.5, -1.8]),
            column="pct_chg",
            dataset="mock/mixed.parquet",
        )
        self.assertEqual(result["unit"], "pct")
        self.assertEqual(result["suspected_unit"], "percent")
        self.assertEqual(result["decimal_votes"], 2)
        self.assertEqual(result["pct_votes"], 2)


class TestScanAndOutput(_TmpMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc = _import()

    def test_scan_flags_mixed_parquet(self):
        dc = self.dc
        tmp = self._make_tmp()
        path = tmp / "mock_returns.parquet"
        pd.DataFrame({
            "pct_chg": [0.01, 0.02, 2.5, -1.8],
            "close": [10.0, 11.0, 12.0, 13.0],
        }).to_parquet(path, index=False)

        result = dc.scan_warehouse(paths=[path], include_kline_sample=False, date="2026-08-13")
        self.assertEqual(result["status"], "mixed_or_suspect")
        self.assertEqual(result["summary"]["scanned_columns"], 1)
        self.assertEqual(result["summary"]["suspicious_columns"], 1)
        record = result["datasets"][0]["columns"][0]
        self.assertEqual(record["column"], "pct_chg")
        self.assertEqual(record["suspected_unit"], "percent")

    def test_scan_output_writes_json_and_md(self):
        dc = self.dc
        tmp = self._make_tmp()
        path = tmp / "mock_returns.parquet"
        pd.DataFrame({
            "pct_chg": [0.01, 0.02],
            "price": [10.0, 11.0],
        }).to_parquet(path, index=False)
        result = dc.scan_warehouse(paths=[path], include_kline_sample=False, date="2026-08-13")
        output = dc.write_report(result, output_dir=tmp / "out")

        json_data = json.loads(Path(output["json"]).read_text(encoding="utf-8"))
        self.assertEqual(json_data["date"], "2026-08-13")
        self.assertEqual(json_data["summary"]["scanned_columns"], 1)
        md_text = Path(output["md"]).read_text(encoding="utf-8")
        self.assertIn("Data Contract Audit 2026-08-13", md_text)
        self.assertIn("pct_chg", md_text)

    def test_no_data_degrades(self):
        dc = self.dc
        tmp = self._make_tmp()
        result = dc.scan_warehouse(paths=[], include_kline_sample=False, date="2026-08-13")
        self.assertEqual(result["status"], "no_data")
        self.assertEqual(result["summary"]["scanned_files"], 0)
        self.assertEqual(result["summary"]["scanned_columns"], 0)


if __name__ == "__main__":
    unittest.main()
