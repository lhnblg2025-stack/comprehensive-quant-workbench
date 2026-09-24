#!/usr/bin/env python3
"""test_decision_workbench_closure.py — 统一决策工作台生产闭环回归 (2026-08-29)

覆盖:
  1. 财务季度断点续传: _latest_period_in_file 识别最新报告期,_probe_latest_report_period
     失败安全返回 None,断点账本幂等。
  2. 执行账本: record_delivery/record_skill_run 幂等追加,ledger_summary 结构稳定。
  3. 旧决策入口隔离: /api/decision_chain 返回 deprecated + 转发统一 workbench,
     不再读 intraday_chain_*/after_close_extra_*/battle_map_* 旧产物。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SERVER = (ROOT / "quant_web" / "server.py").read_text(encoding="utf-8")


class TestFinancialCheckpoint(unittest.TestCase):
    def test_latest_period_in_file_parses_yyyymmdd_columns(self):
        from update_financial_quarterly import _latest_period_in_file
        with patch("update_financial_quarterly.pd.read_parquet") as rd:
            import pandas as pd
            rd.return_value = pd.DataFrame({"选项": ["ROE"], "20260331": [1.0], "20260630": [2.0]})
            self.assertEqual(_latest_period_in_file("000001"), "20260630")

    def test_latest_period_returns_none_on_missing_file(self):
        from update_financial_quarterly import _latest_period_in_file
        self.assertIsNone(_latest_period_in_file("999999"))

    def test_probe_returns_none_on_failure(self):
        from update_financial_quarterly import _probe_latest_report_period
        with patch("akshare.stock_financial_abstract", side_effect=Exception("boom")):
            self.assertIsNone(_probe_latest_report_period())

    def test_checkpoint_load_saves_roundtrip(self):
        from update_financial_quarterly import _load_checkpoint, _save_checkpoint, CHECKPOINT_FILE
        orig = _load_checkpoint()
        _save_checkpoint({"schema": "financial_update_checkpoint/v1",
                          "done": {"000001": {"ok": True}}, "latest_report_period": "20260630"})
        try:
            loaded = _load_checkpoint()
            self.assertEqual(loaded["latest_report_period"], "20260630")
            self.assertIn("000001", loaded["done"])
        finally:
            _save_checkpoint(orig)


class TestExecutionLedger(unittest.TestCase):
    def test_ledger_summary_shape(self):
        from execution_ledger import ledger_summary
        s = ledger_summary()
        self.assertEqual(s["schema"], "execution_ledger/v1")
        for key in ("delivery", "skill_runs"):
            self.assertIn(key, s)
            self.assertIn("count", s[key])
            self.assertIn("index", s[key])

    def test_record_delivery_and_skill_run_are_idempotent(self):
        from execution_ledger import record_delivery, record_skill_run, \
            _load_index, _save_index, DELIVERY_INDEX, SKILL_RUN_INDEX
        # 用唯一测试 id，避免污染真实账本；结束后移除测试条目。
        try:
            record_delivery(delivery_id="__test__:x", record={"status": "acknowledged"})
            record_delivery(delivery_id="__test__:x", record={"status": "failed"})
            d_idx = _load_index(DELIVERY_INDEX)
            self.assertEqual(d_idx["entries"]["__test__:x"]["status"], "failed")

            record_skill_run(execution_id="__test__:s", record={"skill": "turtle", "status": "ok"})
            record_skill_run(execution_id="__test__:s", record={"skill": "turtle", "status": "degraded"})
            s_idx = _load_index(SKILL_RUN_INDEX)
            self.assertEqual(s_idx["entries"]["__test__:s"]["status"], "degraded")
        finally:
            for path in (DELIVERY_INDEX, SKILL_RUN_INDEX):
                if path.exists():
                    data = _load_index(path)
                    data["entries"] = {k: v for k, v in data.get("entries", {}).items()
                                       if not str(k).startswith("__test__")}
                    data["count"] = len(data["entries"])
                    _save_index(path, data)


class TestOldDecisionIsolation(unittest.TestCase):
    def test_decision_chain_is_deprecated_shim(self):
        block = SERVER[SERVER.index("def _handle_decision_chain"):
                        SERVER.index("def _handle_factor_quality")]
        assert "deprecated" in block
        assert "/api/workbench" in block
        assert "content_source" in block and "unified_decision_snapshot" in block

    def test_decision_chain_legacy_fallback_is_date_bound(self):
        # 统一工作台是 canonical；旧产物仅允许按明确日期只读兼容，
        # 不允许无日期扫描或把旧文件作为新的生产入口。
        block = SERVER[SERVER.index("def _handle_decision_chain"):
                        SERVER.index("def _handle_factor_quality")]
        assert "deprecated" in block
        assert "legacy_artifact" in block
        assert "requested_date" in block
        assert "if date and" in block

    def test_execution_ledger_endpoint_registered(self):
        assert 'parsed.path == "/api/execution_ledger"' in SERVER
        assert '"/api/execution_ledger"' in (ROOT / "quant_web" / "api_catalog.py").read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
