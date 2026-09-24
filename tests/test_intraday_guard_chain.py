#!/usr/bin/env python3
"""test_intraday_guard_chain.py — 盘中决策链增强测试 (2026-08-14)

覆盖:
  1. _watch_symbols 只返回自选+持仓(≤50), 不误用全市场409池(超时根因回归)
  2. _market_bias 返回 tone 字段(防御/谨慎/进攻三态)
  3. build_decision_chain 落盘 json 且 rows 含 dominant/summary
  4. opportunity_angles.evaluate_angles 信号归类正确(价值/短线/动量/防御)
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system" / "analysis_core"))

from quant_system.analysis_core.intraday_guard import (  # noqa: E402
    _format_chain,
    _market_bias,
    _watch_symbols,
)
from quant_system.opportunity_angles import evaluate_angles  # noqa: E402


class TestWatchScope(unittest.TestCase):
    def test_watch_symbols_is_narrow_for_risk_checks(self):
        """持仓/自选窄池仅用于逐只风控，决策链另走全市场快照。"""
        syms = _watch_symbols()
        self.assertLessEqual(len(syms), 50, "逐只实时价风控池超过50只")
        self.assertEqual(len(syms), len(set(syms)))
        self.assertNotEqual(syms, ["002714"], "空配置不得注入固定单票冒充扫描范围")


class TestWatchScopeEmpty(unittest.TestCase):
    def test_empty_watchlist_does_not_inject_fixed_stock(self):
        with patch('quant_system.analysis_core.intraday_guard._holdings', return_value=[]):
            with patch('pathlib.Path.exists', return_value=False):
                self.assertEqual(_watch_symbols(), [])


class TestMarketBias(unittest.TestCase):
    def test_bias_returns_tone(self):
        b = _market_bias()
        self.assertIn("tone", b)
        self.assertIn(b["tone"], ("防御", "谨慎", "进攻", "未知"))
        self.assertIn("breadth", b)


class TestChainFormat(unittest.TestCase):
    def test_format_chain_includes_tone_and_angles(self):
        chain = {
            "time": "2026-08-14 12:00:00",
            "bias": {"tone": "防御", "index_pct": -0.8, "advance": 1000,
                     "decline": 4000, "breadth": 0.2},
            "n_scanned": 10,
            "rows": [{"symbol": "600036", "name": "招商银行", "price": 38.0,
                      "change_pct": -0.3, "dominant": "value",
                      "dominant_label": "中长线价值",
                      "summary": "侧重中长线价值 - PE=6.5"}],
        }
        text = _format_chain(chain)
        self.assertIn("防御", text)
        self.assertIn("中长线价值", text)
        self.assertIn("600036", text)
        self.assertIn("大盘基调", text)


class TestAngleEvaluation(unittest.TestCase):
    def test_value_signals_classify(self):
        """低PE/PB/ROE 信号 → 中长线价值视角占主导。"""
        res = evaluate_angles(
            {"low_pe": 1.0, "low_pb": 1.0, "fundamental_roe": 0.5},
            {"low_pe": "PE=6.5", "low_pb": "PB=0.8", "fundamental_roe": "ROE=15%"},
            extra={"pe_ttm": 6.5, "pb": 0.8, "roe": 15.0, "price": 38.0, "ma60": 40.0},
        )
        self.assertEqual(res["dominant_angle"], "value")
        self.assertIn("PE=6.5", res["summary"])

    def test_trend_signals_classify(self):
        """上升趋势+均线支撑 → 短线趋势视角。"""
        res = evaluate_angles(
            {"uptrend": 0.8, "ma60_support": 1.0},
            {"uptrend": "上升趋势", "ma60_support": "MA60=¥19.72"},
            extra={"price": 20.0, "ma60": 19.72},
        )
        self.assertEqual(res["dominant_angle"], "short")

    def test_defense_high_risk(self):
        """跌停/下跌趋势 → 防御视角警示。"""
        res = evaluate_angles(
            {"limit_down": 1.0, "downtrend": -1.0, "today_drop": 1.0},
            {"limit_down": "跌停", "downtrend": "下降趋势", "today_drop": "今日收跌-5%"},
            extra={"price": 10.0, "ma60": 12.0},
        )
        self.assertIn(res["angles"]["defense"]["judge"], ("警惕", "中性", "安全"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAngleRegression(unittest.TestCase):
    """逐行审计回归: 五视角评估权重符号 bug。"""

    def test_downtrend_penalizes_short(self):
        """下降趋势不应给短线视角加分(原 abs() 双符号 bug)。"""
        from quant_system.opportunity_angles import evaluate_angles
        res = evaluate_angles({'downtrend': -1.0, 'today_drop': 1.0},
                              {'downtrend': '下降趋势', 'today_drop': '今日收跌'},
                              extra={'price': 10, 'ma60': 12})
        self.assertIn(res['angles']['short']['judge'], ('看空', '中性'))
        self.assertLess(res['angles']['short']['conf'], 0.5)

    def test_uptrend_boosts_short(self):
        from quant_system.opportunity_angles import evaluate_angles
        res = evaluate_angles({'uptrend': 0.8, 'ma60_support': 1.0},
                              {'uptrend': '上升趋势', 'ma60_support': 'MA60支撑'},
                              extra={'price': 20, 'ma60': 19.5})
        self.assertEqual(res['angles']['short']['judge'], '看多')
        self.assertEqual(res['dominant_angle'], 'short')
