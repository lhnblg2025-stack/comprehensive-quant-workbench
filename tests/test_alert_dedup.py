#!/usr/bin/env python3
"""test_alert_dedup.py — alert_dedup 网关单元测试。

覆盖: 新签名放行 / 同签名静默 / 到期重发 / 夜间静默 / danger 夜间放行 /
每小时硬顶 / normalize_price 价格剔除。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from alert_dedup import AlertDeduper, make_sig, normalize_price, reset_instance  # noqa: E402


class TestAlertDedup(unittest.TestCase):

    def setUp(self):
        os.environ.pop("QUANT_ALERT_SILENT", None)
        reset_instance()
        self.tmp = TemporaryDirectory()
        self.state_file = Path(self.tmp.name) / "alert_dedup_state.json"
        self.d = AlertDeduper(state_file=self.state_file, rate_cap=10)
        self.t0 = datetime(2026, 8, 14, 10, 0, 0)  # 白天 10:00

    def tearDown(self):
        self.tmp.cleanup()
        reset_instance()

    def test_new_sig_sends(self):
        self.assertTrue(self.d.should_send("ch", make_sig("a"), now=self.t0))
        # 不同签名 → 立即放行
        self.assertTrue(self.d.should_send("ch", make_sig("b"), now=self.t0))

    def test_same_sig_silenced_then_reminded(self):
        sig = make_sig("a")
        self.assertTrue(self.d.should_send("ch", sig, now=self.t0))
        # 1 小时后同签名 → 静默（默认 gap 2h）
        t1 = self.t0.replace(hour=11)
        self.assertFalse(self.d.should_send("ch", sig, now=t1))
        # 2 小时后同签名 → 重发一次提醒
        t2 = self.t0.replace(hour=12)
        self.assertTrue(self.d.should_send("ch", sig, now=t2))

    def test_night_silence_except_danger(self):
        t_night = self.t0.replace(hour=2)
        self.assertFalse(self.d.should_send("ch", make_sig("x"), level="warn",
                                            now=t_night))
        self.assertTrue(self.d.should_send("ch", make_sig("y"), level="danger",
                                           now=t_night))

    def test_hourly_rate_cap(self):
        n_sent = sum(
            1 for i in range(30)
            if self.d.should_send("ch", make_sig(f"sig{i}"),
                                  now=self.t0.replace(minute=i))
        )
        self.assertEqual(n_sent, 10)  # 硬顶 10 条/小时

    def test_global_silent_env(self):
        os.environ["QUANT_ALERT_SILENT"] = "1"
        self.assertFalse(self.d.should_send("ch", make_sig("a"), now=self.t0))

    def test_normalize_price(self):
        text = "002714|多因子极端×2|38.37 涨 2.5% 成交 12.3亿"
        norm = normalize_price(text)
        self.assertNotIn("38.37", norm)
        self.assertNotIn("2.5%", norm)
        self.assertIn("#", norm)
        # 归一后同信号不同价格 → 同签名
        a = normalize_price("002714|x|38.37")
        b = normalize_price("002714|x|38.47")
        self.assertEqual(make_sig(a), make_sig(b))

    def test_state_persisted(self):
        sig = make_sig("persist")
        self.assertTrue(self.d.should_send("ch", sig, now=self.t0))
        # 新实例读同一状态文件 → 同签名静默
        d2 = AlertDeduper(state_file=self.state_file)
        self.assertFalse(d2.should_send("ch", sig, now=self.t0.replace(hour=11)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
