"""analysis_core 体系6 风险纪律系统单元测试（risk_system）。

覆盖:
  - 个股止损纪律: 止损=close-1.5×ATR、距止损距离、破位→违规持仓
  - 仓位建议: 2%规则反推、单票20%上限、止损<入场退化
  - 回撤检查: 近20日池回撤、>6%触发月度纪律
  - 盈亏比: 目标位(密集区上沿/前高) vs 止损、<1.5→盈亏比不足
  - 沉没成本: 深亏+无反转→陷阱; 深亏+反转→按信号管理
  - 综合聚合: 高/中/低 + 离场/减仓/观望/持有
  - RiskSystem.detect/report/view 集成（临时K线 + mock RAG/情绪）
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _import(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import importlib
    return importlib.import_module(f"quant_system.analysis_core.{name}")


class _TmpDirMixin:
    def _make_tmp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="risk_system_"))
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        return self._tmp


def _kline(n=60, base=100.0, drift=0.0, vol=1.0):
    """合成日K：base 起步，每期 +drift，high/low 在 close±1 内。"""
    idx = pd.bdate_range("2026-04-01", periods=n)
    closes = base + drift * np.arange(n, dtype=float)
    df = pd.DataFrame({
        "date": idx, "open": closes - 0.2, "high": closes + 1.0,
        "low": closes - 1.0, "close": closes, "volume": vol * 1e6,
    })
    return df


class TestStopDiscipline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def test_stop_is_close_minus_1p5_atr(self):
        df = _kline(base=100.0, drift=0.1)
        close = float(df["close"].iloc[-1])
        atr = float(self.rs.atr14(df))
        res = self.rs._check_stop("000001", df, close, atr)
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["stop"], close - 1.5 * atr, places=6)
        self.assertGreater(res["dist_pct"], 0)
        self.assertFalse(res["broken"])
        self.assertIsNone(res["warning"])

    def test_broken_stop_flagged_violation(self):
        # 盘中最低价跌破止损位（构造暴跌末日）→ 违规持仓（与 watch_card 盘中触支撑同口径）
        df = _kline(base=100.0, drift=0.0, n=40)
        close = float(df["close"].iloc[-1])
        atr = float(self.rs.atr14(df))
        df.loc[df.index[-1], "close"] = close - 1.0 * atr          # 收盘价下移
        close_new = float(df["close"].iloc[-1])
        stop_new = close_new - 1.5 * atr                           # 新止损位
        df.loc[df.index[-1], "low"] = stop_new - 0.5 * atr         # 盘中破止损
        low_new = float(df["low"].iloc[-1])
        res = self.rs._check_stop("000001", df, close_new, atr)
        self.assertTrue(res["low_broken"])
        self.assertTrue(res["broken"])
        w = res["warning"]
        self.assertIsNotNone(w)
        self.assertEqual(w["type"], "违规持仓")
        self.assertEqual(w["level"], "crit")
        self.assertIn(f"{low_new:.2f}", w["msg"])

    def test_atr_missing_marks_unavailable(self):
        df = _kline(n=10)  # 样本不足 ATR14
        res = self.rs._check_stop("000001", df, 100.0, np.nan)
        self.assertFalse(res["ok"])
        self.assertIn("ATR不足", res["note"])


class TestPositionAdvice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def test_two_percent_rule_math(self):
        # 资金100万，入场100/止损90：每股风险10 → 建议金额=2万/10×100=20万(20%)
        res = self.rs._position_advice("000001", 100.0, 100.0, 90.0, 1_000_000, 50)
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["notional"], 200_000.0, places=2)
        self.assertAlmostEqual(res["pct"], 0.20, places=6)
        self.assertEqual(res["shares"], 2000)

    def test_single_stock_cap_20pct(self):
        # 止损很窄 → 2%规则反推仓位超20% → 封顶20%
        res = self.rs._position_advice("000001", 100.0, 100.0, 95.0, 1_000_000, 50)
        self.assertTrue(res["ok"])
        self.assertEqual(res["notional"], 200_000.0)
        self.assertEqual(res["pct"], 0.20)
        self.assertEqual(res["shares"], 2000)
        self.assertTrue(res["capped"])

    def test_degenerate_stop_returns_unavailable(self):
        res = self.rs._position_advice("000001", 100.0, 100.0, 100.0, 1_000_000, 50)
        self.assertFalse(res["ok"])
        res2 = self.rs._position_advice("000001", 100.0, 100.0, None, 1_000_000, 50)
        self.assertFalse(res2["ok"])

    def test_temperature_half_position(self):
        adv = self.rs._position_advice("000001", 100.0, 100.0, 90.0, 1_000_000, 50)
        total = self.rs._total_position_advice([adv], 1_000_000, 30)  # 温度<40 → 半仓
        self.assertTrue(total["half_position"])
        self.assertAlmostEqual(total["total_pct"], 0.10, places=6)  # 20% → 半仓 10%
        total2 = self.rs._total_position_advice([adv], 1_000_000, 45)
        self.assertFalse(total2["half_position"])
        self.assertAlmostEqual(total2["total_pct"], 0.20, places=6)


class TestDrawdown(_TmpDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def _write_kline(self, tmp: Path, code: str, df: pd.DataFrame):
        df.to_parquet(tmp / f"{code}.parquet", index=False)

    def test_pool_drawdown_under_threshold(self):
        tmp = self._make_tmp()
        for c in ("000001", "000002"):
            self._write_kline(tmp, c, _kline(base=100.0, drift=0.1))
        with mock.patch.object(self.rs, "KLINE_DIR", tmp):
            res = self.rs._pool_drawdown(["000001", "000002"],
                                         pd.Timestamp("2026-08-11"))
        self.assertTrue(res["ok"])
        self.assertFalse(res["triggered"])
        self.assertGreater(res["pool_dd"], -0.06)

    def test_drawdown_over_6pct_triggers(self):
        tmp = self._make_tmp()
        self._write_kline(tmp, "000001", _kline(base=100.0, drift=0.0))
        df = _kline(base=100.0, drift=0.0, n=40)
        # 末日暴跌 12% → 20日窗口回撤 > 6%
        df.loc[df.index[-1], "close"] = df["close"].iloc[-2] * 0.88
        df.loc[df.index[-1], "low"] = df["close"].iloc[-1] * 0.99
        self._write_kline(tmp, "000001", df)  # 覆盖，池=单股
        with mock.patch.object(self.rs, "KLINE_DIR", tmp):
            res = self.rs._pool_drawdown(["000001"], pd.Timestamp("2026-08-11"))
        self.assertTrue(res["triggered"])
        self.assertLess(res["pool_dd"], -0.06)
        w = res["warning"]
        self.assertEqual(w["type"], "月度纪律触发")
        self.assertEqual(w["level"], "crit")


class TestPlr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def test_plr_below_1p5_warns(self):
        df = _kline(base=100.0, drift=0.0)
        with mock.patch.object(self.rs, "_target_price", return_value=102.0):
            res = self.rs._check_plr("000001", df, 100.0, 95.0)  # 盈亏比 2/5=0.4
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["plr"], 0.4, places=6)
        w = res["warning"]
        self.assertIsNotNone(w)
        self.assertEqual(w["type"], "盈亏比不足")

    def test_plr_ok_no_warning(self):
        df = _kline(base=100.0, drift=0.0)
        with mock.patch.object(self.rs, "_target_price", return_value=110.0):
            res = self.rs._check_plr("000001", df, 100.0, 95.0)  # 盈亏比 10/5=2.0
        self.assertAlmostEqual(res["plr"], 2.0, places=6)
        self.assertIsNone(res["warning"])

    def test_plr_unavailable_without_stop(self):
        df = _kline(base=100.0, drift=0.0)
        res = self.rs._check_plr("000001", df, 100.0, None)
        self.assertFalse(res["ok"])


class TestSunkCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def test_deep_loss_no_reversal_is_trap(self):
        df = _kline(base=100.0, drift=0.0, n=30)   # 横盘 → 无反转信号
        close = float(df["close"].iloc[-1])
        res = self.rs._check_sunk_cost("000001", close * 1.2, close, df)  # 亏损16.7%
        self.assertTrue(res["ok"])
        self.assertFalse(res["reversal"])
        w = res["warning"]
        self.assertIsNotNone(w)
        self.assertEqual(w["type"], "沉没成本陷阱")
        self.assertEqual(w["level"], "crit")

    def test_deep_loss_with_reversal_not_trap(self):
        # 深亏但价格站上 MA5/MA10 → 按信号管理，不触发陷阱
        df = _kline(base=100.0, drift=1.0, n=30)   # 上行 → 反转信号
        close = float(df["close"].iloc[-1])
        res = self.rs._check_sunk_cost("000001", close * 1.2, close, df)
        self.assertTrue(res["reversal"])
        self.assertIsNone(res["warning"])

    def test_no_entry_skips(self):
        df = _kline(base=100.0, drift=0.0)
        res = self.rs._check_sunk_cost("000001", None, 100.0, df)
        self.assertFalse(res["ok"])
        self.assertIn("跳过", res["note"])


class TestAggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def _w(self, typ, level):
        return {"type": typ, "level": level, "code": "000001", "msg": typ}

    def test_crit_yields_high_exit(self):
        agg = self.rs._aggregate([self._w("违规持仓", "crit")], {"temperature": 50}, 1)
        self.assertEqual(agg["risk_level"], "高")
        self.assertEqual(agg["suggestion"], "离场")
        self.assertEqual(agg["signal"], "高-离场")

    def test_plr_warns_yields_mid_reduce(self):
        agg = self.rs._aggregate([self._w("盈亏比不足", "warn")], {"temperature": 50}, 1)
        self.assertEqual(agg["risk_level"], "中")
        self.assertEqual(agg["suggestion"], "减仓")

    def test_no_warning_yields_low_hold(self):
        agg = self.rs._aggregate([], {"temperature": 50}, 1)
        self.assertEqual(agg["risk_level"], "低")
        self.assertEqual(agg["suggestion"], "持有")

    def test_low_temperature_yields_watch(self):
        agg = self.rs._aggregate([], {"temperature": 30}, 1)
        self.assertEqual(agg["suggestion"], "观望")


class TestRiskSystemIntegration(_TmpDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rs = _import("risk_system")

    def _run(self, positions=None, date="2026-08-11"):
        tmp = self._make_tmp()
        for c in ("601899", "603993"):
            _kline(base=100.0, drift=0.1, n=60).to_parquet(tmp / f"{c}.parquet", index=False)
        patches = [
            mock.patch.object(self.rs, "KLINE_DIR", tmp),
            mock.patch.object(self.rs, "load_names",
                              return_value=({"601899": "甲", "603993": "乙"}, {})),
            mock.patch("quant_system.analysis_core.emotion_system.EmotionSystem.detect",
                       return_value={"temperature": 50.0, "temp_band": "中性",
                                     "stage": "divergence", "stage_cn": "分歧",
                                     "status": "ok"}),
            mock.patch.object(self.rs, "knowledge_rag", mock.MagicMock()),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        self.rs.knowledge_rag.search.return_value = [
            {"file": "skills/x.md", "cat": "心法", "score": 0.9, "text": "截断亏损让利润奔跑"}]
        return self.rs.RiskSystem(capital=1_000_000, watch=["601899", "603993"],
                                  positions=positions,
                                  out_dir=self._tmp / "out")

    def test_detect_structure(self):
        rs = self._run()
        res = rs.detect("2026-08-11")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["date"], "2026-08-11")
        self.assertIn(res["risk_level"], ("低", "中", "高"))
        self.assertIn(res["suggestion"], ("持有", "减仓", "离场", "观望"))
        for c in ("601899", "603993"):
            self.assertIn(c, res["stocks"])
            s = res["stocks"][c]
            self.assertIn("stop", s)
            self.assertIn("position", s)
            self.assertIn("plr", s)
        self.assertIsNotNone(res["drawdown"].get("pool_dd"))
        self.assertEqual(len(res["rag"]), 1)
        self.assertEqual(res["rag"][0]["file"], "skills/x.md")

    def test_report_writes_markdown(self):
        rs = self._run()
        out = rs.report("2026-08-11")  # 统一接口：report() 返回 Path
        self.assertIsInstance(out, Path)
        self.assertTrue(out.exists())
        md = out.read_text(encoding="utf-8")
        self.assertIn("# 🛡️ 风险纪律报告 2026-08-11", md)
        self.assertIn("止损位(close-1.5×ATR)", md)
        self.assertIn("回撤检查", md)
        self.assertIn("RAG 方法论依据", md)

    def test_view_dict(self):
        rs = self._run()
        v = rs.view("2026-08-11")
        self.assertEqual(v["agent"], "风险纪律")
        self.assertIn("signal", v)
        self.assertIn("confidence", v)
        self.assertGreaterEqual(v["confidence"], 0.0)
        self.assertIsInstance(v["evidence"], list)
        self.assertTrue(v["evidence"])

    def test_missing_kline_marked(self):
        tmp = self._make_tmp()
        _kline(n=60).to_parquet(tmp / "601899.parquet", index=False)
        with mock.patch.object(self.rs, "KLINE_DIR", tmp):
            rs = self.rs.RiskSystem(capital=1_000_000, watch=["601899", "603993"],
                                    out_dir=tmp / "out")
            res = rs.detect("2026-08-11")
        self.assertEqual(res["status"], "degraded")  # 部分标的失败 → degraded（对齐 emotion data_status）
        self.assertEqual(res["missing_count"], 1)
        self.assertIn("个股K线", res["data_status"])
        self.assertIn("603993", [m["code"] for m in res["missing"]])


if __name__ == "__main__":
    unittest.main()
