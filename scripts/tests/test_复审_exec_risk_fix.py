"""复审_执行风控层修复回归测试（F2.1 / F1.1 / F-06）。

覆盖:
  * F2.1: trade_db.add_sell 增加 A股 T+1 校验——当日买入的股份不可当日卖出。
          - 当日买入当日卖 → 拒绝
          - 昨日买入今日卖 → 放行
          - 纯存量持仓当日高抛（未当日买入）→ 放行（不误伤正常卖出）
  * F1.1: execution.create_order 增加持仓校验——超持仓卖单在下单阶段即被拒。
          - 无持仓卖单 → 拒绝
          - 超持仓卖单 → 拒绝
          - 正常买单 → 放行
          （买单余额校验因无账户上下文被如实跳过，见代码注释）
  * F-06: pipeline.daily_update 中 build_forces 必须先于 fuse_today，
          保证当天资金合力进当天融合温度。

测试使用临时 SQLite 库（monkeypatch trade_db._DB_PATH），不触碰生产库。
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_system import execution, trade_db

CST = timezone(timedelta(hours=8))


def _today() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (datetime.now(CST) - timedelta(days=1)).strftime("%Y-%m-%d")


class _TradeDbTestCase(unittest.TestCase):
    """基类：每个测试用例使用独立的临时 SQLite 库。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_db = str(Path(self._tmpdir.name) / "test_trade_log.db")
        self._orig_db_path = trade_db._DB_PATH
        trade_db._DB_PATH = tmp_db
        trade_db.init_db()

    def tearDown(self) -> None:
        trade_db._DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()


class TestAddSellTPlus1(_TradeDbTestCase):
    """F2.1: add_sell 的 T+1 校验。"""

    SYM = "TEST_T1"

    def test_same_day_buy_sell_rejected(self) -> None:
        """当日买入当日卖 → add_sell 被拒。"""
        buyd = trade_db.add_buy(self.SYM, "测试T1", shares=1000, price=10.0,
                                trade_date=_today())
        self.assertNotIn("error", buyd, buyd)
        res = trade_db.add_sell(self.SYM, shares=500, price=11.0,
                                trade_date=_today())
        self.assertIn("error", res, "当日买入的股份不应允许当日卖出")
        self.assertIn("T+1", res["error"], res["error"])

    def test_partial_sale_within_yesterday_bought_allowed(self) -> None:
        """昨日买入今日卖 → add_sell 放行（非 T+1 违规场景）。"""
        trade_db.add_buy(self.SYM, "测试T1", shares=1000, price=10.0,
                         trade_date=_yesterday())
        res = trade_db.add_sell(self.SYM, shares=300, price=11.0,
                                trade_date=_today())
        self.assertNotIn("error", res, res)
        pos = trade_db.get_position(self.SYM)
        self.assertEqual(pos["shares"], 700, pos)

    def test_holding_sell_without_same_day_buy_allowed(self) -> None:
        """纯存量（昨日已持有、今日无买入）当日高抛 → 放行（不误伤正常卖出）。"""
        trade_db.add_buy(self.SYM, "测试T1", shares=1000, price=10.0,
                         trade_date=_yesterday())
        # 昨日买入计为昨日，今日无任何 buy 记录
        res = trade_db.add_sell(self.SYM, shares=1000, price=12.0,
                                trade_date=_today())
        self.assertNotIn("error", res, res)
        pos = trade_db.get_position(self.SYM)
        self.assertEqual(pos["shares"], 0, pos)

    def test_oversell_still_rejected(self) -> None:
        """原有超卖校验不受影响。"""
        trade_db.add_buy(self.SYM, "测试T1", shares=500, price=10.0,
                         trade_date=_yesterday())
        res = trade_db.add_sell(self.SYM, shares=600, price=11.0,
                                trade_date=_today())
        self.assertIn("error", res, res)
        self.assertIn("超过持仓", res["error"], res["error"])


class TestCreateOrderHoldingCheck(unittest.TestCase):
    """F1.1: create_order 的持仓校验。"""

    SYM = "TEST_ORD"

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_db = str(Path(self._tmpdir.name) / "test_order_log.db")
        self._orig_db_path = trade_db._DB_PATH
        trade_db._DB_PATH = tmp_db
        trade_db.init_db()

    def tearDown(self) -> None:
        trade_db._DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    def test_sell_without_holding_rejected(self) -> None:
        """无持仓时卖单 → 下单阶段即被拒（防超持仓坏单入库）。"""
        res = execution.create_order(self.SYM, "limit", "sell", shares=100,
                                     price=10.0)
        self.assertIn("error", res, res)
        self.assertIn("超过当前持仓", res["error"], res["error"])
        # 坏单不应入库
        self.assertEqual(execution.get_orders(status="all", days=30), [])

    def test_sell_over_holding_rejected(self) -> None:
        """有持仓但卖单超过持仓 → 拒绝。"""
        trade_db.add_buy(self.SYM, "测试ORD", shares=500, price=10.0)
        res = execution.create_order(self.SYM, "limit", "sell", shares=600,
                                     price=10.0)
        self.assertIn("error", res, res)
        self.assertIn("超过当前持仓", res["error"], res["error"])

    def test_sell_within_holding_allowed(self) -> None:
        """有持仓且卖单未超持仓 → 放行。"""
        trade_db.add_buy(self.SYM, "测试ORD", shares=1000, price=10.0)
        res = execution.create_order(self.SYM, "limit", "sell", shares=300,
                                     price=10.0)
        self.assertNotIn("error", res, res)
        self.assertIn("id", res, res)

    def test_buy_order_normal_allowed(self) -> None:
        """正常买单（100 股整数倍）→ 放行。（余额校验因无账户上下文已如实跳过）"""
        res = execution.create_order(self.SYM, "limit", "buy", shares=400,
                                     price=10.0)
        self.assertNotIn("error", res, res)
        self.assertIn("id", res, res)


class TestPipelineDailyOrder(unittest.TestCase):
    """F-06: pipeline.daily_update 中 build_forces 必须先于 fuse_today。"""

    def test_build_forces_precedes_fuse_today(self) -> None:
        """静态顺序断言：build_forces 调用位置在 fuse_today 之前。"""
        src = inspect.getsource(
            __import__("quant_system.analysis_core.pipeline", fromlist=["x"])
            .daily_update
        )
        # 从函数源码中提取两个关键调用的行号，断言顺序。
        # 匹配实际调用（带 '(' ）而非注释/文档字符串里的提及。
        lines = src.splitlines()
        build_pos = next(i for i, l in enumerate(lines)
                         if "build_forces(" in l)
        fuse_pos = next(i for i, l in enumerate(lines)
                        if "fusion.fuse_today()" in l)
        self.assertLess(
            build_pos, fuse_pos,
            "F-06 违反顺序: build_forces 必须在 fuse_today 之前执行，"
            "否则当天资金合力进不了当天融合温度",
        )


if __name__ == "__main__":
    unittest.main()
