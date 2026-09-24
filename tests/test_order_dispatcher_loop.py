"""
test_order_dispatcher_loop — 盘前链→执行链闭环验收（2026-08-23 C5）

覆盖: order_dispatcher 产出(generated/orders_{date}.json) 被 trade_executor
灌入执行队列(只 pending、去重、不自动下单)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_system import trade_executor as te  # noqa: E402


def _fake_dispatched(tmp_path: Path) -> Path:
    src = tmp_path / "orders_2026-08-21.json"
    src.write_text(json.dumps({"orders": [
        {"code": "600519", "name": "贵州茅台", "strategy": "趋势低吸", "signal_source": "共振85"},
        {"code": "1", "name": "平安银行", "strategy": "打板", "signal_source": "共振80"},  # 1→000001
        {"code": "", "name": "无效", "strategy": "x"},  # 无效 code 跳过
    ]}, ensure_ascii=False), encoding="utf-8")
    return src


def test_ingest_dispatched_orders_closed_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "ORDERS_FILE", tmp_path / "trade_orders.json")
    src = _fake_dispatched(tmp_path)

    orders, n = te.ingest_dispatched_orders([], date="2026-08-21", orders_path=str(src))
    assert n == 2  # 2 有效 + 1 无效跳过
    assert [o["stock"] for o in orders] == ["600519", "000001"]  # code zfill(6)
    assert all(o["status"] == "pending" for o in orders)
    assert all(o["qty_needs_manual"] is True for o in orders)  # 数量人工补全
    assert all(o["action"] == "buy" for o in orders)
    assert all(o["strategy"].startswith("battle_map:") for o in orders)

    # 二次灌入 → 去重, 不新增
    orders2, n2 = te.ingest_dispatched_orders(orders, date="2026-08-21", orders_path=str(src))
    assert n2 == 0 and len(orders2) == 2

    # 落盘可回读（供人工确认队列）
    saved = json.loads(te.ORDERS_FILE.read_text(encoding="utf-8"))
    assert len(saved) == 2


def test_ingest_no_file_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(te, "ORDERS_FILE", tmp_path / "trade_orders.json")
    orders, n = te.ingest_dispatched_orders([], date="2026-08-21",
                                            orders_path=str(tmp_path / "missing.json"))
    assert n == 0 and orders == []
