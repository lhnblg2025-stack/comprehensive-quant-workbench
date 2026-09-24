"""analysis_core/order_dispatcher.py 单元测试（域J：指令转化层）。

全 mock：battle_map 用 tmp 桩、_kline_close 用桩/合成序列，不依赖本地 parquet。
覆盖：攻击组→订单转化、去重归一、相关性约束 0.7 压缩、错误路径结构化返回。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.analysis_core import order_dispatcher as od


def _battle_map(core: list[dict]) -> dict:
    return {
        "position_range": "30%-60%",
        "emotion_stage": "亢奋",
        "macro_veto": "无",
        "regime": "震荡",
        "attack_groups": {"core": core},
    }


def _core(code: str, name: str, score: int, strategy: str = "趋势低吸") -> dict:
    return {
        "leader": {"code": code, "name": name, "boards": 1},
        "score": score,
        "strategy": strategy,
    }


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """ROOT 重定向 tmp，避免写真实 generated/；kline 桩默认无数据（相关性不触发压缩）。"""
    (tmp_path / "generated").mkdir()
    monkeypatch.setattr(od, "ROOT", tmp_path)
    monkeypatch.setattr(od, "_kline_close", lambda code, days=60: None)
    return tmp_path


def _write_battle_map(root: Path, bm: dict, date: str = "2026-08-12"):
    (root / "generated" / f"battle_map_{date}.json").write_text(
        json.dumps(bm, ensure_ascii=False), encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# dispatch 主流程
# ════════════════════════════════════════════════════════════════

class TestDispatch:
    def test_basic_two_orders(self, tmp_root):
        _write_battle_map(tmp_root, _battle_map([
            _core("600000", "A", 85), _core("000001", "B", 80, "打板")]))
        r = od.dispatch("2026-08-12", capital=1_000_000)
        assert len(r["orders"]) == 2
        for o in r["orders"]:
            assert o["ratio"] == pytest.approx(0.3)      # 0.6 / 2 等权
            assert o["amount"] == 300_000
            assert o["signal_source"].startswith("共振")
            assert o["price_cond"] in ("竞价>3%不炸板", "回踩不破均线")
            assert o["stop"] == "-2.5%"                  # 震荡 regime
        assert r["total_exposure"] == pytest.approx(0.6)
        assert r["correlation"]["scale"] == 1.0
        # 输出文件落盘
        out = tmp_root / "generated" / "orders_2026-08-12.json"
        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8"))["date"] == "2026-08-12"

    def test_dedup_keeps_highest_score(self, tmp_root):
        _write_battle_map(tmp_root, _battle_map([
            _core("600000", "A", 85), _core("600000", "A", 90)]))
        r = od.dispatch("2026-08-12", capital=1_000_000)
        assert len(r["orders"]) == 1
        assert r["orders"][0]["score"] == 90
        assert r["orders"][0]["ratio"] == pytest.approx(0.6)
        assert r["orders"][0]["amount"] == 600_000

    def test_correlation_scale_compresses(self, tmp_root, monkeypatch):
        _write_battle_map(tmp_root, _battle_map([_core("600000", "A", 85)]))
        monkeypatch.setattr(
            od, "_correlation_constraint",
            lambda orders: {"note": "高相关组合 1 对 → 总仓位上限压缩30%", "scale": 0.7})
        r = od.dispatch("2026-08-12", capital=1_000_000)
        assert r["orders"][0]["ratio"] == pytest.approx(0.42)   # 0.6 * 0.7
        assert r["orders"][0]["amount"] == 420_000
        assert r["correlation"]["scale"] == 0.7

    def test_missing_battle_map_structured_error(self, tmp_root):
        r = od.dispatch("2099-01-01")
        assert r == {"date": "2099-01-01", "error": "无作战地图，先运行 battle_map"}

    def test_group_without_leader_skipped(self, tmp_root):
        _write_battle_map(tmp_root, _battle_map([{"score": 80, "strategy": "x"}]))
        r = od.dispatch("2026-08-12")
        assert r["orders"] == []
        assert r["total_exposure"] == 0.0

    def test_default_capital(self, tmp_root):
        _write_battle_map(tmp_root, _battle_map([_core("600000", "A", 85)]))
        r = od.dispatch("2026-08-12")
        assert r["orders"][0]["amount"] == 600_000  # 默认 100 万 × 0.6


# ════════════════════════════════════════════════════════════════
# _correlation_constraint（真实逻辑 + _kline_close 桩）
# ════════════════════════════════════════════════════════════════

class TestCorrelationConstraint:
    def _orders(self, codes, names=None):
        names = names or [f"N{i}" for i in range(len(codes))]
        return [{"code": c, "name": n} for c, n in zip(codes, names)]

    def test_fewer_than_two_orders(self):
        r = od._correlation_constraint([{"code": "600000", "name": "A"}])
        assert r["scale"] == 1.0 and "订单<2" in r["note"]

    def test_high_corr_compresses(self, monkeypatch):
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        base = pd.Series(np.linspace(1, 2, n), index=dates)
        monkeypatch.setattr(od, "_kline_close", lambda code, days=60: base if code == "600000" else base * 2)
        r = od._correlation_constraint(self._orders(["600000", "000001"]))
        assert r["scale"] == pytest.approx(0.7)
        assert "高相关组合" in r["note"]

    def test_low_corr_no_compression(self, monkeypatch):
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        s1 = pd.Series(np.sin(np.arange(n)), index=dates)
        s2 = pd.Series(np.cos(np.arange(n)), index=dates)
        monkeypatch.setattr(od, "_kline_close",
                            lambda code, days=60: s1 if code == "600000" else s2)
        r = od._correlation_constraint(self._orders(["600000", "000001"]))
        assert r["scale"] == 1.0
        assert "无高相关组合" in r["note"]

    def test_insufficient_kline_data(self, monkeypatch):
        short = pd.Series([1.0] * 15, index=pd.date_range("2026-01-01", periods=15))
        monkeypatch.setattr(od, "_kline_close", lambda code, days=60: short)
        r = od._correlation_constraint(self._orders(["600000", "000001"]))
        assert r["scale"] == 1.0 and "数据不足" in r["note"]

    def test_missing_kline_falls_back(self, monkeypatch):
        monkeypatch.setattr(od, "_kline_close", lambda code, days=60: None)
        r = od._correlation_constraint(self._orders(["600000", "000001"]))
        assert r["scale"] == 1.0 and "相关性数据不足" in r["note"]

    def test_negative_high_corr_also_compresses(self, monkeypatch):
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        base = pd.Series(np.linspace(1, 2, n), index=dates)
        monkeypatch.setattr(od, "_kline_close", lambda code, days=60: base if code == "600000" else -base)
        r = od._correlation_constraint(self._orders(["600000", "000001"]))
        assert r["scale"] == pytest.approx(0.7)  # abs(corr) > 0.7 同样压缩
