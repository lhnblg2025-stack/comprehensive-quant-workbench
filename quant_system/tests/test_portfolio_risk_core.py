"""portfolio_risk 核心单元测试（域 W 双轨制）。

轨道1 功能矩阵：VaR/CVaR、集中度、回撤、相关性、Kelly、综合报告。
轨道2 已知 bug 回归：confidence 参数真实生效、空/异常输入降级。

全部使用合成持仓、合成行情与临时文件替身，不触网、不读真实数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_system.portfolio_risk as pr


def make_price_panel(codes, n=80, seed=7):
    """构造多标的合成日线收盘价，保证日期完全对齐。"""
    dates = pd.date_range("2026-01-05", periods=n, freq="B").strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    prices = {}
    for offset, code in enumerate(codes):
        rets = rng.normal(0.0004, 0.012, n)
        prices[str(code)] = 100.0 * (1.0 + offset * 0.2) * np.cumprod(1.0 + rets)
    return list(dates), prices


class FakeAkshare:
    """只提供 compute_var / compute_correlation 所需的日线接口。"""

    def __init__(self, dates, prices):
        self.dates = dates
        self.prices = prices

    def stock_zh_a_hist(self, symbol=None, **kwargs):
        code = str(symbol)
        return pd.DataFrame({"日期": self.dates, "收盘": self.prices[code]})


def install_fake_akshare(monkeypatch, codes=("600000", "000001"), n=80, seed=7):
    dates, prices = make_price_panel(codes, n=n, seed=seed)
    monkeypatch.setitem(sys.modules, "akshare", FakeAkshare(dates, prices))
    return dates, prices


def make_positions(codes=("600000", "000001"), prices=(10.0, 20.0), qty=1000):
    return [
        {
            "stock": code,
            "stock_name": f"股票{code}",
            "qty": qty,
            "current_price": price,
        }
        for code, price in zip(codes, prices)
    ]


class TestComputeVar:
    def test_synthetic_returns_structure_and_reasonable_values(self, monkeypatch):
        install_fake_akshare(monkeypatch)
        positions = make_positions()

        result = pr.compute_var(positions, confidence=0.95)

        assert isinstance(result, dict)
        for key in ("var_95", "var_99", "cvar_95", "cvar_99", "var_custom", "cvar_custom"):
            assert key in result
        assert result["total_value"] == pytest.approx(30000.0)
        assert result["confidence_requested"] == pytest.approx(0.95)
        assert isinstance(result["var_threshold_breached"], bool)
        assert all(np.isfinite(result[key]) for key in ("var_95", "var_99", "cvar_95", "cvar_99"))

    def test_empty_positions_does_not_crash(self):
        result = pr.compute_var([], confidence=0.95)

        assert result["var"] == 0
        assert result["cvar"] == 0
        assert result["total_value"] == 0


class TestComputeConcentration:
    def test_single_position_has_full_weight(self, monkeypatch):
        monkeypatch.setattr(pr, "get_stock_sector", lambda code: "行业A")
        positions = [{"stock": "600000", "stock_name": "唯一持仓", "qty": 10, "current_price": 5.0}]

        result = pr.compute_concentration(positions)

        assert result["max_single"] == pytest.approx(100.0)
        assert result["max_single"] / 100.0 == pytest.approx(1.0)
        assert result["top5_weight"] == pytest.approx(100.0)
        assert result["max_single_name"] == "唯一持仓"

    def test_five_equal_weight_positions(self, monkeypatch):
        monkeypatch.setattr(pr, "get_stock_sector", lambda code: "行业A")
        positions = [
            {"stock": f"60000{i}", "stock_name": f"持仓{i}", "qty": 10, "current_price": 2.0}
            for i in range(5)
        ]

        result = pr.compute_concentration(positions)

        assert result["max_single"] == pytest.approx(20.0)
        assert result["max_single"] / 100.0 == pytest.approx(0.2)
        assert result["top5_weight"] == pytest.approx(100.0)
        assert len(result["top5_details"]) == 5

    def test_empty_positions_does_not_crash(self):
        result = pr.compute_concentration([])

        assert set(result) >= {"max_single", "top5", "industry_map", "warnings"}


class TestComputeDrawdown:
    def test_hand_calculated_max_drawdown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "ROOT", tmp_path / "quant_system")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "simulation_state.json").write_text(
            json.dumps({"pnl_history": [{"total_value": v} for v in [1.0, 1.1, 0.9, 1.05]]}),
            encoding="utf-8",
        )

        result = pr.compute_drawdown()

        assert result["max_drawdown"] == pytest.approx(18.1818, rel=1e-3)
        assert result["current_drawdown"] == pytest.approx(4.55, abs=1e-6)
        assert result["peak_value"] == pytest.approx(1.1)
        assert result["current_value"] == pytest.approx(1.05)


class TestComputeCorrelation:
    def test_synthetic_data_correlation_matches_numpy(self, monkeypatch):
        dates, prices = install_fake_akshare(monkeypatch, codes=("600000", "000001"))
        positions = make_positions()
        returns_a = pd.Series(prices["600000"]).pct_change().dropna().values
        returns_b = pd.Series(prices["000001"]).pct_change().dropna().values
        expected = np.corrcoef(returns_a, returns_b)[0, 1]

        result = pr.compute_correlation(positions)

        assert set(result) >= {"matrix", "avg_corr", "max_corr", "min_corr"}
        assert result["avg_corr"] == pytest.approx(expected, abs=5e-4)
        assert result["max_corr"] == pytest.approx(result["avg_corr"])
        assert result["min_corr"] == pytest.approx(result["avg_corr"])
        assert -1.0 <= result["avg_corr"] <= 1.0


class TestKelly:
    def test_compute_kelly_ratio_hand_calculation(self):
        result = pr.compute_kelly_ratio(0.6, 0.2, 0.1)

        assert result["kelly_pct"] == pytest.approx(40.0)
        assert result["half_kelly"] == pytest.approx(20.0)
        assert result["quarter_kelly"] == pytest.approx(10.0)
        assert result["win_rate"] == pytest.approx(60.0)
        assert result["profit_loss_ratio"] == pytest.approx(2.0)

    def test_compute_kelly_from_trades_returns_complete_structure(self):
        trades = [
            {"pnl_pct": 2.0},
            {"pnl_pct": -1.0},
            {"pnl_pct": 3.0},
            {"pnl_pct": -2.0},
        ]

        result = pr.compute_kelly_from_trades(trades)

        assert result["win_rate"] == pytest.approx(50.0)
        # 源码将盈亏比四舍五入到两位小数，保留 1.6667 为真值并放宽绝对容差覆盖 1.67。
        assert result["profit_loss_ratio"] == pytest.approx(1.6667, rel=1e-3, abs=0.004)
        assert result["kelly_pct"] == pytest.approx(20.0)
        assert result["half_kelly"] == pytest.approx(10.0)
        assert result["quarter_kelly"] == pytest.approx(5.0)


class TestFullRiskReport:
    def test_report_contains_all_sections(self, monkeypatch):
        install_fake_akshare(monkeypatch)
        positions = make_positions()
        monkeypatch.setattr(pr, "load_positions_from_db", lambda: positions)
        monkeypatch.setattr(pr, "get_stock_sector", lambda code: "行业A")
        monkeypatch.setattr(pr, "compute_ml_risk_score", lambda: {"ml_risk_score": 50})
        monkeypatch.setattr(pr, "compute_drawdown", lambda positions: {"current_drawdown": 0, "max_drawdown": 0})

        report = pr.full_risk_report()

        assert isinstance(report, dict)
        assert {
            "timestamp",
            "position_count",
            "positions",
            "var",
            "concentration",
            "drawdown",
            "correlation",
            "ml_risk",
        } <= set(report)
        assert report["position_count"] == 2


class TestRegression:
    def test_confidence_parameter_actually_changes_var(self, monkeypatch):
        install_fake_akshare(monkeypatch)
        positions = make_positions()

        result_95 = pr.compute_var(positions, confidence=0.95)
        result_99 = pr.compute_var(positions, confidence=0.99)

        assert result_95["confidence_requested"] == pytest.approx(0.95)
        assert result_99["confidence_requested"] == pytest.approx(0.99)
        assert result_99["var_custom"] > result_95["var_custom"]
        assert result_99["var_custom"] == pytest.approx(result_99["var_99"])
        assert result_95["var_custom"] == pytest.approx(result_95["var_95"])

    def test_none_positions_do_not_raise_uncaught_exception(self):
        assert isinstance(pr.compute_var(None), dict)
        assert isinstance(pr.compute_concentration(None), dict)
