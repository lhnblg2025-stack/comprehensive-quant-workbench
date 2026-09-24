"""factor_model 核心模型单元测试（域 W 双轨制）。

轨道1 功能矩阵：股票池格式、因子计算输出、去极值/标准化/中性化、
beta、短历史容错。
轨道2 已知 bug 回归：beta 必须使用市场基准而非股票自身、特异风险 180 日窗口、
IC 序列按交易日步进。

全部使用合成 DataFrame/Series，不读真实数据、不触网。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_system.factor_model as fm
from quant_system.factor_model import FactorModel, _neutralize, _standardize, _winsorize


DATE = "2026-04-01"


def make_ohlcv(symbol: str, n: int = 80) -> pd.DataFrame:
    """构造合成日线 OHLCV；不同 symbol 有确定性但不同的走势。"""
    seed = int(symbol) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0.0, 0.08, n)) + np.linspace(0.0, 1.5, n)
    close = np.maximum(close, 1.0)
    volume = rng.uniform(100_000.0, 500_000.0, n)
    pct_chg = pd.Series(close).pct_change().fillna(0.0).values
    dates = pd.date_range("2026-01-04", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": volume,
            "amount": volume * close * 100.0,
            "outstanding_share": rng.uniform(50_000_000.0, 200_000_000.0, n),
            "pct_chg": pct_chg,
        }
    )


class FakeStore:
    """不落盘、不联网的数据存储替身。"""

    def __init__(self, dfs: dict[str, pd.DataFrame]):
        self.dfs = {sym: df.copy() for sym, df in dfs.items()}
        self.calls: list[tuple] = []

    def get_many(self, symbols, days=260, end=None):
        return {sym: self.dfs[sym].copy() for sym in symbols if sym in self.dfs}

    def get(self, symbol, days=None, end=None, start=None):
        self.calls.append((symbol, days, start, end))
        if symbol not in self.dfs:
            return pd.DataFrame()
        return self.dfs[symbol].copy()


class FakeFactorStore:
    """避免 compute_factors 写入/读取真实 ~/.quant_system/factors。"""

    def load(self, start_date, end_date):
        return pd.DataFrame()

    def is_cache_safe(self, date_str):
        return False

    def save(self, *args, **kwargs):
        return None


# ════════════════════════════════════════════════════════════════
# 轨道1：功能矩阵
# ════════════════════════════════════════════════════════════════


class TestUniverseAndComputeFactors:
    def test_get_universe_returns_formatted_symbols(self):
        model = FactorModel(universe_size=12)
        universe = model.get_universe()

        assert isinstance(universe, list)
        assert len(universe) == 12
        assert all(isinstance(s, str) and len(s) == 6 and s.isdigit() for s in universe)

    def test_compute_factors_returns_expected_keys(self, monkeypatch):
        symbols = [f"6000{i:02d}" for i in range(12)]
        dfs = {sym: make_ohlcv(sym, 80) for sym in symbols}
        store = FakeStore(dfs)
        monkeypatch.setattr(fm, "get_store", lambda: store)

        model = FactorModel(universe_size=len(symbols))
        model.factor_store = FakeFactorStore()

        market_rng = np.random.default_rng(42)
        monkeypatch.setattr(
            model,
            "_get_market_returns",
            lambda n_days, date_str=None: market_rng.normal(0.0, 0.01, max(int(n_days), 1)),
        )

        def fake_gfv(symbol: str, key: str) -> float:
            return float(1.0 + (int(symbol) % 20) * 0.05 + len(key) * 0.01)

        monkeypatch.setattr(fm, "_gfv_nan", fake_gfv)

        result = model.compute_factors(DATE, symbols=symbols, neutralize=False)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert set(symbols) <= set(result.index)

        expected_keys = {
            "mom_1m",
            "daily_vol_20d",
            "max_dd_60d",
            "beta_60d",
            "amihud_illiq",
            "ln_cap",
        }
        assert expected_keys <= set(result.columns)


class TestWinsorizeStandardizeNeutralize:
    def test_winsorize_clips_extreme_values(self):
        s = pd.Series([-100.0, -2.0, 0.0, 1.0, 2.0, 3.0, 100.0])
        out = _winsorize(s)

        lo = s.quantile(fm.WINSORIZE_LOWER)
        hi = s.quantile(fm.WINSORIZE_UPPER)
        assert out.min() == pytest.approx(lo)
        assert out.max() == pytest.approx(hi)
        assert out.iloc[0] == pytest.approx(lo)
        assert out.iloc[-1] == pytest.approx(hi)
        assert ((out >= lo - 1e-12) & (out <= hi + 1e-12)).all()

    def test_standardize_mean_zero_std_one(self):
        z = _standardize(pd.Series(np.arange(10.0)))
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0, abs=1e-12)

    def test_standardize_constant_returns_zero(self):
        z = _standardize(pd.Series([3.0] * 5))
        assert (z.abs() < 1e-12).all()
        assert len(z) == 5

    def test_neutralize_removes_industry_effects(self):
        n = 60
        index = [f"S{i:03d}" for i in range(n)]
        d1 = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(n)], index=index)
        d2 = pd.Series([1.0 if i % 3 == 0 else 0.0 for i in range(n)], index=index)
        dummies = pd.DataFrame({"industry_a": d1, "industry_b": d2}, index=index)
        noise = np.random.default_rng(7).normal(0.0, 0.01, n)
        factor = 5.0 * d1 + 10.0 * d2 + pd.Series(noise, index=index)

        residual = _neutralize(factor, dummies)

        assert residual.notna().all()
        assert residual.std() > 1e-6

        raw_corr_a = abs(factor.corr(dummies["industry_a"]))
        raw_corr_b = abs(factor.corr(dummies["industry_b"]))
        neutralized_corr_a = abs(residual.corr(dummies["industry_a"]))
        neutralized_corr_b = abs(residual.corr(dummies["industry_b"]))
        assert neutralized_corr_a < 0.1
        assert neutralized_corr_b < 0.1
        assert neutralized_corr_a < raw_corr_a
        assert neutralized_corr_b < raw_corr_b


class TestBetaAndShortHistory:
    def test_calc_beta_matches_cov_over_var(self):
        stock_ret = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
        market_ret = np.array([0.005, 0.01, -0.005, 0.02, -0.01, 0.0])
        expected = np.cov(stock_ret, market_ret)[0, 1] / np.var(market_ret)

        assert FactorModel._calc_beta(stock_ret, market_ret) == pytest.approx(expected)

    def test_calc_beta_constant_series_does_not_crash(self):
        # 约定降级：市场方差为 0 时 beta 未定义，实现约定返回 1.0
        assert FactorModel._calc_beta(np.ones(10), np.ones(10)) == pytest.approx(1.0)
        assert FactorModel._calc_beta(np.ones(10), np.arange(10.0)) == pytest.approx(0.0)

    def test_compute_factors_short_history_does_not_crash(self, monkeypatch):
        symbols = ["600000", "600001", "600002"]
        dfs = {sym: make_ohlcv(sym, 5) for sym in symbols}
        store = FakeStore(dfs)
        monkeypatch.setattr(fm, "get_store", lambda: store)

        model = FactorModel(universe_size=3)
        model.factor_store = FakeFactorStore()
        monkeypatch.setattr(fm, "_gfv_nan", lambda symbol, key: 1.0)
        monkeypatch.setattr(model, "_get_market_returns", lambda n_days, date_str=None: np.zeros(max(n_days, 1)))

        result = model.compute_factors(DATE, symbols=symbols, neutralize=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ════════════════════════════════════════════════════════════════
# 轨道2：已知 bug 回归
# ════════════════════════════════════════════════════════════════


class TestMarketBenchmarkBetaRegression:
    def test_calc_beta_uses_market_returns_not_stock_self(self, monkeypatch):
        stock_ret = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
        market_ret = np.array([0.03, -0.02, 0.01, -0.03, 0.02, -0.01])
        monkeypatch.setattr(FactorModel, "_get_market_returns", lambda self, n_days, date_str=None: market_ret.copy())

        model = FactorModel(universe_size=1)
        benchmark = model._get_market_returns(len(stock_ret))

        assert np.array_equal(benchmark, market_ret)
        expected = np.cov(stock_ret, benchmark)[0, 1] / np.var(benchmark)
        beta = FactorModel._calc_beta(stock_ret, benchmark)
        assert beta == pytest.approx(expected)


class TestSpecificRiskWindowRegression:
    def test_specific_risk_requests_180_day_history_and_short_history_safe(
        self, monkeypatch
    ):
        symbols = ["600000", "600001", "600002", "600003", "600004"]
        short_history = {
            sym: pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-05", periods=30, freq="B"),
                    "pct_chg": np.linspace(-0.01, 0.01, 30),
                }
            )
            for sym in symbols
        }
        store = FakeStore(short_history)
        monkeypatch.setattr(fm, "get_store", lambda: store)

        model = FactorModel(universe_size=5)
        factor_df = pd.DataFrame(
            {
                "ln_cap": [10.0, 11.0, 9.5, 10.5, 9.8],
                "mom_1m": [1.0, -1.0, 0.5, -0.5, 0.2],
            },
            index=symbols,
        )
        monkeypatch.setattr(model, "compute_factors", lambda *a, **k: factor_df)

        result = model.risk_model(DATE, symbols=symbols)

        assert result["status"] == "ok"
        assert len(result["specific_risk"]) == len(symbols)
        # 30 行历史 < 180 天，且截面观测不足，应可见降级为默认 0.02 而不是崩溃。
        assert all(risk == pytest.approx(0.02) for risk in result["specific_risk"])

        get_calls = [call for call in store.calls if call[0] in symbols]
        assert get_calls
        assert all(call[1] == 180 for call in get_calls)


class TestICSeriesStepRegression:
    def test_ic_series_steps_by_trading_date_index_not_calendar_days(self, monkeypatch):
        trading_dates = [
            "2026-08-10",
            "2026-08-11",
            "2026-08-12",
            "2026-08-13",
            "2026-08-14",
            "2026-08-17",
            "2026-08-18",
        ]
        model = FactorModel()
        visited: list[str] = []

        monkeypatch.setattr(
            model,
            "_get_trading_dates",
            lambda start, end: [d for d in trading_dates if start <= d <= end],
        )

        def fake_compute_ic(date_str, forward_days=5):
            visited.append(date_str)
            return pd.Series({"factor_a": 1.0})

        monkeypatch.setattr(model, "compute_ic", fake_compute_ic)

        result = model.compute_ic_series(
            "2026-08-10",
            "2026-08-18",
            step_days=3,
            forward_days=5,
        )

        # 按交易日列表索引 0,3,6 步进，而不是按自然日间隔步进。
        assert visited == [trading_dates[i] for i in (0, 3, 6)]
        assert list(result.index) == visited
        assert list(result.columns) == ["factor_a"]
