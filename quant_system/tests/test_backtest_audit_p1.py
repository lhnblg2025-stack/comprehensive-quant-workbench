"""P1 回测层审计修复测试（audit 回测层.md P1-1/P1-2/P1-4/P1-5/P1-6）。

全部使用合成数据 / 纯函数断言，不读真实数据、不触网。
覆盖：
  - P1-1: market_cap_tiered_slippage 市值分档阈值 + backtest.py 消费路径
  - P1-2: CorporateActionsHandler 事件记账（现金分红入现金 / 送转调股数）
          + backtest.py run_backtest 主成交路径接入
  - P1-4: MetricsCalculator.sharpe 统一口径，backtest_engine.compute_sharpe 转发一致
          + backtest_pro._compute_metrics 减 rf 分支
  - P1-5: 止损/止盈跌停不可卖判定
  - P1-6: DEFAULT_SLIPPAGE_RATE 单一真源，config/backtest/combined/SimBroker 引用一致
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.config import PortfolioConfig, StrategyConfig
from quant_system.metrics_calculator import MetricsCalculator


# ── P1-1: 市值分档滑点 ──────────────────────────────────────────────────
def test_market_cap_tiered_slippage_thresholds():
    from quant_system.execution_broker import market_cap_tiered_slippage

    # < 50亿 → 20bp
    assert market_cap_tiered_slippage(1e8) == pytest.approx(0.002)
    assert market_cap_tiered_slippage(49e8) == pytest.approx(0.002)
    # 50亿~200亿 → 10bp
    assert market_cap_tiered_slippage(50e8) == pytest.approx(0.001)
    assert market_cap_tiered_slippage(150e8) == pytest.approx(0.001)
    assert market_cap_tiered_slippage(199e8) == pytest.approx(0.001)
    # > 200亿 → 5bp
    assert market_cap_tiered_slippage(200e8) == pytest.approx(0.0005)
    assert market_cap_tiered_slippage(1e12) == pytest.approx(0.0005)
    # 未知/非正 → 保守按小盘 20bp
    assert market_cap_tiered_slippage(None) == pytest.approx(0.002)
    assert market_cap_tiered_slippage(0) == pytest.approx(0.002)


def _make_signal_df(n: int = 120):
    """构造可产生有效信号的合成 OHLCV df（q为单调+正弦，MA 可算）。"""
    dates = pd.date_range("2025-01-01", periods=n)
    base = np.linspace(10, 14, n)
    wob = 1 + 0.03 * np.sin(np.arange(n) * 0.3)
    close = base * wob
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.02
    low = np.minimum(open_, close) * 0.98
    vol = np.random.RandomState(1).randint(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "date": dates, "symbol": "600519", "open": open_, "close": close,
        "high": high, "low": low, "volume": vol,
    })


def test_backtest_p1_consumes_market_cap_based_slippage():
    """P1-1: backtest.py 在 slippage_mode=market_cap_based 时按市值分档滑点，
    小盘(高滑点)成交价劣于大盘(低滑点) → 小盘最终净值更低。"""
    from quant_system.backtest import run_backtest

    strat = StrategyConfig(fast_ma=5, slow_ma=10, trend_ma=20, long_trend_ma=30, volume_ma=5)
    small = run_backtest(
        _make_signal_df().assign(market_cap=30e8), strat,
        PortfolioConfig(slippage_mode="market_cap_based"))
    large = run_backtest(
        _make_signal_df().assign(market_cap=1000e8), strat,
        PortfolioConfig(slippage_mode="market_cap_based"))
    assert small["status"] == "ok" and large["status"] == "ok"
    assert small["final_equity"] < large["final_equity"]


# ── P1-2: 动态复权事件记账 ──────────────────────────────────────────────
def test_corporate_actions_dividend_into_cash():
    """P1-2: 一次现金分红正确计入现金（股数不变）。"""
    from quant_system.backtest_pro import CorporateActionsHandler as CAH

    shares, cash = 1000, 5000.0
    new_shares, new_cash = CAH.apply_dividend_to_position(shares, cash, 0.5)
    assert new_shares == 1000                       # 股数不变
    assert new_cash == pytest.approx(5000.0 + 1000 * 0.5)  # 现金 += 股数×每股分红


def test_corporate_actions_split_adjusts_shares_and_cost():
    """P1-2: 一次送转(1→2)调整股数×ratio、总成本不变、每股成本重摊。"""
    from quant_system.backtest_pro import CorporateActionsHandler as CAH

    shares, cash, cost_basis = 500, 0.0, 10.0
    new_shares, new_cash, new_cost = CAH.apply_split_to_position(shares, cash, cost_basis, 2.0)
    assert new_shares == 1000                        # 股数 ×2
    assert new_cash == pytest.approx(cash)           # 现金不变
    assert new_cost == pytest.approx(cost_basis / 2)  # 每股成本 /2


def test_corporate_actions_events_from_columns():
    """P1-2: 轻量适配——从日线显式事件列推导 dividend/split 事件列表。"""
    from quant_system.backtest_pro import CorporateActionsHandler as CAH

    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-05", "2025-01-06", "2025-01-07"]),
        "close": [10, 10, 5],
        "dividend": [np.nan, 0.5, np.nan],
        "split_ratio": [np.nan, np.nan, 2.0],
    })
    events = CAH.events_from_price_columns(df)
    types = [e["type"] for e in events]
    assert "dividend" in types and "split" in types
    div = [e for e in events if e["type"] == "dividend"][0]
    spl = [e for e in events if e["type"] == "split"][0]
    assert div["amount"] == pytest.approx(0.5)
    assert spl["ratio"] == pytest.approx(2.0)


def test_backtest_run_integrates_corporate_actions():
    """P1-2: run_backtest 主成交路径接入事件记账——分红入现金、送转调股数。"""
    from quant_system.backtest import run_backtest

    strat = StrategyConfig(fast_ma=5, slow_ma=10, trend_ma=20, long_trend_ma=30, volume_ma=5)
    df = _make_signal_df()
    df.loc[60, "dividend"] = 0.5
    df.loc[70, "split_ratio"] = 2.0
    res = run_backtest(df, strat, PortfolioConfig())
    assert res["status"] == "ok"
    sides = [t["side"] for t in res["trades"]]
    assert "DIVIDEND" in sides and "SPLIT" in sides


# ── P1-4: 统一 Sharpe 无风险利率口径 ──────────────────────────────────
def test_sharpe_rf_consistency_across_engines():
    """P1-4: 统一口径——MetricsCalculator.sharpe 与 backtest_engine.compute_sharpe 一致，
    且都扣 rf；backtest_pro._compute_metrics 的 sharpe 也复用同一口径(减 rf)。"""
    from quant_system.backtest_engine import compute_sharpe

    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008, 0.012,
                         0.005, -0.002, 0.01, 0.018, -0.008, 0.004, 0.011])
    mc = MetricsCalculator.sharpe(returns, rf=0.02, ddof=1)
    engine = compute_sharpe(returns, risk_free_rate=0.02)
    assert mc == pytest.approx(engine)

    # 与 rf=0 不同（必须扣无风险利率）
    no_rf = MetricsCalculator.sharpe(returns, rf=0.0, ddof=1)
    assert mc != pytest.approx(no_rf)


def test_backtest_pro_sharpe_deducts_rf():
    """P1-4: backtest_pro._compute_metrics 不再 sharpe=ann_ret/ann_vol(不减 rf)，
    而是复用 MetricsCalculator.sharpe（扣 rf）。"""
    from quant_system.backtest_pro import MultiAssetBacktest

    idx = pd.date_range("2025-01-01", periods=30)
    prices = pd.concat([
        pd.Series(100 * (1 + np.random.RandomState(0).normal(0, 0.01, 30)).cumprod(),
                  index=idx, name="A"),
        pd.Series(100 * (1 + np.random.RandomState(1).normal(0.002, 0.005, 30)).cumprod(),
                  index=idx, name="B"),
    ], axis=1)
    signals = pd.DataFrame(np.ones((30, 2)), index=idx, columns=["A", "B"])
    bt = MultiAssetBacktest()
    result = bt.run(prices, signals)
    sharpe_from_metrics = result.metrics["sharpe_ratio"]
    expected = MetricsCalculator.sharpe(result.returns, rf=0.02, ddof=1)
    assert sharpe_from_metrics == pytest.approx(expected)


# ── P1-5: 止损/止盈跌停不可卖 ──────────────────────────────────────────
def test_stop_loss_withheld_on_down_limit_day():
    """P1-5: 当日跌停封板时止损触发应判不可成交（撤回，持仓保留），
    而不是模拟在跌停价卖出。构造：建仓后某日单日深跌到跌停价(<-10%)，
    stop_loss=2% 已触发，但应无 SELL 成交。"""
    from quant_system.backtest import run_backtest

    strat = StrategyConfig(fast_ma=5, slow_ma=60, trend_ma=20, long_trend_ma=30,
                           volume_ma=5, stop_loss_pct=0.02)
    df = _make_signal_df(120)
    # 找出自然买入日，确定持仓窗口
    base = run_backtest(df.copy(), strat, PortfolioConfig())
    buys = [t for t in base["trades"] if t["side"] == "BUY"]
    assert buys, "前提：策略应产生至少一笔记账 BUY"
    buy_day = pd.Timestamp(buys[0]["date"])
    buy_idx = df.index[df["date"] == buy_day][0]
    held_idx = buy_idx + 2  # 建仓后第2个交易日仍持仓
    prev_close = float(df.iloc[held_idx - 1]["close"])
    # 当日直接跌到跌停价之下（-11% < -10% 主跌停），止损 2% 必然触发
    df.loc[held_idx, "close"] = prev_close * 0.89
    df.loc[held_idx, "low"] = df.loc[held_idx, "close"] * 0.98
    df.loc[held_idx, "high"] = prev_close

    res = run_backtest(df, strat, PortfolioConfig())
    crash_date = str(df.iloc[held_idx]["date"].date())
    crash_sells = [t for t in res["trades"]
                   if t["side"] == "SELL" and str(t["date"]) == crash_date]
    assert crash_sells == [], (
        f"跌停日({crash_date})不应产生模拟卖出成交，实际: {crash_sells}"
    )


def test_stop_loss_still_sells_when_not_at_down_limit():
    """P1-5: 对照——同样跌幅但未触及跌停(如-6%)时，止损可正常成交卖出。"""
    from quant_system.backtest import run_backtest

    strat = StrategyConfig(fast_ma=5, slow_ma=60, trend_ma=20, long_trend_ma=30,
                           volume_ma=5, stop_loss_pct=0.02)
    df = _make_signal_df(120)
    base = run_backtest(df.copy(), strat, PortfolioConfig())
    buys = [t for t in base["trades"] if t["side"] == "BUY"]
    buy_idx = df.index[df["date"] == pd.Timestamp(buys[0]["date"])][0]
    held_idx = buy_idx + 2
    prev_close = float(df.iloc[held_idx - 1]["close"])
    # -6%：触发止损(2%)但未跌停(-10%) → 可成交卖出
    df.loc[held_idx, "close"] = prev_close * 0.94
    df.loc[held_idx, "low"] = df.loc[held_idx, "close"] * 0.98
    df.loc[held_idx, "high"] = prev_close

    res = run_backtest(df, strat, PortfolioConfig())
    crash_date = str(df.iloc[held_idx]["date"].date())
    crash_sells = [t for t in res["trades"]
                   if t["side"] == "SELL" and str(t["date"]) == crash_date]
    assert len(crash_sells) == 1, f"-6% 日止损应正常卖出，实际: {crash_sells}"


# ── P1-6: 跨引擎滑点唯一真源 ──────────────────────────────────────────
def test_default_slippage_single_source():
    """P1-6: 滑点默认值统一到 execution_broker.DEFAULT_SLIPPAGE_RATE(10bp)，
    backtest.py/config/combined/SimBroker 不再各自写死差异化数值。"""
    from quant_system import combined_backtest as cb
    from quant_system.config import PortfolioConfig
    from quant_system.execution_broker import DEFAULT_SLIPPAGE_RATE, SimBroker

    assert DEFAULT_SLIPPAGE_RATE == pytest.approx(0.001)
    assert PortfolioConfig().slippage_pct == pytest.approx(DEFAULT_SLIPPAGE_RATE)
    assert cb.SLIPPAGE == pytest.approx(DEFAULT_SLIPPAGE_RATE)
    # SimBroker 缺省滑点走 DEFAULT_SLIPPAGE_RATE（不传 adv/sigma 时恒 fallback）
    broker = SimBroker(initial_cash=1_000_000.0)
    broker.update_market_price("000001", 10.0)
    order = broker.place_order("000001", "buy", 1000, order_type="market")
    assert order.status == "filled"
    assert order.avg_price == pytest.approx(10.0 * (1 + DEFAULT_SLIPPAGE_RATE))
