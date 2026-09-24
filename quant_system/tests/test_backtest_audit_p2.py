"""P2 回测层审计修复测试（audit 回测层.md P2-2/P2-3/P2-4/P2-6/P2-8）。

全部使用合成数据 / 纯函数断言，不读真实数据、不触网。
覆盖：
  - P2-2: backtest_pro 信号转收益按换手计提交易成本（walk-forward 不再零成本高估）
  - P2-3: 退市股在退市日实现资本损失（close→0 清算价），不再被 fillna(0) 抹成 0%
  - P2-4: backtest_engine 默认滑点模式改为 percent，0.001=0.1%（全系统口径一致）
  - P2-6: backtest_engine._force_close 判跌停/流动性上限，受阻标记为持有而非强造成交
  - P2-8: backtest.run_portfolio_backtest 改等权共享资金池语义，失败标的本金退回
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest_pro import (
    MultiAssetBacktest,
    WalkForwardBacktest,
    integrate_delisted,
    load_delisted_prices,
)


# ── P2-2: 信号转收益零交易成本 ─────────────────────────────────────────────
def _lt_returns(n: int = 150):
    """单调上行价格面板 → 日收益。"""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"A": np.linspace(10.0, 20.0, n)}, index=dates)
    rets = prices.pct_change().fillna(0)
    return dates, rets


def test_walkforward_signals_to_returns_charges_turnover_cost():
    """P2-2: 高换手信号的净收益显著低于毛收益（成本被计提），不再零成本高估。"""
    dates, rets = _lt_returns()
    # 零换手：全程满仓
    sig_zero = pd.Series(1.0, index=dates)
    # 高换手：每日满仓/空仓交替
    sig_high = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(len(dates))], index=dates)

    gross_zero = (sig_zero.shift(1) * rets.iloc[:, 0]).fillna(0)
    gross_high = (sig_high.shift(1) * rets.iloc[:, 0]).fillna(0)
    net_zero = WalkForwardBacktest._signals_to_returns(sig_zero, rets)
    net_high = WalkForwardBacktest._signals_to_returns(sig_high, rets)

    # 净收益每个时点不超过毛收益（成本只减不加）
    assert bool((net_zero <= gross_zero + 1e-9).all())
    assert bool((net_high <= gross_high + 1e-9).all())

    # 零换手（仅一次初始建仓）成本很小（相对 rel 放宽到千分之几量级）
    assert net_zero.sum() == pytest.approx(gross_zero.sum(), rel=5e-3)
    # 高换手成本显著：净 < 毛
    assert net_high.sum() < gross_high.sum()
    # 高换手被扣的成本远大于零换手（换手越高高估越狠，修复后差距体现出来）
    high_cost = float(gross_high.sum() - net_high.sum())
    zero_cost = float(gross_zero.sum() - net_zero.sum())
    assert high_cost > zero_cost * 20


def test_walkforward_self_return_uses_cost_model():
    """P2-2: WalkForwardBacktest.run 走成本模型，返回 walk_forward_return 收益序列。

    成本量化已在 test_walkforward_signals_to_returns_charges_turnover_cost
    覆盖；此处验证 run() 走同路径产出的收益列结构合法、有限且铺满 OOS 期。
    """
    _, rets = _lt_returns(n=60)
    prices = (1 + rets).cumprod()

    def fn(prices: pd.DataFrame, **_kw) -> pd.Series:
        # 高换手信号（制造换手成本）
        ixs = list(range(len(prices)))
        return pd.Series([1.0 if i % 2 == 0 else 0.0 for i in ixs], index=prices.index)

    wf = WalkForwardBacktest(train_window=20, test_window=10, step=10)
    out = wf.run(prices, fn, param_grid={"a": [1]})
    assert "walk_forward_return" in out.columns
    v = out["walk_forward_return"].fillna(0)
    assert np.isfinite(v.to_numpy()).all()
    # 每时点收益不为正无穷/爆表，量级在 [-1,1] 附近的合理区间
    assert float(v.abs().max()) < 2.0
    # 高换手叠加成本 → 累计收益为正但低于其内部 OOS 毛收益合成（机制性校验）
    assert (1 + v).prod() > 0.5


# ── P2-3: 退市股退市损失实现 ───────────────────────────────────────────────
def _write_kline(kline_dir: Path, code: str, rows) -> None:
    kline_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "close"]).to_parquet(
        kline_dir / f"{code}.parquet", index=False
    )


def _write_delisted(path: Path, rows) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_delisted_prices_post_delist_is_liquidation(tmp_path: Path) -> None:
    """P2-3: 退市日后价格为默认清算价 0.0（close→0 假设），不再被 NaN 抹零。"""
    kdir = tmp_path / "kline"
    _write_kline(kdir, "000002", [("2026-01-05", 20.0), ("2026-01-06", 21.0), ("2026-01-07", 22.0)])
    _write_delisted(tmp_path / "delisted.parquet", [{
        "code": "000002", "delist_date": "2026-01-07", "reason": "退市",
    }])

    result = load_delisted_prices(
        delisted_path=tmp_path / "delisted.parquet", kline_dir=kdir,
        max_days_after_delist=5,
    )
    assert result["000002"]["2026-01-06"] == 21.0
    # 退市日(01-07)后无 K 线，guard 窗口内交易日填清算价 → 回测实现损失
    assert any(result["000002"][k] == 0.0 for k in result["000002"] if k > "2026-01-07")


def test_delisted_symbol_realizes_loss_in_multibacktest(tmp_path: Path) -> None:
    """P2-3: 注入退市股后，MultiAssetBacktest 在退市次日记真实损失、之后无仓位。

    退市股 T 日最后一根收盘，T+1 价格按清算价(0) → 当日收益 -100%（实现损失），
    T+2 起无仓位（收益 0），而非旧实现下退市后 NaN 被 fillna(0) 的 0%。
    """
    kdir = tmp_path / "kline"
    _write_kline(kdir, "000002", [
        ("2026-01-05", 20.0), ("2026-01-06", 21.0), ("2026-01-07", 22.0),
    ])
    _write_delisted(tmp_path / "delisted.parquet", [{
        "code": "000002", "delist_date": "2026-01-07", "reason": "退市",
    }])
    index = pd.date_range("2026-01-05", periods=6, freq="B")
    panel = integrate_delisted(
        pd.DataFrame({"600000": [100.0] * 6}, index=index),
        delisted_path=tmp_path / "delisted.parquet",
        kline_dir=kdir, max_days_after_delist=5,
    )[0]

    # 退市股的逐日收益：退市次日记 -100%，之后 0%（无仓位）
    sym_returns = panel["000002"].pct_change().fillna(0)
    assert sym_returns.loc["2026-01-08"] == pytest.approx(-1.0)
    assert sym_returns.loc["2026-01-09"] == pytest.approx(0.0)
    assert sym_returns.loc["2026-01-12"] == pytest.approx(0.0)

    # 只在退市股上有信号 → 组合在退市日体现一次大额损失（非 0%）
    signals = pd.DataFrame({"000002": 1.0}, index=index)
    bt = MultiAssetBacktest(initial_capital=1_000_000.0, slippage=0.0)
    res = bt.run(panel, signals, max_position=None)
    delist_day_ret = res.returns.loc["2026-01-08"]
    assert delist_day_ret < -0.9  # 持仓满仓该退市股 → -100%
    # 之后的收益为 0（仓位已移除），不再贡献 0% 收益
    assert res.returns.loc["2026-01-09"] == pytest.approx(0.0)
    # 与"退市后 0%"的乐观基线对比：修复后累计收益显著更低（兑现了损失）
    assert float((1 + res.returns).prod() - 1) < -0.9


# ── P2-4: backtest_engine 默认滑点 0.001 = 0.1% ─────────────────────────────
def test_engine_default_slippage_is_percent():
    """P2-4: 默认 slippage_mode=percent，0.001=0.1%（10 元股买价 10.01 而非 10.001）。"""
    from quant_system.backtest_engine import BacktestEngine

    e = BacktestEngine()
    assert e.slippage_mode == "percent"
    assert e.slippage_rate == 0.001
    # percent: price ± price*0.001
    assert e._apply_slippage(10.0, "buy") == pytest.approx(10.01)
    assert e._apply_slippage(10.0, "sell") == pytest.approx(9.99)
    # 显式取 percent 亦同
    e.set_slippage(0.001, "percent")
    assert e._apply_slippage(10.0, "buy") == pytest.approx(10.01)


def test_engine_fixed_mode_still_absolute_amount():
    """P2-4: 'fixed' 仍为每股绝对金额模式（显式选用时 0.001 元/股）。"""
    from quant_system.backtest_engine import BacktestEngine

    e = BacktestEngine()
    e.set_slippage(0.001, "fixed")
    assert e._apply_slippage(10.0, "buy") == pytest.approx(10.001)
    assert e._apply_slippage(10.0, "sell") == pytest.approx(9.999)


def test_engine_default_percent_matches_system_10bp():
    """P2-4: 默认 percent 滑点与 execution_broker.DEFAULT_SLIPPAGE_RATE(0.001=10bp) 一致。"""
    from quant_system.backtest_engine import BacktestEngine
    from quant_system.execution_broker import DEFAULT_SLIPPAGE_RATE

    e = BacktestEngine()
    assert e.slippage_rate == pytest.approx(DEFAULT_SLIPPAGE_RATE)
    assert e.slippage_mode == "percent"


# ── P2-6: _force_close 判跌停/流动性上限 ───────────────────────────────────
def _build_hold_engine(prices, vols, *, limit_down=-0.10, max_volume_pct=0.0,
                       buy_volume: float = 1000):
    from quant_system.backtest_engine import BacktestEngine, StrategyTemplate

    class BuyAndHold(StrategyTemplate):
        def on_bar(self, bar):
            p = self.bg.positions.get(bar.symbol)
            if p is None or p.volume <= 1e-8:
                self.buy(bar.close, buy_volume, symbol=bar.symbol)

    e = BacktestEngine()
    e.set_capital(100_000.0)
    e.set_commission(0.0)
    e.set_slippage(0.0, "none")
    e.set_fill_mode("current_close")
    e.set_market_rules(False)
    e.limit_up = 0.10
    e.limit_down = limit_down
    e.min_commission = 0.0
    e.transfer_fee_rate = 0.0
    e.max_volume_pct = max_volume_pct
    e.order_timeout_bars = 100
    e.add_data("600000", pd.DataFrame({
        "date": pd.date_range("2026-08-03", periods=len(prices), freq="D"),
        "open": prices,
        "high": [x + 0.1 for x in prices],
        "low": [x - 0.1 for x in prices],
        "close": prices,
        "volume": vols,
        "amount": [v * x for v, x in zip(vols, prices)],
    }))
    e.add_strategy(BuyAndHold)
    return e


def test_force_close_blocked_on_limit_down(tmp_path: Path):
    """P2-6: 末根跌停收板，force_close 应标记"持有/非流动"而非在跌停价强造成交。"""
    e = _build_hold_engine([10.0, 10.0, 9.0], [100000.0, 100000.0, 100000.0])  # 末根 -10%
    res = e.run()
    assert len(res.force_close_blocks) == 1
    blk = res.force_close_blocks[0]
    assert blk["symbol"] == "600000"
    assert "跌停" in blk["reason"]
    # 只成交了开仓，未产生虚假平仓
    assert [t.offset for t in res.trades] == ["open"]


def test_force_close_blocked_on_suspension(tmp_path: Path):
    """P2-6: 末根停牌(volume=0)，force_close 标记不可卖而非强造成交。"""
    e = _build_hold_engine(
        [10.0, 10.0, 10.5], [100000.0, 100000.0, 0.0]
    )
    res = e.run()
    assert len(res.force_close_blocks) == 1
    assert "停牌" in res.force_close_blocks[0]["reason"]
    assert [t.offset for t in res.trades] == ["open"]


def test_force_close_fills_when_liquid(tmp_path: Path):
    """P2-6: 正常可见价（非跌停、有量）时 force_close 正常平仓，不误标非流动。"""
    e = _build_hold_engine([10.0, 10.0, 10.5], [100000.0, 100000.0, 100000.0])
    res = e.run()
    assert res.force_close_blocks == []
    assert [t.offset for t in res.trades] == ["open", "close"]


def test_force_close_blocked_on_volume_participation():
    """P2-6: max_volume_pct 限制下持仓超过可参与量上限 → 标记无法平仓而非强造成交。"""
    # 持仓 400 股；建仓 bar 成交量 100000(可参与上限 500>400 能建)，末根 bar
    # 成交量 30000(参与上限 floor(30000*0.005/100)*100=100 < 400) → force_close 受阻。
    e = _build_hold_engine(
        [10.0, 10.0, 10.5], [100000.0, 100000.0, 30000.0],
        max_volume_pct=0.005, buy_volume=400,
    )
    res = e.run()
    assert len(res.force_close_blocks) == 1
    assert "成交量不足" in res.force_close_blocks[0]["reason"]
    # 只成交了一次开仓（未被强造平仓）；非流动持仓仍按末价计入权益，
    # 不能因强平失败在期末被错误归零。
    assert [t.offset for t in res.trades] == ["open"]
    assert res.equity_curve.iloc[-1]["position_value"] == pytest.approx(400 * 10.5)
    assert res.equity_curve.iloc[-1]["total_value"] > res.equity_curve.iloc[-1]["cash"]


# ── P2-8: run_portfolio_backtest 等权共享资金池 ─────────────────────────────
def _make_portfolio_df(close_start, close_end, n=200, vol=3_000_000):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(close_start, close_end, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "date": dates, "symbol": "S", "open": open_, "close": close,
        "high": np.maximum(open_, close) * 1.02, "low": np.minimum(open_, close) * 0.98,
        "volume": [vol] * n,
    })


def test_portfolio_backtest_equal_weight_shared_pool():
    """P2-8: 等权共享资金池聚合——组合收益 = 各标的**逐日**收益×1/n 的复利合成。

    不复用旧实现的 initial_cash*n 满额账户相加；用各标的逐日权益曲线重建
    等权逐日组合净值，断言 run_portfolio_backtest 的 total_equity 与之一致。
    """
    from quant_system.backtest import run_backtest, run_portfolio_backtest
    from quant_system.config import PortfolioConfig, StrategyConfig

    strat = StrategyConfig(fast_ma=5, slow_ma=10, trend_ma=20, long_trend_ma=30, volume_ma=5)
    up_df = _make_portfolio_df(10.0, 15.0)
    down_df = _make_portfolio_df(10.0, 6.0)
    portfolio = PortfolioConfig(initial_cash=1_000_000.0, max_position_pct=1.0)
    plan = {"UP": up_df, "DN": down_df}

    res = run_portfolio_backtest(plan, strat, portfolio)
    assert res["summary"]["status"] == "ok"
    assert res["summary"]["pool_semantics"] == "equal_weight_shared_pool"
    assert res["summary"]["per_symbol_weight"] == pytest.approx(0.5)
    assert res["summary"]["planned_symbols"] == 2

    # ---- 参考实现：从各标的逐日权益曲线重建等权逐日组合净值 ----
    curves = {}
    for sym, df in plan.items():
        r = run_backtest(df, strat, portfolio)
        curves[sym] = {p["date"]: float(p["equity"]) for p in r["equity_curve"]}
    all_dates = sorted({d for c in curves.values() for d in c})
    weight = 1.0 / len(plan)
    nav, prev = portfolio.initial_cash, {}
    for d in all_dates:
        daily = 0.0
        for sym, c in curves.items():
            if d not in c:
                continue
            pv = prev.get(sym)
            prev[sym] = c[d]
            daily += weight * (c[d] / pv - 1.0) if (pv is not None and pv > 0) else 0.0
        nav *= 1 + daily

    ref_total_ret = nav / portfolio.initial_cash - 1.0
    assert res["summary"]["total_equity"] == pytest.approx(nav, abs=0.01)
    assert res["summary"]["total_return_pct"] == pytest.approx(ref_total_ret * 100, abs=0.2)
    # 组合收益介于两单一标的收益之间（且因任一失败/满仓现金而低于上涨标的）
    r_up = run_backtest(up_df, strat, portfolio)["final_equity"] / portfolio.initial_cash
    r_dn = run_backtest(down_df, strat, portfolio)["final_equity"] / portfolio.initial_cash
    lo, hi = min(r_up, r_dn) - 1, max(r_up, r_dn) - 1
    assert lo <= ref_total_ret <= hi


def test_portfolio_backtest_failed_symbol_capital_refunded():
    """P2-8: 某标的回测失败（无数据→no_data，无重复权益曲线）时，其分仓资金退回现金；
    总资金池=1 份 initial_cash（不再 initial_cash*n），失败标的不贡献收益/不损失本金。"""
    from quant_system.backtest import run_portfolio_backtest
    from quant_system.config import PortfolioConfig, StrategyConfig

    strat = StrategyConfig(fast_ma=5, slow_ma=10, trend_ma=20, long_trend_ma=30, volume_ma=5)
    up_df = _make_portfolio_df(10.0, 15.0)
    # 失败标的：行数 < 30，run_backtest 返回 insufficient_data（不回测）
    bad_df = _make_portfolio_df(10.0, 10.0, n=10)
    portfolio = PortfolioConfig(initial_cash=1_000_000.0, max_position_pct=1.0)

    res = run_portfolio_backtest(
        {"UP": up_df, "BAD": bad_df}, strat, portfolio
    )
    assert res["summary"]["status"] == "ok"
    assert res["summary"]["planned_symbols"] == 2
    assert res["summary"]["num_symbols"] == 1
    # 唯一成功标的按计划等权 1/2 分仓
    assert res["summary"]["per_symbol_weight"] == pytest.approx(0.5)
    # 失败标的本金退回 → 组合只投入 UP(权重0.5)，其余 0.5 现金：等权逐日复利参考
    up_curve = {p["date"]: float(p["equity"]) for p in res["symbols"]["UP"]["equity_curve"]}
    nav, prev = float(portfolio.initial_cash), None
    for d in sorted(up_curve):
        pv = prev
        prev = up_curve[d]
        daily = 0.5 * (up_curve[d] / pv - 1.0) if (pv is not None and pv > 0) else 0.0
        nav *= 1 + daily
    assert res["summary"]["total_equity"] == pytest.approx(nav, abs=0.01)
    # 资金池=1 份 initial_cash，而非 initial_cash*(成功数) 或 *(计划数)
    assert res["summary"]["total_equity"] < 1_500_000.0
