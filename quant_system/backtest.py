"""backtest.py — 信号配置式回测（单标的/多标的/网格/因子分组/组合权重）

D2收敛登记 (2026-08-11): 核心能力（StrategyConfig+signals.generate_signals
信号式回测、因子分组回测、权重函数组合回测、MC 参数扫描）backtest_engine
未覆盖，保留独立实现，标注 'D2收敛登记: 独立能力保留'；grid_search /
sensitivity_analysis 与 backtest_engine.parameter_scan 能力重叠但签名
不兼容，标注 'D1/D2 收敛: 保留独立实现（能力未合并）'。文件与公开 API
均保留（cli.py 依赖 run_backtest），不转发、不强迁。
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import PortfolioConfig, StrategyConfig
from .metrics_calculator import MetricsCalculator
from .signals import generate_signals


def _calc_trade_fees(amount: float, side: str, portfolio: PortfolioConfig) -> dict[str, float]:
    commission = max(amount * portfolio.commission_pct, portfolio.min_commission) if amount > 0 else 0.0
    stamp_tax = max(amount * portfolio.stamp_tax_pct, portfolio.min_stamp_tax) if side == "sell" and amount > 0 else 0.0
    transfer_fee = amount * getattr(portfolio, "transfer_fee_pct", 0.00001) if amount > 0 else 0.0
    return {"commission": commission, "stamp_tax": stamp_tax, "transfer_fee": transfer_fee, "total": commission + stamp_tax + transfer_fee}


def _effective_slippage(portfolio: PortfolioConfig, row: Any) -> float:
    """P1-1 (审计回测层): 根据 slippage_mode 取实际成交滑点率。

    - fixed / 任意其他: 用 portfolio.slippage_pct（默认引用 execution_broker.
      DEFAULT_SLIPPAGE_RATE=10bp）。
    - market_cap_based: 用 market_cap_tiered_slippage(market_cap)，按当日行内
      market_cap（流通市值）分档（小盘20bp / 中盘10bp / 大盘5bp）。
    """
    if getattr(portfolio, "slippage_mode", "fixed") == "market_cap_based":
        try:
            from quant_system.execution_broker import market_cap_tiered_slippage
        except ImportError:  # 独立脚本运行兜底
            from execution_broker import market_cap_tiered_slippage
        mc = row.get("market_cap")
        return market_cap_tiered_slippage(mc)
    return float(portfolio.slippage_pct)


def _safe_float(value: Any, default: float | None) -> float | None:
    """安全的 float 转换；NaN/None/异常 → default。"""
    try:
        if value is None:
            return default
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except (TypeError, ValueError):
        return default


def run_backtest(df: pd.DataFrame, strategy: StrategyConfig, portfolio: PortfolioConfig) -> dict:
    """D2收敛登记: 独立能力保留——StrategyConfig 信号式单标的回测
    （含止损止盈/T+1/涨跌停限制）backtest_engine 未覆盖（engine 为策略类模板）。"""
    data = generate_signals(df, strategy).dropna(subset=["ma_slow", "ma_trend"]).reset_index(drop=True)
    if len(data) < 30:
        return {"status": "insufficient_data", "rows": len(data)}

    # V12.3 审计 P0-1: 用 per-board 涨跌停规则(主板10/ST5/双创20/北交30/新股无限制),
    # 替代恒定 ±10% 硬编码(创业板/科创/ST/北交判错)。
    code = ""
    if "symbol" in df.columns:
        code = str(df["symbol"].iloc[0]).zfill(6)
    name_hint = ""
    if "name" in df.columns and len(df):
        name_hint = str(df["name"].iloc[0])
    try:
        from quant_system.market_rules import get_price_limit_pct as _limit_pct
        _limit = _limit_pct(code, name=name_hint) if code else 10.0
    except Exception:  # noqa: BLE001 - 规则库异常回退默认10%
        _limit = 10.0
    UP_STOP = _limit / 100.0
    DOWN_STOP = _limit / 100.0

    cash = portfolio.initial_cash
    shares = 0
    entry_price = 0.0
    entry_date = ""  # P1-Q14-fix(H02): T+1 建仓日，禁止当日卖出
    peak_equity = portfolio.initial_cash
    equity_values: list[float] = []
    equity_curve: list[dict] = []
    trades: list[dict] = []
    # P1-Q14-fix(H02): 信号在 t 收盘产生，订单植入次日开盘执行（避免同 bar
    # 收盘价成交前视：用当日收盘价算信号又按当日收盘价成交）。pending_action
    # 保存上一 bar 信号留下的待执行动作，在下一 bar 开盘价撮合。
    pending_action: str | None = None
    pending_signal_val = 0.0

    # P1-2 (审计回测层): 动态复权事件记账——从日内显式事件列推导分红/送转事件。
    # 数据层提供原始未复权价 + dividend/split 列时报此路径：分红作现金入账、
    # 送转作股数变动并重摊成本，替代 qfq 全序列缩放。qfq/hfq 复权模式无显式
    # 事件列 → events 为空 → 本步骤 no-op，现有成交/费税逻辑完全不变。
    _ca_events: dict[str, list[dict]] = {}
    try:
        from quant_system.backtest_pro import CorporateActionsHandler as _CAH
        for _ev in _CAH.events_from_price_columns(df):
            _ca_events.setdefault(str(_ev["date"])[:10], []).append(_ev)
    except Exception:  # noqa: BLE001 - 适配器失败则退回纯复权价路径，不阻断回测
        _ca_events = {}

    def _apply_corporate_actions(date_str: str) -> None:
        """对当日持仓执行分红/送转记账（P1-2）。"""
        nonlocal shares, cash, entry_price
        for _ev in _ca_events.get(date_str, []):
            if _ev["type"] == "dividend":
                _, cash = _CAH.apply_dividend_to_position(shares, cash, _ev["amount"])
                trades.append({"date": date_str, "side": "DIVIDEND",
                               "price": round(_ev["amount"], 4), "shares": shares,
                               "pnl": 0.0, "commission": 0.0, "stamp_tax": 0.0,
                               "transfer_fee": 0.0, "cash_in": round(shares * _ev["amount"], 2)})
            elif _ev["type"] == "split":
                shares, cash, entry_price = _CAH.apply_split_to_position(
                    shares, cash, entry_price, _ev["ratio"]
                )
                trades.append({"date": date_str, "side": "SPLIT",
                               "price": _ev["ratio"], "shares": shares,
                               "pnl": 0.0, "commission": 0.0, "stamp_tax": 0.0,
                               "transfer_fee": 0.0})

    for i, (_, row) in enumerate(data.iterrows()):
        price = float(row["close"])
        open_price = float(row.get("open", price))  # 次日开盘价（无 open 列则回退收盘）
        signal = int(row["signal"])
        date = str(row["date"].date())
        prev_close = float(data.iloc[i - 1]["close"]) if i > 0 else price

        # ── 1. 执行上一 bar 信号植入的订单（次日开盘价成交）──
        if pending_action == "buy" and shares == 0:
            # V12.3 P0-1: 用 per-board 涨跌停(UP_STOP)判断能否买入
            if open_price >= prev_close * (1 + UP_STOP):
                pending_action = None  # 涨停无法买入
            else:
                buy_price = open_price * (1 + _effective_slippage(portfolio, row))
                # V12.3 P0-1: 成交价 clamp 到当日 [low, high](开盘已贴涨停时, 滑点后不可越界)
                try:
                    _hi = float(row.get("high", buy_price))
                    buy_price = min(buy_price, _hi)
                except Exception:  # noqa: BLE001
                    pass
                # P1-Q14-fix(H05): 移除 signal_score 动态仓位死代码（generate_signals
                # 从不输出 signal_score，原 pos_multiplier 恒为 0.125，实际建仓仅
                # max_position_pct 的 12.5%，资金长期闲置）。现在直接按 max_position_pct 建仓。
                effective_pos_pct = portfolio.max_position_pct
                budget = min(cash, cash * effective_pos_pct)
                qty = int(budget / buy_price // 100 * 100)
                if qty > 0:
                    gross = qty * buy_price
                    fees = _calc_trade_fees(gross, "buy", portfolio)
                    cost = gross + fees["total"]
                    if cost <= cash:
                        cash -= cost
                        shares = qty
                        entry_price = buy_price
                        entry_date = date
                        trades.append({"date": date, "side": "BUY", "price": round(buy_price, 3), "shares": shares, "pnl": 0.0,
                                       "signal": pending_signal_val, "pos_pct": round(effective_pos_pct * 100, 1),
                                       "commission": round(fees["commission"], 2), "stamp_tax": 0.0,
                                       "transfer_fee": round(fees["transfer_fee"], 2)})
        elif pending_action == "sell" and shares > 0:
            # V12.3 P0-1: 用 per-board 涨跌停(DOWN_STOP)判断能否卖出
            if open_price <= prev_close * (1 - DOWN_STOP):
                pending_action = None  # 跌停无法卖出
            else:
                sell_price = open_price * (1 - _effective_slippage(portfolio, row))
                # V12.3 P0-1: 成交价 clamp 到当日 [low, high]
                try:
                    _lo = float(row.get("low", sell_price))
                    sell_price = max(sell_price, _lo)
                except Exception:  # noqa: BLE001
                    pass
                gross = shares * sell_price
                fees = _calc_trade_fees(gross, "sell", portfolio)
                proceeds = gross - fees["total"]
                pnl = proceeds - shares * entry_price
                cash += proceeds
                trades.append({"date": date, "side": "SELL", "price": round(sell_price, 3), "shares": shares, "pnl": round(pnl, 2),
                               "commission": round(fees["commission"], 2), "stamp_tax": round(fees["stamp_tax"], 2),
                               "transfer_fee": round(fees["transfer_fee"], 2)})
                shares = 0
                entry_price = 0.0
                entry_date = ""
        pending_action = None

        # ── 2. 止损/止盈（价格触发，按当日收盘价成交；T+1 当日不可卖出）──
        if shares > 0 and date != entry_date:
            stop_price = entry_price * (1 - strategy.stop_loss_pct)
            take_profit = entry_price * (1 + strategy.take_profit_pct)
            # 动态追踪止损: 每上升10%收紧止损到盈亏平衡
            if price >= entry_price * 1.1:
                # 移动止损到成本价以上
                stop_price = max(stop_price, entry_price * 0.98)
            if price >= entry_price * 1.2:
                # 锁定部分利润: 止损在-5%以内
                stop_price = max(stop_price, price * 0.95)
            if price <= stop_price or price >= take_profit:
                # P1-5 (审计回测层): 止损/止盈当日收盘触发(不引入前视)，但须判
                # 跌停不可卖 / 停牌不可成交——否则跌停日"收盘价触发却在跌停价卖
                # 出"是现实中卖不掉的模拟成交。
                day_low = _safe_float(row.get("low"), price)
                day_high = _safe_float(row.get("high"), price)
                day_vol = _safe_float(row.get("volume"), None)
                suspended = day_vol is None or day_vol <= 0 or day_low <= 0
                down_stop_px = prev_close * (1 - DOWN_STOP)
                if not suspended and price > down_stop_px:
                    # 未停牌且未跌停封板 → 可成交
                    # 审计 2026-08-16：止损/止盈卖出用 bar 内更保守的 low 作为基准，
                    # 避免“收盘确认后仍按收盘价成交”的乐观假设。
                    slippage = _effective_slippage(portfolio, row)
                    sell_price = day_low * (1 - slippage)
                    # 成交价 clamp 到当日 [low, high]，防滑点后越界
                    sell_price = float(np.clip(sell_price, day_low, day_high))
                    gross = shares * sell_price
                    fees = _calc_trade_fees(gross, "sell", portfolio)
                    proceeds = gross - fees["total"]
                    pnl = proceeds - shares * entry_price
                    cash += proceeds
                    trades.append({"date": date, "side": "SELL", "price": round(sell_price, 3), "shares": shares, "pnl": round(pnl, 2),
                                   "commission": round(fees["commission"], 2), "stamp_tax": round(fees["stamp_tax"], 2),
                                   "transfer_fee": round(fees["transfer_fee"], 2)})
                    shares = 0
                    entry_price = 0.0
                    entry_date = ""
                # 否则：跌停封板或停牌 → 撤回本次止损/止盈，持仓保留，等下一 bar 重判

        # ── 2.5 公司行为记账（P1-2）：分红入现金 / 送转调股数并重摊成本。
        # 仅当日存在显式事件列时生效；qfq/hfq 模式无事件 → no-op。
        _apply_corporate_actions(date)

        # ── 3. 当日净值快照（收盘价标记）──
        equity = cash + shares * price
        peak_equity = max(peak_equity, equity)
        equity_values.append(equity)
        equity_curve.append({"date": date, "equity": round(equity, 2)})

        # ── 4. 记录信号 → 次日开盘执行 ──
        if shares == 0 and signal > 0:
            pending_action = "buy"
            pending_signal_val = float(signal)
        elif shares > 0 and signal < 0:
            pending_action = "sell"

    final_price = float(data.iloc[-1]["close"])
    final_equity = cash + shares * final_price

    returns = pd.Series(equity_values).pct_change().dropna()
    num_days = len(equity_values)

    total_return = final_equity / portfolio.initial_cash - 1

    # Max drawdown
    max_drawdown = min(
        (eq / max(equity_values[: i + 1]) - 1 for i, eq in enumerate(equity_values)),
        default=0.0,
    )

    # Sharpe (annualized, 扣除无风险利率)
    # V12.3 审计 P0-1/P1-4: 统一口径——复用 metrics_calculator.MetricsCalculator.sharpe
    # （扣 rf=0.02、分母样本标准差 ddof=1、年化 sqrt(252)），与 backtest_pro/
    # backtest_engine/signal_backtest 同口径，跨引擎可比。
    sharpe_val = 0.0
    if len(returns) >= 2:
        sharpe_val = MetricsCalculator.sharpe(returns, rf=0.02, ddof=1)

    # Sortino (downside deviation only)
    sortino_val = 0.0
    negative_returns = returns[returns < 0]
    if not returns.empty and len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino_val = float(returns.mean() / negative_returns.std() * math.sqrt(252))

    # Calmar ratio
    calmar_val = 0.0
    if max_drawdown != 0:
        calmar_val = total_return / abs(max_drawdown)

    # Trade-level stats
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]

    total_gain = sum(t["pnl"] for t in wins)
    total_loss = sum(abs(t["pnl"]) for t in losses) if losses else 0.0

    # Profit factor
    profit_factor_val = 0.0
    if total_loss > 0:
        profit_factor_val = total_gain / total_loss

    # Average win/loss as percentage of entry cost
    def _trade_return_pct(t: dict) -> float:
        """Return percentage gained/lost on a SELL trade relative to its cost basis."""
        cost_basis = t["shares"] * t["price"] - t["pnl"]
        if cost_basis == 0:
            return 0.0
        return t["pnl"] / cost_basis * 100

    avg_win_pct = 0.0
    if wins:
        avg_win_pct = sum(_trade_return_pct(t) for t in wins) / len(wins)

    avg_loss_pct = 0.0
    if losses:
        avg_loss_pct = sum(abs(_trade_return_pct(t)) for t in losses) / len(losses)

    # Annualized return
    annual_return_pct = 0.0
    if total_return > -1 and num_days > 0:
        annual_return_pct = ((1 + total_return) ** (252 / num_days) - 1) * 100

    return {
        "status": "ok",
        "start": str(data.iloc[0]["date"].date()),
        "end": str(data.iloc[-1]["date"].date()),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(sharpe_val, 2),
        "sortino_ratio": round(sortino_val, 2),
        "calmar_ratio": round(calmar_val, 2),
        "profit_factor": round(profit_factor_val, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "trade_count": len(sells),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 2) if sells else 0.0,
        "final_equity": round(final_equity, 2),
        "open_shares": shares,
        "equity_curve": equity_curve,
        "trades": trades,
        "last_trades": trades[-8:],
    }


def run_portfolio_backtest(
    symbol_dfs: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
    portfolio: PortfolioConfig,
) -> dict:
    """Run backtest for multiple symbols and aggregate into an equal-weight portfolio.

    P2-8 (审计回测层): 原实现对每个标的各用全额 ``initial_cash`` 满仓单测，
    聚合时 ``total_invested=initial_cash*n``、``total_equity=Σfinal_equity``，
    等价于"n 个独立满额账户相加"，并非一份真实资金池下等权分仓的组合，
    "total_return" 易被误读为等权组合收益。现改为共享资金池的等权组合：
      1. 资金池 = 单份 ``initial_cash``；按计划的全部标的 ``len(symbol_dfs)``
         等权分仓，每标的占 1/N。
      2. 每日组合收益 = Σ_{active 标的} w_i * r_i(t)。其中 active 指状态为
         ok 的标的；失败标的的资金**退回现金**（计入无收益的现金仓位，不回退
         成 0 收益的隐藏损失），故其权重保留为现金、返回 0，不等于损失本金。
      3. 组合净值从 ``initial_cash`` 起按当日组合收益逐日复利。

    D2收敛登记: 独立能力保留——基于 run_backtest 的多标的等权聚合
    backtest_engine 未覆盖（engine.run_multiple_strategies 为策略类并行）。
    """
    symbol_results: dict[str, dict] = {}

    for symbol, df in symbol_dfs.items():
        result = run_backtest(df, strategy, portfolio)
        symbol_results[symbol] = result

    # 按计划的全部标的等权分仓；active=回测成功标的，其资金投入组合，
    # 失败标的资金视为退回现金（0 收益、本金不损）。
    planned = len(symbol_dfs)
    active = {s: r for s, r in symbol_results.items() if r.get("status") == "ok"}
    n_active = len(active)

    if n_active == 0:
        return {"summary": {"status": "no_data"}, "symbols": symbol_results}

    # 等权资金池分仓：每标的 notional = initial_cash / planned
    weight = 1.0 / planned if planned > 0 else 1.0

    # 每个 active 标的的逐日权益曲线 → 日收益（相对其自身该日持仓，起始=initial_cash）
    curves: dict[str, pd.Series] = {}
    for sym, r in active.items():
        pts = r.get("equity_curve", [])
        if not pts:
            continue
        s = pd.Series(
            {p["date"]: float(p["equity"]) for p in pts if p.get("equity") is not None},
            dtype=float,
        ).sort_index()
        curves[sym] = s

    if not curves:
        return {"summary": {"status": "no_data"}, "symbols": symbol_results}

    # 组合日期轴 = 全部 active 标的日期的并集，按序排列
    all_dates_sorted = sorted({d for s in curves.values() for d in s.index})

    # 逐日组合收益：Σ weight * r_i(t)，其中 r_i(t) 为标的当日收益；
    # 首日（无前一日）计 0；失败标的权重保留为现金（不贡献收益，本金不损）。
    daily_rets: dict[str, float] = {}
    prev_eq: dict[str, float] = {}
    for dt in all_dates_sorted:
        port_ret_dt = 0.0
        for sym, s in curves.items():
            if dt not in s.index:
                continue
            # 该标的当日权益及其前一可估值日
            eq_t = float(s.loc[dt])
            prev = prev_eq.get(sym)
            prev_eq[sym] = eq_t
            if prev is None or prev <= 0:
                r_i = 0.0  # 首日计入 cash，不产生收益
            else:
                r_i = eq_t / prev - 1.0
            port_ret_dt += float(weight) * r_i
        daily_rets[dt] = port_ret_dt

    # 从资金池（=1 份 initial_cash）逐日复利
    pool_invested = portfolio.initial_cash
    nav_series = pd.Series(index=all_dates_sorted, dtype=float)
    nav = pool_invested
    for dt in all_dates_sorted:
        nav *= 1 + daily_rets[dt]
        nav_series[dt] = nav

    total_ret = nav_series.iloc[-1] / pool_invested - 1

    # 组合最大回撤（基于逐日净值）
    max_dd = 0.0
    if len(nav_series) > 0:
        peak = nav_series.cummax()
        max_dd = float((nav_series / peak - 1).min())

    # Active 标的等权平均 Sharpe（仅计正数口径，与旧实现保持一致）
    sharpes = [r["sharpe"] for r in active.values() if r.get("sharpe", 0) > 0]
    weighted_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0

    return {
        "summary": {
            "status": "ok",
            "num_symbols": n_active,
            "planned_symbols": planned,
            "total_equity": round(float(nav_series.iloc[-1]), 2),
            "total_return_pct": round(total_ret * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "weighted_sharpe": round(weighted_sharpe, 2),
            # P2-8: 明确聚合语义——等权共享资金池组合（每标的 1/N 分仓）。
            # 失败标的资金退回现金（0 收益、本金不损）。
            "pool_semantics": "equal_weight_shared_pool",
            "per_symbol_weight": round(weight, 6),
        },
        "symbols": symbol_results,
    }


def grid_search(
    symbol_df: pd.DataFrame,
    strategy: StrategyConfig,
    portfolio: PortfolioConfig,
    param_grid: dict,
) -> list[dict]:
    """Grid search over strategy parameters, returning top 50 results by total_return_pct.

    D1/D2 收敛: 保留独立实现（能力未合并）——参数扫描能力与
    backtest_engine.parameter_scan 重叠，但本函数基于 StrategyConfig
    数据类逐组合重建策略（engine 为 strategy_class+data+param_grid），
    签名不兼容。
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    results: list[dict] = []

    fields = {f.name: f.type for f in type(strategy).__dataclass_fields__.values()}

    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))

        # Build a modified strategy config with the new params
        strat_kwargs = {name: getattr(strategy, name) for name in fields}
        strat_kwargs.update(params)
        modified_strategy = type(strategy)(**strat_kwargs)

        result = run_backtest(symbol_df, modified_strategy, portfolio)
        results.append({
            "params": params,
            "result": result,
        })

    # Sort by total_return_pct descending, treat non-ok as worst
    results.sort(
        key=lambda x: x["result"].get("total_return_pct", -9999),
        reverse=True,
    )

    return results[:50]


# ════════════════════════════════════════════════════════════════
# 面板与组合回测扩展
# ════════════════════════════════════════════════════════════════

def _max_drawdown(nav: pd.Series) -> float:
    """计算最大回撤。"""
    if nav.empty:
        return 0.0
    return float((nav / nav.cummax() - 1).min())


def _perf_metrics(returns: pd.Series, nav: pd.Series | None = None) -> dict[str, float]:
    """计算年化收益、波动、夏普和最大回撤。"""
    clean = returns.dropna()
    if nav is None:
        nav = (1 + clean).cumprod()
    if clean.empty or nav.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1) if nav.iloc[0] != 0 else 0.0
    years = max(len(clean) / 252, 1 / 252)
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    annual_vol = float(clean.std() * math.sqrt(252)) if clean.std() > 0 else 0.0
    # 统一所有回测引擎的Sharpe口径：扣年化无风险利率，使用样本标准差。
    sharpe = MetricsCalculator.sharpe(clean, rf=MetricsCalculator.DEFAULT_RISK_FREE_RATE,
                                      ddof=1, trading_days=252)
    return {
        "annual_return": round(float(annual_return), 4),
        "annual_vol": round(annual_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(_max_drawdown(nav), 4),
    }


def _rebalance_dates(index: pd.DatetimeIndex, rebalance: str, lookback: int = 0) -> list[pd.Timestamp]:
    """根据频率生成实际交易日再平衡日期。"""
    freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE"}
    # P2-Q14-fix(L104): "ME"/"QE" 是 pandas>=2.2 的月/季末别名，旧版（<2.2）
    # 需用 "M"/"Q"，否则 resample 抛 ValueError。按版本选择兼容写法。
    if tuple(int(x) for x in pd.__version__.split(".")[:2]) < (2, 2):
        freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "M", "quarterly": "Q"}
    rule = freq_map.get(rebalance, rebalance)
    scheduled = pd.Series(index=index, dtype=float).resample(rule).last().index
    dates: list[pd.Timestamp] = []
    for dt in scheduled:
        pos = index.searchsorted(dt)
        if pos < len(index) and pos >= lookback:
            dates.append(index[pos])
    return dates or ([index[lookback]] if len(index) > lookback else [])


class Benchmark:
    """基准收益与 Alpha/Beta 对比工具。

    D2收敛登记: 独立能力保留——基准行情获取/Alpha/Beta 对比
    backtest_engine 未覆盖。
    """

    def __init__(self, benchmark: str = "000300.SH"):
        self.benchmark = benchmark

    def fetch_returns(self, start: Any, end: Any) -> pd.Series:
        """获取基准收益；失败时返回空 Series，调用方可降级为等权基准。"""
        try:
            import akshare as ak

            symbol = self.benchmark.split(".")[0]
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            if df.empty:
                return pd.Series(dtype=float, name=self.benchmark)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].sort_values("date")
            close_col = "close" if "close" in df.columns else df.columns[-1]
            returns = df.set_index("date")[close_col].astype(float).pct_change().dropna()
            returns.name = self.benchmark
            return returns
        except Exception:
            return pd.Series(dtype=float, name=self.benchmark)

    def compute_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
        """计算 Alpha、Beta、信息比率与组合/基准夏普。"""
        aligned = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            return {"portfolio_sharpe": 0.0, "benchmark_sharpe": 0.0, "excess_sharpe": 0.0,
                    "alpha": 0.0, "beta": 0.0, "information_ratio": 0.0, "tracking_error": 0.0}
        port = aligned["portfolio"]
        bench = aligned["benchmark"]
        excess = port - bench
        beta = float(port.cov(bench) / bench.var()) if bench.var() > 0 else 0.0
        alpha = float((port.mean() - beta * bench.mean()) * 252)
        tracking_error = float(excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0
        information_ratio = float(excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0
        return {
            "portfolio_sharpe": _perf_metrics(port)["sharpe"],
            "benchmark_sharpe": _perf_metrics(bench)["sharpe"],
            "excess_sharpe": _perf_metrics(excess)["sharpe"],
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "information_ratio": round(information_ratio, 3),
            "tracking_error": round(tracking_error, 4),
        }


def factor_panel_backtest(
    factor_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_groups: int = 5,
    rebalance: str = "monthly",
    cost_bps: float = 10,
) -> dict:
    """因子分组回测：按得分分组等权持有，并输出多空与基准对比。

    D2收敛登记: 独立能力保留——因子分组/多空对比回测 backtest_engine 未覆盖。
    """
    scores, rets = factor_scores.align(forward_returns, join="inner", axis=0)
    scores, rets = scores.align(rets, join="inner", axis=1)
    scores = scores.sort_index()
    rets = rets.reindex(scores.index).fillna(0.0)
    if scores.empty or rets.empty:
        return {"status": "no_data"}

    dates = _rebalance_dates(pd.DatetimeIndex(scores.index), rebalance)
    rebalance_set = set(dates)
    # P1-Q14-fix(H04): 首个再平衡日前用初始等权持仓（而非全零权重），避免首段
    # 22 日（月频示例）收益全部丢失。
    n_assets = len(scores.columns)
    initial_weights = pd.Series(1.0 / n_assets, index=scores.columns) if n_assets > 0 else pd.Series(dtype=float)
    group_weights: dict[int, pd.Series] = {g: initial_weights.copy() for g in range(1, n_groups + 1)}
    previous_weights = {g: group_weights[g].copy() for g in group_weights}
    group_returns: dict[int, list[float]] = {g: [] for g in range(1, n_groups + 1)}
    benchmark_returns: list[float] = []

    for date in scores.index:
        day = rets.loc[date]
        # P1-Q14-fix(H04): 先按旧权重计入当日收益（含再平衡日），再换仓。
        # 原实现再平衡日仅记 -turnover*cost 后 continue，丢失当日收益且
        # benchmark 记 0，基准/超额比较双双失真。
        for g in range(1, n_groups + 1):
            group_returns[g].append(float((group_weights[g] * day).sum()))
        benchmark_returns.append(float(day.mean()))
        if date in rebalance_set:
            valid_scores = scores.loc[date].dropna().sort_values(ascending=False)
            if not valid_scores.empty:
                buckets = np.array_split(valid_scores.index.to_numpy(), n_groups)
                for g, names in enumerate(buckets, start=1):
                    weights = pd.Series(0.0, index=scores.columns)
                    if len(names) > 0:
                        weights.loc[list(names)] = 1.0 / len(names)
                    turnover = float((weights - previous_weights[g]).abs().sum() / 2)
                    previous_weights[g] = weights.copy()
                    group_weights[g] = weights
                    group_returns[g][-1] -= turnover * cost_bps / 10000

    ret_df = pd.DataFrame({f"group_{g}": group_returns[g] for g in range(1, n_groups + 1)}, index=scores.index)
    nav = (1 + ret_df).cumprod()
    long_short_returns = ret_df["group_1"] - ret_df[f"group_{n_groups}"]
    long_short_nav = (1 + long_short_returns).cumprod()
    benchmark_series = pd.Series(benchmark_returns, index=scores.index, name="benchmark")
    benchmark_nav = (1 + benchmark_series).cumprod()
    group_metrics = {col: _perf_metrics(ret_df[col], nav[col]) for col in ret_df.columns}
    bench = Benchmark()
    return {
        "status": "ok",
        "group_nav": nav,
        "long_short_nav": long_short_nav,
        "benchmark_nav": benchmark_nav,
        "group_metrics": group_metrics,
        "long_short_metrics": _perf_metrics(long_short_returns, long_short_nav),
        "benchmark_metrics": _perf_metrics(benchmark_series, benchmark_nav),
        "benchmark_comparison": bench.compute_alpha(long_short_returns, benchmark_series),
    }


def portfolio_backtest(
    returns: pd.DataFrame,
    weight_func: Callable[..., Any],
    rebalance_freq: str = "monthly",
    lookback: int = 252,
    cost_bps: float = 10,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """透明组合再平衡框架，weight_func 接收 (returns, date) 并返回权重向量。

    D2收敛登记: 独立能力保留——权重函数式组合再平衡回测
    backtest_engine 未覆盖。
    """
    data = returns.sort_index().dropna(how="all")  # 审计 2026-08-16：缺失收益保留 NaN，不填 0%
    if data.empty or len(data) <= lookback:
        return {"status": "insufficient_data"}
    symbols = data.columns.tolist()
    n_assets = len(symbols)
    current_weights = np.ones(n_assets) / n_assets
    nav = 1.0
    nav_values: list[float] = []
    port_returns: list[float] = []
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    turnover_by_date: dict[pd.Timestamp, float] = {}
    rebal_dates = set(_rebalance_dates(pd.DatetimeIndex(data.index), rebalance_freq, lookback))

    for i, date in enumerate(data.index):
        if i >= lookback and date in rebal_dates:
            hist = data.iloc[:i]
            try:
                raw_weights = weight_func(hist, date)
            except TypeError:
                raw_weights = weight_func(hist)
            if isinstance(raw_weights, pd.Series):
                target = raw_weights.reindex(symbols).fillna(0.0).values.astype(float)
            elif isinstance(raw_weights, dict):
                target = np.array([raw_weights.get(s, 0.0) for s in symbols], dtype=float)
            else:
                target = np.asarray(raw_weights, dtype=float)
            if target.shape[0] != n_assets:
                raise ValueError(f"weight_func returned {target.shape[0]} weights for {n_assets} assets")
            target = np.clip(target, 0.0, 1.0)
            target = target / target.sum() if target.sum() > 0 else np.ones(n_assets) / n_assets
            turnover = float(np.abs(target - current_weights).sum() / 2)
            cost = turnover * cost_bps / 10000
            nav *= (1 - cost)
            current_weights = target
            weights_by_date[date] = {s: round(float(w), 4) for s, w in zip(symbols, current_weights)}
            turnover_by_date[date] = round(turnover, 4)

        # 审计 2026-08-16：缺失收益资产视为不可交易——当日剔除并重新归一化剩余权重
        row = data.loc[date].values.astype(float)
        w = current_weights.copy()
        valid = ~np.isnan(row)
        if not valid.any():
            day_ret = 0.0
        else:
            w[~valid] = 0.0
            if w.sum() > 0:
                w = w / w.sum()
            day_ret = float(np.nansum(row * w))
        nav *= (1 + day_ret)
        port_returns.append(day_ret)
        nav_values.append(nav)
        drifted = current_weights * (1 + data.loc[date].values)
        current_weights = drifted / drifted.sum() if drifted.sum() > 0 else np.ones(n_assets) / n_assets

    returns_series = pd.Series(port_returns, index=data.index, name="portfolio")
    nav_series = pd.Series(nav_values, index=data.index, name="nav")
    if benchmark_returns is None:
        fetched = Benchmark().fetch_returns(data.index.min(), data.index.max())
        benchmark_returns = fetched.reindex(data.index).fillna(0.0) if not fetched.empty else data.mean(axis=1).rename("equal_weight_benchmark")
    else:
        benchmark_returns = benchmark_returns.reindex(data.index).fillna(0.0)
    comparison = Benchmark().compute_alpha(returns_series, benchmark_returns)
    return {
        "status": "ok",
        "nav": nav_series,
        "returns": returns_series,
        "weights": weights_by_date,
        "turnover": pd.Series(turnover_by_date, name="turnover"),
        "metrics": _perf_metrics(returns_series, nav_series),
        "benchmark_returns": benchmark_returns,
        "benchmark_comparison": comparison,
        "excess_returns": returns_series - benchmark_returns,
    }


def sensitivity_analysis(
    returns: pd.DataFrame,
    param_grid: dict,
    weight_func: Callable[..., Any],
    rebalance_freq: str = "monthly",
) -> pd.DataFrame:
    """对参数网格做全组合搜索，每行返回参数组合与绩效指标。

    D1/D2 收敛: 保留独立实现（能力未合并）——参数敏感性搜索能力与
    backtest_engine.parameter_scan 重叠，但本函数基于 weight_func
    回调+portfolio_backtest（engine 为 strategy_class+data），签名不兼容。
    """
    rows: list[dict[str, Any]] = []
    names = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(names, values))

        def wrapped(hist: pd.DataFrame, date: pd.Timestamp, params: dict = params) -> Any:
            return weight_func(hist, date, **params)

        result = portfolio_backtest(returns, wrapped, rebalance_freq=rebalance_freq)
        metrics = result.get("metrics", {}) if result.get("status") == "ok" else {}
        comparison = result.get("benchmark_comparison", {}) if result.get("status") == "ok" else {}
        rows.append({**params, **metrics, **comparison, "status": result.get("status")})
    df = pd.DataFrame(rows)
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        df.attrs["top5"] = df.head(5).copy()
    return df


def monte_carlo_parameter_scan(
    returns: pd.DataFrame,
    param_distributions: dict,
    weight_func: Callable[..., Any],
    n_iter: int = 100,
) -> dict:
    """随机采样参数空间，并输出参数与绩效的相关系数。

    D2收敛登记: 独立能力保留——随机参数采样+相关性分析
    backtest_engine 未覆盖（engine 无 MC 扫描）。
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for _ in range(n_iter):
        params: dict[str, Any] = {}
        for name, dist in param_distributions.items():
            try:
                if callable(dist):
                    params[name] = dist()
                elif isinstance(dist, tuple) and len(dist) == 2:
                    low, high = dist
                    params[name] = float(rng.uniform(low, high))
                else:
                    params[name] = rng.choice(list(dist)).item()
            except Exception:
                params[name] = None

        def wrapped(hist: pd.DataFrame, date: pd.Timestamp, params: dict = params) -> Any:
            return weight_func(hist, date, **params)

        result = portfolio_backtest(returns, wrapped)
        row = {**params, **(result.get("metrics", {}) if result.get("status") == "ok" else {}),
               **(result.get("benchmark_comparison", {}) if result.get("status") == "ok" else {}),
               "status": result.get("status")}
        rows.append(row)
    results = pd.DataFrame(rows)
    numeric = results.select_dtypes(include=[np.number])
    correlations = numeric.corr()[[c for c in ["sharpe", "annual_return", "max_drawdown", "information_ratio"] if c in numeric.columns]] if not numeric.empty else pd.DataFrame()
    return {"results": results, "correlations": correlations}
