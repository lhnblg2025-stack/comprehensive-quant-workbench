"""Optional Backtrader cross-engine adapter for target-weight A-share replay."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest_protocol import UnifiedBacktestResult


class AShareCommission:
    """Factory for Backtrader commission scheme with sell-side stamp duty."""

    @staticmethod
    def create(commission_bps: float = .85, stamp_duty_bps: float = 5.0, transfer_fee_bps: float = .1, min_commission: float = 5.0):
        import backtrader as bt

        class _Commission(bt.CommInfoBase):
            params = dict(commission=commission_bps / 10000.0, stamp=stamp_duty_bps / 10000.0, transfer=transfer_fee_bps / 10000.0, min_comm=min_commission, stocklike=True, commtype=bt.CommInfoBase.COMM_PERC)

            def _getcommission(self, size, price, pseudoexec):
                notional = abs(size) * price
                commission = max(self.p.min_comm, notional * self.p.commission)
                transfer = notional * self.p.transfer
                stamp = notional * self.p.stamp if size < 0 else 0.0
                return commission + transfer + stamp

        return _Commission()


def _annual_metrics(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(values) < 2:
        return None, None, None
    returns = values.pct_change().dropna()
    years = len(returns) / 252.0
    total = values.iloc[-1] / values.iloc[0] - 1.0
    annual = (1 + total) ** (1 / years) - 1 if years > 0 and 1 + total > 0 else None
    std = returns.std(ddof=1)
    sharpe = returns.mean() / std * math.sqrt(252) if std > 0 else None
    drawdown = float((values / values.cummax() - 1).min())
    return annual, float(sharpe) if sharpe is not None else None, drawdown


def run_backtrader_targets(panel: pd.DataFrame, targets: Mapping[object, Mapping[str, float]], *, capital: float = 1_000_000.0, commission_bps: float = .85, stamp_duty_bps: float = 5.0, transfer_fee_bps: float = .1, slippage_bps: float = 10.0, lot_size: int = 100) -> tuple[UnifiedBacktestResult, pd.DataFrame]:
    """Replay target weights using next-open Backtrader market orders.

    Backtrader is used as an independent engine. Historical suspension/limit
    rules still require authoritative source fields and are checked before order
    submission when present in the panel.
    """
    import backtrader as bt

    required = {"date", "code", "raw_open", "raw_high", "raw_low", "raw_close", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"backtrader_panel_missing:{','.join(missing)}")
    normalized_targets = {pd.Timestamp(day).date(): {str(code).zfill(6): float(weight) for code, weight in values.items()} for day, values in targets.items()}
    state = panel.copy(); state["date"] = pd.to_datetime(state["date"]); state["code"] = state["code"].astype(str).str.zfill(6)
    states = {(row.code, row.date.date()): row for row in state.itertuples(index=False)}

    class _Pandas(bt.feeds.PandasData):
        params = (("datetime", None), ("open", "raw_open"), ("high", "raw_high"), ("low", "raw_low"), ("close", "raw_close"), ("volume", "volume"), ("openinterest", -1))

    class _TargetStrategy(bt.Strategy):
        params = dict(targets=normalized_targets, lot_size=lot_size, max_participation=0.10, slippage=slippage_bps)

        def __init__(self):
            self.pending = None; self.equity = []; self.rejected = 0; self.completed = 0; self.total_commission = 0.0

        def _submit_pending(self, day):
            if self.pending is None:
                return
            target = self.pending
            total = float(self.broker.getvalue())
            sells: list[tuple] = []
            buys: list[tuple] = []
            for data in self.datas:
                code = data._name; row = states.get((code, day)); weight = target.get(code, 0.0)
                blocked_buy = bool(getattr(row, "suspended", False) or getattr(row, "limit_up_locked", False)) if row is not None else True
                blocked_sell = bool(getattr(row, "suspended", False) or getattr(row, "limit_down_locked", False)) if row is not None else True
                current = self.getposition(data).size
                open_price = float(data.open[0]) if len(data) else 0.0
                desired = int(total * max(weight, 0.0) / max(open_price, 1e-12) // self.p.lot_size * self.p.lot_size)
                if row is not None and desired > current:
                    adv_amount = float(getattr(row, "adv_amount_20", None) or getattr(row, "amount", 0.0) or 0.0)
                    if np.isfinite(adv_amount) and adv_amount > 0:
                        cap_shares = int(max(0.0, adv_amount * self.p.max_participation) / max(open_price, 1e-12) // self.p.lot_size * self.p.lot_size)
                        desired = min(desired, cap_shares)
                delta = desired - current
                if delta < 0 and not blocked_sell:
                    sells.append((data, -delta))
                elif delta < 0:
                    self.rejected += 1
                elif delta > 0 and not blocked_buy:
                    buys.append((data, delta))
                elif delta > 0:
                    self.rejected += 1
            # Free cash first so buys cannot hit Margin before sells settle.
            for data, shares in sells:
                self.sell(data=data, size=shares)
            cash = float(self.broker.getcash())
            for data, shares in buys:
                price = float(data.open[0]) * (1.0 + self.p.slippage / 10000.0)
                while shares > 0:
                    notional = shares * price
                    fee = max(5.0, notional * 0.000085) + notional * 0.000001
                    if notional + fee <= cash + 1e-8:
                        break
                    shares -= self.p.lot_size
                if shares > 0:
                    self.buy(data=data, size=shares)
                    cash -= shares * price + (max(5.0, shares * price * 0.000085) + shares * price * 0.000001)
            self.pending = None

        def next_open(self):
            # cheat_on_open makes this the current t+1 open, matching the
            # portfolio engine's signal-at-close -> next-open contract.
            day = self.datas[0].datetime.date(0)
            self._submit_pending(day)

        def next(self):
            day = self.datas[0].datetime.date(0)
            self.equity.append({"date": pd.Timestamp(day), "total_value": float(self.broker.getvalue())})
            if day in self.p.targets:
                self.pending = self.p.targets[day]

        def notify_order(self, order):
            if order.status == order.Completed:
                self.completed += 1
                self.total_commission += float(order.executed.comm or 0.0)
            elif order.status in (order.Canceled, order.Margin, order.Rejected):
                self.rejected += 1

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.broker.setcash(capital)
    cerebro.broker.addcommissioninfo(AShareCommission.create(commission_bps, stamp_duty_bps, transfer_fee_bps))
    cerebro.broker.set_slippage_perc(slippage_bps / 10000.0)
    for code, frame in state.groupby("code", sort=True):
        feed = frame.set_index("date").sort_index()
        cerebro.adddata(_Pandas(dataname=feed), name=str(code))
    cerebro.addstrategy(_TargetStrategy)
    strategy = cerebro.run(runonce=False, preload=True)[0]
    equity = pd.DataFrame(strategy.equity).drop_duplicates("date").sort_values("date")
    final_value = float(cerebro.broker.getvalue())
    annual, sharpe, drawdown = _annual_metrics(equity.total_value) if not equity.empty else (None, None, None)
    result = UnifiedBacktestResult(engine="backtrader", status="complete" if strategy.rejected == 0 else "completed_with_rejections", initial_capital=capital, final_value=final_value, total_return=final_value / capital - 1.0, annual_return=annual, sharpe=sharpe, max_drawdown=drawdown, observations=len(equity), orders=strategy.completed + strategy.rejected, trades=strategy.completed, rejected_orders=strategy.rejected, total_fees=strategy.total_commission, metadata={"fill_timing": "next_open", "lot_size": lot_size, "target_dates": len(normalized_targets)})
    return result, equity
