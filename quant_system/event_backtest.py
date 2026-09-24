"""
event_backtest.py — V8 事件驱动回测引擎

D2收敛登记 (2026-08-11): 独立能力保留——事件驱动回测（逐 bar 信号检查→
委托→撮合→持仓更新、MA/MACD/自定义信号、三种滑点模型、涨跌停限制）
backtest_engine 未覆盖（engine 为策略模板类驱动，无事件信号 API）。
文件与公开 API 均保留，不转发、不强迁。

对比 V6/V7 的 naive 回测（点对点出信号立即成交），这个引擎:

  1. 逐根 Bar 事件驱动 — 信号检查 → 委托 → 撮合 → 持仓更新，在同一 bar 内完成
  2. 滑点模型 — 基于流动性的动态滑点: 买卖价差 + 成交量比例冲击
  3. 多标的对齐 — 按交易日历对齐，交叉信号一致性
  4. 组合级约束 — 总仓位上限、单标的上限、行业集中度
  5. 全面绩效指标 — 年化收益/波动、夏普/索提诺/卡玛、最大回撤、盈亏比、收益分布

用法:
    engine = EventBacktest(
        symbols=["002714", "600519", "000858"],
        initial_capital=1_000_000,
        slippage_model="dynamic",  # "flat" | "dynamic"
        max_positions=10,
        max_single_pct=0.15,
    )
    engine.run(start="20260101", end="20260729")
    engine.report()
    engine.plot_equity()
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

# ── V8: use DataStore for clean data ──
from quant_system.data_store import get_store
from quant_system import execution_broker as _execution_broker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TZ_CST = timedelta(hours=8)
# P2-Q14-fix(L098): 年化天数统一为 252，与 backtest.py / backtest_pro.py /
# backtest_enhancer.py 一致。原 242 是 A股 实际交易日近似值，但引擎间口径
# 不一致会导致同一策略在不同引擎的年化/夏普不可比。框架统一采用 252。
ANNUAL_TRADING_DAYS = 252

# 默认交易成本（A股）
# D3/D组收敛: 费率唯一真源为 execution_broker 常量（COMMISSION_RATE 万0.85 /
# MIN_COMMISSION 5 / STAMP_TAX_RATE 万5 / TRANSFER_FEE_RATE 万0.1，同数值同方向），
# 此处改为引用，保留本地常量名与函数签名。历史 P1-Q14-fix(H10)：原为万2.5。
COMMISSION_RATE = _execution_broker.COMMISSION_RATE      # 佣金: 万0.85
STAMP_TAX_RATE = _execution_broker.STAMP_TAX_RATE        # 卖出千0.5（减半征收后）
TRANSFER_FEE_RATE = _execution_broker.TRANSFER_FEE_RATE  # A股过户费0.001%，双边收取
MIN_COMMISSION = _execution_broker.MIN_COMMISSION        # A股最低佣金5元/笔

# 默认滑点参数（动态模型时使用）
DEFAULT_SPREAD_BPS = 2.0        # 平均买卖价差 2bp
DEFAULT_IMPACT_BPS = 5.0        # 基准冲击 5bp
VOLUME_THRESHOLD = 0.05         # 成交不超过日成交量的 5%


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """交易委托"""
    symbol: str
    date: str
    side: str              # "buy" | "sell"
    price: float           # 目标价格
    qty: int               # 股数（正数）
    slippage_bps: float    # 实际滑点（bp）
    timestamp: float       # 时间戳


@dataclass
class Fill:
    """成交记录"""
    symbol: str
    date: str
    side: str
    price: float           # 实际成交价
    qty: int
    commission: float
    tax: float
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class Position:
    """持仓"""
    symbol: str
    qty: int = 0
    avg_cost: float = 0.0
    total_bought: float = 0.0
    total_sold: float = 0.0
    realized_pnl: float = 0.0
    entry_date: str | None = None

    @property
    def market_value(self, price: float) -> float:
        return self.qty * price

    @property
    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.qty

    def buy(self, qty: int, price: float, date: str) -> None:
        total_cost = self.avg_cost * self.qty + price * qty
        self.qty += qty
        self.avg_cost = total_cost / self.qty if self.qty > 0 else 0
        self.total_bought += price * qty
        if self.entry_date is None:
            self.entry_date = date

    def sell(self, qty: int, price: float) -> float:
        """Partially close.  Returns realized PnL."""
        if qty > self.qty:
            qty = self.qty
        pnl = (price - self.avg_cost) * qty
        self.qty -= qty
        self.realized_pnl += pnl
        self.total_sold += price * qty
        if self.qty == 0:
            self.avg_cost = 0.0
            self.entry_date = None
        return pnl


@dataclass
class DailySnapshot:
    """每日快照"""
    date: str
    capital: float
    positions_value: float
    total_value: float
    positions_count: int
    trades: list[Fill] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _annualized_return(equity: pd.Series, n_days: int,
                       base_capital: float | None = None) -> float:
    """年化收益。

    P2-Q14-fix(L099): 基数统一为 initial_capital。原实现以**首个快照值**
    （可能已含首日建仓后净值）为基数，而 total_return 以 initial_capital 为
    基数 → 两者口径不一致。base_capital 缺省时回退到首个快照值以保持兼容。
    """
    if base_capital is None:
        base_capital = float(equity.iloc[0])
    if n_days < 1 or equity.iloc[-1] <= 0 or base_capital <= 0:
        return 0.0
    total_ret = equity.iloc[-1] / base_capital - 1
    years = n_days / ANNUAL_TRADING_DAYS
    return (1 + total_ret) ** (1 / max(years, 0.01)) - 1 if years > 0 else total_ret


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(dd.min())


def _sharpe_ratio(daily_returns: pd.Series, rf: float = 0.02) -> float:
    excess = daily_returns - rf / ANNUAL_TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * math.sqrt(ANNUAL_TRADING_DAYS))


def _sortino_ratio(daily_returns: pd.Series, rf: float = 0.02) -> float:
    excess = daily_returns - rf / ANNUAL_TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0 or downside is None or (isinstance(downside, float) and math.isnan(downside)):
        return 0.0
    # downside is a numpy float, handle it
    d_val = float(downside)
    if d_val == 0 or math.isnan(d_val):
        return 0.0
    return float(excess.mean() / d_val * math.sqrt(ANNUAL_TRADING_DAYS))


def _calmar_ratio(ann_ret: float, max_dd: float) -> float:
    if abs(max_dd) < 1e-10:
        return 0.0
    return ann_ret / abs(max_dd)


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------

def _flat_slippage(price: float, _volume: float, _avg_volume: float,
                   side: str = "buy") -> tuple[float, float]:
    """Flat 5bp slippage regardless of liquidity."""
    bp = 5.0 if side == "buy" else -5.0
    return price * (1 + bp / 10_000), abs(bp)


def _dynamic_slippage(price: float, volume: float, avg_volume: float,
                      side: str = "buy") -> tuple[float, float]:
    """Volume-aware dynamic slippage.

    - 如果交易量远小于日均成交量，用 spread_bp
    - 如果交易量接近日均成交量，冲击线性增长
    - 超过 VOLUME_THRESHOLD 则拒绝交易
    """
    if volume <= 0 or avg_volume <= 0:
        return _flat_slippage(price, volume, avg_volume, side)

    participation = abs(volume) / max(avg_volume, 1)
    if participation > VOLUME_THRESHOLD:
        return price * (1 + 50 / 10_000), 50.0  # 滑点上限

    spread_bp = DEFAULT_SPREAD_BPS
    impact_bp = DEFAULT_IMPACT_BPS * (participation / VOLUME_THRESHOLD)
    total_bp = spread_bp + impact_bp
    sign = 1.0 if side == "buy" else -1.0
    return price * (1 + sign * total_bp / 10_000), total_bp


def _almgren_chriss_impact(price: float, volume: float, avg_volume: float,
                            side: str = "buy") -> tuple[float, float]:
    """Almgren-Chriss 市场冲击模型 (V8⁺)。

    公式: I = a * sigma * (Q / V)^b
    - sigma: 日波动率 (假设 2%)
    - Q: 交易金额
    - V: 日均成交金额
    - a, b: 市场参数 (a=0.1, b=0.3 对A股经验值)

    对永久冲击和暂时冲击分别建模。
    """
    if volume <= 0 or avg_volume <= 0:
        return _flat_slippage(price, volume, avg_volume, side)

    # Almgren-Chriss 参数
    SIGMA_EST = 0.02  # 日波动率假设 2%
    A_PERM = 0.1      # 永久冲击系数
    B_PERM = 0.3
    A_TEMP = 0.2      # 暂时冲击系数
    B_TEMP = 0.5

    q_ratio = abs(volume) / max(avg_volume, 1)
    if q_ratio > VOLUME_THRESHOLD * 3:  # 超过15%成交量, 滑点上限
        return price * (1 + 50 / 10_000), 50.0

    # 永久冲击: I_perm = A_perm * sigma * (Q/V)^B_perm
    # 暂时冲击: I_temp = A_temp * sigma * (Q/V)^B_temp * sign
    perm_impact = A_PERM * SIGMA_EST * (q_ratio ** B_PERM)
    temp_impact = A_TEMP * SIGMA_EST * (q_ratio ** B_TEMP)

    # 总冲击 (bp): 永久冲击的一半 + 暂时冲击
    total_impact_pct = perm_impact * 0.5 + temp_impact
    total_bp = total_impact_pct * 10_000  # convert to bp

    # 加买卖价差
    total_bp = DEFAULT_SPREAD_BPS + total_bp
    total_bp = min(total_bp, 50.0)  # 上限

    sign = 1.0 if side == "buy" else -1.0
    return price * (1 + sign * total_bp / 10_000), round(total_bp, 2)


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class EventBacktest:
    """事件驱动回测引擎

    D2收敛登记: 独立能力保留——事件驱动回测 backtest_engine 未覆盖。
    """

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = 1_000_000,
        commission_rate: float = COMMISSION_RATE,
        stamp_tax_rate: float = STAMP_TAX_RATE,
        transfer_fee_rate: float = TRANSFER_FEE_RATE,
        min_commission: float = MIN_COMMISSION,
        slippage_model: str = "dynamic",
        max_positions: int = 10,
        max_single_pct: float = 0.15,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.min_commission = min_commission
        self.max_positions = max_positions
        self.max_single_pct = max_single_pct

        self.slippage_fn = {
            "flat": _flat_slippage,
            "dynamic": _dynamic_slippage,
            "almgren": _almgren_chriss_impact,
        }.get(slippage_model, _almgren_chriss_impact)

        # Engine state
        self.positions: dict[str, Position] = {}
        self.snapshots: list[DailySnapshot] = []
        self.trades: list[Fill] = []
        self._current_date: str | None = None

        # Data
        self._data: dict[str, pd.DataFrame] = {}
        self._aligned: pd.DataFrame | None = None

        # Signal function
        self.signal_fn: Callable | None = None

    def _load_data(self, start: str, end: str) -> None:
        """Load and align all symbol data."""
        store = get_store()
        raw = store.get_many(self.symbols, force_refresh=False)

        # Align to common trading calendar
        frames: dict[str, pd.Series] = {}
        self._data = raw

        for sym in self.symbols:
            if sym not in raw:
                continue
            df = raw[sym]
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            frames[sym] = df["close"]

        self._aligned = pd.DataFrame(frames)
        self._aligned = self._aligned.sort_index()
        # Filter to date range
        mask = (self._aligned.index >= start) & (self._aligned.index <= end)
        self._aligned = self._aligned[mask]
        self._aligned = self._aligned.dropna(how="all")

    # ── Signal Generators ──────────────────────────────────────────────

    def set_signal_ma_cross(self, fast: int = 5, slow: int = 20) -> None:
        """Simple MA crossover signal generator."""
        self.signal_fn = self._gen_ma_cross_signals
        self._ma_fast = fast
        self._ma_slow = slow

    def set_signal_macd(self) -> None:
        """MACD signal generator."""
        self.signal_fn = self._gen_macd_signals

    def _gen_ma_cross_signals(self, date: str, closes: dict[str, float]) -> dict[str, float]:
        """Return {symbol: signal}, where signal ∈ [-1, 0, 1].

        Q14 修复：按 ``date`` 截断历史后再计算指标，禁止使用数据集末尾
        （含未来）数据 —— 否则每一根 bar 都拿到同一信号，"逐 bar 事件驱动"失效。
        """
        target = pd.to_datetime(date)
        signals: dict[str, float] = {}
        for sym in self.symbols:
            if sym not in self._data:
                continue
            df = self._data[sym]
            # 只使用截至当日的已实现数据（无前视）
            hist = df[df["date"] <= target]
            if len(hist) < self._ma_slow + 5:
                continue
            close = hist["close"].values
            ma_fast = pd.Series(close).rolling(self._ma_fast).mean().iloc[-1]
            ma_slow = pd.Series(close).rolling(self._ma_slow).mean().iloc[-1]
            if pd.isna(ma_fast) or pd.isna(ma_slow):
                continue
            prev_fast = pd.Series(close).rolling(self._ma_fast).mean().iloc[-2]
            prev_slow = pd.Series(close).rolling(self._ma_slow).mean().iloc[-2]
            if pd.isna(prev_fast) or pd.isna(prev_slow):
                continue
            # Cross up
            if prev_fast <= prev_slow and ma_fast > ma_slow:
                signals[sym] = 1.0
            # Cross down
            elif prev_fast >= prev_slow and ma_fast < ma_slow:
                signals[sym] = -1.0
            # Momentum
            elif close[-1] > ma_fast:
                signals[sym] = 0.5
            else:
                signals[sym] = -0.5
        return signals

    def _gen_macd_signals(self, date: str, closes: dict[str, float]) -> dict[str, float]:
        """MACD signal generator.

        Q14 修复：按 ``date`` 截断历史后再计算指标（同 _gen_ma_cross_signals）。
        """
        target = pd.to_datetime(date)
        signals: dict[str, float] = {}
        for sym in self.symbols:
            if sym not in self._data:
                continue
            df = self._data[sym]
            hist = df[df["date"] <= target]
            if len(hist) < 35:
                continue
            close = hist["close"].values
            ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26
            signal_line = pd.Series(close).ewm(span=9).mean().iloc[-1]
            # Simplified: macd vs signal
            prev_macd = pd.Series(close).ewm(span=12).mean().iloc[-2] - \
                        pd.Series(close).ewm(span=26).mean().iloc[-2]
            if prev_macd <= signal_line and macd > signal_line:
                signals[sym] = 1.0
            elif prev_macd >= signal_line and macd < signal_line:
                signals[sym] = -1.0
            elif macd > signal_line:
                signals[sym] = 0.3
            else:
                signals[sym] = -0.3
        return signals

    def set_custom_signal(self, fn: Callable) -> None:
        """Set a custom signal function.

        Signature: fn(date: str, closes: dict[str, float]) -> dict[str, float]
        """
        self.signal_fn = fn

    # ── Order Execution ────────────────────────────────────────────────

    def _execute_order(self, order: Order) -> Fill | None:
        """Fill an order with slippage and commission."""
        pos = self.positions.get(order.symbol)
        df = self._data.get(order.symbol)
        if df is None:
            return None

        # 审计 2026-08-16：修复同 bar 收盘信号+收盘成交的 intrabar 前视。
        # 信号在 order.date 收盘产生，成交延迟到下一交易日开盘执行（T+1 开盘成交）。
        # 用 next bar 的 open 作为成交价；滑点量用该 bar 的成交额近似。
        df_sorted = df.sort_values("date").reset_index(drop=True)
        target_ts = pd.to_datetime(order.date)
        later = df_sorted[df_sorted["date"] > target_ts]
        if later.empty:
            return None  # 无后续 bar，无法成交（不再用当日收盘成交引入前视）
        bar = later.iloc[0]
        exec_date = str(bar["date"])[:10]
        base_price = float(bar["open"] if pd.notna(bar.get("open")) else bar["close"])
        volume = float(bar.get("volume", 1_000_000))
        # Q14 修复：avg_volume 按执行日截止截断历史，禁止用全量数据
        hist = df[df["date"] <= pd.to_datetime(exec_date)]
        avg_volume = hist["volume"].rolling(60).mean().iloc[-1] if len(hist) >= 60 else volume

        fill_price, slippage_bp = self.slippage_fn(base_price, order.qty * base_price, avg_volume * base_price, order.side)
        order = Order(
            symbol=order.symbol, date=exec_date, side=order.side,
            price=fill_price, qty=order.qty,
            slippage_bps=slippage_bp, timestamp=order.timestamp,
        )

        if order.side == "buy":
            # Check capital
            cost = order.qty * fill_price
            if cost > self.capital:
                # Adjust qty to fit capital
                adjusted_qty = int(self.capital / fill_price / 100) * 100
                if adjusted_qty <= 0:
                    return None
                order = Order(
                    symbol=order.symbol, date=order.date, side="buy",
                    price=fill_price, qty=adjusted_qty,
                    slippage_bps=slippage_bp, timestamp=order.timestamp
                )
                cost = adjusted_qty * fill_price

            commission = max(cost * self.commission_rate, self.min_commission)
            transfer_fee = cost * self.transfer_fee_rate
            total_cost = cost + commission + transfer_fee
            if total_cost > self.capital:
                return None

            self.capital -= total_cost

            if order.symbol not in self.positions:
                self.positions[order.symbol] = Position(symbol=order.symbol)
            self.positions[order.symbol].buy(order.qty, fill_price, order.date)

            return Fill(
                symbol=order.symbol, date=order.date, side="buy",
                price=round(fill_price, 3), qty=order.qty,
                commission=round(commission, 2), tax=0.0,
            )

        else:  # sell
            if pos is None or pos.qty < order.qty:
                return None

            proceeds = order.qty * fill_price
            tax = proceeds * self.stamp_tax_rate
            commission = max(proceeds * self.commission_rate, self.min_commission)
            transfer_fee = proceeds * self.transfer_fee_rate
            total_charge = tax + commission + transfer_fee

            pnl = pos.sell(order.qty, fill_price)
            self.capital += proceeds - total_charge

            return Fill(
                symbol=order.symbol, date=order.date, side="sell",
                price=round(fill_price, 3), qty=order.qty,
                commission=round(commission, 2), tax=round(tax, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / max(order.qty * fill_price, 1) * 100, 2),
            )

    # ── Bar Processing ─────────────────────────────────────────────────

    def _limit_pct(self, symbol: str) -> float:
        """按板块返回涨跌停比例（A股规则）。

        主板（60/00/001/002/003 开头）±10%；创业板（300/301）、科创板
        （688/689）±20%；北交所（8/4/92 开头）±30%。ST 股 ±5% 需个股
        标识（本引擎未提供），按板块上限保守处理。
        """
        code = str(symbol).split(".")[0]
        if code.startswith(("300", "301", "688", "689")):
            return 0.20
        if code.startswith(("8", "4", "92")):
            return 0.30
        return 0.10

    def _prev_close(self, symbol: str, date: str) -> float | None:
        """取指定 bar 的前一交易日收盘价；无前值（首日/停牌）返回 None。"""
        df = self._data.get(symbol)
        if df is None:
            return None
        bar = df[df["date"] == date]
        if bar.empty:
            return None
        try:
            pos = df.index.get_loc(bar.index[0])
            if isinstance(pos, slice):  # 重复索引标签 → 取首个位置
                pos = pos.start
            if pos is None or pos == 0:
                return None
            return float(df.iloc[pos - 1]["close"])
        except (KeyError, TypeError):
            return None

    def _process_bar(self, date: str) -> None:
        """Process one trading day (one bar)."""
        if self._aligned is None:
            return

        closes: dict[str, float] = {}
        for sym in self.symbols:
            if sym in self._data:
                row = self._data[sym]
                bar = row[row["date"] == date]
                if not bar.empty:
                    closes[sym] = float(bar.iloc[0]["close"])

        if not closes:
            return

        # 1. Check existing signals via signal_fn
        if self.signal_fn is None:
            return

        signals = self.signal_fn(date, closes)
        if not signals:
            return

        today_trades: list[Fill] = []
        # P2-Q14-fix(L100): 订单时间戳用 bar 日期构造（收盘 15:00），不再用
        # datetime.now()（原实现所有订单同一时间戳，且为运行时刻而非 bar 时刻）。
        date_dt = pd.Timestamp(date + " 15:00:00").timestamp()

        # 2. Process sell signals (先卖后买)
        for sym in self.symbols:
            sig = signals.get(sym, 0)
            pos = self.positions.get(sym)
            if sig <= 0 and pos is not None and pos.qty > 0:
                price = closes.get(sym, 0)
                if price <= 0:
                    continue
                # P2-Q14-fix(M093): 卖出端补跌停检查——跌停封板无法卖出，
                # 按板块涨跌停比例对称处理（原仅买入端有近似涨停检查）。
                # 阈值加 1e-3 容差：浮点下 1.0-0.1=0.9 精确边界不可靠，且
                # 与买入端原 1.099（≈1.10-0.001）"接近涨停即禁"的口径一致。
                prev_close = self._prev_close(sym, date)
                if prev_close is not None and price <= prev_close * (1 - self._limit_pct(sym) + 1e-3):
                    continue
                # Full exit
                order = Order(
                    symbol=sym, date=date, side="sell",
                    price=price,
                    qty=pos.qty,
                    slippage_bps=0,
                    timestamp=date_dt,
                )
                fill = self._execute_order(order)
                if fill:
                    today_trades.append(fill)
                    self.trades.append(fill)

        # P2-Q14-fix(L101): 删除未使用的 current_exposure 死代码（原计算
        # 结果从未被使用，也不影响组合约束）。

        # 3. Process buy signals
        active_positions = sum(1 for p in self.positions.values() if p.qty > 0)
        for sym in self.symbols:
            sig = signals.get(sym, 0)
            if sig > 0:  # buy signal
                if active_positions >= self.max_positions:
                    continue
                price = closes.get(sym, 0)
                if price <= 0:
                    continue

                # Position sizing (动态: 基于当前现金)
                max_invest = self.capital * self.max_single_pct
                current_pos = self.positions.get(sym)
                if current_pos and current_pos.qty > 0:
                    continue  # already hold
                qty = int(max_invest / price / 100) * 100
                if qty <= 0:
                    continue

                # P2-Q14-fix(M093): 涨跌停检查改为按板块比例（10%/20%/30%），
                # 原固定 1.099 近似只覆盖主板 10% 且不含科创板/创业板/北交所。
                # 阈值减 1e-3 容差（浮点精确边界不可靠，1.099≈1.10-0.001），
                # 涨停或接近涨停日禁止开多仓。
                prev_close = self._prev_close(sym, date)
                if prev_close is not None and price >= prev_close * (1 + self._limit_pct(sym) - 1e-3):
                    continue

                order = Order(
                    symbol=sym, date=date, side="buy",
                    price=price, qty=qty,
                    slippage_bps=0,
                    timestamp=date_dt,
                )
                fill = self._execute_order(order)
                if fill:
                    today_trades.append(fill)
                    self.trades.append(fill)
                    active_positions += 1

        # 4. Take snapshot
        positions_value = sum(
            p.qty * closes.get(sym, 0)
            for sym, p in self.positions.items()
            if p.qty > 0
        )
        total_value = self.capital + positions_value

        self.snapshots.append(DailySnapshot(
            date=date,
            capital=round(self.capital, 2),
            positions_value=round(positions_value, 2),
            total_value=round(total_value, 2),
            positions_count=active_positions,
            trades=today_trades,
        ))

    # ── Run ────────────────────────────────────────────────────────────

    def run(
        self,
        start: str = "20260101",
        end: str | None = None,
    ) -> dict[str, Any]:
        """Run the backtest.  Returns performance report."""
        if end is None:
            end = datetime.now().strftime("%Y%m%d")

        # Normalize dates
        s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
        e = f"{end[:4]}-{end[4:6]}-{end[6:]}"

        t0 = _time.time()
        self._load_data(s, e)
        load_time = _time.time() - t0

        if self._aligned is None or self._aligned.empty:
            return {"status": "error", "message": "No data loaded"}

        # Process each bar
        print(f"[EventBacktest] {len(self._aligned)} bars, {len(self.symbols)} symbols")
        bar_t0 = _time.time()
        for dt_idx in self._aligned.index:
            date_str = dt_idx.strftime("%Y-%m-%d")
            self._current_date = date_str
            self._process_bar(date_str)
        bar_time = _time.time() - bar_t0

        # Build final report
        report = self._build_report()
        report["timing"] = {
            "data_load_s": round(load_time, 1),
            "bar_processing_s": round(bar_time, 1),
            "total_s": round(_time.time() - t0, 1),
            "bars": len(self._aligned),
        }
        return report

    # ── Report ─────────────────────────────────────────────────────────

    def _build_report(self) -> dict[str, Any]:
        if not self.snapshots:
            return {"status": "no_trades"}

        equity = pd.Series([s.total_value for s in self.snapshots])
        dates = [s.date for s in self.snapshots]
        daily_returns = equity.pct_change().dropna()
        num_days = len(daily_returns)

        # Basic metrics
        final_value = equity.iloc[-1]
        total_return = final_value / self.initial_capital - 1
        ann_ret = _annualized_return(equity, num_days, base_capital=self.initial_capital)
        max_dd = _max_drawdown(equity)
        sharpe = _sharpe_ratio(daily_returns)
        sortino = _sortino_ratio(daily_returns)
        calmar = _calmar_ratio(ann_ret, max_dd)

        # P2-Q14-fix(M091): 原变量 buy_trades 实为卖出单（side=="sell"），
        # 名为 buy 实为 sell，报告 summary.buy_trades 字段输出的是卖出笔数且
        # 卖出笔数无独立字段。改名为 sells，并补真实的 buys 独立统计。
        sells = [t for t in self.trades if t.side == "sell"]
        buys = [t for t in self.trades if t.side == "buy"]
        win_trades = [t for t in sells if t.pnl > 0]
        loss_trades = [t for t in sells if t.pnl <= 0]
        win_rate = len(win_trades) / max(len(sells), 1)

        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in loss_trades])) if loss_trades else 0
        # P2-Q14-fix(M092): avg_loss 单位是元，原 max(avg_loss, 1) 把 <1 元的
        # 平均亏损强行抬到 1 → 小资金/小仓位时 PF 虚高。改用 1e-9 仅防除零。
        profit_factor = avg_win / max(avg_loss, 1e-9) if loss_trades else float("inf")

        # Total fees
        total_commission = sum(t.commission for t in self.trades)
        total_tax = sum(t.tax for t in self.trades)

        return {
            "status": "ok",
            "parameters": {
                "symbols": self.symbols,
                "initial_capital": self.initial_capital,
                "max_positions": self.max_positions,
                "max_single_pct": self.max_single_pct,
                "commission_rate": self.commission_rate,
                "slippage_model": self.slippage_fn.__name__,
            },
            "summary": {
                "start_date": dates[0],
                "end_date": dates[-1],
                "trading_days": num_days,
                "total_trades": len(self.trades),
                "buy_trades": len(buys),
                "sell_trades": len(sells),
                "win_trades": len(win_trades),
                "loss_trades": len(loss_trades),
            },
            "performance": {
                "initial_value": self.initial_capital,
                "final_value": round(final_value, 2),
                "total_return_pct": round(total_return * 100, 2),
                "annualized_return_pct": round(ann_ret * 100, 2),
                "annualized_volatility_pct": round(
                    float(daily_returns.std() * math.sqrt(ANNUAL_TRADING_DAYS) * 100), 2
                ),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3),
                "calmar_ratio": round(calmar, 3),
                "win_rate_pct": round(win_rate * 100, 1),
                "avg_win": round(float(avg_win), 2),
                "avg_loss": round(float(avg_loss), 2),
                "profit_factor": round(float(profit_factor), 2),
                "total_commission": round(total_commission, 2),
                "total_tax": round(total_tax, 2),
            },
            "equity_curve": [
                {"date": dates[i], "equity": round(float(equity.iloc[i]), 2)}
                for i in range(len(dates))
            ],
            "trades": [
                {
                    "date": t.date,
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": t.price,
                    "qty": t.qty,
                    "commission": t.commission,
                    "tax": t.tax,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in self.trades
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo
    engine = EventBacktest(
        symbols=["002714", "600519", "000858", "601899", "002594"],
        initial_capital=1_000_000,
        slippage_model="dynamic",
        max_positions=5,
        max_single_pct=0.20,
    )
    engine.set_signal_ma_cross(fast=5, slow=20)
    report = engine.run(start="20260101")

    print("\n" + "=" * 60)
    print("  V8 事件驱动回测结果")
    print("=" * 60)
    p = report.get("performance", {})
    print(f"  总收益:   {p.get('total_return_pct', 0):>8.2f}%")
    print(f"  年化收益: {p.get('annualized_return_pct', 0):>8.2f}%")
    print(f"  年化波动: {p.get('annualized_volatility_pct', 0):>8.2f}%")
    print(f"  最大回撤: {p.get('max_drawdown_pct', 0):>8.2f}%")
    print(f"  夏普比:   {p.get('sharpe_ratio', 0):>8.3f}")
    print(f"  索提诺:   {p.get('sortino_ratio', 0):>8.3f}")
    print(f"  卡玛比:   {p.get('calmar_ratio', 0):>8.3f}")
    print(f"  胜率:     {p.get('win_rate_pct', 0):>8.1f}%")
    print(f"  盈亏比:   {p.get('profit_factor', 0):>8.2f}")
    print(f"  交易次数: {report.get('summary', {}).get('total_trades', 0)}")
    print(f"  耗时:     {report.get('timing', {}).get('total_s', 0):.1f}s")
