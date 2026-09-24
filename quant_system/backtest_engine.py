#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件驱动回测引擎 (Event-Driven Backtesting Engine)

D2收敛登记 (2026-08-11): 回测域唯一真源。最近 4 项语义修复
（参数扫描排序/整手拆分/停牌K线/分钟过滤）均落于此模块并被 tests/ 覆盖；
其余回测模块（backtest_pro/event_backtest/backtest/combined_backtest/
backtest_enhancer）只做收敛标注登记，不强迁高风险逻辑。

支持多标的、多策略、多订单类型（market/limit/stop），
包含完整的绩效指标计算、滑点模型、涨跌停限制、并行参数扫描和Walk-Forward回测。
"""

from __future__ import annotations

import math
import os
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from .metrics_calculator import MetricsCalculator
from .execution_broker import COMMISSION_RATE, MIN_COMMISSION, TRANSFER_FEE_RATE, DEFAULT_SLIPPAGE_RATE

# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass
class BarData:
    """K线数据事件"""

    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0

    def __post_init__(self) -> None:
        for attr in ("open", "high", "low", "close", "volume", "amount"):
            val = getattr(self, attr)
            if np.isnan(val) or np.isinf(val):
                raise ValueError(
                    f"{self.symbol} @ {self.datetime}: {attr}={val} is invalid"
                )


@dataclass
class OrderData:
    """订单数据"""

    symbol: str
    order_id: str
    direction: str  # "long" / "short"
    offset: str  # "open" / "close"
    price: float
    volume: float
    filled: float = 0.0
    status: str = "submitted"  # submitted / filled / cancelled / rejected
    datetime: Optional[datetime] = None
    order_type: str = "market"  # market / limit / stop
    strategy_name: str = ""
    pending_bars: int = 0  # P2-Q13-fix(M7): 待成交累计 bar 数，用于超时撤销


@dataclass
class TradeData:
    """成交数据"""

    symbol: str
    trade_id: str
    order_id: str
    direction: str
    offset: str
    price: float
    volume: float
    datetime: datetime


@dataclass
class PositionData:
    """持仓数据"""

    symbol: str
    direction: str = "net"  # "long" / "short" / "net"
    volume: float = 0.0
    frozen: float = 0.0
    price: float = 0.0  # 持仓均价
    pnl: float = 0.0
    pnl_ratio: float = 0.0
    # P2-6 (审计回测层): 末次强制平仓被涨跌停/流动性上限/停牌阻止时置 True，
    # 标记该持仓为"持有/非流动"，而非在现实中卖不掉的价位强造成交。
    force_close_blocked: bool = False
    force_close_blocked_reason: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "net"):
            raise ValueError(
                f"direction must be long/short/net, got {self.direction}"
            )


@dataclass
class BacktestResult:
    """回测结果"""

    equity_curve: pd.DataFrame  # [datetime, total_value, cash, position_value, returns]
    trades: List[TradeData]
    orders: List[OrderData]
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    # P2-6 (审计回测层): 回测结束强制平仓被阻止的持仓清单
    # [{symbol, reason, datetime}]，供调用方辨认"持有/非流动"持仓，而非虚假成交。
    force_close_blocks: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 策略基类
# ---------------------------------------------------------------------------


class StrategyTemplate:
    """策略基类——用户策略继承此类并覆写回调方法。"""

    def __init__(self) -> None:
        self.name: str = ""
        self.bg: Optional[BacktestEngine] = None  # BacktestEngine reference

    def on_init(self) -> None:
        """策略初始化（仅调用一次）。"""
        pass

    def on_start(self) -> None:
        """回测开始前调用（仅调用一次）。"""
        pass

    def on_bar(self, bar: BarData) -> None:
        """每根 K 线回调（必须覆写）。"""
        raise NotImplementedError("on_bar must be overridden by subclass")

    def on_trade(self, trade: TradeData) -> None:
        """每笔成交回调。"""
        pass

    def on_order(self, order: OrderData) -> None:
        """订单状态变更回调。"""
        pass

    # ---- 便捷下单接口 ----

    def buy(
        self,
        price: float,
        volume: float,
        order_type: str = "market",
        symbol: str = "",
    ) -> Optional[str]:
        """开多仓 / 平空仓（long open + long close = buy/sell）
        此处语义：long + open → 买入开多

        Parameters
        ----------
        symbol : str, optional
            标的代码。为空时由成交引擎按当前 K 线补填（单标的兼容旧行为）；
            多标的策略务必显式传入，避免订单被错误盖章到其他标的。
        """
        if self.bg is None:
            return None
        return self.bg.send_order(
            self.name, "long", "open", price, volume, order_type, symbol=symbol
        )

    def sell(
        self,
        price: float,
        volume: float,
        order_type: str = "market",
        symbol: str = "",
    ) -> Optional[str]:
        """平多仓（long close）

        Parameters
        ----------
        symbol : str, optional
            标的代码。为空时由成交引擎按当前 K 线补填（单标的兼容旧行为）。
        """
        if self.bg is None:
            return None
        return self.bg.send_order(
            self.name, "short", "close", price, volume, order_type, symbol=symbol
        )

    def short(
        self,
        price: float,
        volume: float,
        order_type: str = "market",
        symbol: str = "",
    ) -> Optional[str]:
        """开空仓（short open）

        Parameters
        ----------
        symbol : str, optional
            标的代码。为空时由成交引擎按当前 K 线补填（单标的兼容旧行为）。
        """
        if self.bg is None:
            return None
        return self.bg.send_order(
            self.name, "short", "open", price, volume, order_type, symbol=symbol
        )

    def cover(
        self,
        price: float,
        volume: float,
        order_type: str = "market",
        symbol: str = "",
    ) -> Optional[str]:
        """平空仓（short close）

        Parameters
        ----------
        symbol : str, optional
            标的代码。为空时由成交引擎按当前 K 线补填（单标的兼容旧行为）。
        """
        if self.bg is None:
            return None
        return self.bg.send_order(
            self.name, "long", "close", price, volume, order_type, symbol=symbol
        )


# ---------------------------------------------------------------------------
# 事件驱动回测引擎
# ---------------------------------------------------------------------------


class BacktestEngine:
    """事件驱动回测引擎。

    核心流程：
        1. 注册标的 K 线数据
        2. 注册策略实例
        3. 按时间戳遍历所有标的同时刻 K 线
             a. 触发策略 on_bar
             b. 处理所有未成交订单
             c. 更新浮动盈亏 & 权益曲线
        4. 计算最终绩效指标
    """

    TRADE_MODES = ("long", "short", "both")
    SLIPPAGE_MODES = ("fixed", "percent", "none")

    def __init__(self) -> None:
        self.strategies: Dict[str, StrategyTemplate] = {}
        self.bars: Dict[str, List[BarData]] = {}
        self.positions: Dict[str, PositionData] = {}
        self.orders: List[OrderData] = []
        self.trades: List[TradeData] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_dt: Optional[datetime] = None
        self._order_counter: int = 0
        self._trade_counter: int = 0

        # 可配置参数
        self.initial_capital: float = 1_000_000.0
        self.capital: float = 1_000_000.0
        # E2 (审计 2026-08-20): 费率引用执行域唯一真源 execution_broker 常量；
        # 调用方仍可通过 set_commission/set_slippage 覆盖默认值。
        self.commission_rate: float = COMMISSION_RATE  # 万0.85
        self.slippage_rate: float = DEFAULT_SLIPPAGE_RATE  # 千1（0.1%）
        # P2-4 (审计回测层): 默认滑点模式由 fixed 改为 percent，使
        # slippage_rate=0.001 语义为 0.1%（price ± price*0.001），与 config/
        # execution_broker.DEFAULT_SLIPPAGE_RATE=0.001 的百分比口径对齐。
        # 原 fixed 把 0.001 当"每股绝对 0.001 元"，对 10 元股价有效滑点仅 0.01%，
        # 被系统性低估一个量级。fixed 仍保留为显式"每股绝对元"模式供调用方选用。
        self.slippage_mode: str = "percent"  # percent(比例) / fixed(每股绝对金额) / none
        self.trade_mode: str = "both"  # long / short / both
        self.size: int = 100  # 每手100股
        self.min_commission: float = MIN_COMMISSION  # P1-Q13-fix(H1): 最低佣金 5 元/笔（A股标准）
        self.transfer_fee_rate: float = TRANSFER_FEE_RATE  # P1-Q13-fix(H2): 过户费 0.001% 双边收取
        self.limit_up: float = 0.10  # 涨停10% (回退默认，见 enable_market_rules)
        self.limit_down: float = -0.10  # 跌停10% (回退默认，见 enable_market_rules)

        # 基准序列
        self.benchmark_returns: Optional[pd.Series] = None

        # A股前收盘价追踪（用于涨跌停计算）
        self._prev_close: Dict[str, float] = {}

        # M3 fix(P2-Q13-M072): 成交时点模式——current_close(信号bar收盘成交) / next_open(次bar开盘成交)
        self.fill_mode: str = "next_open"

        # M4 fix(P2-Q13-M073): 流动性约束——单bar成交量占比上限(0=不限制) 与 冲击成本系数(平方根模型,0=关闭)
        self.max_volume_pct: float = 0.05
        self.impact_coefficient: float = 0.0

        # M7 fix(P2-Q13-M076): 订单超时撤销（bar 数，0=关闭）；按 symbol 索引的待成交订单桶
        self.order_timeout_bars: int = 5
        self._pending_by_symbol: Dict[str, List[OrderData]] = {}

        # L4 fix(P2-Q13-L083): 年化因子（每年 bar 数）。None=按 bar 频率自动推断，可显式配置
        self.annualization_factor: Optional[float] = None

        # P1-Q13-fix(H4): T+1 上一处理 bar 的日期，用于每日解冻持仓
        self._last_bar_date: Optional[date] = None
        self.end_mode: str = "force_close"  # force_close / mark_to_market
        self.corporate_actions: Dict[date, List[Dict[str, Any]]] = {}
        self.applied_corporate_actions: List[Dict[str, Any]] = []

        # 2026-08-01: 涨跌停规则接入 market_rules（按板块自动识别，支持ST/新股）
        self.symbol_names: Dict[str, str] = {}  # 代码→简称，用于ST判断
        self.enable_market_rules: bool = True   # 关闭则回退到 limit_up/limit_down 固定值

    # ---- 配置 ----

    def set_capital(self, capital: float) -> None:
        """设置初始资金。"""
        if capital <= 0:
            raise ValueError(f"capital must be > 0, got {capital}")
        self.initial_capital = capital
        self.capital = capital

    def set_commission(self, rate: float) -> None:
        """设置佣金费率。"""
        if not 0 <= rate < 1:
            raise ValueError(f"commission rate must be in [0,1), got {rate}")
        self.commission_rate = rate

    def set_slippage(self, rate: float, mode: str = "percent") -> None:
        """设置滑点。默认 mode=percent（0.001=0.1%），与全系统口径一致。"""
        if mode not in self.SLIPPAGE_MODES:
            raise ValueError(
                f"slippage mode must be {self.SLIPPAGE_MODES}, got {mode}"
            )
        if not 0 <= rate < 1:
            raise ValueError(f"slippage rate must be in [0,1), got {rate}")
        self.slippage_rate = rate
        self.slippage_mode = mode

    def set_fill_mode(self, mode: str) -> None:
        """P2-Q13-fix(M3): 设置成交时点模式。

        V12.3 审计 P2-5: ``current_close`` 模式存在 **intrabar 前视**——策略
        ``on_bar`` 已看到完整 bar（含 high/low/close），同日 market 单按 bar.close
        成交、limit/stop 又用当日 high/low 触发，信号输入与成交用同一根 bar 的
        已见区间。该模式**仅用于事后诊断/复现，不得用于实盘校准或参数选择**；
        ``next_open``（默认）为干净口径。调用方选用 current_close 时应自行知悉
        前视风险。

        Parameters
        ----------
        mode : str
            "current_close" — 信号 bar 收盘价成交（旧行为，乐观；仅诊断用）
            "next_open"     — 信号 bar 收盘后、下一根 bar 开盘价成交（稳健，默认）
        """
        if mode not in ("current_close", "next_open"):
            raise ValueError(
                f"fill_mode must be 'current_close'/'next_open', got {mode}"
            )
        if mode == "current_close":
            import warnings as _w
            _w.warn(
                "fill_mode='current_close' 存在intrabar前视(信号与成交同bar已见区间)；"
                "仅限事后诊断，勿用于实盘校准/参数选择。推荐 next_open。",
                UserWarning,
            )
        self.fill_mode = mode

    def set_end_mode(self, mode: str) -> None:
        if mode not in ("force_close", "mark_to_market"):
            raise ValueError("end_mode must be force_close or mark_to_market")
        self.end_mode = mode

    def add_corporate_actions(self, actions: pd.DataFrame) -> None:
        required = {"date", "symbol", "action_type"}
        missing = required - set(actions.columns)
        if missing:
            raise ValueError(f"corporate actions missing columns: {sorted(missing)}")
        self.corporate_actions.clear()
        for row in actions.to_dict("records"):
            action_date = pd.Timestamp(row["date"]).date()
            self.corporate_actions.setdefault(action_date, []).append(row)

    def set_benchmark(self, returns: pd.Series) -> None:
        """设置基准收益率序列（用于计算 alpha/beta/information ratio）。"""
        if not isinstance(returns, pd.Series):
            raise TypeError("benchmark returns must be a pd.Series")
        self.benchmark_returns = returns.copy()

    def set_symbol_name(self, symbol: str, name: str) -> None:
        """注册证券简称（用于涨跌停规则中的 ST/*ST 判断）。"""
        self.symbol_names[symbol] = name

    def set_market_rules(self, enabled: bool) -> None:
        """开关市场规则涨跌停（True=按板块规则，False=固定 limit_up/limit_down）。"""
        self.enable_market_rules = bool(enabled)

    # ---- 数据 ----

    def add_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        datetime_col: str = "date",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        amount_col: str = "amount",
    ) -> None:
        """添加K线数据（支持日线/分钟线）。

        Parameters
        ----------
        symbol : str
            标的代码
        df : pd.DataFrame
            包含OHLCV数据的DataFrame
        datetime_col : str, default "date"
        open_col : str, default "open"
        high_col : str, default "high"
        low_col : str, default "low"
        close_col : str, default "close"
        volume_col : str, default "volume"
        amount_col : str, default "amount"
        """
        if df.empty:
            warnings.warn(f"Empty DataFrame for {symbol}, skipping")
            return

        required = {datetime_col, open_col, high_col, low_col, close_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"DataFrame for {symbol} missing columns: {missing}")

        df_sorted = df.sort_values(datetime_col).reset_index(drop=True)
        bars: List[BarData] = []
        for _, row in df_sorted.iterrows():
            try:
                dt_val = row[datetime_col]
                if isinstance(dt_val, (int, float, np.integer, np.floating)):
                    dt = pd.Timestamp(dt_val).to_pydatetime()
                elif isinstance(dt_val, pd.Timestamp):
                    dt = dt_val.to_pydatetime()
                elif isinstance(dt_val, datetime):
                    dt = dt_val
                else:
                    dt = pd.Timestamp(str(dt_val)).to_pydatetime()

                bar = BarData(
                    symbol=symbol,
                    datetime=dt,
                    open=float(row[open_col]),
                    high=float(row[high_col]),
                    low=float(row[low_col]),
                    close=float(row[close_col]),
                    volume=float(row.get(volume_col, 0.0)),
                    amount=float(row.get(amount_col, 0.0)),
                )
                bars.append(bar)
            except Exception as exc:
                warnings.warn(
                    f"Skip row {_} for {symbol}: {exc}"
                )

        if bars:
            self.bars[symbol] = bars
            self.positions[symbol] = PositionData(symbol=symbol)
        else:
            warnings.warn(f"No valid bars loaded for {symbol}")

    # ---- 策略注册 ----

    def add_strategy(
        self, strategy_class: Type[StrategyTemplate], name: Optional[str] = None
    ) -> StrategyTemplate:
        """注册策略。

        Parameters
        ----------
        strategy_class : Type[StrategyTemplate]
            策略类（非实例）
        name : str, optional
            策略名称；默认使用类名

        Returns
        -------
        StrategyTemplate
            实例化后的策略对象
        """
        if not (isinstance(strategy_class, type) and
                issubclass(strategy_class, StrategyTemplate)):
            raise TypeError("strategy_class must be a subclass of StrategyTemplate")

        strategy = strategy_class()
        strategy.name = name if name else strategy_class.__name__

        # 避免重名
        if strategy.name in self.strategies:
            idx = 2
            while f"{strategy.name}_{idx}" in self.strategies:
                idx += 1
            strategy.name = f"{strategy.name}_{idx}"

        strategy.bg = self
        self.strategies[strategy.name] = strategy
        return strategy

    # ---- 核心事件循环 ----

    def run(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> BacktestResult:
        """启动回测。

        Parameters
        ----------
        start : str, optional
            回测开始日期（YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）
        end : str, optional
            回测结束日期

        Returns
        -------
        BacktestResult
            包含完整绩效指标的回测结果
        """
        if not self.bars:
            raise RuntimeError("No data loaded; call add_data() first")
        if not self.strategies:
            raise RuntimeError("No strategies registered; call add_strategy() first")

        # 重置状态
        self._reset_state()

        # --- 1. 构建对齐的时间线 ---
        all_bars = self._build_timeline(start, end)
        if not all_bars:
            return self._empty_result()

        # --- 2. 策略初始化 ---
        for strategy in self.strategies.values():
            try:
                strategy.on_init()
            except Exception as exc:
                warnings.warn(
                    f"Strategy '{strategy.name}' on_init() failed: {exc}"
                )

        for strategy in self.strategies.values():
            try:
                strategy.on_start()
            except Exception as exc:
                warnings.warn(
                    f"Strategy '{strategy.name}' on_start() failed: {exc}"
                )

        # --- 3. 逐根 K 线处理 (事件循环) ---
        for bar in all_bars:
            self.current_dt = bar.datetime

            # a0) P2-Q13-fix(M3): 开盘前处理遗留订单（next_open 市价单以本 bar 开盘价成交，
            #     使策略 on_bar 即可见新开持仓）
            self._process_orders(bar, new_only=False)

            # a) 触发策略 on_bar
            for strategy in self.strategies.values():
                try:
                    strategy.on_bar(bar)
                except Exception as exc:
                    warnings.warn(
                        f"Strategy '{strategy.name}' on_bar() failed @ "
                        f"{bar.datetime}: {exc}"
                    )

            # b) 处理本 bar 新提交订单（current_close 收盘价成交 / next_open 延至下一根 bar）
            self._process_orders(bar, new_only=True)

            # b2) 更新前收盘价（用于涨跌停计算）
            self._prev_close[bar.symbol] = bar.close

            # c) 更新浮动盈亏
            self._calculate_pnl()

            # d) 记录权益曲线（每个 bar 记录一次）
            self._record_equity(bar.datetime)

        # --- 4. End-of-run accounting ---
        if self.end_mode == "force_close":
            self._force_close(all_bars[-1])
        else:
            for order in self.orders:
                if order.status == "submitted":
                    order.status = "cancelled"
            self._pending_by_symbol.clear()
            self._calculate_pnl()
            final_dt = all_bars[-1].datetime
            position_value = 0.0
            for symbol, pos in self.positions.items():
                if pos.volume <= 1e-8:
                    continue
                bars = [b for b in self.bars.get(symbol, []) if b.datetime <= final_dt]
                if bars:
                    position_value += bars[-1].close * pos.volume if pos.direction == "long" else pos.price * pos.volume + pos.pnl
            final_row = {"datetime": final_dt, "total_value": self.capital + position_value, "cash": self.capital, "position_value": position_value}
            if self.equity_curve and self.equity_curve[-1]["datetime"] == final_dt:
                self.equity_curve[-1] = final_row
            else:
                self.equity_curve.append(final_row)

        # --- 5. 组装结果 ---
        result = self._build_result()
        return self._calculate_performance(result)

    def send_order(
        self,
        strategy_name: str,
        direction: str,
        offset: str,
        price: float,
        volume: float,
        order_type: str = "market",
        symbol: str = "",
    ) -> str:
        """生成订单。

        Parameters
        ----------
        strategy_name : str
        direction : str  "long"/"short"
        offset : str     "open"/"close"
        price : float
        volume : float
        order_type : str "market"/"limit"/"stop"
        symbol : str, optional
            标的代码（C3 修复：订单创建时即绑定标的）。
            为空时由成交引擎按当前 K 线补填（单标的兼容旧行为）；
            多标的/跨标的下单必须显式传入，否则可能被错误盖章。

        Returns
        -------
        str
            订单ID
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be long/short, got {direction}")
        if offset not in ("open", "close"):
            raise ValueError(f"offset must be open/close, got {offset}")
        if order_type not in ("market", "limit", "stop"):
            raise ValueError(
                f"order_type must be market/limit/stop, got {order_type}"
            )

        # 检查交易模式
        if self.trade_mode == "long" and direction == "short" and offset == "open":
            warnings.warn(f"trade_mode=long, short order rejected for {strategy_name}")
            return ""

        if self.trade_mode == "short" and direction == "long" and offset == "open":
            warnings.warn(f"trade_mode=short, long order rejected for {strategy_name}")
            return ""

        # 检查最小手数
        if volume < self.size:
            warnings.warn(
                f"Volume {volume} < min size {self.size}, order rejected"
            )
            return ""

        # 取整到最小手数
        volume = math.floor(volume / self.size) * self.size
        if volume <= 0:
            return ""

        self._order_counter += 1
        order_id = f"O{self._order_counter:010d}-{uuid.uuid4().hex[:8]}"

        order = OrderData(
            symbol=symbol,  # C3 fix: 订单创建时即绑定标的；空则成交时补填
            order_id=order_id,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            filled=0.0,
            status="submitted",
            datetime=self.current_dt,
            order_type=order_type,
            strategy_name=strategy_name,
        )
        self.orders.append(order)

        # P2-Q13-fix(M7): 按 symbol 索引待成交订单，避免每 bar 全量扫描
        self._pending_by_symbol.setdefault(symbol or "", []).append(order)

        # 回调策略
        if strategy_name in self.strategies:
            try:
                self.strategies[strategy_name].on_order(order)
            except Exception as exc:
                warnings.warn(
                    f"Strategy '{strategy_name}' on_order() failed: {exc}"
                )
        return order_id

    # ---- 内部方法 ----

    def _reset_state(self) -> None:
        """重置运行时状态。"""
        self.capital = self.initial_capital
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.current_dt = None
        self._order_counter = 0
        self._trade_counter = 0
        # M1 fix: 前收盘价不跨 run 泄漏（否则污染下一次回测首日涨跌停基准）
        self._prev_close.clear()
        # P2-Q13-fix(M7): 待成交订单索引随 run 重置
        self._pending_by_symbol.clear()
        self._last_bar_date = None
        self.applied_corporate_actions.clear()

        # 重置所有持仓
        for sym in self.positions:
            self.positions[sym] = PositionData(symbol=sym)

    def _build_timeline(
        self, start: Optional[str], end: Optional[str]
    ) -> List[BarData]:
        """构建统一时间线：按 datetime 合并所有标的 K 线并排序。

        Returns
        -------
        List[BarData]
            全局时间线，按 datetime 升序排列
        """
        all_bars: List[BarData] = []
        for symbol, bars in self.bars.items():
            for bar in bars:
                if start:
                    t_start = pd.Timestamp(start)
                    if bar.datetime < t_start.to_pydatetime():
                        continue
                if end:
                    t_end = pd.Timestamp(end)
                    if bar.datetime > t_end.to_pydatetime():
                        continue
                all_bars.append(bar)

        all_bars.sort(key=lambda x: (x.datetime, x.symbol))
        return all_bars

    def _apply_corporate_actions(self, action_date: date) -> None:
        for action in self.corporate_actions.get(action_date, []):
            symbol = str(action["symbol"])
            pos = self.positions.get(symbol)
            if pos is None or pos.volume <= 0:
                continue
            action_type = str(action["action_type"])
            record = {"date": str(action_date), "symbol": symbol, "action_type": action_type, "shares_before": pos.volume, "cash_before": self.capital}
            if action_type == "dividend":
                amount = float(action.get("cash_per_share", action.get("amount", 0.0)) or 0.0)
                self.capital += pos.volume * amount
            elif action_type in ("split", "bonus", "stock_split"):
                ratio = float(action.get("ratio", 1.0) or 1.0)
                if ratio > 0:
                    pos.volume *= ratio; pos.frozen *= ratio; pos.price /= ratio
            record.update({"shares_after": pos.volume, "cash_after": self.capital})
            self.applied_corporate_actions.append(record)

    def _process_orders(self, bar: BarData, new_only: bool = False) -> None:
        """检查待处理订单能否在当前 K 线成交（分两阶段调用）。

        P2-Q13-fix(M3/M7): 两阶段处理，保证 next_open 成交在策略 on_bar 前可见：
        - new_only=False（on_bar 之前调用）：处理此前 bar 遗留的订单
          （next_open 市价单以本 bar 开盘价成交；limit/stop 按本 bar OHLC 触发）；
        - new_only=True（on_bar 之后调用）：处理本 bar 新提交的订单
          （current_close 收盘价成交；next_open 延至下一根 bar 开盘价）。

        成交规则：
        - market  : 用 bar.close 成交（视为收盘价成交）；next_open 模式下延至次 bar 开盘价
        - limit   : 买入限价 ≥ bar.low 则成交（限价高于或触及最低价）
                    卖岀限价 ≤ bar.high 则成交
        - stop    : 买入止损价 ≤ bar.high 触发（价格向上突破）
                    卖岀止损价 ≥ bar.low 触发（价格向下突破）

        P2-Q13-fix(M7): 按 symbol 索引待成交订单（_pending_by_symbol），
        避免每 bar 全量扫描；长时间未成交（限价/止损未触发、无持仓可平、
        流动性不足）自动撤销，杜绝订单永久滞留。
        """
        if not new_only:
            # P1-Q13-fix(H4): T+1 次日解冻——新交易日开始时解冻全部持仓（当日买入的股票次日才可卖出）
            if self._last_bar_date != bar.datetime.date():
                for pos in self.positions.values():
                    pos.frozen = 0.0
                self._apply_corporate_actions(bar.datetime.date())
                self._last_bar_date = bar.datetime.date()

        # 空 symbol 订单（旧式单标的用法）按当前 bar 补填后归入对应 symbol 桶。
        # P2-Q13-fix: 两阶段都要迁移，否则 on_bar 刚提交的无 symbol 订单在 after 阶段被跳过，
        # 导致 current_close 同 bar 成交被推迟到下一根 bar。
        if "" in self._pending_by_symbol and self._pending_by_symbol[""]:
            stamping = self._pending_by_symbol.pop("")
            for o in stamping:
                o.symbol = o.symbol or bar.symbol
                self._pending_by_symbol.setdefault(o.symbol, []).append(o)

        pending = self._pending_by_symbol.get(bar.symbol, [])
        if not pending:
            return

        for order in list(pending):
            if order.status != "submitted":
                # 已结束（filled/cancelled），从索引移除
                self._remove_pending(order)
                continue

            # 阶段筛选：before_on_bar 只处理旧订单；after_on_bar 只处理本 bar 新订单，
            # 避免同一订单在一个 bar 内重复成交（部分成交后被二次撮合）
            if new_only:
                if order.datetime is not None and order.datetime != bar.datetime:
                    continue
            else:
                if order.datetime is not None and order.datetime == bar.datetime:
                    continue

            try:
                self._process_single_order(order, bar)
            except Exception as exc:
                warnings.warn(
                    f"Error processing order {order.order_id}: {exc}"
                )

            if order.status != "submitted":
                self._remove_pending(order)  # 成交/撤销后移出索引

        # after_on_bar 阶段末尾：对仍未成交的订单累计超时 bar 数（每 bar 一次）
        if new_only:
            for order in list(self._pending_by_symbol.get(bar.symbol, [])):
                if order.status != "submitted":
                    self._remove_pending(order)
                    continue
                # 部分成交视为有进展，重置计数
                if order.filled > 1e-8:
                    order.pending_bars = 0
                order.pending_bars += 1
                if self.order_timeout_bars > 0 and order.pending_bars >= self.order_timeout_bars:
                    order.status = "cancelled"
                    self._remove_pending(order)
                    warnings.warn(
                        f"Order {order.order_id} ({order.strategy_name}) "
                        f"cancelled after {order.pending_bars} bars "
                        f"(timeout={self.order_timeout_bars})"
                    )

    def _remove_pending(self, order: OrderData) -> None:
        """P2-Q13-fix(M7): 将订单从待成交索引移除。"""
        sym = order.symbol or ""
        lst = self._pending_by_symbol.get(sym)
        if lst:
            try:
                lst.remove(order)
            except ValueError:
                pass
            if not lst:
                self._pending_by_symbol.pop(sym, None)

    def _process_single_order(self, order: OrderData, bar: BarData) -> None:
        """处理单个订单的成交逻辑。"""
        # 如果订单还没有 symbol，用当前 bar 的 symbol
        if not order.symbol:
            order.symbol = bar.symbol

        # 只处理属于当前 bar.symbol 的订单
        if order.symbol != bar.symbol:
            return

        fill_price: Optional[float] = None
        # P2-Q13-fix(M3): next_open 模式以开盘价作涨跌停基准
        limit_ref_price: Optional[float] = None

        if order.order_type == "market":
            # P2-Q13-fix(M3): next_open 模式——信号 bar 收盘后下单，延至下一根 bar 开盘价成交。
            # 同一 bar 内刚下的市价单跳过（order.datetime == bar.datetime），次 bar 再处理。
            if self.fill_mode == "next_open":
                if order.datetime is not None and order.datetime == bar.datetime:
                    return
                fill_price = bar.open
                limit_ref_price = bar.open
            else:
                # 市价单：用当前收盘价成交
                fill_price = bar.close

        elif order.order_type == "limit":
            # 限价单
            # V11 审计修复（High）: 与市价单一致——next_open 模式下，
            # 信号 bar 收盘后下的限价单必须延至下一根 bar 成交，
            # 否则会用本 bar 的 low/high（收盘后才可知）成交，构成前视偏差。
            if self.fill_mode == "next_open":
                if order.datetime is not None and order.datetime == bar.datetime:
                    return
            if order.offset == "open":
                if order.direction == "long":
                    # 买入开多：price ≥ bar.low 则成交
                    if order.price >= bar.low:
                        fill_price = min(order.price, bar.close)
                else:
                    # 卖岀开空：price ≤ bar.high 则成交
                    if order.price <= bar.high:
                        fill_price = max(order.price, bar.close)
            else:
                if order.direction == "short":
                    # 卖岀平多：price ≤ bar.high 则成交
                    if order.price <= bar.high:
                        fill_price = max(order.price, bar.close)
                else:
                    # 买入平空：price ≥ bar.low 则成交
                    if order.price >= bar.low:
                        fill_price = min(order.price, bar.close)

        elif order.order_type == "stop":
            # 止损单
            # V11 审计修复（High）: next_open 模式同样延至下一根 bar，
            # 避免用本 bar 已发生的 high/low 触发收盘后才下的止损单。
            if self.fill_mode == "next_open":
                if order.datetime is not None and order.datetime == bar.datetime:
                    return
            if order.offset == "open":
                if order.direction == "long":
                    # 买入开多止损：价格向上突破，order.price ≤ bar.high
                    if order.price <= bar.high:
                        fill_price = max(order.price, bar.close)
                else:
                    # 卖岀开空止损：价格向下突破，order.price ≥ bar.low
                    if order.price >= bar.low:
                        fill_price = min(order.price, bar.close)
            else:
                if order.direction == "short":
                    # 卖岀平多止损：order.price ≥ bar.low 触发
                    if order.price >= bar.low:
                        fill_price = min(order.price, bar.close)
                else:
                    # 买入平空止损：order.price ≤ bar.high 触发
                    if order.price <= bar.high:
                        fill_price = max(order.price, bar.close)

        if fill_price is None:
            return  # 未满足成交条件

        # ---- 涨跌停限制 ----
        if not self._check_limit_up_down(bar, order, ref_price=limit_ref_price):
            return

        # ---- 滑点 ----
        # Correctly determine slippage direction (aligns with _execute_trade semantics):
        #   Long open(买入开多)   → buy  side (price goes up)
        #   Long close(买入平空)  → buy  side (price goes up)
        #   Short open(卖出开空)  → sell side (price goes down)
        #   Short close(卖出平多) → sell side (price goes down)
        # P2-Q13-fix: 此前 close 方向与注释相反（平多卖出计 buy 滑点，成交价系统性偏高，
        # 使回测收益偏乐观）；此处按成交动作修正。
        if order.offset == "open":
            side = "buy" if order.direction == "long" else "sell"
        else:  # close
            side = "sell" if order.direction == "short" else "buy"
        fill_price = self._apply_slippage(fill_price, side)

        # ---- 检查冻结数量 & 资金 ----
        remain = order.volume - order.filled
        if remain <= 0:
            order.status = "filled"
            return

        fill_volume = remain  # 默认全部成交

        # P2-Q13-fix(M4): 流动性约束——成交量占比上限（≤ bar.volume 的 max_volume_pct），
        # 对地量标的防止满额成交导致收益系统性偏高；0 成交量（停牌/无量）本 bar 不成交。
        if bar.volume > 1e-8 and self.max_volume_pct > 0:
            max_fill = math.floor(bar.volume * self.max_volume_pct / self.size) * self.size
            fill_volume = min(fill_volume, max_fill)
        elif bar.volume <= 1e-8:
            fill_volume = 0.0

        # P2-Q13-fix(M4): 冲击成本（平方根模型）——随参与率上升而增大，计入成交价
        if fill_volume > 1e-8 and bar.volume > 1e-8 and self.impact_coefficient > 0:
            participation = min(fill_volume / bar.volume, 1.0)
            impact_pct = self.impact_coefficient * math.sqrt(participation)
            if side == "buy":
                fill_price *= (1 + impact_pct)
            else:
                fill_price *= (1 - impact_pct)

        # 检查资金是否足够（仅开仓）
        if order.offset == "open":
            cost = fill_price * fill_volume
            commission = max(cost * self.commission_rate, self.min_commission)  # P1-Q13-fix(H1): 最低佣金5元/笔
            total_cost = cost + commission

            if order.direction == "long" and total_cost > self.capital:
                # 资金不足，按剩余资金成交
                max_cost = self.capital * 0.99  # 留点余量
                fill_volume = math.floor(
                    max_cost / (fill_price * (1 + self.commission_rate)) / self.size
                ) * self.size
                if fill_volume <= 0:
                    return

            if order.direction == "short":
                # 做空保证金要求：一般按市值比例（此处简化：按 100% 保证金）
                if total_cost > self.capital:
                    max_cost = self.capital * 0.99
                    fill_volume = math.floor(
                        max_cost / (fill_price * (1 + self.commission_rate)) / self.size
                    ) * self.size
                    if fill_volume <= 0:
                        return

        # P1-Q13-fix(H4): 检查可平仓量（T+1：当日买入/新开仓数量冻结，不可当日平仓）
        if order.offset == "close":
            pos = self.positions.get(order.symbol)
            if pos is None:
                return
            available = pos.volume - pos.frozen
            if available <= 1e-8:
                return  # 无可卖数量，订单保持 submitted，次日解冻后重试
            if order.direction == "long" and order.offset == "close":
                # long close(cover): 平空仓 = 减少 short 持仓
                if pos.direction == "short":
                    fill_volume = min(fill_volume, available)
                else:
                    return  # 没有空仓可平
            elif order.direction == "short" and order.offset == "close":
                # short close(sell): 平多仓
                if pos.direction == "long":
                    fill_volume = min(fill_volume, available)
                else:
                    return  # 没有多仓可平

        if fill_volume <= 0:
            return

        # ---- 执行成交 ----
        self._execute_trade(order, fill_price, fill_volume, bar.datetime)

    def _execute_trade(
        self,
        order: OrderData,
        fill_price: float,
        fill_volume: float,
        dt: datetime,
    ) -> None:
        """执行成交并更新订单/持仓/资金。"""
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:010d}"

        trade = TradeData(
            symbol=order.symbol,
            trade_id=trade_id,
            order_id=order.order_id,
            direction=order.direction,
            offset=order.offset,
            price=fill_price,
            volume=fill_volume,
            datetime=dt,
        )
        self.trades.append(trade)

        # 更新订单
        order.filled += fill_volume
        if abs(order.filled - order.volume) < 1e-8:
            order.status = "filled"
        else:
            order.status = "submitted"  # 部分成交，继续等待

        # 佣金 / 过户费 / 印花税
        cost = fill_price * fill_volume
        commission = max(cost * self.commission_rate, self.min_commission)  # P1-Q13-fix(H1): 最低佣金5元/笔
        # P1-Q13-fix(H2): 过户费 2022-04-29 起沪深A股统一 0.001% 双边收取
        transfer_fee = cost * self.transfer_fee_rate
        # 印花税（仅卖出时收：平多=方向short/平空=方向long不收）
        stamp_tax = 0.0
        if order.offset == "close" and order.direction == "short":
            stamp_tax = cost * 0.0005  # V4.1 fix: 卖出(平多)才收印花税，买入平空不收

        total_fee = commission + stamp_tax + transfer_fee

        # 更新资金和持仓
        pos = self.positions.setdefault(
            order.symbol, PositionData(symbol=order.symbol)
        )

        if order.offset == "open":
            # ----------------------------------------------------------
            # 开仓（C2 fix: 多空翻转时先按市价清算被对冲的旧反向持仓，
            # 回笼资金后再按净新仓开仓，资金只记一次，杜绝 double-deduct）
            # ----------------------------------------------------------
            closed_vol = 0.0       # 本笔成交中用于对冲旧反向持仓的数量
            net_open = fill_volume  # 净新开仓数量

            if order.direction == "long" and pos.direction == "short" and pos.volume > 1e-8:
                # 先平旧空仓（受 T+1 限制的当日新空仓不可平）
                closed_vol = min(fill_volume, max(pos.volume - pos.frozen, 0.0))
                net_open = fill_volume - closed_vol
                if closed_vol > 1e-8:
                    # 平空：返还保证金(pos.price*vol) + 已实现盈亏((pos.price-fill_price)*vol)
                    self.capital += (
                        pos.price * closed_vol
                        + (pos.price - fill_price) * closed_vol
                    )
                    pos.volume -= closed_vol
                    if pos.volume <= 1e-8:
                        pos.direction = "net"
                        pos.price = 0.0
            elif order.direction == "short" and pos.direction == "long" and pos.volume > 1e-8:
                # 先平旧多仓（受 T+1 限制的当日新买不可卖）
                closed_vol = min(fill_volume, max(pos.volume - pos.frozen, 0.0))
                net_open = fill_volume - closed_vol
                if closed_vol > 1e-8:
                    # 平多：按市价卖出回笼资金
                    # P1-3-fix: 翻转平多属卖出, 漏计卖出印花税(commission/transfer 已在
                    # total_fee 按 full fill_volume 计入=两腿佣金之和, 此处仅补 stamp tax)
                    self.capital += (
                        fill_price * closed_vol
                        - fill_price * closed_vol * 0.0005
                    )
                    pos.volume -= closed_vol
                    if pos.volume <= 1e-8:
                        pos.direction = "net"
                        pos.price = 0.0

            # 净新开仓部分：资金只扣一次（long=买入成本 / short=100%保证金）
            if net_open > 1e-8:
                self.capital -= fill_price * net_open + total_fee
                # P1-Q13-fix(H4): T+1 当日新开仓数量冻结，当日不可平仓
                pos.frozen += net_open

            # ---- 持仓方向/均价更新（保持原有净额逻辑）----
            # net_open <= 0 时本笔全部用于对冲旧仓，持仓已在清算逻辑中更新，无需再建仓
            if net_open > 1e-8:
                if order.direction == "long":
                    if pos.direction == "net":
                        pos.direction = "long"
                        pos.volume = net_open
                        pos.price = fill_price
                    elif pos.direction == "long":
                        # 加仓
                        total_cost_before = pos.price * pos.volume
                        total_cost_now = fill_price * net_open
                        pos.volume += net_open
                        pos.price = (total_cost_before + total_cost_now) / pos.volume
                    elif pos.direction == "short":
                        # 与空仓对冲（closed_vol 已在上方清算，此处仅处理剩余净额）
                        pos.direction = "long"
                        pos.volume = net_open
                        pos.price = fill_price
                else:
                    # 开空仓
                    if pos.direction == "net":
                        pos.direction = "short"
                        pos.volume = net_open
                        pos.price = fill_price
                    elif pos.direction == "short":
                        total_cost_before = pos.price * pos.volume
                        total_cost_now = fill_price * net_open
                        pos.volume += net_open
                        pos.price = (total_cost_before + total_cost_now) / pos.volume
                    elif pos.direction == "long":
                        pos.direction = "short"
                        pos.volume = net_open
                        pos.price = fill_price

        else:
            # 平仓（C2 fix: 平空 = 返还保证金 + 已实现盈亏 - 买入成本）
            if order.direction == "short" and pos.direction == "long":
                # 平多：卖出回笼资金
                self.capital += fill_price * fill_volume - total_fee
                pos.volume -= fill_volume
                if pos.volume <= 1e-8:
                    pos.direction = "net"
                    pos.volume = 0.0
                    pos.price = 0.0
            elif order.direction == "long" and pos.direction == "short":
                # 平空：返还保证金(pos.price*vol) + 盈亏((pos.price-fill_price)*vol) - 费用
                self.capital += (
                    pos.price * fill_volume
                    + (pos.price - fill_price) * fill_volume
                    - total_fee
                )
                pos.volume -= fill_volume
                if pos.volume <= 1e-8:
                    pos.direction = "net"
                    pos.volume = 0.0
                    pos.price = 0.0

        # 回调策略
        strategy_name = order.strategy_name
        if strategy_name in self.strategies:
            try:
                self.strategies[strategy_name].on_trade(trade)
            except Exception as exc:
                warnings.warn(
                    f"Strategy '{strategy_name}' on_trade() failed: {exc}"
                )

        # 回调策略 on_order（状态变更）
        if strategy_name in self.strategies:
            try:
                self.strategies[strategy_name].on_order(order)
            except Exception as exc:
                warnings.warn(
                    f"Strategy '{strategy_name}' on_order() failed: {exc}"
                )

    def _apply_slippage(self, price: float, side: str = "buy") -> float:
        """应用滑点模型。

        Parameters
        ----------
        price : float
            成交价
        side : str
            "buy" 或 "sell"; 买入向上滑点(更贵), 卖出向下滑点(更便宜)

        滑点模式 (P2-4 审计回测层：默认 percent，与全系统 0.1% 口径一致):
        - percent: price * (1 ± slippage_rate)，slippage_rate=0.001 即 0.1%（默认）
        - fixed  : price ± slippage_rate（每股绝对金额，如每股 ±0.01 元，仅显式选用）
        - none   : price
        """
        if self.slippage_mode == "none" or self.slippage_rate == 0:
            return price

        sign = 1.0 if side == "buy" else -1.0
        if self.slippage_mode == "fixed":
            # M2 fix: fixed = 绝对金额滑点
            return price + sign * self.slippage_rate
        elif self.slippage_mode == "percent":
            return price * (1 + sign * self.slippage_rate)
        else:
            return price

    def _check_limit_up_down(
        self, bar: BarData, order: OrderData, ref_price: Optional[float] = None
    ) -> bool:
        """检查涨跌停限制。

        涨停（价格 ≥ 涨停价）：不能开多仓 / 不能平空买入
        跌停（价格 ≤ 跌停价）：不能开空仓 / 不能平多卖出

        Parameters
        ----------
        ref_price : float, optional
            用于判断当日是否封板的参考价（next_open 成交用开盘价）；
            默认 None → 用 bar.close（旧行为）。

        2026-08-01: 接入 market_rules 按板块自动识别涨跌幅
        （主板±10% / 双创±20% / 北交所±30% / ST±5%，支持新股与退市整理期），
        enable_market_rules=False 时回退到固定 limit_up/limit_down。
        """
        # A股涨跌停基准价 = 前收盘价（非当日开盘价）
        # M5 fix: 首根 bar 无前收盘价时跳过涨跌停判断（避免用开盘价作基准失真）
        if bar.symbol not in self._prev_close:
            return True
        base_price = self._prev_close[bar.symbol]

        if self.enable_market_rules:
            try:
                from quant_system.market_rules import get_price_limit_pct
                name = self.symbol_names.get(bar.symbol, "")
                pct = get_price_limit_pct(bar.symbol, name=name)
                if pct > 0:
                    up_price = round(base_price * (1 + pct / 100.0) - 1e-9, 2)
                    down_price = round(base_price * (1 - pct / 100.0) + 1e-9, 2)
                else:
                    return True  # 无涨跌幅限制（新股前5日等）
            except Exception:
                # 规则库异常时回退固定值，保证回测不中断
                up_price = base_price * (1 + self.limit_up)
                down_price = base_price * (1 + self.limit_down)
        else:
            up_price = base_price * (1 + self.limit_up)
            down_price = base_price * (1 + self.limit_down)

        # P2-Q13-fix(M3): 涨停/跌停判定参考价——next_open 成交用开盘价，其余用收盘价
        judge_price = ref_price if ref_price is not None else bar.close
        is_limit_up = judge_price >= up_price - 1e-8
        is_limit_down = judge_price <= down_price + 1e-8

        if is_limit_up and order.direction == "long" and order.offset == "open":
            return False
        if is_limit_down and order.direction == "short" and order.offset == "open":
            return False
        # P1-Q13-fix(H3): A股规则——涨停买不进（禁平空买入）、跌停卖不出（禁平多卖出）
        if is_limit_up and order.direction == "long" and order.offset == "close":
            return False
        if is_limit_down and order.direction == "short" and order.offset == "close":
            return False

        return True

    def _calculate_pnl(self) -> None:
        """计算所有持仓的浮动盈亏。"""
        for symbol, pos in self.positions.items():
            if pos.volume <= 1e-8 or pos.price <= 1e-8:
                pos.pnl = 0.0
                pos.pnl_ratio = 0.0
                continue

            # 获取最新收盘价
            symbol_bars = self.bars.get(symbol, [])
            if not symbol_bars:
                continue
            # 找不超过 current_dt 的最近 bar
            sorted_bars = sorted(
                [b for b in symbol_bars if b.datetime <= self.current_dt],
                key=lambda x: x.datetime,
                reverse=True,
            )
            if not sorted_bars:
                continue
            last_bar = sorted_bars[0]

            if pos.direction == "long":
                pos.pnl = (last_bar.close - pos.price) * pos.volume
                pos.pnl_ratio = (last_bar.close - pos.price) / pos.price
            elif pos.direction == "short":
                pos.pnl = (pos.price - last_bar.close) * pos.volume
                pos.pnl_ratio = (pos.price - last_bar.close) / pos.price
            else:
                pos.pnl = 0.0
                pos.pnl_ratio = 0.0

    def _record_equity(self, dt: datetime) -> None:
        """记录当前时点的权益。"""
        # V4.1 fix: 移除重复的total_position_value死代码（第二段准确计算position_value）
        # 更准确：position_value = 持仓市值
        position_value = 0.0
        for pos in self.positions.values():
            if pos.volume > 1e-8:
                symbol_bars = self.bars.get(pos.symbol, [])
                if symbol_bars:
                    sorted_bars = sorted(
                        [b for b in symbol_bars if b.datetime <= dt],
                        key=lambda x: x.datetime,
                        reverse=True,
                    )
                    if sorted_bars:
                        last_price = sorted_bars[0].close
                        if pos.direction == "long":
                            position_value += last_price * pos.volume
                        elif pos.direction == "short":
                            # 做空持仓价值 = 冻结保证金(pos.price*volume) + 浮动盈亏
                            # （保证金在开空时已从现金扣除，须计入资产端，否则权益被低估）
                            position_value += pos.price * pos.volume + pos.pnl

        total_value = self.capital + position_value

        self.equity_curve.append(
            {
                "datetime": dt,
                "total_value": total_value,
                "cash": self.capital,
                "position_value": position_value,
            }
        )

    def _force_close_blocked_reason(self, ref_bar: BarData, direction: str,
                                    pos: PositionData,
                                    symbol_bars: List[BarData]) -> str:
        """判断强制平仓是否受阻，返回受阻原因；空串表示可正常强制平仓。

        P2-6 (审计回测层): 强制平仓前判涨跌停(per-board)与 max_volume_pct
        （成交量参与率）上限，避免在现实中卖不掉/买不进的价位强造成交。
        - 平多(卖出)：跌停封板卖不出（复用 _check_limit_up_down 判据）+ 停牌/地量
        - 平空(买入)：涨停封板买不进 + 停牌/地量

        涨跌停基准价 = ref_bar 的**前一根**收盘价：回测循环结束后
        ``self._prev_close[symbol]`` 已被 ref_bar 自身 close 覆盖（L577-578），
        须用前收价才能算出当日的真实封板下限。
        """
        # ---- 涨跌停（per-board）----
        # force_close 与 _process_orders 一样视为 close 订单：
        #   平多卖出 → direction=short/offset=close；平空买入 → direction=long/offset=close
        order_direction = "short" if direction == "long" else "long"
        tmp_order = OrderData(
            symbol=ref_bar.symbol,
            order_id="FORCE_CLOSE_LIMIT_CHECK",
            direction=order_direction,
            offset="close",
            price=ref_bar.close,
            volume=pos.volume,
        )

        # 找到 ref_bar 的前一根 bar 收盘价作为涨跌停基准
        prev_close = None
        sorted_bars = sorted(symbol_bars, key=lambda x: x.datetime)
        idx = None
        for i, b in enumerate(sorted_bars):
            if b.datetime == ref_bar.datetime:
                idx = i
                break
        if idx is not None and idx > 0:
            prev_close = float(sorted_bars[idx - 1].close)
        elif idx is None and len(sorted_bars) > 1:
            prev_close = float(sorted_bars[-2].close)

        # 临时将前收价注入，供 _check_limit_up_down 按板块计算当日涨跌停价
        orig_prev = self._prev_close.get(ref_bar.symbol)
        if prev_close is not None:
            self._prev_close[ref_bar.symbol] = prev_close
        blocked_by_limit = False
        try:
            blocked_by_limit = not self._check_limit_up_down(
                ref_bar, tmp_order, ref_price=ref_bar.close
            )
        finally:
            if prev_close is not None:
                if orig_prev is None:
                    self._prev_close.pop(ref_bar.symbol, None)
                else:
                    self._prev_close[ref_bar.symbol] = orig_prev
        if blocked_by_limit:
            if direction == "long":
                return "跌停封板，卖不出"
            return "涨停封板，买不进"

        # ---- 成交量参与率 / 停牌 ----
        # 复刻 _process_orders 的流动性约束：地量/停牌(volume≈0)不可成交；
        # 且持仓量不得超过当日成交量的 max_volume_pct（否则无法在该参与率内平仓）。
        if ref_bar.volume is None or ref_bar.volume <= 1e-8:
            return "停牌/无量，不可成交"
        if self.max_volume_pct > 0:
            max_fill = math.floor(ref_bar.volume * self.max_volume_pct / self.size) * self.size
            if max_fill <= 0 or pos.volume > max_fill + 1e-8:
                return (
                    f"成交量不足无法平仓 (max_volume_pct={self.max_volume_pct}, "
                    f"可平上限 {max_fill} < 持仓 {pos.volume:.0f})"
                )
        return ""

    def _force_close(self, last_bar: BarData) -> None:
        """回测结束时强制平仓。

        C1 fix: 每个标的按自己最近一根 K 线的收盘价平仓，
        不再统一使用全局时间线最后一根 bar 的价格。
        """
        for symbol, pos in list(self.positions.items()):
            if pos.volume <= 1e-8:
                continue

            symbol_bars = self.bars.get(symbol, [])
            if not symbol_bars:
                continue
            # 取该标的不晚于回测末日的最近 bar（数据缺失时回退到其最后一根 bar）
            ref_bars = [b for b in symbol_bars if b.datetime <= last_bar.datetime]
            ref_bar = ref_bars[-1] if ref_bars else symbol_bars[-1]

            direction = pos.direction
            fill_price = ref_bar.close
            # 平多=卖出, 平空=买入
            close_side = "buy" if direction == "short" else "sell"

            # P2-6 (审计回测层): 强制平仓前判涨跌停(per-board)与流动性上限
            # (max_volume_pct 成交量参与率)。现实中跌停卖不出 / 涨停买不进的
            # 持仓、或成交量不足(地量/停牌)的持仓，不应在无流动性价位被强造
            # 成交——标记该持仓为"持有/非流动"并记录原因，保留在 positions 中。
            blocked_reason = self._force_close_blocked_reason(
                ref_bar, direction, pos, symbol_bars
            )
            if blocked_reason:
                pos.force_close_blocked = True
                pos.force_close_blocked_reason = blocked_reason
                warnings.warn(
                    f"P2-6 force_close blocked {symbol} ({direction}): {blocked_reason}"
                )
                continue

            fill_price = self._apply_slippage(fill_price, close_side)

            if direction == "long":
                # 平多
                volume = pos.volume
                cost = fill_price * volume
                commission = max(cost * self.commission_rate, self.min_commission)  # P1-Q13-fix(H1)
                transfer_fee = cost * self.transfer_fee_rate  # P1-Q13-fix(H2)
                stamp_tax = cost * 0.0005
                total_fee = commission + stamp_tax + transfer_fee
                self.capital += cost - total_fee
                pos.volume = 0.0
                pos.direction = "net"
                pos.price = 0.0

                self._trade_counter += 1
                trade = TradeData(
                    symbol=symbol,
                    trade_id=f"T{self._trade_counter:010d}",
                    order_id="FORCE_CLOSE",
                    direction="short",
                    offset="close",
                    price=fill_price,
                    volume=volume,
                    datetime=ref_bar.datetime,
                )
                self.trades.append(trade)

            elif direction == "short":
                # 平空（C2 fix: 返还保证金 + 已实现盈亏 - 买入成本）
                volume = pos.volume
                cost = fill_price * volume
                commission = max(cost * self.commission_rate, self.min_commission)  # P1-Q13-fix(H1)
                transfer_fee = cost * self.transfer_fee_rate  # P1-Q13-fix(H2)
                total_fee = commission + transfer_fee  # 买入平空不收印花税
                self.capital += (
                    pos.price * volume
                    + (pos.price - fill_price) * volume
                    - total_fee
                )
                pos.volume = 0.0
                pos.direction = "net"
                pos.price = 0.0

                self._trade_counter += 1
                trade = TradeData(
                    symbol=symbol,
                    trade_id=f"T{self._trade_counter:010d}",
                    order_id="FORCE_CLOSE",
                    direction="long",
                    offset="close",
                    price=fill_price,
                    volume=volume,
                    datetime=ref_bar.datetime,
                )
                self.trades.append(trade)

        # P2-Q13-fix(M7): 回测结束撤销所有未成交订单（限价/止损未触发、次日开盘未成交等不再有效）
        for order in self.orders:
            if order.status == "submitted":
                order.status = "cancelled"
        self._pending_by_symbol.clear()

        # Revalue any positions that could not be force-closed. They remain
        # illiquid holdings, not a total loss. Writing cash-only equity here would
        # erase their market value and manufacture a terminal drawdown.
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if pos.volume <= 1e-8:
                continue
            symbol_bars = self.bars.get(symbol, [])
            ref_bars = [b for b in symbol_bars if b.datetime <= last_bar.datetime]
            if not ref_bars:
                continue
            price = ref_bars[-1].close
            if pos.direction == "long":
                position_value += price * pos.volume
            elif pos.direction == "short":
                position_value += pos.price * pos.volume + pos.pnl
        final_row = {"datetime": last_bar.datetime, "total_value": self.capital + position_value, "cash": self.capital, "position_value": position_value}
        if self.equity_curve and self.equity_curve[-1]["datetime"] == last_bar.datetime:
            self.equity_curve[-1] = final_row
        else:
            self.equity_curve.append(final_row)

    def _build_result(self) -> BacktestResult:
        """从回测数据组装 BacktestResult。"""
        if not self.equity_curve:
            return BacktestResult(
                equity_curve=pd.DataFrame(
                    columns=["datetime", "total_value", "cash", "position_value"]
                ),
                trades=[],
                orders=[],
            )

        df_eq = pd.DataFrame(self.equity_curve)
        df_eq["returns"] = df_eq["total_value"].pct_change().fillna(0.0)

        # P2-6 (审计回测层): 汇集末次强制平仓被阻止的持仓（持有/非流动）
        force_close_blocks = [
            {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "volume": float(pos.volume),
                "reason": pos.force_close_blocked_reason,
            }
            for pos in self.positions.values()
            if pos.force_close_blocked
        ]

        return BacktestResult(
            equity_curve=df_eq,
            trades=list(self.trades),
            orders=list(self.orders),
            total_trades=len(self.trades),
            force_close_blocks=force_close_blocks,
        )

    def _empty_result(self) -> BacktestResult:
        """返回空的回测结果。"""
        return BacktestResult(
            equity_curve=pd.DataFrame(
                columns=["datetime", "total_value", "cash", "position_value", "returns"]
            ),
            trades=[],
            orders=[],
        )

    def _calculate_performance(self, result: BacktestResult) -> BacktestResult:
        """计算完整绩效指标。

        Parameters
        ----------
        result : BacktestResult
            初步回测结果（含 equity_curve, trades, orders）

        Returns
        -------
        BacktestResult
            补全了所有绩效指标的结果
        """
        eq = result.equity_curve
        if eq.empty or len(eq) < 2:
            return result

        # P2-Q13-fix(L1): 删除未使用的死代码 n_days
        total_return = eq["total_value"].iloc[-1] / self.initial_capital - 1.0
        result.total_return = total_return

        # --- 年化收益率 ---
        # 推断回测年数：根据数据频率
        dt_min = eq["datetime"].iloc[0]
        dt_max = eq["datetime"].iloc[-1]
        years = (dt_max - dt_min).total_seconds() / (365.25 * 86400)
        years = max(years, 1.0 / 365.0)  # 最少1天

        # 审计 2026-08-16：极端亏损(total_return<=-1 → 本金亏光/倒欠)时
        # (1+total_return)^(1/years) 为负数分数次幂 → NaN；显式返回 -1（亏光）
        if total_return <= -1.0:
            result.annual_return = -1.0
        else:
            result.annual_return = (1 + total_return) ** (1.0 / years) - 1.0

        # --- 年化波动率 ---
        daily_returns = eq["returns"].values
        # P2-Q13-fix(L4): 年化因子按 bar 频率感知（日线252/周线52/月线12/日内252×每日bar数），
        # 可显式配置 annualization_factor 覆盖自动推断
        annual_factor = self._infer_annualization_factor(eq)
        daily_vol = np.std(daily_returns, ddof=1)
        result.annual_volatility = daily_vol * math.sqrt(annual_factor)

        # --- Sharpe ---
        rf_rate = 0.02  # 无风险利率2%
        daily_rf = rf_rate / annual_factor
        excess_returns = daily_returns - daily_rf
        if daily_vol > 1e-10:
            result.sharpe_ratio = (
                float(np.mean(excess_returns) / daily_vol) * math.sqrt(annual_factor)
            )
        else:
            result.sharpe_ratio = 0.0

        # --- 最大回撤 ---
        cum_max = np.maximum.accumulate(eq["total_value"].values)
        drawdown = (eq["total_value"].values - cum_max) / cum_max
        result.max_drawdown = float(np.min(drawdown))

        # --- 最大回撤持续期 ---
        underwater = drawdown < -1e-8
        max_duration = 0
        current_duration = 0
        for flag in underwater:
            if flag:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        result.max_drawdown_duration = max_duration

        # --- Calmar ---
        if abs(result.max_drawdown) > 1e-10:
            result.calmar_ratio = result.annual_return / abs(result.max_drawdown)
        else:
            result.calmar_ratio = 0.0

        # --- 胜率 & 盈亏比 ---
        trades = result.trades
        if len(trades) >= 2:
            # P2-Q13-fix(M6): 传入费率，按净盈亏（含佣金/印花税/过户费）统计胜率
            pnl_list = self._calculate_trade_pnl(
                trades,
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
                transfer_fee_rate=self.transfer_fee_rate,
            )
            if pnl_list:
                wins = sum(1 for p in pnl_list if p > 0)
                losses = sum(1 for p in pnl_list if p < 0)
                total_pos = sum(p for p in pnl_list if p > 0)
                total_neg = abs(sum(p for p in pnl_list if p < 0))

                if wins + losses > 0:
                    result.win_rate = wins / (wins + losses)
                if total_neg > 1e-10:
                    result.profit_loss_ratio = total_pos / total_neg
                result.total_trades = len(pnl_list)

        # --- Alpha / Beta / Information Ratio（基于基准） ---
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 1:
            try:
                # 对齐基准与策略收益序列
                bm = self.benchmark_returns.copy()
                # M9 fix: 统一基准索引为 datetime（字符串日期索引会导致 get 全部返回 None）
                if not isinstance(bm.index, pd.DatetimeIndex):
                    bm.index = pd.to_datetime(bm.index)
                eq_dates = pd.to_datetime(eq["datetime"])
                aligned_bm = []
                aligned_strat = []
                for i, dt in enumerate(eq_dates):
                    bm_val = bm.get(dt, None)
                    if bm_val is not None and not np.isnan(bm_val):
                        aligned_bm.append(bm_val)
                        aligned_strat.append(daily_returns[i])

                if len(aligned_bm) < 5:
                    warnings.warn(
                        "benchmark alignment produced <5 matched points; "
                        "alpha/beta/information_ratio skipped"
                    )
                else:
                    bm_arr = np.array(aligned_bm)
                    strat_arr = np.array(aligned_strat)

                    # Beta = Cov(s, bm) / Var(bm)
                    cov_mat = np.cov(strat_arr, bm_arr)
                    var_bm = np.var(bm_arr, ddof=1)
                    if var_bm > 1e-10:
                        result.beta = cov_mat[0, 1] / var_bm
                        result.alpha = float(
                            np.mean(strat_arr) - result.beta * np.mean(bm_arr)
                        ) * annual_factor  # 年化 alpha
                        result.alpha += rf_rate * (1 - result.beta)

                    # Information Ratio
                    te_arr = strat_arr - bm_arr
                    te_vol = np.std(te_arr, ddof=1)
                    if te_vol > 1e-10:
                        result.information_ratio = (
                            float(np.mean(te_arr) / te_vol) * math.sqrt(annual_factor)
                        )
            except Exception as exc:
                warnings.warn(f"Failed to calculate alpha/beta: {exc}")

        return result

    def _infer_annualization_factor(self, eq: pd.DataFrame) -> float:
        """P2-Q13-fix(L4): 按 bar 频率推断年化因子（一年内 bar 数）。

        规则：
        - 显式配置 annualization_factor > 0 时直接采用；
        - 日内（中位间隔 < 12h）：252 交易日 × 每交易日 bar 数；
        - 日线（< 2.5天）：252；
        - 周线（< 8天）：52；
        - 月线（< 40天）：12；
        - 更低频：4。

        替代原 len(returns)/years 的朴素估算，避免分钟/周线数据的年化失真。
        """
        if self.annualization_factor is not None and self.annualization_factor > 0:
            return float(self.annualization_factor)

        dts = pd.to_datetime(eq["datetime"])
        if len(dts) < 2:
            return 252.0
        med_secs = float(dts.diff().dropna().dt.total_seconds().median())
        if med_secs <= 0:
            return 252.0

        day = 86400.0
        if med_secs < day * 0.5:
            bars_per_day = day / med_secs
            return 252.0 * bars_per_day
        if med_secs < day * 2.5:
            return 252.0
        if med_secs < day * 8:
            return 52.0
        if med_secs < day * 40:
            return 12.0
        return 4.0

    @staticmethod
    def _calculate_trade_pnl(
        trades: List[TradeData],
        commission_rate: float = COMMISSION_RATE,
        min_commission: float = MIN_COMMISSION,
        transfer_fee_rate: float = TRANSFER_FEE_RATE,
    ) -> List[float]:
        """计算每笔完整交易的净盈亏（含佣金/印花税/过户费）。

        P2-Q13-fix(M6):
        - 配对时计入费用（此前仅按毛 PnL，不含佣金/印花税/滑点，高估策略质量）；
          TradeData.price 已是滑点后价格，故此处补计交易费用。
        - 对翻转单（open-long 后 open-short 的换向交易）单独成对核算：
          新 open 先与旧反向 open 对冲，剩余部分作为新开仓，避免 FIFO 错配。

        配对规则：按 symbol 分组，FIFO 配对 open/close。
        """
        from collections import defaultdict
        import copy as _copy

        def _fees(price: float, vol: float, is_sell: bool) -> float:
            """单笔成交费用 = 佣金(最低5元) + 过户费(双边0.001%) + 印花税(仅卖出0.05%)。"""
            commission = max(price * vol * commission_rate, min_commission)
            transfer = price * vol * transfer_fee_rate
            stamp = price * vol * 0.0005 if is_sell else 0.0
            return commission + transfer + stamp

        # 按 symbol 分组
        by_symbol: Dict[str, List[TradeData]] = defaultdict(list)
        for t in trades:
            by_symbol[t.symbol].append(t)

        pnl_list: List[float] = []
        for sym, sym_trades in by_symbol.items():
            # 按时间排序；复制 TradeData 再配对，避免修改调用方持有的成交记录（volume 被扣减）
            sym_trades_sorted = sorted(
                (_copy.copy(t) for t in sym_trades), key=lambda x: x.datetime
            )
            open_trades: List[TradeData] = []  # 尚未平仓的开仓记录
            for tr in sym_trades_sorted:
                if tr.offset == "open":
                    # ---- 翻转单：新开仓先与旧反向开仓对冲 ----
                    while open_trades and tr.volume > 1e-8 and open_trades[0].direction != tr.direction:
                        ot = open_trades[0]
                        match_vol = min(ot.volume, tr.volume)
                        if ot.direction == "long":
                            # long open 被 short open 卖出对冲 → 做多盈亏 + 卖出印花税
                            pnl = (tr.price - ot.price) * match_vol
                            is_sell = True
                        else:
                            # short open 被 long open 买回对冲 → 做空盈亏，无印花税
                            pnl = (ot.price - tr.price) * match_vol
                            is_sell = False
                        pnl -= _fees(ot.price, match_vol, False)   # 原开仓费用
                        pnl -= _fees(tr.price, match_vol, is_sell)  # 本次翻转成交费用
                        pnl_list.append(pnl)
                        ot.volume -= match_vol
                        tr.volume -= match_vol
                        if ot.volume <= 1e-8:
                            open_trades.pop(0)
                    if tr.volume > 1e-8:
                        open_trades.append(tr)  # 剩余部分作为新开仓
                elif tr.offset == "close" and open_trades:
                    # ---- 平仓：FIFO 配对 ----
                    while tr.volume > 1e-8 and open_trades:
                        ot = open_trades[0]
                        if ot.direction == tr.direction:
                            break  # 方向相同的异常平仓不配对
                        match_vol = min(ot.volume, tr.volume)
                        if ot.direction == "long" and tr.direction == "short":
                            # 平多（卖出）：做多盈亏 + 印花税
                            pnl = (tr.price - ot.price) * match_vol
                            is_sell = True
                        elif ot.direction == "short" and tr.direction == "long":
                            # 平空（买入）：做空盈亏，无印花税
                            pnl = (ot.price - tr.price) * match_vol
                            is_sell = False
                        else:
                            break
                        pnl -= _fees(ot.price, match_vol, False)   # 开仓费用
                        pnl -= _fees(tr.price, match_vol, is_sell)  # 平仓费用
                        pnl_list.append(pnl)
                        ot.volume -= match_vol
                        tr.volume -= match_vol
                        if ot.volume <= 1e-8:
                            open_trades.pop(0)
                # 忽略未配对平仓
            # 未平仓 open 不产生已实现盈亏（force_close 已生成 close 成交用于配对）
        return pnl_list


# ---------------------------------------------------------------------------
# K线合并器
# ---------------------------------------------------------------------------


class BarMerger:
    """K线合并器——用于对齐多个标的的时间序列。

    用途：
        - 回测前将不同上市时间的标的统一到同一 timeline
        - 填充缺失日期（NaN 或前值填充）
    """

    def __init__(self) -> None:
        self.bar_dict: Dict[str, pd.DataFrame] = {}
        # P2-Q13-fix(L3): 记录每个标的的 datetime 列名（get_aligned_bars 不再硬编码 "date"）
        self.datetime_cols: Dict[str, str] = {}

    def add_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        datetime_col: str = "date",
    ) -> None:
        """添加一个标的的K线数据。

        Parameters
        ----------
        symbol : str
        df : pd.DataFrame
            必须包含 datetime_col 列
        datetime_col : str, default "date"
        """
        if datetime_col not in df.columns:
            raise KeyError(
                f"DataFrame for {symbol} missing datetime column '{datetime_col}'"
            )
        dfc = df.copy()
        dfc[datetime_col] = pd.to_datetime(dfc[datetime_col])
        dfc = dfc.sort_values(datetime_col).reset_index(drop=True)
        self.bar_dict[symbol] = dfc
        self.datetime_cols[symbol] = datetime_col

    def get_aligned_bars(
        self,
        fill_method: str = "ffill",
        price_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取对齐后的K线数据。

        将每个标的的收盘价（或指定列）合并到一张宽表中，
            每列格式 "{symbol}_{col}"。
        缺失值用 fill_method 填充。

        Parameters
        ----------
        fill_method : str, default "ffill"
            缺失值填充方法（"ffill"/"bfill"/None）
        price_cols : list of str, optional
            需要对齐的价格列；默认["close"]

        Returns
        -------
        pd.DataFrame
            index=datetime, columns="{symbol}_{col}"
        """
        if not self.bar_dict:
            return pd.DataFrame()

        if price_cols is None:
            price_cols = ["close"]

        merged: Optional[pd.DataFrame] = None

        for symbol, df in self.bar_dict.items():
            # P2-Q13-fix(L3): 使用 add_symbol 记录的 datetime_col，而非硬编码 "date"
            dt_col = self.datetime_cols.get(symbol, "date")
            for col in price_cols:
                if col not in df.columns:
                    continue
                series = df.set_index(dt_col)[col] if dt_col in df.columns else df.set_index(df.columns[0])[col]
                # 重命名
                series.name = f"{symbol}_{col}"
                if merged is None:
                    merged = series.to_frame()
                else:
                    merged = merged.join(series, how="outer")

        if merged is None:
            return pd.DataFrame()

        merged = merged.sort_index()

        if fill_method == "ffill":
            merged = merged.ffill()
        elif fill_method == "bfill":
            merged = merged.bfill()

        # 删除全 NaN 的行
        merged = merged.dropna(how="all")

        return merged


# ---------------------------------------------------------------------------
# 回测辅助函数
# ---------------------------------------------------------------------------


def run_multiple_strategies(
    strategies: List[Type[StrategyTemplate]],
    data: Dict[str, pd.DataFrame],
    capital: float = 1_000_000,
    commission_rate: float = COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    slippage_mode: str = "percent",
    trade_mode: str = "both",
    start: Optional[str] = None,
    end: Optional[str] = None,
    n_jobs: int = 1,
) -> Dict[str, BacktestResult]:
    """并行跑多个策略。

    Parameters
    ----------
    strategies : list of StrategyTemplate subclasses
        策略类列表
    data : dict of {str: pd.DataFrame}
        {symbol: df} 格式的K线数据
    capital : float, default 1_000_000
    commission_rate : float, default COMMISSION_RATE
    slippage_rate : float, default 0.001
    slippage_mode : str, default "percent"
    trade_mode : str, default "both"
    start : str, optional
    end : str, optional
    n_jobs : int, default 1
        并行数；1 为串行

    Returns
    -------
    dict of {str: BacktestResult}
        {strategy_name: result}
    """
    if n_jobs <= 1:
        # 串行
        results: Dict[str, BacktestResult] = {}
        for strategy_cls in strategies:
            engine = BacktestEngine()
            engine.set_capital(capital)
            engine.set_commission(commission_rate)
            engine.set_slippage(slippage_rate, slippage_mode)
            # P1-Q13-fix(H5): 串行路径传递 trade_mode（此前被忽略，默认 both 导致长/空单过滤失效）
            engine.trade_mode = trade_mode
            for symbol, df in data.items():
                engine.add_data(symbol, df)
            strategy = engine.add_strategy(strategy_cls)
            try:
                result = engine.run(start=start, end=end)
                results[strategy.name] = result
            except Exception as exc:
                warnings.warn(f"Strategy {strategy_cls.__name__} failed: {exc}")
                results[strategy.name] = BacktestResult(
                    equity_curve=pd.DataFrame(
                        columns=["datetime", "total_value", "cash", "position_value", "returns"]
                    ),
                    trades=[],
                    orders=[],
                )
        return results

    # 并行
    try:
        from joblib import Parallel, delayed

        def _run_one(cls: Type[StrategyTemplate]) -> Tuple[str, BacktestResult]:
            eng = BacktestEngine()
            eng.set_capital(capital)
            eng.set_commission(commission_rate)
            eng.set_slippage(slippage_rate, slippage_mode)
            # P1-Q13-fix(H5): 并行路径同样传递 trade_mode
            eng.trade_mode = trade_mode
            for sym, df in data.items():
                eng.add_data(sym, df)
            st = eng.add_strategy(cls)
            try:
                res = eng.run(start=start, end=end)
                return st.name, res
            except Exception as exc:
                warnings.warn(f"Strategy {cls.__name__} failed: {exc}")
                return st.name, BacktestResult(
                    equity_curve=pd.DataFrame(
                        columns=["datetime", "total_value", "cash", "position_value", "returns"]
                    ),
                    trades=[],
                    orders=[],
                )

        out = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(cls) for cls in strategies
        )
        results = dict(out)
        return results
    except ImportError:
        warnings.warn("joblib not installed, falling back to serial execution")
        return run_multiple_strategies(
            strategies, data, capital,
            commission_rate, slippage_rate, slippage_mode,
            trade_mode, start, end, n_jobs=1,
        )


def parameter_scan(
    strategy_class: Type[StrategyTemplate],
    data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    capital: float = 1_000_000,
    commission_rate: float = COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    slippage_mode: str = "percent",
    start: Optional[str] = None,
    end: Optional[str] = None,
    n_jobs: int = 1,
    metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """参数网格扫描。

    Parameters
    ----------
    strategy_class : StrategyTemplate subclass
    data : dict of {str: pd.DataFrame}
    param_grid : dict of {str: list}
        参数搜索空间，例如 {"fast_ma": [5, 10, 20], "slow_ma": [30, 60]}
    capital : float
    commission_rate : float
    slippage_rate : float
    slippage_mode : str
    start : str, optional
    end : str, optional
    n_jobs : int, default 1
    metric : str, default "sharpe_ratio"
        用于结果排序的指标名

    Returns
    -------
    pd.DataFrame
        所有参数组合的结果，按 metric 降序排列
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    # 生成所有参数组合
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _run_with_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """用给定参数运行一次回测。"""
        engine = BacktestEngine()
        engine.set_capital(capital)
        engine.set_commission(commission_rate)
        engine.set_slippage(slippage_rate, slippage_mode)
        for sym, df in data.items():
            engine.add_data(sym, df)

        # 创建策略实例
        strategy = engine.add_strategy(strategy_class)
        # 注入参数
        for k, v in params.items():
            if hasattr(strategy, k):
                setattr(strategy, k, v)

        try:
            result = engine.run(start=start, end=end)
            # 提取指标
            row = {
                **params,
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "annual_volatility": result.annual_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_loss_ratio": result.profit_loss_ratio,
                "total_trades": result.total_trades,
                "calmar_ratio": result.calmar_ratio,
                "information_ratio": result.information_ratio,
                "alpha": result.alpha,
                "beta": result.beta,
                "status": "ok",
            }
            return row
        except Exception as exc:
            warnings.warn(
                f"Parameter combination {params} failed: {exc}"
            )
            # V4.1 audit fix (parameter_scan 全0排序): 失败组合指标用 NaN 而非 0，
            # 并标记 status=failed。此前返回全 0 会与"正常但无交易"的组合无法区分，
            # 且 0 值参与排序导致结果表顶部全是失败组合。
            row = {
                **params,
                "total_return": float("nan"),
                "annual_return": float("nan"),
                "annual_volatility": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "win_rate": float("nan"),
                "profit_loss_ratio": float("nan"),
                "total_trades": 0,
                "calmar_ratio": float("nan"),
                "information_ratio": float("nan"),
                "alpha": float("nan"),
                "beta": float("nan"),
                "status": "failed",
            }
            return row

    if n_jobs <= 1:
        rows = [_run_with_params(p) for p in all_combos]
    else:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs)(
                delayed(_run_with_params)(p) for p in all_combos
            )
        except ImportError:
            warnings.warn("joblib not installed, falling back to serial")
            rows = [_run_with_params(p) for p in all_combos]

    df = pd.DataFrame(rows)
    if metric in df.columns:
        # V4.1 audit fix: 失败组合(status=failed)不参与排名——指标为 NaN，
        # pandas sort_values 默认 NaN 排最后。无交易组合(total_trades=0)排在
        # 有交易组合之后（sharpe=0 的"空跑"组合不应压过真实交易结果），
        # 同交易量内再按 metric 降序。
        df["_has_trades"] = df.get("total_trades", pd.Series(0, index=df.index)) > 0
        df = df.sort_values(
            ["status", "_has_trades", metric],
            ascending=[False, False, False],
            na_position="last",
        ).drop(columns=["_has_trades"]).reset_index(drop=True)
    return df


def walk_forward_backtest(
    strategy_class: Type[StrategyTemplate],
    data: Dict[str, pd.DataFrame],
    window: int = 252,
    step: int = 63,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    capital: float = 1_000_000,
    commission_rate: float = COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    slippage_mode: str = "percent",
    start: Optional[str] = None,
    end: Optional[str] = None,
    param_scan_n: int = 10,
) -> List[BacktestResult]:
    """Walk-Forward 回测。

    流程：
        1. 将全周期划分为多个滑动窗口（训练期 + 验证期）
        2. 每个窗口：在训练期内扫描最优参数
        3. 用最优参数在验证期运行回测
        4. 汇总所有验证期的结果

    Parameters
    ----------
    strategy_class : StrategyTemplate subclass
    data : dict of {str: pd.DataFrame}
        标的数据（至少包含一个标的的完整K线）
    window : int, default 252
        训练期长度（K线数）
    step : int, default 63
        验证期长度（K线数）
    param_bounds : dict of {str: (low, high)}, optional
        参数搜索范围；若为 None 则使用默认值
    capital : float
    commission_rate : float
    slippage_rate : float
    slippage_mode : str
    start : str, optional
    end : str, optional
    param_scan_n : int, default 10
        每个参数扫描的点数

    Returns
    -------
    list of BacktestResult
        每个验证期的回测结果
    """
    if not data:
        raise ValueError("data must be non-empty")

    # 取第一个标的的完整时间线作为参考
    first_symbol = list(data.keys())[0]
    first_df = data[first_symbol].copy()
    if "date" in first_df.columns:
        dt_col = "date"
    else:
        dt_col = first_df.columns[0]
    first_df = first_df.sort_values(dt_col).reset_index(drop=True)
    if start:
        first_df = first_df[first_df[dt_col] >= pd.Timestamp(start)].reset_index(
            drop=True
        )
    if end:
        first_df = first_df[first_df[dt_col] <= pd.Timestamp(end)].reset_index(
            drop=True
        )
    total_rows = len(first_df)

    if total_rows < window + step:
        raise ValueError(
            f"Total rows {total_rows} < window {window} + step {step}"
        )

    # 构建参数网格（如果 param_bounds 提供）
    def _make_param_grid(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
        grid: Dict[str, List[float]] = {}
        for k, (lo, hi) in bounds.items():
            if lo >= hi:
                grid[k] = [lo]
            else:
                step_size = (hi - lo) / (param_scan_n - 1) if param_scan_n > 1 else 0
                if isinstance(lo, int) and isinstance(hi, int):
                    grid[k] = [int(lo + i * step_size) for i in range(param_scan_n)]
                else:
                    grid[k] = [lo + i * step_size for i in range(param_scan_n)]
        return grid

    all_results: List[BacktestResult] = []

    for train_end in range(window, total_rows, step):
        if train_end + step > total_rows:
            test_end = total_rows
        else:
            test_end = train_end + step

        train_start_idx = train_end - window
        train_end_idx = train_end
        test_start_idx = train_end
        test_end_idx = test_end

        # 提取训练期和验证期的日期范围
        train_start_dt = first_df.iloc[train_start_idx][dt_col]
        train_end_dt = first_df.iloc[train_end_idx - 1][dt_col]
        test_start_dt = first_df.iloc[test_start_idx][dt_col]
        test_end_dt = first_df.iloc[test_end_idx - 1][dt_col]

        # ---- 1. 训练期参数扫描 ----
        best_params: Dict[str, Any] = {}
        if param_bounds:
            param_grid = _make_param_grid(param_bounds)
            scan_result = parameter_scan(
                strategy_class=strategy_class,
                data=data,
                param_grid=param_grid,
                capital=capital,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                slippage_mode=slippage_mode,
                start=str(train_start_dt),
                end=str(train_end_dt),
                n_jobs=1,
                metric="sharpe_ratio",
            )
            if not scan_result.empty:
                # V4.1 audit fix: 跳过失败组合，取第一个 status=ok 的行
                ok_rows = (
                    scan_result[scan_result.get("status", "ok") != "failed"]
                    if "status" in scan_result.columns
                    else scan_result
                )
                if ok_rows.empty:
                    ok_rows = scan_result
                best_row = ok_rows.iloc[0]
                best_params = {
                    k: best_row[k] for k in param_bounds.keys()
                }

        # ---- 2. 验证期回测 ----
        engine = BacktestEngine()
        engine.set_capital(capital)
        engine.set_commission(commission_rate)
        engine.set_slippage(slippage_rate, slippage_mode)
        for sym, df in data.items():
            engine.add_data(sym, df)

        strategy = engine.add_strategy(strategy_class)
        for k, v in best_params.items():
            if hasattr(strategy, k):
                setattr(strategy, k, v)

        try:
            result = engine.run(start=str(test_start_dt), end=str(test_end_dt))
            all_results.append(result)
        except Exception as exc:
            warnings.warn(
                f"Walk-forward test window "
                f"[{test_start_dt}, {test_end_dt}] failed: {exc}"
            )

        # 如果到达数据末尾，停止
        if test_end_idx >= total_rows:
            break

    return all_results


# P2-Q13-fix(L6): 移除被注释且未维护的 __main__ 示例块（死代码，含潜在过时 API 用法）。
# 使用示例见 tests/ 与 backtest.py / quick_backtest()。

# ---------------------------------------------------------------------------
# 扩展风险度量工具 (RiskMetrics)
# ---------------------------------------------------------------------------


class RiskMetrics:
    """扩展风险度量工具集。

    在 BacktestResult 的基础上计算更为细致的风险指标，
    包括 Sortino 比率、VaR、CVaR、下行波动率、盈亏分布峰度/偏度等。
    这些指标不直接影响回测逻辑，仅供分析和报告使用。
    """

    def __init__(
        self,
        result: BacktestResult,
        risk_free_rate: float = 0.02,
        confidence_level: float = 0.95,
        trading_days: int = 252,
    ) -> None:
        """
        Parameters
        ----------
        result : BacktestResult
            已完成回测的结果对象
        risk_free_rate : float, default 0.02
            年化无风险利率
        confidence_level : float, default 0.95
            VaR / CVaR 置信水平
        trading_days : int, default 252
            年化交易天数
        """
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be a BacktestResult instance")
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0,1), got {confidence_level}"
            )

        self.result = result
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.trading_days = trading_days

        # 缓存计算结果
        self._cache: Dict[str, float] = {}

    @property
    def daily_returns(self) -> np.ndarray:
        """获取日收益率序列（缓存）。"""
        eq = self.result.equity_curve
        if eq.empty:
            return np.array([])
        if "returns" in eq.columns:
            return eq["returns"].values
        return eq["total_value"].pct_change().fillna(0.0).values

    def sortino_ratio(self) -> float:
        """Sortino 比率：仅考虑下行波动。"""
        rets = self.daily_returns
        if len(rets) < 2:
            return 0.0
        daily_rf = self.risk_free_rate / self.trading_days
        excess = rets - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 0.0
        downside_vol = np.std(downside, ddof=1)
        if downside_vol < 1e-10:
            return 0.0
        annual_excess = float(np.mean(excess)) * self.trading_days
        annual_downside = downside_vol * np.sqrt(self.trading_days)
        return annual_excess / annual_downside

    def var(self) -> float:
        """Value at Risk (历史模拟法)。"""
        rets = self.daily_returns
        if len(rets) < 2:
            return 0.0
        return float(np.percentile(rets, (1 - self.confidence_level) * 100))

    def cvar(self) -> float:
        """Conditional VaR (CVaR / Expected Shortfall)。"""
        rets = self.daily_returns
        if len(rets) < 2:
            return 0.0
        threshold = np.percentile(rets, (1 - self.confidence_level) * 100)
        tail = rets[rets <= threshold]
        if len(tail) == 0:
            return threshold
        return float(np.mean(tail))

    def downside_volatility(self) -> float:
        """下行波动率（年化）。"""
        rets = self.daily_returns
        if len(rets) < 2:
            return 0.0
        downside = rets[rets < 0]
        if len(downside) == 0:
            return 0.0
        return float(np.std(downside, ddof=1) * np.sqrt(self.trading_days))

    def upside_volatility(self) -> float:
        """上行波动率（年化）。"""
        rets = self.daily_returns
        if len(rets) < 2:
            return 0.0
        upside = rets[rets > 0]
        if len(upside) == 0:
            return 0.0
        return float(np.std(upside, ddof=1) * np.sqrt(self.trading_days))

    def ulcer_index(self) -> float:
        """Ulcer Index：回撤深度与持续时间的综合度量。"""
        eq = self.result.equity_curve
        if eq.empty or "total_value" not in eq.columns:
            return 0.0
        values = eq["total_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        squared_dd = drawdown ** 2
        return float(np.sqrt(np.mean(squared_dd)))

    def sterling_ratio(self) -> float:
        """Sterling 比率：年化收益 / 平均回撤。"""
        ann_ret = self.result.annual_return
        mdds = self._average_drawdown()
        if abs(mdds) < 1e-10:
            return 0.0
        return ann_ret / abs(mdds)

    def _average_drawdown(self) -> float:
        """计算平均回撤深度。"""
        eq = self.result.equity_curve
        if eq.empty or "total_value" not in eq.columns:
            return 0.0
        values = eq["total_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        negative = drawdown[drawdown < 0]
        if len(negative) == 0:
            return 0.0
        return float(np.mean(negative))

    def tail_ratio(self) -> float:
        """尾部比率：95% 分位收益 / 5% 分位收益。"""
        rets = self.daily_returns
        if len(rets) < 10:
            return 1.0
        p95 = np.percentile(rets, 95)
        p05 = np.percentile(rets, 5)
        if abs(p05) < 1e-10:
            return 0.0
        return p95 / abs(p05)

    def skewness(self) -> float:
        """日收益率偏度。"""
        rets = self.daily_returns
        if len(rets) < 3:
            return 0.0
        return float(pd.Series(rets).skew())

    def kurtosis(self) -> float:
        """日收益率峰度（超额峰度，正态分布=3）。"""
        rets = self.daily_returns
        if len(rets) < 4:
            return 0.0
        return float(pd.Series(rets).kurtosis()) + 3.0

    def to_dict(self) -> Dict[str, float]:
        """将所有指标汇总为字典。"""
        return {
            "sortino_ratio": self.sortino_ratio(),
            "var_95": self.var(),
            "cvar_95": self.cvar(),
            "downside_volatility": self.downside_volatility(),
            "upside_volatility": self.upside_volatility(),
            "ulcer_index": self.ulcer_index(),
            "sterling_ratio": self.sterling_ratio(),
            "tail_ratio": self.tail_ratio(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
        }

    def summary_str(self, decimal: int = 4) -> str:
        """生成文本摘要。"""
        metrics = self.to_dict()
        lines = ["=== RiskMetrics Summary ==="]
        lines.append(f"{'Metric':<25} {'Value':>12}")
        lines.append("-" * 40)
        for k, v in metrics.items():
            lines.append(f"{k:<25} {v:>12.{decimal}f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 多策略结果对比分析 (PortfolioAnalyzer)
# ---------------------------------------------------------------------------


class PortfolioAnalyzer:
    """组合/多策略结果对比分析器。

    支持将多个策略的回测结果整合分析，包括等权组合或自定义权重组合的
    权益曲线合成、相关性矩阵、风险分解等。
    """

    def __init__(self, results: Dict[str, BacktestResult]) -> None:
        """
        Parameters
        ----------
        results : dict of {str: BacktestResult}
            策略名到回测结果的映射
        """
        if not results:
            raise ValueError("results dict must be non-empty")
        self.results = results

    def _returns_frame(self) -> pd.DataFrame:
        """收集各策略日收益率并按 datetime 对齐（P1-Q13-fix H6）。"""
        rets_dict: Dict[str, pd.Series] = {}
        for name, res in self.results.items():
            eq = res.equity_curve
            if eq.empty or "returns" not in eq.columns or "datetime" not in eq.columns:
                continue
            s = eq["returns"].copy()
            s.index = pd.to_datetime(eq["datetime"])
            rets_dict[name] = s

        if not rets_dict:
            return pd.DataFrame()
        df = pd.DataFrame(rets_dict)
        return df.sort_index()

    def correlation_matrix(self) -> pd.DataFrame:
        """计算各策略日收益率的相关性矩阵（按 datetime 对齐）。"""
        df = self._returns_frame()
        if df.empty:
            return pd.DataFrame()
        return df.corr()

    def equal_weight_portfolio(self) -> BacktestResult:
        """生成等权组合的回测结果（按 datetime 对齐后等权）。"""
        df = self._returns_frame()
        if df.empty or df.shape[1] == 0:
            return BacktestResult(
                equity_curve=pd.DataFrame(
                    columns=["datetime", "total_value", "cash",
                             "position_value", "returns"]
                ),
                trades=[],
                orders=[],
            )

        # P1-Q13-fix(H6): 按 datetime 对齐后，每行取当日有效策略的等权收益（不再按行位置对齐）
        portfolio_rets = df.mean(axis=1, skipna=True)
        portfolio_rets = portfolio_rets.dropna()

        # 构建组合净值
        initial_value = 1.0
        equity = [initial_value]
        for r in portfolio_rets:
            equity.append(equity[-1] * (1 + r))
        equity = equity[1:]

        eq_df = pd.DataFrame(
            {
                "datetime": portfolio_rets.index,
                "total_value": equity,
                "cash": equity,
                "position_value": [0.0] * len(equity),
                "returns": portfolio_rets.values,
            }
        )

        # 合并所有标的的 trade / order
        all_trades: List[TradeData] = []
        all_orders: List[OrderData] = []
        for res in self.results.values():
            all_trades.extend(res.trades)
            all_orders.extend(res.orders)

        # 用 BacktestEngine 计算组合指标
        engine = BacktestEngine()
        engine.initial_capital = 1_000_000.0
        engine.capital = 1_000_000.0
        result = BacktestResult(
            equity_curve=eq_df,
            trades=all_trades,
            orders=all_orders,
            total_trades=len(all_trades),
        )

        # 手动计算各项指标
        total_return = equity[-1] / initial_value - 1.0 if equity else 0.0
        result.total_return = total_return

        dt_min = eq_df["datetime"].iloc[0] if not eq_df.empty else datetime.now()
        dt_max = eq_df["datetime"].iloc[-1] if not eq_df.empty else datetime.now()
        years = max(
            (dt_max - dt_min).total_seconds() / (365.25 * 86400), 1.0 / 365.0
        )
        # 审计 2026-08-16：极端亏损(total_return<=-1 → 本金亏光/倒欠)时
        # (1+total_return)^(1/years) 为负数分数次幂 → NaN；显式返回 -1（亏光）
        if total_return <= -1.0:
            result.annual_return = -1.0
        else:
            result.annual_return = (1 + total_return) ** (1.0 / years) - 1.0

        daily_returns = portfolio_rets.values
        # P2-Q13-fix(L4): 与主回测路径一致，年化因子按 bar 频率感知
        annual_factor = engine._infer_annualization_factor(eq_df)
        daily_vol = float(np.std(daily_returns, ddof=1))
        result.annual_volatility = daily_vol * math.sqrt(annual_factor)

        rf_rate = 0.02
        daily_rf = rf_rate / annual_factor
        excess = daily_returns - daily_rf
        if daily_vol > 1e-10:
            result.sharpe_ratio = (
                float(np.mean(excess) / daily_vol) * math.sqrt(annual_factor)
            )

        cum_max = np.maximum.accumulate(equity)
        drawdown = (np.array(equity) - cum_max) / cum_max
        result.max_drawdown = float(np.min(drawdown))

        # Calmar
        if abs(result.max_drawdown) > 1e-10:
            result.calmar_ratio = result.annual_return / abs(result.max_drawdown)

        return result

    def ranking(self, metric: str = "sharpe_ratio") -> pd.DataFrame:
        """按指定指标排序策略。"""
        rows = []
        for name, res in self.results.items():
            rows.append(
                {
                    "strategy": name,
                    "total_return": res.total_return,
                    "annual_return": res.annual_return,
                    "annual_volatility": res.annual_volatility,
                    "sharpe_ratio": res.sharpe_ratio,
                    "max_drawdown": res.max_drawdown,
                    "win_rate": res.win_rate,
                    "profit_loss_ratio": res.profit_loss_ratio,
                    "total_trades": res.total_trades,
                    "calmar_ratio": res.calmar_ratio,
                    "information_ratio": res.information_ratio,
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        return df

    def plot_data(self) -> pd.DataFrame:
        """准备用于绘图的净值曲线数据。

        Returns
        -------
        pd.DataFrame
            columns=["datetime", "strategy", "equity"] 的 long-form DataFrame
        """
        records = []
        for name, res in self.results.items():
            eq = res.equity_curve
            if eq.empty:
                continue
            dt_list = eq["datetime"].tolist()
            val_list = eq["total_value"].tolist()
            for dt, val in zip(dt_list, val_list):
                records.append({"datetime": dt, "strategy": name, "equity": val})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 回测报告输出工具 (BacktestReporter)
# ---------------------------------------------------------------------------


class BacktestReporter:
    """回测结果报告生成器。

    将 BacktestResult 转换为易读的文本报告或结构化数据。
    """

    def __init__(self, result: BacktestResult, name: str = "Strategy") -> None:
        self.result = result
        self.name = name

    def text_report(self, decimal: int = 4) -> str:
        """生成文本格式的回测报告。"""
        r = self.result
        lines = [
            f"{'=' * 60}",
            f"  回测报告: {self.name}",
            f"{'=' * 60}",
            "",
            f"{'指标':<30} {'数值':>12}",
            "-" * 42,
            f"{'总收益率 (Total Return)':<30} {r.total_return:>12.{decimal}%}",
            f"{'年化收益率 (Annual Return)':<30} {r.annual_return:>12.{decimal}%}",
            f"{'年化波动率 (Annual Volatility)':<30} {r.annual_volatility:>12.{decimal}%}",
            f"{'夏普比率 (Sharpe Ratio)':<30} {r.sharpe_ratio:>12.{decimal}f}",
            f"{'最大回撤 (Max Drawdown)':<30} {r.max_drawdown:>12.{decimal}%}",
            f"{'最大回撤持续期 (天)':<30} {r.max_drawdown_duration:>12d}",
            f"{'胜率 (Win Rate)':<30} {r.win_rate:>12.{decimal}%}",
            f"{'盈亏比 (PL Ratio)':<30} {r.profit_loss_ratio:>12.{decimal}f}",
            f"{'总交易次数':<30} {r.total_trades:>12d}",
            f"{'卡玛比率 (Calmar Ratio)':<30} {r.calmar_ratio:>12.{decimal}f}",
            f"{'信息比率 (Info Ratio)':<30} {r.information_ratio:>12.{decimal}f}",
            f"{'Alpha':<30} {r.alpha:>12.{decimal}f}",
            f"{'Beta':<30} {r.beta:>12.{decimal}f}",
            "",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)

    def trades_report(self) -> str:
        """生成成交明细报告。"""
        trades = self.result.trades
        if not trades:
            return "无成交记录。"

        lines = [
            f"{'=' * 90}",
            f"  成交明细: {self.name}  (共 {len(trades)} 笔)",
            f"{'=' * 90}",
            "",
            f"{'成交ID':<15} {'订单ID':<15} {'标的':<10} {'方向':<8} "
            f"{'开平':<6} {'价格':<10} {'数量':<10} {'时间':<20}",
            "-" * 90,
        ]
        for t in trades:
            lines.append(
                f"{t.trade_id:<15} {t.order_id:<15} {t.symbol:<10} {t.direction:<8} "
                f"{t.offset:<6} {t.price:<10.4f} {t.volume:<10.0f} {t.datetime:<20}"
            )
        return "\n".join(lines)

    def orders_report(self) -> str:
        """生成订单明细报告。"""
        orders = self.result.orders
        if not orders:
            return "无订单记录。"

        lines = [
            f"{'=' * 100}",
            f"  订单明细: {self.name}  (共 {len(orders)} 笔)",
            f"{'=' * 100}",
            "",
            f"{'订单ID':<15} {'标的':<10} {'方向':<8} {'开平':<6} "
            f"{'类型':<8} {'价格':<10} {'数量':<10} {'已成交':<10} {'状态':<12} {'时间':<20}",
            "-" * 100,
        ]
        for o in orders:
            dt_str = str(o.datetime) if o.datetime else ""
            lines.append(
                f"{o.order_id:<15} {o.symbol:<10} {o.direction:<8} {o.offset:<6} "
                f"{o.order_type:<8} {o.price:<10.4f} {o.volume:<10.0f} "
                f"{o.filled:<10.0f} {o.status:<12} {dt_str:<20}"
            )
        return "\n".join(lines)

    def full_report(self) -> str:
        """生成完整回测报告。"""
        parts = [
            self.text_report(),
            "",
            self.trades_report(),
            "",
            self.orders_report(),
        ]
        return "\n".join(parts)

    def to_csv(self, trade_path: str = "", order_path: str = "") -> None:
        """将成交和订单明细导出 CSV。"""
        trades = self.result.trades
        if trades and trade_path:
            rows = []
            for t in trades:
                rows.append(
                    {
                        "trade_id": t.trade_id,
                        "order_id": t.order_id,
                        "symbol": t.symbol,
                        "direction": t.direction,
                        "offset": t.offset,
                        "price": t.price,
                        "volume": t.volume,
                        "datetime": t.datetime,
                    }
                )
            pd.DataFrame(rows).to_csv(trade_path, index=False)

        orders = self.result.orders
        if orders and order_path:
            rows = []
            for o in orders:
                rows.append(
                    {
                        "order_id": o.order_id,
                        "symbol": o.symbol,
                        "direction": o.direction,
                        "offset": o.offset,
                        "price": o.price,
                        "volume": o.volume,
                        "filled": o.filled,
                        "status": o.status,
                        "order_type": o.order_type,
                        "datetime": o.datetime,
                        "strategy_name": o.strategy_name,
                    }
                )
            pd.DataFrame(rows).to_csv(order_path, index=False)


# ---------------------------------------------------------------------------
# 内置示例策略 (Example Strategies)
# ---------------------------------------------------------------------------


class SmaCrossStrategy(StrategyTemplate):
    """双均线交叉策略。

    快线上穿慢线 → 买入开多
    快线下穿慢线 → 卖出平多
    """

    def __init__(self) -> None:
        super().__init__()
        self.fast: int = 5
        self.slow: int = 20
        # P2-Q13-fix(M8): 按 symbol 分桶维护价格序列（多标的回测避免跨标的价格混入）
        self.prices: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData) -> None:
        prices = self.prices.setdefault(bar.symbol, [])
        prices.append(bar.close)
        if len(prices) < self.slow + 1:
            return

        fast_ma = np.mean(prices[-self.fast:])
        slow_ma = np.mean(prices[-self.slow:])
        prev_fast = np.mean(prices[-(self.fast + 1): -1])
        prev_slow = np.mean(prices[-(self.slow + 1): -1])

        pos = self.bg.positions.get(bar.symbol) if self.bg else None

        # 金叉: prev_fast <= prev_slow and fast_ma > slow_ma
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            if pos is None or pos.volume <= 1e-8 or pos.direction == "net":
                self.buy(bar.close, 1000)

        # 死叉: prev_fast >= prev_slow and fast_ma < slow_ma
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            if pos is not None and pos.volume > 1e-8 and pos.direction == "long":
                self.sell(bar.close, pos.volume)


class BbiStrategy(StrategyTemplate):
    """BBI (多空指标) 策略。

    BBI = (MA3 + MA6 + MA12 + MA24) / 4
    价格上穿 BBI → 做多
    价格下穿 BBI → 平多/做空
    """

    def __init__(self) -> None:
        super().__init__()
        self.periods: List[int] = [3, 6, 12, 24]
        # P2-Q13-fix(M8): 按 symbol 分桶维护价格序列
        self.prices: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData) -> None:
        prices = self.prices.setdefault(bar.symbol, [])
        prices.append(bar.close)
        max_period = max(self.periods)
        if len(prices) < max_period + 1:
            return

        mas = [np.mean(prices[-p:]) for p in self.periods]
        bbi = np.mean(mas)
        prev_mas = [np.mean(prices[-(p + 1): -1]) for p in self.periods]
        prev_bbi = np.mean(prev_mas)

        pos = self.bg.positions.get(bar.symbol) if self.bg else None

        if prev_bbi >= prices[-2] and bbi < prices[-1]:
            # 下穿BBI
            if pos is not None and pos.volume > 1e-8 and pos.direction == "long":
                self.sell(bar.close, pos.volume)
        elif prev_bbi <= prices[-2] and bbi > prices[-1]:
            # 上穿BBI
            if pos is None or pos.volume <= 1e-8 or pos.direction == "net":
                self.buy(bar.close, 1000)


class RsiStrategy(StrategyTemplate):
    """RSI 反转策略。

    RSI < oversold → 买入
    RSI > overbought → 卖出
    """

    def __init__(self) -> None:
        super().__init__()
        self.period: int = 14
        self.oversold: float = 30.0
        self.overbought: float = 70.0
        # P2-Q13-fix(M8): 按 symbol 分桶维护价格序列
        self.prices: Dict[str, List[float]] = {}

    @staticmethod
    def compute_rsi(prices: Sequence[float], period: int) -> float:
        """计算 RSI。"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = deltas[deltas > 0].sum() if np.any(deltas > 0) else 0.0
        losses = (-deltas[deltas < 0]).sum() if np.any(deltas < 0) else 1e-10
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: BarData) -> None:
        prices = self.prices.setdefault(bar.symbol, [])
        prices.append(bar.close)
        if len(prices) < self.period + 1:
            return

        rsi = self.compute_rsi(prices, self.period)
        prev_rsi = self.compute_rsi(prices[:-1], self.period)
        pos = self.bg.positions.get(bar.symbol) if self.bg else None

        if prev_rsi > self.oversold and rsi <= self.oversold:
            # 进入超卖区 → 买入
            if pos is None or pos.volume <= 1e-8 or pos.direction == "net":
                self.buy(bar.close, 1000)
        elif prev_rsi < self.overbought and rsi >= self.overbought:
            # 进入超买区 → 卖出
            if pos is not None and pos.volume > 1e-8 and pos.direction == "long":
                self.sell(bar.close, pos.volume)


class BollingerStrategy(StrategyTemplate):
    """布林带策略。

    价格触及下轨 → 买入
    价格触及上轨 → 卖出
    """

    def __init__(self) -> None:
        super().__init__()
        self.period: int = 20
        self.num_std: float = 2.0
        # P2-Q13-fix(M8): 按 symbol 分桶维护价格序列
        self.prices: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData) -> None:
        prices = self.prices.setdefault(bar.symbol, [])
        prices.append(bar.close)
        if len(prices) < self.period:
            return

        window = prices[-self.period:]
        middle = np.mean(window)
        std = np.std(window, ddof=1)
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        pos = self.bg.positions.get(bar.symbol) if self.bg else None

        if bar.close <= lower:
            if pos is None or pos.volume <= 1e-8 or pos.direction == "net":
                self.buy(bar.close, 1000)
        elif bar.close >= upper:
            if pos is not None and pos.volume > 1e-8 and pos.direction == "long":
                self.sell(bar.close, pos.volume)


class TurtleStrategy(StrategyTemplate):
    """海龟交易策略（简化版）。

    突破N日高点 → 买入
    跌破N日低点 → 卖出
    """

    def __init__(self) -> None:
        super().__init__()
        self.entry_period: int = 20
        self.exit_period: int = 10
        # P2-Q13-fix(M8): 按 symbol 分桶维护高低价序列
        self.highs: Dict[str, List[float]] = {}
        self.lows: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarData) -> None:
        highs = self.highs.setdefault(bar.symbol, [])
        lows = self.lows.setdefault(bar.symbol, [])
        highs.append(bar.high)
        lows.append(bar.low)

        if len(highs) < max(self.entry_period, self.exit_period):
            return

        entry_high = max(highs[-self.entry_period:])
        exit_low = min(lows[-self.exit_period:])

        pos = self.bg.positions.get(bar.symbol) if self.bg else None

        if bar.close >= entry_high * 0.99:
            if pos is None or pos.volume <= 1e-8 or pos.direction == "net":
                self.buy(bar.close, 1000)
        elif bar.close <= exit_low * 1.01:
            if pos is not None and pos.volume > 1e-8 and pos.direction == "long":
                self.sell(bar.close, pos.volume)


# ---------------------------------------------------------------------------
# 数据加载与预处理工具
# ---------------------------------------------------------------------------


def load_csv_data(
    filepath: str,
    symbol: str = "",
    datetime_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    amount_col: str = "amount",
    date_format: str = "",
    **kwargs: Any,
) -> pd.DataFrame:
    """从 CSV 文件加载 K 线数据。

    Parameters
    ----------
    filepath : str
        CSV 文件路径
    symbol : str, optional
        标的代码（为空则从文件名推断）
    datetime_col : str, default "date"
    open_col : str, default "open"
    high_col : str, default "high"
    low_col : str, default "low"
    close_col : str, default "close"
    volume_col : str, default "volume"
    amount_col : str, default "amount"
    date_format : str, optional
        pd.read_csv 的 parse_dates 参数格式
    **kwargs : Any
        传递给 pd.read_csv 的其他参数

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, **kwargs)
    if datetime_col in df.columns:
        if date_format:
            df[datetime_col] = pd.to_datetime(df[datetime_col], format=date_format)
        else:
            df[datetime_col] = pd.to_datetime(df[datetime_col])

    # P2-Q13-fix(L2): 实现 docstring 声称的 symbol 推断——为空时从文件名推断并附到结果
    if not symbol:
        symbol = os.path.splitext(os.path.basename(filepath))[0]
    if symbol:
        df["symbol"] = symbol
    return df


def data_split(
    df: pd.DataFrame,
    split_date: str,
    datetime_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按日期分割数据集。

    Parameters
    ----------
    df : pd.DataFrame
    split_date : str
        分割日期（YYYY-MM-DD），该日期之前为训练集，之后为测试集
    datetime_col : str, default "date"

    Returns
    -------
    (train_df, test_df)
    """
    split_dt = pd.Timestamp(split_date)
    train = df[df[datetime_col] < split_dt].copy()
    test = df[df[datetime_col] >= split_dt].copy()
    return train, test


def resample_bars(
    df: pd.DataFrame,
    rule: str = "W",
    datetime_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """将 K 线重采样到更高时间周期。

    Parameters
    ----------
    df : pd.DataFrame
    rule : str, default "W"
        pandas resample rule: "D", "W", "M", "H", "30min", etc.
    datetime_col : str, default "date"
    open_col : str, default "open"
    high_col : str, default "high"
    low_col : str, default "low"
    close_col : str, default "close"
    volume_col : str, default "volume"
    amount_col : str, default "amount"

    Returns
    -------
    pd.DataFrame
        重采样后的 DataFrame，包含 OHLCV
    """
    dfc = df.copy()
    dfc[datetime_col] = pd.to_datetime(dfc[datetime_col])
    dfc = dfc.set_index(datetime_col)

    agg_dict: Dict[str, Callable] = {
        open_col: "first",
        high_col: "max",
        low_col: "min",
        close_col: "last",
    }
    if volume_col in dfc.columns:
        agg_dict[volume_col] = "sum"
    if amount_col in dfc.columns:
        agg_dict[amount_col] = "sum"

    resampled = dfc.resample(rule).agg(agg_dict).dropna()
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={resampled.columns[0]: datetime_col})

    # P2-Q13-fix(L5): 首行 open / 末行 close 已由 agg 的 "first"/"last" 正确聚合，无需额外修复
    return resampled


# ---------------------------------------------------------------------------
# 回测结果保存与加载
# ---------------------------------------------------------------------------


def save_backtest_result(
    result: BacktestResult,
    path: str,
    suffix: str = "",
) -> None:
    """将回测结果保存到磁盘。

    保存为 HDF5 或 pickle 格式。

    Parameters
    ----------
    result : BacktestResult
    path : str
        保存路径（目录），会自动创建子目录
    suffix : str, optional
        文件名后缀
    """
    import os
    import pickle

    os.makedirs(path, exist_ok=True)

    name_suffix = f"_{suffix}" if suffix else ""
    eq_path = os.path.join(path, f"equity_curve{name_suffix}.csv")
    result.equity_curve.to_csv(eq_path, index=False)

    # 用 pickle 保存完整对象（不含 equity_curve）
    obj_path = os.path.join(path, f"result{name_suffix}.pkl")
    save_obj = BacktestResult(
        equity_curve=pd.DataFrame(),  # 不重复存 equity_curve
        trades=result.trades,
        orders=result.orders,
        total_return=result.total_return,
        annual_return=result.annual_return,
        annual_volatility=result.annual_volatility,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        max_drawdown_duration=result.max_drawdown_duration,
        win_rate=result.win_rate,
        profit_loss_ratio=result.profit_loss_ratio,
        total_trades=result.total_trades,
        calmar_ratio=result.calmar_ratio,
        information_ratio=result.information_ratio,
        alpha=result.alpha,
        beta=result.beta,
    )
    try:
        with open(obj_path, "wb") as f:
            pickle.dump(save_obj, f)
    except (OSError, PermissionError) as e:
        warnings.warn(f"保存 result pickle 失败 ({obj_path}): {e}")


def load_backtest_result(path: str, suffix: str = "") -> Optional[BacktestResult]:
    """从磁盘加载回测结果。

    Parameters
    ----------
    path : str
        保存目录
    suffix : str, optional
        文件名后缀

    Returns
    -------
    BacktestResult
    """
    import os
    import pickle

    name_suffix = f"_{suffix}" if suffix else ""
    eq_path = os.path.join(path, f"equity_curve{name_suffix}.csv")
    obj_path = os.path.join(path, f"result{name_suffix}.pkl")

    try:
        eq_df = pd.read_csv(eq_path, parse_dates=["datetime"])
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        warnings.warn(f"加载 equity_curve 失败 ({eq_path}): {e}")
        return None

    try:
        with open(obj_path, "rb") as f:
            obj: BacktestResult = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        warnings.warn(f"加载 result pickle 失败 ({obj_path}): {e}")
        return None

    obj.equity_curve = eq_df
    return obj


def merge_walk_forward_results(
    results: List[BacktestResult],
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """合并 Walk-Forward 回测的多个阶段结果。

    将多个验证期的 equity_curve 拼接，形成完整的净值曲线。

    Parameters
    ----------
    results : list of BacktestResult
    initial_capital : float, default 1_000_000

    Returns
    -------
    BacktestResult
    """
    if not results:
        return BacktestResult(
            equity_curve=pd.DataFrame(
                columns=["datetime", "total_value", "cash",
                         "position_value", "returns"]
            ),
            trades=[],
            orders=[],
        )

    # 拼接 equity_curve（P1-Q13-fix H7: 各窗口均以 initial_capital 起步，
    # 按前一窗口期末净值复利链式缩放，消除拼接边界的锯齿跳变）
    eq_parts = []
    running = initial_capital
    for res in results:
        eq = res.equity_curve
        if eq.empty or "total_value" not in eq.columns:
            continue
        eq = eq.copy()
        start_val = eq["total_value"].iloc[0]
        scale = running / start_val if start_val > 0 else 1.0
        eq["total_value"] = eq["total_value"] * scale
        if "cash" in eq.columns:
            eq["cash"] = eq["cash"] * scale
        if "position_value" in eq.columns:
            eq["position_value"] = eq["position_value"] * scale
        if "returns" in eq.columns:
            eq["returns"] = eq["returns"].fillna(0.0)
        running = eq["total_value"].iloc[-1]
        eq_parts.append(eq)

    if eq_parts:
        full_eq = pd.concat(eq_parts, ignore_index=True).drop_duplicates(
            subset=["datetime"]
        ).sort_values("datetime").reset_index(drop=True)
    else:
        full_eq = pd.DataFrame(
            columns=["datetime", "total_value", "cash",
                     "position_value", "returns"]
        )

    # 合并 trade & order
    all_trades: List[TradeData] = []
    all_orders: List[OrderData] = []
    for res in results:
        all_trades.extend(res.trades)
        all_orders.extend(res.orders)

    merged = BacktestResult(
        equity_curve=full_eq,
        trades=all_trades,
        orders=all_orders,
        total_trades=len(all_trades),
    )

    # 计算合并后的指标
    if not full_eq.empty and len(full_eq) >= 2:
        merged.total_return = (
            full_eq["total_value"].iloc[-1] / initial_capital - 1.0
        )
        # 重算 returns（链式拼接后原 returns 不再连续有效）
        full_eq = full_eq.copy()
        full_eq["returns"] = full_eq["total_value"].pct_change().fillna(0.0)
        merged.equity_curve = full_eq
        dt_min = full_eq["datetime"].iloc[0]
        dt_max = full_eq["datetime"].iloc[-1]
        years = max(
            (dt_max - dt_min).total_seconds() / (365.25 * 86400), 1.0 / 365.0
        )
        merged.annual_return = (1 + merged.total_return) ** (1.0 / years) - 1.0

    return merged


# ---------------------------------------------------------------------------
# 常量与版本信息
# ---------------------------------------------------------------------------


__version__ = "1.0.0"
__author__ = "OpenClaw Quant System"
__description__ = "Event-Driven Backtesting Engine for Python"


# 支持常用数据频率的周期映射（供引用参考）
FREQ_MAP: Dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
    "D": 1440,
    "W": 10080,
    "M": 43200,
}

# 中国A股市场常量
A_SHARE_CONSTANTS = {
    "size": 100,  # 每手股数
    "limit_up": 0.10,  # 涨停幅度
    "limit_down": -0.10,  # 跌停幅度
    "commission_rate": COMMISSION_RATE,  # 佣金费率万0.85
    "stamp_tax_rate": 0.0005,  # 印花税率（2023年8月起减半至万5）
    "min_commission": 5.0,  # P1-Q13-fix(H1): 最低佣金5元/笔（A股基准）
    "transfer_fee_rate": 0.00001,  # P1-Q13-fix(H2): 过户费 0.001% 双边（2022-04-29 起沪深A股统一）
    "trading_days": 252,  # 年化交易日
}

# 期货市场常量
FUTURES_CONSTANTS = {
    "size": 1,  # 每手
    "commission_rate": 0.0001,  # 手续费率
    "margin_rate": 0.10,  # 保证金率
    "trading_days": 252,
}


# ---------------------------------------------------------------------------
# 策略分析辅助函数
# ---------------------------------------------------------------------------


def compute_sharpe(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
    annualize: bool = True,
) -> float:
    """计算 Sharpe 比率。

    V12.3 审计 P1-4: 统一口径转发——复用 metrics_calculator.MetricsCalculator.sharpe
    （扣 rf、样本标准差 ddof=1、年化 sqrt(252)）。本函数保持为兼容壳，避免本引擎
    调用方签名变更；数值与全系统 MetricsCalculator 严格一致。

    Parameters
    ----------
    returns : array-like
        收益率序列
    risk_free_rate : float, default 0.02
        年化无风险利率
    trading_days : int, default 252
    annualize : bool, default True
        是否年化

    Returns
    -------
    float
    """
    return MetricsCalculator.sharpe(
        returns, rf=risk_free_rate, ddof=1, trading_days=trading_days, annualize=annualize
    )


def compute_max_drawdown(
    equity: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """计算最大回撤。

    Parameters
    ----------
    equity : array-like
        净值序列

    Returns
    -------
    float
        最大回撤（负值，如 -0.15 表示 15% 回撤）
    """
    arr = np.asarray(equity, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(np.min(dd))


def compute_drawdown_duration(
    equity: Union[pd.Series, np.ndarray, List[float]],
) -> int:
    """计算最大回撤持续期（连续处于回撤状态的最大天数）。

    Parameters
    ----------
    equity : array-like
        净值序列

    Returns
    -------
    int
    """
    arr = np.asarray(equity, dtype=np.float64)
    if len(arr) < 2:
        return 0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    underwater = dd < -1e-8
    max_dur = 0
    cur = 0
    for flag in underwater:
        if flag:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def compute_sortino(
    returns: Union[pd.Series, np.ndarray, List[float]],
    target_return: float = 0.0,
    trading_days: int = 252,
) -> float:
    """计算 Sortino 比率。

    Parameters
    ----------
    returns : array-like
    target_return : float, default 0.0
        目标收益率（日化）
    trading_days : int, default 252

    Returns
    -------
    float
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    excess = arr - target_return
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = np.std(downside, ddof=1)
    if downside_vol < 1e-10:
        return 0.0
    mean_excess = float(np.mean(excess))
    sortino = mean_excess / downside_vol
    sortino *= math.sqrt(trading_days)
    return sortino


def compute_calmar(
    annual_return: float,
    max_drawdown: float,
) -> float:
    """计算 Calmar 比率。

    Parameters
    ----------
    annual_return : float
        年化收益率
    max_drawdown : float
        最大回撤（负值）

    Returns
    -------
    float
    """
    if abs(max_drawdown) < 1e-10:
        return 0.0
    return annual_return / abs(max_drawdown)


def compute_consecutive_wins_losses(
    trades: List[TradeData],
) -> Tuple[int, int]:
    """计算最大连胜次数和最大连败次数。

    Parameters
    ----------
    trades : list of TradeData
        成交记录

    Returns
    -------
    (max_consecutive_wins, max_consecutive_losses)
    """
    pnl_list = BacktestEngine._calculate_trade_pnl(trades)
    if not pnl_list:
        return 0, 0

    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0

    for pnl in pnl_list:
        if pnl > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        elif pnl < 0:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)

    return max_wins, max_losses


# ---------------------------------------------------------------------------
# 快捷回测入口 (quick_backtest)
# ---------------------------------------------------------------------------


def quick_backtest(
    strategy_class: Type[StrategyTemplate],
    data: Dict[str, pd.DataFrame],
    capital: float = 1_000_000.0,
    commission_rate: float = COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    slippage_mode: str = "percent",
    start: Optional[str] = None,
    end: Optional[str] = None,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    print_report: bool = True,
) -> BacktestResult:
    """快速回测入口——一行代码完成回测。

    Parameters
    ----------
    strategy_class : Type[StrategyTemplate]
    data : dict of {str: pd.DataFrame}
    capital : float
    commission_rate : float
    slippage_rate : float
    slippage_mode : str
    start : str, optional
    end : str, optional
    strategy_kwargs : dict, optional
        传递给策略构造器的参数（会通过 setattr 注入）
    print_report : bool, default True
        是否打印文本报告

    Returns
    -------
    BacktestResult
    """
    engine = BacktestEngine()
    engine.set_capital(capital)
    engine.set_commission(commission_rate)
    engine.set_slippage(slippage_rate, slippage_mode)

    for symbol, df in data.items():
        engine.add_data(symbol, df)

    strategy = engine.add_strategy(strategy_class)
    if strategy_kwargs:
        for k, v in strategy_kwargs.items():
            if hasattr(strategy, k):
                setattr(strategy, k, v)

    result = engine.run(start=start, end=end)

    if print_report:
        reporter = BacktestReporter(result, strategy.name)
        print(reporter.text_report())

    return result


# ---------------------------------------------------------------------------
# 策略性能对比 (BacktestComparison)
# ---------------------------------------------------------------------------


class BacktestComparison:
    """回测结果对比——可视化与其他分析的基础结构。"""

    def __init__(self) -> None:
        self.results: Dict[str, BacktestResult] = {}

    def add_result(self, name: str, result: BacktestResult) -> None:
        """添加回测结果。"""
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be a BacktestResult")
        self.results[name] = result

    def summary(self) -> pd.DataFrame:
        """生成对比摘要表。"""
        analyzer = PortfolioAnalyzer(self.results)
        return analyzer.ranking()

    def best(self, metric: str = "sharpe_ratio") -> Tuple[str, BacktestResult]:
        """返回指定指标最优的策略。"""
        df = self.summary()
        if df.empty or metric not in df.columns:
            if self.results:
                name = list(self.results.keys())[0]
                return name, self.results[name]
            raise ValueError("No results available")
        best_name = df.iloc[0]["strategy"]
        return best_name, self.results[best_name]

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, name: str) -> BacktestResult:
        return self.results[name]

    def __contains__(self, name: str) -> bool:
        return name in self.results

