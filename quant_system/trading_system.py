"""
trading_system.py — 完整交易执行系统。

V4.1 feature: trading_system

本模块提供一个可独立运行、也可接入外部券商网关的交易执行闭环：

1. OrderManager
   - 订单生命周期管理：pending → submitted → partial → filled/cancelled/rejected
   - 订单簿 OrderBook 维护
   - 撤单、改单、状态同步、批量下单
2. PositionManager
   - 多标的多空持仓管理
   - 加权平均 / FIFO 成本核算
   - 浮动盈亏、已实现盈亏、仓位限制、调仓指令、P&L 报告
3. PreTradeRiskCheck
   - 资金、保证金、集中度、行业、日内次数、涨跌停、偏离度、合规检查
4. TCA
   - Arrival Cost、VWAP Shortfall、Implementation Shortfall
   - 市场冲击、滑点统计、时间滑点、日度/标的报告
5. TradeReporter
   - 成交记录、佣金税费、每日汇总、CSV/Excel 导出
6. TradingSystem
   - 策略信号到订单、风控、模拟成交、对账、综合报告

设计说明：
- 顶层系统默认使用 sim 模式，便于研究系统直接验证策略执行链路。
- 实盘 broker_type 会保留订单提交状态，具体柜台对接可在外部同步状态。
- numpy/sklearn 按要求使用 try/except 延迟导入；pandas 也做可选导入。
- 所有金额单位默认人民币元，数量单位默认股，费用率均为 decimal。

D3收敛登记 (2026-08-11): 执行域券商成本模型唯一真源 = execution_broker.py
（P2-Q28 文档化：佣金万0.85/最低5元/印花税万5仅卖出/过户费万0.1双边/滑点）。
费率常量转发 execution_broker 模块级常量；TradeReporter.calculate_fee 费用公式
已薄壳转发 execution_broker.estimate_trade_cost（零额保护与 TradeFee 返回结构
保留本地语义）。OrderManager/PositionManager/PreTradeRiskCheck/TCA/TradingSystem
等执行/策略/风控/状态机能力与 broker 模型签名不兼容，标注 'D3收敛: 能力未合并'
或 'D3收敛登记: 独立能力保留'，保留独立实现（不强迁）。
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from quant_system.utils import to_float as _safe_float

try:  # pandas 不是延迟导入要求的一部分，但这里保持可选，便于最小环境编译。
    import pandas as pd
except Exception:  # pragma: no cover - 运行环境缺 pandas 时走降级路径。
    pd = None  # type: ignore[assignment]


# =============================================================================
# 基础常量与工具函数
# =============================================================================

ACTIVE_ORDER_STATUSES: frozenset[str] = frozenset({"pending", "submitted", "partial"})
TERMINAL_ORDER_STATUSES: frozenset[str] = frozenset({"filled", "cancelled", "rejected", "expired"})
VALID_ORDER_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"submitted", "cancelled", "rejected", "expired"},
    "submitted": {"partial", "filled", "cancelled", "rejected", "expired"},
    "partial": {"partial", "filled", "cancelled", "rejected", "expired"},
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
    "expired": set(),
}

# P1-Q18-fix: 外部柜台状态同步白名单。sync_order_status 只允许直写这些柜台字段，
# 其余参数一律落入 tags，防止外部直写 filled_quantity/remaining_quantity/status 等
# 派生字段破坏订单状态机不变量。
ORDER_SYNC_WHITELIST: frozenset[str] = frozenset({
    "broker_order_id",
    "client_order_id",
    "parent_order_id",
    "submitted_at",
    "completed_at",
    "reject_reason",
    "cancel_reason",
    "avg_fill_price",
})

# V4.1 feature: trading_system — 与 execution.py 的状态机命名保持显式映射。
# execution.py 使用 pending/approved/sent/partial_filled，交易系统门面使用
# pending/submitted/partial。这里不强行改写本模块状态机，而是在需要落库或对接
# 旧执行引擎时做窄适配，避免破坏用户要求的生命周期命名。
EXECUTION_STATE_MAP: dict[str, str] = {
    "pending": "pending",
    "submitted": "sent",
    "partial": "partial_filled",
    "filled": "filled",
    "cancelled": "cancelled",
    "rejected": "rejected",
    "expired": "expired",
}

# V4.1 feature: trading_system — 与 execution_broker.py 简单 DTO 的字段映射。
# 本模块 Order/PositionRecord 字段更丰富，因此保留本地模型并提供转换方法。
BROKER_MODEL_FIELD_MAP: dict[str, str] = {
    "quantity": "volume",
    "filled_quantity": "filled_volume",
    "avg_fill_price": "avg_price",
    "avg_cost": "cost_price",
    "market_value": "market_value",
}

DEFAULT_LOT_SIZE: int = 100
# D3收敛 (2026-08-11): 费用率唯一真源 = execution_broker（P2-Q28 文档化），此处转发。
from quant_system.execution_broker import (
    COMMISSION_RATE as DEFAULT_COMMISSION_RATE,
    MIN_COMMISSION as DEFAULT_MIN_COMMISSION,
    STAMP_TAX_RATE as DEFAULT_STAMP_TAX_RATE,
    TRANSFER_FEE_RATE as DEFAULT_TRANSFER_FEE_RATE,
    estimate_trade_cost,
)
DEFAULT_MARGIN_RATE: float = 1.0
DEFAULT_SHORT_MARGIN_RATE: float = 1.3
DEFAULT_PRICE_DEVIATION_BPS: float = 300.0
DEFAULT_DAILY_TRADE_LIMIT: int = 200


def _now() -> datetime:
    """返回当前本地时间，集中封装便于测试替换。"""
    return datetime.now()


def _date_key(ts: datetime | None = None) -> str:
    """把时间戳转成 YYYY-MM-DD 字符串。"""
    return (ts or _now()).date().isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转 int，用于外部信号或数量字段容错。"""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _round_lot(quantity: float, lot_size: int = DEFAULT_LOT_SIZE) -> int:
    """按 A 股默认手数向下取整，保证交易数量合法。"""
    if lot_size <= 1:
        return int(quantity)
    sign = 1 if quantity >= 0 else -1
    return sign * (abs(int(quantity)) // lot_size) * lot_size


def _round_lot_sell(delta: int, current_qty: int, lot_size: int = DEFAULT_LOT_SIZE) -> int:
    """A 股卖出取整：delta 为负值（卖出）。

    规则：买入必须整手（100 股整数倍），但**零股可一次性卖出**——
    - 清仓（卖出量 >= 当前持仓）时，允许含零股一次性全部卖出；
    - 卖出后剩余为整手倍数时，允许把零股部分一并卖出；
    - 其余部分卖出仍按整手向下取整。

    实测场景：持仓 150 股清仓，delta=-150 → 返回 -150 而非 -100，
    避免剩余 50 股因 delta 取整为 0 永远卖不掉。
    """
    sell_qty = abs(delta)
    if sell_qty >= current_qty:
        return -current_qty
    remaining = current_qty - sell_qty
    if remaining % lot_size == 0:
        return -sell_qty
    return -_round_lot(sell_qty, lot_size)


def _side_sign(side: str) -> int:
    """买入为 +1，卖出为 -1。"""
    text = str(side).lower()
    if text in {"buy", "b", "long", "cover"}:
        return 1
    if text in {"sell", "s", "short"}:
        return -1
    raise ValueError(f"未知交易方向: {side}")


def _normalize_side(side: str) -> str:
    """把方向字段规范成 buy/sell。"""
    return "buy" if _side_sign(side) > 0 else "sell"


def _notional(price: float, quantity: int) -> float:
    """计算名义金额。"""
    return abs(price * quantity)


def _pct(num: float, den: float) -> float:
    """安全比例计算。"""
    return num / den if den else 0.0


def _bps(num: float, den: float) -> float:
    """安全基点计算。"""
    return _pct(num, den) * 10_000.0


def _lazy_numpy() -> Any | None:
    """V4.1 feature: trading_system — 延迟导入 numpy。"""
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def _lazy_sklearn_linear_regression() -> Any | None:
    """V4.1 feature: trading_system — 延迟导入 sklearn。"""
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
        return LinearRegression
    except Exception:
        return None


def _lazy_market_impact() -> Any | None:
    """延迟导入 market_impact.py，避免交易系统被研究依赖阻塞。"""
    try:
        from quant_system.market_impact import ACParams, MarketImpact
        return MarketImpact(ACParams.a_share_defaults())
    except Exception:
        try:
            from .market_impact import ACParams, MarketImpact
            return MarketImpact(ACParams.a_share_defaults())
        except Exception:
            return None


def _format_money(value: float) -> str:
    """金额格式化。"""
    return f"{value:,.2f}"


def _format_pct(value: float) -> str:
    """百分比格式化。"""
    return f"{value * 100:.2f}%"


def _ensure_parent(path: str | Path) -> Path:
    """确保导出文件目录存在。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# 枚举与数据模型
# =============================================================================


class OrderStatus(str, Enum):
    """订单状态枚举。"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(str, Enum):
    """订单方向枚举。"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型枚举。"""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """订单有效期枚举。"""

    DAY = "day"
    IOC = "ioc"
    FOK = "fok"
    GTC = "gtc"


class CostMethod(str, Enum):
    """持仓成本核算方式。"""

    WEIGHTED_AVERAGE = "weighted_average"
    FIFO = "fifo"


@dataclass
class Order:
    """订单对象，覆盖订单生命周期的核心字段。"""

    symbol: str
    side: str
    quantity: int
    order_type: str = OrderType.LIMIT.value
    price: float = 0.0
    order_id: str = ""
    status: str = OrderStatus.PENDING.value
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    remaining_quantity: int = 0
    strategy_name: str = ""
    account_id: str = "sim"
    time_in_force: str = TimeInForce.DAY.value
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    parent_order_id: str | None = None
    client_order_id: str | None = None
    broker_order_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    reject_reason: str = ""
    cancel_reason: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        self.side = _normalize_side(self.side)
        self.quantity = int(abs(self.quantity))
        # P2-Q18-fix: 订单数量必须为正。quantity=0 的订单此前会进入 _simulate_fill
        # 生成 0 股 Fill，进而在 record_fill 抛未捕获 ValueError 使 execute_signals 崩溃。
        if self.quantity <= 0:
            raise ValueError(f"订单数量必须为正: {self.quantity}")
        if not self.order_id:
            self.order_id = f"ORD-{_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
        if self.remaining_quantity <= 0:
            self.remaining_quantity = max(self.quantity - self.filled_quantity, 0)
        if self.client_order_id is None:
            self.client_order_id = self.order_id

    def __str__(self) -> str:
        return (
            f"Order({self.order_id}, {self.symbol}, {self.side}, "
            f"qty={self.quantity}, filled={self.filled_quantity}, status={self.status})"
        )

    @property
    def is_active(self) -> bool:
        """是否仍处于可成交/可操作状态。"""
        return self.status in ACTIVE_ORDER_STATUSES

    @property
    def is_terminal(self) -> bool:
        """是否处于终态。"""
        return self.status in TERMINAL_ORDER_STATUSES

    @property
    def signed_quantity(self) -> int:
        """买入为正，卖出为负。"""
        return self.quantity * _side_sign(self.side)

    @property
    def notional(self) -> float:
        """按委托价格计算的名义金额。"""
        return _notional(self.price, self.quantity)

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        data = asdict(self)
        for key in ("created_at", "updated_at", "submitted_at", "completed_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class Fill:
    """成交回报对象。"""

    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    fill_id: str = ""
    timestamp: datetime = field(default_factory=_now)
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    broker_fee: float = 0.0
    liquidity_flag: str = "unknown"
    venue: str = "sim"
    trade_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = _normalize_side(self.side)
        self.quantity = int(abs(self.quantity))
        if not self.fill_id:
            self.fill_id = f"FILL-{self.timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.trade_date:
            self.trade_date = self.timestamp.date().isoformat()

    def __str__(self) -> str:
        return f"Fill({self.fill_id}, {self.symbol}, {self.side}, {self.quantity}@{self.price:.4f})"

    @property
    def signed_quantity(self) -> int:
        return self.quantity * _side_sign(self.side)

    @property
    def gross_amount(self) -> float:
        return self.price * self.quantity

    @property
    def total_fee(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee + self.broker_fee

    @property
    def net_cash_flow(self) -> float:
        """买入为现金流出负值，卖出为现金流入正值。"""
        if self.side == OrderSide.BUY.value:
            return -(self.gross_amount + self.total_fee)
        return self.gross_amount - self.total_fee

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["gross_amount"] = self.gross_amount
        data["total_fee"] = self.total_fee
        data["net_cash_flow"] = self.net_cash_flow
        return data


@dataclass
class PositionLot:
    """FIFO 成本批次；quantity 使用有符号数量。"""

    symbol: str
    quantity: int
    price: float
    opened_at: datetime = field(default_factory=_now)
    lot_id: str = ""

    def __post_init__(self) -> None:
        if not self.lot_id:
            self.lot_id = f"LOT-{self.opened_at.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    def __str__(self) -> str:
        return f"PositionLot({self.symbol}, qty={self.quantity}, cost={self.price:.4f})"

    @property
    def direction(self) -> int:
        return 1 if self.quantity > 0 else -1 if self.quantity < 0 else 0

    @property
    def abs_quantity(self) -> int:
        return abs(self.quantity)


@dataclass
class PositionRecord:
    """单标的持仓账本。"""

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_price: float = 0.0
    realized_pnl: float = 0.0
    lots: list[PositionLot] = field(default_factory=list)
    updated_at: datetime = field(default_factory=_now)
    turnover_buy: float = 0.0
    turnover_sell: float = 0.0
    total_fees: float = 0.0

    def __str__(self) -> str:
        return (
            f"PositionRecord({self.symbol}, qty={self.quantity}, "
            f"avg={self.avg_cost:.4f}, mtm={self.market_price:.4f})"
        )

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def gross_market_value(self) -> float:
        return abs(self.quantity * self.market_price)

    @property
    def cost_value(self) -> float:
        return abs(self.quantity) * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:
            return (self.market_price - self.avg_cost) * self.quantity
        if self.quantity < 0:
            return (self.avg_cost - self.market_price) * abs(self.quantity)
        return 0.0

    @property
    def pnl_pct(self) -> float:
        return _pct(self.unrealized_pnl, self.cost_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "gross_market_value": self.gross_market_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "pnl_pct": self.pnl_pct,
            "turnover_buy": self.turnover_buy,
            "turnover_sell": self.turnover_sell,
            "total_fees": self.total_fees,
            "updated_at": self.updated_at.isoformat(),
            "lots": [asdict(lot) | {"opened_at": lot.opened_at.isoformat()} for lot in self.lots],
        }


@dataclass
class RiskRuleResult:
    """单条风控规则检查结果。"""

    rule_name: str
    passed: bool
    severity: str = "error"
    message: str = ""
    value: float | str | None = None
    limit: float | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"RiskRuleResult({self.rule_name}, {status}, {self.message})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeFee:
    """交易费用分解。"""

    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    broker_fee: float = 0.0

    def __str__(self) -> str:
        return f"TradeFee(total={self.total:.4f})"

    @property
    def total(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee + self.broker_fee

    def to_dict(self) -> dict[str, float]:
        return asdict(self) | {"total": self.total}


@dataclass
class TCARecord:
    """单笔订单 TCA 结果。"""

    order_id: str
    symbol: str
    side: str
    quantity: int
    avg_exec_price: float
    arrival_price: float
    decision_price: float
    vwap_price: float
    arrival_cost_bps: float
    vwap_shortfall_bps: float
    implementation_shortfall_bps: float
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    timing_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    delay_seconds: float = 0.0
    trade_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"TCARecord({self.order_id}, {self.symbol}, "
            f"IS={self.implementation_shortfall_bps:.2f}bps)"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# OrderBook — 订单簿维护
# =============================================================================


class OrderBook:
    """简单内存订单簿，按标的维护活跃买卖委托。"""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._by_symbol: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"buy": [], "sell": []})

    def __repr__(self) -> str:
        return f"OrderBook(active_orders={len(self._orders)})"

    def add(self, order: Order) -> None:
        """把活跃订单加入订单簿。"""
        if not order.is_active:
            return
        self.remove(order.order_id, silent=True)
        self._orders[order.order_id] = order
        self._by_symbol[order.symbol][order.side].append(order.order_id)
        self._sort_symbol_side(order.symbol, order.side)

    def remove(self, order_id: str, silent: bool = False) -> bool:
        """从订单簿移除订单。"""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False if not silent else True
        ids = self._by_symbol.get(order.symbol, {}).get(order.side, [])
        if order_id in ids:
            ids.remove(order_id)
        return True

    def modify(self, order: Order) -> None:
        """改单后重排订单簿。"""
        if order.is_active:
            self.add(order)
        else:
            self.remove(order.order_id, silent=True)

    def get(self, order_id: str) -> Order | None:
        """查询订单簿中的活跃订单。"""
        return self._orders.get(order_id)

    def active_orders(self, symbol: str | None = None, side: str | None = None) -> list[Order]:
        """返回活跃订单列表，可按标的和方向过滤。"""
        if symbol is None:
            orders = list(self._orders.values())
        else:
            if side is None:
                ids = self._by_symbol.get(symbol, {}).get("buy", []) + self._by_symbol.get(symbol, {}).get("sell", [])
            else:
                ids = self._by_symbol.get(symbol, {}).get(_normalize_side(side), [])
            orders = [self._orders[oid] for oid in ids if oid in self._orders]
        return [o for o in orders if o.is_active]

    def top_of_book(self, symbol: str) -> dict[str, Order | None]:
        """返回内部订单簿最优买卖委托。"""
        buys = self.active_orders(symbol, "buy")
        sells = self.active_orders(symbol, "sell")
        return {"bid": buys[0] if buys else None, "ask": sells[0] if sells else None}

    def snapshot(self, symbol: str | None = None, depth: int = 5) -> dict[str, Any]:
        """返回订单簿快照，用于诊断和对账。"""
        symbols = [symbol] if symbol else sorted(self._by_symbol.keys())
        result: dict[str, Any] = {}
        for sym in symbols:
            result[sym] = {
                "buy": [o.to_dict() for o in self.active_orders(sym, "buy")[:depth]],
                "sell": [o.to_dict() for o in self.active_orders(sym, "sell")[:depth]],
            }
        return result

    def clear_terminal(self) -> None:
        """清理误留在订单簿中的终态订单。"""
        for order_id, order in list(self._orders.items()):
            if order.is_terminal:
                self.remove(order_id, silent=True)

    def _sort_symbol_side(self, symbol: str, side: str) -> None:
        """买单高价优先，卖单低价优先；同价按创建时间优先。"""
        normalized = _normalize_side(side)
        reverse = normalized == "buy"
        ids = self._by_symbol[symbol][normalized]
        ids.sort(key=lambda oid: (self._orders[oid].price, -self._orders[oid].created_at.timestamp()), reverse=reverse)


# =============================================================================
# OrderManager — 订单生命周期管理
# =============================================================================


class OrderManager:
    """订单状态机与内存订单仓库。

    D3收敛: 能力未合并——订单生命周期与 execution_broker.Broker 重叠，但本类为
    本地 OrderBook + 状态机白名单（ORDER_SYNC_WHITELIST）模型，与 broker
    Order DTO 字段/语义签名不兼容，保留独立实现，不强迁。
    """

    def __init__(self, account_id: str = "sim", strategy_name: str = "default") -> None:
        self.account_id = account_id
        self.strategy_name = strategy_name
        self.order_book = OrderBook()
        self.orders: dict[str, Order] = {}
        self.fills: dict[str, list[Fill]] = defaultdict(list)
        self.event_log: list[dict[str, Any]] = []
        # P2-Q18-fix: 订单对象级重入锁，防止并发成交回报对同一订单重复记账。
        self._fill_lock = threading.RLock()

    def __repr__(self) -> str:
        return f"OrderManager(account_id={self.account_id!r}, orders={len(self.orders)})"

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float = 0.0,
        order_type: str = OrderType.LIMIT.value,
        **kwargs: Any,
    ) -> Order:
        """创建 pending 订单，不自动提交。"""
        # P2-Q18-fix: 入口校验数量 > 0，避免 0 股订单进入模拟成交后崩溃。
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            raise ValueError(f"订单数量非法: {quantity!r}") from None
        if quantity <= 0:
            raise ValueError(f"订单数量必须为正: {quantity}")
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            account_id=kwargs.pop("account_id", self.account_id),
            strategy_name=kwargs.pop("strategy_name", self.strategy_name),
            time_in_force=kwargs.pop("time_in_force", TimeInForce.DAY.value),
            parent_order_id=kwargs.pop("parent_order_id", None),
            client_order_id=kwargs.pop("client_order_id", None),
            tags=kwargs.pop("tags", {}),
        )
        if kwargs:
            order.tags.update(kwargs)
        self.orders[order.order_id] = order
        self.order_book.add(order)
        self._log_event(order.order_id, "create", {"status": order.status})
        return order

    def submit_order(self, order: Order | str, broker_order_id: str | None = None) -> Order:
        """pending → submitted。"""
        obj = self._resolve_order(order)
        self._transition(obj, OrderStatus.SUBMITTED.value)
        obj.submitted_at = _now()
        obj.broker_order_id = broker_order_id or obj.broker_order_id
        self.order_book.modify(obj)
        self._log_event(obj.order_id, "submit", {"broker_order_id": obj.broker_order_id})
        return obj

    def reject_order(self, order: Order | str, reason: str) -> Order:
        """拒绝订单，通常由风控或柜台返回。"""
        obj = self._resolve_order(order)
        self._transition(obj, OrderStatus.REJECTED.value)
        obj.reject_reason = reason
        obj.completed_at = _now()
        self.order_book.remove(obj.order_id, silent=True)
        self._log_event(obj.order_id, "reject", {"reason": reason})
        return obj

    def cancel_order(self, order: Order | str, reason: str = "") -> Order:
        """撤单，支持 pending/submitted/partial。"""
        obj = self._resolve_order(order)
        self._transition(obj, OrderStatus.CANCELLED.value)
        obj.cancel_reason = reason
        obj.completed_at = _now()
        self.order_book.remove(obj.order_id, silent=True)
        self._log_event(obj.order_id, "cancel", {"reason": reason})
        return obj

    def modify_order(
        self,
        order: Order | str,
        new_quantity: int | None = None,
        new_price: float | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> Order:
        """改单：支持未完成订单调整剩余数量和价格。"""
        obj = self._resolve_order(order)
        if obj.status not in ACTIVE_ORDER_STATUSES:
            raise ValueError(f"终态订单不可改单: {obj.order_id} ({obj.status})")
        if new_quantity is not None:
            if new_quantity < obj.filled_quantity:
                raise ValueError("新数量不能小于已成交数量")
            obj.quantity = int(abs(new_quantity))
            obj.remaining_quantity = max(obj.quantity - obj.filled_quantity, 0)
        if new_price is not None:
            if new_price < 0:
                raise ValueError("价格不能为负")
            obj.price = float(new_price)
        if tags:
            obj.tags.update(dict(tags))
        obj.version += 1
        obj.updated_at = _now()
        self.order_book.modify(obj)
        self._log_event(obj.order_id, "modify", {"quantity": obj.quantity, "price": obj.price, "version": obj.version})
        return obj

    def record_fill(self, order: Order | str, fill: Fill) -> Order:
        """登记成交并驱动 submitted/partial/filled 状态。"""
        with self._fill_lock:
            obj = self._resolve_order(order)
            # P2-Q18-fix: 按 fill_id 幂等去重——同一成交回报重放/部分阶段重入时
            # 直接返回当前状态，避免重复成交导致超量（如同一 fill_id 两次 → 50+50）。
            if any(existing.fill_id == fill.fill_id for existing in self.fills.get(obj.order_id, [])):
                return obj
            # P2-Q18-fix: 成交方向必须与订单方向一致，防止 sell fill 写入 buy 订单
            # 被接受后污染持仓账本。
            if fill.side != obj.side:
                raise ValueError(f"成交方向与订单不一致: order={obj.side}, fill={fill.side}")
            if obj.status == OrderStatus.PENDING.value:
                self.submit_order(obj)
            if obj.status not in {OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value}:
                raise ValueError(f"订单当前状态不能登记成交: {obj.order_id} ({obj.status})")
            if fill.quantity <= 0:
                raise ValueError("成交数量必须为正")
            if fill.quantity > obj.remaining_quantity:
                raise ValueError("成交数量超过订单剩余数量")

            old_filled = obj.filled_quantity
            old_value = obj.avg_fill_price * old_filled
            new_value = fill.price * fill.quantity
            obj.filled_quantity += fill.quantity
            obj.remaining_quantity = max(obj.quantity - obj.filled_quantity, 0)
            obj.avg_fill_price = (old_value + new_value) / obj.filled_quantity if obj.filled_quantity else 0.0
            obj.updated_at = fill.timestamp
            self.fills[obj.order_id].append(fill)

            if obj.remaining_quantity == 0:
                self._transition(obj, OrderStatus.FILLED.value)
                obj.completed_at = fill.timestamp
                self.order_book.remove(obj.order_id, silent=True)
            else:
                self._transition(obj, OrderStatus.PARTIAL.value)
                self.order_book.modify(obj)

            self._log_event(obj.order_id, "fill", fill.to_dict())
            return obj

    def sync_order_status(self, order_id: str, external_status: str, **kwargs: Any) -> Order:
        """按外部柜台状态同步订单。

        P1-Q18-fix: 外部同步不再任意直写订单属性，避免破坏状态机不变量：
        - 只允许同步 ORDER_SYNC_WHITELIST 白名单字段，其余参数落入 tags；
        - filled_quantity 需校验（0 ≤ filled ≤ quantity 且不回退），并由系统重算
          remaining_quantity，禁止外部直写 remaining_quantity；
        - 同步到 filled 必须满足全额成交（filled_quantity == quantity）；
        - 同步到 partial/filled 时必须已有成交登记或显式提供 filled_quantity，
          避免 OrderManager.fills 与订单状态脱节。
        """
        obj = self._resolve_order(order_id)
        target = str(external_status).lower()
        if target not in VALID_ORDER_TRANSITIONS:
            raise ValueError(f"未知外部订单状态: {external_status}")

        explicit_filled = kwargs.get("filled_quantity")
        if explicit_filled is not None:
            try:
                filled = int(explicit_filled)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"filled_quantity 必须为整数: {order_id} ({explicit_filled!r})") from exc
            if filled < 0 or filled > obj.quantity:
                raise ValueError(
                    f"外部同步成交数量越界: {order_id} filled_quantity={filled}, quantity={obj.quantity}"
                )
            if filled < obj.filled_quantity:
                raise ValueError(
                    f"外部同步成交数量不能回退: {order_id} {obj.filled_quantity} -> {filled}"
                )
            obj.filled_quantity = filled
            obj.remaining_quantity = max(obj.quantity - filled, 0)

        if target == OrderStatus.FILLED.value and obj.remaining_quantity != 0:
            raise ValueError(
                f"外部同步到 filled 但订单未全额成交: {order_id} "
                f"filled={obj.filled_quantity}/{obj.quantity}"
            )
        if target == OrderStatus.PARTIAL.value and (obj.filled_quantity <= 0 or obj.remaining_quantity <= 0):
            raise ValueError(
                f"外部同步到 partial 但成交数量矛盾: {order_id} "
                f"filled={obj.filled_quantity}, remaining={obj.remaining_quantity}"
            )
        if target in {OrderStatus.FILLED.value, OrderStatus.PARTIAL.value}:
            registered = sum(f.quantity for f in self.fills.get(obj.order_id, []))
            if registered <= 0 and explicit_filled is None:
                raise ValueError(
                    f"外部同步到 {target} 但既无已登记成交也未显式提供 filled_quantity: {order_id}"
                )

        if target != obj.status:
            self._transition(obj, target)

        for key, value in kwargs.items():
            if key == "filled_quantity":
                continue  # 已在上面受控校验并重算 remaining_quantity
            if key in ORDER_SYNC_WHITELIST:
                if key == "avg_fill_price" and float(value) < 0:
                    raise ValueError(f"成交均价不能为负: {order_id} ({value!r})")
                setattr(obj, key, value)
            elif key == "remaining_quantity":
                raise ValueError(f"remaining_quantity 由系统按 filled_quantity 重算，禁止外部直写: {order_id}")
            else:
                obj.tags[key] = value

        obj.updated_at = _now()
        self.order_book.modify(obj)
        if obj.is_terminal:
            obj.completed_at = obj.completed_at or _now()
            self.order_book.remove(obj.order_id, silent=True)
        self._log_event(obj.order_id, "sync", {"external_status": external_status, **kwargs})
        return obj

    def batch_submit(self, orders: Sequence[Order]) -> list[Order]:
        """批量提交订单。"""
        submitted: list[Order] = []
        for order in orders:
            if order.order_id not in self.orders:
                self.orders[order.order_id] = order
            submitted.append(self.submit_order(order))
        return submitted

    def batch_cancel(self, orders: Sequence[Order | str], reason: str = "batch_cancel") -> list[Order]:
        """批量撤单。"""
        cancelled: list[Order] = []
        for order in orders:
            obj = self._resolve_order(order)
            if obj.is_active:
                cancelled.append(self.cancel_order(obj, reason=reason))
        return cancelled

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def get_orders(self, status: str | None = None, symbol: str | None = None) -> list[Order]:
        orders = list(self.orders.values())
        if status is not None:
            orders = [o for o in orders if o.status == status]
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        if order_id:
            return list(self.fills.get(order_id, []))
        result: list[Fill] = []
        for fills in self.fills.values():
            result.extend(fills)
        return result

    def active_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_active]

    def order_summary(self) -> dict[str, Any]:
        """订单状态汇总。"""
        by_status: dict[str, int] = defaultdict(int)
        by_symbol: dict[str, int] = defaultdict(int)
        for order in self.orders.values():
            by_status[order.status] += 1
            by_symbol[order.symbol] += 1
        return {
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders()),
            "total_fills": sum(len(v) for v in self.fills.values()),
            "by_status": dict(by_status),
            "by_symbol": dict(by_symbol),
            "order_book": self.order_book.snapshot(depth=3),
        }

    def purge_terminal(self, before: datetime | None = None) -> int:
        """清理终态旧订单，默认仅清理订单簿。"""
        self.order_book.clear_terminal()
        if before is None:
            return 0
        removed = 0
        for order_id, order in list(self.orders.items()):
            if order.is_terminal and order.updated_at < before:
                self.orders.pop(order_id, None)
                self.fills.pop(order_id, None)
                removed += 1
        return removed
    def _resolve_order(self, order: Order | str) -> Order:
        if isinstance(order, Order):
            if order.order_id not in self.orders:
                self.orders[order.order_id] = order
            return order
        obj = self.orders.get(order)
        if obj is None:
            raise KeyError(f"订单不存在: {order}")
        return obj

    def _transition(self, order: Order, new_status: str) -> None:
        current = order.status
        if new_status == current:
            return
        allowed = VALID_ORDER_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise ValueError(f"非法订单状态转换: {current} → {new_status}")
        order.status = new_status
        order.updated_at = _now()

    def _log_event(self, order_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        self.event_log.append(
            {
                "timestamp": _now().isoformat(),
                "order_id": order_id,
                "event_type": event_type,
                "payload": dict(payload),
            }
        )


# =============================================================================
# PositionManager — 持仓、成本、P&L
# =============================================================================


class PositionManager:
    """多标的持仓管理器，支持加权平均和 FIFO 成本核算。

    D3收敛登记: 独立能力保留——持仓核算/调仓指令/P&L 报告 execution_broker
    未覆盖。
    """

    def __init__(
        self,
        cost_method: str = CostMethod.WEIGHTED_AVERAGE.value,
        allow_short: bool = False,
        lot_size: int = DEFAULT_LOT_SIZE,
        sector_map: Mapping[str, str] | None = None,
        max_symbol_weight: float = 0.20,
        max_sector_weight: float = 0.40,
        max_abs_position_value: float | None = None,
        max_quantity_per_symbol: int | None = None,
    ) -> None:
        self.cost_method = cost_method
        self.allow_short = allow_short
        self.lot_size = lot_size
        self.sector_map = dict(sector_map or {})
        self.max_symbol_weight = max_symbol_weight
        self.max_sector_weight = max_sector_weight
        self.max_abs_position_value = max_abs_position_value
        self.max_quantity_per_symbol = max_quantity_per_symbol
        self.positions: dict[str, PositionRecord] = {}
        self.realized_pnl_history: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"PositionManager(positions={len(self.positions)}, method={self.cost_method!r})"

    def update_market_prices(self, market_prices: Mapping[str, float]) -> None:
        """批量更新持仓市价。"""
        for symbol, price in market_prices.items():
            self.get_or_create(symbol).market_price = _safe_float(price)
            self.get_or_create(symbol).updated_at = _now()

    def get_or_create(self, symbol: str) -> PositionRecord:
        """获取或创建单标的持仓记录。"""
        if symbol not in self.positions:
            self.positions[symbol] = PositionRecord(symbol=symbol)
        return self.positions[symbol]

    def get_position(self, symbol: str) -> PositionRecord | None:
        return self.positions.get(symbol)

    def apply_fill(self, fill: Fill) -> PositionRecord:
        """把成交写入持仓账本。"""
        position = self.get_or_create(fill.symbol)
        position.total_fees += fill.total_fee
        if fill.side == OrderSide.BUY.value:
            position.turnover_buy += fill.gross_amount
        else:
            position.turnover_sell += fill.gross_amount

        if self.cost_method == CostMethod.FIFO.value:
            realized = self._apply_fill_fifo(position, fill)
        else:
            realized = self._apply_fill_weighted_average(position, fill)

        # P2-Q18-fix: 以「盈亏+费用是否为零」判定，开仓/加仓返回 0.0 不记事件，
        # 平仓（含按成本价卖出的纯费用亏损）返回非零净盈亏记入 realized_pnl_history。
        if realized != 0.0:
            event = {
                "timestamp": fill.timestamp.isoformat(),
                "trade_date": fill.trade_date,
                "symbol": fill.symbol,
                "order_id": fill.order_id,
                "fill_id": fill.fill_id,
                "realized_pnl": realized,
                "fee": fill.total_fee,
            }
            self.realized_pnl_history.append(event)
        position.updated_at = fill.timestamp
        if position.quantity == 0:
            position.avg_cost = 0.0
            position.lots.clear()
        return position

    def _apply_fill_weighted_average(self, position: PositionRecord, fill: Fill) -> float:
        """加权平均成本核算。"""
        old_qty = position.quantity
        trade_qty = fill.signed_quantity
        price = fill.price
        realized = 0.0
        # P2-Q18-fix: 区分「开仓/加仓」与「平仓」。平仓时即使 realized==0（按成本价卖出）
        # 也必须返回 realized - fee（含纯费用亏损），否则 position.realized_pnl 已扣费而
        # realized_pnl_history 不记录，日志与账本不一致。
        closing = old_qty != 0 and old_qty * trade_qty < 0

        if not closing:
            new_qty = old_qty + trade_qty
            old_cost_value = abs(old_qty) * position.avg_cost
            new_cost_value = abs(trade_qty) * price
            position.quantity = new_qty
            position.avg_cost = (old_cost_value + new_cost_value) / abs(new_qty) if new_qty else 0.0
        else:
            close_qty = min(abs(old_qty), abs(trade_qty))
            if old_qty > 0:
                realized = (price - position.avg_cost) * close_qty
            else:
                realized = (position.avg_cost - price) * close_qty
            new_qty = old_qty + trade_qty
            position.quantity = new_qty
            position.realized_pnl += realized - fill.total_fee
            if new_qty == 0:
                position.avg_cost = 0.0
            elif old_qty * new_qty > 0:
                position.avg_cost = position.avg_cost
            else:
                position.avg_cost = price

        if position.quantity:
            position.lots = [PositionLot(symbol=position.symbol, quantity=position.quantity, price=position.avg_cost, opened_at=fill.timestamp)]
        else:
            position.lots.clear()
        return realized - fill.total_fee if closing else 0.0

    def _apply_fill_fifo(self, position: PositionRecord, fill: Fill) -> float:
        """FIFO 成本核算。"""
        trade_qty = fill.signed_quantity
        remaining = trade_qty
        realized = 0.0

        if not position.lots or position.quantity * trade_qty >= 0:
            position.lots.append(PositionLot(fill.symbol, trade_qty, fill.price, fill.timestamp))
            self._rebuild_position_from_lots(position)
            return 0.0

        new_lots: list[PositionLot] = []
        for lot in position.lots:
            if remaining == 0:
                new_lots.append(lot)
                continue
            if lot.quantity * remaining > 0:
                new_lots.append(lot)
                continue

            close_qty = min(abs(lot.quantity), abs(remaining))
            if lot.quantity > 0:
                realized += (fill.price - lot.price) * close_qty
                lot.quantity -= close_qty
                remaining += close_qty
            else:
                realized += (lot.price - fill.price) * close_qty
                lot.quantity += close_qty
                remaining -= close_qty
            if lot.quantity != 0:
                new_lots.append(lot)

        if remaining != 0:
            new_lots.append(PositionLot(fill.symbol, remaining, fill.price, fill.timestamp))
        position.lots = new_lots
        position.realized_pnl += realized - fill.total_fee
        self._rebuild_position_from_lots(position)
        # P2-Q18-fix: 平仓路径始终返回净已实现盈亏（含 realized==0 时的纯费用亏损），
        # 与 _apply_fill_weighted_average 保持一致，保证日志与账本一致。
        return realized - fill.total_fee

    def _rebuild_position_from_lots(self, position: PositionRecord) -> None:
        """根据 FIFO lot 重算数量和平均成本。"""
        position.quantity = sum(lot.quantity for lot in position.lots)
        total_abs_qty = sum(abs(lot.quantity) for lot in position.lots)
        total_cost = sum(abs(lot.quantity) * lot.price for lot in position.lots)
        position.avg_cost = total_cost / total_abs_qty if total_abs_qty else 0.0

    def portfolio_market_value(self) -> float:
        """净市值，多头为正、空头为负。"""
        return sum(p.market_value for p in self.positions.values())

    def portfolio_gross_value(self) -> float:
        """总风险暴露市值，多空取绝对值。"""
        return sum(p.gross_market_value for p in self.positions.values())

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def total_fees(self) -> float:
        return sum(p.total_fees for p in self.positions.values())

    def account_equity(self, cash: float) -> float:
        """权益 = 现金 + 净持仓市值。"""
        return cash + self.portfolio_market_value()

    def exposure_by_symbol(self, equity: float) -> dict[str, float]:
        """按标的计算权重暴露。"""
        return {symbol: _pct(pos.gross_market_value, equity) for symbol, pos in self.positions.items() if pos.quantity}

    def exposure_by_sector(self, equity: float) -> dict[str, float]:
        """按行业计算权重暴露。"""
        sector_value: dict[str, float] = defaultdict(float)
        for symbol, pos in self.positions.items():
            sector = self.sector_map.get(symbol, "UNKNOWN")
            sector_value[sector] += pos.gross_market_value
        return {sector: _pct(value, equity) for sector, value in sector_value.items()}

    def check_position_limits(
        self,
        cash: float,
        proposed_orders: Sequence[Order] | None = None,
        market_prices: Mapping[str, float] | None = None,
    ) -> list[RiskRuleResult]:
        """检查当前和拟交易后的仓位限制。"""
        if market_prices:
            self.update_market_prices(market_prices)
        simulated = self.snapshot_positions()
        for order in proposed_orders or []:
            price = order.price or (market_prices or {}).get(order.symbol, 0.0)
            pos = simulated.setdefault(order.symbol, PositionRecord(order.symbol, market_price=price))
            pos.quantity += order.signed_quantity
            pos.market_price = price
            if pos.quantity and pos.avg_cost == 0:
                pos.avg_cost = price
        equity = cash + sum(p.market_value for p in simulated.values())
        results: list[RiskRuleResult] = []

        for symbol, pos in simulated.items():
            if not self.allow_short and pos.quantity < 0:
                results.append(RiskRuleResult("allow_short", False, "error", f"禁止卖空: {symbol}", pos.quantity, 0))
            if self.max_quantity_per_symbol is not None and abs(pos.quantity) > self.max_quantity_per_symbol:
                results.append(
                    RiskRuleResult(
                        "max_quantity_per_symbol",
                        False,
                        "error",
                        f"{symbol} 数量超过限制",
                        abs(pos.quantity),
                        self.max_quantity_per_symbol,
                    )
                )
            if self.max_abs_position_value is not None and pos.gross_market_value > self.max_abs_position_value:
                results.append(
                    RiskRuleResult(
                        "max_abs_position_value",
                        False,
                        "error",
                        f"{symbol} 市值超过限制",
                        pos.gross_market_value,
                        self.max_abs_position_value,
                    )
                )
            weight = _pct(pos.gross_market_value, equity)
            if weight > self.max_symbol_weight:
                results.append(
                    RiskRuleResult("max_symbol_weight", False, "error", f"{symbol} 集中度过高", weight, self.max_symbol_weight)
                )

        sector_exposure: dict[str, float] = defaultdict(float)
        for symbol, pos in simulated.items():
            sector_exposure[self.sector_map.get(symbol, "UNKNOWN")] += pos.gross_market_value
        for sector, value in sector_exposure.items():
            weight = _pct(value, equity)
            if weight > self.max_sector_weight:
                results.append(RiskRuleResult("max_sector_weight", False, "error", f"行业 {sector} 暴露过高", weight, self.max_sector_weight))
        if not results:
            results.append(RiskRuleResult("position_limits", True, "info", "仓位限制检查通过"))
        return results

    def generate_adjustment_orders(
        self,
        target_weights: Mapping[str, float],
        equity: float,
        market_prices: Mapping[str, float],
        lot_size: int | None = None,
        order_type: str = OrderType.LIMIT.value,
    ) -> list[Order]:
        """根据目标权重生成调仓订单。"""
        lot = lot_size or self.lot_size
        symbols = sorted(set(target_weights) | set(self.positions))
        orders: list[Order] = []
        for symbol in symbols:
            price = _safe_float(market_prices.get(symbol))
            if price <= 0:
                continue
            target_weight = float(target_weights.get(symbol, 0.0))
            target_qty = _round_lot(target_weight * equity / price, lot)
            current_qty = self.positions.get(symbol, PositionRecord(symbol)).quantity
            delta = target_qty - current_qty
            # P2-Q18-fix: 卖出路径保留零股（清仓/剩余整手时可一次性卖出零股），
            # 买入保持整手向下取整。修复持仓 150 股清仓只卖 100 股的问题。
            if delta < 0:
                delta = _round_lot_sell(delta, current_qty, lot)
            else:
                delta = _round_lot(delta, lot)
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            orders.append(Order(symbol=symbol, side=side, quantity=abs(delta), price=price, order_type=order_type))
        return orders

    def generate_target_quantity_orders(
        self,
        target_quantities: Mapping[str, int],
        market_prices: Mapping[str, float],
        lot_size: int | None = None,
        order_type: str = OrderType.LIMIT.value,
    ) -> list[Order]:
        """根据目标股数生成调仓订单。"""
        lot = lot_size or self.lot_size
        orders: list[Order] = []
        for symbol, target in target_quantities.items():
            price = _safe_float(market_prices.get(symbol))
            if price <= 0:
                continue
            current = self.positions.get(symbol, PositionRecord(symbol)).quantity
            # P2-Q18-fix: 卖出路径保留零股一次性卖出，买入保持整手向下取整。
            delta = int(target) - current
            if delta < 0:
                delta = _round_lot_sell(delta, current, lot)
            else:
                delta = _round_lot(delta, lot)
            if delta == 0:
                continue
            orders.append(Order(symbol=symbol, side="buy" if delta > 0 else "sell", quantity=abs(delta), price=price, order_type=order_type))
        return orders

    def pnl_report(self, cash: float = 0.0) -> dict[str, Any]:
        """生成结构化 P&L 报告。"""
        equity = self.account_equity(cash)
        positions = [p.to_dict() for p in sorted(self.positions.values(), key=lambda x: x.symbol) if p.quantity]
        return {
            "cash": cash,
            "equity": equity,
            "net_market_value": self.portfolio_market_value(),
            "gross_market_value": self.portfolio_gross_value(),
            "realized_pnl": self.total_realized_pnl(),
            "unrealized_pnl": self.total_unrealized_pnl(),
            "total_pnl": self.total_realized_pnl() + self.total_unrealized_pnl(),
            "total_fees": self.total_fees(),
            "positions": positions,
            "exposure_by_symbol": self.exposure_by_symbol(equity),
            "exposure_by_sector": self.exposure_by_sector(equity),
        }

    def pnl_report_text(self, cash: float = 0.0) -> str:
        """生成文本 P&L 报告。"""
        report = self.pnl_report(cash)
        lines = [
            "持仓与 P&L 报告",
            f"现金: {_format_money(report['cash'])}",
            f"权益: {_format_money(report['equity'])}",
            f"净市值: {_format_money(report['net_market_value'])}",
            f"总暴露: {_format_money(report['gross_market_value'])}",
            f"已实现盈亏: {_format_money(report['realized_pnl'])}",
            f"浮动盈亏: {_format_money(report['unrealized_pnl'])}",
            f"费用合计: {_format_money(report['total_fees'])}",
            "",
            "明细:",
        ]
        for p in report["positions"]:
            lines.append(
                f"- {p['symbol']}: qty={p['quantity']}, avg={p['avg_cost']:.4f}, "
                f"px={p['market_price']:.4f}, mv={_format_money(p['market_value'])}, "
                f"uPnL={_format_money(p['unrealized_pnl'])} ({_format_pct(p['pnl_pct'])})"
            )
        return "\n".join(lines)

    def snapshot_positions(self) -> dict[str, PositionRecord]:
        """复制当前持仓，用于风险模拟。"""
        copied: dict[str, PositionRecord] = {}
        for symbol, pos in self.positions.items():
            copied[symbol] = PositionRecord(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=pos.market_price,
                realized_pnl=pos.realized_pnl,
                lots=[PositionLot(l.symbol, l.quantity, l.price, l.opened_at, l.lot_id) for l in pos.lots],
                updated_at=pos.updated_at,
                turnover_buy=pos.turnover_buy,
                turnover_sell=pos.turnover_sell,
                total_fees=pos.total_fees,
            )
        return copied


# =============================================================================
# PreTradeRiskCheck — 事前风控
# =============================================================================


class PreTradeRiskCheck:
    """事前风控检查器。

    D3收敛登记: 独立能力保留——资金/保证金/集中度/行业/日内次数/涨跌停/偏离度/
    合规检查为策略风控能力，execution_broker 未覆盖。
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        margin_rate: float = DEFAULT_MARGIN_RATE,
        short_margin_rate: float = DEFAULT_SHORT_MARGIN_RATE,
        max_gross_leverage: float = 1.0,
        max_symbol_weight: float = 0.20,
        max_sector_weight: float = 0.40,
        daily_trade_limit: int = DEFAULT_DAILY_TRADE_LIMIT,
        price_deviation_bps: float = DEFAULT_PRICE_DEVIATION_BPS,
        allow_short: bool = False,
        allow_st: bool = False,
        allow_delisted: bool = False,
        allow_beijing: bool = True,
        sector_map: Mapping[str, str] | None = None,
        security_flags: Mapping[str, Mapping[str, Any]] | None = None,
        limit_prices: Mapping[str, Mapping[str, float] | tuple[float, float]] | None = None,
        compliance_hook: Callable[[Order], RiskRuleResult | list[RiskRuleResult] | None] | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.margin_rate = margin_rate
        self.short_margin_rate = short_margin_rate
        self.max_gross_leverage = max_gross_leverage
        self.max_symbol_weight = max_symbol_weight
        self.max_sector_weight = max_sector_weight
        self.daily_trade_limit = daily_trade_limit
        self.price_deviation_bps = price_deviation_bps
        self.allow_short = allow_short
        self.allow_st = allow_st
        self.allow_delisted = allow_delisted
        self.allow_beijing = allow_beijing
        self.sector_map = dict(sector_map or {})
        self.security_flags = {k: dict(v) for k, v in (security_flags or {}).items()}
        self.limit_prices = dict(limit_prices or {})
        self.compliance_hook = compliance_hook
        self._trade_count_by_date: dict[str, int] = defaultdict(int)

    def __repr__(self) -> str:
        return f"PreTradeRiskCheck(max_gross_leverage={self.max_gross_leverage}, trade_limit={self.daily_trade_limit})"

    def record_trade_count(self, count: int = 1, trade_date: str | None = None) -> None:
        """成交后更新日内交易次数。"""
        self._trade_count_by_date[trade_date or _date_key()] += count

    def reset_daily_count(self, trade_date: str | None = None) -> None:
        """重置指定日期交易次数。"""
        self._trade_count_by_date[trade_date or _date_key()] = 0

    def check_order(
        self,
        order: Order,
        cash: float,
        positions: Mapping[str, PositionRecord],
        market_prices: Mapping[str, float],
        pending_orders: Sequence[Order] | None = None,
    ) -> dict[str, Any]:
        """检查单笔订单，返回是否通过和规则列表。"""
        results: list[RiskRuleResult] = []
        results.extend(self._check_cash_and_margin(order, cash, market_prices))
        results.extend(self._check_position_and_exposure(order, cash, positions, market_prices, pending_orders or []))
        results.extend(self._check_daily_trade_count(order))
        results.extend(self._check_limit_price(order, market_prices))
        results.extend(self._check_price_deviation(order, market_prices))
        results.extend(self._check_compliance(order))

        if self.compliance_hook:
            extra = self.compliance_hook(order)
            if isinstance(extra, RiskRuleResult):
                results.append(extra)
            elif isinstance(extra, list):
                results.extend(extra)

        passed = all(r.passed or r.severity != "error" for r in results)
        return {
            "passed": passed,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "results": [r.to_dict() for r in results],
            "failed_rules": [r.rule_name for r in results if not r.passed and r.severity == "error"],
        }

    def check_batch(
        self,
        orders: Sequence[Order],
        cash: float,
        positions: Mapping[str, PositionRecord],
        market_prices: Mapping[str, float],
    ) -> dict[str, Any]:
        """批量风控，逐笔模拟资金占用。"""
        remaining_cash = cash
        accepted: list[Order] = []
        reports: list[dict[str, Any]] = []
        for order in orders:
            report = self.check_order(order, remaining_cash, positions, market_prices, accepted)
            reports.append(report)
            if report["passed"]:
                accepted.append(order)
                if order.side == "buy":
                    remaining_cash -= self._required_cash(order, market_prices, positions)
        return {
            "passed": all(r["passed"] for r in reports),
            "orders_checked": len(orders),
            "orders_passed": sum(1 for r in reports if r["passed"]),
            "remaining_cash_after_acceptance": remaining_cash,
            "reports": reports,
        }

    def _required_cash(self, order: Order, market_prices: Mapping[str, float], positions: Mapping[str, PositionRecord]) -> float:
        price = order.price or _safe_float(market_prices.get(order.symbol))
        notional = price * order.quantity
        if order.side == "buy":
            return notional * self.margin_rate
        # P2-Q18-fix: 传入真实持仓字典判断卖出是否已有持仓覆盖。
        # 此前恒传空字典 {}，_has_long_inventory 恒 False，导致有持仓覆盖的卖出
        # 仍按 notional * short_margin_rate 错误占用资金。
        return notional * self.short_margin_rate if not self._has_long_inventory(order.symbol, order.quantity, positions) else 0.0

    def _check_cash_and_margin(self, order: Order, cash: float, market_prices: Mapping[str, float]) -> list[RiskRuleResult]:
        price = order.price or _safe_float(market_prices.get(order.symbol))
        if price <= 0:
            return [RiskRuleResult("valid_price", False, "error", "订单价格或行情价格无效", price, ">0")]
        notional = price * order.quantity
        if order.side == "buy":
            required = notional * self.margin_rate
            return [RiskRuleResult("cash_sufficiency", cash >= required, "error", "资金充足性检查", cash, required)]
        if not self.allow_short:
            return [RiskRuleResult("short_margin", True, "info", "卖出不占用新增保证金")]
        required = notional * self.short_margin_rate
        return [RiskRuleResult("short_margin", cash >= required, "error", "卖空保证金检查", cash, required)]

    def _check_position_and_exposure(
        self,
        order: Order,
        cash: float,
        positions: Mapping[str, PositionRecord],
        market_prices: Mapping[str, float],
        pending_orders: Sequence[Order],
    ) -> list[RiskRuleResult]:
        simulated_qty: dict[str, int] = {symbol: pos.quantity for symbol, pos in positions.items()}
        for pending in pending_orders:
            # 防御性去重：若调用方仍把被检查订单自身传入 pending_orders，跳过避免双计
            if pending.order_id == order.order_id:
                continue
            simulated_qty[pending.symbol] = simulated_qty.get(pending.symbol, 0) + pending.signed_quantity
        simulated_qty[order.symbol] = simulated_qty.get(order.symbol, 0) + order.signed_quantity

        if not self.allow_short and simulated_qty.get(order.symbol, 0) < 0:
            return [RiskRuleResult("position_inventory", False, "error", f"{order.symbol} 持仓不足，禁止卖空", simulated_qty[order.symbol], 0)]

        equity = cash
        gross_by_symbol: dict[str, float] = {}
        sector_gross: dict[str, float] = defaultdict(float)
        for symbol, qty in simulated_qty.items():
            price = _safe_float(market_prices.get(symbol)) or positions.get(symbol, PositionRecord(symbol)).market_price
            value = abs(qty * price)
            gross_by_symbol[symbol] = value
            equity += qty * price
            sector_gross[self.sector_map.get(symbol, "UNKNOWN")] += value

        gross_total = sum(gross_by_symbol.values())
        results = [
            RiskRuleResult(
                "gross_leverage",
                _pct(gross_total, equity) <= self.max_gross_leverage,
                "error",
                "总杠杆检查",
                _pct(gross_total, equity),
                self.max_gross_leverage,
            )
        ]
        symbol_weight = _pct(gross_by_symbol.get(order.symbol, 0.0), equity)
        results.append(
            RiskRuleResult("symbol_concentration", symbol_weight <= self.max_symbol_weight, "error", "单票集中度检查", symbol_weight, self.max_symbol_weight)
        )
        for sector, value in sector_gross.items():
            weight = _pct(value, equity)
            results.append(
                RiskRuleResult("sector_concentration", weight <= self.max_sector_weight, "error", f"行业集中度检查: {sector}", weight, self.max_sector_weight)
            )
        return results

    def _check_daily_trade_count(self, order: Order) -> list[RiskRuleResult]:
        today = _date_key(order.created_at)
        count = self._trade_count_by_date.get(today, 0)
        return [RiskRuleResult("daily_trade_limit", count + 1 <= self.daily_trade_limit, "error", "日内交易次数限制", count + 1, self.daily_trade_limit)]

    def _check_limit_price(self, order: Order, market_prices: Mapping[str, float]) -> list[RiskRuleResult]:
        limits = self.limit_prices.get(order.symbol)
        if not limits:
            return [RiskRuleResult("limit_price", True, "info", "未提供涨跌停数据，跳过硬限制")]
        if isinstance(limits, tuple):
            limit_down, limit_up = limits
        else:
            limit_up = _safe_float(limits.get("limit_up"))
            limit_down = _safe_float(limits.get("limit_down"))
        price = order.price or _safe_float(market_prices.get(order.symbol))
        passed = limit_down <= price <= limit_up if limit_up and limit_down else True
        return [RiskRuleResult("limit_price", passed, "error", "涨跌停板检查", price, f"{limit_down}-{limit_up}")]

    def _check_price_deviation(self, order: Order, market_prices: Mapping[str, float]) -> list[RiskRuleResult]:
        ref_price = _safe_float(market_prices.get(order.symbol))
        price = order.price or ref_price
        if ref_price <= 0 or price <= 0:
            return [RiskRuleResult("price_deviation", False, "error", "价格偏离度检查缺少有效行情", price, ref_price)]
        deviation = abs(price / ref_price - 1.0) * 10_000.0
        return [RiskRuleResult("price_deviation", deviation <= self.price_deviation_bps, "error", "价格偏离度检查", deviation, self.price_deviation_bps)]

    def _check_compliance(self, order: Order) -> list[RiskRuleResult]:
        flags = self.security_flags.get(order.symbol, {})
        results: list[RiskRuleResult] = []
        is_st = bool(flags.get("is_st") or flags.get("st")) or "ST" in str(flags.get("name", "")).upper()
        is_delisted = bool(flags.get("is_delisted") or flags.get("delisted"))
        exchange = str(flags.get("exchange", "")).upper()
        is_beijing = exchange in {"BJ", "BSE", "BEIJING"} or order.symbol.endswith(".BJ") or order.symbol[:1] in {"4", "8"}
        results.append(RiskRuleResult("compliance_st", self.allow_st or not is_st, "error", "ST 股票合规检查", str(is_st), str(self.allow_st)))
        results.append(
            RiskRuleResult("compliance_delisted", self.allow_delisted or not is_delisted, "error", "退市股票合规检查", str(is_delisted), str(self.allow_delisted))
        )
        results.append(
            RiskRuleResult("compliance_beijing", self.allow_beijing or not is_beijing, "error", "北交所股票合规检查", str(is_beijing), str(self.allow_beijing))
        )
        return results

    def _has_long_inventory(self, symbol: str, quantity: int, positions: Mapping[str, PositionRecord]) -> bool:
        pos = positions.get(symbol)
        return bool(pos and pos.quantity >= quantity)


# =============================================================================
# TCA — Transaction Cost Analysis
# =============================================================================


class TCA:
    """交易成本分析器。

    D3收敛登记: 独立能力保留——Arrival/VWAP/Implementation Shortfall 与滑点
    统计分析 execution_broker 未覆盖。
    """

    def __init__(self, market_impact_model: Any | None = None) -> None:
        self.records: list[TCARecord] = []
        self.market_impact_model = market_impact_model if market_impact_model is not None else _lazy_market_impact()

    def __repr__(self) -> str:
        return f"TCA(records={len(self.records)})"

    def record_order_execution(
        self,
        order: Order,
        fills: Sequence[Fill],
        benchmarks: Mapping[str, Any] | None = None,
    ) -> TCARecord | None:
        """登记订单执行 TCA。"""
        if not fills:
            return None
        bench = dict(benchmarks or {})
        qty = sum(fill.quantity for fill in fills)
        notional = sum(fill.quantity * fill.price for fill in fills)
        avg_exec = notional / qty if qty else 0.0
        arrival = _safe_float(bench.get("arrival_price"), order.price or avg_exec)
        decision = _safe_float(bench.get("decision_price"), arrival)
        vwap = _safe_float(bench.get("vwap_price"), arrival)
        side_multiplier = 1.0 if order.side == "buy" else -1.0
        arrival_cost = side_multiplier * _bps(avg_exec - arrival, arrival)
        vwap_shortfall = side_multiplier * _bps(avg_exec - vwap, vwap)
        implementation_shortfall = side_multiplier * _bps(avg_exec - decision, decision)
        slippage = side_multiplier * _bps(avg_exec - (order.price or arrival), order.price or arrival)
        first_ts = min(fill.timestamp for fill in fills)
        decision_time = bench.get("decision_time") or order.created_at
        if isinstance(decision_time, str):
            try:
                decision_time = datetime.fromisoformat(decision_time)
            except ValueError:
                decision_time = order.created_at
        delay_seconds = max((first_ts - decision_time).total_seconds(), 0.0) if isinstance(decision_time, datetime) else 0.0
        impact = self.compute_market_impact(order.symbol, order.side, qty, arrival, bench)
        record = TCARecord(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            avg_exec_price=avg_exec,
            arrival_price=arrival,
            decision_price=decision,
            vwap_price=vwap,
            arrival_cost_bps=arrival_cost,
            vwap_shortfall_bps=vwap_shortfall,
            implementation_shortfall_bps=implementation_shortfall,
            market_impact_bps=impact.get("total_cost_bps", 0.0),
            spread_cost_bps=impact.get("spread_cost_bps", 0.0),
            timing_cost_bps=impact.get("timing_cost_bps", 0.0),
            slippage_bps=slippage,
            delay_seconds=delay_seconds,
            trade_date=fills[-1].trade_date,
            metadata={"fill_count": len(fills), "benchmark": bench, "impact": impact},
        )
        self.records.append(record)
        return record

    def compute_market_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        arrival_price: float,
        benchmarks: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """对接 market_impact.py 计算市场冲击。

        D3收敛: 能力未合并——市场冲击/滑点与 execution_broker.SimBroker 的滑点
        模拟重叠，但本方法为 TCA 报告接口（返回 bps 分解 dict），与 broker
        下单内嵌滑点签名不兼容，保留独立实现，不强迁。
        """
        if self.market_impact_model is None or quantity <= 0 or arrival_price <= 0:
            return {"total_cost_bps": 0.0, "spread_cost_bps": 0.0, "timing_cost_bps": 0.0}
        bench = dict(benchmarks or {})
        try:
            result = self.market_impact_model.total_cost(
                shares=float(quantity),
                arrival_price=float(arrival_price),
                daily_vol=bench.get("daily_vol"),
                horizon_days=float(bench.get("horizon_days", 1.0)),
                total_shares_outstanding=bench.get("total_shares_outstanding"),
                side=_normalize_side(side),
            )
            return {
                "total_cost_bps": _safe_float(getattr(result, "total_cost_bps", 0.0)),
                "total_cost_value": _safe_float(getattr(result, "total_cost_value", 0.0)),
                "permanent_impact_bps": _safe_float(getattr(result, "permanent_impact", 0.0)),
                "temporary_impact_bps": _safe_float(getattr(result, "temporary_impact", 0.0)),
                "spread_cost_bps": _safe_float(getattr(result, "spread_cost", 0.0)),
                "timing_cost_bps": _safe_float(getattr(result, "timing_risk", 0.0)),
            }
        except Exception as exc:
            return {"total_cost_bps": 0.0, "spread_cost_bps": 0.0, "timing_cost_bps": 0.0, "error": str(exc)}

    def as_dataframe(self) -> Any:
        """返回 pandas DataFrame；无 pandas 时返回 list[dict]。"""
        rows = [record.to_dict() for record in self.records]
        if pd is None:
            return rows
        return pd.DataFrame(rows)

    def slippage_statistics(self, symbol: str | None = None) -> dict[str, Any]:
        """滑点统计。"""
        records = [r for r in self.records if symbol is None or r.symbol == symbol]
        values = [r.slippage_bps for r in records]
        if not values:
            return {"count": 0, "mean_bps": 0.0, "median_bps": 0.0, "stdev_bps": 0.0, "p95_bps": 0.0}
        sorted_values = sorted(values)
        p95_idx = min(int(0.95 * (len(sorted_values) - 1)), len(sorted_values) - 1)
        return {
            "count": len(values),
            "mean_bps": statistics.mean(values),
            "median_bps": statistics.median(values),
            "stdev_bps": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min_bps": min(values),
            "max_bps": max(values),
            "p95_bps": sorted_values[p95_idx],
        }

    def time_slippage_analysis(self) -> dict[str, Any]:
        """分析等待时间和滑点的关系。"""
        if not self.records:
            return {"count": 0, "correlation": 0.0, "buckets": {}}
        pairs = [(r.delay_seconds, r.slippage_bps) for r in self.records]
        np = _lazy_numpy()
        corr = 0.0
        if np is not None and len(pairs) > 1:
            try:
                corr = float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1])
                if math.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
        buckets: dict[str, list[float]] = {"0-5s": [], "5-30s": [], "30-120s": [], ">120s": []}
        for delay, slip in pairs:
            if delay <= 5:
                buckets["0-5s"].append(slip)
            elif delay <= 30:
                buckets["5-30s"].append(slip)
            elif delay <= 120:
                buckets["30-120s"].append(slip)
            else:
                buckets[">120s"].append(slip)
        bucket_summary = {k: {"count": len(v), "mean_bps": statistics.mean(v) if v else 0.0} for k, v in buckets.items()}
        return {"count": len(pairs), "correlation": corr, "buckets": bucket_summary}

    def fit_slippage_model(self) -> dict[str, Any]:
        """用 sklearn 做一个可选滑点回归，用于诊断成交质量。"""
        LinearRegression = _lazy_sklearn_linear_regression()
        np = _lazy_numpy()
        if LinearRegression is None or np is None or len(self.records) < 3:
            return {"available": False, "message": "sklearn/numpy 不可用或样本不足"}
        try:
            x = np.array([[r.quantity, r.delay_seconds, abs(r.market_impact_bps)] for r in self.records], dtype=float)
            y = np.array([r.slippage_bps for r in self.records], dtype=float)
            model = LinearRegression().fit(x, y)
            return {
                "available": True,
                "intercept": float(model.intercept_),
                "coef_quantity": float(model.coef_[0]),
                "coef_delay_seconds": float(model.coef_[1]),
                "coef_market_impact": float(model.coef_[2]),
                "r2": float(model.score(x, y)),
            }
        except Exception as exc:
            return {"available": False, "message": str(exc)}

    def daily_report(self, trade_date: str | None = None) -> dict[str, Any]:
        """每日 TCA 报告。"""
        target = trade_date or _date_key()
        records = [r for r in self.records if r.trade_date == target]
        return self._aggregate_records(records, group_name=target)

    def symbol_report(self, symbol: str) -> dict[str, Any]:
        """按标的 TCA 报告。"""
        records = [r for r in self.records if r.symbol == symbol]
        return self._aggregate_records(records, group_name=symbol)

    def report_by_symbol(self) -> dict[str, dict[str, Any]]:
        """所有标的 TCA 汇总。"""
        symbols = sorted({r.symbol for r in self.records})
        return {symbol: self.symbol_report(symbol) for symbol in symbols}

    def generate_report(self) -> str:
        """生成文本 TCA 报告。"""
        total = self._aggregate_records(self.records, "ALL")
        slip = self.slippage_statistics()
        time_report = self.time_slippage_analysis()
        lines = [
            "TCA 交易成本分析",
            f"订单数: {total['count']}",
            f"总数量: {total['quantity']}",
            f"平均 Arrival Cost: {total['avg_arrival_cost_bps']:.2f} bps",
            f"平均 VWAP Shortfall: {total['avg_vwap_shortfall_bps']:.2f} bps",
            f"平均 Implementation Shortfall: {total['avg_implementation_shortfall_bps']:.2f} bps",
            f"平均市场冲击: {total['avg_market_impact_bps']:.2f} bps",
            f"滑点均值/中位数: {slip['mean_bps']:.2f}/{slip['median_bps']:.2f} bps",
            f"等待时间-滑点相关性: {time_report['correlation']:.4f}",
        ]
        return "\n".join(lines)

    def _aggregate_records(self, records: Sequence[TCARecord], group_name: str) -> dict[str, Any]:
        if not records:
            return {
                "group": group_name,
                "count": 0,
                "quantity": 0,
                "avg_arrival_cost_bps": 0.0,
                "avg_vwap_shortfall_bps": 0.0,
                "avg_implementation_shortfall_bps": 0.0,
                "avg_market_impact_bps": 0.0,
                "avg_slippage_bps": 0.0,
            }
        qty = sum(r.quantity for r in records)

        def wavg(values: Iterable[float]) -> float:
            pairs = list(zip(values, [r.quantity for r in records], strict=False))
            den = sum(w for _, w in pairs)
            return sum(v * w for v, w in pairs) / den if den else 0.0

        return {
            "group": group_name,
            "count": len(records),
            "quantity": qty,
            "avg_arrival_cost_bps": wavg([r.arrival_cost_bps for r in records]),
            "avg_vwap_shortfall_bps": wavg([r.vwap_shortfall_bps for r in records]),
            "avg_implementation_shortfall_bps": wavg([r.implementation_shortfall_bps for r in records]),
            "avg_market_impact_bps": wavg([r.market_impact_bps for r in records]),
            "avg_slippage_bps": wavg([r.slippage_bps for r in records]),
            "total_delay_seconds": sum(r.delay_seconds for r in records),
        }


# =============================================================================
# TradeReporter — 成交记录、费用和导出
# =============================================================================


class TradeReporter:
    """交易流水与费用报表。

    D3收敛登记: 独立能力保留——成交流水/汇总/CSV/Excel 报表 execution_broker
    未覆盖；其中费用公式已薄壳转发 execution_broker.estimate_trade_cost。
    """

    def __init__(
        self,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        min_commission: float = DEFAULT_MIN_COMMISSION,
        stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
        transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE,
    ) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.fills: list[Fill] = []
        self.order_records: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"TradeReporter(fills={len(self.fills)})"

    def calculate_fee(self, side: str, quantity: int, price: float) -> TradeFee:
        """计算佣金、印花税和过户费。

        D3收敛 (2026-08-11): 费用公式薄壳转发 execution_broker.estimate_trade_cost
        （执行域唯一真源）；零额保护（gross<=0 返回全零）与 TradeFee 返回结构
        保留本地语义，行为不变。
        """
        gross = abs(quantity * price)
        if gross <= 0:
            return TradeFee(commission=0.0, stamp_tax=0.0, transfer_fee=0.0, broker_fee=0.0)
        fee = estimate_trade_cost(
            side=_normalize_side(side),
            quantity=abs(quantity),
            price=abs(price),
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
            stamp_tax_rate=self.stamp_tax_rate,
            transfer_fee_rate=self.transfer_fee_rate,
        )
        return TradeFee(
            commission=fee["commission"],
            stamp_tax=fee["stamp_tax"],
            transfer_fee=fee["transfer_fee"],
            broker_fee=0.0,
        )

    def enrich_fill_fee(self, fill: Fill) -> Fill:
        """给成交补全费用。"""
        fee = self.calculate_fee(fill.side, fill.quantity, fill.price)
        fill.commission = fill.commission or fee.commission
        fill.stamp_tax = fill.stamp_tax or fee.stamp_tax
        fill.transfer_fee = fill.transfer_fee or fee.transfer_fee
        fill.broker_fee = fill.broker_fee or fee.broker_fee
        return fill

    def record_fill(self, fill: Fill) -> Fill:
        """登记成交。"""
        enriched = self.enrich_fill_fee(fill)
        self.fills.append(enriched)
        return enriched

    def record_order(self, order: Order) -> None:
        """登记订单快照。"""
        self.order_records.append(order.to_dict())

    def fills_as_rows(self) -> list[dict[str, Any]]:
        return [fill.to_dict() for fill in self.fills]

    def as_dataframe(self) -> Any:
        rows = self.fills_as_rows()
        if pd is None:
            return rows
        return pd.DataFrame(rows)

    def fee_summary(self, trade_date: str | None = None) -> dict[str, float]:
        fills = [f for f in self.fills if trade_date is None or f.trade_date == trade_date]
        return {
            "gross_amount": sum(f.gross_amount for f in fills),
            "commission": sum(f.commission for f in fills),
            "stamp_tax": sum(f.stamp_tax for f in fills),
            "transfer_fee": sum(f.transfer_fee for f in fills),
            "broker_fee": sum(f.broker_fee for f in fills),
            "total_fee": sum(f.total_fee for f in fills),
        }

    def daily_summary(self, trade_date: str | None = None) -> dict[str, Any]:
        target = trade_date or _date_key()
        fills = [f for f in self.fills if f.trade_date == target]
        by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {"buy_qty": 0, "sell_qty": 0, "buy_amount": 0.0, "sell_amount": 0.0, "fee": 0.0})
        for fill in fills:
            bucket = by_symbol[fill.symbol]
            if fill.side == "buy":
                bucket["buy_qty"] += fill.quantity
                bucket["buy_amount"] += fill.gross_amount
            else:
                bucket["sell_qty"] += fill.quantity
                bucket["sell_amount"] += fill.gross_amount
            bucket["fee"] += fill.total_fee
        summary = self.fee_summary(target)
        return {
            "trade_date": target,
            "fill_count": len(fills),
            "symbols": dict(by_symbol),
            **summary,
        }

    def symbol_summary(self, symbol: str) -> dict[str, Any]:
        fills = [f for f in self.fills if f.symbol == symbol]
        buy_qty = sum(f.quantity for f in fills if f.side == "buy")
        sell_qty = sum(f.quantity for f in fills if f.side == "sell")
        buy_amount = sum(f.gross_amount for f in fills if f.side == "buy")
        sell_amount = sum(f.gross_amount for f in fills if f.side == "sell")
        return {
            "symbol": symbol,
            "fill_count": len(fills),
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
            "avg_buy_price": buy_amount / buy_qty if buy_qty else 0.0,
            "avg_sell_price": sell_amount / sell_qty if sell_qty else 0.0,
            "total_fee": sum(f.total_fee for f in fills),
        }

    def export_csv(self, path: str | Path) -> str:
        """导出成交 CSV。"""
        p = _ensure_parent(path)
        rows = self.fills_as_rows()
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["fill_id", "order_id", "symbol", "side", "quantity", "price"]
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(p)

    def export_excel(self, path: str | Path) -> str:
        """导出成交 Excel；缺少 pandas/openpyxl 时抛出清晰错误。"""
        if pd is None:
            raise RuntimeError("pandas 不可用，无法导出 Excel")
        p = _ensure_parent(path)
        with pd.ExcelWriter(p) as writer:  # type: ignore[attr-defined]
            pd.DataFrame(self.fills_as_rows()).to_excel(writer, sheet_name="fills", index=False)
            pd.DataFrame([self.daily_summary(d) for d in sorted({f.trade_date for f in self.fills})]).to_excel(writer, sheet_name="daily", index=False)
            pd.DataFrame(self.order_records).to_excel(writer, sheet_name="orders", index=False)
        return str(p)

    def generate_report(self, trade_date: str | None = None) -> str:
        """生成每日交易汇总文本。"""
        summary = self.daily_summary(trade_date)
        lines = [
            f"每日交易汇总: {summary['trade_date']}",
            f"成交笔数: {summary['fill_count']}",
            f"成交金额: {_format_money(summary['gross_amount'])}",
            f"佣金: {_format_money(summary['commission'])}",
            f"印花税: {_format_money(summary['stamp_tax'])}",
            f"过户费: {_format_money(summary['transfer_fee'])}",
            f"总费用: {_format_money(summary['total_fee'])}",
            "标的汇总:",
        ]
        for symbol, data in sorted(summary["symbols"].items()):
            lines.append(
                f"- {symbol}: 买 {data['buy_qty']} 股 / 卖 {data['sell_qty']} 股, "
                f"买入额 {_format_money(data['buy_amount'])}, 卖出额 {_format_money(data['sell_amount'])}, 费用 {_format_money(data['fee'])}"
            )
        return "\n".join(lines)


# =============================================================================
# TradingSystem — 顶层整合
# =============================================================================


class TradingSystem:
    """完整交易执行系统顶层门面。

    D3收敛登记: 独立能力保留——策略信号→订单（execute_signals）、风控
    （PreTradeRiskCheck）、持仓状态机（PositionManager/OrderManager）、对账与
    报告为策略层能力，execution_broker 未覆盖；sim 撮合（_simulate_fill）与
    SimBroker 重叠但签名不兼容，保留独立实现，不强迁。
    """

    def __init__(self, strategy_name: str, broker_type: str = "sim", **kwargs: Any) -> None:
        self.strategy_name = strategy_name
        self.broker_type = broker_type
        self.account_id = kwargs.get("account_id", f"{broker_type}-{strategy_name}")
        self.initial_cash = float(kwargs.get("initial_cash", 1_000_000.0))
        self.cash = float(kwargs.get("cash", self.initial_cash))
        self.lot_size = int(kwargs.get("lot_size", DEFAULT_LOT_SIZE))
        self.signal_mode = kwargs.get("signal_mode", "weight")
        self.default_order_type = kwargs.get("default_order_type", OrderType.LIMIT.value)
        self.slippage_bps = float(kwargs.get("slippage_bps", 5.0))
        self.t_plus_1 = bool(kwargs.get("t_plus_1", True))  # P1-Q18-fix: A股 T+1 规则开关，默认开启
        # P1-Q18-fix: 按 (trade_date, symbol) 跟踪当日买入股数，用于卖出可用数量校验。
        self._intraday_buys: dict[tuple[str, str], int] = defaultdict(int)
        self.market_prices: dict[str, float] = {}
        self.order_manager = OrderManager(account_id=self.account_id, strategy_name=strategy_name)
        self.position_manager = PositionManager(
            cost_method=kwargs.get("cost_method", CostMethod.WEIGHTED_AVERAGE.value),
            allow_short=bool(kwargs.get("allow_short", False)),
            lot_size=self.lot_size,
            sector_map=kwargs.get("sector_map"),
            max_symbol_weight=float(kwargs.get("max_symbol_weight", 0.20)),
            max_sector_weight=float(kwargs.get("max_sector_weight", 0.40)),
            max_abs_position_value=kwargs.get("max_abs_position_value"),
            max_quantity_per_symbol=kwargs.get("max_quantity_per_symbol"),
        )
        self.risk_checker = PreTradeRiskCheck(
            initial_cash=self.initial_cash,
            margin_rate=float(kwargs.get("margin_rate", DEFAULT_MARGIN_RATE)),
            short_margin_rate=float(kwargs.get("short_margin_rate", DEFAULT_SHORT_MARGIN_RATE)),
            max_gross_leverage=float(kwargs.get("max_gross_leverage", 1.0)),
            max_symbol_weight=float(kwargs.get("max_symbol_weight", 0.20)),
            max_sector_weight=float(kwargs.get("max_sector_weight", 0.40)),
            daily_trade_limit=int(kwargs.get("daily_trade_limit", DEFAULT_DAILY_TRADE_LIMIT)),
            price_deviation_bps=float(kwargs.get("price_deviation_bps", DEFAULT_PRICE_DEVIATION_BPS)),
            allow_short=bool(kwargs.get("allow_short", False)),
            allow_st=bool(kwargs.get("allow_st", False)),
            allow_delisted=bool(kwargs.get("allow_delisted", False)),
            allow_beijing=bool(kwargs.get("allow_beijing", True)),
            sector_map=kwargs.get("sector_map"),
            security_flags=kwargs.get("security_flags"),
            limit_prices=kwargs.get("limit_prices"),
            compliance_hook=kwargs.get("compliance_hook"),
        )
        self.reporter = TradeReporter(
            commission_rate=float(kwargs.get("commission_rate", DEFAULT_COMMISSION_RATE)),
            min_commission=float(kwargs.get("min_commission", DEFAULT_MIN_COMMISSION)),
            stamp_tax_rate=float(kwargs.get("stamp_tax_rate", DEFAULT_STAMP_TAX_RATE)),
            transfer_fee_rate=float(kwargs.get("transfer_fee_rate", DEFAULT_TRANSFER_FEE_RATE)),
        )
        self.tca = TCA(kwargs.get("market_impact_model"))
        self.reconciliation_log: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"TradingSystem(strategy={self.strategy_name!r}, broker_type={self.broker_type!r}, cash={self.cash:.2f})"

    def execute_signals(self, signals: "pd.Series", market_prices: dict[str, float]) -> list[Order]:
        """把策略信号转成订单并执行。

        signals 默认解释为目标权重：
        - 0.10 表示目标持仓为账户权益的 10%
        - -0.05 表示目标空头 5%，若 allow_short=False 会被风控拒绝
        若初始化传入 signal_mode='shares'，则把 signals 值解释为目标股数。
        """
        self.market_prices.update({k: _safe_float(v) for k, v in market_prices.items()})
        self.position_manager.update_market_prices(self.market_prices)
        signal_map = self._signals_to_dict(signals)
        equity = self.position_manager.account_equity(self.cash)
        if self.signal_mode == "shares":
            candidate_orders = self.position_manager.generate_target_quantity_orders(
                {symbol: _safe_int(value) for symbol, value in signal_map.items()},
                self.market_prices,
                lot_size=self.lot_size,
                order_type=self.default_order_type,
            )
        else:
            candidate_orders = self.position_manager.generate_adjustment_orders(
                {symbol: float(value) for symbol, value in signal_map.items()},
                equity,
                self.market_prices,
                lot_size=self.lot_size,
                order_type=self.default_order_type,
            )

        executed: list[Order] = []
        for order in candidate_orders:
            # 风控先行：构造未登记入订单簿的候选订单进行事前风控，
            # 通过后才 create_order，避免订单以 pending/active 身份进入订单簿后
            # 在 _check_position_and_exposure 中被自身双计（Q18 修复）。
            candidate = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
                strategy_name=self.strategy_name,
                account_id=self.account_id,
                tags={"source": "execute_signals", "signal": signal_map.get(order.symbol)},
            )
            risk = self.risk_checker.check_order(
                candidate,
                self.cash,
                self.position_manager.positions,
                self.market_prices,
                self.order_manager.active_orders(),
            )
            if not risk["passed"]:
                reason = "; ".join(risk["failed_rules"]) or "pre_trade_risk_failed"
                import logging
                logging.getLogger(__name__).warning(
                    f"[TradingSystem] 风控拒绝 {order.symbol} {order.side} {order.quantity}: {reason}"
                )
                continue

            managed = self.order_manager.create_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
                tags={"source": "execute_signals", "signal": signal_map.get(order.symbol), "risk_report": risk},
            )
            self.order_manager.submit_order(managed)
            if self.broker_type == "sim":
                self._simulate_fill(managed)
            executed.append(managed)
        return executed

    def check_risk(self) -> dict[str, Any]:
        """检查账户当前风险状态。"""
        self.position_manager.update_market_prices(self.market_prices)
        pnl = self.position_manager.pnl_report(self.cash)
        equity = pnl["equity"]
        gross = pnl["gross_market_value"]
        leverage = _pct(gross, equity)
        concentration = pnl["exposure_by_symbol"]
        sector = pnl["exposure_by_sector"]
        violations: list[dict[str, Any]] = []
        if leverage > self.risk_checker.max_gross_leverage:
            violations.append({"rule": "gross_leverage", "value": leverage, "limit": self.risk_checker.max_gross_leverage})
        for symbol, weight in concentration.items():
            if weight > self.risk_checker.max_symbol_weight:
                violations.append({"rule": "symbol_concentration", "symbol": symbol, "value": weight, "limit": self.risk_checker.max_symbol_weight})
        for sector_name, weight in sector.items():
            if weight > self.risk_checker.max_sector_weight:
                violations.append({"rule": "sector_concentration", "sector": sector_name, "value": weight, "limit": self.risk_checker.max_sector_weight})
        return {
            "passed": len(violations) == 0,
            "cash": self.cash,
            "equity": equity,
            "gross_market_value": gross,
            "gross_leverage": leverage,
            "positions": pnl["positions"],
            "violations": violations,
            "active_orders": [o.to_dict() for o in self.order_manager.active_orders()],
        }

    def generate_report(self) -> str:
        """生成综合交易系统报告。"""
        risk = self.check_risk()
        order_summary = self.order_manager.order_summary()
        sections = [
            f"交易系统报告 — {self.strategy_name}",
            f"V4.1 feature: trading_system",
            f"Broker: {self.broker_type}",
            f"现金: {_format_money(self.cash)}",
            f"权益: {_format_money(risk['equity'])}",
            f"总杠杆: {_format_pct(risk['gross_leverage'])}",
            f"订单: total={order_summary['total_orders']}, active={order_summary['active_orders']}, fills={order_summary['total_fills']}",
            "",
            self.position_manager.pnl_report_text(self.cash),
            "",
            self.reporter.generate_report(),
            "",
            self.tca.generate_report(),
        ]
        if risk["violations"]:
            sections.append("\n风险告警:")
            for item in risk["violations"]:
                sections.append(f"- {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(sections)

    def run_reconciliation(self) -> dict[str, Any]:
        """运行对账：订单成交、报表成交、现金和持仓一致性。"""
        order_fills = self.order_manager.get_fills()
        reporter_fills = self.reporter.fills
        order_fill_ids = {f.fill_id for f in order_fills}
        reporter_fill_ids = {f.fill_id for f in reporter_fills}
        missing_in_reporter = sorted(order_fill_ids - reporter_fill_ids)
        extra_in_reporter = sorted(reporter_fill_ids - order_fill_ids)
        cash_recalc = self.initial_cash + sum(f.net_cash_flow for f in reporter_fills)
        cash_diff = self.cash - cash_recalc
        position_qty_from_fills: dict[str, int] = defaultdict(int)
        for fill in reporter_fills:
            position_qty_from_fills[fill.symbol] += fill.signed_quantity
        position_diffs: dict[str, dict[str, int]] = {}
        symbols = sorted(set(position_qty_from_fills) | set(self.position_manager.positions))
        for symbol in symbols:
            from_fills = position_qty_from_fills.get(symbol, 0)
            from_book = self.position_manager.positions.get(symbol, PositionRecord(symbol)).quantity
            if from_fills != from_book:
                position_diffs[symbol] = {"fills_quantity": from_fills, "position_quantity": from_book}
        result = {
            "passed": not missing_in_reporter and not extra_in_reporter and abs(cash_diff) < 1e-6 and not position_diffs,
            "missing_in_reporter": missing_in_reporter,
            "extra_in_reporter": extra_in_reporter,
            "cash": self.cash,
            "cash_recalculated": cash_recalc,
            "cash_diff": cash_diff,
            "position_diffs": position_diffs,
            "checked_at": _now().isoformat(),
        }
        self.reconciliation_log.append(result)
        return result

    def export_reports(self, directory: str | Path) -> dict[str, str]:
        """导出 CSV/Excel 和文本报告。"""
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        stamp = _now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reporter.export_csv(base / f"fills_{stamp}.csv")
        txt_path = base / f"trading_report_{stamp}.txt"
        txt_path.write_text(self.generate_report(), encoding="utf-8")
        result = {"fills_csv": csv_path, "report_txt": str(txt_path)}
        if pd is not None:
            try:
                result["fills_excel"] = self.reporter.export_excel(base / f"fills_{stamp}.xlsx")
            except Exception as exc:
                result["fills_excel_error"] = str(exc)
        return result

    def _signals_to_dict(self, signals: Any) -> dict[str, float]:
        """把 pandas Series / dict / list 转成 symbol -> signal。"""
        if pd is not None and isinstance(signals, pd.Series):  # type: ignore[arg-type]
            return {str(k): _safe_float(v) for k, v in signals.dropna().items()}
        if isinstance(signals, Mapping):
            return {str(k): _safe_float(v) for k, v in signals.items() if v is not None}
        if isinstance(signals, Sequence) and not isinstance(signals, (str, bytes)):
            result: dict[str, float] = {}
            for item in signals:
                if isinstance(item, Mapping) and "symbol" in item:
                    result[str(item["symbol"])] = _safe_float(item.get("signal", item.get("weight", item.get("quantity", 0.0))))
            return result
        raise TypeError("signals 必须是 pandas.Series、dict 或包含 symbol/signal 的列表")

    def _simulate_fill(self, order: Order) -> Fill | None:
        """sim 模式撮合：默认全额成交，并应用固定滑点。

        D3收敛: 能力未合并——撮合/滑点/费用结算与 execution_broker.SimBroker
        ._do_place_order 重叠，但本方法以本地 Order/Fill/门面状态（cash/T+1/风控）
        运行，与 broker place_order 签名不兼容，保留独立实现，不强迁。
        """
        if not order.is_active:
            return None
        # P2-Q18-fix: 剩余数量 <= 0 时直接拒单返回，避免生成 0 股 Fill 触发
        # record_fill「成交数量必须为正」未捕获异常导致 execute_signals 崩溃。
        if order.remaining_quantity <= 0:
            self.order_manager.reject_order(order, "模拟成交剩余数量非正")
            return None
        # P1-Q18-fix: 限价单滑点必须以市价为基准，不能对限价本身施加滑点；
        # 滑点后成交价再与限价取 min/max（买：min(限价, 市价+滑点)；卖：max(限价, 市价-滑点)）。
        market_price = self.market_prices.get(order.symbol, 0.0)
        base_price = market_price if market_price > 0 else (order.price if order.price > 0 else 0.0)
        if base_price <= 0:
            self.order_manager.reject_order(order, "模拟成交缺少有效价格")
            return None
        side_multiplier = 1.0 if order.side == "buy" else -1.0
        exec_price = base_price * (1.0 + side_multiplier * self.slippage_bps / 10_000.0)
        if order.order_type == OrderType.LIMIT.value and order.price > 0:
            exec_price = min(exec_price, order.price) if order.side == "buy" else max(exec_price, order.price)
        fee = self.reporter.calculate_fee(order.side, order.remaining_quantity, exec_price)
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=exec_price,
            commission=fee.commission,
            stamp_tax=fee.stamp_tax,
            transfer_fee=fee.transfer_fee,
            broker_fee=fee.broker_fee,
            venue="sim",
            metadata={"base_price": base_price, "slippage_bps": self.slippage_bps},
        )
        if order.side == "buy" and -fill.net_cash_flow > self.cash + 1e-8:
            self.order_manager.reject_order(order, "模拟成交资金不足")
            return None
        if order.side == "sell":
            pos = self.position_manager.positions.get(order.symbol, PositionRecord(order.symbol))
            # P1-Q18-fix: A股 T+1 校验——当日买入的股份次日方可卖出。
            # 可用数量 = 持仓 - 当日买入；仅对多头持仓收窄卖出额度。
            if self.t_plus_1 and pos.quantity > 0:
                day_key = (fill.trade_date, order.symbol)
                today_buys = self._intraday_buys.get(day_key, 0)
                available = pos.quantity - today_buys
                if available < fill.quantity:
                    self.order_manager.reject_order(
                        order,
                        f"T+1 当日买入不可卖出: {order.symbol} 可用 {available} < 卖出 {fill.quantity}",
                    )
                    return None
            if not self.position_manager.allow_short:
                current_qty = pos.quantity
                if current_qty < fill.quantity:
                    self.order_manager.reject_order(order, "模拟成交持仓不足")
                    return None
        self.cash += fill.net_cash_flow
        self.order_manager.record_fill(order, fill)
        self.position_manager.apply_fill(fill)
        if order.side == "buy":
            # P1-Q18-fix: 记录当日买入股数，供 T+1 卖出校验使用。
            self._intraday_buys[(fill.trade_date, order.symbol)] += fill.quantity
        self.reporter.record_fill(fill)
        self.reporter.record_order(order)
        self.risk_checker.record_trade_count(1, fill.trade_date)
        benchmarks = {
            "arrival_price": base_price,
            "decision_price": order.tags.get("decision_price", base_price),
            "vwap_price": order.tags.get("vwap_price", base_price),
            "decision_time": order.created_at,
            "daily_vol": order.tags.get("daily_vol"),
            "horizon_days": order.tags.get("horizon_days", 1.0),
        }
        self.tca.record_order_execution(order, [fill], benchmarks)
        return fill


__all__ = [
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "CostMethod",
    "Order",
    "Fill",
    "PositionLot",
    "PositionRecord",
    "RiskRuleResult",
    "TradeFee",
    "TCARecord",
    "OrderBook",
    "OrderManager",
    "PositionManager",
    "PreTradeRiskCheck",
    "TCA",
    "TradeReporter",
    "TradingSystem",
]
