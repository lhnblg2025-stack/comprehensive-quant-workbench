"""
execution_broker.py — 实盘执行层（Broker 抽象接口）
V4.1 feature

提供统一的交易执行接口，支持：
1. Broker 抽象基类（模拟/实盘统一 API）
2. SimBroker — 模拟执行器（用于策略验证）
3. QmtBroker — 迅投 QMT 对接桩
4. PTradeBroker — 恒生 PTrade 对接桩
5. 订单管理（OrderManager）
6. 交易流水持久化

D3收敛登记 (2026-08-11): 执行域券商成本模型唯一真源。
P2-Q28 文档化 A股成本参数（佣金万0.85/最低5元/印花税万5仅卖出/过户费万0.1双边/滑点）
以模块级常量 COMMISSION_RATE/MIN_COMMISSION/STAMP_TAX_RATE/TRANSFER_FEE_RATE
与 estimate_trade_cost() 落地，SimBroker 与 execution.py/trading_system.py
的成本计算统一以此为准（薄壳转发，不强迁）。
"""

import os
import time
import uuid
import logging
import sqlite3
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# 数据模型
# ══════════════════════════════════════

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """订单数据结构"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""          # "buy" / "sell"
    order_type: str = "limit"
    price: float = 0.0
    volume: int = 0
    filled_volume: int = 0
    avg_price: float = 0.0
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""
    reason: str = ""

    def is_active(self) -> bool:
        return self.status in ("pending", "submitted", "partial")

    def pnl(self, market_price: float = 0.0) -> float:
        """估算持仓盈亏（需传入当前市价）"""
        price = market_price if market_price > 0 else self.avg_price
        if self.side == "buy":
            return (price - self.avg_price) * self.filled_volume
        else:
            return (self.avg_price - price) * self.filled_volume


@dataclass
class Position:
    """持仓数据结构"""
    symbol: str = ""
    volume: int = 0          # 持仓股数（正=多头，负=空头）
    cost_price: float = 0.0   # 持仓成本价
    market_price: float = 0.0 # 当前市价
    market_value: float = 0.0 # 市值
    pnl: float = 0.0          # 浮动盈亏
    pnl_pct: float = 0.0      # 盈亏比例
    updated_at: str = ""
    # P1-Q19-fix: T+1 跟踪——当日买入股数与最近买入日期
    today_buy_volume: int = 0
    last_buy_date: str = ""

    @property
    def is_long(self) -> bool:
        return self.volume > 0

    @property
    def is_short(self) -> bool:
        return self.volume < 0


@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str = ""
    total_asset: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    frozen_cash: float = 0.0
    profit: float = 0.0
    positions: list[Position] = field(default_factory=list)
    updated_at: str = ""


# ══════════════════════════════════════
# 券商成本模型（D3 唯一真源）
# ══════════════════════════════════════

# P2-Q28 文档化 A股交易成本参数（D3收敛登记 2026-08-11）：
# 佣金 万0.85（双边）、单笔最低 5 元、印花税 万5（仅卖出）、过户费 万0.1（双边）。
COMMISSION_RATE = 0.000085
MIN_COMMISSION = 5.0
STAMP_TAX_RATE = 0.0005
TRANSFER_FEE_RATE = 0.00001

# P1-6 (审计回测层): 跨引擎滑点默认值唯一真源（10bp）。
# 此前 backtest.py/config.py/SimBroker 各写 0.001(v10bp)、combined_backtest 写
# 0.0005(5bp)，同策略换引擎成本差一倍。此处统一为 0.001，全引擎默认滑点引用此常量。
DEFAULT_SLIPPAGE_RATE = 0.001


# P1-1 (审计回测层): 按流通市值分档的冲击成本滑点。
# 小盘/低流动性标的冲击被系统性低估：全引擎此前都是单一定值。现按流通市值分档——
#   流通市值 < 50亿        → 20bp（0.0020）
#   流通市值 50亿~200亿     → 10bp（0.0010）
#   流通市值 > 200亿        →  5bp（0.0005）
# 参考 hft-backtesting-reality §四-2"忽略冲击=低估成本" 与 audit 报告 P1-1：
# 小盘 20bp+、大盘 ≈5bp 的区分。输入可传美元/元的任一货币单位，仅比较量级。
def market_cap_tiered_slippage(market_cap: Optional[float]) -> float:
    """按流通市值返回冲击成本滑点率（0.0005 ~ 0.002）。

    Args:
        market_cap: 流通市值，单位为【元】。50亿=5e9、200亿=2e10。
                    None/0/非正 → 保守按小盘 20bp（未知流动性按最不利计）。
                    注: 若调用方持有"亿元"值, 请先 ×1e8 换算成元再传入;
                    本函数不做单位猜测, 阈值按【元】写死。

    Returns:
        滑点率，如 0.002 表示 20bp。
    """
    if market_cap is None or market_cap <= 0:
        return 0.002  # 未知市值 → 保守按小盘 20bp
    if market_cap < 50e8:          # < 50亿
        return 0.002
    if market_cap < 200e8:         # 50亿 ~ 200亿
        return 0.001
    return 0.0005                  # > 200亿


def estimate_trade_cost(
    side: str,
    quantity: int,
    price: float,
    commission_rate: float = COMMISSION_RATE,
    min_commission: float = MIN_COMMISSION,
    stamp_tax_rate: float = STAMP_TAX_RATE,
    transfer_fee_rate: float = TRANSFER_FEE_RATE,
) -> dict:
    """A股单笔成交成本模型（D3收敛：执行域唯一真源）。

    佣金 = max(成交额 × 佣金率, 单笔最低佣金)（双边）
    印花税 = 成交额 × 印花税率（仅卖出）
    过户费 = 成交额 × 过户费率（双边）

    Args:
        side: buy / sell（仅卖出计印花税）
        quantity: 成交股数（正数）
        price: 成交单价
        commission_rate / min_commission / stamp_tax_rate / transfer_fee_rate:
            费率覆盖项，默认取 P2-Q28 文档化 A股标准值。

    Returns:
        {"gross": 成交额, "commission": 佣金, "stamp_tax": 印花税,
         "transfer_fee": 过户费}
    """
    gross = quantity * price
    commission = max(gross * commission_rate, min_commission)
    stamp_tax = gross * stamp_tax_rate if side == "sell" else 0.0
    transfer_fee = gross * transfer_fee_rate
    return {
        "gross": gross,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "transfer_fee": transfer_fee,
    }


# ══════════════════════════════════════
# Broker 抽象基类
# ══════════════════════════════════════

class Broker:
    """交易执行器抽象基类。

    所有具体对接实现（模拟/QMT/PTrade）需继承此类并实现全部抽象方法。
    """

    def __init__(self, account_id: str = "default"):
        self.account_id = account_id
        self._orders: dict[str, Order] = {}
        self._order_db = OrderDB()

    def place_order(self, symbol: str, side: str, volume: int,
                    order_type: str = "limit", price: float = 0.0,
                    **kwargs) -> Order:
        """下单。返回 Order 对象（含 order_id）。
        子类必须实现 _do_place_order()。
        """
        order = Order(
            order_id=self._gen_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            volume=volume,
            status="pending",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # 调用具体实现
        result = self._do_place_order(order, **kwargs)
        self._orders[order.order_id] = result
        self._order_db.save_order(result)
        logger.info(f"[Broker] 下单: {order.order_id} {symbol} {side} {volume}@{price}")
        return result

    def cancel_order(self, order_id: str) -> bool:
        """撤单。返回是否成功。"""
        if order_id not in self._orders:
            logger.warning(f"[Broker] 订单不存在: {order_id}")
            return False

        order = self._orders[order_id]
        if not order.is_active():
            logger.warning(f"[Broker] 订单已终态: {order_id} ({order.status})")
            return False

        success = self._do_cancel_order(order_id)
        if success:
            order.status = "cancelled"
            order.updated_at = datetime.now().isoformat()
            self._order_db.save_order(order)
        return success

    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        if order_id in self._orders:
            return self._orders[order_id]
        # 从数据库查
        return self._order_db.load_order(order_id)

    def get_orders(self, status: Optional[str] = None) -> list[Order]:
        """获取订单列表。

        P2-Q19-fix(M181): 合并内存订单与 OrderDB 持久化记录——原实现仅查内存
        _orders，进程重启后历史订单不在内存中导致列表不完整。
        """
        merged: dict[str, Order] = {}
        for o in self._orders.values():
            merged[o.order_id] = o
        for o in self._order_db.load_orders():
            merged.setdefault(o.order_id, o)
        result = list(merged.values())
        if status:
            result = [o for o in result if o.status == status]
        return result

    def get_positions(self) -> list[Position]:
        """获取持仓"""
        return self._do_get_positions()

    def get_account(self) -> AccountInfo:
        """获取账户信息"""
        return self._do_get_account()

    def sync(self) -> None:
        """同步订单状态和账户信息（从柜台拉取最新数据）"""
        self._do_sync()

    # ── 子类需实现的内部方法 ──

    def _do_place_order(self, order: Order, **kwargs) -> Order:
        raise NotImplementedError

    def _do_cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def _do_get_positions(self) -> list[Position]:
        raise NotImplementedError

    def _do_get_account(self) -> AccountInfo:
        raise NotImplementedError

    def _do_sync(self) -> None:
        pass

    def _gen_order_id(self) -> str:
        return f"{self.account_id}_{uuid.uuid4().hex[:12].upper()}"


# ══════════════════════════════════════
# SimBroker — 模拟执行器
# ══════════════════════════════════════

class SimBroker(Broker):
    """模拟执行器。

    用于策略验证：在不受实盘条件约束的环境下测试下单逻辑。
    支持滑点模拟（基于 MarketImpact 模块）。
    """

    def __init__(self, account_id: str = "sim", initial_cash: float = 1_000_000.0,
                 commission_rate: float = COMMISSION_RATE, min_commission: float = MIN_COMMISSION,
                 stamp_tax_rate: float = STAMP_TAX_RATE, transfer_fee_rate: float = TRANSFER_FEE_RATE):
        super().__init__(account_id)
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self._positions: dict[str, Position] = {}
        self._market_prices: dict[str, float] = {}
        self._trade_history: list[dict] = []

        # 尝试导入市场冲击模型
        try:
            from quant_system.market_impact import MarketImpact, ACParams
            self._impact = MarketImpact(ACParams.a_share_defaults())
        except ImportError:
            self._impact = None

    def update_market_price(self, symbol: str, price: float):
        """更新市价"""
        self._market_prices[symbol] = price

    def update_market_prices(self, prices: dict[str, float]):
        """批量更新市价"""
        self._market_prices.update(prices)

    # ── 实现抽象方法 ──

    def _do_place_order(self, order: Order, **kwargs) -> Order:
        # 模拟滑点
        market_price = self._market_prices.get(order.symbol, 0)
        price = order.price if order.price > 0 else market_price
        if price <= 0:
            order.status = "rejected"
            order.reason = "无效价格"
            return order

        # P2-Q19-fix(M175): 限价单按限价条件撮合——条件不满足保持 pending，不立即成交。
        # 原实现无视限价：市价 8 限价 10 的买单按 10.009 成交（应约 8）、
        # 市价 10 限价 8 的买单也成交（应挂单等待）。
        if order.order_type == "limit":
            limit = order.price
            if limit <= 0:
                order.status = "rejected"
                order.reason = "限价单缺少限价"
                return order
            if market_price <= 0:
                # 无行情无法判定是否触发，保持 pending 等待行情
                order.reason = "等待行情（限价未判定）"
                return order
            if order.side == "buy" and market_price > limit:
                order.reason = f"限价未触发 (市价{market_price:.2f} > 限价{limit:.2f})"
                return order
            if order.side == "sell" and market_price < limit:
                order.reason = f"限价未触发 (市价{market_price:.2f} < 限价{limit:.2f})"
                return order
            # 触发后按市价（±滑点）成交，优于限价
            price = market_price

        # 应用滑点
        if self._impact and 'adv' in kwargs and 'sigma' in kwargs:
            cost = self._impact.total_cost(
                trade_value=price * order.volume,
                stock_price=price,
                adv_shares=kwargs['adv'],
                sigma=kwargs['sigma'],
                side=order.side,
            )
            slippage = cost['cost_rate']
        else:
            # P1-6 (审计回测层): 默认滑点引用唯一真源 DEFAULT_SLIPPAGE_RATE(10bp)
            slippage = DEFAULT_SLIPPAGE_RATE  # 默认万10滑点

        if order.side == "buy":
            exec_price = price * (1 + slippage)
        else:
            exec_price = price * (1 - slippage)

        # 成本核算（D3收敛: 转发 estimate_trade_cost 唯一真源，公式见模块级券商成本模型）
        fee = estimate_trade_cost(
            side=order.side,
            quantity=order.volume,
            price=exec_price,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
            stamp_tax_rate=self.stamp_tax_rate,
            transfer_fee_rate=self.transfer_fee_rate,
        )
        cost = fee["gross"]
        commission = fee["commission"]
        stamp_tax = fee["stamp_tax"]
        transfer_fee = fee["transfer_fee"]
        total_cost = cost + commission + stamp_tax + transfer_fee

        if order.side == "buy":
            if total_cost > self.cash:
                order.status = "rejected"
                order.reason = "资金不足"
                return order
            self.cash -= total_cost
        else:
            pos = self._positions.get(order.symbol)
            if pos is None or pos.volume < order.volume:
                order.status = "rejected"
                order.reason = "持仓不足"
                return order
            # P1-Q19-fix: A股 T+1——当日买入股数不可当日卖出
            sellable = self._sellable_volume(pos)
            if order.volume > sellable:
                order.status = "rejected"
                order.reason = f"T+1: 可卖{sellable} < 卖出{order.volume}（当日买入次日方可卖出）"
                return order
            self.cash += cost - commission - stamp_tax - transfer_fee
            # 持仓扣减统一由下方 _update_position 处理（Q19 修复：此处不再直接扣减，
            # 避免与 _update_position 的卖出分支重复扣减导致持仓凭空消失）

        order.filled_volume = order.volume
        order.avg_price = exec_price
        order.status = "filled"
        order.updated_at = datetime.now().isoformat()

        # 更新持仓
        self._update_position(order.symbol, order.side, order.volume, exec_price)

        # 记录成交
        self._trade_history.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'volume': order.volume,
            'price': exec_price,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'transfer_fee': transfer_fee,
            'time': order.updated_at,
        })

        return order

    def _do_cancel_order(self, order_id: str) -> bool:
        return True  # 模拟环境立即成功

    def _do_get_positions(self) -> list[Position]:
        # 更新市值
        for pos in self._positions.values():
            pos.market_price = self._market_prices.get(pos.symbol, 0)
            pos.market_value = pos.market_price * pos.volume
            pos.pnl = (pos.market_price - pos.cost_price) * pos.volume
            if pos.cost_price > 0:
                pos.pnl_pct = pos.pnl / (pos.cost_price * pos.volume)
        return list(self._positions.values())

    def _do_get_account(self) -> AccountInfo:
        positions = self._do_get_positions()
        mv = sum(p.market_value for p in positions if p.volume > 0)
        return AccountInfo(
            account_id=self.account_id,
            total_asset=self.cash + mv,
            cash=self.cash,
            market_value=mv,
            positions=positions,
            updated_at=datetime.now().isoformat(),
        )

    def _do_sync(self) -> None:
        pass  # 模拟环境无需同步

    def _sellable_volume(self, pos: Position) -> int:
        """T+1 可卖数量 = 持仓 - 当日买入（当日买入次日方可卖出）"""
        today = datetime.now().strftime("%Y-%m-%d")
        if pos.last_buy_date != today:
            return pos.volume
        return max(0, pos.volume - pos.today_buy_volume)

    def _update_position(self, symbol: str, side: str, volume: int, price: float):
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        pos = self._positions[symbol]
        if side == "buy":
            # P1-Q19-fix: 记录当日买入股数（T+1 冻结），跨日自动重置
            today = datetime.now().strftime("%Y-%m-%d")
            if pos.last_buy_date != today:
                pos.today_buy_volume = 0
                pos.last_buy_date = today
            pos.today_buy_volume += volume
            old_cost = pos.cost_price * pos.volume
            new_cost = price * volume
            pos.volume += volume
            pos.cost_price = (old_cost + new_cost) / pos.volume if pos.volume > 0 else 0
        else:
            pos.volume -= volume

        if pos.volume <= 0:
            del self._positions[symbol]

    def reset(self, cash: float = 1_000_000.0):
        """重置模拟账户"""
        # P2-Q19-fix(L184): 全量清理——连同订单与行情上下文一并重置，
        # 避免上一轮 _orders/_market_prices 残留影响下一轮撮合与查询
        self.cash = cash
        self._positions.clear()
        self._trade_history.clear()
        self._orders.clear()
        self._market_prices.clear()


# ══════════════════════════════════════
# QmtBroker — 迅投 QMT 对接桩
# ══════════════════════════════════════

class QmtBroker(Broker):
    """迅投 QMT 交易接口对接桩。

    生产环境需要：
    - QMT 客户端（迅投极速交易终端）
    - 配置 account_id、client_ip、port

    当前为桩实现，仅打印日志。接入实盘时需填充 _do_place_order 等。
    """

    def __init__(self, account_id: str = "", client_ip: str = "127.0.0.1",
                 port: int = 10000):
        super().__init__(account_id)
        self.client_ip = client_ip
        self.port = port
        self._connected = False

    def connect(self) -> bool:
        """连接 QMT 客户端"""
        logger.info(f"[QMT] 连接: {self.client_ip}:{self.port} account={self.account_id}")
        # P2-Q19-fix(L183): 桩实现——不会建立真实柜台连接，禁止静默"成功"
        logger.warning("[QMT] 桩实现：仅置 _connected=True，未建立真实柜台连接")
        self._connected = True
        return True

    def disconnect(self):
        """断开连接"""
        self._connected = False
        logger.info("[QMT] 已断开")

    def _do_place_order(self, order: Order, **kwargs) -> Order:
        """通过 QMT API 下单。

        实盘时需调用 xtquant.xttrader.XtQuantTrader 相关接口：
        - order_stock(account, stock_code, order_type, order_volume, price_type, price)
        """
        if not self._connected:
            order.status = "rejected"
            order.reason = "QMT 未连接"
            return order

        logger.info(f"[QMT] 下单指令: {order.symbol} {order.side} {order.volume}@{order.price}")
        # P2-Q19-fix(L183): 桩实现——订单未发送到真实柜台，禁止静默"成功"
        logger.warning("[QMT] 桩实现：未真实下单，仅置 submitted 状态（接入实盘前不可视为成交）")
        order.status = "submitted"
        order.filled_volume = 0
        # 实盘：监听异步回调更新状态
        return order

    def _do_cancel_order(self, order_id: str) -> bool:
        # P2-Q19-fix(L183): 桩实现——撤单未提交真实柜台，返回 True 仅为仿真
        logger.warning(f"[QMT] 桩实现：撤单 {order_id} 未提交真实柜台，返回 True 仅为仿真")
        return True

    def _do_get_positions(self) -> list[Position]:
        """同步持仓。

        实盘时调用 xtquant 的 get_stock_positions()。
        """
        # P2-Q19-fix(L183): 桩实现——返回空持仓，未同步真实柜台
        logger.warning("[QMT] 桩实现：返回空持仓，未同步真实柜台")
        return []

    def _do_get_account(self) -> AccountInfo:
        # P2-Q19-fix(L183): 桩实现——返回空账户，未同步真实柜台
        logger.warning("[QMT] 桩实现：返回空账户信息，未同步真实柜台")
        return AccountInfo(account_id=self.account_id)

    def _do_sync(self):
        """定时同步订单状态"""
        logger.debug("[QMT] 同步中...")


# ══════════════════════════════════════
# PTradeBroker — 恒生 PTrade 对接桩
# ══════════════════════════════════════

class PTradeBroker(Broker):
    """恒生 PTrade 交易接口对接桩。

    生产环境需 ptrade 终端环境支持：
    - 在 PTrade 策略运行环境下使用 ptrade 内建 API

    当前为桩实现。
    """

    def __init__(self, account_id: str = "", server: str = "localhost", port: int = 1111):
        super().__init__(account_id)
        self.server = server
        self.port = port
        self._connected = False

    def connect(self) -> bool:
        logger.info(f"[PTrade] 连接: {self.server}:{self.port} account={self.account_id}")
        # P2-Q19-fix(L183): 桩实现——不会建立真实柜台连接，禁止静默"成功"
        logger.warning("[PTrade] 桩实现：仅置 _connected=True，未建立真实柜台连接")
        self._connected = True
        return True

    def _do_place_order(self, order: Order, **kwargs) -> Order:
        if not self._connected:
            order.status = "rejected"
            order.reason = "PTrade 未连接"
            return order

        logger.info(f"[PTrade] 下单: {order.symbol} {order.side} {order.volume}@{order.price}")
        # P2-Q19-fix(L183): 桩实现——订单未发送到真实柜台，禁止静默"成功"
        logger.warning("[PTrade] 桩实现：未真实下单，仅置 submitted 状态（接入实盘前不可视为成交）")
        order.status = "submitted"
        return order

    def _do_cancel_order(self, order_id: str) -> bool:
        # P2-Q19-fix(L183): 桩实现——撤单未提交真实柜台，返回 True 仅为仿真
        logger.warning(f"[PTrade] 桩实现：撤单 {order_id} 未提交真实柜台，返回 True 仅为仿真")
        return True

    def _do_get_positions(self) -> list[Position]:
        # P2-Q19-fix(L183): 桩实现——返回空持仓，未同步真实柜台
        logger.warning("[PTrade] 桩实现：返回空持仓，未同步真实柜台")
        return []

    def _do_get_account(self) -> AccountInfo:
        # P2-Q19-fix(L183): 桩实现——返回空账户，未同步真实柜台
        logger.warning("[PTrade] 桩实现：返回空账户信息，未同步真实柜台")
        return AccountInfo(account_id=self.account_id)

    def _do_sync(self):
        pass


# ══════════════════════════════════════
# OrderDB — 交易流水持久化
# ══════════════════════════════════════

def _resolve_orders_db_path() -> str:
    """W2.5 收尾：家目录只读时回退仓库内 generated/quant_state（同 trade_db 教训）。"""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.expanduser('~/.quant_system/trade_logs'),
        os.path.join(repo, 'generated', 'quant_state', 'trade_logs'),
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, '.wtest')
            with open(test, 'w', encoding='utf-8') as f:
                f.write('ok')
            os.remove(test)
            return os.path.join(d, 'orders.db')
        except OSError:
            continue
    return os.path.join(candidates[-1], 'orders.db')


class OrderDB:
    """订单数据库（SQLite），保存所有成交记录。

    DB 路径: ~/.quant_system/trade_logs/orders.db（只读时回退 generated/quant_state）
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = _resolve_orders_db_path()
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                price REAL,
                volume INTEGER,
                filled_volume INTEGER,
                avg_price REAL,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                reason TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT,
                symbol TEXT,
                side TEXT,
                volume INTEGER,
                price REAL,
                commission REAL,
                stamp_tax REAL,
                trade_time TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_order(self, order: Order):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO orders
            (order_id, symbol, side, order_type, price, volume,
             filled_volume, avg_price, status, created_at, updated_at, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (order.order_id, order.symbol, order.side, order.order_type,
              order.price, order.volume, order.filled_volume, order.avg_price,
              order.status, order.created_at, order.updated_at, order.reason))
        conn.commit()
        conn.close()

    def load_order(self, order_id: str) -> Optional[Order]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        conn.close()
        if row:
            # P2-Q19-fix(M181): 列序与 Order 字段一致，直接整体展开；
            # 原 *row[1:], order_id=row[0] 会把 order_id 重复绑定导致 TypeError
            return Order(*row)
        return None

    def load_orders(self, symbol: Optional[str] = None,
                    date_from: Optional[str] = None,
                    date_to: Optional[str] = None) -> list[Order]:
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT * FROM orders WHERE 1=1"
        params = []
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        if date_from:
            sql += " AND created_at>=?"
            params.append(date_from)
        if date_to:
            sql += " AND created_at<=?"
            params.append(date_to)
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        # P2-Q19-fix(M181): 同 load_order——直接整体展开
        return [Order(*r) for r in rows]


# ══════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════

def create_broker(broker_type: str = "sim", **kwargs) -> Broker:
    """Broker 工厂。

    Parameters
    ----------
    broker_type : str
        "sim" — 模拟执行器（默认）
        "qmt" — 迅投 QMT
        "ptrade" — 恒生 PTrade
    **kwargs : dict
        传递给具体 Broker 构造函数的参数

    Returns
    -------
    Broker
    """
    if broker_type == "sim":
        return SimBroker(**kwargs)
    elif broker_type == "qmt":
        return QmtBroker(**kwargs)
    elif broker_type == "ptrade":
        return PTradeBroker(**kwargs)
    else:
        raise ValueError(f"未知 Broker 类型: {broker_type}")


# ══════════════════════════════════════
# StrategyRunner — 策略在线运行器
# ══════════════════════════════════════

class StrategyRunner:
    """策略在线运行器。

    将回测策略包装为实盘/模拟信号处理器：
    1. 接收实时信号
    2. 计算目标持仓
    3. 生成调仓指令
    4. 通过 Broker 执行
    """

    def __init__(self, broker: Broker, strategy_name: str = "default"):
        self.broker = broker
        self.strategy_name = strategy_name
        self.target_positions: dict[str, int] = {}

    def set_target(self, symbol: str, target_volume: int):
        """设置目标持仓"""
        self.target_positions[symbol] = target_volume

    def set_targets(self, targets: dict[str, int]):
        """批量设置目标持仓"""
        self.target_positions.update(targets)

    def execute(self, market_prices: dict[str, float]) -> list[Order]:
        """根据目标持仓与当前持仓之差，执行调仓。

        Parameters
        ----------
        market_prices : dict[str, float]
            symbol -> 当前市价

        Returns
        -------
        list[Order]
            本次调仓产生的订单列表
        """
        if isinstance(self.broker, SimBroker):
            self.broker.update_market_prices(market_prices)

        current_pos = {p.symbol: p.volume for p in self.broker.get_positions()}
        orders = []

        for symbol, target in self.target_positions.items():
            current = current_pos.get(symbol, 0)
            diff = target - current
            if diff == 0:
                continue

            price = market_prices.get(symbol)
            if not price or price <= 0:
                logger.warning(f"[StrategyRunner] 跳过 {symbol}: 市价不可用")
                continue

            side = "buy" if diff > 0 else "sell"
            order = self.broker.place_order(
                symbol=symbol, side=side, volume=abs(diff),
                order_type="market", price=price,
            )
            orders.append(order)

        return orders

    def close_all(self) -> list[Order]:
        """平掉所有持仓"""
        current_pos = {p.symbol: p.volume for p in self.broker.get_positions()}
        self.target_positions = {s: 0 for s in current_pos}
        # V4.1 fix: 传入当前市价而非零价格，否则execute会跳过所有卖出
        prices = {p.symbol: p.market_price for p in self.broker.get_positions()}
        return self.execute(prices)
