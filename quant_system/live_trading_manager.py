"""
live_trading_manager.py — 实盘交易管理与监控
V4.1 feature

提供实盘信号执行、订单管理、持仓追踪、风控检查。

⚠️ 模块状态说明：行情已接入 akshare 真实数据；
   下单/执行链路仍为仿真（ExecutionEngine 内存撮合，未接券商柜台），
   接入真实柜台前不应视为实盘可用。
"""

import time
import uuid
import logging
import threading
from datetime import datetime
from typing import Optional, Callable
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# D3/D组收敛: 费率唯一真源为 execution_broker 常量（同数值同方向），此处改为引用。
# P2-Q19-fix(M173): A股交易成本模型（与 execution / SimBroker / backtest_engine 保持一致）
from quant_system import execution_broker as _execution_broker

COMMISSION_RATE = _execution_broker.COMMISSION_RATE       # 佣金率 万0.85
MIN_COMMISSION = _execution_broker.MIN_COMMISSION         # 佣金最低 5 元/笔（A股标准）
STAMP_TAX_RATE = _execution_broker.STAMP_TAX_RATE         # 印花税 0.05% 仅卖出
TRANSFER_FEE_RATE = _execution_broker.TRANSFER_FEE_RATE   # 过户费 0.001% 双边


class LiveOrder:
    """实盘订单"""
    def __init__(self, symbol: str, side: str, volume: int,
                 order_type: str = "limit", price: float = 0.0):
        self.order_id = f"L{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.symbol = symbol
        self.side = side
        self.volume = volume
        self.order_type = order_type
        self.price = price
        self.filled_volume = 0
        self.avg_price = 0.0
        self.status = "pending"  # pending/submitted/partial/filled/cancelled/rejected
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.messages: list[str] = []

    def fill(self, volume: int, price: float):
        """部分/全部成交"""
        self.filled_volume += volume
        total_cost = self.avg_price * (self.filled_volume - volume) + price * volume
        self.avg_price = total_cost / max(self.filled_volume, 1)
        self.status = "partial" if self.filled_volume < self.volume else "filled"
        self.updated_at = datetime.now()
        self.messages.append(f"成交 {volume}@{price:.2f}")

    def cancel(self):
        self.status = "cancelled"
        self.updated_at = datetime.now()
        self.messages.append("已撤单")

    def reject(self, reason: str = ""):
        self.status = "rejected"
        self.updated_at = datetime.now()
        self.messages.append(f"拒绝: {reason}")

    @property
    def remaining(self) -> int:
        return self.volume - self.filled_volume

    @property
    def is_active(self) -> bool:
        return self.status in ("pending", "submitted", "partial")

    def __repr__(self):
        return f"Order({self.order_id[:12]} {self.symbol} {self.side} {self.filled_volume}/{self.volume} {self.status})"


class LivePosition:
    """实盘持仓"""
    def __init__(self, symbol: str, volume: int = 0,
                 cost_price: float = 0.0):
        self.symbol = symbol
        self.volume = volume
        self.cost_price = cost_price
        self.market_price = cost_price
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.updated_at = datetime.now()
        # P1-Q19-fix: T+1 跟踪——当日买入股数与最近买入日期
        self.today_buy_volume = 0
        self.last_buy_date = ""

    def update_market(self, price: float):
        """更新市价"""
        self.market_price = price
        self.pnl = (price - self.cost_price) * self.volume
        self.pnl_pct = (price / max(self.cost_price, 1e-12) - 1) * 100 if self.volume > 0 else 0
        self.updated_at = datetime.now()

    @property
    def sellable(self) -> int:
        """T+1 可卖数量 = 持仓 - 当日买入（当日买入次日方可卖出）"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_buy_date != today:
            return self.volume
        return max(0, self.volume - self.today_buy_volume)

    def add(self, volume: int, price: float):
        """加仓"""
        # P1-Q19-fix: 记录当日买入股数（跨日自动重置）
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_buy_date != today:
            self.today_buy_volume = 0
            self.last_buy_date = today
        self.today_buy_volume += volume
        total_cost = self.cost_price * self.volume + price * volume
        self.volume += volume
        self.cost_price = total_cost / max(self.volume, 1)
        self.update_market(price)

    def reduce(self, volume: int, price: float) -> float:
        """减仓，返回已实现盈亏"""
        ratio = volume / max(self.volume, 1)
        realized_pnl = (price - self.cost_price) * volume
        self.volume -= volume
        if self.volume <= 0:
            self.volume = 0
            self.cost_price = 0
        else:
            self.update_market(price)
        return realized_pnl

    def __repr__(self):
        return f"Pos({self.symbol} {self.volume} @ {self.cost_price:.2f} PnL={self.pnl:.2f})"


class Account:
    """交易账户"""
    def __init__(self, name: str = "default", initial_cash: float = 1_000_000.0):
        self.name = name
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, LivePosition] = {}
        self.orders: list[LiveOrder] = []
        self.trades: list[dict] = []
        self.total_pnl = 0.0
        self.created_at = datetime.now()

    @property
    def market_value(self) -> float:
        return sum(p.volume * p.market_price for p in self.positions.values())

    @property
    def total_asset(self) -> float:
        return self.cash + self.market_value

    @property
    def return_pct(self) -> float:
        return (self.total_asset / self.initial_cash - 1) * 100

    def summary(self) -> str:
        lines = [
            f"账户: {self.name}",
            f"总资产: {self.total_asset:,.2f}",
            f"现金: {self.cash:,.2f}",
            f"市值: {self.market_value:,.2f}",
            f"收益率: {self.return_pct:.2f}%",
            f"持仓数: {len(self.positions)}",
        ]
        return "\n".join(lines)


class MarketDataFeed:
    """市场数据推送"""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, symbol: str, callback: Callable):
        """订阅行情"""
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
        self._subscribers[symbol].append(callback)

    def unsubscribe(self, symbol: str, callback: Callable):
        if symbol in self._subscribers and callback in self._subscribers[symbol]:
            self._subscribers[symbol].remove(callback)

    def on_tick(self, symbol: str, price: float, volume: int = 0, timestamp: str = ""):
        """处理Tick"""
        for cb in self._subscribers.get(symbol, []):
            try:
                cb(symbol, price, volume, timestamp)
            except Exception as e:
                logger.error(f"回调执行失败 {symbol}: {e}")


class RiskChecks:
    """实盘风控检查"""

    def __init__(self, max_position_value: float = 1e7,
                 max_position_pct: float = 0.1,
                 max_daily_loss: float = 0.05,
                 max_leverage: float = 1.0,
                 min_cash_reserve: float = 1e5):
        self.max_position_value = max_position_value
        self.max_position_pct = max_position_pct
        self.max_daily_loss = max_daily_loss
        self.max_leverage = max_leverage
        self.min_cash_reserve = min_cash_reserve
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._max_daily_trades = 100
        self._reset_hour = 9  # 每日重置时间
        # P2-Q19-fix(M176): 记录最近一次重置日期，跨日自动重置日内风控计数
        self._daily_date = datetime.now().strftime("%Y-%m-%d")

    def reset_daily(self):
        """每日重置"""
        self._daily_pnl = 0.0
        self._daily_trades = 0

    def _check_daily_reset(self):
        """P2-Q19-fix(M176): 跨日自动重置日内计数（_daily_pnl/_daily_trades）。"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_date != today:
            self.reset_daily()
            self._daily_date = today

    def add_daily_pnl(self, pnl: float):
        """P2-Q19-fix(M176): 累计当日已实现盈亏（负值=亏损），成交时由执行引擎调用。"""
        self._check_daily_reset()
        self._daily_pnl += pnl

    def daily_loss_limit(self, account: Account) -> float:
        """P2-Q19-fix(M176): 当日亏损熔断阈值（金额）= max_daily_loss × 初始资金。"""
        return self.max_daily_loss * account.initial_cash

    def check_order(self, account: Account, symbol: str, side: str,
                    volume: int, price: float) -> tuple[bool, str]:
        """检查订单是否可执行"""
        self._check_daily_reset()
        value = price * volume

        # P2-Q19-fix(M176): 当日亏损熔断——已实现亏损达阈值时禁止新开仓/加仓；
        # 卖出减仓仍放行，避免亏损时无法止损离场
        if side == "buy" and self._daily_pnl < 0 and -self._daily_pnl >= self.daily_loss_limit(account):
            return False, (f"日内亏损熔断: 当日亏损 {-self._daily_pnl:.0f} 超过上限 "
                           f"{self.daily_loss_limit(account):.0f}")

        # P1-Q19-fix: 卖出必须持有足额持仓（A股无融券不可卖空）+ T+1 可卖数量校验
        if side == "sell":
            pos = account.positions.get(symbol)
            held = pos.volume if pos else 0
            if pos is None or held < volume:
                return False, f"持仓不足: 持有{held} < 卖出{volume}"
            if pos.sellable < volume:
                return False, f"T+1限制: 可卖{pos.sellable} < 卖出{volume}（当日买入次日方可卖出）"

        # 资金检查
        if side == "buy" and value > account.cash - self.min_cash_reserve:
            return False, f"资金不足: 需{value:.0f} 可用{account.cash:.0f}"

        # 持仓集中度
        if side == "buy":
            new_pos_value = value + account.positions.get(symbol, LivePosition(symbol)).volume * price
            if new_pos_value > account.total_asset * self.max_position_pct:
                return False, f"持仓超限: {new_pos_value:.0f} > {account.total_asset * self.max_position_pct:.0f}"

        # 杠杆检查
        if side == "buy":
            new_exposure = account.market_value + value
            if new_exposure > account.total_asset * self.max_leverage:
                return False, f"杠杆超限: {new_exposure / max(account.total_asset, 1):.2f}x"

        # 日交易次数
        if self._daily_trades >= self._max_daily_trades:
            return False, f"日内交易超限: {self._daily_trades}"

        return True, "OK"

    def check_position(self, account: Account) -> list[str]:
        """全仓风控检查"""
        self._check_daily_reset()
        warnings = []

        # P2-Q19-fix(M176): 当日亏损熔断预警
        if self._daily_pnl < 0 and -self._daily_pnl >= self.daily_loss_limit(account):
            warnings.append(
                f"日内亏损熔断: 当日亏损 {-self._daily_pnl:.0f} 超过上限 "
                f"{self.daily_loss_limit(account):.0f}"
            )

        # 集中度
        for sym, pos in account.positions.items():
            value = pos.volume * pos.market_price
            if value > account.total_asset * self.max_position_pct:
                warnings.append(f"{sym} 超集中度: {value/account.total_asset*100:.1f}%")

        # 杠杆
        leverage = account.market_value / max(account.total_asset, 1)
        if leverage > self.max_leverage:
            warnings.append(f"杠杆超限: {leverage:.2f}x")

        # 现金储备
        if account.cash < self.min_cash_reserve:
            warnings.append(f"现金不足: {account.cash:.0f}")

        return warnings


class ExecutionEngine:
    """交易执行引擎"""

    def __init__(self, account: Account, risk_checks: RiskChecks):
        self.account = account
        self.risk = risk_checks
        self._pending: list[LiveOrder] = []
        self._execution_delay = 0.1  # 执行延迟模拟
        # P2-Q19-fix(M177): 最近行情价缓存，供无显式价格的订单做风控计价
        self._last_prices: dict[str, float] = {}

    def place_order(self, symbol: str, side: str, volume: int,
                     order_type: str = "market", price: float = 0.0,
                     check_risk: bool = True) -> Optional[LiveOrder]:
        """下单"""
        if check_risk:
            # P2-Q19-fix(M177): 用传入价/最近行情价做风控计价，无有效价则拒绝下单。
            # 原 `price or 1` 按 1 元/股计价，资金/集中度/杠杆校验完全失真。
            eff_price = price if price and price > 0 else self._last_prices.get(symbol, 0)
            if eff_price <= 0:
                logger.warning(f"[Exec] 拒绝下单 {symbol}: 无有效行情价，无法进行资金/集中度/杠杆风控")
                return None
            ok, msg = self.risk.check_order(self.account, symbol, side, volume, eff_price)
            if not ok:
                logger.warning(f"[Exec] 风控拒绝: {msg}")
                return None
        else:
            eff_price = price
            if eff_price <= 0:
                eff_price = self._last_prices.get(symbol, 0)

        order = LiveOrder(symbol, side, volume, order_type, eff_price)
        self._pending.append(order)
        self.account.orders.append(order)
        logger.info(f"[Exec] 下单: {order}")
        return order

    def execute_pending(self, market_prices: dict[str, float]) -> list[LiveOrder]:
        """执行所有待处理订单"""
        filled = []
        # P2-Q19-fix(M177): 记录最近行情价供后续订单风控计价
        self._last_prices.update({k: v for k, v in market_prices.items() if v and v > 0})
        for order in self._pending[:]:
            if not order.is_active:
                self._pending.remove(order)
                continue

            price = market_prices.get(order.symbol, order.price)
            if price <= 0:
                continue

            # 市价单立即成交
            if order.order_type == "market":
                fill_volume = order.remaining
                actual_price = price * (1.001 if order.side == "buy" else 0.999)
                try:
                    # P1-Q19-fix: 先校验并更新账户（超卖/T+1），失败则拒单而非标记成交
                    self._update_account(order, fill_volume, actual_price)
                except ValueError as e:
                    order.reject(str(e))
                    logger.warning(f"[Exec] 拒绝成交 {order}: {e}")
                    self._pending.remove(order)
                    continue
                order.fill(fill_volume, actual_price)
                filled.append(order)
                self._pending.remove(order)
            # 限价单
            elif order.order_type == "limit":
                if (order.side == "buy" and price <= order.price) or (
                        order.side == "sell" and price >= order.price):
                    try:
                        # P1-Q19-fix: 同上，先校验并更新账户
                        self._update_account(order, order.remaining, price)
                    except ValueError as e:
                        order.reject(str(e))
                        logger.warning(f"[Exec] 拒绝成交 {order}: {e}")
                        self._pending.remove(order)
                        continue
                    order.fill(order.remaining, price)
                    filled.append(order)
                    self._pending.remove(order)

        return filled

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        for order in self._pending:
            if order.order_id == order_id:
                order.cancel()
                self._pending.remove(order)
                return True
        return False

    def cancel_all(self) -> int:
        """全部撤单"""
        n = len(self._pending)
        for order in self._pending:
            order.cancel()
        self._pending.clear()
        return n

    def _update_account(self, order: LiveOrder, volume: int, price: float):
        """更新账户持仓（含 A股交易成本核算）"""
        gross = price * volume
        # P2-Q19-fix(M173): 成交成本——佣金(最低5元/笔)、印花税(卖出0.05%)、过户费(双边0.001%)
        commission = max(gross * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = gross * STAMP_TAX_RATE if order.side == "sell" else 0.0
        transfer_fee = gross * TRANSFER_FEE_RATE
        total_cost = commission + stamp_tax + transfer_fee

        if order.side == "sell":
            pos = self.account.positions.get(order.symbol)
            # P1-Q19-fix: 超卖/T+1 防御——无券不可卖空，当日买入不可卖出
            if pos is None or pos.volume < volume:
                raise ValueError(
                    f"超卖: {order.symbol} 持仓{pos.volume if pos else 0} < 卖出{volume}"
                )
            if pos.sellable < volume:
                raise ValueError(
                    f"T+1 限制: {order.symbol} 可卖{pos.sellable} < 卖出{volume}（当日买入次日方可卖出）"
                )
            realized = pos.reduce(volume, price)
            self.account.cash += gross - total_cost
            self.account.total_pnl += realized
            # P2-Q19-fix(M176): 当日已实现盈亏 = 卖出实现盈亏 - 交易成本
            self.risk.add_daily_pnl(realized - total_cost)
        else:
            pos = self.account.positions.get(order.symbol)
            if pos is None:
                pos = LivePosition(order.symbol)
                self.account.positions[order.symbol] = pos
            pos.add(volume, price)
            self.account.cash -= gross + total_cost
            # P2-Q19-fix(M176): 买入仅支出成本（未实现盈亏不计入当日已实现）
            self.risk.add_daily_pnl(-total_cost)

        self.account.trades.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "volume": volume,
            "price": price,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "time": datetime.now().isoformat(),
        })
        self.risk._daily_trades += 1


class LiveTradingManager:
    """实盘交易管理器"""

    def __init__(self, account_name: str = "main",
                 initial_cash: float = 1_000_000.0):
        self.account = Account(account_name, initial_cash)
        self.risk = RiskChecks()
        self.executor = ExecutionEngine(self.account, self.risk)
        self.feed = MarketDataFeed()
        self._strategies: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_strategy(self, name: str, signal_fn: Callable,
                     symbols: list[str], params: dict = None):
        """添加交易策略"""
        self._strategies.append({
            "name": name,
            "signal_fn": signal_fn,
            "symbols": symbols,
            "params": params or {},
        })
        logger.info(f"[LTM] 添加策略: {name}")

    def start(self, interval: float = 5.0):
        """启动策略循环"""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval,),
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[LTM] 启动, 间隔{interval}s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[LTM] 停止")

    def _run_loop(self, interval: float):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"[LTM] 运行异常: {e}")
            time.sleep(interval)

    def _tick(self):
        """执行一次策略循环"""
        # 获取行情
        prices = self._get_prices()

        # 风控检查
        warnings = self.risk.check_position(self.account)
        if warnings:
            for w in warnings:
                logger.warning(f"[LTM] 风控: {w}")

        # 执行策略
        for strategy in self._strategies:
            try:
                signals = strategy["signal_fn"](prices, **strategy["params"])
                for sym in strategy["symbols"]:
                    if sym in signals:
                        signal = signals[sym]
                        if signal > 0.5:
                            # P1-Q19-fix: A股买入必须 100 股整数倍，
                            # 向下取整（如 0.55 → 550 → 500 股）
                            buy_vol = (int(signal * 1000) // 100) * 100
                            if buy_vol <= 0:
                                continue
                            # P2-Q19-fix(M177): 传入当前行情价供风控校验（无价则由引擎拒绝）
                            self.executor.place_order(sym, "buy", buy_vol, price=prices.get(sym, 0))
                        elif signal < -0.5:
                            pos = self.account.positions.get(sym)
                            vol = pos.volume if pos else 0
                            if vol > 0:
                                self.executor.place_order(sym, "sell", vol, price=prices.get(sym, 0))
            except Exception as e:
                logger.error(f"[LTM] 策略 {strategy['name']} 失败: {e}")

        # 执行订单
        self.executor.execute_pending(prices)

    def _get_prices(self) -> dict[str, float]:
        """获取当前行情（akshare 真实行情）。

        优先使用 akshare 东财全市场快照（stock_zh_a_spot_em）拉取真实价格；
        接口/网络异常时记录告警并返回空 dict —— 不再硬编码伪价格，
        避免市价单以虚构的 10.0 元成交（Q19 修复）。
        """
        prices: dict[str, float] = {}
        all_symbols: set[str] = set()
        for s in self._strategies:
            all_symbols.update(s["symbols"])
        if not all_symbols:
            return prices

        # 归一化为 6 位代码（兼容 "600000" / "sh600000" / "SH600000" 写法）
        wanted: dict[str, str] = {}
        for sym in all_symbols:
            code = str(sym).strip().lower()
            for prefix in ("sh", "sz", "bj"):
                if code.startswith(prefix):
                    code = code[len(prefix):]
                    break
            code = code.zfill(6)
            if code.isdigit():
                wanted.setdefault(code, str(sym))
        if not wanted:
            logger.warning(f"[LTM] 无效的标的代码: {sorted(all_symbols)}")
            return prices

        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is not None and not spot.empty and "代码" in spot.columns and "最新价" in spot.columns:
                code_to_price = {
                    str(row["代码"]).zfill(6): row["最新价"]
                    for _, row in spot[["代码", "最新价"]].iterrows()
                }
                for code, original in wanted.items():
                    try:
                        price = float(code_to_price.get(code))
                    except (TypeError, ValueError):
                        continue
                    if price > 0:
                        prices[original] = price
            else:
                logger.warning("[LTM] akshare 快照为空或缺少代码/最新价列")
        except Exception as e:
            logger.warning(f"[LTM] 行情获取失败(akshare): {e}")

        missing = [s for s in all_symbols if s not in prices]
        if missing:
            logger.warning(f"[LTM] 未获取到行情的标的: {missing}（策略将跳过这些标的）")
        return prices

    def status(self) -> str:
        """管理器状态报告"""
        warnings = self.risk.check_position(self.account)
        lines = [
            "=" * 55,
            "实盘交易管理器 (V4.1 feature)",
            f"运行状态: {'运行中' if self._running else '已停止'}",
            "=" * 55,
            "",
            self.account.summary(),
            "",
            f"策略数: {len(self._strategies)}",
            f"待处理订单: {len(self.executor._pending)}",
        ]
        if warnings:
            lines.extend(["", "⚠️ 风控警告:"])
            for w in warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class PortfolioTracker:
    """投资组合追踪器"""

    def __init__(self):
        self._snapshots: list[dict] = []

    def snapshot(self, account: Account):
        """记录快照"""
        self._snapshots.append({
            "time": datetime.now().isoformat(),
            "total_asset": account.total_asset,
            "cash": account.cash,
            "market_value": account.market_value,
            "return_pct": account.return_pct,
            "positions": {s: pos.volume for s, pos in account.positions.items()},
        })

    def equity_curve(self) -> pd.Series:
        """净值曲线"""
        if not self._snapshots:
            return pd.Series(dtype=float)
        return pd.Series(
            [s["total_asset"] for s in self._snapshots],
            index=pd.to_datetime([s["time"] for s in self._snapshots]),
        )

    def performance(self) -> dict:
        """绩效统计"""
        if len(self._snapshots) < 2:
            return {"error": "快照不足"}
        total_asset = pd.Series([s["total_asset"] for s in self._snapshots])
        returns = total_asset.pct_change().dropna()
        return {
            "start_value": self._snapshots[0]["total_asset"],
            "end_value": self._snapshots[-1]["total_asset"],
            "total_return": (self._snapshots[-1]["total_asset"] / self._snapshots[0]["total_asset"] - 1) * 100,
            "sharpe": returns.mean() / max(returns.std(), 1e-12) * np.sqrt(252),
            "max_dd": (total_asset / total_asset.cummax() - 1).min() * 100,
            "n_snapshots": len(self._snapshots),
        }


__all__ = [
    "LiveOrder", "LivePosition", "Account",
    "MarketDataFeed", "RiskChecks",
    "ExecutionEngine", "LiveTradingManager",
    "PortfolioTracker",
]
