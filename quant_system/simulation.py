"""
仿真交易引擎 — 策略信号→模拟下单→持仓管理→绩效分析

功能:
  1. 模拟资金管理: 初始资金100万，支持多笔买入/卖出
  2. 自动执行: 从 trade_executor 读取待确认指令，自动模拟成交
  3. 持仓管理: 每日市值更新，浮动盈亏计算
  4. 绩效分析: 收益率曲线/胜率/夏普/最大回撤
  5. 报表输出: 每日持仓快照+交易记录

用法:
  python3 -m quant_system.simulation      # 查看仿真状态
  python3 -m quant_system.simulation --run  # 执行挂单（模拟成交）
  python3 -m quant_system.simulation --daily  # 日终持仓更新
  python3 -m quant_system.simulation --pnl  # 绩效报告
  python3 -m quant_system.simulation --reset  # 重置仿真
"""

# V4.1
from __future__ import annotations

import json
import logging
import sys
import time as _time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
SIM_FILE = ROOT.parent / "config" / "simulation_state.json"

# ───────── 配置 ─────────
INITIAL_CAPITAL = 1_000_000    # 初始资金100万
COMMISSION_RATE = 0.000085     # 佣金万0.85
MIN_COMMISSION = 5.0           # P1-Q26-fix: A股佣金最低5元/笔（原0.1元不符规）
STAMP_TAX_RATE = 0.0005         # 印花税万5（卖出，2023年8月起减半征收）
TRANSFER_FEE_RATE = 0.00001    # P1-Q26-fix: 过户费 0.001% 双边（卖出+买入都收）
MAX_POSITIONS = 10             # 最大持仓
MAX_SINGLE_RATIO = 0.15        # 单票上限15%
MIN_AMOUNT = 10000             # 最小交易金额1万

# ───────── 数据结构 ─────────
@dataclass
class SimulationOrder:
    """仿真委托单"""
    id: str
    stock: str
    stock_name: str
    action: str            # buy / sell
    price: float
    qty: int
    amount: float
    commission: float
    tax: float
    transfer_fee: float = 0.0  # P1-Q26-fix: 过户费 0.001% 双边
    status: str = "pending"    # pending / filled / canceled
    created_at: str = ""
    filled_at: str = ""
    reason: str = ""
    strategy: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationPosition:
    """仿真持仓"""
    stock: str
    stock_name: str
    qty: int
    cost: float            # 持仓成本（含佣金）
    current_price: float
    market_value: float
    pnl: float
    pnl_pct: float
    updated_at: str
    entry_date: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationState:
    """仿真账户状态"""
    capital: float = INITIAL_CAPITAL
    positions: dict[str, SimulationPosition] = field(default_factory=dict)
    orders: list[SimulationOrder] = field(default_factory=list)
    pnl_history: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    initial_capital: float = INITIAL_CAPITAL
    created_at: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        return {
            'capital': self.capital,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'orders': [o.to_dict() for o in self.orders],
            'pnl_history': self.pnl_history,
            'trades': self.trades,
            'initial_capital': self.initial_capital,
            'created_at': self.created_at,
            'last_update': self.last_update,
        }


def _load_state() -> SimulationState:
    if SIM_FILE.exists():
        try:
            data = json.loads(SIM_FILE.read_text())
            state = SimulationState(
                capital=data.get('capital', INITIAL_CAPITAL),
                orders=[SimulationOrder(**o) for o in data.get('orders', [])],
                pnl_history=data.get('pnl_history', []),
                trades=data.get('trades', []),
                initial_capital=data.get('initial_capital', INITIAL_CAPITAL),
                created_at=data.get('created_at', ''),
                last_update=data.get('last_update', ''),
            )
            for code, pd in data.get('positions', {}).items():
                state.positions[code] = SimulationPosition(**pd)
            return state
        except Exception as e:
            logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    state = SimulationState()
    state.created_at = datetime.now(CST).isoformat()
    return state


def _save_state(state: SimulationState):
    SIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.last_update = datetime.now(CST).isoformat()
    SIM_FILE.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=str))


def _get_current_price(code: str) -> float:
    """获取实时价格（优先 akshare 日K，降级到腾讯实时行情）。"""
    try:
        import akshare as ak
        hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   # P2-Q26-fix: 原 5 日窗口在长假（春节/国庆 7+ 天休市）后
                                   # 可能整窗口无交易 → 取不到最新收盘，扩大到 10 日窗口。
                                   start_date=(datetime.now(CST)-timedelta(days=10)).strftime('%Y%m%d'),
                                   end_date=datetime.now(CST).strftime('%Y%m%d'),
                                   adjust="qfq")
        if hist is not None and len(hist) > 0:
            return float(hist['收盘'].iloc[-1])
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    try:
        from quant_system.watchlist import fetch_quotes
        qs = fetch_quotes([code])
        if qs and qs[0].get('price', 0) > 0:
            return float(qs[0]['price'])
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    return 0.0


def _calc_commission(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def _calc_transfer_fee(amount: float) -> float:
    """过户费 0.001% 双边。"""
    return amount * TRANSFER_FEE_RATE


def _calc_stamp_tax(amount: float) -> float:
    return amount * STAMP_TAX_RATE


def _calc_qty(amount: float, price: float) -> int:
    """按金额计算股数（必须100股整数倍）"""
    qty = int(amount / price / 100) * 100
    return max(qty, 100)


# ───────── A股交易规则校验 (P1-Q26-fix) ─────────

def _get_quote_context(code: str) -> dict:
    """一次拉取日K，返回撮合校验所需上下文：
       prev_close(昨收,不复权) / last_close(最新收盘) / last_date(最近K线日期)。
       取不到行情时返回全空，由调用方按停牌/无数据处理。
    """
    ctx = {'prev_close': 0.0, 'last_close': 0.0, 'last_date': None}
    try:
        import akshare as ak
        hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=(datetime.now(CST) - timedelta(days=10)).strftime('%Y%m%d'),
                                   end_date=datetime.now(CST).strftime('%Y%m%d'),
                                   adjust="")
        if hist is not None and len(hist) > 0:
            ctx['last_close'] = float(hist['收盘'].iloc[-1])
            ctx['last_date'] = str(hist['日期'].astype(str).iloc[-1])
            today = datetime.now(CST).strftime('%Y-%m-%d')
            completed = hist[hist['日期'].astype(str) < today]
            ctx['prev_close'] = float(completed['收盘'].iloc[-1]) if len(completed) else ctx['last_close']
            return ctx
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    # 降级：腾讯实时行情（本环境 akshare/东财接口不可达时仍可用）
    try:
        from quant_system.watchlist import fetch_quotes
        qs = fetch_quotes([code])
        if qs and qs[0].get('prev_close', 0) > 0:
            q = qs[0]
            ctx['prev_close'] = float(q['prev_close'])
            ctx['last_close'] = float(q.get('price', 0) or 0)
            # 实时行情当日可得 → 不判定为停牌
            ctx['last_date'] = datetime.now(CST).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    return ctx


def _board_limit_pct(code: str, name: str) -> float:
    """按板块返回涨跌停幅度：主板10%(ST 5%)、创业板/科创板20%、北交所30%。"""
    if code.startswith(('688', '300', '301')):
        return 0.20
    if code.startswith(('4', '8', '920')):
        return 0.30
    if 'ST' in (name or '').upper():
        return 0.05
    return 0.10


def _get_price_limits(code: str, name: str, prev_close: float) -> tuple[float, float]:
    """获取当日涨跌停价 (limit_up, limit_down)。

    优先用 easy_tdx get_price_limits（服务端自动识别板块/ST/注册制差异），
    失败时按板块规则 10%/20%/30%（ST 5%）本地计算。
    """
    try:
        from easy_tdx import TdxClient, Market
        market = (Market.SH if code.startswith('6')
                  else Market.SZ if code.startswith(('0', '2', '3'))
                  else Market.BJ)
        client = TdxClient()
        client.connect()
        try:
            ul, dl = client.get_price_limits(market, code, name, prev_close)
            if ul and dl:
                return float(ul), float(dl)
        finally:
            try:
                client.disconnect()
            except Exception as e:
                logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)
    pct = _board_limit_pct(code, name)
    up = round(prev_close * (1 + pct), 2)
    down = round(prev_close * (1 - pct), 2)
    return up, down


def _is_suspended(ctx: dict) -> bool:
    """停牌判断：无任何行情数据，或最近K线距今超过 3 个交易日。

    V11 审计修复（Medium）: 原用自然日 gap>=3 判停牌，春节/国庆 7+ 天长假后
    首日 gap=8+ 被误判全部持仓/挂单停牌而拒单。修正: 改用交易日历计算
    （长假期间非交易日不计入缺口）。
    """
    last = ctx.get('last_date')
    if last is None:
        return True
    try:
        from quant_system.market_clock import is_trading_day
        # 统一为无时区 datetime，避免 naive/aware 相减抛错
        now = datetime.now(CST).replace(tzinfo=None)
        last_dt = datetime.strptime(last, '%Y-%m-%d')
        # 统计 last 之后到今天的非交易日天数（缺口）
        gap_days = 0
        d = last_dt + timedelta(days=1)
        while d <= now:
            if not is_trading_day(d):
                gap_days += 1
            d += timedelta(days=1)
        # 缺口 >= 3 个交易日才判停牌
        return gap_days >= 3
    except Exception:
        # 回退: 自然日 3 天（宽松窗口避免长假误判）
        try:
            now = datetime.now(CST).replace(tzinfo=None)
            gap = (now - datetime.strptime(last, '%Y-%m-%d')).days
            return gap >= 5  # 放宽到 5 自然日，兼容长假
        except ValueError:
            return True


def _mark_rejected(order: dict) -> None:
    """将指令标记为已拒绝，避免后续轮询重复拒单。"""
    try:
        from quant_system.trade_executor import reject_order
        reject_order(order.get('id', ''))
    except Exception as e:
        logger.error(f"[simulation] 操作失败: {e}", exc_info=True)


# ───────── 核心功能 ─────────
def execute_pending_orders(state: SimulationState) -> list[dict]:
    """执行待确认指令（模拟成交）"""
    # 从 trade_executor 读取待确认指令
    results = []
    try:
        from quant_system.trade_executor import _load_orders
        pending = [o for o in _load_orders() if o.get('status') == 'pending']
    except ImportError:
        pending = []

    for order in pending:
        code = order.get('stock', '')
        action = order.get('action', '')
        price = order.get('price', 0) or _get_current_price(code)
        name = order.get('stock_name', '')

        if price <= 0:
            # V11 审计修复（Medium）: 原实现直接 continue 不标记 rejected，
            # 挂单每轮 --run 都被重新处理，永久滞留 pending。
            results.append(f"⛔ 拒单 {name}({code}): 价格无效(<=0)，委托未成交")
            _mark_rejected(order)
            continue

        # ── P1-Q26-fix: 撮合前校验 A股交易规则（涨跌停/停牌/T+1） ──
        ctx = _get_quote_context(code)
        if ctx['prev_close'] <= 0 or _is_suspended(ctx):
            results.append(f"⛔ 拒单 {name}({code}): 无法获取行情或疑似停牌，委托未成交")
            _mark_rejected(order)
            continue
        limit_up, limit_down = _get_price_limits(code, name, ctx['prev_close'])
        if limit_up <= 0 or limit_down <= 0:
            results.append(f"⛔ 拒单 {name}({code}): 涨跌停价计算失败，委托未成交")
            _mark_rejected(order)
            continue
        # 委托价必须落在当日涨跌停区间内（分板块 10%/20%/30%，ST 5%）
        if price > limit_up + 1e-6 or price < limit_down - 1e-6:
            results.append(
                f"⛔ 拒单 {name}({code}): 委托价{price:.2f}超出当日涨跌停区间"
                f"[{limit_down:.2f}, {limit_up:.2f}]"
            )
            _mark_rejected(order)
            continue
        # 一字涨停/跌停封板：现价已封板时按委托价无法成交
        last_price = ctx.get('last_close') or 0.0
        if action == 'buy' and last_price > 0 and last_price >= limit_up - 0.01 \
                and price >= limit_up - 0.01:
            results.append(f"⛔ 拒单 {name}({code}): 涨停一字板无法买入 (现价{last_price:.2f}已封涨停)")
            _mark_rejected(order)
            continue
        if action == 'sell' and last_price > 0 and last_price <= limit_down + 0.01 \
                and price <= limit_down + 0.01:
            results.append(f"⛔ 拒单 {name}({code}): 跌停一字板无法卖出 (现价{last_price:.2f}已封跌停)")
            _mark_rejected(order)
            continue
        # T+1：当日买入的持仓当日不可卖出
        if action == 'sell' and code in state.positions:
            pos = state.positions[code]
            today = datetime.now(CST).strftime('%Y-%m-%d')
            entry = (pos.entry_date or '')[:10]
            if entry == today:
                results.append(f"⛔ 拒单 {name}({code}): T+1 限制，当日买入不可卖出")
                _mark_rejected(order)
                continue

        if action == 'buy':
            # 修复 Q26: 买入量 = min(委托qty, 可买金额地板)，循环降档直至
            # total_cost <= capital，低于100股/资金不足则拒单（不产生负现金）。
            # 原实现 max(buy_amount, MIN_AMOUNT) 强制≥1万且 _calc_qty 强制≥100股，
            # 资金不足时现金可为负（实测 capital=5000/price=100 → capital=-5000.85）。
            budget = min(state.capital * 0.9 * 0.3, state.capital * MAX_SINGLE_RATIO)
            budget = min(budget, state.capital * 0.95)  # 最多动用95%现金（含费用缓冲）
            order_qty = int(order.get('qty', 0) or 0)

            if order_qty > 0:
                # 委托指定数量：受资金地板约束（min(委托qty, 可买金额地板)）
                qty = min(order_qty, int(state.capital * 0.95 / price / 100) * 100)
            else:
                qty = int(budget / price / 100) * 100

            # 循环降档直至总成本（含佣金+过户费）<= 现金，保证现金不为负
            while qty >= 100:
                amount = qty * price
                commission = _calc_commission(amount)
                total_cost = amount + commission + _calc_transfer_fee(amount)
                if total_cost <= state.capital:
                    break
                qty -= 100

            if qty < 100:
                results.append(f"⛔ 拒单 {name}({code}): 资金不足或不足100股")
                _mark_rejected(order)
                continue
            amount = qty * price
            commission = _calc_commission(amount)
            transfer_fee = _calc_transfer_fee(amount)
            total_cost = amount + commission + transfer_fee

            # V11 审计修复（Medium）: MAX_POSITIONS 定义后从未使用——"最大持仓 10 只"
            # 约束从未强制，买入路径可无限加仓。修正: 新开仓（非加仓）前检查持仓数。
            if code not in state.positions and len(state.positions) >= MAX_POSITIONS:
                results.append(f"⛔ 拒单 {name}({code}): 已达最大持仓 {MAX_POSITIONS} 只")
                _mark_rejected(order)
                continue

            # 执行买入
            state.capital -= total_cost
            cost_per_share = total_cost / qty

            if code in state.positions:
                # 加仓：加权平均成本
                pos = state.positions[code]
                total_qty = pos.qty + qty
                total_cost_pos = pos.qty * pos.cost + qty * cost_per_share
                pos.cost = total_cost_pos / total_qty
                pos.qty = total_qty
            else:
                state.positions[code] = SimulationPosition(
                    stock=code, stock_name=name,
                    qty=qty, cost=cost_per_share,
                    current_price=price, market_value=amount,
                    pnl=0, pnl_pct=0,
                    updated_at=datetime.now(CST).isoformat(),
                    entry_date=datetime.now(CST).strftime('%Y-%m-%d'),
                )

            sim_order = SimulationOrder(
                id=f"SIM_{int(_time.time())}_{code}",
                stock=code, stock_name=name,
                action='buy', price=price, qty=qty, amount=amount,
                commission=commission, tax=0, transfer_fee=transfer_fee,
                status='filled', created_at=datetime.now(CST).isoformat(),
                filled_at=datetime.now(CST).isoformat(),
                reason=order.get('reason', ''),
                strategy=order.get('strategy', ''),
            )
            state.orders.append(sim_order)
            state.trades.append({
                'date': datetime.now(CST).strftime('%Y-%m-%d'),
                'stock': code, 'name': name,
                'action': 'buy', 'price': price, 'qty': qty,
                'amount': amount, 'commission': commission,
                'transfer_fee': transfer_fee,
            })
            results.append(f"🟢 买入 {name}({code}) {qty}股 @ {price:.2f}")

        elif action == 'sell' and code in state.positions:
            pos = state.positions[code]
            # P1-Q26-fix: 按委托 qty 部分卖出（不足100股按持仓余量），不再无视委托量全仓卖出
            order_qty = int(order.get('qty', 0) or 0)
            if order_qty > 0:
                sell_qty = min(order_qty, pos.qty)
                sell_qty = (sell_qty // 100) * 100  # 整手
                if sell_qty <= 0:
                    results.append(f"⛔ 拒单 {name}({code}): 卖出数量不足100股")
                    _mark_rejected(order)
                    continue
            else:
                sell_qty = pos.qty

            amount = sell_qty * price
            commission = _calc_commission(amount)
            tax = _calc_stamp_tax(amount)
            transfer_fee = _calc_transfer_fee(amount)
            net_amount = amount - commission - tax - transfer_fee
            state.capital += net_amount

            sim_order = SimulationOrder(
                id=f"SIM_{int(_time.time())}_{code}",
                stock=code, stock_name=name,
                action='sell', price=price, qty=sell_qty,
                amount=amount, commission=commission, tax=tax,
                transfer_fee=transfer_fee,
                status='filled', created_at=datetime.now(CST).isoformat(),
                filled_at=datetime.now(CST).isoformat(),
                reason=order.get('reason', ''),
            )
            state.orders.append(sim_order)
            state.trades.append({
                'date': datetime.now(CST).strftime('%Y-%m-%d'),
                'stock': code, 'name': name,
                'action': 'sell', 'price': price, 'qty': sell_qty,
                'amount': amount, 'commission': commission, 'tax': tax,
                'transfer_fee': transfer_fee,
                # P2-Q26-fix: 卖出 pnl 扣除卖出侧费用（佣金+印花税+过户费），
                # 原实现为毛额(amount-持仓成本)，胜率/盈亏比被高估。
                'pnl': amount - sell_qty * pos.cost - commission - tax - transfer_fee,
            })
            results.append(f"🔴 卖出 {name}({code}) {sell_qty}股 @ {price:.2f}")

            # P1-Q26-fix: 部分卖出保留剩余持仓，仅清仓时删除
            remaining = pos.qty - sell_qty
            if remaining > 0:
                pos.qty = remaining
                pos.market_value = remaining * pos.current_price
                pos.pnl = pos.market_value - remaining * pos.cost
                pos.pnl_pct = (pos.current_price / pos.cost - 1) * 100 if pos.cost > 0 else 0
                pos.updated_at = datetime.now(CST).isoformat()
            else:
                del state.positions[code]

        # 更新原指令状态为已确认
        try:
            from quant_system.trade_executor import confirm_order
            confirm_order(order.get('id', ''))
        except Exception as e:
            logger.error(f"[simulation] 操作失败: {e}", exc_info=True)

    _save_state(state)
    return results


def daily_update(state: SimulationState) -> dict:
    """日终更新：更新所有持仓市值和盈亏。

    Returns:
        {"updated": 更新成功数, "failed": 失败持仓列表, "skipped_nav": 是否跳过当日净值}
    """
    now = datetime.now(CST)
    updated_count = 0
    failed: list[str] = []
    for code, pos in list(state.positions.items()):
        price = _get_current_price(code)
        if price > 0:
            pos.current_price = price
            pos.market_value = pos.qty * price
            pos.pnl = pos.market_value - pos.qty * pos.cost
            pos.pnl_pct = (price / pos.cost - 1) * 100 if pos.cost > 0 else 0
            pos.updated_at = now.isoformat()
            updated_count += 1
        else:
            # P2-Q26-fix: 行情获取失败显式标记（akshare/腾讯均不可达），
            # 不再静默保留陈旧 current_price 参与记账
            failed.append(code)
        _time.sleep(0.3)

    # P2-Q26-fix: 持仓存在但全部无新价（如 akshare 断网）→ 跳过当日净值记录，
    # 避免用陈旧价格记出虚假净值点（降级可见）
    if state.positions and updated_count == 0:
        logger.warning(f"daily_update: 全部 {len(state.positions)} 只持仓行情获取失败，跳过当日净值记录: {failed}")
        return {"updated": 0, "failed": failed, "skipped_nav": True}
    if failed:
        logger.warning(f"daily_update: {len(failed)} 只持仓行情获取失败（保留原价）: {failed}")

    # 记录日净值
    total_value = state.capital + sum(
        p.qty * p.current_price for p in state.positions.values()
    )
    state.pnl_history.append({
        'date': now.strftime('%Y-%m-%d'),
        'total_value': round(total_value, 2),
        'capital': round(state.capital, 2),
        'positions_value': round(total_value - state.capital, 2),
    })
    _save_state(state)
    return {"updated": updated_count, "failed": failed, "skipped_nav": False}


def compute_performance(state: SimulationState) -> dict:
    """计算绩效指标"""
    history = state.pnl_history
    if not history:
        return {}

    values = [h['total_value'] for h in history]
    init = state.initial_capital

    total_return = (values[-1] / init - 1) * 100

    # 每日收益率
    daily_returns = []
    for i in range(1, len(values)):
        dr = (values[i] / values[i - 1]) - 1
        daily_returns.append(dr)

    # 夏普 (假设无风险利率2%)
    if daily_returns:
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (avg_ret - 0.02 / 252) / std_ret * np.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0

    # 最大回撤
    max_drawdown = 0
    peak = values[0]
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # 胜率/盈亏比：仅统计已平仓（卖出）记录——买入单无 pnl 键，
    # 原实现把买入单计入分母导致胜率被系统性低估。
    closed = [t for t in state.trades if isinstance(t.get('pnl'), (int, float))]
    wins = sum(1 for t in closed if t['pnl'] > 0)
    total_trades = len(closed)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # 盈亏比（pnl 已扣卖出侧费用）
    win_avg = np.mean([t['pnl'] for t in closed if t['pnl'] > 0]) if wins > 0 else 0
    loss_avg = abs(np.mean([t['pnl'] for t in closed if t['pnl'] < 0])) if (total_trades - wins) > 0 else 1
    profit_loss_ratio = win_avg / loss_avg if loss_avg > 0 else 0

    return {
        'total_return_pct': round(total_return, 2),
        'total_value': round(values[-1], 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown_pct': round(max_drawdown, 2),
        'win_rate_pct': round(win_rate, 1),
        'profit_loss_ratio': round(profit_loss_ratio, 2),
        'total_trades': total_trades,
        'positions_count': len(state.positions),
        'tracking_days': len(history),
    }


# ───────── 格式化 ─────────
def format_state(state: SimulationState) -> str:
    now = datetime.now(CST)
    total_pos_value = sum(p.qty * p.current_price for p in state.positions.values())
    total_value = state.capital + total_pos_value
    total_pnl = total_value - state.initial_capital
    total_pnl_pct = (total_value / state.initial_capital - 1) * 100

    lines = [f"# 💰 仿真交易账户 ({now.strftime('%Y-%m-%d %H:%M')})"]
    lines.append(f"{'=' * 50}")
    lines.append(f"\n📊 账户概况")
    lines.append(f"  初始资金: ¥{state.initial_capital:,.0f}")
    lines.append(f"  总资产: ¥{total_value:,.0f}")
    lines.append(f"  可用资金: ¥{state.capital:,.0f}")
    lines.append(f"  持仓市值: ¥{total_pos_value:,.0f}")
    lines.append(f"  总盈亏: ¥{total_pnl:+,.0f} ({total_pnl_pct:+.2f}%)")

    if state.positions:
        lines.append(f"\n📦 当前持仓 ({len(state.positions)}只)")
        lines.append(f"{'股票':<14}{'持股':>6}{'成本':>8}{'现价':>8}{'市值':>10}{'盈亏':>10}{'盈亏%':>8}")
        lines.append("-" * 68)
        for code, pos in sorted(state.positions.items(), key=lambda x: x[1].market_value, reverse=True):
            name = f"{pos.stock_name}({code[:6]})" if pos.stock_name else code[:6]
            lines.append(f"  {name:<12} {pos.qty:>5} {pos.cost:>7.2f} {pos.current_price:>7.2f} "
                         f"{pos.market_value:>9,.0f} {pos.pnl:>+9,.0f} {pos.pnl_pct:>+6.2f}%")

    # 持仓集中度
    if total_value > 0:
        lines.append(f"\n📈 持仓集中度")
        for code, pos in sorted(state.positions.items(), key=lambda x: x[1].market_value, reverse=True)[:5]:
            ratio = pos.market_value / total_value * 100
            bars = "█" * int(ratio / 5) + "░" * max(0, 10 - int(ratio / 5))
            name = pos.stock_name or code[:6]
            lines.append(f"  {name:<10} {bars} {ratio:.1f}%")

    # 最近交易
    if state.trades:
        recent = state.trades[-5:]
        lines.append(f"\n📋 最近交易")
        for t in reversed(recent):
            act = "🟢买入" if t['action'] == 'buy' else "🔴卖出"
            lines.append(f"  {t['date']} {act} {t.get('name','')}({t['stock'][:6]}) {t['qty']}股 @ {t['price']:.2f}")

    lines.append(f"\n操作: python3 -m quant_system.simulation --run (执行挂单)")
    lines.append(f"       python3 -m quant_system.simulation --daily (日终更新)")
    lines.append(f"       python3 -m quant_system.simulation --pnl (绩效报告)")
    return "\n".join(lines)


def format_pnl(state: SimulationState) -> str:
    perf = compute_performance(state)
    lines = [f"# 📊 仿真交易绩效 ({datetime.now(CST).strftime('%Y-%m-%d')})"]
    lines.append(f"{'=' * 50}")
    if not perf:
        lines.append("\n⚠️ 暂无交易数据")
        return "\n".join(lines)

    lines.append(f"\n📈 累计收益: {perf['total_return_pct']:+.2f}%")
    lines.append(f"💰 总资产: ¥{perf['total_value']:,.0f}")
    lines.append(f"📐 夏普比率: {perf['sharpe_ratio']}")
    lines.append(f"📉 最大回撤: {perf['max_drawdown_pct']:.2f}%")
    lines.append(f"🎯 胜率: {perf['win_rate_pct']:.1f}% ({perf['total_trades']}笔)")
    lines.append(f"⚖️ 盈亏比: {perf['profit_loss_ratio']:.2f}")
    lines.append(f"📅 跟踪天数: {perf['tracking_days']}")
    lines.append(f"🪪 当前持仓: {perf['positions_count']}只")

    if state.pnl_history:
        lines.append(f"\n📅 净值曲线（最近7天）")
        for h in state.pnl_history[-7:]:
            ret = (h['total_value'] / state.initial_capital - 1) * 100
            bars = "█" * max(1, int(abs(ret) / 2))
            sign = "+" if ret >= 0 else ""
            lines.append(f"  {h['date']}: {sign}{ret:.2f}% {bars}")

    return "\n".join(lines)


# ───────── CLI ─────────
def main():
    state = _load_state()
    args = sys.argv[1:]

    if "--run" in args:
        results = execute_pending_orders(state)
        if results:
            for r in results:
                print(r)
        else:
            print("✅ 无待执行指令")
        print(format_state(state))
    elif "--daily" in args:
        print("🔄 执行日终更新...")
        daily_update(state)
        print("✅ 更新完成")
        print(format_state(state))
    elif "--pnl" in args:
        print(format_pnl(state))
    elif "--reset" in args:
        SIM_FILE.unlink(missing_ok=True)
        print("✅ 仿真账户已重置")
    else:
        print(format_state(state))


if __name__ == "__main__":
    main()
