"""
交易执行链路 — 策略引擎→信号→交易指令→人工确认

功能:
  1. 信号评估: 从strategy_engine获取推荐，结合持仓做终审
  2. 指令生成: 生成买入/卖出/减仓交易指令
  3. 人工确认: 交易指令生成飞书消息，等待确认后执行
  4. 执行记录: 记录指令状态(待确认/已执行/已拒绝)
  5. 循环引擎: 定时扫描+评估+生成指令

用法:
  python3 -m quant_system.trade_executor              # 全链路扫描
  python3 -m quant_system.trade_executor --confirm ID  # 确认某条指令
  python3 -m quant_system.trade_executor --reject ID   # 拒绝
  python3 -m quant_system.trade_executor --daemon      # 循环模式
"""

from __future__ import annotations
import logging

import json
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_system.context import RunContext, coerce_run_context
from quant_system.market_clock import latest_completed_trading_day, prev_trading_day


ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
CONFIG_DIR = ROOT.parent / "config"
ORDERS_FILE = CONFIG_DIR / "trade_orders.json"
MARKET_FUSION_FILE = ROOT.parent / "data_warehouse" / "market" / "fusion.parquet"
MAX_STALE_TRADING_DAYS = 3

# 仓位限制
MAX_POSITIONS = 10             # 最大持仓数
MAX_SINGLE_WEIGHT = 0.15       # 单票上限
CASH_RESERVE = 0.2             # 现金保留


def _orders_lock_path() -> Path:
    return CONFIG_DIR / "trade_orders.lock"


def _read_orders_unlocked() -> list[dict]:
    if ORDERS_FILE.exists():
        try:
            return json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logging.getLogger(__name__).error(f"[trade_executor] 操作失败: {e}", exc_info=True)
    return []


def _write_orders_unlocked(orders: list[dict]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ORDERS_FILE.with_suffix(ORDERS_FILE.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(orders, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, ORDERS_FILE)


def _load_orders() -> list[dict]:
    """加载指令记录（共享锁，避免读到半写文件）。"""
    import fcntl
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with _orders_lock_path().open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            return _read_orders_unlocked()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save_orders(orders: list[dict]):
    """原子保存指令记录（独占锁 + temp rename）。"""
    import fcntl
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with _orders_lock_path().open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _write_orders_unlocked(orders)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _mutate_orders(mutator):
    """在同一把锁内完成读-改-写，防止 confirm/reject/set-qty 覆盖并发更新。"""
    import fcntl
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with _orders_lock_path().open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            orders = _read_orders_unlocked()
            result = mutator(orders)
            _write_orders_unlocked(orders)
            return result
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _merge_new_orders(new_orders: list[dict]) -> list[dict]:
    """Merge generated orders under one exclusive read-modify-write lock."""
    def _merge(current: list[dict]) -> list[dict]:
        existing_ids = {o.get("id") for o in current}
        pending_keys = {
            (o.get("stock"), o.get("strategy"), o.get("action"))
            for o in current if o.get("status") == "pending"
        }
        for order in new_orders:
            key = (order.get("stock"), order.get("strategy"), order.get("action"))
            if order.get("id") not in existing_ids and key not in pending_keys:
                current.append(order)
                existing_ids.add(order.get("id"))
                pending_keys.add(key)
        return [dict(o) for o in current if o.get("status") == "pending"]

    return _mutate_orders(_merge)


def ingest_dispatched_orders(orders: list[dict] | None = None,
                             date: str | None = None,
                             orders_path: str | None = None) -> tuple[list[dict], int]:
    """把盘前 order_dispatcher 订单（generated/orders_{date}.json）灌入执行队列。

    2026-08-23 C5: 修复盘前链(battle_map→order_dispatcher)与执行链(trade_executor)
    断链——dispatcher 产出的订单此前从不被 executor 消费。此桥只追加 `pending`
    状态、按 (code, strategy) 去重、**绝不自动下单**（仍走 confirm_order 人工确认）。

    返回 (orders, n_added)。orders 缺省时自行加载；orders_path 供测试注入。
    """
    date = date or datetime.now(CST).date().isoformat()
    orders = _load_orders() if orders is None else orders
    src = Path(orders_path) if orders_path else (ROOT.parent / "generated" / f"orders_{date}.json")
    if not src.exists():
        return orders, 0
    try:
        dispatched = json.loads(src.read_text(encoding="utf-8")).get("orders", [])
    except Exception as e:
        logging.getLogger(__name__).warning(f"[trade_executor] 盘前订单读取失败: {e}")
        return orders, 0
    if not isinstance(dispatched, list):
        return orders, 0

    existing = {(o.get("stock"), o.get("strategy"))
                for o in orders if o.get("status") == "pending"}
    n = 0
    for d in dispatched:
        if not isinstance(d, dict):
            continue
        code = str(d.get("code") or "").strip().zfill(6)
        if not code or code == "000000":
            continue
        key = (code, f"battle_map:{d.get('strategy', '')}")
        if key in existing:
            continue
        orders.append({
            "id": f"ORD_{_time.time_ns()}_{code}",
            "stock": code,
            "stock_name": str(d.get("name", "")),
            "action": "buy",
            "price": 0.0,
            "qty": 0,
            "qty_needs_manual": True,
            "stop_loss": None,
            "reason": f"盘前作战地图 {str(d.get('signal_source', ''))[:50]}",
            "strategy": f"battle_map:{d.get('strategy', '')}",
            "status": "pending",
            "created_at": datetime.now(CST).isoformat(),
            "confirmed_at": None,
        })
        existing.add(key)
        n += 1
    if n:
        _save_orders(orders)
    return orders, n


def _market_temperature(as_of: str | datetime | None = None) -> int | None:
    """读取截至最近已完成交易日且足够新鲜的融合温度。"""
    try:
        import pandas as pd
        cutoff = latest_completed_trading_day(as_of)
        oldest = cutoff
        for _ in range(MAX_STALE_TRADING_DAYS):
            oldest = prev_trading_day(oldest) or oldest
        if not MARKET_FUSION_FILE.exists():
            return None
        frame = pd.read_parquet(MARKET_FUSION_FILE)
        date_col = next((c for c in ("date", "日期", "trade_date") if c in frame.columns), None)
        if date_col is None or "temperature" not in frame.columns:
            return None
        dates = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None)
        frame = frame.assign(_date=dates).dropna(subset=["_date", "temperature"])
        frame = frame[frame["_date"].dt.date <= cutoff].sort_values("_date")
        if frame.empty or frame["_date"].iloc[-1].date() < oldest:
            return None
        value = float(frame["temperature"].iloc[-1])
        if not 0 <= value <= 100:
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _max_new_buys(temp: int | None) -> int:
    """关键市场门禁缺失时 fail-closed，不生成新增买单。"""
    if temp is None:
        return 0
    if temp < 20:
        return 0      # 冰点：不新增买入
    if temp < 40:
        return 2      # 寒冷：防守，只上最强 2 个
    if temp < 60:
        return 4      # 温和：中性
    return 8          # 温暖/过热：放开


def _fetch_quote_price(code: str) -> float:
    """获取最近收盘价：本地数据仓库优先（免网络、快），缺失才回退 akshare。

    P2-Q19-fix(M179/L185): 行情获取失败时返回 0，由调用方决定
    拒绝指令或回退，避免生成 price=0 的不可执行指令。
    2026-08-25 加速: 本机境外 IP 访问东财 push2his 被代理阻断，逐股 akshare
    反复重试是模拟盘扫描卡死的根因；改为先读 data_warehouse/kline 本地快照。
    """
    try:
        from pathlib import Path
        _p = Path(__file__).resolve().parent.parent / "data_warehouse" / "kline" / f"{code}.parquet"
        if _p.exists():
            import pandas as pd
            _df = pd.read_parquet(_p)
            if len(_df) and "close" in _df.columns:
                return float(_df["close"].iloc[-1])
    except Exception:
        pass
    try:
        import akshare as ak
        hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                   start_date=(datetime.now(CST) - timedelta(days=5)).strftime('%Y%m%d'),
                                   end_date=datetime.now(CST).strftime('%Y%m%d'),
                                   adjust="qfq")
        if hist is not None and len(hist) > 0:
            return float(hist['收盘'].iloc[-1])
    except Exception as e:
        logging.getLogger(__name__).error(f"[trade_executor] 操作失败: {e}", exc_info=True)
    return 0.0


def evaluate_signal(signal: dict, position: dict = None) -> dict:
    """终审一条信号，决定是否生成指令

    Args:
        signal: {'stock':, 'direction':, 'confidence':, 'strategy':, 'reason':}
        position: 当前持仓 {'qty':, 'cost':, 'pnl_pct':}

    Returns:
        dict: {'action':, 'price':, 'qty':, 'reason':, 'decision': 'execute'|'hold'|'reject'}
    """
    code = signal.get('stock', '')
    direction = signal.get('direction', 0)
    confidence = signal.get('confidence', 0)
    has_position = position is not None

    # 决策逻辑
    if direction == 1:  # 买入信号
        if has_position:
            return {'action': 'hold', 'decision': 'hold', 'reason': '已持仓'}
        if confidence < 0.4:
            return {'action': 'hold', 'decision': 'reject', 'reason': f'置信度不足({confidence:.0%})'}
        # 估算买入量
        price = _fetch_quote_price(code)
        # P2-Q19-fix(M179): 行情获取失败时拒绝生成买入指令，
        # 避免 price=0 的不可执行指令流入确认队列
        if price <= 0:
            return {
                'action': 'buy', 'decision': 'reject',
                'reason': f'行情获取失败，无法生成买入指令({code})',
            }
        return {
            'action': 'buy',
            'decision': 'execute',
            'price': price,
            'qty': 0,  # 需要用户确认实际资金
            'qty_needs_manual': True,  # P2-Q19-fix(M179): 标记数量待人工补全
            'stop_loss': round(price * 0.92, 2),
            'reason': signal.get('reason', ''),
            'strategy': signal.get('strategy', ''),
        }

    elif direction == -1:  # 卖出信号
        if not has_position:
            return {'action': 'hold', 'decision': 'reject', 'reason': '无持仓'}
        # P2-Q19-fix(L185): trade_db.current_price 可为 NULL → 空价回退行情接口
        price = position.get('current_price', 0) or 0
        if price <= 0:
            price = _fetch_quote_price(code)
        qty = position.get('qty', position.get('shares', 0)) or 0
        if price <= 0 or qty <= 0:
            return {
                'action': 'sell', 'decision': 'reject',
                'reason': f'行情或持仓数量无效，无法生成卖出指令({code})',
            }
        return {
            'action': 'sell',
            'decision': 'execute',
            'price': price,
            'qty': qty,
            'reason': signal.get('reason', ''),
            'strategy': signal.get('strategy', ''),
        }

    return {'action': 'hold', 'decision': 'hold', 'reason': '无信号'}


def scan_and_execute(*, as_of: str | datetime | None = None, context: RunContext | None = None) -> list[dict]:
    """全链路扫描: 策略引擎→信号评估→生成指令"""
    orders = _load_orders()
    run_context = coerce_run_context(context, as_of=as_of)
    cutoff = run_context.iso_date

    # 0. 2026-08-23 C5: 盘前链订单(battle_map→order_dispatcher)灌入执行队列
    #    只追加 pending、不自动下单；失败不阻断主扫描。
    try:
        orders, _n_ingested = ingest_dispatched_orders(orders)
        if _n_ingested:
            logging.getLogger(__name__).info(
                f"[trade_executor] 已灌入盘前作战地图订单 {_n_ingested} 条")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(f"[trade_executor] 盘前订单灌入失败: {e}")

    # 1. 获取策略推荐
    try:
        from quant_system.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        result = engine.run(as_of=cutoff, context=run_context)
        signals = result.get('allocations', [])
        meta = result.get('meta', {})
    except ImportError:
        return []

    # 2. 获取当前持仓
    positions_dict = {}
    try:
        from quant_system.trade_db import TradeDB
        db = TradeDB()
        for pos in db.get_positions():
            symbol = pos.get('stock', pos.get('symbol', ''))
            if symbol:
                normalized = dict(pos)
                normalized.setdefault('stock', symbol)
                normalized.setdefault('qty', pos.get('shares', 0))
                positions_dict[str(symbol)] = normalized
    except ImportError:
        pass

    # 3. 评估信号
    new_orders = []
    for alloc in signals:
        code = getattr(alloc, 'stock', None)
        if not code:
            continue
        direction = 1
        # P2-Q19-fix: StrategyAllocation 是 dataclass 无 .get()，旧写法 getattr(alloc, x, alloc.get(x))
        # 会先求值 alloc.get(x) 抛 AttributeError，导致模拟盘扫描一直断链。
        for sig in (getattr(alloc, 'signals', None) or []):
            if hasattr(sig, 'direction'):
                direction = sig.direction
            elif isinstance(sig, dict):
                direction = sig.get('direction', 1)

        signal = {
            'stock': code,
            'stock_name': getattr(alloc, 'stock_name', ''),
            'direction': direction,
            'confidence': getattr(alloc, 'confidence', 0.5),
            'reason': f"组合引擎推荐",
            'strategy': 'strategy_engine',
        }
        position = positions_dict.get(code)
        evaluation = evaluate_signal(signal, position)

        if evaluation['decision'] == 'execute':
            order = {
                # P2-Q19-fix(M180): 纳秒级时间戳——原秒级精度同秒同股生成重复 ID 被去重丢弃
                'id': f"ORD_{_time.time_ns()}_{code}",
                'stock': code,
                'stock_name': signal.get('stock_name', ''),
                'action': evaluation['action'],
                'price': evaluation.get('price', 0),
                'qty': evaluation.get('qty', 0),
                'qty_needs_manual': evaluation.get('qty_needs_manual', False),  # P2-Q19-fix(M179)
                'stop_loss': evaluation.get('stop_loss'),
                'reason': evaluation['reason'],
                'strategy': evaluation.get('strategy', ''),
                'status': 'pending',
                'created_at': datetime.now(CST).isoformat(),
                'confirmed_at': None,
            }
            new_orders.append(order)

    # 3.5 市场状态择时：冰点空仓/寒冷防守，限制新增买单数量（长仓回撤风控）。
    _buy_orders = [o for o in new_orders if o.get("action") == "buy"]
    _max_buys = _max_new_buys(_market_temperature(as_of=cutoff))
    if len(_buy_orders) > _max_buys:
        _keep = set(o["id"] for o in _buy_orders[:_max_buys])
        new_orders = [o for o in new_orders if o.get("action") != "buy" or o["id"] in _keep]

    # 4. Re-read and merge under one exclusive lock.  This preserves a GUI
    # confirm/reject/set-qty update that may happen while strategy scanning runs.
    return _merge_new_orders(new_orders)


def _place_execution_order(o: dict) -> dict:
    """根据已确认指令调用 execution.create_order 创建真实执行订单。

    返回 {'created': True, 'execution_order_id': N} 或
          {'created': False, 'reason': '...'}（失败原因可见，不静默吞掉）。
    """
    symbol = o.get('stock', '')
    action = o.get('action', '')
    qty = int(o.get('qty', 0) or 0)
    price = o.get('price', 0) or 0

    if not symbol or not action:
        return {"created": False, "reason": "缺少股票代码或方向"}
    if qty <= 0:
        return {"created": False, "reason": "数量为0（需人工指定资金）"}

    # P1-Q19-fix: A股买入须 100 股整数倍，向下取整（卖出允许零股）
    if action == 'buy':
        qty = (qty // 100) * 100
        if qty <= 0:
            return {"created": False, "reason": "买入数量不足100股"}

    try:
        from quant_system import execution
        r = execution.create_order(
            symbol=symbol,
            order_type="market",
            direction=action,
            shares=qty,
            price=price if price > 0 else None,
            notes=f"人工确认指令 {o.get('id', '')}: {o.get('reason', '')[:60]}",
        )
        if "error" in r:
            return {"created": False, "reason": r["error"]}
        created = {"created": True, "execution_order_id": r.get("id")}

        # P2-Q19-fix(M179): stop_loss 接入订单——买入指令确认后挂保护性止损卖单，
        # 原 stop_loss 仅展示无下游使用
        stop_loss = o.get('stop_loss') or 0
        if action == 'buy' and stop_loss > 0:
            try:
                stop_r = execution.create_order(
                    symbol=symbol,
                    order_type="stop",
                    direction="sell",
                    shares=qty,
                    trigger_price=stop_loss,
                    notes=f"保护性止损(指令{o.get('id', '')}) 参考买入价{o.get('price', 0)}",
                )
                if "error" in stop_r:
                    created["stop_order_error"] = stop_r["error"]
                else:
                    created["stop_order_id"] = stop_r.get("id")
            except Exception as e:
                created["stop_order_error"] = str(e)
        return created
    except Exception as e:
        return {"created": False, "reason": str(e)}


def confirm_order(order_id: str, place_execution: bool = False) -> bool:
    """确认执行某条指令。

    place_execution=True 时：除翻转 JSON 状态外，调用 execution.create_order 真实下单，
    由 execution.check_orders 撮合（人工确认 → 真实执行链路）。
    默认 False 保持"仅记录"（仿真流程沿用，避免双重记账）。
    """
    def _mut(orders: list[dict]) -> bool:
        for o in orders:
            if o.get('id') == order_id:
                # Real execution must succeed before the order becomes confirmed.
                if place_execution:
                    execution = _place_execution_order(o)
                    o['execution'] = execution
                    if not execution.get('created'):
                        o['status'] = 'execution_failed'
                        o['confirmed_at'] = None
                        return False
                else:
                    o.setdefault('execution', {'created': False, 'reason': '记录模式：仅确认，未下单'})
                o['status'] = 'confirmed'
                o['confirmed_at'] = datetime.now(CST).isoformat()
                return True
        return False
    return bool(_mutate_orders(_mut))


def reject_order(order_id: str) -> bool:
    """拒绝某条指令"""
    def _mut(orders: list[dict]) -> bool:
        for o in orders:
            if o.get('id') == order_id:
                o['status'] = 'rejected'
                o['confirmed_at'] = datetime.now(CST).isoformat()
                return True
        return False
    return bool(_mutate_orders(_mut))


def set_order_qty(order_id: str, qty: int, price: float | None = None) -> bool:
    """人工补全 pending 指令股数；买入按 100 股整数手向下归整。"""
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        return False
    if qty <= 0:
        return False

    def _mut(orders: list[dict]) -> bool:
        for o in orders:
            if o.get('id') == order_id:
                action = o.get('action')
                q = (qty // 100) * 100 if action == 'buy' else qty
                if q <= 0:
                    return False
                o['qty'] = q
                o['qty_needs_manual'] = False
                o['qty_updated_at'] = datetime.now(CST).isoformat()
                if price is not None:
                    try:
                        p = float(price)
                    except (TypeError, ValueError):
                        p = 0.0
                    if p > 0:
                        o['price'] = p
                return True
        return False
    return bool(_mutate_orders(_mut))


def format_pending_orders(orders: list[dict]) -> str:
    if not orders:
        return "✅ 当前无待执行指令"

    lines = ["# 📋 待确认交易指令\n"]
    lines.append(f"{'ID':<30}{'股票':<12}{'动作':<8}{'价格':>8}{'止损':>8}{'理由'}")
    lines.append("-" * 80)
    for o in orders:
        oid = o.get('id', '')
        stock = f"{o.get('stock_name','')}({o.get('stock','')[:6]})"
        action = "🟢买入" if o.get('action') == 'buy' else "🔴卖出"
        price = o.get('price', 0)
        sl = o.get('stop_loss', 0)
        reason = o.get('reason', '')[:30]
        # P2-Q19-fix(M179): 数量待人工补全的指令显式标注，避免误确认不可执行指令
        marker = " ⚠️待补数量" if o.get('qty_needs_manual') else ""
        lines.append(f"  {oid[:28]:<28} {stock:<10} {action:<8} {price:>7.2f} {sl:>7.2f} {reason}{marker}")
    lines.append(f"\n补数量: python3 -m quant_system.trade_executor --set-qty <ID> <股数> [价格]")
    lines.append(f"确认: python3 -m quant_system.trade_executor --confirm <ID>")
    lines.append(f"拒绝: python3 -m quant_system.trade_executor --reject <ID>")
    return "\n".join(lines)


def format_execution_report() -> str:
    parts = [f"# 🔄 交易执行链路 ({datetime.now(CST).strftime('%Y-%m-%d %H:%M')})"]
    parts.append("=" * 50)

    # 获取待执行指令
    pending = scan_and_execute()
    parts.append(format_pending_orders(pending))

    # 历史记录
    all_orders = _load_orders()
    confirmed = [o for o in all_orders if o.get('status') == 'confirmed']
    rejected = [o for o in all_orders if o.get('status') == 'rejected']
    parts.append(f"\n📊 历史: 已确认{len(confirmed)}条 / 已拒绝{len(rejected)}条")

    # 仓位统计
    try:
        from quant_system.trade_db import TradeDB
        db = TradeDB()
        positions = db.get_positions()
        parts.append(f"\n📦 当前持仓: {len(positions)}只")
        total_pnl = sum(float(p.get('pnl', 0) or 0) for p in positions)
        parts.append(f"  浮动盈亏: {total_pnl:+.2f}")
    except ImportError:
        pass

    return "\n".join(parts)


def main():
    args = sys.argv[1:]

    if any(a.startswith('--set-qty') for a in args):
        idx = args.index('--set-qty')
        if idx + 2 < len(args):
            price = float(args[idx + 3]) if idx + 3 < len(args) else None
            if set_order_qty(args[idx + 1], int(args[idx + 2]), price=price):
                print(f"✅ 已补全指令 {args[idx + 1]} 数量={args[idx + 2]}" + (f" 价格={price:.2f}" if price else ""))
            else:
                print(f"❌ 补全失败 {args[idx + 1]}")
        else:
            print("用法: python3 -m quant_system.trade_executor --set-qty <ID> <股数> [价格]")
    elif any(a.startswith('--confirm') for a in args):
        idx = args.index('--confirm')
        if idx + 1 < len(args):
            if confirm_order(args[idx + 1], place_execution=True):
                print(f"✅ 已确认指令 {args[idx + 1]}")
                # P1-Q19-fix: 展示真实下单结果，保证"确认后执行"可观测
                for o in _load_orders():
                    if o.get('id') == args[idx + 1]:
                        exe = o.get('execution')
                        if exe and exe.get('created'):
                            print(f"  → 已创建执行订单 #{exe.get('execution_order_id')}")
                            # P2-Q19-fix(M179): 展示保护性止损单创建结果（可见不静默）
                            if exe.get('stop_order_id'):
                                print(f"  → 已挂保护性止损单 #{exe.get('stop_order_id')}")
                            elif exe.get('stop_order_error'):
                                print(f"  ⚠️ 止损单创建失败: {exe.get('stop_order_error')}")
                        elif exe:
                            print(f"  ⚠️ 未下单: {exe.get('reason')}")
                        break
            else:
                print(f"❌ 未找到指令 {args[idx + 1]}")
    elif any(a.startswith('--reject') for a in args):
        idx = args.index('--reject')
        if idx + 1 < len(args):
            if reject_order(args[idx + 1]):
                print(f"✅ 已拒绝指令 {args[idx + 1]}")
            else:
                print(f"❌ 未找到指令 {args[idx + 1]}")
    elif "--daemon" in args:
        print("🔄 交易执行循环启动 (每300秒扫描)")
        while True:
            pending = scan_and_execute()
            if pending:
                print(f"\n[{datetime.now(CST).strftime('%H:%M:%S')}] {len(pending)}条待确认")
                for o in pending:
                    print(f"  {o['action'].upper()} {o['stock_name']}({o['stock']}) @ {o['price']}")
            else:
                print(f"[{datetime.now(CST).strftime('%H:%M:%S')}] 无待执行指令")
            _time.sleep(300)
    else:
        print(format_execution_report())


if __name__ == "__main__":
    main()
