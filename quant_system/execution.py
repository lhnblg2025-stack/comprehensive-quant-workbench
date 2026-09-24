"""
execution.py — 执行引擎 + 订单管理。

订单类型: market(市价), limit(限价), stop(止损), stop_limit(止损限价), trailing(追踪止损)
状态机: pending → approved → sent → partial_filled → filled
                                    → cancelled → rejected → expired

所有下单数据持久化到 trade_db 的 orders / order_fills 表。

D3收敛登记 (2026-08-11): 执行域券商成本模型唯一真源 = execution_broker.py
（P2-Q28 文档化：佣金万0.85/最低5元/印花税万5仅卖出/过户费万0.1双边/滑点）。
本模块成本核算（_fill_internal）已薄壳转发 execution_broker.estimate_trade_cost，
费率常量转发 execution_broker 模块级常量；estimate_slippage/split_order 与 broker
签名不兼容，标注 'D3收敛: 能力未合并'，保留独立实现（不强迁）。
"""

from __future__ import annotations
import logging

import sys
from datetime import datetime, timedelta, timezone
from typing import Any

CST = timezone(timedelta(hours=8))

# D3收敛 (2026-08-11): A股交易成本模型唯一真源 = execution_broker（P2-Q28 文档化）。
# 费率常量与成本计算统一转发 execution_broker，本模块不再独立定义费率。
from quant_system.execution_broker import (
    COMMISSION_RATE,
    MIN_COMMISSION,
    STAMP_TAX_RATE,
    TRANSFER_FEE_RATE,
    estimate_trade_cost,
)

# ── Order state machine transitions ────────────────────────────
_VALID_TRANSITIONS: dict[str, set[str]] = {
    # P2-Q19-fix(L182): 补 pending→sent 边——市价单创建路径直接置 'sent'（自动成交），
    # 原定义缺该边导致创建路径绕开状态机定义
    "pending": {"approved", "rejected", "cancelled", "sent"},
    "approved": {"sent", "cancelled"},
    "sent": {"partial_filled", "filled", "cancelled"},
    "partial_filled": {"filled", "cancelled", "partial_filled"},
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
    "expired": set(),
}

_TERMINAL_STATUSES = frozenset({"filled", "cancelled", "rejected", "expired"})


# ════════════════════════════════════════════════════════════════
# 内部 DB 工具
# ════════════════════════════════════════════════════════════════

_fill_cost_cols_checked = False


def _ensure_fill_cost_columns() -> None:
    """P2-Q19-fix(M173): 幂等迁移——order_fills 增加 transfer_fee 列（A股过户费 0.001% 双边）。"""
    global _fill_cost_cols_checked
    if _fill_cost_cols_checked:
        return
    from quant_system import trade_db
    with trade_db._conn() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(order_fills)").fetchall()]
        if "transfer_fee" not in cols:
            conn.execute("ALTER TABLE order_fills ADD COLUMN transfer_fee REAL DEFAULT 0")
    _fill_cost_cols_checked = True


def _db() -> Any:
    """Lazy import trade_db and return a connection."""
    from quant_system import trade_db
    trade_db.init_db()
    _ensure_fill_cost_columns()
    return trade_db._conn()


def _row_to_dict(row: Any) -> dict:
    """Convert sqlite3.Row to dict."""
    return dict(row) if row else {}


# ════════════════════════════════════════════════════════════════
# 1. 订单创建
# ════════════════════════════════════════════════════════════════

def create_order(
    symbol: str,
    order_type: str,
    direction: str,
    shares: int,
    price: float | None = None,
    trigger_price: float | None = None,
    trailing_pct: float | None = None,
    parent_order_id: int | None = None,
    notes: str = "",
) -> dict:
    """
    创建订单。

    订单类型:
      - market: 市价单，立即成交
      - limit: 限价单，达到 price 时成交
      - stop: 止损单，达到 trigger_price 时触发市价单
      - stop_limit: 止损限价单，达到 trigger_price 时生成限价单(price)
      - trailing_stop: 追踪止损，价格从最高回落 trailing_pct% 时触发

    返回订单 dict（含 id）。
    """
    # 校验
    valid_types = {"market", "limit", "stop", "stop_limit", "trailing_stop"}
    if order_type not in valid_types:
        return {"error": f"无效订单类型: {order_type}，支持: {valid_types}"}

    valid_dirs = {"buy", "sell"}
    if direction not in valid_dirs:
        return {"error": f"无效方向: {direction}，支持: {valid_dirs}"}

    if shares <= 0:
        return {"error": "股份数量必须 > 0"}

    # P1-Q19-fix: A股买入必须 100 股整数倍（卖出允许零股一次性申报）
    if direction == "buy" and shares % 100 != 0:
        return {"error": f"A股买入必须 100 股整数倍，收到 {shares} 股"}

    # F1.1-fix: 卖单下单时校验持仓 → 阻止超持仓坏单入库。
    # 持仓上下文可从 trade_db 安全取得（不需要账户上下文），故在 create_order 阶段就拦截。
    if direction == "sell":
        from quant_system import trade_db
        held = trade_db.get_position(symbol)
        held_shares = held["shares"] if held else 0
        if shares > held_shares:
            return {"error": f"卖出{shares}股超过当前持仓{held_shares}股"}
    # F1.1-fix: 买单下单时余额校验——需账户/可用资金上下文。
    # 本系统无实时账户可用资金（cash/balance）数据源，无法安全校验下单金额 <= 可用资金，
    # 故此处如实跳过，不编造校验；超资金买单仍由成交侧(_fill_internal)兜底拒绝。
    # TODO(需账户上下文, 未校验): 接入资金账户后在此补充 "下单金额 <= 可用资金" 校验。

    if order_type == "limit" and (price is None or price <= 0):
        return {"error": "限价单需要有效的 price"}
    if order_type == "stop" and (trigger_price is None or trigger_price <= 0):
        return {"error": "止损单需要有效的 trigger_price"}
    if order_type == "stop_limit" and (trigger_price is None or trigger_price <= 0 or price is None or price <= 0):
        return {"error": "止损限价单需要有效的 trigger_price 和 price"}
    if order_type == "trailing_stop" and (trailing_pct is None or trailing_pct <= 0):
        return {"error": "追踪止损需要有效的 trailing_pct"}
    # P2-Q19-fix(M178): 追踪止损仅支持卖出方向（按持仓最高价回撤触发）。
    # 买入方向需"从最低点回升"触发，持仓表无最低价字段无法正确实现，
    # 明确拒绝创建，避免下单后永不触发的静默失效。
    if order_type == "trailing_stop" and direction == "buy":
        return {"error": "追踪止损(买入方向)暂不支持：需跟踪最低价回升触发，"
                          "当前持仓模型无法提供该数据，请改用 stop / stop_limit / limit 单"}

    with _db() as conn:
        c = conn.execute(
            """INSERT INTO orders
               (symbol, order_type, direction, shares, price, trigger_price,
                trailing_pct, status, parent_order_id, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (symbol, order_type, direction, shares, price, trigger_price,
             trailing_pct, parent_order_id, notes),
        )
        order_id = c.lastrowid

        # 市价单直接批准并送到 sent
        if order_type == "market":
            conn.execute(
                "UPDATE orders SET status = 'sent', updated_at = datetime('now','localtime') WHERE id = ?",
                (order_id,),
            )

        # 重新读取完整记录
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()

    result = _row_to_dict(row)
    result["message"] = f"✅ 订单 #{order_id} 已创建 ({order_type}/{direction} {shares}股 {symbol})"
    return result


def get_order(order_id: int) -> dict:
    """获取单个订单详情。"""
    from quant_system import trade_db
    with trade_db._conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not row:
            return {"error": f"订单 #{order_id} 不存在"}
        # 获取成交明细
        fills = conn.execute(
            "SELECT * FROM order_fills WHERE order_id = ? ORDER BY filled_at",
            (order_id,),
        ).fetchall()
    result = _row_to_dict(row)
    result["fills"] = [_row_to_dict(f) for f in fills]
    return result


def get_orders(
    status: str = "",
    symbol: str = "",
    order_type: str = "",
    days: int = 7,
    limit: int = 100,
) -> list[dict]:
    """查询订单列表。"""
    from quant_system import trade_db
    conditions = ["1=1"]
    params: list[Any] = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if order_type:
        conditions.append("order_type = ?")
        params.append(order_type)

    conditions.append(
        "created_at >= datetime('now', ?, 'localtime')"
    )
    params.append(f"-{days} days")

    sql = f"SELECT * FROM orders WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with trade_db._conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════
# 2. 订单操作
# ════════════════════════════════════════════════════════════════

def _transition(order_id: int, new_status: str, conn: Any) -> bool:
    """内部：安全转换订单状态。"""
    row = conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchone()
    if not row:
        return False
    current = row["status"]

    if new_status not in _VALID_TRANSITIONS.get(current, set()):
        return False

    conn.execute(
        "UPDATE orders SET status = ?, updated_at = datetime('now','localtime') WHERE id = ?",
        (new_status, order_id),
    )
    return True


def approve_order(order_id: int) -> dict:
    """批准订单 (pending → approved)。"""
    from quant_system import trade_db
    with trade_db._conn() as conn:
        ok = _transition(order_id, "approved", conn)
    return {"success": ok, "order_id": order_id,
            "message": f"订单 #{order_id} {'已批准' if ok else '状态转换失败'}"}


def reject_order(order_id: int, reason: str = "") -> dict:
    """拒绝订单 (pending → rejected)。"""
    from quant_system import trade_db
    with trade_db._conn() as conn:
        ok = _transition(order_id, "rejected", conn)
        if ok and reason:
            conn.execute(
                "UPDATE orders SET notes = CASE WHEN notes != '' THEN notes || '; ' || ? ELSE ? END WHERE id = ?",
                (reason, reason, order_id),
            )
    return {"success": ok, "order_id": order_id,
            "message": f"订单 #{order_id} {'已拒绝' if ok else '状态转换失败'}"}


def cancel_order(order_id: int) -> dict:
    """
    撤销订单。
    只有 pending / approved / sent / partial_filled 状态的订单可以撤销。
    """
    from quant_system import trade_db
    with trade_db._conn() as conn:
        ok = _transition(order_id, "cancelled", conn)
    return {"success": ok, "order_id": order_id,
            "message": f"订单 #{order_id} {'已撤销' if ok else '无法撤销（状态不允许或不存在）'}"}


def expire_order(order_id: int) -> dict:
    """使订单过期 (→ expired)。"""
    from quant_system import trade_db
    with trade_db._conn() as conn:
        ok = _transition(order_id, "expired", conn)
    return {"success": ok, "order_id": order_id}


# ════════════════════════════════════════════════════════════════
# 3. 订单检查与自动成交
# ════════════════════════════════════════════════════════════════

def _is_in_market_hours(now: datetime | None = None) -> bool:
    """
    检查是否在A股交易时间内。

    交易时间: 9:30-11:30, 13:00-15:00
    收盘前15分钟 (14:45-15:00) 不执行新止损单。
    """
    if now is None:
        now = datetime.now(CST)
    weekday = now.weekday()
    if weekday >= 5:  # 周六日
        return False

    minute_of_day = now.hour * 60 + now.minute

    # 早盘 9:30-11:30
    if 570 <= minute_of_day <= 690:
        return True
    # 午盘 13:00-15:00
    if 780 <= minute_of_day <= 900:
        return True
    return False


def _is_stop_cutoff(now: datetime | None = None) -> bool:
    """
    检查是否在K线边界（收盘前15分钟）。
    收盘前15分钟不执行新止损单 (14:45-15:00)。
    """
    if now is None:
        now = datetime.now(CST)
    minute_of_day = now.hour * 60 + now.minute
    return 885 <= minute_of_day <= 900  # 14:45 - 15:00


def check_orders(current_prices: dict[str, float] | None = None) -> list[dict]:
    """
    检查所有待成交订单，自动成交满足条件的订单。

    current_prices: {symbol: price} 字典，None 时自动从 watchlist.fetch_quotes 获取。

    返回成交/触发的订单列表。
    """
    from quant_system import trade_db

    if current_prices is None:
        try:
            from quant_system.watchlist import fetch_quotes
            pending_symbols = _get_pending_symbols()
            if pending_symbols:
                quotes = fetch_quotes(pending_symbols)
                current_prices = {q["symbol"]: q["price"] for q in quotes if q.get("price", 0) > 0}
            else:
                return []
        except Exception:
            return []

    if not current_prices:
        return []

    now = datetime.now(CST)
    in_market = _is_in_market_hours(now)
    stop_cutoff = _is_stop_cutoff(now)

    triggered: list[dict] = []

    with trade_db._conn() as conn:
        rows = conn.execute(
            """SELECT * FROM orders
               WHERE status IN ('approved', 'sent', 'partial_filled')
               ORDER BY created_at""",
        ).fetchall()

        for row in rows:
            order = _row_to_dict(row)
            oid = order["id"]
            sym = order["symbol"]
            otype = order["order_type"]
            direction = order["direction"]
            shares = order["shares"]
            filled = order.get("filled_shares", 0) or 0
            remaining = shares - filled
            price = order.get("price")
            trigger = order.get("trigger_price")
            trailing = order.get("trailing_pct")

            curr_price = current_prices.get(sym)
            if curr_price is None or curr_price <= 0:
                continue

            # P1-Q19-fix: 非交易时间一律跳过执行（含市价单），
            # 避免市价单在周末/盘后以过期价格自动成交。
            if not in_market:
                continue

            fill_this: bool = False
            fill_price: float = curr_price
            fill_reason: str = ""

            if otype == "market":
                fill_this = True
                fill_reason = "市价单自动成交"

            elif otype == "limit":
                # 限价买：市价 <= 限价; 限价卖：市价 >= 限价
                if direction == "buy" and price is not None and curr_price <= price:
                    fill_this = True
                    fill_reason = f"限价单触发 (现价{curr_price:.2f} ≤ 限价{price:.2f})"
                elif direction == "sell" and price is not None and curr_price >= price:
                    fill_this = True
                    fill_reason = f"限价单触发 (现价{curr_price:.2f} ≥ 限价{price:.2f})"

            elif otype == "stop":
                # 止损触发：卖单触发价 >= 现价（跌破）, 买单触发器 >= 现价（涨破）？
                # 标准定义：止损卖单：市价 <= trigger_price 时触发
                #           止损买单：市价 >= trigger_price 时触发
                if stop_cutoff:
                    continue  # K线边界不执行新止损单
                if direction == "sell" and trigger is not None and curr_price <= trigger:
                    fill_this = True
                    fill_reason = f"止损卖单触发 (现价{curr_price:.2f} ≤ 触发价{trigger:.2f})"
                elif direction == "buy" and trigger is not None and curr_price >= trigger:
                    fill_this = True
                    fill_reason = f"止损买单触发 (现价{curr_price:.2f} ≥ 触发价{trigger:.2f})"

            elif otype == "stop_limit":
                # 达到 trigger_price 后激活限价单
                if stop_cutoff:
                    continue
                limit_triggered = False
                if direction == "sell" and trigger is not None and curr_price <= trigger:
                    limit_triggered = True
                elif direction == "buy" and trigger is not None and curr_price >= trigger:
                    limit_triggered = True

                if limit_triggered:
                    # 然后检查限价是否可成交
                    if direction == "buy" and price is not None and curr_price <= price:
                        fill_this = True
                        fill_reason = f"止损限价触发 (现价{curr_price:.2f}, 限价{price:.2f})"
                    elif direction == "sell" and price is not None and curr_price >= price:
                        fill_this = True
                        fill_reason = f"止损限价触发 (现价{curr_price:.2f}, 限价{price:.2f})"

            elif otype == "trailing_stop":
                if stop_cutoff:
                    continue
                # P2-Q19-fix(M178): 买入方向追踪止损不支持（创建时已拒绝），
                # 历史遗留买单在此显式跳过并告警，禁止永不触发的静默失效。
                if direction == "buy":
                    import logging
                    logging.getLogger(__name__).warning(
                        f"追踪止损买单#{oid} 不受支持，跳过（请改用 stop/limit 单）"
                    )
                    continue
                # 追踪止损需要读取当前持仓的最高价
                pos = trade_db.get_position(sym)
                highest = pos.get("highest_price") if pos else None
                if trailing and highest and highest > 0:
                    trailing_price = highest * (1 - trailing / 100)
                    if curr_price <= trailing_price:
                        fill_this = True
                        fill_reason = f"追踪止损触发 (最高{highest:.2f}, 回撤{trailing:.0f}%, 价{trailing_price:.2f})"

            if fill_this:
                # P2-Q19-fix(M173): 按方向应用滑点得到预估成交价（不再以现价直接成交）；
                # 限价/止损限价单以限价为上下限（成交价不得越过限价）
                slip_pct = estimate_slippage(sym, direction, remaining)
                if direction == "buy":
                    fill_price = curr_price * (1 + slip_pct / 100)
                else:
                    fill_price = curr_price * (1 - slip_pct / 100)
                if otype in ("limit", "stop_limit") and price and price > 0:
                    if direction == "buy":
                        fill_price = min(fill_price, price)
                    else:
                        fill_price = max(fill_price, price)
                fill_price = round(fill_price, 4)
                try:
                    result = _fill_internal(conn, oid, fill_price, remaining, fill_reason)
                    if result:
                        triggered.append(result)
                except ValueError as e:
                    # P1-Q19-fix: 显式降级——订单无法成交（剩余量校验/T+1 校验失败），
                    # 标记为 rejected 并在 notes 记录原因，避免静默吞掉。
                    conn.execute(
                        """UPDATE orders SET status = 'rejected',
                           notes = CASE WHEN notes != '' THEN notes || '; ' || ? ELSE ? END,
                           updated_at = datetime('now','localtime')
                           WHERE id = ?""",
                        (str(e), str(e), oid),
                    )
                    import logging
                    logging.getLogger(__name__).warning(f"订单#{oid} 拒绝执行: {e}")

    return triggered


def _get_pending_symbols() -> list[str]:
    """获取所有待成交订单中的股票代码（用于拉取实时行情）。"""
    from quant_system import trade_db
    with trade_db._conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT symbol FROM orders
               WHERE status IN ('approved', 'sent', 'partial_filled')"""
        ).fetchall()
    return [r["symbol"] for r in rows]


def _fill_internal(conn: Any, order_id: int, fill_price: float,
                   fill_shares: int, reason: str = "") -> dict | None:
    """
    内部：执行成交（不重新 acquire connection）。
    更新订单状态、记录成交明细、更新持仓。
    """
    from quant_system import trade_db

    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if not row:
        return None
    order = _row_to_dict(row)
    curr_status = order["status"]
    if curr_status in _TERMINAL_STATUSES:
        return None

    now_str = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
    symbol = order["symbol"]
    direction = order["direction"]
    total_shares = order["shares"]
    old_filled = order.get("filled_shares", 0) or 0
    remaining = total_shares - old_filled
    # P1-Q19-fix: 成交数量不得超出剩余量，防止手动 fill 超买超卖
    # （如先 fill 60/100 再 fill 100 → 成交 160 > 100）
    if fill_shares <= 0:
        raise ValueError(f"订单#{order_id} 成交数量必须 > 0（收到 {fill_shares}）")
    if fill_shares > remaining:
        raise ValueError(
            f"订单#{order_id} 成交数量 {fill_shares} 超过剩余可成交数量 {remaining}"
        )
    new_filled = old_filled + fill_shares
    is_full = new_filled >= total_shares

    # P1-Q19-fix: A股 T+1 校验——当日买入股数不可当日卖出。
    # 必须在订单标记成交/写入 order_fills 之前校验，否则 T+1 拒绝会留下
    # "订单已成交但持仓未同步"的半成品状态。
    # 可卖数量 = 持仓股数 - 当日累计买入（加权平均成本模型下成立）。
    if direction == "sell":
        today = datetime.now(CST).strftime("%Y-%m-%d")
        buy_row = conn.execute(
            """SELECT COALESCE(SUM(shares), 0) AS s FROM trades
               WHERE symbol = ? AND trade_type = 'buy' AND trade_date = ?""",
            (symbol, today),
        ).fetchone()
        today_buys = buy_row["s"] if buy_row else 0
        pos_row = trade_db.get_position(symbol)
        held = pos_row["shares"] if pos_row else 0
        sellable = max(0, held - today_buys)
        if fill_shares > sellable:
            raise ValueError(
                f"T+1 限制: {symbol} 可卖 {sellable} 股 < 卖出 {fill_shares} 股"
                f"（当日买入 {today_buys} 股次日方可卖出）"
            )

    # P2-Q19-fix(M173): A股交易成本核算入库——佣金(最低5元/笔)、印花税(卖出0.05%)、过户费(双边0.001%)
    # D3收敛 (2026-08-11): 成本计算薄壳转发 execution_broker.estimate_trade_cost（唯一真源）。
    fee = estimate_trade_cost(
        side=direction,
        quantity=fill_shares,
        price=fill_price,
        commission_rate=COMMISSION_RATE,
        min_commission=MIN_COMMISSION,
        stamp_tax_rate=STAMP_TAX_RATE,
        transfer_fee_rate=TRANSFER_FEE_RATE,
    )
    commission = fee["commission"]
    stamp_tax = fee["stamp_tax"]
    transfer_fee = fee["transfer_fee"]
    _ensure_fill_cost_columns()
    # 记录成交明细
    conn.execute(
        """INSERT INTO order_fills (order_id, filled_shares, filled_price,
                                    commission, stamp_tax, transfer_fee, filled_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (order_id, fill_shares, fill_price, commission, stamp_tax, transfer_fee, now_str),
    )

    # 更新订单
    new_status = "filled" if is_full else "partial_filled"
    conn.execute(
        """UPDATE orders SET
               status = ?, filled_shares = ?, filled_price = ?,
               filled_at = ?,
               updated_at = datetime('now','localtime')
           WHERE id = ?""",
        (new_status, new_filled, fill_price, now_str if is_full else None, order_id),
    )

    # 先提交订单侧写入，释放写锁（WAL 单写者），再在独立连接/独立事务中更新持仓。
    # 修复：此前在未提交事务内通过第二个连接调用 trade_db.add_buy/add_sell，
    # 触发 database is locked 被静默吞掉，订单标记成交但持仓永不更新（Q19 修复）。
    conn.commit()

    # 更新持仓（独立事务）
    import logging
    _log = logging.getLogger(__name__)
    try:
        if direction == "buy":
            pos_result = trade_db.add_buy(
                symbol, order.get("name", "") or "", fill_shares, fill_price,
                notes=f"订单#{order_id}成交: {reason}",
            )
        else:
            pos_result = trade_db.add_sell(
                symbol, fill_shares, fill_price,
                notes=f"订单#{order_id}成交: {reason}",
            )
    except Exception as e:
        # 显式失败：订单已标记成交但持仓未同步，必须让调用方感知，不再静默吞掉
        _log.error(f"持仓更新异常 (订单#{order_id}): {e}", exc_info=True)
        raise
    if isinstance(pos_result, dict) and pos_result.get("error"):
        _log.error(f"持仓更新失败 (订单#{order_id}): {pos_result['error']}")
        raise RuntimeError(f"订单#{order_id} 持仓更新失败: {pos_result['error']}")

    # P2-Q19-fix(L186): 成交后重读 DB 返回最新订单记录，避免返回成交前快照
    #（此前仅手动补丁 filled/status/fill_price，其余字段均为旧值）
    fresh = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if fresh is None:
        raise RuntimeError(f"订单#{order_id} 成交后重读记录失败")
    result = _row_to_dict(fresh)
    result["reason"] = reason
    return result


def fill_order(order_id: int, fill_price: float, fill_shares: int) -> dict:
    """
    手动成交订单。更新持仓（调用 trade_db.add_buy / add_sell）。
    """
    from quant_system import trade_db
    try:
        with trade_db._conn() as conn:
            result = _fill_internal(conn, order_id, fill_price, fill_shares, "手动成交")
    except ValueError as e:
        # P1-Q19-fix: 显式报错（超量/非正数成交），不再静默吞掉
        return {"error": f"订单 #{order_id} 无法成交: {e}"}
    if result:
        return result
    return {"error": f"订单 #{order_id} 无法成交（不存在或已终态）"}


# ════════════════════════════════════════════════════════════════
# 4. 滑点模型
# ════════════════════════════════════════════════════════════════

def estimate_slippage(
    symbol: str,
    direction: str,
    shares: int,
    market_cap_yi: float | None = None,
) -> float:
    """
    预估滑点百分比。

    Args:
        symbol: 股票代码
        direction: buy / sell
        shares: 买入/卖出股数
        market_cap_yi: 市值(亿)，不传则尝试从行情获取

    Returns:
        预估滑点百分比 (如 0.1 表示 0.1%)

    D3收敛: 能力未合并——按市值档次×成交量×方向的滑点模型 execution_broker 未提供
    同签名函数（SimBroker 使用 MarketImpact/固定万10），签名不兼容，保留独立实现，不强迁。
    """
    if market_cap_yi is None or market_cap_yi <= 0:
        try:
            from quant_system.watchlist import fetch_quotes
            quotes = fetch_quotes([symbol])
            if quotes:
                market_cap_yi = quotes[0].get("market_cap_yi", 0)
        except Exception as e:
            logging.getLogger(__name__).error(f"[execution] 操作失败: {e}", exc_info=True)

    # 按市值档次定滑点
    if market_cap_yi and market_cap_yi >= 500:
        base_slippage = 0.1
    elif market_cap_yi and market_cap_yi >= 100:
        base_slippage = 0.2
    elif market_cap_yi and market_cap_yi >= 50:
        base_slippage = 0.3
    else:
        base_slippage = 0.5

    # P2-Q19-fix(M172): 成交量系数——分支从大到小排列，避免首个分支恒命中
    #（原顺序 >10000 在前，>50000 / >100000 永不可达，6万/20万股均按 1.5 计）
    vol_factor = 1.0
    if shares > 100000:
        vol_factor = 3.0
    elif shares > 50000:
        vol_factor = 2.0
    elif shares > 10000:
        vol_factor = 1.5

    # 方向系数：卖出滑点通常略大
    dir_factor = 1.1 if direction == "sell" else 1.0

    return round(base_slippage * vol_factor * dir_factor, 4)


# ════════════════════════════════════════════════════════════════
# 5. K线边界检查
# ════════════════════════════════════════════════════════════════

def is_stop_cutoff_period(now: datetime | None = None) -> bool:
    """
    检查当前是否在止损拒绝执行时段。
    A股收盘前15分钟 (14:45-15:00) 不执行新止损单。
    """
    return _is_stop_cutoff(now)


# ════════════════════════════════════════════════════════════════
# 6. VWAP 拆分
# ════════════════════════════════════════════════════════════════

def split_order(order_id: int, n_slices: int) -> list[dict]:
    """
    将大订单拆分为 n_slices 个子订单，按时间均匀分布。

    子订单时间：
      - 如果当前在交易时间，从当前时间开始均匀分布到收盘
      - 如果不在交易时间，从下一个开盘开始均匀分布

    返回创建的子订单列表。

    D3收敛: 能力未合并——DB 级 VWAP 拆分（子单落库/整手+零股守恒）execution_broker
    无对应能力（StrategyRunner 为内存目标量执行，签名不兼容），保留独立实现，不强迁。
    """
    if n_slices < 1:
        return [{"error": "拆分份数必须 >= 1"}]

    from quant_system import trade_db

    parent = get_order(order_id)
    if "error" in parent:
        return [parent]

    if parent["status"] in _TERMINAL_STATUSES:
        return [{"error": f"父订单 #{order_id} 已是终态，无法拆分"}]

    total_shares = parent["shares"]
    order_type = parent["order_type"]
    direction = parent["direction"]
    symbol = parent["symbol"]
    price = parent.get("price")
    trigger_price = parent.get("trigger_price")
    trailing_pct = parent.get("trailing_pct")

    # P1-Q19-fix: 父单置终态 'cancelled'——不再留在 check_orders 候选集
    # （'sent' 仍在候选集，shares 未减会被下次检查全额成交）；
    # 拆分原因写入 notes 以便追溯。
    with trade_db._conn() as conn:
        conn.execute(
            """UPDATE orders SET status = 'cancelled', parent_order_id = ?,
               notes = CASE WHEN notes != '' THEN notes || '; ' || ? ELSE ? END,
               updated_at = datetime('now','localtime') WHERE id = ?""",
            (order_id, f"已拆分为{n_slices}个子单", f"已拆分为{n_slices}个子单", order_id),
        )

    # 每份股数（V4.1 audit fix: A股按 100 股整数倍拆分，余量并入尾单，
    # 否则子单如 166 股无法按整手下单/成交）
    LOT_SIZE = 100
    total_lots = total_shares // LOT_SIZE
    base_lots = total_lots // n_slices
    remainder_lots = total_lots % n_slices
    base_shares = base_lots * LOT_SIZE
    remainder = remainder_lots * LOT_SIZE  # 前 remainder_lots 个整手余量
    non_lot = total_shares - total_lots * LOT_SIZE  # 不足一手的零股，并入尾单

    # 计算时间间隔
    now = datetime.now(CST)
    minute_of_day = now.hour * 60 + now.minute

    # 交易时段定义 (分钟从0点起)
    session_ranges = [(570, 690), (780, 900)]  # 9:30-11:30, 13:00-15:00

    remaining_minutes = 0
    for start, end in session_ranges:
        if minute_of_day < end:
            if minute_of_day < start:
                remaining_minutes += end - start
            else:
                remaining_minutes += end - minute_of_day
            break
        # 如果过了这个时段，不加

    # 如果今天已收盘，从明天9:30开始
    if remaining_minutes <= 5:
        remaining_minutes = 240  # 明天全天 4小时

    interval = max(1, remaining_minutes // (n_slices + 1))

    sub_orders: list[dict] = []
    with trade_db._conn() as conn:
        for i in range(n_slices):
            slice_shares = base_shares + (LOT_SIZE if i < remainder_lots else 0)
            if i == n_slices - 1:
                # V4.1 audit fix: 零股(不足一手)并入尾单，保证拆分总量守恒
                slice_shares += non_lot
            if slice_shares <= 0:
                continue

            # P1-Q19-fix: 子单以 'sent' 创建——进入 check_orders 待执行候选集
            #（此前 'pending' 不在候选集，子单永不成交）
            c = conn.execute(
                """INSERT INTO orders
                   (symbol, order_type, direction, shares, price, trigger_price,
                    trailing_pct, status, parent_order_id, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'sent', ?, ?)""",
                (symbol, order_type, direction, slice_shares, price, trigger_price,
                 trailing_pct, order_id,
                 f"VWAP拆分 #{i + 1}/{n_slices} 父订单#{order_id}"),
            )
            sub_order_id = c.lastrowid
            sub = _row_to_dict(conn.execute("SELECT * FROM orders WHERE id = ?", (sub_order_id,)).fetchone())
            sub["scheduled_slice"] = i + 1
            sub["total_slices"] = n_slices
            sub_orders.append(sub)

    return sub_orders


# ════════════════════════════════════════════════════════════════
# 7. 批量操作工具
# ════════════════════════════════════════════════════════════════

def process_pending_orders() -> dict:
    """
    一站式处理所有待成交订单：
      1. 拉取实时价格
      2. 检查并自动成交满足条件的订单
      3. 返回处理摘要
    """
    from quant_system import trade_db

    # 统计待处理订单
    with trade_db._conn() as conn:
        pending_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM orders WHERE status IN ('approved','sent','partial_filled')"
        ).fetchone()["cnt"]

    triggered = check_orders()

    # 统计结果
    filled_count = len(triggered)
    remaining = pending_count - filled_count

    return {
        "ts": _ts(),
        "pending_before": pending_count,
        "filled": filled_count,
        "pending_remaining": max(0, remaining),
        "triggered_orders": triggered,
    }


# ════════════════════════════════════════════════════════════════
# 内部工具
# ════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _ts_short() -> str:
    return datetime.now(CST).strftime("%H:%M")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="执行引擎 — 订单管理")
    sub = parser.add_subparsers(dest="cmd")

    # create
    p_create = sub.add_parser("create", help="创建订单")
    p_create.add_argument("--symbol", required=True)
    p_create.add_argument("--type", dest="order_type", required=True,
                          choices=["market", "limit", "stop", "stop_limit", "trailing_stop"])
    p_create.add_argument("--direction", required=True, choices=["buy", "sell"])
    p_create.add_argument("--shares", type=int, required=True)
    p_create.add_argument("--price", type=float)
    p_create.add_argument("--trigger", type=float, dest="trigger_price")
    p_create.add_argument("--trailing", type=float, dest="trailing_pct")
    p_create.add_argument("--notes", default="")

    # cancel
    p_cancel = sub.add_parser("cancel", help="撤销订单")
    p_cancel.add_argument("order_id", type=int)

    # fill
    p_fill = sub.add_parser("fill", help="手动成交订单")
    p_fill.add_argument("order_id", type=int)
    p_fill.add_argument("--price", type=float, required=True)
    p_fill.add_argument("--shares", type=int, required=True)

    # check
    p_check = sub.add_parser("check", help="检查并自动成交")

    # get
    p_get = sub.add_parser("get", help="查询订单")
    p_get.add_argument("order_id", type=int, nargs="?", default=0)

    # list
    p_list = sub.add_parser("list", help="列出订单")
    p_list.add_argument("--status", default="")
    p_list.add_argument("--symbol", default="")
    p_list.add_argument("--days", type=int, default=7)

    # split
    p_split = sub.add_parser("split", help="VWAP拆分订单")
    p_split.add_argument("order_id", type=int)
    p_split.add_argument("--n", type=int, dest="n_slices", required=True)

    # slippage
    p_slip = sub.add_parser("slippage", help="预估滑点")
    p_slip.add_argument("--symbol", required=True)
    p_slip.add_argument("--direction", required=True, choices=["buy", "sell"])
    p_slip.add_argument("--shares", type=int, required=True)

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "create":
        r = create_order(args.symbol, args.order_type, args.direction,
                         args.shares, args.price, args.trigger_price,
                         args.trailing_pct, notes=args.notes)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ {r['message']}")

    elif args.cmd == "cancel":
        r = cancel_order(args.order_id)
        print(r["message"])

    elif args.cmd == "fill":
        r = fill_order(args.order_id, args.price, args.shares)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ 订单 #{args.order_id} 成交 {args.shares}股 @ {args.price:.2f}")

    elif args.cmd == "check":
        results = check_orders()
        print(f"检查完成: {len(results)} 单自动成交")
        for r in results:
            print(f"  #{r['id']} {r['symbol']} - {r.get('reason', '')}")

    elif args.cmd == "get":
        if args.order_id:
            r = get_order(args.order_id)
            if "error" in r:
                print(f"❌ {r['error']}")
            else:
                print(f"订单 #{r['id']}: {r['symbol']} {r['order_type']}/{r['direction']} "
                      f"{r['shares']}股 [{r['status']}]")
                fills = r.get("fills", [])
                if fills:
                    for f in fills:
                        print(f"  成交: {f['filled_shares']}股 @ {f['filled_price']:.2f} 于 {f['filled_at']}")

    elif args.cmd == "list":
        orders = get_orders(status=args.status, symbol=args.symbol, days=args.days)
        if not orders:
            print("暂无订单")
        else:
            for o in orders:
                status_emoji = {"pending": "⏳", "approved": "✅", "sent": "📨",
                                "partial_filled": "🔄", "filled": "✔️",
                                "cancelled": "❌", "rejected": "🚫", "expired": "⏰"}
                emoji = status_emoji.get(o["status"], "📄")
                print(f"  {emoji} #{o['id']} {o['symbol']} {o['order_type']}/{o['direction']} "
                      f"{o['shares']}股 [{o['status']}]")

    elif args.cmd == "split":
        subs = split_order(args.order_id, args.n_slices)
        print(f"拆分完成: {len(subs)} 个子订单")
        for s in subs:
            print(f"  #{s['id']} {s['shares']}股 [{s['status']}]")

    elif args.cmd == "slippage":
        slip = estimate_slippage(args.symbol, args.direction, args.shares)
        print(f"预估滑点: {slip:.2f}%")
