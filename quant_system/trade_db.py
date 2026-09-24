"""
持仓管理数据库层。

SQLite 本地存储，无需外部依赖。
三个表：positions（当前持仓）、trades（交易流水）、signal_records（信号记录）

路径: ~/.quant_system/trade_log.db  （用户主目录下，不随 workspace 变动）
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

CST = timezone(timedelta(hours=8))

# Database path: user home directory
# 2026-08-22 稳定化: ~/.quant_system 在部署环境为只读 → sqlite 打开报错噪音;
# 只有可写时才用家目录, 否则回退仓库内 generated/quant_state（纸面日志持久可写）。
def _resolve_db_path() -> Path:
    env = os.environ.get("TRADE_DB_DIR")
    if env:
        p = Path(env)
        try:
            p.mkdir(parents=True, exist_ok=True)
            (p / ".wtest").write_text("ok", encoding="utf-8")
            (p / ".wtest").unlink()
            return p / "trade_log.db"
        except OSError:
            pass
    home_dir = Path.home() / ".quant_system"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        (home_dir / ".wtest").write_text("ok", encoding="utf-8")
        (home_dir / ".wtest").unlink()
        return home_dir / "trade_log.db"
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / "generated" / "quant_state"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / "trade_log.db"


_DB_DIR = _resolve_db_path().parent
_DB_PATH = _resolve_db_path()


# P2-Q3-fix(M401): sqlite3.Connection 的上下文管理器只 commit 不 close，
# 高频调用（refresh/check_stops）会连接堆积、依赖 GC 关闭。改为 contextmanager，
# 退出时 commit + 显式 close（异常时 rollback 后 close）。
@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables and run migrations."""
    with _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                shares INTEGER NOT NULL DEFAULT 0,
                cost_price REAL NOT NULL DEFAULT 0,
                total_cost REAL NOT NULL DEFAULT 0,
                current_price REAL,
                total_value REAL,
                pnl REAL,
                pnl_pct REAL,
                buy_date TEXT,
                sector TEXT DEFAULT '',
                signal_type TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                stop_loss_price REAL,
                take_profit_price REAL,
                stop_pct REAL DEFAULT -8.0,
                take_profit_pct REAL DEFAULT 15.0,
                last_refresh TEXT,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT DEFAULT '',
                trade_type TEXT NOT NULL CHECK(trade_type IN ('buy','sell')),
                shares INTEGER NOT NULL,
                price REAL NOT NULL,
                total_amount REAL NOT NULL,
                trade_date TEXT NOT NULL,
                notes TEXT DEFAULT '',
                signal_type TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT DEFAULT '',
                signal_type TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                entry_date TEXT,
                exit_date TEXT,
                pnl_pct REAL,
                outcome TEXT DEFAULT 'pending' CHECK(outcome IN ('pending','win','loss','breakeven')),
                shares INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
            CREATE INDEX IF NOT EXISTS idx_signal_type ON signal_records(signal_type);
            CREATE INDEX IF NOT EXISTS idx_signal_outcome ON signal_records(outcome);

            -- Orders table
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL CHECK(order_type IN ('market','limit','stop','stop_limit','trailing_stop')),
                direction TEXT NOT NULL CHECK(direction IN ('buy','sell')),
                shares INTEGER NOT NULL,
                price REAL,
                trigger_price REAL,
                trailing_pct REAL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','approved','sent','partial_filled','filled','cancelled','rejected','expired')),
                filled_shares INTEGER DEFAULT 0,
                filled_price REAL,
                filled_at TEXT,
                parent_order_id INTEGER,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS order_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                execution_id TEXT,
                filled_shares INTEGER NOT NULL,
                filled_price REAL NOT NULL,
                commission REAL DEFAULT 0,
                stamp_tax REAL DEFAULT 0,
                transfer_fee REAL DEFAULT 0,
                filled_at TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                UNIQUE(order_id, execution_id)
            );

            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_order_fills_order ON order_fills(order_id);
        """)

        # Migration: add columns if missing
        cols = [r[1] for r in db.execute("PRAGMA table_info(positions)").fetchall()]
        migrations = {
            "stop_loss_price": "ALTER TABLE positions ADD COLUMN stop_loss_price REAL",
            "take_profit_price": "ALTER TABLE positions ADD COLUMN take_profit_price REAL",
            "stop_pct": "ALTER TABLE positions ADD COLUMN stop_pct REAL DEFAULT -8.0",
            "take_profit_pct": "ALTER TABLE positions ADD COLUMN take_profit_pct REAL DEFAULT 15.0",
            "highest_price": "ALTER TABLE positions ADD COLUMN highest_price REAL",
            "trailing_stop_pct": "ALTER TABLE positions ADD COLUMN trailing_stop_pct REAL DEFAULT 5.0",
        }
        for col, sql in migrations.items():
            if col not in cols:
                db.execute(sql)
        # Migration: add close_reason to trades
        t_cols = [r[1] for r in db.execute("PRAGMA table_info(trades)").fetchall()]
        if "close_reason" not in t_cols:
            db.execute("ALTER TABLE trades ADD COLUMN close_reason TEXT DEFAULT 'manual'")

        # Migration: add fee columns to trades; legacy rows remain zero-cost.
        fee_migrations = {
            "commission": "ALTER TABLE trades ADD COLUMN commission REAL DEFAULT 0",
            "stamp_tax": "ALTER TABLE trades ADD COLUMN stamp_tax REAL DEFAULT 0",
            "transfer_fee": "ALTER TABLE trades ADD COLUMN transfer_fee REAL DEFAULT 0",
            "gross_amount": "ALTER TABLE trades ADD COLUMN gross_amount REAL DEFAULT 0",
        }
        for col, sql in fee_migrations.items():
            if col not in t_cols:
                db.execute(sql)

        o_cols = [r[1] for r in db.execute("PRAGMA table_info(orders)").fetchall()]
        if not o_cols:
            _create_orders_tables(db)
            o_cols = [r[1] for r in db.execute("PRAGMA table_info(orders)").fetchall()]
        order_migrations = {
            "candidate_id": "ALTER TABLE orders ADD COLUMN candidate_id TEXT",
            "release_id": "ALTER TABLE orders ADD COLUMN release_id TEXT",
            "suggested_price": "ALTER TABLE orders ADD COLUMN suggested_price REAL",
            "suggested_at": "ALTER TABLE orders ADD COLUMN suggested_at TEXT",
        }
        for col, sql in order_migrations.items():
            if col not in o_cols:
                db.execute(sql)
        fill_cols = [r[1] for r in db.execute("PRAGMA table_info(order_fills)").fetchall()]
        fill_migrations = {
            "execution_id": "ALTER TABLE order_fills ADD COLUMN execution_id TEXT",
            "transfer_fee": "ALTER TABLE order_fills ADD COLUMN transfer_fee REAL DEFAULT 0",
        }
        for col, sql in fill_migrations.items():
            if col not in fill_cols:
                db.execute(sql)
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_order_fills_execution ON order_fills(order_id, execution_id) WHERE execution_id IS NOT NULL")


def _create_orders_tables(db: sqlite3.Connection) -> None:
    """Create orders and order_fills tables (migration helper)."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            order_type TEXT NOT NULL CHECK(order_type IN ('market','limit','stop','stop_limit','trailing_stop')),
            direction TEXT NOT NULL CHECK(direction IN ('buy','sell')),
            shares INTEGER NOT NULL,
            price REAL,
            trigger_price REAL,
            trailing_pct REAL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','approved','sent','partial_filled','filled','cancelled','rejected','expired')),
            filled_shares INTEGER DEFAULT 0,
            filled_price REAL,
            filled_at TEXT,
            parent_order_id INTEGER,
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now','localtime')),
            updated_at TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS order_fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            execution_id TEXT,
            filled_shares INTEGER NOT NULL,
            filled_price REAL NOT NULL,
            commission REAL DEFAULT 0,
            stamp_tax REAL DEFAULT 0,
            transfer_fee REAL DEFAULT 0,
            filled_at TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            UNIQUE(order_id, execution_id)
        );
        CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_order_fills_order ON order_fills(order_id);
    """)


def _validate_fees(*, gross: float, commission, stamp_tax, transfer_fee,
                   side: str) -> tuple[tuple[float, float, float] | None, str | None]:
    values = []
    for name, raw in (("commission", commission), ("stamp_tax", stamp_tax),
                      ("transfer_fee", transfer_fee)):
        try:
            value = float(raw or 0)
        except (TypeError, ValueError):
            return None, f"{name} 非法: {raw!r}"
        if not math.isfinite(value) or value < 0:
            return None, f"{name} 必须为有限非负数 (received {raw!r})"
        values.append(round(value, 2))
    if side == "sell" and sum(values) > gross:
        return None, f"卖出费用合计不能超过成交额 (fees={sum(values):.2f}, gross={gross:.2f})"
    return (values[0], values[1], values[2]), None


def create_paper_order(*, symbol: str, direction: str, shares: int,
                       suggested_price: float, release_id: str,
                       order_type: str = "limit", notes: str = "",
                       candidate_id: str | None = None) -> dict:
    """Create an auditable paper order bound to immutable research inputs."""
    init_db()
    direction = str(direction).lower()
    if direction not in {"buy", "sell"}:
        return {"error": f"direction must be buy/sell: {direction}"}
    try:
        shares = int(shares); suggested_price = float(suggested_price)
    except (TypeError, ValueError):
        return {"error": "shares/suggested_price invalid"}
    if shares <= 0 or suggested_price <= 0 or not release_id:
        return {"error": "shares, suggested_price and release_id are required"}
    now = datetime.now(CST).isoformat(timespec="seconds")
    with _conn() as db:
        cur = db.execute("""INSERT INTO orders
            (symbol, order_type, direction, shares, price, status, notes,
             release_id, suggested_price, suggested_at, candidate_id)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
            (str(symbol), order_type, direction, shares, suggested_price, notes,
             release_id, suggested_price, now, candidate_id))
        order_id = int(cur.lastrowid)
    return get_paper_order(order_id) or {"id": order_id}


def get_paper_order(order_id: int) -> dict | None:
    init_db()
    with _conn() as db:
        row = db.execute("SELECT * FROM orders WHERE id = ?", (int(order_id),)).fetchone()
    return dict(row) if row else None


def record_paper_fill(order_id: int, *, filled_shares: int, filled_price: float,
                      commission: float = 0.0, stamp_tax: float = 0.0,
                      transfer_fee: float = 0.0, execution_id: str | None = None,
                      filled_at: str | None = None) -> dict:
    """Append one fill and update order state; execution_id makes retries idempotent."""
    init_db()
    try:
        filled_shares = int(filled_shares); filled_price = float(filled_price)
    except (TypeError, ValueError):
        return {"error": "filled_shares/filled_price invalid"}
    if filled_shares <= 0 or filled_price <= 0 or not math.isfinite(filled_price):
        return {"error": "fill values must be finite and positive"}
    fee_values, fee_error = _validate_fees(gross=filled_shares * filled_price,
                                            commission=commission, stamp_tax=stamp_tax,
                                            transfer_fee=transfer_fee, side="sell")
    if fee_error:
        return {"error": fee_error}
    commission, stamp_tax, transfer_fee = fee_values
    execution_id = str(execution_id).strip() if execution_id else None
    filled_at = filled_at or datetime.now(CST).isoformat(timespec="seconds")
    with _conn() as db:
        db.execute("BEGIN IMMEDIATE")
        order = db.execute("SELECT * FROM orders WHERE id = ?", (int(order_id),)).fetchone()
        if not order:
            return {"error": f"order not found: {order_id}"}
        if execution_id:
            existing_fill = db.execute("SELECT id FROM order_fills WHERE order_id=? AND execution_id=?", (int(order_id), execution_id)).fetchone()
            if existing_fill:
                result = dict(order)
                result.update({"duplicate": True, "execution_id": execution_id})
                return result
        current = int(order["filled_shares"] or 0)
        if current + filled_shares > int(order["shares"]):
            return {"error": "fill exceeds requested shares"}
        db.execute("""INSERT INTO order_fills
            (order_id, execution_id, filled_shares, filled_price, commission, stamp_tax, transfer_fee, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(order_id), execution_id, filled_shares, filled_price, commission, stamp_tax, transfer_fee, filled_at))
        rows = db.execute("SELECT filled_shares, filled_price FROM order_fills WHERE order_id = ?", (int(order_id),)).fetchall()
        total_shares = sum(int(row["filled_shares"]) for row in rows)
        weighted = sum(int(row["filled_shares"]) * float(row["filled_price"]) for row in rows) / total_shares
        status = "filled" if total_shares == int(order["shares"]) else "partial_filled"
        db.execute("""UPDATE orders SET filled_shares=?, filled_price=?, filled_at=?,
            status=?, updated_at=datetime('now','localtime') WHERE id=?""",
            (total_shares, weighted, filled_at, status, int(order_id)))
    return get_paper_order(order_id) or {"id": order_id}


def list_paper_orders(*, release_id: str | None = None, limit: int = 1000) -> list[dict]:
    """Return paper intents with one row per order for audit/reconciliation."""
    init_db()
    where, params = "", []
    if release_id:
        where, params = "WHERE release_id = ?", [release_id]
    safe_limit = max(1, min(int(limit), 10000))
    with _conn() as db:
        rows = db.execute(f"SELECT * FROM orders {where} ORDER BY id DESC LIMIT ?", [*params, safe_limit]).fetchall()
    return [dict(row) for row in rows]


def list_paper_fills(*, release_id: str | None = None, limit: int = 10000) -> list[dict]:
    """Return immutable fill events joined to their originating order."""
    init_db()
    where, params = "", []
    if release_id:
        where, params = "WHERE o.release_id = ?", [release_id]
    safe_limit = max(1, min(int(limit), 100000))
    sql = f"""SELECT f.id AS fill_id, f.order_id, o.release_id, o.candidate_id,
        o.symbol, o.direction AS side, o.shares AS requested_shares,
        o.suggested_price, f.execution_id, f.filled_shares, f.filled_price, f.commission,
        f.stamp_tax, f.transfer_fee, f.filled_at
        FROM order_fills f JOIN orders o ON o.id = f.order_id
        {where} ORDER BY f.id DESC LIMIT ?"""
    with _conn() as db:
        rows = db.execute(sql, [*params, safe_limit]).fetchall()
    return [dict(row) for row in rows]


def paper_reconciliation(*, release_id: str | None = None) -> dict:
    """Reconcile every paper order against fill events without CSV exports."""
    import pandas as pd
    from scripts.paper_execution_reconciliation import reconcile

    orders = list_paper_orders(release_id=release_id, limit=10000)
    fills = list_paper_fills(release_id=release_id, limit=100000)
    order_rows = [{"order_id": row["id"], "symbol": row["symbol"], "side": row["direction"],
                   "requested_shares": row["shares"], "suggested_price": row.get("suggested_price") or row.get("price")}
                  for row in orders]
    fill_rows = [{"order_id": row["order_id"], "filled_shares": row["filled_shares"],
                  "filled_price": row["filled_price"]} for row in fills]
    order_frame = pd.DataFrame(order_rows, columns=sorted(REQUIRED_PAPER_ORDER_COLUMNS))
    fill_frame = pd.DataFrame(fill_rows, columns=sorted(REQUIRED_PAPER_FILL_COLUMNS))
    rows, summary = reconcile(order_frame, fill_frame)
    return {"schema": "paper-execution-reconciliation/v1", "release_id": release_id,
            "summary": summary, "rows": json.loads(rows.to_json(orient="records"))}


REQUIRED_PAPER_ORDER_COLUMNS = {"order_id", "symbol", "side", "requested_shares", "suggested_price"}
REQUIRED_PAPER_FILL_COLUMNS = {"order_id", "filled_shares", "filled_price"}


def paper_execution_stats(*, release_id: str | None = None) -> dict:
    """Aggregate fill rate and adverse execution slippage for paper orders."""
    init_db()
    where, params = "", []
    if release_id:
        where, params = "WHERE release_id = ?", [release_id]
    with _conn() as db:
        rows = [dict(row) for row in db.execute(f"SELECT * FROM orders {where} ORDER BY id", params).fetchall()]
    slips = []
    requested = filled = 0
    for row in rows:
        requested += int(row.get("shares") or 0); filled += int(row.get("filled_shares") or 0)
        suggested = float(row.get("suggested_price") or row.get("price") or 0)
        actual = row.get("filled_price")
        if actual is None or suggested <= 0:
            continue
        sign = 1 if row.get("direction") == "buy" else -1
        slips.append((float(actual) - suggested) / suggested * sign * 10000)
    slips.sort()
    return {
        "release_id": release_id, "orders": len(rows), "requested_shares": requested,
        "filled_shares": filled, "fill_rate": filled / requested if requested else 0.0,
        "unfilled_orders": sum(1 for row in rows if int(row.get("filled_shares") or 0) == 0),
        "partial_orders": sum(1 for row in rows if row.get("status") == "partial_filled"),
        "mean_adverse_slippage_bps": sum(slips) / len(slips) if slips else None,
        "median_adverse_slippage_bps": slips[len(slips) // 2] if slips else None,
    }


# ════════════════════════════════════════════════════════════════
# 持仓操作
# ════════════════════════════════════════════════════════════════

def get_positions() -> list[dict]:
    """Get all open positions (shares > 0)."""
    with _conn() as db:
        rows = db.execute("""
            SELECT * FROM positions WHERE shares > 0 ORDER BY total_cost DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_position(symbol: str) -> dict | None:
    """Get single position."""
    with _conn() as db:
        r = db.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
    return dict(r) if r else None


def add_buy(symbol: str, name: str, shares: int, price: float,
            signal_type: str = "", notes: str = "",
            sector: str = "", trade_date: str = "",
            commission: float = 0.0, stamp_tax: float = 0.0,
            transfer_fee: float = 0.0) -> dict:
    """
    买入记录。

    - 如果已有该股持仓，自动计算加权平均成本
    - 记录到 trades 表
    """
    # P2-Q3-fix(M400): 入口校验，禁止 shares<=0 / price<=0 入库；
    # shares=0 且无持仓时 new_total_cost/new_shares=0/0 会 ZeroDivisionError，
    # price=0 会让后续 add_sell/refresh_prices 的除零崩溃。
    try:
        shares = int(shares)
        price = float(price)
    except (TypeError, ValueError):
        return {"error": f"买入数量/价格非法: shares={shares!r}, price={price!r}"}
    if shares <= 0:
        return {"error": f"买入股数必须>0 (received {shares})"}
    if not math.isfinite(price) or price <= 0:
        return {"error": f"买入价格必须为有限正数 (received {price})"}

    trade_date = trade_date or datetime.now(CST).strftime("%Y-%m-%d")
    gross = round(shares * price, 2)
    fees, fee_error = _validate_fees(
        gross=gross, commission=commission, stamp_tax=stamp_tax,
        transfer_fee=transfer_fee, side="buy",
    )
    if fee_error:
        return {"error": fee_error}
    commission, stamp_tax, transfer_fee = fees
    total = round(gross + commission + stamp_tax + transfer_fee, 2)

    with _conn() as db:
        # P2-Q3-fix(L402): 单连接内完成 read-modify-write；BEGIN IMMEDIATE 抢占写锁，
        # 防止 WAL 下两个并发连接各自读到旧值后互相覆盖（丢失更新）。
        db.execute("BEGIN IMMEDIATE")
        # Check existing position
        existing = db.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()

        if existing:
            existing = dict(existing)
            old_shares = existing["shares"]
            old_cost = existing["total_cost"]
            new_shares = old_shares + shares
            new_total_cost = old_cost + total
            new_cost_price = round(new_total_cost / new_shares, 4)

            db.execute("""
                UPDATE positions SET
                    shares = ?, cost_price = ?, total_cost = ?,
                    name = CASE WHEN ? != '' THEN ? ELSE name END,
                    sector = CASE WHEN ? != '' THEN ? ELSE sector END,
                    signal_type = CASE WHEN ? != '' THEN ? ELSE signal_type END,
                    notes = CASE WHEN ? != '' THEN ? ELSE notes END,
                    buy_date = CASE WHEN buy_date IS NULL THEN ? ELSE buy_date END,
                    updated_at = datetime('now','localtime')
                WHERE symbol = ?
            """, (new_shares, new_cost_price, new_total_cost,
                  name, name, sector, sector,
                  signal_type, signal_type, notes, notes,
                  trade_date, symbol))
        else:
            db.execute("""
                INSERT INTO positions
                    (symbol, name, shares, cost_price, total_cost,
                     buy_date, sector, signal_type, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, name, shares, round(total / shares, 4), total,
                  trade_date, sector, signal_type, notes))

        # Record trade
        db.execute("""
            INSERT INTO trades
                (symbol, name, trade_type, shares, price, total_amount,
                 gross_amount, commission, stamp_tax, transfer_fee,
                 trade_date, notes, signal_type)
            VALUES (?, ?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, name, shares, price, total, gross, commission, stamp_tax,
              transfer_fee, trade_date, notes, signal_type))

        # Record signal entry
        if signal_type:
            db.execute("""
                INSERT INTO signal_records
                    (symbol, name, signal_type, entry_price, entry_date,
                     shares, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, name, signal_type, price, trade_date, shares, notes))

    return get_position(symbol) or {"symbol": symbol, "shares": 0}


def add_sell(symbol: str, shares: int, price: float,
             notes: str = "", trade_date: str = "",
             commission: float = 0.0, stamp_tax: float = 0.0,
             transfer_fee: float = 0.0) -> dict:
    """
    卖出记录。

    - 减少持仓
    - 记录交易
    - 如果全部卖出，更新信号 outcome
    """
    # P2-Q3-fix(M400): 入口校验 + 除数保护
    try:
        shares = int(shares)
        price = float(price)
    except (TypeError, ValueError):
        return {"error": f"卖出数量/价格非法: shares={shares!r}, price={price!r}"}
    if shares <= 0:
        return {"error": f"卖出股数必须>0 (received {shares})"}
    if not math.isfinite(price) or price <= 0:
        return {"error": f"卖出价格必须为有限正数 (received {price})"}

    trade_date = trade_date or datetime.now(CST).strftime("%Y-%m-%d")
    gross = round(shares * price, 2)
    fees, fee_error = _validate_fees(
        gross=gross, commission=commission, stamp_tax=stamp_tax,
        transfer_fee=transfer_fee, side="sell",
    )
    if fee_error:
        return {"error": fee_error}
    commission, stamp_tax, transfer_fee = fees
    total = round(gross - commission - stamp_tax - transfer_fee, 2)

    with _conn() as db:
        # P2-Q3-fix(L402): 单连接内完成 read-modify-write；BEGIN IMMEDIATE 抢占写锁
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not existing:
            return {"error": f"未找到 {symbol} 的持仓记录"}

        existing = dict(existing)
        old_shares = existing["shares"]
        cost_price = existing["cost_price"]
        name = existing["name"]

        if shares > old_shares:
            return {"error": f"卖出{shares}股超过持仓{old_shares}股"}

        # F2.1-fix: A股 T+1 校验——当日买入的股份次日方可卖出。
        # 口径与 execution.py _fill_internal 的 T+1 检查保持一致：
        # 可卖数量 = 持仓股数 - 当日累计买入。仅当"当日买 + 当日卖"才拦截，
        # 不影响正常的当日高抛（纯存量持仓卖出）。
        today_buy_row = db.execute(
            """SELECT COALESCE(SUM(shares), 0) AS s FROM trades
               WHERE symbol = ? AND trade_type = 'buy' AND trade_date = ?""",
            (symbol, trade_date),
        ).fetchone()
        today_buys = today_buy_row["s"] if today_buy_row else 0
        sellable = max(0, old_shares - today_buys)
        if shares > sellable:
            logging.getLogger(__name__).warning(
                "F2.1 T+1 限制: %s 当日买入 %s 股不可当日卖出（可卖 %s 股 < 卖出 %s 股）",
                symbol, today_buys, sellable, shares,
            )
            return {"error": (
                f"T+1 限制: {symbol} 当日买入 {today_buys} 股不可当日卖出"
                f"（可卖 {sellable} 股 < 卖出 {shares} 股）"
            )}

        # Realized P&L includes sell-side commission, stamp tax and transfer fee.
        sell_pnl = round(total - cost_price * shares, 2)
        # P2-Q3-fix(M400): cost_price 可能为 0（历史脏数据），除零保护
        sell_pnl_pct = round((price / cost_price - 1) * 100, 2) if cost_price > 0 else 0.0

        new_shares = old_shares - shares
        new_total_cost = round(new_shares * cost_price, 2)

        if new_shares == 0:
            # P2-Q3-fix(L403): 清仓不再 DELETE —— 保留行(shares=0)以留存
            # 止损/止盈/追踪参数与最高价历史，positions 表可回溯历史持仓；
            # 查询端 get_positions() 已按 shares>0 过滤，不影响持仓视图。
            db.execute("""
                UPDATE positions SET
                    shares = 0,
                    total_cost = 0,
                    total_value = NULL,
                    pnl = NULL,
                    pnl_pct = NULL,
                    current_price = NULL,
                    updated_at = datetime('now','localtime')
                WHERE symbol = ?
            """, (symbol,))
            # Close pending signal records
            db.execute("""
                UPDATE signal_records SET
                    exit_price = ?, exit_date = ?,
                    pnl_pct = ?, outcome = CASE
                        WHEN ? > 5 THEN 'win'
                        WHEN ? < -5 THEN 'loss'
                        ELSE 'breakeven'
                    END
                WHERE symbol = ? AND outcome = 'pending'
            """, (price, trade_date, sell_pnl_pct, sell_pnl_pct, sell_pnl_pct, symbol))
        else:
            db.execute("""
                UPDATE positions SET
                    shares = ?, total_cost = ?,
                    updated_at = datetime('now','localtime')
                WHERE symbol = ?
            """, (new_shares, new_total_cost, symbol))

        # Record trade
        sell_note = notes + " | 实现盈亏" + ("%+.2f" % sell_pnl) + " (" + ("%+.2f" % sell_pnl_pct) + "%)" if notes else \
                    "实现盈亏" + ("%+.2f" % sell_pnl) + " (" + ("%+.2f" % sell_pnl_pct) + "%)"
        db.execute("""
            INSERT INTO trades
                (symbol, name, trade_type, shares, price, total_amount,
                 gross_amount, commission, stamp_tax, transfer_fee,
                 trade_date, notes)
            VALUES (?, ?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, name, shares, price, total, gross, commission, stamp_tax,
              transfer_fee, trade_date, sell_note))

    # 部分卖出返回最新持仓；清仓返回 realized_pnl 结果
    if new_shares > 0:
        pos = get_position(symbol)
        if pos is not None:
            return pos
    return {"symbol": symbol, "name": name, "shares": new_shares,
            "realized_pnl": sell_pnl,
            "realized_pnl_pct": sell_pnl_pct}


def refresh_prices() -> int:
    """
    用真实市场数据刷新持仓价格。
    Returns: 刷新成功的股票数
    """
    positions = get_positions()
    if not positions:
        return 0

    symbols = [p["symbol"] for p in positions]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from quant_system.watchlist import fetch_quotes
        quotes = fetch_quotes(symbols)
    except Exception:
        return 0

    now_str = datetime.now(CST).strftime("%Y-%m-%d %H:%M")
    updated = 0

    with _conn() as db:
        for sym in symbols:
            q = next((s for s in quotes if s.get("symbol") == sym), None)
            if q is None:
                continue
            current = q.get("price") or q.get("close", 0)
            if current <= 0:
                continue
            cost_row = db.execute(
                "SELECT cost_price, shares, highest_price FROM positions WHERE symbol = ?",
                (sym,)
            ).fetchone()
            if not cost_row:
                continue
            cost_price = cost_row["cost_price"]
            shares = cost_row["shares"]
            total_value = round(current * shares, 2)
            pnl = round((current - cost_price) * shares, 2)
            # P2-Q3-fix(M400): cost_price 可能为 0（历史脏数据），除零保护
            pnl_pct = round((current / cost_price - 1) * 100, 2) if cost_price > 0 else 0.0

            # 更新最高价（用于追踪止损）
            highest = cost_row["highest_price"] or cost_price
            if current > highest:
                highest = current

            db.execute("""
                UPDATE positions SET
                    current_price = ?, total_value = ?,
                    pnl = ?, pnl_pct = ?, highest_price = ?,
                    last_refresh = ?
                WHERE symbol = ?
            """, (current, total_value, pnl, pnl_pct, highest, now_str, sym))
            updated += 1

    return updated


# ════════════════════════════════════════════════════════════════
# 止损止盈管理
# ════════════════════════════════════════════════════════════════

def set_stop(symbol: str, stop_loss_price: float = None,
             take_profit_price: float = None,
             stop_pct: float = None, take_profit_pct: float = None) -> dict:
    """
    Set stop-loss and/or take-profit levels for a position.
    
    Args:
        symbol: 股票代码
        stop_loss_price: 固定止损价
        take_profit_price: 固定止盈价
        stop_pct: 止损百分比(从成本价算,默认-8%)
        take_profit_pct: 止盈百分比(从成本价算,默认15%)
    """
    pos = get_position(symbol)
    if not pos:
        return {"error": f"未找到 {symbol} 的持仓"}

    updates = []
    params = []
    if stop_loss_price is not None:
        updates.append("stop_loss_price = ?")
        params.append(stop_loss_price)
    if take_profit_price is not None:
        updates.append("take_profit_price = ?")
        params.append(take_profit_price)
    if stop_pct is not None:
        updates.append("stop_pct = ?")
        params.append(stop_pct)
    if take_profit_pct is not None:
        updates.append("take_profit_pct = ?")
        params.append(take_profit_pct)

    if not updates:
        return {"error": "未指定止损或止盈参数"}

    updates.append("updated_at = datetime('now','localtime')")
    params.append(symbol)

    with _conn() as db:
        db.execute(
            f"UPDATE positions SET {', '.join(updates)} WHERE symbol = ?",
            params
        )

    return get_position(symbol)


def check_stops() -> list[dict]:
    """
    Check all positions against stop-loss/take-profit levels.
    Returns list of triggered alerts.
    """
    refresh_prices()
    positions = get_positions()
    alerts = []

    for p in positions:
        curr = p.get("current_price")
        if not curr or curr <= 0:
            continue
        cost = p["cost_price"]
        # P2-Q3-fix(M400): cost_price 为 0 时避免除零（历史脏数据保护）
        pnl_pct = (curr / cost - 1) * 100 if cost > 0 else 0.0

        # Check stop-loss (fixed price or percentage)
        stop_price = p.get("stop_loss_price")
        if stop_price is not None and curr <= stop_price:
            alerts.append({
                "type": "STOP_LOSS",
                "symbol": p["symbol"],
                "name": p["name"],
                "shares": p["shares"],
                "cost_price": cost,
                "current_price": curr,
                "stop_price": stop_price,
                "pnl_pct": round(pnl_pct, 2),
                "message": f"🚨 {p['symbol']} {p['name'][:8]} 触发止损！"
                           f"现价{curr:.2f} ≤ 止损{stop_price:.2f} ({pnl_pct:+.2f}%)",
            })
            continue

        # Check percentage stop
        stop_pct = p.get("stop_pct")
        if stop_pct and pnl_pct <= stop_pct:
            alerts.append({
                "type": "STOP_LOSS_PCT",
                "symbol": p["symbol"],
                "name": p["name"],
                "shares": p["shares"],
                "cost_price": cost,
                "current_price": curr,
                "pnl_pct": round(pnl_pct, 2),
                "stop_pct": stop_pct,
                "message": f"🚨 {p['symbol']} {p['name'][:8]} 触发百分比止损！"
                           f"盈亏{pnl_pct:+.2f}% ≤ 止损{stop_pct:+.0f}%",
            })
            continue

        # Check take-profit (fixed price)
        tp_price = p.get("take_profit_price")
        if tp_price is not None and curr >= tp_price:
            alerts.append({
                "type": "TAKE_PROFIT",
                "symbol": p["symbol"],
                "name": p["name"],
                "shares": p["shares"],
                "cost_price": cost,
                "current_price": curr,
                "tp_price": tp_price,
                "pnl_pct": round(pnl_pct, 2),
                "message": f"💰 {p['symbol']} {p['name'][:8]} 触发止盈！"
                           f"现价{curr:.2f} ≥ 止盈{tp_price:.2f} ({pnl_pct:+.2f}%)",
            })
            continue

        # Check percentage take-profit
        tp_pct = p.get("take_profit_pct")
        if tp_pct and pnl_pct >= tp_pct:
            alerts.append({
                "type": "TAKE_PROFIT_PCT",
                "symbol": p["symbol"],
                "name": p["name"],
                "shares": p["shares"],
                "cost_price": cost,
                "current_price": curr,
                "pnl_pct": round(pnl_pct, 2),
                "tp_pct": tp_pct,
                "message": f"💰 {p['symbol']} {p['name'][:8]} 触发百分比止盈！"
                           f"盈亏{pnl_pct:+.2f}% ≥ 止盈{tp_pct:+.0f}%",
            })
            continue

        # Check trailing stop
        trailing_pct = p.get("trailing_stop_pct", 0)
        highest = p.get("highest_price")
        if trailing_pct and trailing_pct > 0 and highest and highest > cost:
            trailing_stop_price = highest * (1 - trailing_pct / 100)
            if curr <= trailing_stop_price:
                drawdown_from_peak = (curr / highest - 1) * 100
                alerts.append({
                    "type": "TRAILING_STOP",
                    "symbol": p["symbol"],
                    "name": p["name"],
                    "shares": p["shares"],
                    "cost_price": cost,
                    "current_price": curr,
                    "highest_price": highest,
                    "trailing_stop_pct": trailing_pct,
                    "pnl_pct": round(pnl_pct, 2),
                    "drawdown_pct": round(drawdown_from_peak, 2),
                    "message": f"🎯 {p['symbol']} {p['name'][:8]} 触发追踪止损！"
                               f"最高¥{highest:.2f}→现¥{curr:.2f} "
                               f"回撤{drawdown_from_peak:.1f}% > 追踪{trailing_pct:.0f}% "
                               f"(盈亏{pnl_pct:+.2f}%)",
                })

    return alerts


# ════════════════════════════════════════════════════════════════
# 交易日志查询
# ════════════════════════════════════════════════════════════════

def get_trades(symbol: str = "", days: int = 30, limit: int = 50) -> list[dict]:
    """Get trade history."""
    with _conn() as db:
        if symbol:
            rows = db.execute("""
                SELECT * FROM trades
                WHERE symbol = ? ORDER BY trade_date DESC, id DESC LIMIT ?
            """, (symbol, limit)).fetchall()
        else:
            since = (datetime.now(CST) - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = db.execute("""
                SELECT * FROM trades WHERE trade_date >= ?
                ORDER BY trade_date DESC, id DESC LIMIT ?
            """, (since, limit)).fetchall()
    return [dict(r) for r in rows]


def get_signal_performance(signal_type: str = "") -> list[dict]:
    """Get signal performance statistics."""
    with _conn() as db:
        if signal_type:
            rows = db.execute("""
                SELECT signal_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                       ROUND(AVG(CASE WHEN pnl_pct IS NOT NULL THEN pnl_pct END),2) as avg_pnl,
                       ROUND(AVG(CASE WHEN outcome='win' THEN pnl_pct END),2) as avg_win,
                       ROUND(AVG(CASE WHEN outcome='loss' THEN pnl_pct END),2) as avg_loss,
                       ROUND(SUM(CASE WHEN outcome='win' THEN ABS(pnl_pct) ELSE 0 END) * 1.0 /
                             NULLIF(SUM(CASE WHEN outcome='loss' THEN ABS(pnl_pct) ELSE 0 END), 0), 2)
                             as profit_factor
                FROM signal_records
                WHERE outcome != 'pending' AND signal_type = ?
                GROUP BY signal_type
            """, (signal_type,)).fetchall()
        else:
            rows = db.execute("""
                SELECT signal_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                       ROUND(AVG(CASE WHEN pnl_pct IS NOT NULL THEN pnl_pct END),2) as avg_pnl
                FROM signal_records
                WHERE outcome != 'pending'
                GROUP BY signal_type
                ORDER BY total DESC
            """).fetchall()
    return [dict(r) for r in rows]


def get_summary() -> dict:
    """
    Portfolio summary.
    Returns {total_cost, total_value, total_pnl, total_pnl_pct,
             num_positions, sector_allocation, ...}
    """
    positions = get_positions()
    if not positions:
        return {"total_cost": 0, "total_value": 0, "total_pnl": 0,
                "total_pnl_pct": 0, "num_positions": 0}

    total_cost = sum(p.get("total_cost", 0) for p in positions)
    total_value = sum(p.get("total_value", 0) or 0 for p in positions)
    total_pnl = round(total_value - total_cost, 2)
    total_pnl_pct = round((total_value / total_cost - 1) * 100, 2) if total_cost > 0 else 0

    # Sector allocation
    sectors = {}
    for p in positions:
        sec = p.get("sector", "未分类") or "未分类"
        val = p.get("total_value", 0) or (p["shares"] * (p.get("current_price") or p["cost_price"]))
        sectors[sec] = sectors.get(sec, 0) + val

    sector_pct = {
        k: round(v / total_value * 100, 1) if total_value > 0 else 0
        for k, v in sorted(sectors.items(), key=lambda x: -x[1])
    }

    return {
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "num_positions": len(positions),
        "sector_allocation": sector_pct,
    }


# ════════════════════════════════════════════════════════════════
# 格式化输出
# ════════════════════════════════════════════════════════════════

def format_positions(positions: list[dict] = None) -> str:
    """Format positions for output."""
    if positions is None:
        positions = get_positions()

    if not positions:
        return "📭 当前无持仓"

    summary = get_summary()
    lines = [
        f"📦 **持仓总览**",
        f"  持仓数量: {summary['num_positions']}只",
        f"  总成本: {summary['total_cost']:.2f}",
        f"  总市值: {summary['total_value']:.2f}",
    ]

    pnl = summary["total_pnl"]
    pnl_pct = summary["total_pnl_pct"]
    if pnl >= 0:
        lines.append(f"  总盈亏: 🟢 **+{pnl:.2f} (+{pnl_pct:.2f}%)**")
    else:
        lines.append(f"  总盈亏: 🔴 **{pnl:.2f} ({pnl_pct:.2f}%)**")

    # Sector allocation
    sectors = summary.get("sector_allocation", {})
    if sectors:
        lines.append(f"  行业分布:")
        for sec, pct in sectors.items():
            bar = "█" * max(1, int(pct / 5))
            lines.append(f"    {sec}: {pct:.1f}% {bar}")

    lines.append(f"\n{'='*50}")
    lines.append(f"📋 **持仓明细**")
    for i, p in enumerate(positions):
        symbol = p["symbol"]
        name = p["name"][:8]
        shares = p["shares"]
        cost_p = p["cost_price"]
        curr = p.get("current_price")
        if curr:
            pnl_val = p.get("pnl", 0)
            pnl_p = p.get("pnl_pct", 0)
            value = p.get("total_value", 0)
            pnl_str = f"🟢+{pnl_val:.2f}" if pnl_val >= 0 else f"🔴{pnl_val:.2f}"
            pnl_p_str = f"+{pnl_p:.2f}%" if pnl_p >= 0 else f"{pnl_p:.2f}%"
            price_str = f"{curr:.2f}"
        else:
            value = shares * cost_p
            pnl_str = "待刷新"
            pnl_p_str = ""
            price_str = f"{cost_p:.2f}*"

        sig = f" [{p['signal_type']}]" if p.get("signal_type") else ""
        lines.append(
            f"  {i+1}. {symbol} {name:<8} {shares}股 "
            f"@{price_str} 市值{value:.0f} "
            f"({pnl_str} {pnl_p_str}){sig}"
        )

    return "\n".join(lines)


def format_trades(trades: list[dict]) -> str:
    """Format trade history."""
    if not trades:
        return "📭 暂无交易记录"

    lines = [f"📋 **最近交易 ({len(trades)}笔)**"]
    for t in trades:
        t_type = "🟢买" if t["trade_type"] == "buy" else "🔴卖"
        lines.append(
            f"  {t['trade_date']} {t_type} {t['symbol']} {t['name'][:8]} "
            f"{t['shares']}股 @{t['price']:.2f} "
            f"金额{t['total_amount']:.0f}"
        )
        if t.get("notes"):
            lines.append(f"     └─ {t['notes'][:60]}")
    return "\n".join(lines)


def format_signal_performance(stats: list[dict]) -> str:
    """Format signal performance."""
    if not stats:
        return "📭 暂无信号记录（还没有完成买卖闭环的信号）"

    lines = [f"📊 **信号胜率统计**"]
    for s in stats:
        total = s["total"]
        wins = s["wins"]
        losses = s["losses"]
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        avg_pnl = s.get("avg_pnl", 0) or 0
        profit_factor = s.get("profit_factor", 0) or 0

        bar = "🟢" * int(win_rate / 10) + "🔴" * max(0, 10 - int(win_rate / 10))
        lines.append(
            f"  {s['signal_type']}: {total}次 胜率{win_rate:.0f}% {bar}"
        )
        lines.append(
            f"    🏆{wins}胜/{losses}负  "
            f"平均盈亏{avg_pnl:+.2f}%  "
            f"盈亏比{profit_factor:.2f}"
        )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════


class TradeDB:
    """V11 审计修复（High-1）: trade_executor/portfolio_risk 依赖 TradeDB 类，
    但本模块是纯函数式实现——补一个门面类包装函数式接口，消除 ImportError 静默吞掉。
    """

    def get_positions(self) -> list[dict]:
        return get_positions()

    def get_position(self, symbol: str) -> dict | None:
        return get_position(symbol)

    def add_buy(self, symbol: str, name: str, shares: int, price: float,
                signal_type: str = "", notes: str = "", sector: str = "",
                trade_date: str = "", stop_loss: float | None = None) -> dict:
        """Class facade for add_buy, using keyword args to preserve field semantics."""
        res = add_buy(symbol, name, shares, price, signal_type=signal_type,
                      notes=notes, sector=sector, trade_date=trade_date)
        if stop_loss is not None and not res.get("error"):
            stop_res = set_stop(symbol, stop_loss_price=stop_loss)
            if isinstance(stop_res, dict) and stop_res.get("error"):
                res["stop_error"] = stop_res["error"]
            else:
                res["stop"] = stop_res
        return res

    def add_sell(self, symbol: str, shares: int, price: float,
                 notes: str = "", trade_date: str = "") -> dict:
        return add_sell(symbol, shares, price, notes=notes, trade_date=trade_date)

    def get_trades(self, symbol: str = "", days: int = 30, limit: int = 50) -> list[dict]:
        return get_trades(symbol, days, limit)

    def get_summary(self) -> dict:
        return get_summary()

    def set_stop(self, symbol: str, stop_loss_price: float = None,
                 take_profit_price: float = None, stop_pct: float = None,
                 take_profit_pct: float = None) -> dict:
        return set_stop(symbol, stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        stop_pct=stop_pct, take_profit_pct=take_profit_pct)

    def check_stops(self) -> list[dict]:
        return check_stops()

    def refresh_prices(self) -> int:
        return refresh_prices()

    def init_db(self) -> None:
        return init_db()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="持仓管理与交易日志")
    sub = parser.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="初始化数据库")

    # buy
    p_buy = sub.add_parser("buy", help="买入记录")
    p_buy.add_argument("--symbol", required=True)
    p_buy.add_argument("--name", default="", help="股票名称")
    p_buy.add_argument("--shares", type=int, required=True)
    p_buy.add_argument("--price", type=float, required=True)
    p_buy.add_argument("--signal", default="", help="触发信号类型")
    p_buy.add_argument("--sector", default="", help="行业")
    p_buy.add_argument("--notes", default="", help="买入理由")

    # sell
    p_sell = sub.add_parser("sell", help="卖出记录")
    p_sell.add_argument("--symbol", required=True)
    p_sell.add_argument("--shares", type=int, required=True)
    p_sell.add_argument("--price", type=float, required=True)
    p_sell.add_argument("--notes", default="")

    # status
    p_status = sub.add_parser("status", help="查看持仓")
    p_status.add_argument("--refresh", action="store_true", help="刷新价格")
    p_status.add_argument("--format", choices=["text", "json"], default="text")

    # trades
    p_trades = sub.add_parser("trades", help="交易记录")
    p_trades.add_argument("--symbol", default="")
    p_trades.add_argument("--days", type=int, default=30)

    # signals
    p_signals = sub.add_parser("signals", help="信号胜率")
    p_signals.add_argument("--signal", default="")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    if args.cmd == "init":
        init_db()
        print(f"✅ 数据库已初始化: {_DB_PATH}")
        sys.exit(0)

    # Ensure initialized
    init_db()

    if args.cmd == "buy":
        r = add_buy(args.symbol, args.name, args.shares, args.price,
                     args.signal, args.notes, args.sector)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ 买入记录: {args.symbol} {args.shares}股 @{args.price:.2f}")
            if args.signal:
                print(f"  信号类型: {args.signal}")

    elif args.cmd == "sell":
        r = add_sell(args.symbol, args.shares, args.price, args.notes)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ 卖出记录: {args.symbol} {args.shares}股 @{args.price:.2f}")
            if "realized_pnl" in r:
                print(f"  实现盈亏: {r['realized_pnl']:+.2f} ({r['realized_pnl_pct']:+.2f}%)")

    elif args.cmd == "status":
        if args.refresh:
            n = refresh_prices()
            print(f"✅ 已刷新 {n} 只持仓价格\n")
        positions = get_positions()
        print(format_positions(positions))

        if args.format == "json":
            print("\n--- JSON ---")
            print(json.dumps(get_summary(), ensure_ascii=False, indent=2))

    elif args.cmd == "trades":
        trades = get_trades(args.symbol, args.days)
        print(format_trades(trades))

    elif args.cmd == "signals":
        stats = get_signal_performance(args.signal)
        print(format_signal_performance(stats))
