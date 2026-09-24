"""
db.py — QuantV6 SQLite 存储
连接池（线程局部）、建表、upsert、事务、WAL 模式。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable


_LOCAL = threading.local()
_DEFAULT_DB = "/root/quant/state/quantv6.db"


def _conn(db_path: str) -> sqlite3.Connection:
    conns = getattr(_LOCAL, "conns", None)
    if conns is None:
        conns = {}
        _LOCAL.conns = conns
    conn = conns.get(db_path)  # P2-Q12-fix: 线程局部连接按 db_path 隔离，避免同线程跨库复用连接
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conns[db_path] = conn
    return conn


class KVStore:
    """通用键值存储（JSON 值）。"""

    def __init__(self, db_path: str = _DEFAULT_DB, table: str = "kv"):
        self.db_path = db_path
        self.table = table
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")

    def _cursor(self):
        return _conn(self.db_path).cursor()

    def put(self, key: str, value: Any) -> None:
        with self._cursor() as cur:
            cur.execute(
                f"INSERT OR REPLACE INTO {self.table} (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False, default=str), time_now()),
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._cursor() as cur:
            cur.execute(f"SELECT value FROM {self.table} WHERE key=?", (key,))
            row = cur.fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

    def delete(self, key: str) -> None:
        with self._cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE key=?", (key,))

    def keys(self) -> list[str]:
        with self._cursor() as cur:
            cur.execute(f"SELECT key FROM {self.table}")
            return [r[0] for r in cur.fetchall()]


def time_now() -> float:
    import time
    return time.time()


class TableStore:
    """通用表存储：create/upsert/query。"""

    def __init__(self, db_path: str = _DEFAULT_DB, table: str = "records",
                 columns: Iterable[tuple[str, str]] = (("id", "TEXT PRIMARY KEY"), ("payload", "TEXT"), ("ts", "REAL"))):
        self.db_path = db_path
        self.table = table
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        col_sql = ", ".join(f"{c} {t}" for c, t in columns)
        with _conn(db_path).cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_sql})")

    def upsert(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        ph = ", ".join("?" for _ in row)
        update = ", ".join(f"{k}=excluded.{k}" for k in row if k != "id")
        sql = f"INSERT INTO {self.table} ({cols}) VALUES ({ph})"
        if update:
            sql += f" ON CONFLICT(id) DO UPDATE SET {update}"
        with _conn(self.db_path).cursor() as cur:
            cur.execute(sql, list(row.values()))

    def query(self, where: str = "", params: tuple = (), limit: int = 100) -> list[sqlite3.Row]:
        sql = f"SELECT * FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        sql += f" LIMIT {limit}"
        with _conn(self.db_path).cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def count(self, where: str = "", params: tuple = ()) -> int:
        sql = f"SELECT COUNT(*) FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        with _conn(self.db_path).cursor() as cur:
            cur.execute(sql, params)
            return int(cur.fetchone()[0])


def transaction(db_path: str = _DEFAULT_DB):
    """事务上下文管理器。"""
    import contextlib

    @contextlib.contextmanager
    def _tx():
        conn = _conn(db_path)
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return _tx()
