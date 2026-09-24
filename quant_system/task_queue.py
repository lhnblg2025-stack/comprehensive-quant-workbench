"""
量化系统 3.1 — 后台任务队列

将长时间计算（Fama-MacBeth、风险模型等）放入后台执行，
前端通过轮询或 SSE 获取结果。

特性:
  - 自动超时保护（60s）
  - 失败自动重试（指数退避，最多 3 次）
  - 并发上限保护（默认最多 2 个并发执行）
  - 任务统计 / 看板接口
  - 自动清理过期任务

用法:
    from quant_system.task_queue import submit_task, get_task_result

    task_id = submit_task("fama_macbeth", {"symbols": [...], "step": 5})
    result = get_task_result(task_id)

    stats = get_queue_stats()  # 看板数据
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("quant_tasks")


def _connect() -> contextlib.AbstractContextManager[sqlite3.Connection]:
    """安全打开任务队列 DB 连接：with 退出时先 commit 再 close。

    注意：`with sqlite3.connect(...) as conn:` 只提交/回滚事务、不关闭连接，
    反复调用会泄漏 fd（quant-web 曾因此耗尽 1024 个 fd 上限导致
    "unable to open database file"）；而 contextlib.closing 只 close 不 commit，
    会丢事务。本函数两者兼顾：读操作无写入时 commit 无害，写操作保证落盘。
    """
    @contextlib.contextmanager
    def _manager() -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(_DB))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    return _manager()

_DB: Optional[Path] = None
# P2-Q3-fix(M394): 从单 worker 改为 MAX_CONCURRENT 个 worker 线程（真正并发），
# 信号量在成功 acquire 后才 release（用 acquired 标志位）。
_WORKERS: list[threading.Thread] = []
_RUNNING = False
_HANDLERS: dict[str, Callable] = {}

# ── 配置 ──
MAX_CONCURRENT = 2          # 最大并发执行任务数
TASK_TIMEOUT = 60           # 普通任务超时秒数
TASK_TIMEOUT_BY_KIND = {
    "quant_backtest": 900,       # 十年多标的研究任务
    "strategy_lab_run": 900,     # 5/10年 Backtrader execution reconciliation
}
MAX_RETRIES = 3             # 最大重试次数
RETRY_BACKOFF_BASE = 5      # 首次重试延迟秒数
RETRY_BACKOFF_MULT = 2      # 退避乘数
CLEANUP_AGE_HOURS = 24      # 清理超过 N 小时的任务
_PENDING_SEM = threading.Semaphore(MAX_CONCURRENT)


def _require_init() -> None:
    """P2-Q3-fix(L399): init() 未调用时抛清晰异常，不再 sqlite3.connect(str(None))
    在 CWD 创建名为 "None" 的数据库文件。"""
    if _DB is None:
        raise RuntimeError("task_queue.init(db_path) 必须先调用后才能使用任务队列")


def init(db_path: Path) -> None:
    """Initialize task queue database."""
    global _DB
    _DB = db_path
    _DB.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                error TEXT,
                created REAL NOT NULL,
                started REAL,
                completed REAL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                next_retry_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_next_retry ON tasks(next_retry_at)")
        # 安全迁移：旧表没有 retry_count / max_retries / next_retry_at 列
        migration_cols = [
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ("max_retries", "INTEGER NOT NULL DEFAULT 3"),
            ("next_retry_at", "REAL"),
        ]
        for col_name, col_def in migration_cols:
            try:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass  # 列已存在

        # P2-Q3-fix(M395): 进程崩溃/重启后 running 任务会永久滞留（cleanup 只清
        # completed 且超龄记录）。启动时把超过 TASK_TIMEOUT+裕量 的 running 任务
        # 重置为 failed 并调度重试（重试次数未耗尽时）。
        stale_cutoff = time.time() - (TASK_TIMEOUT + 60)
        stale_running = conn.execute(
            "SELECT id, started, retry_count, max_retries FROM tasks "
            "WHERE status='running' AND started IS NOT NULL AND started < ?",
            (stale_cutoff,),
        ).fetchall()
        for (tid, started, retries, max_r) in stale_running:
            if retries < max_r:
                conn.execute(
                    "UPDATE tasks SET status='failed', error=?, completed=?, retry_count=retry_count+1, next_retry_at=? WHERE id=?",
                    (f"Recovered after restart: task was running at {started:.0f}, likely interrupted",
                     time.time(), time.time() + RETRY_BACKOFF_BASE, tid),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status='failed', error=?, completed=? WHERE id=?",
                    (f"Recovered after restart: task was running at {started:.0f}, retries exhausted",
                     time.time(), tid),
                )
            logger.warning("Recovered stale running task %s (started %.0f)", tid[:12], started)
    _ensure_worker()


def register_handler(kind: str, handler: Callable) -> None:
    """Register a task handler function.

    Handler receives (params: dict) and returns result dict.
    """
    _HANDLERS[kind] = handler


def submit_task(kind: str, params: dict, max_retries: int = MAX_RETRIES) -> str:
    """Submit a background task. Returns task_id."""
    _require_init()  # P2-Q3-fix(L399)
    task_id = uuid.uuid4().hex[:16]
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tasks (id, kind, params, status, created, max_retries) VALUES (?, ?, ?, 'pending', ?, ?)",
            (task_id, kind, json.dumps(params, ensure_ascii=False), now, max_retries),
        )
    _ensure_worker()
    return task_id


def resubmit_failed(kind: str | None = None, max_tasks: int = 10) -> int:
    """Resubmit failed/retryable tasks."""
    _require_init()  # P2-Q3-fix(L399)
    count = 0
    with _connect() as conn:
        now = time.time()
        # V11 审计修复（Medium）: 原实现不检查 next_retry_at，把仍在退避期的
        # failed 任务立即置 pending 重跑，与 worker 自动重试双重消耗 retry_count。
        # 修正: 只重提已过退避期（next_retry_at <= now）的任务。
        where = "status='failed' AND retry_count < max_retries AND (next_retry_at IS NULL OR next_retry_at <= ?)"
        args: list[Any] = [now]
        if kind:
            where += " AND kind=?"
            args.append(kind)
        rows = conn.execute(
            f"SELECT id FROM tasks WHERE {where} LIMIT ?",
            (*args, max_tasks),
        ).fetchall()
        for (task_id,) in rows:
            conn.execute(
                "UPDATE tasks SET status='pending', error=NULL, completed=NULL, next_retry_at=?, retry_count=retry_count+1 WHERE id=?",
                (now + RETRY_BACKOFF_BASE, task_id),
            )
            count += 1
    if count:
        _ensure_worker()
    return count


def get_task_result(task_id: str) -> Optional[dict]:
    """Get task result. Returns dict with status and result/error."""
    _require_init()  # P2-Q3-fix(L399)
    with _connect() as conn:
        row = conn.execute(
            "SELECT status, result, error, created, started, completed, retry_count, max_retries FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    if not row:
        return {"error": "task not found"}
    status, result_json, error, created, started, completed, retries, max_r = row
    base = {
        "status": status,
        "created": created,
        "started": started,
        "completed": completed,
        "elapsed": round(completed - started, 1) if completed and started else None,
        "retry_count": retries,
        "max_retries": max_r,
    }
    if status == "completed" and result_json:
        base["result"] = json.loads(result_json)
    elif status in {"failed", "timed_out"} and error:
        base["error"] = error
    return base


def list_tasks(kind: str | None = None, limit: int = 20) -> list[dict]:
    """List recent tasks."""
    _require_init()  # P2-Q3-fix(L399)
    with _connect() as conn:
        if kind:
            rows = conn.execute(
                "SELECT id, kind, status, created, started, completed, retry_count, max_retries FROM tasks WHERE kind=? ORDER BY created DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, status, created, started, completed, retry_count, max_retries FROM tasks ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [
        {
            "id": r[0],
            "kind": r[1],
            "status": r[2],
            "created": r[3],
            "started": r[4],
            "completed": r[5],
            "retry_count": r[6],
            "max_retries": r[7],
        }
        for r in rows
    ]


def get_queue_stats() -> dict:
    """返回任务队列看板统计。"""
    _require_init()  # P2-Q3-fix(L399)
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        by_status = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        }
        by_kind = [
            {"kind": r[0], "count": r[1]}
            for r in conn.execute(
                "SELECT kind, COUNT(*) FROM tasks GROUP BY kind ORDER BY COUNT(*) DESC"
            ).fetchall()
        ]
        avg_elapsed = conn.execute(
            "SELECT AVG(completed - started) FROM tasks WHERE status='completed' AND completed IS NOT NULL AND started IS NOT NULL"
        ).fetchone()[0]
        recent_failures = conn.execute(
            "SELECT id, kind, error, completed, retry_count FROM tasks WHERE status IN ('failed', 'timed_out') ORDER BY completed DESC LIMIT 10"
        ).fetchall()
    return {
        "total": total,
        "by_status": by_status,
        "by_kind": by_kind,
        "avg_elapsed_seconds": round(avg_elapsed, 1) if avg_elapsed else None,
        "recent_failures": [
            {"id": r[0], "kind": r[1], "error": (r[2] or "")[:200], "completed": r[3], "retry_count": r[4]}
            for r in recent_failures
        ],
        "pending": by_status.get("pending", 0),  # P2-Q3-fix(L398): 无任何路径产生 retry_pending 状态，删除死代码
        "running": by_status.get("running", 0),
        "completed": by_status.get("completed", 0),
        "failed": by_status.get("failed", 0),
        "timed_out": by_status.get("timed_out", 0),
        "config": {
            "max_concurrent": MAX_CONCURRENT,
            "task_timeout": TASK_TIMEOUT,
            "max_retries": MAX_RETRIES,
        },
    }


def cleanup_old_tasks(max_age_hours: int = CLEANUP_AGE_HOURS) -> int:
    """Remove completed/failed tasks older than max_age_hours."""
    _require_init()  # P2-Q3-fix(L399)
    cutoff = time.time() - max_age_hours * 3600
    with _connect() as conn:
        conn.execute("DELETE FROM tasks WHERE status IN ('completed', 'failed', 'timed_out') AND completed IS NOT NULL AND completed < ?", (cutoff,))
        deleted = conn.total_changes
    logger.info("Cleaned %d old tasks (age > %dh)", deleted, max_age_hours)
    return deleted


def _ensure_worker() -> None:
    """P2-Q3-fix(M394): 维持 MAX_CONCURRENT 个 worker 线程，实现真正的并发上限。"""
    global _WORKERS, _RUNNING
    _RUNNING = True
    while len(_WORKERS) < MAX_CONCURRENT:
        t = threading.Thread(target=_worker_loop, daemon=True,
                             name=f"task-worker-{len(_WORKERS)}")
        t.start()
        _WORKERS.append(t)
    for i, t in enumerate(_WORKERS):
        if not t.is_alive():
            t = threading.Thread(target=_worker_loop, daemon=True,
                                 name=f"task-worker-{i}")
            t.start()
            _WORKERS[i] = t


def _claim_next_task() -> Optional[tuple[str, str, dict]]:
    """P2-Q3-fix(M394): 原子认领下一个 pending 任务。

    多 worker 下先 SELECT 再 UPDATE 会抢到同一任务；这里用
    ``UPDATE ... WHERE id=? AND status='pending'`` 的条件更新保证
    只有一个 worker 认领成功（rowcount==1）。
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, kind, params FROM tasks WHERE status='pending' ORDER BY created ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        task_id, kind, params_json = row
        cur = conn.execute(
            "UPDATE tasks SET status='running', started=? WHERE id=? AND status='pending'",
            (time.time(), task_id),
        )
        if cur.rowcount == 0:
            return None  # 已被其它 worker 认领
        try:
            return task_id, kind, json.loads(params_json)
        except (json.JSONDecodeError, TypeError):
            # params 损坏：任务已标记 running，直接置为 failed，避免永久滞留
            conn.execute(
                "UPDATE tasks SET status='failed', error=?, completed=? WHERE id=?",
                (f"Corrupted task params: {params_json[:100]}", time.time(), task_id),
            )
            return None


def _worker_loop() -> None:
    while _RUNNING:
        acquired = False  # P2-Q3-fix(M394): 仅在成功 acquire 后才 release
        try:
            if _DB is None:
                time.sleep(0.5)
                continue

            # 1) 检查定时重试任务
            now = time.time()
            with _connect() as conn:
                retry_rows = conn.execute(
                    "SELECT id FROM tasks WHERE status='failed' AND retry_count < max_retries AND next_retry_at IS NOT NULL AND next_retry_at <= ?",
                    (now,),
                ).fetchall()
                for (tid,) in retry_rows:
                    conn.execute(
                        "UPDATE tasks SET status='pending', error=NULL, completed=NULL, started=NULL WHERE id=?",
                        (tid,),
                    )

            # 2) 并发上限控制（先占信号量再认领任务）
            if not _PENDING_SEM.acquire(blocking=False):
                time.sleep(0.5)
                continue
            acquired = True

            # 3) 原子认领一个 pending 任务
            task = _claim_next_task()
            if task is None:
                # 没有空闲任务 → 清理 + sleep（信号量在 finally 释放）
                _periodic_cleanup()
                time.sleep(1)
                continue

            task_id, kind, params = task

            # 4) 执行
            _execute_with_timeout(task_id, kind, params)

        except Exception as exc:
            logger.error("Task worker error: %s", exc)
            time.sleep(2)
        finally:
            if acquired:
                try:
                    _PENDING_SEM.release()
                except ValueError:
                    pass


def _execute_with_timeout(task_id: str, kind: str, params: dict) -> None:
    """在子线程中执行 handler，带超时保护与失败自动重试调度。"""
    handler = _HANDLERS.get(kind)
    if handler is None:
        with _connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='failed', error=?, completed=? WHERE id=?",
                (f"No handler registered for kind: {kind}", time.time(), task_id),
            )
        return

    # P2-Q3-fix(L397): 记录本任务实际执行起点，完成耗时不再恒为 0
    exec_start = time.time()
    _result_holder: list[Any] = []
    _exc_holder: list[Exception] = []
    _exc_tb: list[str] = []

    def _run():
        try:
            _result_holder.append(handler(params))
        except Exception as e:
            _exc_holder.append(e)
            # P2-Q3-fix(L396): 在异常发生的子线程内捕获 traceback；
            # 否则主线程调用 traceback.format_exc() 拿到的是 "NoneType: None"
            _exc_tb.append(traceback.format_exc())

    _t = threading.Thread(target=_run, daemon=True, name=f"task-exec-{task_id[:8]}")
    _t.start()
    _t.join(timeout=TASK_TIMEOUT_BY_KIND.get(kind, TASK_TIMEOUT))

    now = time.time()

    with _connect() as conn:
        row = conn.execute(
            "SELECT retry_count, max_retries FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        retry_count, max_retries = row

    # ── 超时 ──
    if _t.is_alive():
        # Python 线程无法安全强杀；旧 handler 仍可能带副作用继续运行。
        # 将任务置为不可自动重试的终态，避免同一任务并发执行多份。
        with _connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='timed_out', error=?, completed=?, retry_count=retry_count+1, next_retry_at=NULL WHERE id=?",
                (f"Task timed out after {TASK_TIMEOUT}s; automatic retry disabled because the worker thread may still be running", now, task_id),
            )
            logger.error("Task %s timed out; automatic retry disabled to avoid duplicate execution", task_id[:12])
        return

    # ── 异常 ──
    if _exc_holder:
        # P2-Q3-fix(L396): 使用子线程内捕获的 traceback 字符串
        tb = (_exc_tb[0] if _exc_tb else traceback.format_exc())[-500:]
        with _connect() as conn:
            if retry_count < max_retries:
                delay = RETRY_BACKOFF_BASE * (RETRY_BACKOFF_MULT ** retry_count)
                conn.execute(
                    "UPDATE tasks SET status='failed', error=?, completed=?, retry_count=retry_count+1, next_retry_at=? WHERE id=?",
                    (f"{_exc_holder[0]} | {tb}", now, now + delay, task_id),
                )
                logger.warning("Task %s failed with %s, retry %d/%d in %.0fs",
                               task_id[:12], _exc_holder[0], retry_count + 1, max_retries, delay)
            else:
                conn.execute(
                    "UPDATE tasks SET status='failed', error=?, completed=? WHERE id=?",
                    (f"{_exc_holder[0]} | {tb}", now, task_id),
                )
                logger.error("Task %s failed, %d retries exhausted: %s", task_id[:12], max_retries, _exc_holder[0])
        return

    # ── 成功 ──
    result = _result_holder[0]
    with _connect() as conn:
        conn.execute(
            "UPDATE tasks SET status='completed', result=?, completed=?, next_retry_at=NULL WHERE id=?",
            (json.dumps(result, ensure_ascii=False, default=str), now, task_id),
        )
    # P2-Q3-fix(L397): 此前 row 只有 2 列(retry_count,max_retries)，恒走 else → 0.0s；
    # 改用 exec_start 计算真实耗时。
    logger.info("Task %s completed in %.1fs", task_id[:12], now - exec_start)


_last_cleanup = 0.0

def _periodic_cleanup() -> None:
    """每 30 秒最多清理一次过期任务 + 重试超时任务。"""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < 30:
        return
    _last_cleanup = now
    cleanup_old_tasks(CLEANUP_AGE_HOURS)

    # 重试 pending 超过 12h 的任务标记为死信
    with _connect() as conn:
        stale = conn.execute(
            "SELECT id FROM tasks WHERE status='pending' AND created < ?",
            (now - 12 * 3600,),
        ).fetchall()
        for (tid,) in stale:
            conn.execute(
                "UPDATE tasks SET status='failed', error='Stale (pending >12h)', completed=? WHERE id=?",
                (now, tid),
            )


def shutdown() -> None:
    """Stop the worker thread."""
    global _RUNNING
    _RUNNING = False
    logger.info("Task worker shutdown requested")
