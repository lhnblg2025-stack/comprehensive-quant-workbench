#!/usr/bin/env python3
"""牧云天枢 · 服务看门狗
自动检测 quant_web server 健康状态，僵死/崩溃时自动重启。
每60秒检查一次 /api/livez，连续3次失败才通过唯一启动入口重启。
"""
from __future__ import annotations

import os
import sys
import time
import signal
import logging
import subprocess
import fcntl
from pathlib import Path
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT)
SERVER_SCRIPT = os.path.join(ROOT, "server.py")
START_SCRIPT = os.path.join(WORKSPACE_ROOT, "start_quant.sh")
# 服务只能由 start_quant.sh 统一拉起；它使用与当前 quant_web 数据依赖
# 一致的 Anaconda Python，避免 watchdog 与手工启动产生第二套运行时。
PYTHON_BIN = os.environ.get("QUANT_PYTHON", sys.executable)
LOCK_FILE = os.environ.get("QUANT_WATCHDOG_LOCK", os.path.join(WORKSPACE_ROOT, "generated", "run", "watchdog.lock"))
# 与 start_quant.sh 使用同一端口环境变量，避免 QUANT_WEB_PORT 非 8600 时
# 看门狗检查/重启验证打到错误端口并形成循环重启。
PORT = int(os.environ.get("QUANT_WEB_PORT", "8600"))
# 看门狗只负责判断进程是否存活，不把可选预热状态当作崩溃。
HEALTH_URL = f"http://127.0.0.1:{PORT}/api/livez"
CHECK_INTERVAL = 60   # 秒
MAX_FAILURES = 3       # 连续失败次数触发重启

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "watchdog.log"), mode="a"),
    ],
)


def is_alive() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def find_server_pid() -> int | None:
    """找到当前运行的 server.py 进程 PID"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"python.*server\\.py.*{PORT}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        # 排除自己（watchdog）和 grep
        for pid_str in pids:
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            try:
                cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", errors="replace")
                if "server.py" in cmdline and str(PORT) in cmdline:
                    return pid
            except (IOError, OSError):
                continue
        return None
    except Exception:
        return None


def restart_server() -> None:
    """Delegate restart to the single owner, preserving one PID/log contract."""
    logging.warning("Restarting through %s", START_SCRIPT)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # 明确把看门狗端口传给启动脚本，保持检查端口与重启端口一致。
    if "QUANT_WEB_PORT" not in env:
        env["QUANT_WEB_PORT"] = str(PORT)
    proc = subprocess.run(
        ["bash", START_SCRIPT],
        cwd=WORKSPACE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.stdout:
        logging.info("start_quant: %s", proc.stdout.strip().replace("\n", " | ")[-1000:])
    if proc.stderr:
        logging.warning("start_quant stderr: %s", proc.stderr.strip().replace("\n", " | ")[-1000:])
    if proc.returncode != 0:
        logging.error("Unified server restart failed with exit=%s", proc.returncode)
        return
    if is_alive():
        logging.info("Server liveness check PASSED after restart.")
    else:
        logging.error("Server liveness check FAILED after restart!")


def main_loop() -> None:
    Path(LOCK_FILE).parent.mkdir(parents=True, exist_ok=True)
    lock_handle = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.error("已有 watchdog 实例运行，当前实例退出")
        return
    logging.info("=== 牧云天枢 看门狗启动 ===")
    logging.info(f"健康检查: {HEALTH_URL}, 每{CHECK_INTERVAL}s, 连续{MAX_FAILURES}次失败重启")

    failures = 0
    while True:
        alive = is_alive()
        if alive:
            if failures > 0:
                logging.info(f"服务已恢复 (连续失败{failures}次后)")
            failures = 0
        else:
            failures += 1
            logging.warning(f"健康检查失败 ({failures}/{MAX_FAILURES})")
            if failures >= MAX_FAILURES:
                logging.error(f"连续{failures}次失败，触发重启")
                restart_server()
                failures = 0

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("看门狗收到退出信号")
        sys.exit(0)
