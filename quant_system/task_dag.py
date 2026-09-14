"""轻量 DAG 调度器：任务依赖图，父失败子挂起。

解决 crontab/schtasks 纯时间触发下"父任务失败/超时，子任务仍准时启动并用旧数据
跑出伪新鲜报告"的问题：子任务仅在父任务退出码为 0 时启动；父任务失败或超时则
子任务挂起（``skipped_dep_failed``）并明确记录原因。

示例::

    dag = TaskDAG("pipeline")
    dag.add_task("a", "python3 -c exit(0)")
    dag.add_task("b", "python3 -c exit(0)", deps=("a",))
    results = dag.run()
    print(render_report(results))

CLI::

    python -m quant_system.task_dag --dag-file dag.json --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_now = time.time
_sleep = time.sleep


class DAGCycleError(ValueError):
    """DAG 存在环，拒绝执行。"""


class TaskSpec:
    """单个任务定义。"""

    __slots__ = ("task_id", "cmd", "deps", "timeout_s", "retries", "env")

    def __init__(self, task_id, cmd, deps, timeout_s, retries, env):
        self.task_id = task_id
        self.cmd = cmd
        self.deps = deps
        self.timeout_s = timeout_s
        self.retries = retries
        self.env = env


class TaskDAG:
    """任务依赖图调度器。

    - 拓扑排序（Kahn 算法）按依赖层级执行，有环抛 :class:`DAGCycleError` 且不执行；
    - 子任务仅在父任务退出码为 0 时启动，父失败/超时/挂起则子任务跳过；
    - 退出码非 0 为失败，可重试 ``retries`` 次（每次间隔 2s）；
    - ``run_until`` 在截止时间前反复整 DAG 重拉，直到全部成功或时间到。
    """

    def __init__(self, name: str = "dag"):
        self.name = name
        self._tasks: Dict[str, TaskSpec] = {}

    def add_task(
        self,
        task_id: str,
        cmd,
        deps: Sequence[str] = (),
        timeout_s: float = 300,
        retries: int = 0,
        env: Optional[dict] = None,
    ) -> "TaskDAG":
        """注册任务；``deps`` 为其依赖（父任务）id 列表。"""
        if task_id in self._tasks:
            raise ValueError(f"duplicate task id: {task_id}")
        self._tasks[task_id] = TaskSpec(
            task_id, cmd, tuple(deps), timeout_s, int(retries), env
        )
        return self

    # ------------------------------------------------------------------ #

    def _layers(self) -> List[List[str]]:
        """Kahn 拓扑排序，按依赖分层；有环抛 DAGCycleError，未知依赖抛 ValueError。"""
        children: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        in_deg: Dict[str, int] = {tid: 0 for tid in self._tasks}
        for tid, spec in self._tasks.items():
            for dep in spec.deps:
                if dep not in self._tasks:
                    raise ValueError(f"task {tid} 引用未知依赖 {dep}")
                children[dep].append(tid)
                in_deg[tid] += 1

        layers: List[List[str]] = []
        frontier = [tid for tid, deg in in_deg.items() if deg == 0]
        while frontier:
            layers.append(frontier)
            nxt: List[str] = []
            for tid in frontier:
                for child in children[tid]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        nxt.append(child)
            frontier = nxt
        if sum(len(layer) for layer in layers) != len(self._tasks):
            raise DAGCycleError(
                f"检测到环，涉及 {len(self._tasks) - sum(len(l) for l in layers)} 个任务"
            )
        return layers

    def _first_bad_dep(self, task_id: str, results: dict) -> Optional[str]:
        """返回第一个未成功（失败/超时/跳过）的依赖 id，全部成功则 None。"""
        for dep in self._tasks[task_id].deps:
            dep_result = results.get(dep)
            if dep_result is not None and dep_result["status"] != "success":
                return dep
        return None

    def _run_one(self, spec: TaskSpec, log_dir: Optional[str]) -> dict:
        """执行单个任务（含重试），返回结果 dict。"""
        start = _now()
        status, exit_code, error = "success", None, None
        output = ""
        for attempt in range(spec.retries + 1):
            try:
                proc = subprocess.run(
                    spec.cmd,
                    shell=isinstance(spec.cmd, str),
                    capture_output=True,
                    text=True,
                    timeout=spec.timeout_s,
                    env=spec.env,
                )
                exit_code = proc.returncode
                output = _as_text(getattr(proc, "stdout", None)) + _as_text(
                    getattr(proc, "stderr", None)
                )
                if exit_code == 0:
                    status, error = "success", None
                    break
                status, error = "failed", f"exit code {exit_code}"
            except subprocess.TimeoutExpired as exc:
                status, error = "timeout", f"timeout after {spec.timeout_s}s"
                output = _as_text(getattr(exc, "stdout", None)) + _as_text(
                    getattr(exc, "stderr", None)
                )
                break
            except OSError as exc:
                status, error = "failed", str(exc)
                output = ""
                break
            if attempt < spec.retries:
                _sleep(2)
        if log_dir:
            log_path = Path(log_dir) / f"{spec.task_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"[{status}] {error or ''}\n{output}", encoding="utf-8"
            )
        return {
            "status": status,
            "exit_code": exit_code,
            "duration_s": round(_now() - start, 3),
            "skipped_dep": None,
            "error": error,
        }

    def run(self, max_parallel: int = 1, log_dir: Optional[str] = None) -> dict:
        """按依赖层级执行整个 DAG，返回 ``{task_id: result}``。"""
        max_parallel = max(1, int(max_parallel))
        layers = self._layers()
        results: dict = {}
        for layer in layers:
            ready: List[str] = []
            for task_id in layer:
                bad = self._first_bad_dep(task_id, results)
                if bad is None:
                    ready.append(task_id)
                else:
                    results[task_id] = {
                        "status": "skipped_dep_failed",
                        "exit_code": None,
                        "duration_s": 0.0,
                        "skipped_dep": bad,
                        "error": f"因依赖 {bad} 失败而跳过",
                    }
            if not ready:
                continue
            workers = min(max_parallel, len(ready))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_one, self._tasks[tid], log_dir): tid
                    for tid in ready
                }
                for future in futures:
                    results[futures[future]] = future.result()
        return results

    def run_until(
        self,
        deadline_ts: float,
        max_parallel: int = 1,
        log_dir: Optional[str] = None,
    ) -> dict:
        """截止时间前所有任务尝试；首轮失败且 deadline 未到则整 DAG 重拉（redo）。"""
        results: dict = {}
        while True:
            results = self.run(max_parallel=max_parallel, log_dir=log_dir)
            if (
                all(r["status"] == "success" for r in results.values())
                or _now() >= deadline_ts
            ):
                break
        return results


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def render_report(results: dict) -> str:
    """把 run() 结果渲染成 markdown 表格（task/status/duration_s/skip_reason）。"""
    lines = [
        "| task | status | duration_s | skip_reason |",
        "|------|--------|------------|-------------|",
    ]
    for task_id, result in results.items():
        if result["status"] == "skipped_dep_failed":
            reason = f"因依赖 {result['skipped_dep']} 失败而跳过"
        else:
            reason = result["error"] or ""
        lines.append(
            f"| {task_id} | {result['status']} | "
            f"{result['duration_s']:.2f} | {reason} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI：--dag-file dag.json / --run-until / --max-parallel。"""
    parser = argparse.ArgumentParser(
        description="轻量 DAG 调度器：父任务失败则子任务挂起"
    )
    parser.add_argument(
        "--dag-file",
        required=True,
        help='dag.json，形如 {"tasks": {"a": {"cmd": "...", "deps": [], "timeout_s": 30}}}',
    )
    parser.add_argument(
        "--run-until",
        type=float,
        default=None,
        help="epoch 秒截止时间；截止前有失败则整 DAG 重拉",
    )
    parser.add_argument("--max-parallel", type=int, default=1, help="最大并行任务数")
    parser.add_argument("--log-dir", default=None, help="任务输出日志目录")
    args = parser.parse_args(argv)

    with open(args.dag_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    dag = TaskDAG(name=data.get("name", "dag"))
    for task_id, spec in data.get("tasks", {}).items():
        dag.add_task(
            task_id,
            spec["cmd"],
            deps=spec.get("deps", []),
            timeout_s=spec.get("timeout_s", 300),
            retries=spec.get("retries", 0),
            env=spec.get("env"),
        )

    if args.run_until is not None:
        results = dag.run_until(
            args.run_until, max_parallel=args.max_parallel, log_dir=args.log_dir
        )
    else:
        results = dag.run(max_parallel=args.max_parallel, log_dir=args.log_dir)

    print(render_report(results))
    return 0 if all(r["status"] == "success" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
