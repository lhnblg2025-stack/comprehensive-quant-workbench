"""task_dag 轻量 DAG 调度器测试。

真 subprocess 跑 echo/false/true（python3 -c exit(N)），禁网络；重试/redo 用
monkeypatch 模拟 subprocess.run 与时间，避免真实等待。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_system.task_dag as td


class FakeProc:
    """模拟 subprocess.CompletedProcess。"""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_run(monkeypatch, codes):
    """按调用顺序返回 returncode 的 fake subprocess.run，并记录调用次数。"""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        code = codes[min(len(calls) - 1, len(codes) - 1)]
        return FakeProc(code)

    monkeypatch.setattr(td.subprocess, "run", fake_run)
    return calls


def test_linear_deps_all_success():
    """线性依赖 a→b→c 全部成功。"""
    dag = td.TaskDAG("pipe")
    dag.add_task("a", "python3 -c 'exit(0)'")
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    dag.add_task("c", "python3 -c 'exit(0)'", deps=("b",))
    results = dag.run()
    assert set(results) == {"a", "b", "c"}
    for r in results.values():
        assert r["status"] == "success"
        assert r["exit_code"] == 0
        assert r["skipped_dep"] is None


def test_parent_fail_child_skipped():
    """父任务失败 → 子任务 skipped_dep_failed 并记录原因。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(1)'")
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    dag.add_task("c", "python3 -c 'exit(0)'", deps=("b",))
    results = dag.run()
    assert results["a"]["status"] == "failed"
    assert results["a"]["exit_code"] == 1
    assert results["b"]["status"] == "skipped_dep_failed"
    assert results["b"]["skipped_dep"] == "a"
    assert results["b"]["error"] == "因依赖 a 失败而跳过"
    assert results["c"]["status"] == "skipped_dep_failed"
    assert results["c"]["skipped_dep"] == "b"
    assert results["c"]["error"] == "因依赖 b 失败而跳过"


def test_timeout_parent_child_skipped():
    """父任务超时 → status timeout，子任务挂起。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'import time; time.sleep(5)'", timeout_s=1)
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    results = dag.run()
    assert results["a"]["status"] == "timeout"
    assert results["a"]["exit_code"] is None
    assert "timeout" in results["a"]["error"]
    assert results["b"]["status"] == "skipped_dep_failed"
    assert results["b"]["skipped_dep"] == "a"


def test_retries_fail_then_success(monkeypatch):
    """失败重试：第一次失败第二次成功（monkeypatch subprocess.run）。"""
    calls = _install_fake_run(monkeypatch, codes=[1, 0])
    monkeypatch.setattr(td, "_sleep", lambda seconds: None)
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(1)'", retries=1)
    results = dag.run()
    assert len(calls) == 2
    assert results["a"]["status"] == "success"
    assert results["a"]["exit_code"] == 0


def test_cycle_raises_and_no_execution(monkeypatch):
    """a→b→a 成环 → DAGCycleError，且不执行任何任务。"""
    calls = _install_fake_run(monkeypatch, codes=[0])
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(0)'", deps=("b",))
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    with pytest.raises(td.DAGCycleError):
        dag.run()
    assert calls == []


def test_parallel_two_independent_tasks():
    """max_parallel=2 时两个独立任务并行完成。"""
    dag = td.TaskDAG()
    dag.add_task("x", "python3 -c 'import time; time.sleep(1.5)'", timeout_s=10)
    dag.add_task("y", "python3 -c 'import time; time.sleep(1.5)'", timeout_s=10)
    start = time.monotonic()
    results = dag.run(max_parallel=2)
    elapsed = time.monotonic() - start
    assert results["x"]["status"] == "success"
    assert results["y"]["status"] == "success"
    assert elapsed < 2.5, f"串行执行约 3s，实测 {elapsed:.2f}s，应并行"


def test_result_structure():
    """结果结构包含 status/exit_code/duration_s/skipped_dep/error 字段。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(0)'")
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    results = dag.run()
    assert set(results) == {"a", "b"}
    for r in results.values():
        assert set(("status", "exit_code", "duration_s", "skipped_dep", "error")) <= set(r)
        assert r["status"] in ("success", "failed", "timeout", "skipped_dep_failed")
        assert isinstance(r["duration_s"], (int, float))
    assert results["a"]["exit_code"] == 0
    assert results["b"]["skipped_dep"] is None


def test_run_until_redo_before_deadline(monkeypatch):
    """deadline 未到且首轮失败 → 整 DAG 重拉，最终成功。"""
    calls = _install_fake_run(monkeypatch, codes=[1, 0])
    monkeypatch.setattr(td, "_now", lambda: 1000.0)
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(1)'")
    results = dag.run_until(deadline_ts=2000.0)
    assert len(calls) == 2, "首轮失败后应在 deadline 前重拉一次"
    assert results["a"]["status"] == "success"
    assert results["a"]["exit_code"] == 0


def test_run_until_deadline_passed_no_redo(monkeypatch):
    """deadline 已过 → 不再整 DAG 重拉，保留失败结果。"""
    calls = _install_fake_run(monkeypatch, codes=[1])
    monkeypatch.setattr(td, "_now", lambda: 5000.0)
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(1)'")
    results = dag.run_until(deadline_ts=1000.0)
    assert len(calls) == 1
    assert results["a"]["status"] == "failed"
    assert results["a"]["exit_code"] == 1


def test_render_report_contains_table_rows():
    """render_report 输出 markdown 表格行与跳过原因。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(1)'")
    dag.add_task("b", "python3 -c 'exit(0)'", deps=("a",))
    report = td.render_report(dag.run())
    assert "| task | status | duration_s | skip_reason |" in report
    assert "| a | failed |" in report
    assert "| b | skipped_dep_failed |" in report
    assert "因依赖 a 失败而跳过" in report


def test_add_task_duplicate_raises():
    """重复 task_id 直接报错。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(0)'")
    with pytest.raises(ValueError):
        dag.add_task("a", "python3 -c 'exit(0)'")


def test_unknown_dep_raises(monkeypatch):
    """依赖引用未知任务 → ValueError，且不执行。"""
    calls = _install_fake_run(monkeypatch, codes=[0])
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'exit(0)'", deps=("ghost",))
    with pytest.raises(ValueError):
        dag.run()
    assert calls == []


def test_log_dir_writes_task_log(tmp_path):
    """log_dir 下为每个任务写入日志文件。"""
    dag = td.TaskDAG()
    dag.add_task("a", "python3 -c 'print(\"hello\")'")
    dag.run(log_dir=str(tmp_path))
    log_file = tmp_path / "a.log"
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")
