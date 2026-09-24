"""scripts/chaos_drill.py 三断混沌演练测试。

测试先行的同提交约束：
  - 通过 importlib 从 workspace/scripts/chaos_drill.py 加载脚本，避免污染包结构。
  - 全部故障注入与输出目录使用 mock/tmp_path，绝不真实断网、删文件或触达飞书。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKSPACE / "scripts" / "chaos_drill.py"


def _load_cd():
    spec = importlib.util.spec_from_file_location("chaos_drill_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cd():
    return _load_cd()


def _fail_then_ok(failures: int, payload=None):
    """返回一个连续失败 failures 次后成功的网络 fetcher。"""
    state = {"calls": 0}

    def fetcher():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise TimeoutError("mock network timeout")
        return payload or {"source": "primary", "rows": 10}

    return fetcher, state


class _BackoffRecorder:
    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


def test_source_primary_raises_fallback_used(cd):
    result = cd.run_source_drill(
        primary_fetcher=lambda: (_ for _ in ()).throw(ConnectionError("primary down")),
        fallback_fetcher=lambda: {"source": "fallback", "rows": 42},
    )

    assert result.passed is True
    assert "primary" in result.path
    assert "fallback" in result.path
    assert "unavailable" not in result.path
    assert result.detail["active_chain"] == ["primary", "fallback"]


def test_source_degradation_chain_marks_unavailable(cd):
    result = cd.run_source_drill(
        primary_fetcher=lambda: (_ for _ in ()).throw(ConnectionError("primary down")),
        fallback_fetcher=lambda: {"source": "fallback", "rows": 1},
    )

    assert "unavailable" in result.detail["degradation_chain"]
    assert "unavailable" not in result.detail["active_chain"]


def test_source_both_fail_marks_unavailable_and_fails(cd):
    def boom():
        raise RuntimeError("all sources down")

    result = cd.run_source_drill(primary_fetcher=boom, fallback_fetcher=boom)

    assert result.passed is False
    assert result.path.endswith("unavailable")
    assert result.rpo["type"] == "full_outage"
    assert "unavailable" in result.detail["active_chain"]


def test_db_corrupt_parquet_impute_does_not_crash(cd):
    def corrupt_read():
        raise OSError("mock parquet corruption")

    def impute(error):
        assert isinstance(error, OSError)
        return [{"symbol": "MOCK", "close": None, "imputed": True}]

    result = cd.run_db_drill(read_frame=corrupt_read, impute_frame=impute)

    assert result.passed is True
    assert "impute" in result.path
    assert result.rpo["imputed_rows"] == 1
    assert "read_degraded" in result.detail["degradation_chain"]


def test_db_corrupt_parquet_none_fallback_is_allowed(cd):
    def corrupt_read():
        raise OSError("corrupt")

    result = cd.run_db_drill(read_frame=corrupt_read, impute_frame=lambda error: None)

    assert result.passed is True
    assert result.path == "parquet -> read_degraded"
    assert result.rpo["mode"] == "none"
    assert result.rpo["imputed_rows"] == 0


def test_db_imputer_also_fails_returns_failure(cd):
    def corrupt_read():
        raise OSError("corrupt")

    def broken_imputer(error):
        raise RuntimeError("imputer unavailable")

    result = cd.run_db_drill(read_frame=corrupt_read, impute_frame=broken_imputer)

    assert result.passed is False
    assert result.path.endswith("unavailable")


def test_net_timeout_retry_backoff_degrade(cd):
    def always_timeout():
        raise TimeoutError("network down")

    recorder = _BackoffRecorder()
    result = cd.run_net_drill(
        fetcher=always_timeout,
        degraded_fetcher=lambda: {"source": "cache", "lag_seconds": 30},
        backoff=recorder,
        delays=[0.1, 0.2],
    )

    assert result.passed is True
    assert result.detail["retry_count"] == 3
    assert recorder.delays == [0.1, 0.2]
    assert "retry" in result.path
    assert "degrade" in result.path
    assert result.rpo["type"] == "cache_lag"


def test_net_success_after_timeout_skips_degrade(cd):
    fetcher, state = _fail_then_ok(1)
    recorder = _BackoffRecorder()

    result = cd.run_net_drill(
        fetcher=fetcher,
        degraded_fetcher=lambda: {"source": "unused"},
        backoff=recorder,
        delays=[0.01, 0.02],
    )

    assert result.passed is True
    assert "degrade" not in result.path
    assert "success" in result.path
    assert result.detail["retry_count"] == 1
    assert recorder.delays == [0.01]


def test_net_all_fail_degrade_fails(cd):
    def timeout():
        raise TimeoutError("network down")

    recorder = _BackoffRecorder()
    result = cd.run_net_drill(
        fetcher=timeout,
        degraded_fetcher=lambda: (_ for _ in ()).throw(RuntimeError("cache empty")),
        backoff=recorder,
        delays=[0.01, 0.02],
    )

    assert result.passed is False
    assert "unavailable" in result.path
    assert "unavailable" in result.detail["active_chain"]


def test_rto_rpo_record_structure(cd, tmp_path):
    report = cd.run_all("2026-08-13", only="db", output_root=tmp_path)

    assert report["summary"]["all_passed"] is True
    drill = report["drills"][0]
    assert isinstance(drill["rto_ms"], int)
    assert drill["rto_ms"] >= 0
    assert isinstance(drill["rpo"], dict)
    assert "type" in drill["rpo"]
    assert drill["dry_run"] is True


def test_only_filter_runs_selected_drill(cd, tmp_path):
    report = cd.run_all("2026-08-13", only="source", output_root=tmp_path)

    assert len(report["drills"]) == 1
    assert report["drills"][0]["key"] == "source"


def test_run_all_writes_json_and_md_reports(cd, tmp_path):
    report = cd.run_all("2026-08-13", output_root=tmp_path)

    out_dir = tmp_path / "20260813"
    json_path = out_dir / "chaos_2026-08-13.json"
    md_path = out_dir / "chaos_2026-08-13.md"
    assert json_path.exists()
    assert md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["date"] == "2026-08-13"
    assert set(data["drills"][0]["rpo"].keys()) == {"type", "value", "note"}
    md_text = md_path.read_text(encoding="utf-8")
    assert "免疫正常" in md_text
    assert "RTO(ms)" in md_text


def test_dry_run_has_no_unlink_or_network_side_effect(cd, tmp_path, monkeypatch):
    unlink_calls = []
    monkeypatch.setattr(Path, "unlink", lambda self, **kwargs: unlink_calls.append(str(self)))

    report = cd.run_all("2026-08-13", output_root=tmp_path)

    assert report["dry_run"] is True
    assert unlink_calls == []
    assert (tmp_path / "20260813" / "chaos_2026-08-13.json").exists()


def test_failure_summary_reports_not_immune(cd, tmp_path):
    report = cd.run_all(
        "2026-08-13",
        only="db",
        output_root=tmp_path,
        injections={
            "db": {
                "read_frame": lambda: (_ for _ in ()).throw(OSError("corrupt")),
                "impute_frame": lambda error: (_ for _ in ()).throw(RuntimeError("down")),
            }
        },
    )

    assert report["summary"]["all_passed"] is False
    assert report["summary"]["message"] != "免疫正常"
    assert "失败" in report["summary"]["message"]


def test_failure_alert_reuses_feishu(cd, monkeypatch):
    sent = []
    monkeypatch.setattr(cd, "_send_feishu_text", lambda text, title=None: sent.append((text, title)))

    report = {
        "date": "2026-08-13",
        "summary": {"all_passed": False, "message": "存在失败演练", "failed": 1},
        "drills": [
            {"key": "source", "passed": False, "path": "primary -> fallback -> unavailable",
             "rto_ms": 1, "error": "boom"}
        ],
    }
    cd.send_failure_alert(report)

    assert sent
    assert "Chaos Drill" in sent[0][1]
    assert "存在失败演练" in sent[0][0]


def test_invalid_only_rejected(cd):
    with pytest.raises(ValueError):
        cd.select_drills("not_a_drill")


def test_date_folder_uses_yyyymmdd(cd, tmp_path):
    cd.write_reports(
        {
            "date": "2026-08-13",
            "generated_at": "2026-08-13T00:00:00+08:00",
            "dry_run": True,
            "mode": "mock_fault_injection",
            "summary": {"all_passed": True, "message": "免疫正常", "passed": 1, "failed": 0},
            "drills": [{
                "key": "source",
                "name": "断源",
                "passed": True,
                "elapsed_ms": 3,
                "rto_ms": 3,
                "path": "primary -> fallback",
                "rpo": {"type": "source_outage", "value": 0, "note": "mock"},
                "detail": {},
                "error": "",
            }],
        },
        tmp_path,
    )

    assert (tmp_path / "20260813" / "chaos_2026-08-13.md").exists()
    assert not (tmp_path / "2026-08-13").exists()
