from scripts import daily_delivery_report as delivery
from scripts.sync_fusion_workbench import build_feishu


def test_task_summary_excludes_unknown_from_success_rate():
    status = {
        "ok": ["quant_ok"],
        "fail": ["quant_fail"],
        "running": ["quant_running"],
        "unknown": ["query unavailable"],
    }
    summary = delivery._task_summary(status)
    assert summary["determined"] == 2
    assert summary["rate"] == "50.0%"
    assert summary["unknown"] == 1


def test_task_parser_marks_connection_failure_unknown():
    status = delivery._parse_task_status("ssh: connect to host: Connection timed out")
    assert not status["ok"]
    assert not status["fail"]
    assert status["unknown"]


def test_task_parser_requires_verbose_last_result_column():
    row = r'"\\quant_daily","08/28/2026 18:45","Ready","Interactive","08/27/2026 18:45","0x0","N/A","Enabled"'
    status = delivery._parse_task_status(row)
    assert status["ok"] == [r"\\quant_daily"]
    assert not status["fail"]

    short_row = r'"\\quant_daily","Ready"'
    status = delivery._parse_task_status(short_row)
    assert status["unknown"] == [r"\\quant_daily（无最近结果）"]


def test_feishu_renders_summary_gates_and_risks():
    snapshot = {
        "as_of": "2026-08-26",
        "scope": {"execution": "仅输出主板高分候选"},
        "market": {
            "emotion_stage": "修复",
            "temperature": 41,
            "force_index": 25,
            "breadth": 0.5,
            "risk_flags": ["资金合力偏弱"],
        },
        "counts": {"scanned": 5202, "candidates": 197, "execution_candidates": 0,
                   "hard_risk_candidates": 2},
        "intraday_coverage": {"status": "complete", "signal_evaluated": 5202,
                               "snapshot_rows": 5202},
        "source_chain": {"intraday_chain": True, "research": True},
        "evidence": {
            "research": {"reports": 12, "all_codes": 5, "all_companies": 5, "concepts": []},
            "regulatory": {"total_hits": 8, "codes": 2, "tier_counts": {"硬风险": 2}, "top_codes": []},
            "holders": {"available": 1, "total": 2, "items": []},
            "factor_decay": {"status": "insufficient_snapshot", "n_windows": 0},
        },
        "opportunities": [],
    }
    report = build_feishu(snapshot)
    assert "## 数据摘要" in report
    assert "## 执行门控" in report
    assert "GATE CAUTION" in report
    assert "## 风险结构" in report
    assert "5202/5202 (complete)" in report
