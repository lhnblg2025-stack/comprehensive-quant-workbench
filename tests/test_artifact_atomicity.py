from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_html_atomic_write_publishes_complete_content(tmp_path):
    mod = _load("html_report_generator_test", ROOT / "scripts/html_report_generator.py")
    target = tmp_path / "report.html"
    target.write_text("old", encoding="utf-8")
    mod._atomic_write(target, "new complete html")
    assert target.read_text(encoding="utf-8") == "new complete html"
    assert list(tmp_path.glob("*.tmp")) == []


def test_artifact_guard_rejects_tampered_html_even_with_new_mtime(tmp_path, monkeypatch):
    mod = _load("artifact_guard_test", ROOT / "scripts/artifact_guard.py")
    day = "2026-08-30"
    report_root = tmp_path / "研究报告" / day
    report_root.mkdir(parents=True)
    html = report_root / f"A股详细研报_{day}.html"
    html.write_text("complete", encoding="utf-8")
    meta = {
        "schema": "quant-report-artifact/v1",
        "report_date": day,
        "run_id": "run-1",
        "sha256": hashlib.sha256(b"complete").hexdigest(),
        "html": html.name,
    }
    html.with_suffix(html.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    html.write_text("tampered", encoding="utf-8")
    monkeypatch.setenv("QUANT_RUN_ID", "run-1")
    assert mod._find_html(tmp_path, day) is None


def test_artifact_guard_requires_matching_run_and_date(tmp_path, monkeypatch):
    mod = _load("artifact_guard_binding_test", ROOT / "scripts/artifact_guard.py")
    day = "2026-08-30"
    report_root = tmp_path / "研究报告" / day
    report_root.mkdir(parents=True)
    html = report_root / f"A股详细研报_{day}.html"
    body = "complete"
    html.write_text(body, encoding="utf-8")
    html.with_suffix(html.suffix + ".meta.json").write_text(json.dumps({
        "schema": "quant-report-artifact/v1", "report_date": day,
        "run_id": "run-old", "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "html": html.name,
    }), encoding="utf-8")
    monkeypatch.setenv("QUANT_RUN_ID", "run-current")
    assert mod._find_html(tmp_path, day) is None
    monkeypatch.setenv("QUANT_RUN_ID", "run-old")
    assert mod._find_html(tmp_path, "2026-08-31") is None
