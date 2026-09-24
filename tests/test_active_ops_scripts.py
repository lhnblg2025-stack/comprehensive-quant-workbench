from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_health_guardian_uses_configured_paths_and_strict_known_hosts(monkeypatch, tmp_path):
    mod = _load("health_guardian_ops_test", "scripts/health_guardian.py")
    monkeypatch.setattr(mod, "CLOUD_HOST", "user@example")
    monkeypatch.setattr(mod, "CLOUD_PEM", "/tmp/key")
    monkeypatch.setattr(mod, "KNOWN_HOSTS", tmp_path / "known_hosts")
    (tmp_path / "known_hosts").write_text("example ssh-ed25519 key\n", encoding="utf-8")
    calls = []

    class Result:
        returncode = 0
        stdout = "".encode()

    monkeypatch.setattr(mod.subprocess, "run", lambda args, **kwargs: calls.append(args) or Result())
    mod._cloud_ssh("echo ok")
    args = calls[0]
    assert "StrictHostKeyChecking=yes" in args
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in args


def test_daily_delivery_requires_cloud_configuration(monkeypatch):
    mod = _load("daily_delivery_ops_test", "scripts/daily_delivery_report.py")
    monkeypatch.setattr(mod, "HOST_TX", "")
    monkeypatch.setattr(mod, "PEM", "")
    try:
        mod._ssh_tx("echo ok")
    except RuntimeError as exc:
        assert "QUANT_CLOUD_HOST" in str(exc)
    else:
        raise AssertionError("missing cloud configuration must fail closed")


def test_shell_entries_use_root_relative_runtime_paths():
    for name in ("after_close_update.sh", "run_daily_after_close.sh", "cloud_pull_fallback.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'QUANT_ROOT:-' in text
        assert 'QUANT_LOG_DIR:-$ROOT/generated/logs' in text
        assert 'QUANT_RUN_DIR:-$ROOT/generated/run' in text or name == "run_daily_after_close.sh"


def test_shell_scripts_are_syntactically_valid():
    for name in ("after_close_update.sh", "run_daily_after_close.sh", "cloud_pull_fallback.sh"):
        result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
