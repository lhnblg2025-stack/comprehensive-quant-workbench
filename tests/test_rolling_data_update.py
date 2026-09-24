from __future__ import annotations

import json
from pathlib import Path

from scripts import rolling_data_update as updater


def test_dry_run_has_real_commands_and_explicit_unsupported_domain(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "ROOT", tmp_path)
    monkeypatch.setattr(updater, "GEN", tmp_path / "generated")
    monkeypatch.setattr(updater, "STATUS", tmp_path / "generated" / "data_update_status.json")
    monkeypatch.setattr(updater, "LOCK", tmp_path / "rolling.lock")
    assert updater.TASKS["kline"][0][-2:] == ["--days", "15"]
    result = updater._run_domain("kline_hfq", None, 0, 1)
    assert result["status"] == "not_supported"
    assert "不能用前复权数据冒充" in result["error"]


def test_update_status_file_is_machine_readable(tmp_path):
    status = {"schema": "data_update_status/v1", "target_day": "2026-08-28",
              "domains": {"kline": {"status": "ok", "latest": "2026-08-28"}}}
    path = tmp_path / "data_update_status.json"
    path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["domains"]["kline"]["latest"] == "2026-08-28"
