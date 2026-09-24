from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import experiment_manifest as em


def _record_by_path(records, path):
    return next((r for r in records if r["path"] == str(path)), None)


def test_manifest_records_hashes_and_finalizes_atomically(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "result.json"
    manifest_path = tmp_path / "run.json"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    output.write_text('{"ok": true}', encoding="utf-8")
    manifest = em.start(as_of="2026-08-30", command="demo", parameters={"seed": 42}, inputs=[source], run_id="run-test", output=manifest_path)
    assert manifest_path.is_file()
    input_record = _record_by_path(manifest["inputs"], source)
    assert input_record is not None
    assert input_record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    result = em.finalize(manifest, outputs=[output], output=manifest_path)
    assert result["status"] == "ok"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["finished_at"]
    output_record = _record_by_path(payload["outputs"], output)
    assert output_record is not None
    assert output_record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_does_not_hash_sensitive_paths(tmp_path):
    secret = tmp_path / "api_secret.key"
    secret.write_text("do-not-read", encoding="utf-8")
    manifest = em.start(inputs=[secret])
    assert manifest["inputs"] == []


def test_manifest_rejects_symlinks(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    manifest = em.start(inputs=[link])
    assert manifest["inputs"] == []
