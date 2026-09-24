from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import model_version_snapshot as mvs


def test_sidecar_records_hash_and_versions(tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"pickled-model")
    sidecar = mvs.write_sidecar(model, params={"n": 150})
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema"] == "quant-model-version/v1"
    assert payload["sha256"] == hashlib.sha256(b"pickled-model").hexdigest()
    assert payload["python"]
    assert payload["packages"]["numpy"] is not None
    assert payload["parameters"] == {"n": 150}
    assert not list(tmp_path.glob("*.tmp"))
