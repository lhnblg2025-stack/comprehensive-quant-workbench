from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_system import production_pipeline as pipeline


def test_target_release_mismatch_is_not_consumed(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    pd.DataFrame([{
        "date": "2026-08-28", "code": "000001", "target_weight": 0.2,
        "release_id": "old-release",
    }]).to_parquet(generated / "factor_strategy_targets_2026-08-28.parquet", index=False)
    (generated / "factor_quality_registry.json").write_text(json.dumps({"factors": [{"tier": "core"}]}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    frame, meta = pipeline._load_factor_targets("2026-08-28", "new-release")
    assert frame.empty
    assert meta["reason"] == "target_release_mismatch"
