"""
test_fusion_v11 — fusion.read_fusion_latest 损坏文件降级（域G）

运行: python3 -m pytest tests/test_fusion_v11.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant_system"))


@pytest.fixture
def fusion_mod(monkeypatch, tmp_path):
    import quant_system.analysis_core.fusion as fusion
    monkeypatch.setattr(fusion, "MARKET_DIR", tmp_path)
    return fusion


def test_read_fusion_latest_normal_file_returns_data(fusion_mod, tmp_path):
    """正常 parquet：≤ref 返回最新一期 dict，行为不变。"""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-05"],
        "temperature": [40.0, 61.0],
        "tag": ["old", "new"],
    })
    df.to_parquet(tmp_path / "fusion.parquet", index=False)

    out = fusion_mod.read_fusion_latest("2026-01-06", cols=["temperature", "tag"])
    assert out is not None
    assert out["date"] == "2026-01-05"
    assert float(out["temperature"]) == 61.0
    assert out["tag"] == "new"


def test_read_fusion_latest_corrupt_file_returns_none(fusion_mod, tmp_path):
    """损坏 parquet（随机字节）：返回 None 不抛，降级链不失效。"""
    (tmp_path / "fusion.parquet").write_bytes(b"\x00\x01garbage-not-parquet\xff" * 64)

    out = fusion_mod.read_fusion_latest("2026-01-06")
    assert out is None


def test_read_fusion_latest_missing_file_returns_none(fusion_mod):
    """文件缺失：返回 None（原有正常路径）。"""
    assert fusion_mod.read_fusion_latest("2026-01-06") is None
