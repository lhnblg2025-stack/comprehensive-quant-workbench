"""validate_calibrate_direction — direction 校准逻辑测试 (V12.3 补覆盖)。

用临时 IC CSV + 假 registry 验证: sign(IC) 校准 / IC=0 保持 / 不在注册表跳过 / 报告结构。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_system.validate_calibrate_direction as vcd  # noqa: E402


class _FakeMeta:
    def __init__(self, base_direction: int):
        self.base_direction = base_direction
        self.direction = base_direction


class _FakeRegistry:
    def __init__(self, factors: dict):
        self._factors = factors

    def get_factor(self, name):
        return self._factors.get(name)

    def ensure_loaded(self):
        return None


def _write_ic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["factor", "category", "ic_mean", "direction"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run(monkeypatch, tmp_path, ic_rows, factors) -> dict:
    import quant_system.ic_factors.registry as _reg_mod  # noqa: PLC0415
    ic_csv = tmp_path / "ic.csv"
    out_json = tmp_path / "out.json"
    _write_ic_csv(ic_csv, ic_rows)
    monkeypatch.setattr(vcd, "IC_CSV", ic_csv)
    monkeypatch.setattr(vcd, "OUT_JSON", out_json)
    monkeypatch.setattr(vcd, "REPORT_MD", tmp_path / "out.md")
    fake = _FakeRegistry(factors)
    monkeypatch.setattr(_reg_mod, "get_factor", fake.get_factor)
    monkeypatch.setattr(_reg_mod, "ensure_loaded", fake.ensure_loaded)
    rc = vcd.main()
    assert rc == 0
    return json.loads(out_json.read_text(encoding="utf-8"))


def test_calibrate_positive_ic_sets_plus_one(monkeypatch, tmp_path):
    """IC>0 → direction 校准为 1。"""
    meta = _FakeMeta(-1)
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "f1", "category": "mom", "ic_mean": "0.05", "direction": "-1"}],
               {"f1": meta})
    assert rep["total"] == 1 and rep["changed"] == 1
    assert rep["details"][0]["new_direction"] == 1
    assert meta.direction == 1


def test_calibrate_negative_ic_sets_minus_one(monkeypatch, tmp_path):
    """IC<0 → direction 校准为 -1。"""
    meta = _FakeMeta(1)
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "f1", "category": "mom", "ic_mean": "-0.03", "direction": "1"}],
               {"f1": meta})
    assert rep["changed"] == 1
    assert rep["details"][0]["new_direction"] == -1
    assert meta.direction == -1


def test_calibrate_consistent_keeps(monkeypatch, tmp_path):
    """IC 符号与 base_direction 一致 → 不改。"""
    meta = _FakeMeta(1)
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "f1", "category": "mom", "ic_mean": "0.04", "direction": "1"}],
               {"f1": meta})
    assert rep["changed"] == 0 and rep["consistent"] == 1
    assert meta.direction == 1


def test_ic_zero_skipped(monkeypatch, tmp_path):
    """IC=0 → 跳过保持原方向。"""
    meta = _FakeMeta(-1)
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "f1", "category": "mom", "ic_mean": "0.0", "direction": "-1"}],
               {"f1": meta})
    assert rep["ic_zero"] == 1 and rep["changed"] == 0
    assert meta.direction == -1


def test_not_in_registry_reported(monkeypatch, tmp_path):
    """不在注册表的因子 → not_in_registry 列表。"""
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "ghost", "category": "x", "ic_mean": "0.02", "direction": "1"}],
               {})
    assert rep["not_in_registry"] == ["ghost"]


def test_rule_documented(monkeypatch, tmp_path):
    rep = _run(monkeypatch, tmp_path,
               [{"factor": "f1", "category": "mom", "ic_mean": "0.01", "direction": "1"}],
               {"f1": _FakeMeta(1)})
    assert "sign(ic_mean)" in rep["rule"]
    assert rep["calibrated_at"]
