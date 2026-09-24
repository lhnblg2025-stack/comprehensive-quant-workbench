from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def _fusion():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import fusion_decision
    return fusion_decision


def test_safe_collect_timeout_preserves_default_shape():
    from daily_review_collectors import _safe_collect
    block = _safe_collect(lambda: (_ for _ in ()).throw(TimeoutError("slow")), "x", "s", {"directions": [], "components": {}}, timeout_seconds=1)
    assert block.error
    assert block.value == {"directions": [], "components": {}}


def test_mainline_falls_back_to_engine_dimensions_on_timeout():
    f = _fusion()
    blocks = {
        "strong_direction": SimpleNamespace(value={"directions": []}, error="timeout"),
        "engine_fusion": {"dimensions": {"expert": {"name": "专精特新", "score": 72}}},
        "leader_sentiment": {"value": {"components": {"ladder": {}}}},
    }
    score, note, available = f._score_mainline(blocks)
    assert available is True
    assert "回填引擎证据" in note
    assert score > 50


def test_overseas_single_quote_is_degraded():
    f = _fusion()
    blocks = {"overseas": {"quotes": [{"label": "纳指", "chg_pct": -1.2}]}}
    score, note, available = f._score_overseas(blocks)
    assert available is False
    assert "降级" in note


def test_etf_history_deduplicates_columns_before_merge(monkeypatch, tmp_path):
    import scripts.backfill_etf_sector_history as mod

    old_path = tmp_path / "old.parquet"
    state_path = tmp_path / "state.parquet"
    out_path = tmp_path / "out.parquet"
    pd.DataFrame({"code": ["510300"], "date": ["2026-08-25"], "price": [4.0], "name": ["old"], "name_state": ["stale"]}).to_parquet(old_path, index=False)
    pd.DataFrame({"code": ["510300"], "date": ["2026-08-25"], "name": ["new"]}).to_parquet(state_path, index=False)
    monkeypatch.setattr(mod, "ETF_HIST", old_path)
    monkeypatch.setattr(mod, "ETF_STATE", state_path)
    monkeypatch.setattr(mod, "_atomic", lambda df, path: df.to_parquet(out_path, index=False))

    class FakeAk:
        def fund_etf_hist_em(self, **kwargs):
            return pd.DataFrame({"日期": ["2026-08-25"], "收盘": [4.1], "成交量": [1], "成交额": [2]})

    import sys
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(**{"fund_etf_hist_em": FakeAk().fund_etf_hist_em}))
    result = mod.collect_etf(["510300"], "20260801")
    assert result["ok"]
    written = pd.read_parquet(out_path)
    assert written.columns.duplicated().sum() == 0
