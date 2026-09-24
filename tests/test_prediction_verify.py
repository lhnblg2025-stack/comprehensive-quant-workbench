"""
test_prediction_verify — 预测验证闭环单测（2026-08-22 融合升级）

覆盖: market_trend 预测按沪深300次日涨跌自动验证(判对/判错/flat/无下一交易日跳过)、
描述类目标超期作废、pending 队列诚实化。

运行: python3 -m pytest tests/test_prediction_verify.py -q --no-header
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def index_tmp(tmp_path):
    """合成沪深300日线: 2026-08-17..08-21 五连阳(+1%每日)。"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-17", "2026-08-18", "2026-08-19",
                                "2026-08-20", "2026-08-21"]),
        "open": [100.0] * 5,
        "high": [102.0] * 5,
        "low": [99.0] * 5,
        "close": [100.0, 101.0, 102.03, 103.05, 104.08],
        "volume": [1e8] * 5,
    })
    p = tmp_path / "index_daily.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def preds_tmp(tmp_path, monkeypatch):
    """合成预测库: 3条 market_trend(up/down/flat, target 08-19) + 1条无下一交易日(08-21) + 1条描述类。"""
    import quant_system.analysis_core.predictions as predictions
    p = tmp_path / "predictions.parquet"
    monkeypatch.setattr(predictions, "PREDICTIONS", p)
    # 预置: 直接写库(用 add_prediction 会产生真实 created_at, 描述类按需旧化)
    ids = []
    ids.append(predictions.add_prediction(target_type="market_trend", target="2026-08-19",
                                          direction="up", probability=0.7,
                                          source_module="multi_agent", note="t1"))
    ids.append(predictions.add_prediction(target_type="market_trend", target="2026-08-19",
                                          direction="down", probability=0.6,
                                          source_module="emotion_cycle", note="t2"))
    ids.append(predictions.add_prediction(target_type="market_trend", target="2026-08-19",
                                          direction="flat", probability=0.5,
                                          source_module="battle_map", note="t3"))
    ids.append(predictions.add_prediction(target_type="market_trend", target="2026-08-21",
                                          direction="up", source_module="emotion_cycle", note="nxt_na"))
    ids.append(predictions.add_prediction(target_type="情绪", target="明日情绪阶段",
                                          direction="修复", source_module="scenario", note="desc"))
    # 描述类旧化 30 天(触发作废窗口)
    df = predictions._load()
    df.loc[df["pred_id"] == ids[-1], "created_at"] = "2026-07-01T05:00:00+08:00"
    predictions._save(df)
    return ids, predictions


def test_verify_market_trend_against_index(index_tmp, preds_tmp, monkeypatch):
    import prediction_verify as pv
    monkeypatch.setattr(pv, "_INDEX_PATH", index_tmp)
    ids, predictions = preds_tmp
    # 08-19→08-20 涨 +1.0% （close 102.03→103.05）；flat 阈 ±0.3%
    # up 判对 / down 判错 / flat 判错(涨1%不在±0.3%) / 08-21 无下一交易日跳过
    r = pv.verify_pending("2026-08-22", void_stale=False)
    assert r["verified_n"] == 3
    assert r["correct_n"] == 1
    assert r["hit"] == pytest.approx(1 / 3, abs=0.01)  # 模块内 round(...,3)
    assert r["skipped_n"] == 1  # 08-21
    df = predictions._load()
    st = {row["pred_id"]: row["status"] for _, row in df.iterrows()}
    assert st[ids[0]] == "correct"
    assert st[ids[1]] == "wrong"
    assert st[ids[2]] == "wrong"
    assert st[ids[3]] == "pending"  # 无下一交易日
    # actual 回填为小数涨跌
    actual = df.loc[df["pred_id"] == ids[0], "actual"].iloc[0]
    assert abs(float(actual) - (104.08 / 103.05 - 1)) < 0.01


def test_void_stale_descriptive_preds(preds_tmp, index_tmp, monkeypatch):
    import prediction_verify as pv
    monkeypatch.setattr(pv, "_INDEX_PATH", index_tmp)
    ids, predictions = preds_tmp
    r = pv.verify_pending("2026-08-22", void_stale=True)
    df = predictions._load()
    desc = df[df["pred_id"] == ids[-1]].iloc[0]
    assert desc["status"] == "void"
    assert r["voided_n"] == 1


def test_render_verify_md_shape():
    import prediction_verify as pv
    md = pv.render_verify_md({"verified_n": 0, "pending": 5, "total": 9, "note": "无下一交易日"})
    assert "预测验证闭环" in md
    md2 = pv.render_verify_md({"verified_n": 2, "correct_n": 1, "hit": 0.5,
                               "pending": 1, "total": 3, "brier": 0.25,
                               "by_module": [{"module": "multi_agent", "n": 2, "hit": 0.5}]})
    assert "本次验证 **2** 条" in md2
    assert "multi_agent" in md2