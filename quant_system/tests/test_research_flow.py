# -*- coding: utf-8 -*-
"""研报工作流 research_flow 测试：NLP 提取 / OCR 错误 / run_today 降级。"""
from __future__ import annotations

import json

from quant_system.analysis_core import research_flow as rf


def test_extract_research_entities_rating_target_price():
    text = (
        "公司 600519 贵州茅台发布研报，目标价 2500 元，评级买入。"
        "公司是白酒行业龙头，市占率第一，受益于消费升级与政策催化。"
        "产业链位于下游消费环节，订单持续增长。"
    )
    ent = rf.extract_research_entities(text)
    assert "600519" in ent["codes"]
    assert ent["rating"] == "strong"
    assert ent["target_price"] == 2500.0
    assert "龙头" in ent["leader_flags"]
    assert any("下游" in l for l in ent["chain_layers"])
    assert "政策" in ent["catalysts"]


def test_extract_research_entities_empty():
    ent = rf.extract_research_entities("")
    assert ent["codes"] == []
    assert ent["rating"] is None
    assert ent["target_price"] is None


def test_ocr_report_image_missing_file():
    res = rf.ocr_report_image("/tmp/definitely_not_exist_xyz.png")
    assert res["ok"] is False
    assert res.get("error")


def test_run_today_empty_local_dir_ok(tmp_path, monkeypatch):
    # 指向不存在的研报目录，应返回空结果不崩溃
    monkeypatch.setattr(rf, "RESEARCH_DIR", tmp_path / "no_reports")
    monkeypatch.setattr(rf, "IMA_EXPORT_DIR", tmp_path / "no_ima")
    monkeypatch.setattr(rf, "OUT_DIR", tmp_path / "out")
    res = rf.run_today("2026-08-20")
    assert res["n_reports"] == 0
    assert res["n_ocr"] == 0
    assert res["all_concepts"] == []
    assert (tmp_path / "out" / "research_flow_2026-08-20.json").exists()


def test_ingest_youdao_notes_filters_short_content():
    notes = [
        {"title": "研报A", "content": "贵州茅台目标价2500元，买入评级，白酒龙头。"},
        {"title": "空", "content": ""},
    ]
    reports = rf.ingest_youdao_notes(notes)
    assert len(reports) == 1
    assert reports[0]["source"] == "youdao_ynote"


def test_ingest_news_text_source():
    rep = rf.ingest_news_text("新闻标题", "某公司中标大单，产业链受益，龙头催化。")
    assert rep["source"] == "news"
    ent = rf.extract_research_entities(rep["content"])
    assert "中标" in ent["catalysts"]


def test_scan_ocr_dir(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("not image")
    paths = rf.scan_ocr_dir(tmp_path)
    assert len(paths) == 2
    assert all(str(tmp_path / x) in paths for x in ("a.png", "b.jpg"))
