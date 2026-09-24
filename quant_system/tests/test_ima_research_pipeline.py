# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from scripts import ima_research_pipeline as pipeline


def test_enumerate_kb_paginates_and_deduplicates(tmp_path, monkeypatch):
    calls = []

    def fake_call(api, body):
        calls.append(dict(body))
        if body.get("cursor") == "next":
            return {"data": {"knowledge_list": [{"media_id": "m2", "title": "第二篇", "media_type": 7}], "is_end": True}}
        return {"data": {"knowledge_list": [{"media_id": "m1", "title": "第一篇", "media_type": 7}, {"media_id": "m2", "title": "第二篇", "media_type": 7}], "next_cursor": "next", "is_end": False}}

    monkeypatch.setattr(pipeline, "call", fake_call)
    stats = pipeline.enumerate_kb("kb", tmp_path, page_size=10, sleep=0)
    rows = [json.loads(x) for x in (tmp_path / "items.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert [r["media_id"] for r in rows] == ["m1", "m2"]
    assert stats["duplicates"] == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["done"] == ["__root__"]


def test_priority_matches_business_order():
    paths = [
        "七、每日复盘数据", "十一、调研会议纪要", "九、题材概念产业库",
        "二、高盛、花旗", "八、大摩闭门会", "十、中金、中信",
        "一：彭博Bloomberg", "三、红宝书", "五、财联社VIP",
    ]
    assert [pipeline._priority(path) for path in paths] == list(range(9))


def test_manifest_records_full_enumeration_contract(tmp_path):
    (tmp_path / "items.jsonl").write_text(json.dumps({"media_id": "m1", "kind": "media"}) + "\n", encoding="utf-8")
    manifest = pipeline.build_manifest(tmp_path, "kb", {"pages": 1, "media": 1, "folders": 0, "duplicates": 0, "errors": 0}, expected_count=1)
    assert manifest["schema"] == "ima-research-pipeline-v1"
    assert manifest["total_media"] == 1
    assert (tmp_path / "manifest.json").exists()
