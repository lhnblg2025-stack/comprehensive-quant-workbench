from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from quant_platform.openclaw_api import _date_column
from quant_web.handlers import review
from quant_web import server


class _CaptureAssetHandler(server.QuantHandler):
    def __init__(self) -> None:
        self.wfile = io.BytesIO()
        self.sent: list[tuple[dict, int]] = []

    def _send_json(self, data: dict, status: int = 200) -> None:
        self.sent.append((data, status))

    def send_response(self, code: int, message: str | None = None) -> None:
        pass

    def send_header(self, keyword: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass


def test_date_column_falls_back_to_first_parseable_candidate() -> None:
    frame = pd.DataFrame({"日期": [None, None], "上榜日": ["2026-08-29", "2026-08-30"]})
    assert _date_column(frame) == "上榜日"


def test_review_rejects_invalid_date_and_symlink(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(review, "GEN", tmp_path / "generated")
    review.GEN.mkdir()
    (review.GEN / "review_2026-08-30.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert review._load_review("2026-02-30") == (None, None)
    assert review._load_review("../2026-08-30") == (None, None)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"secret": True}), encoding="utf-8")
    link = review.GEN / "review_2026-08-31.json"
    link.symlink_to(outside)
    assert review._load_review("2026-08-31") == (None, None)


def test_report_asset_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "reports"
    date_dir = root / "2026-08-30" / "assets"
    date_dir.mkdir(parents=True)
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    (date_dir / "safe.png").symlink_to(outside)
    monkeypatch.setattr(server, "_report_roots", lambda: [root])
    handler = _CaptureAssetHandler()
    handler._serve_report_asset("/report_assets/2026-08-30/assets/safe.png")
    assert handler.sent == [({"ok": False, "error": "report asset not found"}, 404)]
