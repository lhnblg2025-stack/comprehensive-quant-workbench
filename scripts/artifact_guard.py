#!/usr/bin/env python3
"""盘后产物归档与投递守卫。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

_DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_REQUIRED_MARKERS = (
    "统一短线决策：全市场主线扫描与主板精选",
    "统一决策审计：覆盖、候选、监管与筹码证据",
    "ETF", "fusionFlowChart", "短线分", "盘中覆盖",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_html(root: Path, day: str) -> Path | None:
    """Find only a report with a matching metadata sidecar, never by mtime."""
    candidates = []
    configured = os.environ.get("QUANT_REPORT_ROOT", "").strip()
    roots = ([Path(configured).expanduser()] if configured else []) + [
        Path.home() / "Desktop" / "研报共享", root / "研究报告"
    ]
    for base in roots:
        if base.is_dir():
            candidates.extend(base.glob(f"**/综合详细研报_{day}.html"))
    valid = []
    expected_run = os.environ.get("QUANT_RUN_ID", "").strip()
    for path in candidates:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("schema") != "quant-report-artifact/v1":
                continue
            if meta.get("report_date") != day or meta.get("html") != path.name:
                continue
            if expected_run and meta.get("run_id") != expected_run:
                continue
            if meta.get("sha256") != _sha256(path):
                continue
            valid.append(path)
        except (OSError, ValueError, TypeError):
            continue
    return valid[0] if len(valid) == 1 else (valid[0] if valid else None)


def validate_artifacts(root: Path, day: str) -> Path:
    if not _DATE_RE.fullmatch(day):
        raise SystemExit(f"日期无效: {day}")
    gen = root / "generated"
    sys.path.insert(0, str(root))
    try:
        from quant_system.product_contract import PRODUCT_VERSION, validate_production_contract
        production = validate_production_contract(root)
    except Exception as exc:
        raise SystemExit(f"生产路径契约读取失败: {exc}")
    if production.get("status") != "PASS":
        raise SystemExit(f"生产路径契约无效: {production.get('errors')}")
    try:
        from quant_system.data_release import load_release
        release = load_release(root, expected_day=day)
    except Exception as exc:
        raise SystemExit(f"数据发布版本缺失或未通过: {exc}")

    long_report_path = gen / "runs" / f"frozen_amp20_100_2020_2026_{PRODUCT_VERSION}-r2" / "long_backtest_report.json"
    if not long_report_path.exists():
        raise SystemExit(f"冻结包长回测报告缺失: {long_report_path}")
    try:
        long_report = json.loads(long_report_path.read_text(encoding="utf-8"))
        expected_hash = long_report.get("report_sha256")
        payload = dict(long_report); payload.pop("report_sha256", None)
        actual_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    except Exception as exc:
        raise SystemExit(f"冻结包长回测报告损坏: {exc}")
    if long_report.get("product_version") != PRODUCT_VERSION:
        raise SystemExit(f"冻结包长回测版本错误: {long_report.get('product_version')} expected={PRODUCT_VERSION}")
    if (long_report.get("quality_gate") or {}).get("status") != "PASS":
        raise SystemExit("冻结包质量门禁未通过")
    if (long_report.get("accounting_audit") or {}).get("status") != "PASS":
        raise SystemExit("冻结包长回测会计审计未通过")
    if expected_hash != actual_hash:
        raise SystemExit("冻结包长回测报告哈希无效")

    review = gen / f"review_{day}.json"
    decision = gen / f"decision_snapshot_after_close_{day}.json"
    short = gen / "short_term_daily.md"
    for path in (review, decision, short):
        if not path.exists():
            raise SystemExit(f"产物缺失: {path}")

    try:
        review_data = json.loads(review.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"综合复盘JSON损坏: {exc}")
    review_release = ((review_data.get("blocks") or {}).get("data_release") or {}).get("value") or {}
    if review_release.get("release_id") != release.get("release_id"):
        raise SystemExit(f"复盘数据发布版本不一致: {review_release.get('release_id')} != {release.get('release_id')}")

    contract_candidates = [review.with_suffix(".md.research.json"), review.with_suffix(review.suffix + ".research.json")]
    contract_path = next((path for path in contract_candidates if path.exists()), None)
    if contract_path is None:
        raise SystemExit(f"研究契约缺失: {contract_candidates[0]}")
    try:
        sys.path.insert(0, str(root / "scripts"))
        from report_contract import validate_report_contract
        research = json.loads(contract_path.read_text(encoding="utf-8"))
        errors = validate_report_contract(research)
        if errors:
            raise SystemExit(f"研究契约无效: {errors[:12]}")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"研究契约读取失败: {exc}")

    html = _find_html(root, day)
    if html is None:
        raise SystemExit(f"产物缺失或未绑定当前运行: 综合详细研报_{day}.html")
    meta_path = html.with_suffix(html.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_run = os.environ.get("QUANT_RUN_ID", "").strip()
    if expected_run and meta.get("run_id") != expected_run:
        raise SystemExit(f"HTML运行绑定错误: {meta.get('run_id')} expected={expected_run}")

    try:
        decision_data = json.loads(decision.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"统一决策快照损坏: {exc}")
    if decision_data.get("as_of") != day or decision_data.get("mode") != "after_close":
        raise SystemExit(f"统一决策快照契约错误: as_of={decision_data.get('as_of')} mode={decision_data.get('mode')} expected={day}/after_close")
    html_text = html.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in _REQUIRED_MARKERS if marker not in html_text]
    if missing:
        raise SystemExit(f"HTML缺少短线决策能力: {missing}")
    coverage = decision_data.get("intraday_coverage") or {}
    if coverage.get("status") == "complete":
        if int(decision_data.get("counts", {}).get("scanned") or 0) != int(coverage.get("snapshot_rows") or 0):
            raise SystemExit("完整盘中链与盘后扫描数量不一致")
        if html.stat().st_size < 80_000:
            print(f"WARN: 主报告偏短但核心契约完整: {html.stat().st_size} bytes", file=sys.stderr)
    elif "盘中覆盖降级" not in html_text:
        raise SystemExit(f"覆盖状态{coverage.get('status') or 'missing'}但HTML未显示降级")
    text = short.read_text(encoding="utf-8", errors="replace")[:300]
    match = re.search(r"20\d{2}-\d{2}-\d{2}", text)
    if not match or match.group(0) != day:
        raise SystemExit(f"短线日报日期错误: {match.group(0) if match else 'missing'} expected={day}")
    return html


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: artifact_guard.py YYYY-MM-DD", file=sys.stderr)
        return 2
    html = validate_artifacts(Path(os.environ.get("QUANT_ROOT", "${PROJECT_ROOT}")), sys.argv[1])
    print(f"artifacts ok: {sys.argv[1]} html={html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
