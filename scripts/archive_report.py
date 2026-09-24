#!/usr/bin/env python3
"""archive_report.py — 盘后产物归档与备份 (2026-08-25)
======================================================
职责（盘后全链路最后几步之一，位于 html_report 之后、feishu_push 之前）:
  1. 把当天全部研报/决策/因子/行业链/舆情产物归档到「研报共享/{date}/」，
     即用户硬诉求「所有产出的研报要保存到研报共享文件夹，分文件夹加上日期」。
  2. 生成 archive_manifest_{date}.json 记录归档清单，便于审计与网页回显。
  3. 把当日研报 markdown 摘要备份到有道云笔记（best-effort，失败不阻断归档）。
  4. 主 HTML 详细研报同样作为一条有道云笔记备份（best-effort，过大则降级为路径记录）。

归档根目录与 html_report_generator 一致：优先 ~/Desktop/研报共享，不可写时
退到 workspace/研究报告（同一份逻辑，避免两处口径漂移）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GEN = ROOT / "generated"


def report_root() -> Path:
    configured = os.environ.get("QUANT_REPORT_ROOT", "").strip()
    desktop = Path(configured).expanduser() if configured else Path.home() / "Desktop" / "研报共享"
    try:
        desktop.mkdir(parents=True, exist_ok=True)
        probe = desktop / ".wtest"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return desktop
    except Exception:
        return ROOT / "研究报告"


def _collect_artifacts(date: str) -> dict[str, Path]:
    """返回 {归档文件名: 源路径}，只保留真实存在的源文件。"""
    cands: list[tuple[str, Path]] = [
        ("A股详细研报.html", _report_html(date)),
        ("A股融合研报.html", _fusion_html(date)),
        ("短线决策卡.md", GEN / "short_term_daily.md"),
        ("行动卡.md", GEN / f"battle_map_{date}.md"),
        ("复盘综述.md", GEN / f"review_{date}.md"),
        ("盘后龙虎榜与低估池.md", GEN / f"after_close_extra_{date}.md"),
        ("盘后龙虎榜与低估池.json", GEN / f"after_close_extra_{date}.json"),
        ("研究融合快照.json", GEN / f"research_fusion_snapshot_{date}.json"),
        ("统一决策快照.json", _latest_json(f"decision_snapshot_after_close_{date}*.json")),
        ("校准信任报告.md", GEN / "calibration_report.md"),
        ("行业链最新.json", GEN / "industry_chain_latest.json"),
        ("K线形态报告.md", GEN / f"pattern_report_{date}.md"),
        ("因子IC报告.csv", GEN / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.csv"),
        ("因子IC报告.md", GEN / "ic_report" / "FACTOR_IC_REPORT_VECTORIZED.md"),
        ("因子OOS报告.md", GEN / "ic_report" / "IC_OOS_REPORT.md"),
        ("因子OOS报告.json", GEN / "ic_report" / "IC_OOS_REPORT.json"),
        ("因子质量登记.json", _latest_json("factor_quality_registry*.json")),
    ]
    out: dict[str, Path] = {}
    for name, src in cands:
        if src and src.exists():
            out[name] = src
    return out


def _report_html(date: str) -> Path | None:
    root = report_root()
    p = root / date / f"综合详细研报_{date}.html"
    if p.exists():
        return p
    alt = ROOT / "研究报告" / date / f"综合详细研报_{date}.html"
    return alt if alt.exists() else None


def _fusion_html(date: str) -> Path | None:
    """兼容旧版/共享文件夹中的融合研报命名，避免归档漏掉用户实际查看的文件。"""
    roots = [report_root() / date, report_root(), ROOT / "研究报告" / date]
    names = [f"A股融合研报_{date}.html", f"A股多因子复盘日报_{date}.html"]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    return None


def _latest_json(pattern: str) -> Path | None:
    hits = sorted(GEN.glob(pattern))
    return hits[-1] if hits else None


def _read_text(p: Path, limit: int = 400_000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _build_digest(date: str, artifacts: dict[str, Path]) -> str:
    """把当日 markdown 产物拼成一条可读的有道云笔记正文（截断控制体积）。"""
    parts = [f"# A股量化盘后研报 {date}\n"]
    order = ["短线决策卡.md", "行动卡.md", "复盘综述.md", "盘后龙虎榜与低估池.md",
             "校准信任报告.md", "K线形态报告.md", "因子IC报告.md", "因子OOS报告.md"]
    for name in order:
        p = artifacts.get(name)
        if p and p.suffix == ".md":
            body = _read_text(p, 60_000)
            if body.strip():
                parts.append(f"\n<!-- ===== {name} ===== -->\n")
                parts.append(body)
    if not any(p.suffix == ".md" for p in artifacts.values()):
        parts.append("\n（当日无 markdown 研报产物）\n")
    parts.append("\n\n---\n归档清单见 workspace 研报共享目录与 archive_manifest。")
    return "\n".join(parts)


def archive(date: str, *, backup_youdao: bool = True) -> dict:
    root = report_root()
    day_dir = root / date
    day_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _collect_artifacts(date)
    copied: list[str] = []
    for name, src in artifacts.items():
        if src.resolve() == (day_dir / name).resolve():
            copied.append(name)  # 主 HTML 已就位，无需复制
            continue
        try:
            shutil.copy2(src, day_dir / name)
            copied.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[archive] 复制失败 {name}: {exc}", flush=True)

    manifest = {
        "date": date,
        "report_root": str(root),
        "day_dir": str(day_dir),
        "copied": sorted(copied),
        "total": len(artifacts),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = day_dir / f"archive_manifest_{date}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    youdao_status: dict = {"backed_up": False, "reason": "backup disabled"}
    if backup_youdao:
        youdao_status = _backup_youdao(date, artifacts)
    manifest["youdao"] = youdao_status
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "date": date, "day_dir": str(day_dir),
                      "copied": len(copied), "youdao": youdao_status}, ensure_ascii=False))
    return manifest


def _backup_youdao(date: str, artifacts: dict[str, Path]) -> dict:
    """有道云备份：markdown 摘要 + 主 HTML。任一失败都不抛出。"""
    try:
        from scripts.report_delivery import backup_to_youdao_note, load_config  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        try:
            from report_delivery import backup_to_youdao_note, load_config  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return {"backed_up": False, "reason": f"report_delivery 导入失败: {exc}"}

    cfg = load_config()
    digest = _build_digest(date, artifacts)
    results = []
    try:
        st = backup_to_youdao_note(cfg, title=f"A股量化盘后研报_{date}.md", content=digest)
        results.append({"note": "研报摘要", "status": st.status, "reason": st.reason, "target": st.target})
    except Exception as exc:  # noqa: BLE001
        results.append({"note": "研报摘要", "status": "未完成", "reason": str(exc)[:200]})

    for artifact_name, note_name in (("A股详细研报.html", "详细研报HTML"), ("A股融合研报.html", "融合研报HTML")):
        html = artifacts.get(artifact_name)
        if html and html.stat().st_size <= 500_000:
            try:
                st = backup_to_youdao_note(cfg, title=f"{artifact_name[:-5]}_{date}.html", content=_read_text(html))
                results.append({"note": note_name, "status": st.status, "reason": st.reason, "target": st.target})
            except Exception as exc:  # noqa: BLE001
                results.append({"note": note_name, "status": "未完成", "reason": str(exc)[:200]})
        elif html:
            results.append({"note": note_name, "status": "跳过", "reason": "体积过大，已落盘研报共享"})

    backed = any(r.get("status") == "已备份" for r in results)
    return {"backed_up": backed, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="盘后研报归档与有道云备份")
    ap.add_argument("date", help="交易日 YYYY-MM-DD")
    ap.add_argument("--no-youdao-backup", action="store_true", help="跳过有道云备份")
    args = ap.parse_args()
    archive(args.date, backup_youdao=not args.no_youdao_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
