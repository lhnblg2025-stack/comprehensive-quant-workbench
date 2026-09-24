#!/usr/bin/env python3
"""Classify Git dirty entries without deleting or restoring user files."""
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def category(path: str) -> str:
    p = path.strip('"')
    if p in {".gitignore", "pytest.ini"}:
        return "governance"
    if p.startswith(("data_cache/", "tmp_tx/")):
        return "cache"
    if "__pycache__" in p or p.endswith(".pyc") or p.startswith(".cache/") or p in {".coverage", ":memory:-shm", ":memory:-wal"}:
        return "cache"
    if p.startswith(("generated/", "data_warehouse/")):
        return "runtime_artifact"
    if p.startswith("logs/") or p.endswith(".log"):
        return "log"
    if p.startswith(("tmp/", "tmp_out/")):
        return "temporary"
    if p.startswith((".archive/", "archive/", "垃圾文件夹/")):
        return "archive"
    if p.startswith(("tests/", "quant_system/tests/")):
        return "test"
    if p.startswith(("scripts/", "quant_web/", "quant_platform/", "quant_system/")) or p.endswith((".py", ".js", ".html", ".sh")):
        return "source"
    if p.startswith(("项目文档/", "audit_fusion_", "memory/", "运维记录/")) or p.endswith(".md"):
        return "documentation"
    if p.startswith(("skills/", "mcp/", ".clawhub/")):
        return "external_or_plugin"
    if p.startswith("config/") or p.endswith((".sqlite", ".db", ".db-shm", ".db-wal")):
        return "runtime_state"
    return "review_required"


def main() -> int:
    raw = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in raw:
        status = line[:2]
        path = line[3:]
        groups[category(path)].append({"status": status, "path": path})
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(raw),
        "counts": dict(Counter({k: len(v) for k, v in groups.items()})),
        "groups": dict(groups),
        "policy": {
            "preserve": ["source", "test", "documentation", "governance", "review_required", "external_or_plugin"],
            "ignore_or_lifecycle": ["cache", "runtime_artifact", "runtime_state", "log", "temporary"],
            "archive_with_manifest": ["archive"],
            "destructive_actions_performed": False,
        },
    }
    out_dir = ROOT / "generated" / "workspace_hygiene"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "dirty_inventory.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "total": report["total"], "counts": report["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
