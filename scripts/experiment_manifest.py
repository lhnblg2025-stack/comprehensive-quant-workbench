#!/usr/bin/env python3
"""Create reproducible research run manifests without storing secrets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CST = timezone(timedelta(hours=8))
SENSITIVE_NAMES = {".env", ".env.secrets", "credentials", "secrets"}
SENSITIVE_SUFFIXES = (".key", ".pem")


def _safe_path(path: str | Path) -> bool:
    """Reject secret-bearing path components; never inspect file contents."""
    parts = {part.lower() for part in Path(path).parts}
    name = Path(path).name.lower()
    return not (parts & SENSITIVE_NAMES or name.endswith(SENSITIVE_SUFFIXES))


def sha256_file(path: str | Path) -> str | None:
    p = Path(path).expanduser()
    if not _safe_path(p) or not p.is_file() or p.is_symlink():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any] | None:
    digest = sha256_file(path)
    if digest is None:
        return None
    p = Path(path).expanduser().resolve()
    try:
        display = p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display = str(p)
    return {"path": display, "sha256": digest, "bytes": p.stat().st_size}


def _file_records(paths: list[str] | None) -> list[dict[str, Any]]:
    records = []
    for path in paths or []:
        record = _file_record(path)
        if record is not None:
            records.append(record)
    return records


def _git(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "pyarrow", "scikit-learn", "akshare"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def start(*, as_of: str | None = None, command: str = "", parameters: dict[str, Any] | None = None, inputs: list[str] | None = None, run_id: str | None = None, output: str | Path | None = None) -> dict[str, Any]:
    rid = run_id or f"exp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    manifest: dict[str, Any] = {
        "schema": "quant-experiment/v1",
        "run_id": rid,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": as_of,
        "command": command,
        "parameters": parameters or {},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "packages": _package_versions()},
        "git": {"root": _git(ROOT), "quant_system": _git(ROOT / "quant_system")},
        "inputs": _file_records(inputs),
        "outputs": [],
    }
    if output:
        write_manifest(output, manifest)
    return manifest


def finalize(manifest: dict[str, Any], *, outputs: list[str] | None = None, status: str = "ok", output: str | Path | None = None) -> dict[str, Any]:
    result = dict(manifest)
    result["status"] = status
    result["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["outputs"] = _file_records(outputs)
    if output:
        write_manifest(output, result)
    return result


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="write a reproducible experiment manifest")
    parser.add_argument("output", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--command", default="")
    parser.add_argument("--input", action="append", default=[])
    args = parser.parse_args()
    manifest = start(as_of=args.as_of, command=args.command, inputs=args.input, output=args.output)
    print(json.dumps({"run_id": manifest["run_id"], "manifest": str(args.output), "status": "running"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
