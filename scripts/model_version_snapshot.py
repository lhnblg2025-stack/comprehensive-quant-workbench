#!/usr/bin/env python3
"""Write a versioned sidecar for a persisted ML model.

The sidecar records the loading-time dependency versions and the file hash so a
model is only treated as reproducible when both match on re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

CST = timezone(timedelta(hours=8))


def _versions() -> dict[str, str | None]:
    import importlib.metadata

    out: dict[str, str | None] = {}
    for name in ("scikit-learn", "numpy", "pandas", "scipy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_sidecar(model_path: Path, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "quant-model-version/v1",
        "model": model_path.name,
        "sha256": sha256_file(model_path),
        "bytes": model_path.stat().st_size,
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "packages": _versions(),
        "parameters": params or {},
    }


def write_sidecar(model_path: Path, *, params: dict[str, Any] | None = None) -> Path:
    sidecar = model_path.with_suffix(model_path.suffix + ".version.json")
    payload = build_sidecar(model_path, params=params)
    tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, sidecar)
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="write a model version sidecar")
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    sidecar = write_sidecar(args.model)
    print(json.dumps({"model": str(args.model), "sidecar": str(sidecar)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
