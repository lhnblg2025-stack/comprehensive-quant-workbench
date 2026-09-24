from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS = Path("generated") / "experiments"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "size": path.stat().st_size}


@dataclass
class Experiment:
    name: str
    as_of: str
    parameters: dict[str, Any] = field(default_factory=dict)
    strategy_ref: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            seed = {"name": self.name, "as_of": self.as_of, "parameters": self.parameters, "strategy_ref": self.strategy_ref}
            self.run_id = _json_hash(seed)[:16]
        self.manifest_path = RUNS / f"{self.run_id}.json"
        self._started = False
        self._manifest: dict[str, Any] | None = None

    def _params(self) -> dict[str, Any]:
        params = self.strategy_ref.get("parameters", self.parameters)
        return {
            "parameters_sha256": _json_hash(self.parameters),
            "strategy_params_sha256": _json_hash(params),
        }

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def start(self, inputs: list[str | Path] | None = None) -> dict[str, Any]:
        input_files = []
        for item in inputs or []:
            path = Path(item)
            if path.exists() and path.is_file():
                input_files.append(_file_digest(path))
            else:
                input_files.append({"path": str(path), "missing": True})
        manifest = {
            "schema": "quant-experiment/v2",
            "run_id": self.run_id,
            "name": self.name,
            "as_of": self.as_of,
            "status": "running",
            "parameters": self.parameters,
            "strategy_ref": self.strategy_ref,
            "inputs": input_files,
            "checkpoints": [],
            "started_at": _utc_now(),
            "finished_at": None,
            **self._params(),
        }
        self._manifest = manifest
        self._started = True
        self._write_manifest(manifest)
        return dict(manifest)

    def checkpoint(self, message: str, **extra: Any) -> dict[str, Any]:
        if not self._started or self._manifest is None:
            raise RuntimeError("experiment has not been started")
        item = {"at": _utc_now(), "message": message, **extra}
        self._manifest.setdefault("checkpoints", []).append(item)
        self._write_manifest(self._manifest)
        return item

    def finalize(self, status: str = "completed", **extra: Any) -> dict[str, Any]:
        if not self._started or self._manifest is None:
            raise RuntimeError("experiment has not been started")
        self._manifest.update({"status": status, "finished_at": _utc_now(), **extra})
        self._write_manifest(self._manifest)
        return dict(self._manifest)
