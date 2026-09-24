"""Compatibility shim: the vectorized IC engine lives in ``scripts/ic_vectorized.py``.

Some callers import the module as a top-level name (``import ic_vectorized``)
after putting ``scripts/`` on ``sys.path``. Because this file shares that name,
it loads the implementation directly from its path and re-exports its namespace
(including the underscore-prefixed helpers callers use). No symlink required.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "scripts" / "ic_vectorized.py"

_spec = importlib.util.spec_from_file_location("_ic_vectorized_impl", _IMPL_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"cannot load IC engine implementation from {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = [n for n in dir(_impl) if not n.startswith("_")]
