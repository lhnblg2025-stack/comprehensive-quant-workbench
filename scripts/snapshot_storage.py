"""Lossless dated snapshot storage. Keep five dates hot; gzip older dates.

No automatic deletion: archived dates remain available for historical replay.
Only the two explicitly named snapshot families are rotated.
"""
import gzip
import json
import os
import re
import tempfile
from pathlib import Path

FAMILIES = ("intraday_chain_", "decision_snapshot_after_close_")


class SnapshotPath(type(Path())):
    """Logical .json path which also resolves an archived .json.gz file."""

    def _physical(self):
        plain = Path(str(self))
        return plain if plain.exists() else Path(str(self) + ".gz")

    def exists(self):
        return self._physical().exists()

    def is_file(self):
        return self._physical().is_file()

    def stat(self, **kwargs):
        return self._physical().stat(**kwargs)

    def read_text(self, encoding=None, errors=None):
        physical = self._physical()
        if physical.suffix == ".gz":
            with gzip.open(physical, "rt", encoding=encoding or "utf-8", errors=errors) as handle:
                return handle.read()
        return physical.read_text(encoding=encoding, errors=errors)


def snapshot_paths(directory, pattern):
    directory = Path(directory)
    names = {p.name for p in directory.glob(pattern)}
    names.update(p.name[:-3] for p in directory.glob(pattern + ".gz"))
    return [SnapshotPath(directory / name) for name in sorted(names)]


def write_snapshot(path, payload, **kwargs):
    """Atomic compact JSON write; preserve every field and value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def archive_snapshots(directory, keep=5):
    """Archive older per-family dates, verify bytes before removing plain JSON.

    A changed file is left alone. Temporary outputs never match reader globs.
    keep counts observed dates, not calendar days (weekends do not expire data).
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    directory = Path(directory)
    archived = []
    for family in FAMILIES:
        files = sorted(p for p in directory.glob(family + "*.json")
                       if re.fullmatch(re.escape(family) + r"20\d{2}-\d{2}-\d{2}\.json", p.name)
                       and p.is_file() and not p.is_symlink())
        for path in files[:-keep]:
            before = path.stat()
            raw = path.read_bytes()
            fd, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as handle:
                    with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as zipped:
                        zipped.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                with gzip.open(temporary, "rb") as handle:
                    if handle.read() != raw:
                        raise OSError("snapshot archive verification failed")
                after = path.stat()
                if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
                    continue
                os.replace(temporary, str(path) + ".gz")
                path.unlink()
                archived.append(path.name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return archived
