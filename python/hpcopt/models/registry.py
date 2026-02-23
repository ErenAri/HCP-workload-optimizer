from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from hpcopt.utils.io import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-platform advisory file lock
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(lock_path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Acquire an advisory file lock, blocking up to *timeout* seconds.

    Uses ``msvcrt.locking`` on Windows and ``fcntl.flock`` on POSIX.
    Falls back to no-op if locking is unsupported (e.g. some network mounts).
    """
    import time

    ensure_dir(lock_path.parent)
    fh = lock_path.open("w", encoding="utf-8")
    deadline = time.monotonic() + timeout

    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except (OSError, IOError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Could not acquire lock on {lock_path}")
                    time.sleep(0.05)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (OSError, IOError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Could not acquire lock on {lock_path}")
                    time.sleep(0.05)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError):
            pass
        fh.close()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_STATUSES = {"registered", "production", "archived"}

DEFAULT_REGISTRY_PATH = Path("outputs/models/registry.jsonl")


@dataclass
class RegistryEntry:
    """Single row in the model registry."""

    model_id: str
    model_dir: str
    status: str  # registered | production | archived
    registered_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryEntry:
        return cls(
            model_id=str(data["model_id"]),
            model_dir=str(data["model_dir"]),
            status=str(data["status"]),
            registered_at=str(data["registered_at"]),
            metadata=data.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Append-only registry backed by a JSONL file.

    Thread-safety is provided via a ``threading.Lock`` so that multiple
    in-process callers can safely read/write concurrently.  Cross-process
    safety is provided via an advisory file lock on ``<registry>.lock``.
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path: Path = registry_path or DEFAULT_REGISTRY_PATH
        self._lock = threading.Lock()
        self._lock_path = self._path.with_suffix(".jsonl.lock")
        # Eagerly create the parent directory so later writes never fail due
        # to a missing directory.
        ensure_dir(self._path.parent)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[RegistryEntry]:
        """Load every entry from the JSONL backing file."""
        if not self._path.exists():
            return []

        entries: list[RegistryEntry] = []
        for line_no, raw_line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
                entries.append(RegistryEntry.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed registry line %d in %s: %s",
                    line_no,
                    self._path,
                    exc,
                )
        return entries

    def _backup(self) -> None:
        """Create a backup copy of the registry before destructive writes."""
        if self._path.exists():
            bak_path = self._path.with_suffix(".jsonl.bak")
            shutil.copy2(self._path, bak_path)

    def _write_all(self, entries: list[RegistryEntry]) -> None:
        """Overwrite the backing file atomically with *entries*.

        Creates a ``.bak`` backup, writes to a temporary file, then renames
        it into place to prevent partial writes from corrupting the registry.
        Cross-process safety via advisory file lock.
        """
        ensure_dir(self._path.parent)
        with _file_lock(self._lock_path):
            self._backup()
            lines = [json.dumps(entry.to_dict(), sort_keys=True) for entry in entries]
            content = "\n".join(lines) + "\n"
            tmp_path = self._path.with_suffix(".jsonl.tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(self._path)

    def _append_one(self, entry: RegistryEntry) -> None:
        """Append a single entry without rewriting the whole file."""
        ensure_dir(self._path.parent)
        with _file_lock(self._lock_path):
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    def _find_entry(
        self, entries: list[RegistryEntry], model_id: str
    ) -> RegistryEntry | None:
        for entry in entries:
            if entry.model_id == model_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        model_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a newly-trained model.

        Parameters
        ----------
        model_id:
            Unique identifier for this model version.
        model_dir:
            Path to the directory containing model artifacts.
        metadata:
            Arbitrary key/value pairs (hyperparams, dataset id, etc.).

        Returns
        -------
        dict
            The full registry entry as a plain dict.

        Raises
        ------
        ValueError
            If *model_id* is already registered.
        """
        with self._lock:
            existing = self._read_all()
            if self._find_entry(existing, model_id) is not None:
                raise ValueError(
                    f"Model '{model_id}' is already registered. "
                    "Use a unique model_id or archive the existing one first."
                )

            entry = RegistryEntry(
                model_id=model_id,
                model_dir=str(model_dir),
                status="registered",
                registered_at=dt.datetime.now(tz=dt.UTC).isoformat(),
                metadata=metadata or {},
            )
            self._append_one(entry)
            logger.info("Registered model %s at %s", model_id, model_dir)
            return entry.to_dict()

    def get(self, model_id: str) -> dict[str, Any]:
        """Return the registry entry for *model_id*.

        Raises
        ------
        KeyError
            If no entry exists for the given *model_id*.
        """
        with self._lock:
            entries = self._read_all()
        match = self._find_entry(entries, model_id)
        if match is None:
            raise KeyError(f"No model registered with id '{model_id}'")
        return match.to_dict()

    def list(self) -> list[dict[str, Any]]:
        """Return every registry entry, ordered by registration time."""
        with self._lock:
            entries = self._read_all()
        return [entry.to_dict() for entry in entries]

    def get_production(self) -> dict[str, Any] | None:
        """Return the current production model, or ``None`` if none is promoted."""
        with self._lock:
            entries = self._read_all()
        for entry in entries:
            if entry.status == "production":
                return entry.to_dict()
        return None

    def promote(self, model_id: str) -> dict[str, Any]:
        """Promote *model_id* to production.

        Any previously-promoted model is demoted back to ``registered``.

        Raises
        ------
        KeyError
            If *model_id* is not found.
        ValueError
            If *model_id* is archived.
        """
        with self._lock:
            entries = self._read_all()
            target = self._find_entry(entries, model_id)
            if target is None:
                raise KeyError(f"No model registered with id '{model_id}'")
            if target.status == "archived":
                raise ValueError(
                    f"Cannot promote archived model '{model_id}'. "
                    "Re-register it first."
                )

            # Demote any existing production model.
            for entry in entries:
                if entry.status == "production" and entry.model_id != model_id:
                    entry.status = "registered"
                    logger.info(
                        "Demoted model %s from production to registered",
                        entry.model_id,
                    )

            target.status = "production"
            self._write_all(entries)
            logger.info("Promoted model %s to production", model_id)
            try:
                from hpcopt.utils.audit import audit_log
                audit_log("model.promote", details={"model_id": model_id})
            except (ImportError, OSError):
                pass
            return target.to_dict()

    def archive(self, model_id: str) -> dict[str, Any]:
        """Archive a model so it is no longer eligible for production.

        Raises
        ------
        KeyError
            If *model_id* is not found.
        """
        with self._lock:
            entries = self._read_all()
            target = self._find_entry(entries, model_id)
            if target is None:
                raise KeyError(f"No model registered with id '{model_id}'")

            target.status = "archived"
            self._write_all(entries)
            logger.info("Archived model %s", model_id)
            try:
                from hpcopt.utils.audit import audit_log
                audit_log("model.archive", details={"model_id": model_id})
            except (ImportError, OSError):
                pass
            return target.to_dict()
