from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protected-artifact resolution
# ---------------------------------------------------------------------------


def _get_production_model_dir() -> Path | None:
    """Resolve the current production model directory from the registry.

    Returns ``None`` if no production model is set or if the registry
    cannot be read.
    """
    try:
        from hpcopt.models.registry import ModelRegistry

        registry = ModelRegistry()
        entry = registry.get_production()
        if entry is not None:
            return Path(str(entry["model_dir"]))
    except (OSError, ImportError) as exc:
        logger.debug("Could not resolve production model from registry: %s", exc)
    return None


def _get_active_dossier_refs(outputs_dir: Path) -> set[Path]:
    """Scan dossier/export JSON files for referenced artifact paths.

    Any path mentioned inside a dossier is considered *active* and will not
    be cleaned up.
    """
    refs: set[Path] = set()
    dossier_dir = outputs_dir / "reports"
    if not dossier_dir.exists():
        return refs

    for json_file in dossier_dir.rglob("*_export.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            artifacts = data.get("artifacts", {})
            for key in ("report_files", "simulation_files", "model_files"):
                for entry in artifacts.get(key, []):
                    refs.add(Path(str(entry)).resolve())
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.debug("Skipping dossier ref scan for %s: %s", json_file, exc)
    return refs


# ---------------------------------------------------------------------------
# Age check
# ---------------------------------------------------------------------------


def _is_older_than(path: Path, max_age_days: int, now: dt.datetime) -> bool:
    """Return ``True`` if *path* was last modified more than *max_age_days* ago."""
    try:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
        return (now - mtime).days > max_age_days
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cleanup_artifacts(
    outputs_dir: str | Path = Path("outputs"),
    max_age_days: int = 90,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Scan for stale artifacts and optionally delete them.

    Parameters
    ----------
    outputs_dir:
        Root directory to scan (typically ``outputs/``).
    max_age_days:
        Artifacts older than this many days are candidates for deletion.
    dry_run:
        If ``True`` (default), no files are actually removed.  The return
        value still lists what *would* be deleted.

    Returns
    -------
    dict
        Summary with keys ``candidates``, ``protected``, ``deleted``
        (or ``would_delete`` if dry_run), and counts.
    """
    outputs_dir = Path(outputs_dir)
    now = dt.datetime.now(tz=dt.UTC)

    # ------------------------------------------------------------------
    # Build the protected set
    # ------------------------------------------------------------------
    protected_paths: set[Path] = set()

    # 1. Production model directory (and everything inside it).
    production_model_dir = _get_production_model_dir()
    if production_model_dir is not None:
        resolved = production_model_dir.resolve()
        protected_paths.add(resolved)
        for child in resolved.rglob("*"):
            protected_paths.add(child.resolve())
        logger.info("Protecting production model dir: %s", production_model_dir)

    # 2. Active dossier references.
    dossier_refs = _get_active_dossier_refs(outputs_dir)
    protected_paths |= dossier_refs
    if dossier_refs:
        logger.info("Protecting %d dossier-referenced artifacts.", len(dossier_refs))

    # 3. Always protect the registry file itself.
    registry_path = (outputs_dir / "models" / "registry.jsonl").resolve()
    protected_paths.add(registry_path)

    # ------------------------------------------------------------------
    # Scan for candidates
    # ------------------------------------------------------------------
    candidates: list[str] = []
    protected_skipped: list[str] = []
    deleted: list[str] = []

    if not outputs_dir.exists():
        logger.info("Outputs directory does not exist: %s", outputs_dir)
        return {
            "outputs_dir": str(outputs_dir),
            "max_age_days": max_age_days,
            "dry_run": dry_run,
            "candidates": [],
            "protected": [],
            "deleted": [],
            "summary": {"candidates_count": 0, "protected_count": 0, "deleted_count": 0},
        }

    for item in sorted(outputs_dir.rglob("*")):
        if not item.is_file():
            continue
        if not _is_older_than(item, max_age_days, now):
            continue

        candidates.append(str(item))
        resolved_item = item.resolve()

        if resolved_item in protected_paths:
            protected_skipped.append(str(item))
            continue

        if dry_run:
            deleted.append(str(item))
        else:
            try:
                item.unlink()
                deleted.append(str(item))
                logger.info("Deleted: %s", item)
            except OSError as exc:
                logger.warning("Could not delete %s: %s", item, exc)

    # Clean up empty directories (non-dry-run only).
    if not dry_run:
        for dirpath in sorted(outputs_dir.rglob("*"), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                try:
                    dirpath.rmdir()
                    logger.info("Removed empty directory: %s", dirpath)
                except OSError:
                    pass

    summary: dict[str, Any] = {
        "outputs_dir": str(outputs_dir),
        "max_age_days": max_age_days,
        "dry_run": dry_run,
        "scanned_at_utc": now.isoformat(),
        "candidates": candidates,
        "protected": protected_skipped,
        "deleted" if not dry_run else "would_delete": deleted,
        "summary": {
            "candidates_count": len(candidates),
            "protected_count": len(protected_skipped),
            "deleted_count" if not dry_run else "would_delete_count": len(deleted),
        },
    }
    logger.info(
        "Retention scan complete: %d candidates, %d protected, %d %s.",
        len(candidates),
        len(protected_skipped),
        len(deleted),
        "deleted" if not dry_run else "would be deleted (dry run)",
    )
    return summary
