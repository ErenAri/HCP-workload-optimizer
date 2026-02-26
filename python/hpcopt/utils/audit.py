"""Structured audit trail for security-relevant operations."""

from __future__ import annotations

import datetime as dt
import json
import logging

audit_logger = logging.getLogger("hpcopt.audit")


def audit_log(
    action: str,
    actor: str = "system",
    details: dict | None = None,
) -> None:
    """Emit a structured audit event to the ``hpcopt.audit`` logger.

    Parameters
    ----------
    action:
        Short verb phrase describing the action (e.g. ``model.promote``).
    actor:
        Who performed the action (API key id, CLI user, ``system``).
    details:
        Additional context (model_id, old_status, new_status, etc.).
    """
    record = {
        "timestamp": dt.datetime.now(tz=dt.UTC).isoformat(),
        "action": action,
        "actor": actor,
        "details": details or {},
    }
    audit_logger.info(json.dumps(record, sort_keys=True))
