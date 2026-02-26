"""Shared helper functions for Batsim integration."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
SUPPORTED_EDC_MODES = {"library_file", "library_str", "socket_file", "socket_str"}


def windows_path_to_wsl(path: Path | str) -> str:
    raw = str(path)
    if not _WINDOWS_ABS_PATH_RE.match(raw):
        return raw.replace("\\", "/")
    drive = raw[0].lower()
    tail = raw[2:].replace("\\", "/")
    return f"/mnt/{drive}{tail}"


def coerce_positive_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed


def coerce_positive_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def resolve_default_fcfs_library_path(wsl_distro: str) -> str:
    if shutil.which("wsl") is None:
        return str((Path.home() / ".nix-profile" / "lib" / "libfcfs.so").resolve())
    proc = subprocess.run(
        ["wsl", "-d", wsl_distro, "--", "bash", "-lc", 'printf "%s" "$HOME/.nix-profile/lib/libfcfs.so"'],
        capture_output=True,
        text=True,
        check=False,
    )
    candidate = proc.stdout.strip()
    if proc.returncode == 0 and candidate:
        return candidate
    return "$HOME/.nix-profile/lib/libfcfs.so"


def build_edc_args(
    edc_mode: str,
    edc_library_path: str | None,
    edc_socket_endpoint: str | None,
    edc_init_file: Path,
    edc_init_inline: str,
) -> list[str]:
    if edc_mode not in SUPPORTED_EDC_MODES:
        raise ValueError(f"unsupported edc_mode '{edc_mode}'")

    if edc_mode in {"library_file", "library_str"} and not edc_library_path:
        raise ValueError("edc_library_path is required for library EDC modes")
    if edc_mode in {"socket_file", "socket_str"} and not edc_socket_endpoint:
        raise ValueError("edc_socket_endpoint is required for socket EDC modes")

    if edc_mode == "library_file":
        return ["--edc-library-file", str(edc_library_path), str(edc_init_file)]
    if edc_mode == "library_str":
        return ["--edc-library-str", str(edc_library_path), edc_init_inline]
    if edc_mode == "socket_file":
        return ["--edc-socket-file", str(edc_socket_endpoint), str(edc_init_file)]
    return ["--edc-socket-str", str(edc_socket_endpoint), edc_init_inline]


def extract_cli_arg_value(cli_args: list[Any], flag: str) -> str | None:
    for idx, arg in enumerate(cli_args[:-1]):
        if str(arg) == flag:
            return str(cli_args[idx + 1])
    return None


def wsl_path_to_windows(raw: str) -> str:
    parts = raw.split("/")
    if len(parts) >= 4 and parts[0] == "" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        tail = "\\".join(parts[3:])
        return f"{drive}:\\{tail}"
    return raw


def resolve_local_path(raw: str) -> Path:
    candidate = os.path.expandvars(os.path.expanduser(raw))
    path = Path(candidate)
    if path.exists():
        return path
    return Path(wsl_path_to_windows(raw))


def parse_job_id(raw: Any, fallback: int) -> int:
    text = str(raw)
    if "!" in text:
        text = text.split("!")[-1]
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return fallback
    return parsed


def to_int_ts(value: Any, fallback: int = 0) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return max(0, int(math.floor(parsed)))
