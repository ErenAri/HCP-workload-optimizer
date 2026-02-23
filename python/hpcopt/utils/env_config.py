"""Environment-specific configuration loader.

Reads from ``configs/environments/{env}.yaml`` based on the ``HPCOPT_ENV``
environment variable (default: ``dev``).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "rate_limit_per_minute": 60,
    "log_level": "INFO",
    "model_staleness_warn_days": 30,
    "health_min_disk_free_gb": 0.1,
}

_CONFIG_DIR = Path("configs/environments")


def get_env_name() -> str:
    """Return the current environment name from ``HPCOPT_ENV``."""
    return os.getenv("HPCOPT_ENV", "dev")


def load_env_config() -> dict[str, Any]:
    """Load the environment-specific config, merged with defaults.

    Returns the default config if the YAML file is missing or ``pyyaml``
    is not installed.
    """
    env = get_env_name()
    config_path = _CONFIG_DIR / f"{env}.yaml"
    config = dict(_DEFAULTS)

    if not config_path.exists():
        logger.debug("No env config at %s; using defaults", config_path)
        return config

    try:
        import yaml
        with config_path.open("r", encoding="utf-8") as fh:
            file_config = yaml.safe_load(fh) or {}
        if isinstance(file_config, dict):
            config.update(file_config)
            logger.info("Loaded env config for %s from %s", env, config_path)
        else:
            logger.warning("Invalid env config format in %s; using defaults", config_path)
    except ImportError:
        logger.debug("pyyaml not installed; using default config")
    except (ValueError, TypeError):
        logger.warning("Failed to load env config from %s", config_path, exc_info=True)

    return config
