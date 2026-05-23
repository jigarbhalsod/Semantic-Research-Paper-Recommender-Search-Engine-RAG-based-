"""Weights & Biases logging helpers for embedding and retrieval experiments.

The helper runs in offline mode when no W&B API key is available, which keeps
local development and CI usable without external authentication.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
PROJECT_NAME = "semantic-paper-search"
_RUN: Any | None = None


def init_run(config: dict[str, Any], project_name: str = PROJECT_NAME) -> Any | None:
    """Initialize a W&B run and return it, or None if W&B is unavailable."""
    global _RUN
    load_dotenv()

    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb is not installed; experiment tracking is disabled.")
        _RUN = None
        return None

    mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
    try:
        _RUN = wandb.init(project=project_name, config=config, mode=mode)
    except Exception as exc:  # pragma: no cover - depends on local W&B state.
        LOGGER.warning("Could not initialize W&B; tracking is disabled: %s", exc)
        _RUN = None

    return _RUN


def log_metrics(metrics_dict: dict[str, float | int | str]) -> None:
    """Log metrics to the active W&B run when one exists."""
    if _RUN is None:
        LOGGER.debug("Skipping W&B metrics because no run is active: %s", metrics_dict)
        return
    _RUN.log(metrics_dict)


def finish() -> None:
    """Finish the active W&B run when one exists."""
    global _RUN
    if _RUN is not None:
        _RUN.finish()
        _RUN = None
