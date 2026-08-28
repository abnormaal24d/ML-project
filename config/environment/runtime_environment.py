"""Centralized access to runtime process environment variables.

Only the dedicated configuration facades read ``os.environ`` directly.
Product modules (crawler, training, datasets, augmentation, orchestration
runtime) must go through this facade so environment-variable access stays
auditable in one place (ADR-0004 rule 4).

The values exposed here are *process/runtime* environment variables that are
not part of the settings tree:

- distributed launcher contracts (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``)
- runtime framework contracts (``CUBLAS_WORKSPACE_CONFIG``)
- deployment identity (``CONTAINER_IMAGE_DIGEST``)
- operator-configured secret names (``HF_TOKEN`` and friends)
- optional tool discovery (``POPPLER_PATH``, ``POPPLER_BIN``,
  ``LOCALAPPDATA``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import overload

CONTAINER_IMAGE_DIGEST_ENV = "CONTAINER_IMAGE_DIGEST"
DEFAULT_CONTAINER_IMAGE_DIGEST = "local-unpinned"

CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"

WORLD_SIZE_ENV = "WORLD_SIZE"
RANK_ENV = "RANK"
LOCAL_RANK_ENV = "LOCAL_RANK"

POPPLER_PATH_ENV = "POPPLER_PATH"
POPPLER_BIN_ENV = "POPPLER_BIN"
LOCAL_APP_DATA_ENV = "LOCALAPPDATA"


@overload
def get(name: str, default: None = None) -> str | None: ...


@overload
def get(name: str, default: str) -> str: ...


def get(name: str, default: str | None = None) -> str | None:
    """Return one process environment variable, or ``default`` when unset."""
    return os.environ.get(name, default)


def all_values() -> dict[str, str]:
    """Return an isolated snapshot of the complete process environment."""

    return dict(os.environ)


def set(name: str, value: str) -> None:
    """Set one process environment variable for the current process."""
    os.environ[name] = value


def unset(name: str) -> None:
    """Remove one process environment variable if it is present."""
    os.environ.pop(name, None)


def snapshot(names: tuple[str, ...]) -> dict[str, str | None]:
    """Capture the current values of the named variables for restore."""
    return {name: os.environ.get(name) for name in names}


def restore(snapshot: dict[str, str | None]) -> None:
    """Restore previously captured environment variable values."""
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def world_size() -> int:
    """Return the distributed launcher world size (default 1)."""
    return _environment_int(WORLD_SIZE_ENV, default=1)


def rank() -> int:
    """Return the distributed launcher process rank (default 0)."""
    return _environment_int(RANK_ENV, default=0)


def local_rank() -> int:
    """Return the distributed launcher local rank (default 0)."""
    return _environment_int(LOCAL_RANK_ENV, default=0)


def cublas_workspace_config() -> str | None:
    """Return the configured cuBLAS workspace config, or None."""
    return get(CUBLAS_WORKSPACE_CONFIG_ENV)


def container_image_digest() -> str:
    """Return the deployed container image digest or ``local-unpinned``."""
    return get(
        CONTAINER_IMAGE_DIGEST_ENV,
        default=DEFAULT_CONTAINER_IMAGE_DIGEST,
    )


def configured_token_value(environment_variable: str) -> str | None:
    """Return the secret value named by an operator-configured variable."""
    if not environment_variable:
        return None
    return get(environment_variable)


def poppler_environment_paths() -> tuple[Path, ...]:
    """Return candidate poppler directories named by the environment."""
    candidates: list[Path] = []
    for name in (POPPLER_PATH_ENV, POPPLER_BIN_ENV):
        value = get(name)
        if value:
            candidates.append(Path(value))
    return tuple(candidates)


def local_app_data() -> Path | None:
    """Return the Windows LOCALAPPDATA directory, or None."""
    value = get(LOCAL_APP_DATA_ENV)
    return Path(value) if value else None


def _environment_int(name: str, *, default: int) -> int:
    value = get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
