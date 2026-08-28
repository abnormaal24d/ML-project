"""Fail-closed network rules for effective scratch-model training.

External services may be used while creating preprocessing artefacts.  Once a
finalized snapshot enters training, however, the process must not open network
connections.  Container-level isolation remains the production boundary; this
module adds a deterministic in-process guard so local runs and tests enforce the
same schema.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

from config.environment.runtime_environment import (
    restore,
    snapshot,
)
from config.environment.runtime_environment import (
    set as set_environment,
)

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings


class OfflineTrainingRulesError(RuntimeError):
    """Raised when effective training attempts to use the network."""


_OFFLINE_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "NO_PROXY": "*",
    "TRANSFORMERS_OFFLINE": "1",
}
_STRICT_RELEASE_STAGES = frozenset({"candidate", "production_model"})


def enforce_offline_if_requested(*, settings: TrainingSettings) -> bool:
    """Validate the rules, configure hub clients, and return guard state.

    ``offline`` defaults to true in :class:`TrainingSettings`.  Production-like
    release stages are fail-closed when a caller explicitly disables it.
    """

    offline = settings.offline
    release_stage = settings.release_stage
    if release_stage in _STRICT_RELEASE_STAGES and not offline:
        raise OfflineTrainingRulesError(
            f"release_stage={release_stage!r} requires offline training"
        )
    if offline:
        for name, value in _OFFLINE_ENVIRONMENT.items():
            set_environment(name, value)
    return offline


@contextmanager
def offline_training_guard(*, settings: TrainingSettings) -> Iterator[bool]:
    """Block DNS and socket I/O for the duration of a training operation."""

    previous_environment = snapshot(tuple(_OFFLINE_ENVIRONMENT))
    offline = enforce_offline_if_requested(settings=settings)
    if not offline:
        yield False
        return

    try:
        with (
            patch.object(socket, "create_connection", _blocked_network_call),
            patch.object(socket, "getaddrinfo", _blocked_network_call),
            patch.object(socket.socket, "connect", _blocked_network_call),
            patch.object(socket.socket, "connect_ex", _blocked_network_call),
            patch.object(socket.socket, "sendto", _blocked_network_call),
        ):
            yield True
    finally:
        restore(previous_environment)


def _blocked_network_call(*_args: object, **_kwargs: object) -> None:
    raise OfflineTrainingRulesError(
        "network access is forbidden during effective model training"
    )
