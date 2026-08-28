"""Atomic JSON persistence for training-job status documents."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from training.runtime.job_status.models import (
    TrainingCampaignIdentity,
    TrainingJobIdentity,
)

GenerateId = Callable[[], str]

LifecycleIdentity = TrainingJobIdentity | TrainingCampaignIdentity


class TrainingJobStatusError(RuntimeError):
    """A training job status could not be serialized or persisted."""


class AtomicTrainingJobStatusWriter:
    """Persist status payloads atomically under attempt-scoped paths."""

    def __init__(
        self,
        *,
        root: str | Path,
        generate_id: GenerateId,
        replace_retry_attempts: int,
        replace_retry_delay_seconds: float,
    ) -> None:
        self._root = Path(root)
        self._generate_id = generate_id
        self._replace_retry_attempts = replace_retry_attempts
        self._replace_retry_delay_seconds = replace_retry_delay_seconds

    def path_for(self, *, identity: TrainingJobIdentity) -> Path:
        return (
            self._root
            / identity.snapshot_id
            / "attempts"
            / identity.attempt_id
            / "status.json"
        )

    def campaign_path_for(self, *, identity: TrainingCampaignIdentity) -> Path:
        return (
            self._root
            / identity.snapshot_id
            / "campaigns"
            / identity.campaign_id
            / "status.json"
        )

    def path_for_identity(self, *, identity: LifecycleIdentity) -> Path:
        if isinstance(identity, TrainingCampaignIdentity):
            return self.campaign_path_for(identity=identity)
        return self.path_for(identity=identity)

    def list_attempts(
        self, *, snapshot_id: str
    ) -> tuple[TrainingJobIdentity, ...]:
        attempts_root = self._root / snapshot_id / "attempts"
        if not attempts_root.is_dir():
            return ()
        identities: list[TrainingJobIdentity] = []
        for path in attempts_root.iterdir():
            if not (path / "status.json").is_file():
                continue
            identities.append(
                TrainingJobIdentity(
                    snapshot_id=snapshot_id,
                    attempt_id=path.name,
                )
            )
        return tuple(sorted(identities, key=lambda item: item.attempt_id))

    def list_campaigns(
        self, *, snapshot_id: str
    ) -> tuple[TrainingCampaignIdentity, ...]:
        campaigns_root = self._root / snapshot_id / "campaigns"
        if not campaigns_root.is_dir():
            return ()
        identities: list[TrainingCampaignIdentity] = []
        for path in campaigns_root.iterdir():
            if not (path / "status.json").is_file():
                continue
            identities.append(
                TrainingCampaignIdentity(
                    snapshot_id=snapshot_id,
                    campaign_id=path.name,
                )
            )
        return tuple(sorted(identities, key=lambda item: item.campaign_id))

    def read(self, *, identity: LifecycleIdentity) -> dict[str, object] | None:
        path = self.path_for_identity(identity=identity)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingJobStatusError(
                "Training job status could not be read"
            ) from exc
        if not isinstance(payload, dict):
            raise TrainingJobStatusError(
                "Training job status must contain a JSON object"
            )
        return payload

    def write(
        self,
        *,
        identity: LifecycleIdentity,
        payload: Mapping[str, object],
    ) -> Path:
        path = self.path_for_identity(identity=identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(
            f"{path.name}.{self._generate_id()}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_with_retry(
                source=temporary_path,
                target=path,
            )
            if os.name != "nt":
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

        return path

    def _replace_with_retry(
        self,
        *,
        source: Path,
        target: Path,
    ) -> None:
        for attempt in range(self._replace_retry_attempts):
            try:
                source.replace(target)
                return
            except PermissionError:
                if attempt == self._replace_retry_attempts - 1:
                    raise
                time.sleep(self._replace_retry_delay_seconds * (attempt + 1))


__all__ = [
    "AtomicTrainingJobStatusWriter",
    "LifecycleIdentity",
    "TrainingJobStatusError",
]
