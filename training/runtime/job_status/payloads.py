"""Lifecycle projection for training attempt and campaign status documents.

One projector derives every persisted field from the canonical
:class:`TrainingLifecycleState`. Provenance evidence (training results)
survives every transition because it is re-projected from the existing
document rather than re-supplied at every call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from training.runtime.job_status.models import (
    TrainingCampaignIdentity,
    TrainingJobIdentity,
    TrainingLifecycleState,
    TrainingLifecycleStatus,
)

STATUS_SCHEMA_VERSION = 4

# Training outcomes persisted as structured evidence. They must survive
# every lifecycle transition, so they live in one canonical place.
_RESULT_FIELDS = (
    "checkpoint_path",
    "model_seed",
    "train_loss",
    "validation_loss",
    "test_loss",
    "average_loss",
    "last_epoch_loss",
)


class TrainingResultPayload(Protocol):
    """Structural training result consumed by status projection."""

    def to_payload(self) -> dict[str, object]:
        """Return the persisted training-result representation."""


LifecycleIdentity = TrainingJobIdentity | TrainingCampaignIdentity


def lifecycle_payload(
    *,
    identity: LifecycleIdentity,
    state: TrainingLifecycleState,
    training_root: str | Path,
    dataset_manifest_hash: str | None,
    result: TrainingResultPayload | None,
    acceptance: dict[str, object] | None,
    error: BaseException | None,
    started_at: str,
    timestamp: str,
    existing: Mapping[str, object] | None = None,
    attempt_ids: tuple[str, ...] | None = None,
    primary_attempt_id: str | None = None,
) -> dict[str, object]:
    """Project one lifecycle transition into its persisted JSON contract.

    ``existing`` carries forward previously captured evidence so later
    transitions never erase already-verified facts.
    """

    evidence = dict(_evidence_of(existing))
    if result is not None:
        evidence.update(_result_evidence(result))

    payload: dict[str, object] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "phase": "training",
        "snapshot_id": identity.snapshot_id,
        "training_root": Path(training_root).as_posix(),
        "dataset_manifest_hash": dataset_manifest_hash,
        "started_at": started_at,
        "updated_at": timestamp,
        "evidence": evidence,
        # Canonical lifecycle document.
        "lifecycle": state.to_payload(),
    }

    if isinstance(identity, TrainingJobIdentity):
        payload["attempt_id"] = identity.attempt_id
    else:
        payload["campaign_id"] = identity.campaign_id

    if existing is not None:
        for key in ("attempt_ids", "primary_attempt_id"):
            value = existing.get(key)
            if value is not None:
                payload[key] = value

    if attempt_ids is not None:
        payload["attempt_ids"] = list(attempt_ids)
    if primary_attempt_id is not None:
        payload["primary_attempt_id"] = primary_attempt_id

    if state.status in {
        TrainingLifecycleStatus.COMPLETED,
        TrainingLifecycleStatus.FAILED,
        TrainingLifecycleStatus.CANCELLED,
    }:
        payload["completed_at"] = timestamp

    if acceptance is not None:
        payload["acceptance"] = acceptance
    elif _acceptance_of(existing) is not None:
        payload["acceptance"] = _acceptance_of(existing)

    if error is not None:
        payload.update(_error_summary(error))

    return payload


def _evidence_of(existing: Mapping[str, object] | None) -> dict[str, object]:
    if existing is None:
        return {}
    evidence = existing.get("evidence")
    if isinstance(evidence, dict):
        return dict(evidence)
    return {}


def _acceptance_of(
    existing: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if existing is None:
        return None
    acceptance = existing.get("acceptance")
    if isinstance(acceptance, dict):
        return dict(acceptance)
    return None


def _result_evidence(result: TrainingResultPayload) -> dict[str, object]:
    payload = result.to_payload()
    return {field: payload.get(field) for field in _RESULT_FIELDS}


def _error_summary(error: BaseException) -> dict[str, object]:
    return {
        "error_type": type(error).__name__,
        "error_message": str(error) or None,
    }


__all__ = [
    "LifecycleIdentity",
    "STATUS_SCHEMA_VERSION",
    "TrainingResultPayload",
    "lifecycle_payload",
]
