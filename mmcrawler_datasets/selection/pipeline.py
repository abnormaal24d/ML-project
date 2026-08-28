"""Ordered training-sample selection: permission, quality, dedupe, quota."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )
from mmcrawler_datasets.training_samples.models import TrainingSample

from .contracts import (
    BLOCKED_USAGE_RULES,
    QuotaState,
    RejectReason,
    _source_type,
    _topic,
)


def select_samples(
    samples: Iterable[TrainingSample],
    settings: TrainingSnapshotAssemblerSettings,
    validator_settings: DatasetValidatorSettings,
) -> tuple[TrainingSample, ...]:
    """Apply permission, quality, dedupe and quota gates in fixed order."""

    from .quality import quality_reject

    accepted: list[TrainingSample] = []
    seen: set[str] = set()
    quota = QuotaState()
    for sample in samples:
        reason = permission_reject(sample, settings)
        reason = reason or quality_reject(
            sample,
            settings,
            validator_settings,
        )
        if reason is not None:
            continue
        if is_duplicate(sample, seen):
            continue
        if quota_exceeded(sample, quota, settings):
            continue
        record_keys(sample, seen)
        quota.record(sample)
        accepted.append(sample)
    return tuple(accepted)


def permission_reject(
    sample: TrainingSample,
    settings: TrainingSnapshotAssemblerSettings,
) -> RejectReason | None:
    """Return why source governance forbids a sample, if applicable."""

    if not sample.source_url.strip():
        return RejectReason.NO_SOURCE
    if not sample.domain.strip():
        return RejectReason.NO_DOMAIN
    if sample.governance.robots_status == "disallowed":
        return RejectReason.ROBOTS
    if (sample.governance.usage_rules or "").casefold() in BLOCKED_USAGE_RULES:
        return RejectReason.USAGE
    if settings.require_license_rules and not license_allowed(sample):
        return RejectReason.LICENSE
    return None


def license_allowed(sample: TrainingSample) -> bool:
    """Return whether explicit licensing permits model training."""

    governance = sample.governance
    if governance.allow_training is not True or not governance.license:
        return False
    return (governance.usage_rules or "").casefold() not in BLOCKED_USAGE_RULES


def dedupe_keys(sample: TrainingSample) -> tuple[str, ...]:
    """Return keys that preserve distinct signals across modalities."""

    fingerprints = (
        sample.content_fingerprints.normalized_items()
        if sample.content_fingerprints
        else ()
    )
    if sample.modality == "text" and bool(sample.chunk_id):
        keys = [f"{name}:{value}" for name, value in fingerprints]
        if sample.chunk_id:
            keys.append(f"chunk:{sample.chunk_id}")
        return tuple(dict.fromkeys(keys))

    modality = sample.modality or "unknown"
    signal = f"{modality}:{sample.task_target.task_type or 'unknown'}"
    keys = [f"{signal}:{name}:{value}" for name, value in fingerprints]
    if sample.near_duplicate_cluster_id:
        keys.append(f"{signal}:near:{sample.near_duplicate_cluster_id}")
    if sample.source_url:
        keys.append(f"{signal}:url:{sample.source_url}")
    return tuple(dict.fromkeys(keys))


def is_duplicate(sample: TrainingSample, seen: set[str]) -> bool:
    """Return whether any stable key for a sample was already accepted."""

    return bool(seen.intersection(dedupe_keys(sample)))


def record_keys(sample: TrainingSample, seen: set[str]) -> None:
    """Record every stable key after a sample passes all selection gates."""

    seen.update(dedupe_keys(sample))


def quota_exceeded(
    sample: TrainingSample,
    state: QuotaState,
    settings: TrainingSnapshotAssemblerSettings,
) -> bool:
    """Return whether accepting a sample would exceed any configured cap."""

    if (
        settings.max_samples_per_domain
        and state.domains[sample.domain] >= settings.max_samples_per_domain
    ):
        return True
    key = (sample.domain, sample.modality)
    if (
        settings.max_samples_per_domain_modality
        and state.domain_modalities[key]
        >= settings.max_samples_per_domain_modality
    ):
        return True
    topic = _topic(sample)
    if (
        settings.max_samples_per_topic
        and state.topics[topic] >= settings.max_samples_per_topic
    ):
        return True
    source_type = _source_type(sample)
    return bool(
        settings.max_samples_per_source_type
        and state.source_types[source_type]
        >= settings.max_samples_per_source_type
    )


__all__ = [
    "QuotaState",
    "RejectReason",
    "license_allowed",
    "permission_reject",
    "select_samples",
]
