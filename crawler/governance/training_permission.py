"""Single fail-closed training-permission decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingPermissionDecision:
    allowed: bool
    violations: tuple[str, ...]


class _CollectionEvidence(Protocol):
    @property
    def allowed(self) -> bool: ...


class _AccessEvidence(Protocol):
    @property
    def checked(self) -> bool: ...

    @property
    def decision(self) -> str: ...


class _RightsEvidence(Protocol):
    @property
    def checked(self) -> bool: ...

    @property
    def decision(self) -> str: ...

    @property
    def rights_reserved(self) -> bool: ...

    @property
    def tdm_allowed(self) -> bool: ...

    def is_current(self, *, now: datetime) -> bool: ...


class _CheckEvidence(Protocol):
    @property
    def checked(self) -> bool: ...

    @property
    def result(self) -> str: ...


class _LineageEvidence(Protocol):
    @property
    def complete(self) -> bool: ...


def resolve_training_permission(
    *,
    collection: _CollectionEvidence,
    access: _AccessEvidence,
    rights: _RightsEvidence,
    privacy: _CheckEvidence,
    dedupe: _CheckEvidence,
    quality: _CheckEvidence,
    lineage: _LineageEvidence,
    processing_activity_allowed: bool,
    dpia_approved: bool,
    now: datetime,
) -> TrainingPermissionDecision:
    """Resolve permission from explicit evidence; missing evidence always denies."""

    violations: list[str] = []
    if not collection.allowed:
        violations.append("collection_not_allowed")
    if not access.checked or access.decision != "allow":
        violations.append(f"access_{access.decision}")
    if not rights.checked or rights.decision != "allow":
        violations.append("rights_not_allowed")
    elif not rights.is_current(now=now):
        violations.append("rights_review_expired")
    if rights.rights_reserved:
        violations.append("rights_reserved")
    if not rights.tdm_allowed:
        violations.append("tdm_not_allowed")
    if not privacy.checked or privacy.result != "pass":
        violations.append("privacy_not_passed")
    if not dedupe.checked or dedupe.result != "pass":
        violations.append("dedupe_not_passed")
    if not quality.checked or quality.result != "pass":
        violations.append("quality_not_passed")
    if not lineage.complete:
        violations.append("lineage_incomplete")
    if not processing_activity_allowed:
        violations.append("processing_activity_not_allowed")
    if not dpia_approved:
        violations.append("dpia_not_approved")
    return TrainingPermissionDecision(
        allowed=not violations,
        violations=tuple(violations),
    )
