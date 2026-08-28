"""Core typed training-sample aggregate models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Protocol, runtime_checkable

from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)
from preprocessing.privacy.clearance import (
    ApprovedObjectRole,
    PrivacyClearance,
)
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION

from .common import _require_sha256, _validated_object_sha256
from .fingerprints import ContentFingerprintInputs, ContentFingerprints
from .targets import TrainingTaskTarget


@runtime_checkable
class AssetContextContract(Protocol):
    """Structural contract for persisted non-content asset lineage."""

    @property
    def safety_status(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class GovernanceEvidence:
    """Immutable evidence released with training-eligible content (training-sample component)."""

    license: str | None = None
    license_url: str | None = None
    allow_training: bool | None = None
    governance_note: str | None = None
    robots_status: str | None = None
    terms_source: str | None = None
    usage_rules: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "license",
            "license_url",
            "governance_note",
            "robots_status",
            "terms_source",
            "usage_rules",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"governance {name} must be text")
        if self.allow_training is not None and not isinstance(
            self.allow_training,
            bool,
        ):
            raise TypeError("governance allow_training must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: object) -> GovernanceEvidence:
        return cls(
            **{item.name: getattr(record, item.name) for item in fields(cls)}
        )


@dataclass(frozen=True, slots=True)
class TrainingObject:
    """A referenced object participating in a multimodal sample."""

    object_id: str
    object_path: ValidatedArtifactPath
    object_sha256: str
    object_mime_type: str | None
    role: ApprovedObjectRole
    derived_from_sha256: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    page_number: int | None = None
    frame_timestamp_seconds: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("training object_id must be non-empty")
        if not isinstance(self.object_path, ValidatedArtifactPath):
            raise TypeError("training object_path must be validated")
        if not isinstance(self.role, ApprovedObjectRole):
            raise TypeError("training object role must be typed")
        _require_sha256(self.object_sha256, field_name="object_sha256")
        if self.derived_from_sha256 is not None:
            _require_sha256(
                self.derived_from_sha256,
                field_name="derived_from_sha256",
            )

    def current_digest(self) -> str | None:
        """Hash the validated artifact bound to this object."""

        try:
            return _validated_object_sha256(self.object_path)
        except OSError:
            return None


@dataclass(frozen=True, slots=True)
class TrainingTextSpan:
    """Structured text span aligned to an object, page, or time window."""

    text: str
    source: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    page_number: int | None = None
    confidence: float | None = None
    text_start: int | None = None
    text_end: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    frame_indices: tuple[int, ...] = ()
    relation_type: str | None = None
    evidence_source: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """Curated training sample with lineage and modality-aware metadata."""

    schema_version: str = TRAINING_DATASET_SCHEMA_VERSION
    sample_id: str = ""
    snapshot_id: str = ""
    split: str = "train"
    modality: str = "text"
    task_target: TrainingTaskTarget = field(default_factory=TrainingTaskTarget)
    document_id: str | None = None
    chunk_id: str | None = None
    object_id: str | None = None
    text: str = ""
    paired_text_source: str | None = None
    token_count_estimate: int = 0
    language: str | None = None
    title: str | None = None
    domain: str = ""
    source_url: str = ""
    quality_score: float = 0.0
    context_score: float | None = None
    exact_duplicate_key: str | None = None
    near_duplicate_cluster_id: str | None = None
    target_source: str | None = None
    builder_source: str | None = None
    text_quality_diagnostics: dict[str, object] | None = None
    tokenizer_backend: str | None = None
    label: int | None = None
    source_object_ids: tuple[str, ...] = ()
    dataset_version: str | None = None
    content_hash: str | None = None
    processing_version: str | None = None
    language_confidence: float | None = None
    language_script: str | None = None
    safety_status: str = "unchecked"
    safety_labels: tuple[str, ...] = ()
    pii_status: str = "unchecked"
    quarantine_reason: str | None = None
    content_fingerprints: ContentFingerprints | None = None
    fingerprint_inputs: ContentFingerprintInputs = field(
        default_factory=ContentFingerprintInputs,
    )
    objects: tuple[TrainingObject, ...] = ()
    text_spans: tuple[TrainingTextSpan, ...] = ()
    page_range_start: int | None = None
    page_range_end: int | None = None
    source_document_id: str | None = None
    normalized_url: str | None = None
    content_family_id: str | None = None
    alignment_group_id: str | None = None
    fetch_mode: str = "full"
    is_complete_payload: bool = True
    observed_bytes: int | None = None
    source_content_length: int | None = None
    media_kind: str | None = None
    parent_document_id: str | None = None
    asset_context: AssetContextContract | None = None
    is_trainable: bool = True
    governance: GovernanceEvidence = field(
        default_factory=GovernanceEvidence,
    )
    non_trainable_reason: str | None = None
    pairability_score: float | None = None
    pair_source: str | None = None
    privacy_clearance: PrivacyClearance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_target, TrainingTaskTarget):
            raise TypeError("task_target must be typed")
        if self.content_fingerprints is not None and not isinstance(
            self.content_fingerprints,
            ContentFingerprints,
        ):
            raise TypeError("content_fingerprints must be typed")
        if not isinstance(self.fingerprint_inputs, ContentFingerprintInputs):
            raise TypeError("fingerprint_inputs must be typed")
        if not isinstance(self.objects, tuple) or not all(
            isinstance(value, TrainingObject) for value in self.objects
        ):
            raise TypeError("objects must contain typed training objects")
        if not isinstance(self.text_spans, tuple) or not all(
            isinstance(value, TrainingTextSpan) for value in self.text_spans
        ):
            raise TypeError("text_spans must contain typed spans")
        if self.asset_context is not None and not isinstance(
            self.asset_context,
            AssetContextContract,
        ):
            raise TypeError("asset_context must be typed")
        if not isinstance(
            self.governance,
            GovernanceEvidence,
        ):
            raise TypeError("governance must be typed")
        if self.privacy_clearance is not None and not isinstance(
            self.privacy_clearance,
            PrivacyClearance,
        ):
            raise TypeError("privacy_clearance must be typed")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical schema-3 JSON representation."""

        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "fingerprint_inputs":
                continue
            if item.name == "privacy_clearance":
                payload[item.name] = (
                    value.to_dict()
                    if isinstance(value, PrivacyClearance)
                    else None
                )
            elif item.name == "asset_context":
                payload[item.name] = (
                    value.to_dict()
                    if isinstance(value, AssetContextContract)
                    else None
                )
            elif item.name == "governance":
                payload[item.name] = (
                    value.to_dict()
                    if isinstance(value, GovernanceEvidence)
                    else None
                )
            elif item.name == "task_target":
                payload[item.name] = self.task_target.to_mapping()
            elif item.name == "content_fingerprints":
                payload[item.name] = (
                    asdict(value)
                    if isinstance(value, ContentFingerprints)
                    else None
                )
            elif item.name in {"objects", "text_spans"}:
                payload[item.name] = (
                    [_training_object_dict(child) for child in value]
                    if item.name == "objects"
                    else [asdict(child) for child in value]
                )
            else:
                payload[item.name] = value
        return payload


def _training_object_dict(value: TrainingObject) -> dict[str, object]:
    return {
        "object_id": value.object_id,
        "object_path": value.object_path.relative_path,
        "object_sha256": value.object_sha256,
        "object_mime_type": value.object_mime_type,
        "role": value.role.value,
        "derived_from_sha256": value.derived_from_sha256,
        "start_seconds": value.start_seconds,
        "end_seconds": value.end_seconds,
        "page_number": value.page_number,
        "frame_timestamp_seconds": value.frame_timestamp_seconds,
        "confidence": value.confidence,
    }


def _build_object(
    *,
    object_id: str,
    object_path: ValidatedArtifactPath | None,
    object_mime_type: str | None,
    role: ApprovedObjectRole,
    derived_from_sha256: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    page_number: int | None = None,
    frame_timestamp_seconds: float | None = None,
    confidence: float | None = None,
) -> TrainingObject | None:
    """TrainingObject construction helper (moved from objects.py for consolidation)."""
    if not object_id or not object_path:
        return None

    return TrainingObject(
        object_id=object_id,
        object_path=object_path,
        object_sha256=_validated_object_sha256(object_path),
        object_mime_type=object_mime_type,
        role=role,
        derived_from_sha256=derived_from_sha256,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        page_number=page_number,
        frame_timestamp_seconds=frame_timestamp_seconds,
        confidence=confidence,
    )
