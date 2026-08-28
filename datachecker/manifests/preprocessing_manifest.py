"""Manifest model for preprocessing workflow output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datachecker.manifests.artifact_manifest import ArtifactManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus

# Canonical for preprocessing
PREPROCESSED_LIFECYCLE = "preprocessed"


@dataclass(frozen=True, slots=True)
class PreprocessingManifest(ArtifactManifest):
    """Persisted proof that preprocessing is current for crawl input."""

    crawl_manifest_hash: str
    crawl_output_fingerprint: str
    preprocessing_settings_hash: str
    normalization_settings_hash: str
    deduplication_settings_hash: str
    splitting_settings_hash: str
    validation_settings_hash: str
    output_fingerprint: str
    curated_snapshot_directory: Path
    curated_snapshot_manifest_path: Path
    training_snapshot_directory: Path
    training_dataset_manifest_path: Path
    training_dataset_manifest_hash: str
    input_document_count: int
    output_document_count: int
    output_chunk_count: int
    output_image_count: int
    output_audio_count: int
    output_video_count: int
    output_alignment_count: int
    training_sample_count: int
    rejected_document_count: int
    rejected_image_count: int
    rejected_audio_count: int
    rejected_video_count: int
    image_coverage: dict[str, object]
    audio_coverage: dict[str, object]
    video_coverage: dict[str, object]
    built_at: str | None
    lifecycle_stage: str = PREPROCESSED_LIFECYCLE
    status: WorkflowLifecycleStatus = WorkflowLifecycleStatus.COMPLETED
    final: bool = True

    def __post_init__(self) -> None:
        ArtifactManifest.__post_init__(self)
        if self.lifecycle_stage != PREPROCESSED_LIFECYCLE:
            raise ValueError(
                "preprocessing lifecycle_stage must be preprocessed"
            )
        if (
            self.status is not WorkflowLifecycleStatus.COMPLETED
            or not self.final
        ):
            raise ValueError(
                "preprocessing manifest must be completed and final"
            )
        for name, val in (
            ("input_document_count", self.input_document_count),
            ("output_document_count", self.output_document_count),
            ("output_chunk_count", self.output_chunk_count),
            ("output_image_count", self.output_image_count),
            ("output_audio_count", self.output_audio_count),
            ("output_video_count", self.output_video_count),
            ("output_alignment_count", self.output_alignment_count),
            ("training_sample_count", self.training_sample_count),
            ("rejected_document_count", self.rejected_document_count),
            ("rejected_image_count", self.rejected_image_count),
            ("rejected_audio_count", self.rejected_audio_count),
            ("rejected_video_count", self.rejected_video_count),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >=0, got {val}")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
    ) -> PreprocessingManifest:
        """Build a manifest instance from JSON payload data."""

        return cls(
            **cls.identity_from_payload(payload),
            crawl_manifest_hash=str(payload.get("crawl_manifest_hash", "")),
            crawl_output_fingerprint=str(
                payload.get("crawl_output_fingerprint", "")
            ),
            preprocessing_settings_hash=str(
                payload.get("preprocessing_settings_hash", "")
            ),
            normalization_settings_hash=str(
                payload.get("normalization_settings_hash", "")
            ),
            deduplication_settings_hash=str(
                payload.get("deduplication_settings_hash", "")
            ),
            splitting_settings_hash=str(
                payload.get("splitting_settings_hash", "")
            ),
            validation_settings_hash=str(
                payload.get("validation_settings_hash", "")
            ),
            output_fingerprint=str(payload.get("output_fingerprint", "")),
            curated_snapshot_directory=cls._opt_path(
                payload.get("curated_snapshot_directory")
            ),
            curated_snapshot_manifest_path=cls._opt_path(
                payload.get("curated_snapshot_manifest_path")
            ),
            training_snapshot_directory=cls._opt_path(
                payload.get("training_snapshot_directory")
            ),
            training_dataset_manifest_path=cls._opt_path(
                payload.get("training_dataset_manifest_path")
            ),
            training_dataset_manifest_hash=str(
                payload.get("training_dataset_manifest_hash", "")
            ),
            input_document_count=cls.as_int(
                payload.get("input_document_count")
            ),
            output_document_count=cls.as_int(
                payload.get("output_document_count")
            ),
            output_chunk_count=cls.as_int(payload.get("output_chunk_count")),
            output_image_count=cls.as_int(payload.get("output_image_count")),
            output_audio_count=cls.as_int(payload.get("output_audio_count")),
            output_video_count=cls.as_int(payload.get("output_video_count")),
            output_alignment_count=cls.as_int(
                payload.get("output_alignment_count")
            ),
            training_sample_count=cls.as_int(
                payload.get("training_sample_count")
            ),
            rejected_document_count=cls.as_int(
                payload.get("rejected_document_count")
            ),
            rejected_image_count=cls.as_int(
                payload.get("rejected_image_count")
            ),
            rejected_audio_count=cls.as_int(
                payload.get("rejected_audio_count")
            ),
            rejected_video_count=cls.as_int(
                payload.get("rejected_video_count")
            ),
            image_coverage=cls._coverage_dict(payload.get("image_coverage")),
            audio_coverage=cls._coverage_dict(payload.get("audio_coverage")),
            video_coverage=cls._coverage_dict(payload.get("video_coverage")),
            built_at=cls.as_opt_str(payload.get("built_at")),
            lifecycle_stage=str(
                payload.get("lifecycle_stage", PREPROCESSED_LIFECYCLE)
            ),
            status=WorkflowLifecycleStatus.parse(payload.get("status")),
            final=cls.as_bool(payload.get("final")),
        )

    @staticmethod
    def _coverage_dict(value: object) -> dict[str, object]:
        # Point 53: strict (no silent {})
        if not isinstance(value, dict):
            raise TypeError("coverage field must be dict")
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _opt_path(raw: object) -> Path:
        text = PreprocessingManifest.as_opt_str(raw)
        if not text or text.strip() == ".":
            raise ValueError(
                "preprocessing manifest path field must be valid non-. path"
            )
        return Path(text)
