"""Validation for preprocessing workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datachecker.validation.shared_validation import ArtifactPathPresence
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowDecisionReason,
)

if TYPE_CHECKING:
    from datachecker.inventory.curated_snapshot_inventory import (
        CuratedInventory,
    )
    from datachecker.inventory.training_snapshot_inventory import (
        TrainingInventory,
    )
    from datachecker.manifests.preprocessing_manifest import (
        PreprocessingManifest,
    )


class PreprocessingArtifactValidator:
    """
    Validate preprocessing output against current crawl and settings state.
    """

    def __init__(
        self,
        *,
        minimum_documents: int,
        minimum_chunks: int,
        minimum_images: int = 0,
        minimum_audio: int = 0,
        minimum_video: int = 0,
        minimum_cross_modal_alignments: int = 0,
        minimum_transcript_coverage: float = 0.0,
        minimum_ocr_coverage: float = 0.0,
        minimum_keyframe_coverage: float = 0.0,
    ) -> None:
        self._minimum_documents = minimum_documents
        self._minimum_chunks = minimum_chunks
        self._minimum_images = minimum_images
        self._minimum_audio = minimum_audio
        self._minimum_video = minimum_video
        self._minimum_cross_modal_alignments = minimum_cross_modal_alignments
        self._minimum_transcript_coverage = minimum_transcript_coverage
        self._minimum_ocr_coverage = minimum_ocr_coverage
        self._minimum_keyframe_coverage = minimum_keyframe_coverage

    def validate(
        self,
        *,
        manifest: PreprocessingManifest | None,
        curated_inventory: CuratedInventory,
        training_inventory: TrainingInventory,
        current_crawl_manifest_hash: str | None,
        current_preprocessing_settings_hash: str,
        current_normalization_settings_hash: str,
        current_deduplication_settings_hash: str,
        current_splitting_settings_hash: str,
        current_validation_settings_hash: str,
    ) -> ValidationResult:
        """Validate preprocessing outputs against current upstream state."""

        if manifest is None:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_MISSING,
                details=("top-level preprocessing manifest is missing",),
            )
        if current_crawl_manifest_hash is None:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_INPUT_CHANGED,
                details=("current crawl manifest hash is unavailable",),
            )
        artifact_validation = self._validate_manifest_artifacts(
            manifest=manifest,
            curated_inventory=curated_inventory,
            training_inventory=training_inventory,
        )
        if artifact_validation is not None:
            return artifact_validation
        if manifest.crawl_manifest_hash != current_crawl_manifest_hash:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_INPUT_CHANGED,
                details=(
                    "crawl manifest hash changed since preprocessing ran",
                    f"stored={manifest.crawl_manifest_hash}",
                    f"current={current_crawl_manifest_hash}",
                ),
            )
        settings_validation = self._validate_settings_hashes(
            manifest=manifest,
            current_preprocessing_settings_hash=(
                current_preprocessing_settings_hash
            ),
            current_normalization_settings_hash=(
                current_normalization_settings_hash
            ),
            current_deduplication_settings_hash=(
                current_deduplication_settings_hash
            ),
            current_splitting_settings_hash=current_splitting_settings_hash,
            current_validation_settings_hash=current_validation_settings_hash,
        )
        if settings_validation is not None:
            return settings_validation
        count_validation = self._validate_output_counts(
            inventory=curated_inventory,
        )
        if count_validation is not None:
            return count_validation

        coverage_validation = self._validate_multimodal_coverage_checks(
            manifest=manifest,
            minimum_transcript_coverage=self._minimum_transcript_coverage,
            minimum_ocr_coverage=self._minimum_ocr_coverage,
            minimum_keyframe_coverage=self._minimum_keyframe_coverage,
        )
        if coverage_validation is not None:
            return coverage_validation

        if training_inventory.fingerprint != manifest.output_fingerprint:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "preprocessing output fingerprint no longer matches",
                    f"stored={manifest.output_fingerprint}",
                    f"current={training_inventory.fingerprint}",
                ),
            )

        media_validation = self._validate_curated_media_files_on_disk(
            snapshot_directory=manifest.curated_snapshot_directory,
        )
        if media_validation is not None:
            return media_validation

        return ValidationResult.valid(
            reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        )

    def _validate_manifest_artifacts(
        self,
        *,
        manifest: PreprocessingManifest,
        curated_inventory: CuratedInventory,
        training_inventory: TrainingInventory,
    ) -> ValidationResult | None:
        if (
            manifest.lifecycle_stage != "preprocessed"
            or manifest.status != "completed"
            or not manifest.final
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "preprocessing manifest is not finalized",
                    f"lifecycle_stage={manifest.lifecycle_stage}",
                    f"status={manifest.status}",
                    f"final={manifest.final}",
                ),
            )
        if self._outputs_missing(
            curated_inventory=curated_inventory,
            training_inventory=training_inventory,
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_MISSING,
                details=(
                    "curated snapshot or training snapshot output is missing",
                ),
            )
        absent_paths = ArtifactPathPresence.missing(
            manifest.curated_snapshot_directory,
            manifest.curated_snapshot_manifest_path,
            manifest.training_snapshot_directory,
            manifest.training_dataset_manifest_path,
        )
        if absent_paths:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_MISSING,
                details=(
                    "preprocessing manifest references missing "
                    "physical artifacts",
                    *absent_paths,
                ),
            )
        if (
            manifest.curated_snapshot_directory != curated_inventory.directory
            or manifest.curated_snapshot_manifest_path
            != curated_inventory.manifest_path
            or manifest.training_snapshot_directory
            != training_inventory.directory
            or manifest.training_dataset_manifest_path
            != training_inventory.manifest_path
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "preprocessing manifest does not match the selected "
                    "curated or training snapshot",
                ),
            )
        return None

    @staticmethod
    def _validate_settings_hashes(
        *,
        manifest: PreprocessingManifest,
        current_preprocessing_settings_hash: str,
        current_normalization_settings_hash: str,
        current_deduplication_settings_hash: str,
        current_splitting_settings_hash: str,
        current_validation_settings_hash: str,
    ) -> ValidationResult | None:
        expected_hashes = (
            (
                "preprocessing",
                manifest.preprocessing_settings_hash,
                current_preprocessing_settings_hash,
            ),
            (
                "normalization",
                manifest.normalization_settings_hash,
                current_normalization_settings_hash,
            ),
            (
                "deduplication",
                manifest.deduplication_settings_hash,
                current_deduplication_settings_hash,
            ),
            (
                "splitting",
                manifest.splitting_settings_hash,
                current_splitting_settings_hash,
            ),
            (
                "validation",
                manifest.validation_settings_hash,
                current_validation_settings_hash,
            ),
        )
        for label, stored_hash, current_hash in expected_hashes:
            if stored_hash != current_hash:
                return ValidationResult.invalid(
                    reason=WorkflowDecisionReason.PREPROCESSING_SETTINGS_CHANGED,
                    details=(
                        f"{label} settings fingerprint changed",
                        f"stored={stored_hash}",
                        f"current={current_hash}",
                    ),
                )
        return None

    @staticmethod
    def _validate_curated_media_files_on_disk(
        *,
        snapshot_directory: Path | None,
        sample_limit: int = 24,
    ) -> ValidationResult | None:
        if snapshot_directory is None or not snapshot_directory.exists():
            return None

        missing_paths: list[str] = []
        checked = 0
        for relative_name in (
            "images/images.jsonl",
            "audio/audio.jsonl",
            "video/video.jsonl",
        ):
            manifest_path = snapshot_directory / relative_name
            if not manifest_path.exists():
                continue
            try:
                lines = manifest_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                missing_paths.append(relative_name)
                continue

            for line in lines:
                if checked >= sample_limit:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                media_path = str(payload.get("media_path") or "").strip()
                if not media_path:
                    continue
                checked += 1
                resolved = (
                    snapshot_directory / media_path
                    if not Path(media_path).is_absolute()
                    else Path(media_path)
                )
                if not resolved.is_file():
                    missing_paths.append(media_path)

        if missing_paths:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "curated media files missing on disk",
                    *missing_paths[:8],
                ),
            )
        return None

    @staticmethod
    def _outputs_missing(
        *,
        curated_inventory: CuratedInventory,
        training_inventory: TrainingInventory,
    ) -> bool:
        return (
            curated_inventory.directory is None
            or curated_inventory.manifest_path is None
            or training_inventory.directory is None
            or training_inventory.manifest_path is None
            or not curated_inventory.schema_valid
            or not training_inventory.schema_valid
        )

    def _validate_output_counts(
        self,
        *,
        inventory: CuratedInventory,
    ) -> ValidationResult | None:
        checks = (
            (
                "document",
                self._minimum_documents,
                inventory.document_count,
            ),
            ("chunk", self._minimum_chunks, inventory.chunk_count),
            ("image", self._minimum_images, inventory.image_count),
            ("audio", self._minimum_audio, inventory.audio_count),
            ("video", self._minimum_video, inventory.video_count),
            (
                "cross-modal alignment",
                self._minimum_cross_modal_alignments,
                inventory.alignment_count,
            ),
        )
        for label, minimum, current in checks:
            if current >= minimum:
                continue
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    f"preprocessed {label} count is below the "
                    "required minimum",
                    f"minimum={minimum}",
                    f"current={current}",
                ),
            )
        return None

    @staticmethod
    def _validate_multimodal_coverage_checks(
        *,
        manifest: PreprocessingManifest,
        minimum_transcript_coverage: float,
        minimum_ocr_coverage: float,
        minimum_keyframe_coverage: float,
    ) -> ValidationResult | None:
        if "trainable" not in manifest.video_coverage:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "video_coverage.trainable is required by schema 3.0",
                ),
            )
        transcript_numerator = PreprocessingArtifactValidator._as_float(
            manifest.audio_coverage.get("accepted_with_transcript")
        ) + PreprocessingArtifactValidator._as_float(
            manifest.video_coverage.get("accepted_with_transcript")
        )
        transcript_denominator = PreprocessingArtifactValidator._as_float(
            manifest.audio_coverage.get("curated_accepted")
        ) + PreprocessingArtifactValidator._as_float(
            manifest.video_coverage.get("curated_accepted")
        )
        if (
            transcript_numerator < 0
            or transcript_denominator < 0
            or transcript_numerator > transcript_denominator
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    "transcript coverage has impossible accepted counts",
                    f"accepted_with_transcript={transcript_numerator}",
                    f"curated_accepted={transcript_denominator}",
                ),
            )

        transcript_coverage = (
            1.0
            if transcript_denominator <= 0
            else transcript_numerator / transcript_denominator
        )
        checks = (
            (
                "transcript",
                minimum_transcript_coverage,
                transcript_coverage,
            ),
            (
                "ocr",
                minimum_ocr_coverage,
                PreprocessingArtifactValidator._coverage_ratio(
                    coverage=manifest.image_coverage,
                    numerator_key="accepted_with_ocr",
                    denominator_key="curated_accepted",
                ),
            ),
            (
                "keyframe",
                minimum_keyframe_coverage,
                PreprocessingArtifactValidator._video_readiness_ratio(
                    coverage=manifest.video_coverage,
                    denominator_key="curated_accepted",
                ),
            ),
        )
        for label, minimum, current in checks:
            if current >= minimum:
                continue
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.PREPROCESSING_OUTPUT_INVALID,
                details=(
                    f"preprocessed {label} coverage is below the required "
                    f"minimum",
                    f"minimum={minimum}",
                    f"current={round(current, 4)}",
                ),
            )
        return None

    @staticmethod
    def _coverage_ratio(
        *,
        coverage: dict[str, object],
        numerator_key: str,
        denominator_key: str,
    ) -> float:
        denominator = PreprocessingArtifactValidator._as_float(
            coverage.get(denominator_key)
        )
        if denominator <= 0:
            return 1.0
        return (
            PreprocessingArtifactValidator._as_float(
                coverage.get(numerator_key)
            )
            / denominator
        )

    @staticmethod
    def _video_readiness_ratio(
        *,
        coverage: dict[str, object],
        denominator_key: str,
    ) -> float:
        denominator = PreprocessingArtifactValidator._as_float(
            coverage.get(denominator_key)
        )
        if denominator <= 0:
            return 1.0

        trainable = PreprocessingArtifactValidator._as_float(
            coverage.get("trainable")
        )
        return trainable / denominator

    @staticmethod
    def _as_float(value: object) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0
