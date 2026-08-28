"""Shared media preprocessing loop for image, audio and video."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from logger.project_logger import ProjectLogger
from preprocessing.media.media_fingerprint import (
    build_media_fingerprint,
    build_media_id,
)
from preprocessing.media.transcript_segment_normalizer import (
    estimate_text_tokens,
)
from preprocessing.preprocessing_quality import PreprocessingQualityResult
from preprocessing.preprocessing_result import PreprocessingQuarantineRecord
from preprocessing.privacy.artifacts import (
    PrivacyArtifactWorkspace,
    build_receipt,
    canonical_sha256,
    file_sha256,
    privacy_artifact_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from preprocessing.media.media_input_validation import (
        MediaValidationResult,
    )
    from preprocessing.media.ports import EmbeddedMetadataAdapter
    from preprocessing.preprocessing_input import PreprocessingInput
    from preprocessing.privacy.artifacts import PublishedPrivacyArtifact

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class MediaPreprocessingResult(Generic[T_co]):
    """Explicit output of one media preprocessing pipeline run."""

    items: tuple[T_co, ...]
    quarantine_records: tuple[PreprocessingQuarantineRecord, ...]


class BaseMediaPreprocessor(ABC, Generic[T]):
    """Template loop: validate, optionally reject, build record, report."""

    def __init__(
        self,
        *,
        modality: str,
        logger: ProjectLogger,
        now: Callable[[], datetime],
        generate_id: Callable[[], str],
    ) -> None:
        self._modality = modality
        self._logger = logger
        self._now = now
        self._generate_id = generate_id

    def process(
        self,
        *,
        inputs: Iterable[PreprocessingInput],
    ) -> MediaPreprocessingResult[T]:
        """Validate each input and build accepted media records."""

        items: list[T] = []
        quarantine_records: list[PreprocessingQuarantineRecord] = []
        for item in inputs:
            validation = self._validate(item=item)
            if not validation.accepted:
                quarantine_records.append(
                    self._record_rejection(
                        item=item,
                        reason=validation.rejection_reason or "media_rejected",
                        quality_signals=validation.signals,
                    )
                )
                continue
            built = self._build_record(item=item, validation=validation)
            if isinstance(built, PreprocessingQuarantineRecord):
                quarantine_records.append(built)
                continue
            items.append(built)

        self._logger.info(
            f"{self._modality}_preprocessing_completed",
            emitted=len(items),
            quarantined=len(quarantine_records),
        )
        return MediaPreprocessingResult(
            items=tuple(items),
            quarantine_records=tuple(quarantine_records),
        )

    @abstractmethod
    def _validate(
        self,
        *,
        item: PreprocessingInput,
    ) -> MediaValidationResult:
        """Validate one media input."""

    @abstractmethod
    def _build_record(
        self,
        *,
        item: PreprocessingInput,
        validation: MediaValidationResult,
    ) -> T | PreprocessingQuarantineRecord:
        """Build one accepted record, or a quarantine record on late reject."""

    def _record_rejection(
        self,
        *,
        item: PreprocessingInput,
        reason: str,
        quality_signals: dict[str, object],
    ) -> PreprocessingQuarantineRecord:
        self._logger.warning(
            "preprocessing_item_rejected",
            source_id=item.source_id,
            modality=item.modality,
            reason=reason,
        )
        return PreprocessingQuarantineRecord.from_input(
            item=item,
            reason=reason,
            quality_signals=quality_signals,
        )

    def _prepare_embedded_metadata(
        self,
        *,
        item: PreprocessingInput,
        adapter: EmbeddedMetadataAdapter,
    ) -> tuple[
        dict[str, str],
        PublishedPrivacyArtifact | None,
        str | None,
    ]:
        """Inspect and transform one immutable private snapshot of the source."""

        modality = self._modality
        media_path = _selected_media_path(item)
        source = Path(media_path)
        fields: dict[str, str] = {}
        try:
            with PrivacyArtifactWorkspace(
                source_path=source,
                stage="metadata-clean",
                run_id=self._generate_id(),
            ) as workspace:
                source_digest = workspace.source_snapshot.sha256
                metadata = adapter.inspect(
                    path=workspace.source_path,
                    modality=modality,
                )
                if metadata is None:
                    return {}, None, "embedded_metadata_inspection_failed"
                fields = {
                    f"metadata:embedded:{index}": text
                    for index, value in enumerate(metadata)
                    if (text := str(value).strip())
                }
                if not fields and modality == "image":
                    return {}, None, None

                temporary = workspace.new_external_output_path(
                    suffix=source.suffix
                )
                if not adapter.remove(
                    source=workspace.source_path,
                    destination=temporary,
                    modality=modality,
                ):
                    return (
                        fields,
                        None,
                        "embedded_metadata_sanitization_failed",
                    )
                residual = adapter.inspect(
                    path=temporary,
                    modality=modality,
                )
                if residual is None or residual:
                    return fields, None, "residual_embedded_metadata_detected"
                output_digest = file_sha256(temporary)
                if file_sha256(workspace.source_path) != source_digest:
                    return fields, None, "embedded_metadata_source_changed"
                transform_path = Path(
                    inspect.getsourcefile(type(adapter)) or __file__
                )
                receipt = build_receipt(
                    workspace=workspace,
                    source_path=workspace.original_source_path,
                    source_sha256=source_digest,
                    transform_input_sha256=source_digest,
                    output_path=temporary,
                    output_sha256=output_digest,
                    source_mime_type=item.mime_type,
                    output_mime_type=item.mime_type,
                    transform_id=f"{modality}-embedded-metadata-strip",
                    transform_version="2.0.0",
                    transform_artifact_path=transform_path,
                    configuration={
                        "modality": modality,
                        "remove_all_nontechnical_metadata": True,
                        "stream_retention_policy": {
                            "image": "image_pixels_only",
                            "audio": "primary_audio_only",
                            "video": "primary_video_and_primary_audio_only",
                        }.get(modality, "unsupported_modality"),
                        **adapter.provenance(),
                    },
                    residual_inspection_sha256=canonical_sha256(
                        {"embedded_metadata": list(residual)}
                    ),
                    created_at=self._now(),
                )
                final_name = privacy_artifact_name(
                    Path(media_path), stage="metadata-clean"
                )
                artifact = workspace.publish(
                    temporary_path=temporary,
                    receipt=receipt,
                    final_name=final_name,
                )
                return fields, artifact, None
        except (OSError, RuntimeError, ValueError, TypeError):
            return fields, None, "embedded_metadata_sanitization_failed"

    def _media_id(self, *, item: PreprocessingInput) -> str:
        return build_media_id(modality=self._modality, item=item)

    def _fingerprints(
        self,
        *,
        item: PreprocessingInput,
        primary_text: str | None,
    ) -> dict[str, str]:
        return build_media_fingerprint(
            modality=self._modality,
            item=item,
            primary_text=primary_text,
        )

    def _quality_for_valid_item(
        self,
        *,
        item: PreprocessingInput,
        validation: MediaValidationResult,
        semantic_text: str | None,
        has_alignment_material: bool,
        extra_signals: dict[str, object] | None = None,
    ) -> PreprocessingQualityResult:
        signals = dict(validation.signals)
        signals.update(extra_signals or {})
        score = 0.85 if has_alignment_material else 0.65
        if semantic_text:
            score = min(1.0, score + 0.05)
        return PreprocessingQualityResult(
            score=round(score, 4),
            bucket="gold" if score >= 0.85 else "silver",
            rejection_reason=None,
            token_count_estimate=estimate_text_tokens(
                text=semantic_text or "",
            ),
            modality=self._modality,
            language=item.resolved_language(),
            alignment_score=score if has_alignment_material else 0.0,
            signals={
                key: value
                for key, value in signals.items()
                if isinstance(value, (float, int, bool, str)) or value is None
            },
            modality_signals=signals,
        )


def _selected_media_path(item: PreprocessingInput) -> str:
    """Return only the typed source path; payload paths are untrusted."""

    value = item.media_path
    if isinstance(value, str) and value.strip() and Path(value).is_file():
        return value
    return ""
