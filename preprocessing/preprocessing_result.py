"""Structured result models emitted by preprocessing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from preprocessing.provenance import ProducerProvenance

if TYPE_CHECKING:
    from preprocessing.preprocessed_document import PreprocessedDocument
    from preprocessing.preprocessed_media import (
        PreprocessedAudio,
        PreprocessedImage,
        PreprocessedVideo,
    )
    from preprocessing.preprocessing_input import PreprocessingInput


@dataclass(frozen=True, slots=True)
class PreprocessingQuarantineRecord:
    source_id: str
    reason: str
    source_url: str | None = None
    path: str | None = None
    finding_counts: dict[str, int] = field(default_factory=dict)
    pii_spans: tuple[dict[str, object], ...] = ()
    modality: str = "text"
    media_path: str | None = None
    mime_type: str | None = None
    byte_size: int | None = None
    quality_signals: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_input(
        cls,
        *,
        item: PreprocessingInput,
        reason: str,
        finding_counts: dict[str, int] | None = None,
        pii_spans: tuple[dict[str, object], ...] = (),
        quality_signals: dict[str, object] | None = None,
    ) -> PreprocessingQuarantineRecord:
        """Build a sanitized quarantine record from one input."""

        return cls(
            source_id=item.source_id,
            reason=reason,
            source_url=item.normalized_url or item.source_url,
            path=item.path,
            finding_counts=dict(finding_counts or {}),
            pii_spans=tuple(pii_spans),
            modality=item.modality,
            media_path=item.media_path,
            mime_type=item.mime_type,
            byte_size=item.byte_size,
            quality_signals=dict(quality_signals or {}),
        )


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    documents: tuple[PreprocessedDocument, ...]
    skipped_sources: dict[str, str] = field(default_factory=dict)
    quarantine_records: tuple[PreprocessingQuarantineRecord, ...] = ()
    images: tuple[PreprocessedImage, ...] = ()
    audio: tuple[PreprocessedAudio, ...] = ()
    video: tuple[PreprocessedVideo, ...] = ()
    provenance: tuple[ProducerProvenance, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)
