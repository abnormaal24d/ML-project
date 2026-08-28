"""Text selection and pairability scoring for training samples."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from mmcrawler_datasets.curated.training_projection import (
    TimedMediaTrainingInput,
    TrainingAudioInput,
    TrainingVideoInput,
)
from preprocessing.privacy.clearance import PrivacyClearance

TimedMedia = CuratedAudioRecord | CuratedVideoRecord | TimedMediaTrainingInput
PairableMedia = TimedMedia | CuratedImageRecord
MAX_DOCUMENT_TEXT_BYTES = 4 * 1024 * 1024


class DocumentTextRejection(ValueError):
    """A deterministic rejection at the curated-document file boundary."""


def read_document_text(
    *,
    snapshot_directory: Path,
    text_path: str,
    privacy_clearance: PrivacyClearance | PrivacyClearanceRecord,
    cache: dict[str, str | None] | None = None,
    max_bytes: int = MAX_DOCUMENT_TEXT_BYTES,
) -> str | None:
    """Read and verify one bounded curated-document buffer exactly once."""

    if cache is not None and text_path in cache:
        return cache[text_path]
    relative = Path(text_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DocumentTextRejection("document_text_path_not_relative")
    try:
        root = snapshot_directory.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except FileNotFoundError:
        if cache is not None:
            cache[text_path] = None
        return None
    except (OSError, RuntimeError, ValueError) as exc:
        raise DocumentTextRejection(
            "document_text_path_outside_snapshot"
        ) from exc
    try:
        with path.open("rb") as handle:
            if os.fstat(handle.fileno()).st_size > max_bytes:
                raise DocumentTextRejection("document_text_exceeds_byte_limit")
            raw = handle.read(max_bytes + 1)
    except DocumentTextRejection:
        raise
    except OSError as exc:
        raise DocumentTextRejection("document_text_io_failed") from exc
    if len(raw) > max_bytes:
        raise DocumentTextRejection("document_text_exceeds_byte_limit")
    if hashlib.sha256(raw).hexdigest() != privacy_clearance.output_digest:
        raise DocumentTextRejection("document_text_digest_mismatch")
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DocumentTextRejection("document_text_decode_failed") from exc
    approved_body = privacy_clearance.approved_text("body")
    if (
        not privacy_clearance.permits_training
        or approved_body is None
        or decoded != approved_body
    ):
        raise DocumentTextRejection("document_text_not_privacy_approved")
    normalized = decoded if len(decoded.strip()) >= 16 else None
    if cache is not None:
        cache[text_path] = normalized
    return normalized


def select_image_text(image: CuratedImageRecord) -> tuple[str, str] | None:
    """Select one privacy-approved image text field by stable precedence.

    Includes all available text sources for contrastive alignment tasks.
    """

    clearance = image.privacy_clearance
    if clearance is None or not clearance.permits_training:
        return None
    candidates = (
        ("caption_text", "caption", image.caption_text),
        ("ocr_text", "ocr", image.ocr_text),
        ("ocr_preview", "ocr", image.ocr_preview),
        ("figcaption", "figcaption", image.figcaption),
        ("alt_text", "alt", image.alt_text),
        ("surrounding_text", "surrounding", image.surrounding_text),
        ("page_title", "page_title", image.page_title),
    )
    return _select_approved_candidate(clearance, candidates)


def select_image_caption(image: CuratedImageRecord) -> tuple[str, str] | None:
    """Select one privacy-approved image caption field by strict precedence.

    Only uses high-quality caption sources: caption_text, figcaption,
    and carefully filtered alt_text. Excludes OCR, surrounding text,
    page titles, and other noisy sources that would leak target text
    into the encoder for causal captioning tasks.
    """

    clearance = image.privacy_clearance
    if clearance is None or not clearance.permits_training:
        return None
    candidates = (
        ("caption_text", "caption", image.caption_text),
        ("figcaption", "figcaption", image.figcaption),
        ("alt_text", "alt", image.alt_text),
    )
    return _select_approved_candidate(clearance, candidates)


def image_quality(image: CuratedImageRecord) -> float:
    """Return the bounded visual/text alignment quality for an image pair."""

    alignment = min(image.caption_quality_score, image.context_score)
    visual = image.image_quality_score
    if visual is None:
        visual = alignment
    weighted = (
        image.caption_quality_score * 0.55
        + image.context_score * 0.30
        + visual * 0.15
    )
    return round(max(0.0, min(1.0, alignment, weighted)), 4)


def select_media_text(record: TimedMedia) -> tuple[str, str] | None:
    """Select one privacy-approved audio/video text field."""

    clearance = record.privacy_clearance
    if clearance is None or not clearance.permits_training:
        return None
    candidates = (
        ("transcript_text", "transcript", record.transcript_text),
        (
            "transcript_preview",
            "transcript_preview",
            record.transcript_preview,
        ),
        (
            "frame_ocr_text",
            "frame_ocr",
            getattr(record, "frame_ocr_text", None)
            if record.modality == "video"
            else None,
        ),
        (
            "frame_ocr_preview",
            "frame_ocr_preview",
            (
                getattr(record, "frame_ocr_preview", None)
                if record.modality == "video"
                else None
            ),
        ),
        ("surrounding_text", "surrounding", record.surrounding_text),
        ("page_title", "page_title", record.page_title),
        ("html_context", "html_context", record.html_context),
    )
    return _select_approved_candidate(clearance, candidates)


def select_speech_transcript(record: TimedMedia) -> tuple[str, str] | None:
    """Select one privacy-approved speech transcript field by strict precedence.

    Only uses the full approved transcript_text for transcription tasks.
    Excludes transcript_preview, surrounding_text, page_title, html_context,
    and other noisy sources that would leak target text into the encoder
    for causal transcription tasks.
    """

    clearance = record.privacy_clearance
    if clearance is None or not clearance.permits_training:
        return None
    candidates = (("transcript_text", "transcript", record.transcript_text),)
    return _select_approved_candidate(clearance, candidates)


def select_video_caption(record: TimedMedia) -> tuple[str, str] | None:
    """Select one privacy-approved video caption field by strict precedence.

    Only uses explicit video caption sources for causal video captioning tasks.
    Excludes transcript, OCR, surrounding text, page titles, and other noisy
    sources that would leak target text into the encoder for causal
    captioning tasks.
    """

    clearance = record.privacy_clearance
    if clearance is None or not clearance.permits_training:
        return None
    candidates = (
        (
            "video_caption_text",
            "video_caption",
            getattr(record, "video_caption_text", None),
        ),
    )
    return _select_approved_candidate(clearance, candidates)


def fallback_text(record: TimedMedia) -> str:
    """Build deterministic standalone text when contextual text exists."""

    modality = normalize_text(record.modality) or record.modality
    page_title = normalize_text(record.page_title)
    mime_type = normalize_text(record.media_mime_type)
    duration = (
        getattr(record, "audio_duration_seconds", None)
        if record.modality == "audio"
        else getattr(record, "video_duration_seconds", None)
    )
    width = (
        getattr(record, "video_width", None)
        if record.modality == "video"
        else None
    )
    height = (
        getattr(record, "video_height", None)
        if record.modality == "video"
        else None
    )
    label = modality.capitalize()
    if page_title and duration is not None:
        return f"{page_title}. {label} object with duration {duration:.1f} seconds."
    if page_title and width is not None and height is not None:
        return (
            f"{page_title}. {label} object with resolution {width}x{height}."
        )
    if page_title:
        return f"{page_title}. {label} object."
    if duration is not None:
        return f"{label} object with duration {duration:.1f} seconds."
    if width is not None and height is not None:
        return f"{label} object with resolution {width}x{height}."
    if mime_type:
        return f"{label} object of type {mime_type}."
    if record.source_url:
        return f"{label} object from {record.source_url}."
    if record.media_id:
        return f"{label} object with id {record.media_id}."
    return f"{label} object."


def pairability_score(
    *,
    media_record: PairableMedia | Mapping[str, object],
    parent_text: str | None,
) -> float:
    """Return how suitable one media record is for a text-pair sample."""

    row = (
        media_record.to_dict()
        if isinstance(
            media_record,
            (
                CuratedAudioRecord,
                CuratedImageRecord,
                CuratedVideoRecord,
                TrainingAudioInput,
                TrainingVideoInput,
            ),
        )
        else media_record
    )
    context_value = row.get("asset_context")
    context = context_value if isinstance(context_value, Mapping) else {}

    score = 0.0

    if (
        _has_text(
            context,
            (
                "caption_text",
                "video_caption_text",
                "audio_caption_text",
            ),
        )
        or row.get("caption_text")
        or row.get("figcaption")
    ):
        score += 0.4

    if (
        _has_text(
            context,
            (
                "transcript_text",
                "transcript_preview",
                "transcript_segments",
            ),
        )
        or row.get("transcript_text")
        or row.get("transcript_preview")
        or row.get("transcript_segments")
    ):
        score += 0.35

    if (
        _has_text(
            context,
            ("frame_ocr_text", "frame_ocr_preview"),
        )
        or row.get("frame_ocr_text")
        or row.get("frame_ocr_preview")
        or row.get("ocr_text")
        or row.get("ocr_preview")
    ):
        score += 0.35

    if (
        _has_text(
            context,
            (
                "video_summary_text",
                "audio_summary_text",
                "summary_text",
            ),
        )
        or row.get("video_summary_text")
        or row.get("audio_summary_text")
        or row.get("summary_text")
    ):
        score += 0.25

    if context.get("alt_text") or row.get("alt_text"):
        score += 0.2

    if (
        _has_text(
            context,
            (
                "surrounding_text",
                "html_context",
                "page_title",
                "parent_title",
                "parent_text_preview",
            ),
        )
        or row.get("surrounding_text")
        or row.get("html_context")
        or row.get("page_title")
    ):
        score += 0.2

    if parent_text:
        score += 0.1

    asset_fetch_mode = row.get("asset_fetch_mode") or context.get(
        "asset_fetch_mode"
    )

    if str(asset_fetch_mode or "").strip() in {
        "full",
        "full_payload",
    }:
        score += 0.1

    return min(score, 1.0)


def _select_approved_candidate(
    clearance: PrivacyClearance | PrivacyClearanceRecord,
    candidates: tuple[tuple[str, str, object], ...],
) -> tuple[str, str] | None:
    for field_name, source, value in candidates:
        approved = clearance.approved_text(field_name)
        if (
            isinstance(value, str)
            and value
            and approved is not None
            and value == approved
        ):
            return approved, source
    return None


def normalize_text(value: object) -> str | None:
    """Collapse whitespace and reject empty text values."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _has_text(context: Mapping[str, object], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple)) and value:
            return True
    return False


__all__ = [
    "PairableMedia",
    "TimedMedia",
    "DocumentTextRejection",
    "fallback_text",
    "image_quality",
    "normalize_text",
    "pairability_score",
    "read_document_text",
    "select_image_caption",
    "select_image_text",
    "select_media_text",
    "select_speech_transcript",
    "select_video_caption",
]
