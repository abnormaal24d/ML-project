"""Build stable media fingerprints after normalization."""

from __future__ import annotations

import hashlib

from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.provenance import stable_identifier


def build_media_fingerprint(
    *,
    modality: str,
    item: PreprocessingInput,
    primary_text: str | None,
) -> dict[str, str]:
    """Return media and optional semantic-text fingerprints."""

    parts = (
        item.normalized_url or item.source_url,
        item.media_path or "",
        str(item.byte_size or ""),
        str(item.duration_seconds or ""),
        str(item.width or ""),
        str(item.height or ""),
    )
    media_fingerprint = stable_identifier(
        prefix=f"{modality}_fingerprint",
        parts=parts,
    )
    fingerprints = {
        "media_fingerprint": media_fingerprint,
    }
    if primary_text:
        fingerprints["semantic_text_hash"] = stable_identifier(
            prefix=f"{modality}_text",
            parts=(" ".join(primary_text.casefold().split()),),
        )
    return fingerprints


def build_media_id(
    *,
    modality: str,
    item: PreprocessingInput,
) -> str:
    """Build the canonical preprocessed media identifier."""

    return stable_identifier(
        prefix=modality,
        parts=(
            item.normalized_url or item.source_url,
            item.media_path or "",
            item.source_id,
        ),
    )


def build_contextual_media_fingerprint(
    *,
    modality: str,
    row_source: str,
    source_url: str,
    byte_size: int,
    transcript_text: str | None,
    ocr_text: str | None,
) -> str:
    """Build a curated media fingerprint from source and semantic context."""

    semantic_text = _normalize_text(transcript_text or ocr_text) or ""
    payload = "|".join(
        (
            modality,
            row_source,
            source_url,
            str(byte_size),
            semantic_text.casefold(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None
