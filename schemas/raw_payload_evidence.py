"""Raw payload evidence matching for persisted dataset artifacts.

Owns the canonical modality → (MIME, suffix) admissibility policy.
Both the crawler writer and the datachecker validator consume this
module independently, keeping the raw artifact invariant in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawPayloadEvidencePolicy:
    """Evidence allowed for persisted raw-dataset payloads."""

    mime_types: frozenset[str] = frozenset()
    mime_prefixes: tuple[str, ...] = ()
    suffixes: frozenset[str] = frozenset()


SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)

RAW_PAYLOAD_EVIDENCE_POLICIES: dict[
    str,
    RawPayloadEvidencePolicy,
] = {
    "page": RawPayloadEvidencePolicy(
        mime_types=frozenset(
            {
                "text/html",
                "application/xhtml+xml",
            }
        ),
        suffixes=frozenset(
            {
                ".htm",
                ".html",
            }
        ),
    ),
    "document": RawPayloadEvidencePolicy(
        mime_types=frozenset(
            {
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document",
                "application/x-research-info-systems",
            }
        ),
        suffixes=frozenset(
            {
                ".pdf",
                ".doc",
                ".docx",
                ".ris",
            }
        ),
    ),
    "image": RawPayloadEvidencePolicy(
        mime_types=SUPPORTED_IMAGE_MIME_TYPES,
    ),
    "audio": RawPayloadEvidencePolicy(
        mime_prefixes=("audio/",),
        suffixes=frozenset(
            {
                ".aac",
                ".flac",
                ".m4a",
                ".mp3",
                ".ogg",
                ".wav",
            }
        ),
    ),
    "video": RawPayloadEvidencePolicy(
        mime_prefixes=("video/",),
        suffixes=frozenset(
            {
                ".mkv",
                ".mov",
                ".mp4",
                ".webm",
            }
        ),
    ),
}


def _normalize_optional_suffix(suffix: str | None) -> str | None:
    """Normalize optional suffix evidence."""

    if suffix is None:
        return None

    normalized = suffix.strip().casefold()

    if not normalized:
        return None

    if not normalized.startswith("."):
        normalized = f".{normalized}"

    return normalized


def raw_payload_evidence_matches(
    *,
    modality: str,
    mime_type: str | None,
    suffix: str | None,
) -> bool:
    """Return whether evidence permits one persisted raw payload.

    The ``modality`` is a plain string (e.g. ``"page"``, ``"image"``),
    not a ``MediaKind`` enum.  This keeps the raw artifact invariant
    independent of the crawler classification stack.
    """

    normalized_modality = modality.strip().casefold()
    policy = RAW_PAYLOAD_EVIDENCE_POLICIES.get(normalized_modality)

    if policy is None:
        return False

    normalized_suffix = _normalize_optional_suffix(suffix)

    if mime_type is not None:
        normalized_mime = mime_type.strip().lower() or None
    else:
        normalized_mime = None

    if normalized_mime is not None and normalized_mime in policy.mime_types:
        return True

    if normalized_mime is not None:
        for prefix in policy.mime_prefixes:
            if normalized_mime.startswith(prefix):
                return True

    return (
        normalized_suffix is not None and normalized_suffix in policy.suffixes
    )


__all__ = [
    "RAW_PAYLOAD_EVIDENCE_POLICIES",
    "RawPayloadEvidencePolicy",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "raw_payload_evidence_matches",
]
