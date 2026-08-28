"""Canonical media-kind acceptance data and unambiguous matching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from crawler.classification.media_kind import MediaKind
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)


@dataclass(frozen=True, slots=True)
class MediaKindDefinition:
    """Accepted extensions and MIME types for one media kind."""

    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]


MEDIA_KIND_DEFINITIONS: dict[MediaKind, MediaKindDefinition] = {
    MediaKind.PAGE: MediaKindDefinition(
        extensions=(
            ".html",
            ".htm",
        ),
        mime_types=(
            "text/html",
            "application/xhtml+xml",
            "text/plain",
        ),
    ),
    MediaKind.FEED: MediaKindDefinition(
        extensions=(
            ".rss",
            ".xml",
            ".atom",
        ),
        mime_types=(
            "application/rss+xml",
            "application/atom+xml",
            "application/xml",
            "text/xml",
            "application/json",
        ),
    ),
    MediaKind.IMAGE: MediaKindDefinition(
        # Supported training images are raster formats only.
        # SVG may be detected elsewhere but is intentionally not accepted
        # as a supported training-image payload.
        extensions=(
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".tif",
            ".tiff",
            ".avif",
            ".bmp",
        ),
        mime_types=(
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/tiff",
            "image/avif",
            "image/bmp",
        ),
    ),
    MediaKind.AUDIO: MediaKindDefinition(
        extensions=(
            ".mp3",
            ".wav",
            ".ogg",
            ".m4a",
            ".webm",
            ".aac",
            ".flac",
            ".opus",
            ".aiff",
            ".mpga",
            ".mpeg",
        ),
        mime_types=(
            "audio/mpeg",
            "audio/mp4",
            "audio/mp4a-latm",
            "audio/ogg",
            "audio/wav",
            "audio/webm",
            "audio/x-m4a",
            "audio/x-wav",
            "audio/aac",
            "audio/flac",
        ),
    ),
    MediaKind.VIDEO: MediaKindDefinition(
        extensions=(
            ".mp4",
            ".webm",
            ".mov",
            ".m4v",
            ".mkv",
            ".avi",
            ".m3u8",
        ),
        mime_types=(
            "video/mp4",
            "video/webm",
            "video/quicktime",
            "video/ogg",
            "video/x-matroska",
            "video/x-msvideo",
        ),
    ),
    MediaKind.DOCUMENT: MediaKindDefinition(
        extensions=(
            ".pdf",
            ".txt",
            ".md",
            ".vtt",
            ".srt",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".zip",
            ".epub",
            ".json",
            ".csv",
            ".tsv",
            ".ttml",
            ".ris",
            ".xls",
            ".xlsx",
        ),
        mime_types=(
            "application/pdf",
            "application/ttml+xml",
            "application/x-subrip",
            "text/plain",
            "text/markdown",
            "text/srt",
            "text/vtt",
            "application/msword",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation",
            "application/zip",
            "application/x-zip-compressed",
            "application/epub+zip",
            "application/json",
            "text/csv",
            "text/tab-separated-values",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet",
            "application/x-research-info-systems",
        ),
    ),
}

_AMBIGUOUS_EXTENSIONS = frozenset(
    {
        ".json",
        ".mpeg",
        ".webm",
        ".xml",
    }
)

_AMBIGUOUS_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/octet-stream",
        "application/ogg",
        "application/xml",
        "text/plain",
        "text/xml",
    }
)

_MIME_PREFIX_KINDS: tuple[
    tuple[str, MediaKind],
    ...,
] = (
    ("image/", MediaKind.IMAGE),
    ("audio/", MediaKind.AUDIO),
    ("video/", MediaKind.VIDEO),
)

_extension_kinds_builder: dict[str, set[MediaKind]] = defaultdict(set)
_mime_kinds_builder: dict[str, set[MediaKind]] = defaultdict(set)

for _kind, _definition in MEDIA_KIND_DEFINITIONS.items():
    for _extension in _definition.extensions:
        _normalized = _extension.strip().casefold()
        if _normalized and not _normalized.startswith("."):
            _normalized = f".{_normalized}"
        if _normalized:
            _extension_kinds_builder[_normalized].add(_kind)

    for _mime_type in _definition.mime_types:
        _normalized = normalize_mime_type(_mime_type)
        if _normalized:
            _mime_kinds_builder[_normalized].add(_kind)

_EXTENSION_KINDS = {
    ext: frozenset(kinds) for ext, kinds in _extension_kinds_builder.items()
}
_MIME_KINDS = {
    mime: frozenset(kinds) for mime, kinds in _mime_kinds_builder.items()
}

del _extension_kinds_builder, _mime_kinds_builder


def candidate_suffixes(
    *,
    url: str,
) -> tuple[str, ...]:
    """Return compound and individual suffix candidates from a URL."""

    try:
        path = urlsplit(url).path.casefold()
    except ValueError:
        return ()

    if not path:
        return ()

    suffixes = [
        suffix.casefold() for suffix in PurePosixPath(path).suffixes if suffix
    ]

    if not suffixes:
        return ()

    candidates: list[str] = []

    for start_index in range(len(suffixes)):
        combined = "".join(suffixes[start_index:])

        if combined not in candidates:
            candidates.append(combined)

    for suffix in reversed(suffixes):
        if suffix not in candidates:
            candidates.append(suffix)

    return tuple(candidates)


def definition_for(
    kind: MediaKind,
) -> MediaKindDefinition:
    """Return acceptance data for one kind."""

    return MEDIA_KIND_DEFINITIONS[kind]


def known_extensions() -> frozenset[str]:
    """Return all registered extensions, including ambiguous ones."""

    return frozenset(
        extension
        for definition in MEDIA_KIND_DEFINITIONS.values()
        for extension in definition.extensions
    )


def match_extension(
    url: str,
) -> MediaKind | None:
    """Resolve an unambiguous media kind from URL suffix evidence."""

    for suffix in candidate_suffixes(url=url):
        if suffix in _AMBIGUOUS_EXTENSIONS:
            return None

        candidates = _EXTENSION_KINDS.get(
            suffix,
            frozenset(),
        )

        if len(candidates) == 1:
            return next(iter(candidates))

        if candidates:
            return None

    return None


def match_mime_type(
    mime_type: str,
) -> MediaKind | None:
    """Resolve an unambiguous media kind from MIME evidence."""

    normalized = normalize_mime_type(mime_type)

    if normalized is None or normalized in _AMBIGUOUS_MIME_TYPES:
        return None

    candidates = _MIME_KINDS.get(normalized)

    if candidates is None:
        for prefix, kind in _MIME_PREFIX_KINDS:
            if normalized.startswith(prefix):
                return kind
        return None

    if len(candidates) != 1:
        return None

    return next(iter(candidates))


__all__ = [
    "MediaKindDefinition",
    "definition_for",
    "known_extensions",
    "candidate_suffixes",
    "match_extension",
    "match_mime_type",
]
