"""Resolve response MIME evidence from headers and strong signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import unquote, urlparse

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.classification import (
        ContentTypeDetectorSettings,
    )
    from crawler.classification.mime_signature_detector import (
        MimeSignatureDetector,
    )


def normalize_mime_type(
    mime_type: str | None,
) -> str | None:
    """Return a lowercase parameter-free canonical MIME value."""

    if mime_type is None:
        return None

    normalized = mime_type.partition(";")[0].strip().casefold()

    if not normalized or "/" not in normalized:
        return None

    major, separator, subtype = normalized.partition("/")

    if not separator or not major or not subtype:
        return None

    return normalized


def mime_major_type(
    mime_type: str | None,
) -> str | None:
    """Return the normalized major MIME component."""

    normalized = normalize_mime_type(mime_type)

    if normalized is None:
        return None

    major, _, _ = normalized.partition("/")

    return major or None


_EXTENSION_TO_MIME: Final[dict[str, str]] = {
    ".pdf": "application/pdf",
    ".epub": "application/epub+zip",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".xml": "application/xml",
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".avif": "image/avif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
}


def _mime_type_from_filename_or_url(
    *,
    filename: str | None = None,
    url: str | None = None,
) -> str | None:
    """Infer MIME evidence from a filename or URL path."""

    for value, is_url in (
        (filename, False),
        (url, True),
    ):
        if not value:
            continue

        path = urlparse(value).path if is_url else value

        normalized_path = unquote(path).replace("\\", "/")

        extension = PurePosixPath(normalized_path).suffix.casefold()

        mime_type = _EXTENSION_TO_MIME.get(extension)

        if mime_type is not None:
            return mime_type

    return None


@dataclass(frozen=True, slots=True)
class MimeResolution:
    """Resolved response MIME evidence."""

    raw_header: str | None
    mime_type: str | None
    source: (
        Literal[
            "header",
            "signature",
            "extension",
            "fallback",
        ]
        | None
    )
    exact_conflict: bool = False
    major_conflict: bool = False


class MimeTypeResolver:
    """Resolve MIME evidence by signature, header, extension, then fallback."""

    def __init__(
        self,
        *,
        settings: ContentTypeDetectorSettings,
        logger: ProjectLogger,
        mime_signature_detector: MimeSignatureDetector,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._mime_signature_detector = mime_signature_detector

        self._generic_header_mime_types = frozenset(
            mime
            for value in settings.generic_header_mime_types
            if (mime := normalize_mime_type(value))
        ) | {
            "application/octet-stream",
            "binary/octet-stream",
        }

        self._signature_validation_mime_types = frozenset(
            mime
            for value in settings.signature_validation_mime_types
            if (mime := normalize_mime_type(value))
        )

        self._signature_validation_major_types = frozenset(
            value.strip().casefold()
            for value in settings.signature_validation_major_types
            if value.strip()
        )

        self._signature_exempt_mime_types = frozenset(
            mime
            for value in settings.signature_exempt_mime_types
            if (mime := normalize_mime_type(value))
        )

    def detect_resolution(
        self,
        *,
        url: str,
        content_type_header: str | None,
        sample: bytes,
        filename: str | None = None,
    ) -> MimeResolution:
        raw_header = content_type_header

        if not self._settings.enabled:
            return MimeResolution(
                raw_header,
                None,
                None,
            )

        header_mime = (
            normalize_mime_type(raw_header)
            if self._settings.detect_from_headers
            else None
        )

        signature_mime = (
            normalize_mime_type(
                self._mime_signature_detector.detect(sample=sample)
            )
            if (self._settings.detect_from_signature and sample)
            else None
        )

        exact_conflict = bool(
            header_mime and signature_mime and header_mime != signature_mime
        )

        major_conflict = bool(
            exact_conflict
            and mime_major_type(header_mime) != mime_major_type(signature_mime)
        )

        signature_trusted = bool(
            signature_mime
            and signature_mime not in self._signature_exempt_mime_types
        )

        if exact_conflict:
            self._logger.debug(
                "content_type_mismatch_detected",
                url=url,
                header_mime=header_mime,
                signature_mime=signature_mime,
                major_conflict=major_conflict,
            )

        if signature_trusted:
            return MimeResolution(
                raw_header,
                signature_mime,
                "signature",
                exact_conflict=exact_conflict,
                major_conflict=major_conflict,
            )

        trusted_header = self._trusted_header_mime(
            url=url,
            header_mime=header_mime,
            signature_mime=signature_mime,
        )

        if trusted_header is not None:
            return MimeResolution(
                raw_header,
                trusted_header,
                "header",
                exact_conflict=exact_conflict,
                major_conflict=major_conflict,
            )

        if self._settings.detect_from_extension:
            extension_mime = _mime_type_from_filename_or_url(
                filename=filename,
                url=url,
            )

            if extension_mime is not None:
                return MimeResolution(
                    raw_header,
                    extension_mime,
                    "extension",
                    exact_conflict=exact_conflict,
                    major_conflict=major_conflict,
                )

        fallback = normalize_mime_type(self._settings.fallback_content_type)

        if fallback is None:
            self._logger.debug(
                "mime_type_resolver_no_match",
                url=url,
            )

        return MimeResolution(
            raw_header,
            fallback,
            "fallback" if fallback else None,
            exact_conflict=exact_conflict,
            major_conflict=major_conflict,
        )

    def _trusted_header_mime(
        self,
        *,
        url: str,
        header_mime: str | None,
        signature_mime: str | None,
    ) -> str | None:
        if header_mime is None:
            return None

        if header_mime in self._generic_header_mime_types:
            self._logger.debug(
                "content_type_generic_header_ignored",
                url=url,
                header_mime=header_mime,
                signature_mime=signature_mime,
            )
            return None

        if not self._requires_signature_validation(header_mime):
            return header_mime

        if signature_mime == header_mime:
            return header_mime

        self._logger.debug(
            "content_type_signature_validation_failed",
            url=url,
            claimed_mime=header_mime,
            signature_mime=signature_mime,
        )

        return None

    def _requires_signature_validation(
        self,
        mime_type: str,
    ) -> bool:
        if mime_type in self._signature_exempt_mime_types:
            return False

        return (
            mime_type in self._signature_validation_mime_types
            or mime_major_type(mime_type)
            in self._signature_validation_major_types
        )
