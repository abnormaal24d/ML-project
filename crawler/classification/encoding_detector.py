"""Deterministic text encoding detection from header, BOM and UTF-8."""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.classification import EncodingDetectorSettings


_BOMS = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

_LOW_INFORMATION_ENCODINGS = frozenset(
    {
        "cp1252",
        "iso8859-1",
        "iso-8859-1",
        "latin-1",
        "latin1",
    }
)


@dataclass(frozen=True, slots=True)
class EncodingDetection:
    """Resolved encoding with confidence and provenance."""

    value: str | None
    confidence: float | None = None
    source: str | None = None


class EncodingDetector:
    """Resolve a valid Python codec using ordered evidence."""

    def __init__(
        self,
        settings: EncodingDetectorSettings,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._charset_re = re.compile(settings.charset_pattern)

    def detect(
        self,
        *,
        content_type_header: str | None,
        body: bytes,
    ) -> EncodingDetection:
        if not self._settings.enabled:
            return EncodingDetection(None, None, None)

        header_value = content_type_header or ""
        match = self._charset_re.search(header_value)

        if match:
            encoding = _canonical_encoding(match.group(1))

            if encoding is not None:
                return EncodingDetection(
                    encoding,
                    1.0,
                    "header",
                )

            self._logger.debug(
                "encoding_detector_invalid_header_charset",
                charset=match.group(1),
            )

        sample = body[: self._settings.sample_size_bytes]

        if self._settings.detect_bom:
            for marker, encoding in _BOMS:
                if sample.startswith(marker):
                    return EncodingDetection(
                        encoding,
                        1.0,
                        "bom",
                    )

        if sample:
            try:
                sample.decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError:
                pass
            else:
                return EncodingDetection(
                    "utf-8",
                    0.95,
                    "strict_utf8",
                )

        for configured in self._settings.allowed_encodings:
            encoding = _canonical_encoding(configured)

            if encoding is None or encoding in _LOW_INFORMATION_ENCODINGS:
                continue

            if encoding.startswith(
                (
                    "utf-8",
                    "utf-16",
                    "utf-32",
                )
            ):
                continue

            try:
                sample.decode(
                    encoding,
                    errors="strict",
                )
            except UnicodeDecodeError:
                continue

            return EncodingDetection(
                encoding,
                0.45,
                "trial_decode",
            )

        self._logger.debug("encoding_detector_fallback")

        if self._settings.fallback_to_utf8:
            return EncodingDetection(
                "utf-8",
                0.1,
                "fallback",
            )

        default = _canonical_encoding(self._settings.default_encoding)

        return EncodingDetection(
            default,
            0.1 if default else None,
            "default" if default else None,
        )


def _canonical_encoding(
    value: object,
) -> str | None:
    if not isinstance(value, str):
        return None

    text = value.strip()

    if not text:
        return None

    try:
        return codecs.lookup(text).name
    except LookupError:
        return None
