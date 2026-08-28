"""Canonical content kinds used throughout the crawler."""

from __future__ import annotations

from enum import StrEnum


class MediaKind(StrEnum):
    PAGE = "page"
    FEED = "feed"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"

    @classmethod
    def parse(cls, value: str | MediaKind) -> MediaKind:
        """Parse untrusted input into a canonical crawler media kind."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("media kind must be a string or MediaKind")

        normalized = value.strip().casefold()
        if not normalized:
            raise ValueError("media kind cannot be empty")

        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unsupported media kind: {value!r}") from exc
