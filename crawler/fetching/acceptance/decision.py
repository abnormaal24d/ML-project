"""Resolved fetch acceptance schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)

if TYPE_CHECKING:
    from crawler.classification.media_kind import MediaKind


@dataclass(frozen=True, slots=True)
class FetchAcceptance:
    """Resolved request acceptance for one fetch operation."""

    requested_kind: MediaKind
    allowed_content_types: tuple[str, ...]
    max_bytes: int
    allow_metadata_only_when_oversized: bool = False
    allow_streaming_when_oversized: bool = False
    allow_partial_when_oversized: bool = False
    max_bytes_by_content_type: dict[str, int] = field(default_factory=dict)

    def allows_content_type(self, content_type: str | None) -> bool:
        """Return whether the response content type is allowed."""

        if not self.allowed_content_types:
            return True
        if content_type is None:
            return True
        if content_type in self.allowed_content_types:
            return True

        for candidate in self.allowed_content_types:
            if candidate.endswith("/*") and content_type.startswith(
                candidate[:-1]
            ):
                return True

        return False

    def max_bytes_for_content_type(self, content_type: str | None) -> int:
        """Return the most specific configured transport cap for a response."""

        if content_type is None:
            return int(self.max_bytes)

        normalized = normalize_mime_type(content_type)
        if not normalized:
            return int(self.max_bytes)

        exact = self.max_bytes_by_content_type.get(normalized)
        if exact is not None:
            return int(exact)

        for candidate, max_bytes in self.max_bytes_by_content_type.items():
            if candidate.endswith("/*") and normalized.startswith(
                candidate[:-1]
            ):
                return int(max_bytes)

        return int(self.max_bytes)
