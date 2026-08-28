"""Module implementation for the crawler runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from crawler.classification.content_classifier import ClassifiedContent
    from crawler.fetching.results.payload import FetchedPayload


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Immutable fetch result metadata with a persisted payload reference."""

    url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    fetched_at: str

    content_type: str | None
    mime_type: str | None
    encoding: str | None
    language: str | None
    kind: MediaKind

    language_confidence: float | None = None
    language_source: str | None = None
    language_detector_version: str | None = None

    payload: FetchedPayload | None = None
    body_sha256: str | None = None

    category: str | None = None
    relevance_score: float | None = None
    content_signature: str | None = None

    mime_conflict: bool = False

    def read_body_optional(self) -> bytes:
        if self.payload is None:
            return b""
        return self.payload.read_bytes()

    def read_body_required(self) -> bytes:
        if self.payload is None:
            raise ValueError(
                f"FetchResult payload is missing for URL: {self.final_url}"
            )
        return self.payload.read_bytes()

    @property
    def has_payload(self) -> bool:
        return self.payload is not None

    @property
    def payload_path(self) -> str | None:
        if self.payload is None:
            return None
        return str(self.payload.temp_path)

    @property
    def body_size(self) -> int:
        if self.payload is None:
            return 0
        return self.payload.byte_size

    @property
    def sniff_bytes(self) -> bytes:
        if self.payload is None:
            return b""
        return self.payload.sniff_bytes

    @classmethod
    def build(
        cls,
        *,
        requested_url: str,
        final_url: str,
        status_code: int,
        response_headers: Mapping[str, str],
        payload: FetchedPayload,
        body_sha256: str,
        classified: ClassifiedContent,
        fetched_at: str,
    ) -> FetchResult:
        from crawler.fetching.response.snapshot import safe_response_headers

        return cls(
            url=requested_url,
            final_url=final_url,
            status_code=status_code,
            headers=dict(safe_response_headers(response_headers)),
            payload=payload,
            fetched_at=fetched_at,
            content_type=classified.raw_content_type_header,
            mime_type=classified.normalized_mime_type,
            encoding=classified.encoding,
            language=classified.language,
            kind=classified.kind,
            language_confidence=classified.language_confidence,
            language_source=classified.language_source,
            language_detector_version=classified.language_detector_version,
            body_sha256=body_sha256,
            category=classified.category,
            relevance_score=classified.relevance_score,
            content_signature=cls._content_signature(
                payload=payload,
                body_sha256=body_sha256,
            ),
            mime_conflict=classified.mime_conflict,
        )

    @staticmethod
    def _content_signature(
        *,
        payload: FetchedPayload,
        body_sha256: str,
    ) -> str:
        fetch_mode = str(payload.fetch_mode or "").strip().lower()

        if fetch_mode in {
            "head_only_oversized",
            "partial_probe_failed_fallback_head_only",
        }:
            length = (
                payload.source_content_length or payload.observed_bytes or 0
            )
            return f"head-only-sha256:{body_sha256}:{length}"

        if payload.truncated:
            length = payload.source_content_length or payload.byte_size
            return f"partial-sha256:{body_sha256}:{length}"

        return f"sha256:{body_sha256}"
