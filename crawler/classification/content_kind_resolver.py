"""Resolve crawler media kind from MIME, content, URL and request intent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import (
    match_extension,
    match_mime_type,
)
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.extraction.links.sitemap_extractor import is_sitemap_markup

if TYPE_CHECKING:
    from config.settings.classification import ContentKindResolverSettings


_GENERIC_XML_MIME_TYPES = frozenset(
    {
        "application/xml",
        "text/xml",
    }
)

_GENERIC_JSON_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/jsonl",
        "application/x-ndjson",
    }
)

_CONTEXT_DEPENDENT_MIME_TYPES = (
    _GENERIC_XML_MIME_TYPES
    | _GENERIC_JSON_MIME_TYPES
    | {
        "application/octet-stream",
        "text/plain",
    }
)

_TEXT_METADATA_MIME_TYPES = frozenset(
    {
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/json",
        "application/ld+json",
        "application/jsonl",
        "application/x-ndjson",
    }
)


@dataclass(frozen=True, slots=True)
class KindResolution:
    """Resolved crawler kind and the evidence that selected it."""

    kind: MediaKind
    source: Literal[
        "mime",
        "content",
        "extension",
        "requested_kind",
        "fallback",
    ]
    role: str | None = None


class ContentKindResolver:
    """Resolve one media kind using ordered, auditable evidence."""

    def __init__(
        self,
        *,
        settings: ContentKindResolverSettings,
    ) -> None:
        self._settings = settings

        self._fallback_kind = MediaKind.parse(settings.fallback_kind)

        self._exact_kind_map = {
            mime: MediaKind.parse(kind)
            for raw_mime, kind in settings.exact_kind_map.items()
            if (mime := normalize_mime_type(raw_mime))
        }

        self._prefix_kind_map = {
            normalized_prefix: MediaKind.parse(kind)
            for prefix, kind in settings.prefix_kind_map.items()
            if (normalized_prefix := prefix.strip().casefold())
        }

        self._document_mime_types = frozenset(
            mime
            for value in settings.document_mime_types
            if (mime := normalize_mime_type(value))
        )

        self._registry = None

    def resolve(
        self,
        *,
        content_type: str | None,
        body: bytes,
        url: str = "",
        requested_kind: str | MediaKind | None = None,
    ) -> MediaKind:
        """Resolve only the final media kind."""

        return self.resolve_with_metadata(
            content_type=content_type,
            body=body,
            url=url,
            requested_kind=requested_kind,
        ).kind

    def resolve_with_metadata(
        self,
        *,
        content_type: str | None,
        body: bytes,
        url: str = "",
        requested_kind: str | MediaKind | None = None,
    ) -> KindResolution:
        """Resolve media kind together with its selecting evidence."""

        mime_type = normalize_mime_type(content_type)

        content_resolution = self._resolve_context_dependent_content(
            mime_type=mime_type,
            body=body,
        )

        if content_resolution is not None:
            return content_resolution

        mime_resolution = self._resolve_from_mime_type(mime_type)

        if mime_resolution is not None:
            return mime_resolution

        extension_resolution = self._resolve_from_extension(url)

        if extension_resolution is not None:
            return extension_resolution

        if requested_kind is not None:
            return KindResolution(
                kind=MediaKind.parse(requested_kind),
                source="requested_kind",
            )

        return KindResolution(
            kind=self._fallback_kind,
            source="fallback",
        )

    def _resolve_context_dependent_content(
        self,
        *,
        mime_type: str | None,
        body: bytes,
    ) -> KindResolution | None:
        """Resolve generic XML or JSON from bounded content evidence."""

        if mime_type in _GENERIC_XML_MIME_TYPES:
            if self.looks_like_feed(body):
                return KindResolution(
                    kind=MediaKind.FEED,
                    source="content",
                    role="feed",
                )

            if is_sitemap_markup(
                body,
                max_bytes=self._settings.feed_snippet_bytes,
            ):
                return KindResolution(
                    kind=MediaKind.PAGE,
                    source="content",
                    role="sitemap",
                )

        if mime_type in _GENERIC_JSON_MIME_TYPES and self.looks_like_json_feed(
            body
        ):
            return KindResolution(
                kind=MediaKind.FEED,
                source="content",
                role="json_feed",
            )

        return None

    def _resolve_from_mime_type(
        self,
        mime_type: str | None,
    ) -> KindResolution | None:
        """Resolve media kind from non-context-dependent MIME evidence."""

        if mime_type is None or mime_type in _CONTEXT_DEPENDENT_MIME_TYPES:
            return None

        registry_kind = match_mime_type(mime_type)

        if registry_kind is not None:
            return KindResolution(
                kind=registry_kind,
                source="mime",
            )

        exact_kind = self._exact_kind_map.get(mime_type)

        if exact_kind is not None:
            return KindResolution(
                kind=exact_kind,
                source="mime",
            )

        for prefix, kind in self._prefix_kind_map.items():
            if mime_type.startswith(prefix):
                return KindResolution(
                    kind=kind,
                    source="mime",
                )

        if mime_type in self._document_mime_types:
            return KindResolution(
                kind=MediaKind.DOCUMENT,
                source="mime",
            )

        return None

    def _resolve_from_extension(
        self,
        url: str,
    ) -> KindResolution | None:
        """Resolve media kind from URL extension evidence."""

        if not url:
            return None

        extension_kind = match_extension(url)

        if extension_kind is None:
            return None

        return KindResolution(
            kind=extension_kind,
            source="extension",
        )

    def looks_like_feed(
        self,
        body: bytes,
    ) -> bool:
        """Return whether bounded XML-like content resembles a feed."""

        sample = self._decoded_sample(body)

        if not sample:
            return False

        return any(
            normalized_marker in sample
            for marker in self._settings.feed_markers
            if (normalized_marker := marker.strip().casefold())
        )

    def looks_like_json_feed(
        self,
        body: bytes,
    ) -> bool:
        """Return whether bounded JSON content satisfies JSON Feed shape."""

        if not body:
            return False

        sample = body[: self._settings.feed_snippet_bytes]

        try:
            payload = json.loads(
                sample.decode(
                    "utf-8",
                    errors="strict",
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return False

        if not isinstance(payload, dict):
            return False

        version_value = payload.get("version")

        if not isinstance(
            version_value,
            str,
        ):
            return False

        version = version_value.strip().casefold()

        if "jsonfeed.org/version/" not in version:
            return False

        return isinstance(
            payload.get("items"),
            list,
        )

    def _decoded_sample(
        self,
        body: bytes,
    ) -> str:
        """Decode the bounded feed-detection sample."""

        if not body:
            return ""

        return (
            body[: self._settings.feed_snippet_bytes]
            .decode(
                "utf-8",
                errors="ignore",
            )
            .casefold()
        )

    @staticmethod
    def should_detect_text_metadata(
        content_type: str | None,
    ) -> bool:
        """Return whether text metadata detection applies to a MIME type."""

        mime_type = normalize_mime_type(content_type)

        if mime_type is None:
            return False

        return (
            mime_type.startswith("text/")
            or mime_type in _TEXT_METADATA_MIME_TYPES
        )
