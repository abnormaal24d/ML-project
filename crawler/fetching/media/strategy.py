"""Oversized-media transport strategy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )


class HeadPreflightAction(StrEnum):
    """Transport action selected by HEAD preflight."""

    FETCH_FULL = "fetch_full"
    FETCH_STREAMING = "fetch_streaming"
    FETCH_PARTIAL = "fetch_partial"
    METADATA_ONLY = "metadata_only"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class HeadPreflightResult:
    """Outcome metadata for an optional HEAD preflight."""

    attempted: bool
    allowed: bool = True
    action: HeadPreflightAction = HeadPreflightAction.FETCH_FULL
    status_code: int | None = None
    final_url: str | None = None
    content_type: str | None = None
    content_length: int | None = None
    failure_type: str | None = None
    soft_rejected: bool = False
    rejection_reason: str | None = None
    action_reason: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    record_kind: MediaKind | None = None


@dataclass(frozen=True, slots=True)
class MediaFetchStrategy:
    """Selected transport action for an oversized media response."""

    action: HeadPreflightAction
    reason: str


class MediaFetchStrategyResolver:
    """Resolve metadata-only, streaming or partial oversized-media handling."""

    @staticmethod
    def is_embed_metadata_context(*, context: FetchRequestContext) -> bool:
        return (
            MediaFetchStrategyResolver._context_text(
                context=context,
                key="asset_fetch_mode",
            )
            == "embed_metadata"
        )

    def resolve(
        self,
        *,
        context: FetchRequestContext,
        content_type: str | None = None,
    ) -> MediaFetchStrategy | None:
        """Return the configured oversized-media transport strategy."""

        media_kind = context.requested_kind
        normalized_content_type = (content_type or "").strip().lower()
        if media_kind not in {MediaKind.AUDIO, MediaKind.VIDEO}:
            if normalized_content_type.startswith("audio/"):
                media_kind = MediaKind.AUDIO
            elif normalized_content_type.startswith("video/"):
                media_kind = MediaKind.VIDEO

        if media_kind not in {MediaKind.AUDIO, MediaKind.VIDEO}:
            return None

        acceptance = context.acceptance
        if acceptance.allow_metadata_only_when_oversized:
            return MediaFetchStrategy(
                action=HeadPreflightAction.METADATA_ONLY,
                reason="oversized_media_metadata_only",
            )
        if acceptance.allow_streaming_when_oversized:
            return MediaFetchStrategy(
                action=HeadPreflightAction.FETCH_STREAMING,
                reason="oversized_media_streaming_allowed",
            )
        if acceptance.allow_partial_when_oversized:
            return MediaFetchStrategy(
                action=HeadPreflightAction.FETCH_PARTIAL,
                reason="oversized_media_partial_allowed",
            )
        return None

    @staticmethod
    def should_build_head_only_result(
        *,
        settings: FetcherSettings,
        context: FetchRequestContext,
        head_preflight_result: HeadPreflightResult | None,
    ) -> bool:
        """Return whether HEAD metadata can replace an oversized video GET."""

        if settings.oversized_video_metadata_mode != "head_only":
            return False
        if context.requested_kind is not MediaKind.VIDEO:
            return False
        if head_preflight_result is None or not head_preflight_result.allowed:
            return False
        if head_preflight_result.action != HeadPreflightAction.METADATA_ONLY:
            return False

        content_length = head_preflight_result.content_length
        if content_length is None:
            return (
                head_preflight_result.action_reason
                == "oversized_media_metadata_only"
            )
        return int(content_length) > int(context.acceptance.max_bytes)

    @staticmethod
    def _context_text(
        *,
        context: FetchRequestContext,
        key: str,
    ) -> str | None:
        value = context.task_context.get(key)
        if value is None:
            return None
        text = " ".join(str(value).split())
        return text or None
