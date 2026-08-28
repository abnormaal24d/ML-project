"""Media response body read decisions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from crawler.classification.media_kind import MediaKind
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.fetching.errors.exceptions import IgnoredFetchError
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.fetching.request.body_plan import BodyReadPlan
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.fetching.results.materializer import ResponseBodyReadResult


class MediaBodyReadStrategy:
    """Own media throttling and size-based pre-streaming decisions."""

    _THROTTLED_KINDS = frozenset({MediaKind.AUDIO, MediaKind.VIDEO})
    _METADATA_MODES = frozenset({"metadata_probe", "metadata_only"})

    def __init__(self, *, logger: ProjectLogger) -> None:
        self._logger = logger

    def should_throttle(self, *, requested_kind: MediaKind) -> bool:
        """Return whether a response body should use the media semaphore."""

        return requested_kind in self._THROTTLED_KINDS

    async def read_with_optional_throttle(
        self,
        *,
        requested_kind: MediaKind,
        media_semaphore: asyncio.Semaphore,
        read_body: Callable[[], Awaitable[ResponseBodyReadResult]],
    ) -> ResponseBodyReadResult:
        """
        Run a body read under the media semaphore when the kind requires it.
        """

        if self.should_throttle(requested_kind=requested_kind):
            async with media_semaphore:
                return await read_body()

        return await read_body()

    def ensure_size_acceptance_allows_streaming(
        self,
        *,
        context: FetchRequestContext,
        response: Any,
        final_url: str,
        read_plan: BodyReadPlan | None = None,
    ) -> None:
        """
        Raise when media headers show an oversized body should be skipped.
        """

        requested_kind = context.requested_kind
        if requested_kind not in self._THROTTLED_KINDS:
            return

        content_length = self._content_length_from_headers(response)
        if content_length is None:
            return

        content_type = self._content_type_from_headers(response)
        max_bytes = int(
            context.acceptance.max_bytes_for_content_type(content_type)
        )
        if content_length <= max_bytes:
            return

        if self._accepted_by_read_plan(
            context=context,
            final_url=final_url,
            requested_kind=requested_kind,
            content_length=content_length,
            max_bytes=max_bytes,
            read_plan=read_plan,
        ):
            return

        self._logger.warning(
            "fetch_skipped_oversized_media",
            url=context.url,
            final_url=final_url,
            requested_kind=requested_kind,
            acceptance_mode=context.acceptance_mode,
            content_length=content_length,
            max_bytes=max_bytes,
        )
        raise IgnoredFetchError(
            reason="content_length_exceeded",
            observed_bytes=0,
        )

    def _accepted_by_read_plan(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
        requested_kind: MediaKind,
        content_length: int,
        max_bytes: int,
        read_plan: BodyReadPlan | None,
    ) -> bool:
        if read_plan is None:
            return False

        if read_plan.mode in self._METADATA_MODES:
            self._logger.info(
                "fetch_accepted_metadata_only_media",
                url=context.url,
                final_url=final_url,
                requested_kind=requested_kind,
                acceptance_mode=context.acceptance_mode,
                content_length=content_length,
                probe_bytes=read_plan.max_bytes,
                max_bytes=max_bytes,
            )
            return True

        if read_plan.mode == "fetch_partial":
            self._logger.info(
                "fetch_accepted_partial_media",
                url=context.url,
                final_url=final_url,
                requested_kind=requested_kind,
                acceptance_mode=context.acceptance_mode,
                content_length=content_length,
                partial_bytes=read_plan.max_bytes,
                max_bytes=max_bytes,
            )
            return True

        if (
            read_plan.mode == "fetch_streaming"
            and context.acceptance.allow_streaming_when_oversized
        ):
            self._logger.info(
                "fetch_accepted_oversized_media_streaming",
                url=context.url,
                final_url=final_url,
                requested_kind=requested_kind,
                acceptance_mode=context.acceptance_mode,
                content_length=content_length,
                max_bytes=max_bytes,
                allow_partial=read_plan.allow_partial,
            )
            return True

        return False

    @staticmethod
    def _content_length_from_headers(response: Any) -> int | None:
        value = response.headers.get("Content-Length")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _content_type_from_headers(response: Any) -> str | None:
        value = response.headers.get("Content-Type")
        if value is None:
            value = response.headers.get("content-type")
        if value is None:
            return None
        return normalize_mime_type(value)
