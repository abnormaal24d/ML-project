"""Synthetic fetch-result construction for metadata-only media."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from crawler.classification.content_classifier import ClassifiedContent
from crawler.classification.media_kind import MediaKind
from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.fetching.media.strategy import HeadPreflightAction
from crawler.fetching.results.result import FetchResult

if TYPE_CHECKING:
    from crawler.fetching.media.strategy import HeadPreflightResult
    from crawler.fetching.request.context import FetchRequestContext
    from crawler.fetching.results.materializer import (
        FetchedPayloadMaterializer,
    )
    from logger.project_logger import ProjectLogger


_EMBED_METADATA_MIME_TYPE = "video/embed+json"


class MediaMetadataResultBuilder:
    """Build persisted synthetic results without issuing a media GET."""

    def __init__(
        self,
        *,
        payload_materializer: FetchedPayloadMaterializer,
        logger: ProjectLogger,
        now_utc: Callable[[], datetime],
    ) -> None:
        self._payload_materializer = payload_materializer
        self._logger = logger
        self._now_utc = now_utc

    def build_embed_metadata(
        self,
        *,
        context: FetchRequestContext,
    ) -> FetchResult:
        """Build a metadata-only video result for an embedded player URL."""

        final_url = (
            self._context_text(
                context=context,
                key="embed_url",
            )
            or context.url
        )

        status_code = 200

        headers = {
            "Content-Type": _EMBED_METADATA_MIME_TYPE,
            "X-Crawler-Fetch-Mode": "embed_metadata",
        }

        result = self._build_metadata_result(
            context=context,
            final_url=final_url,
            status_code=status_code,
            headers=headers,
            body=self._embed_metadata_body(
                context=context,
                final_url=final_url,
                status_code=status_code,
                headers=headers,
            ),
            fetch_mode="embed_metadata",
            source_content_length=None,
            classified=ClassifiedContent(
                raw_content_type_header=_EMBED_METADATA_MIME_TYPE,
                normalized_mime_type=_EMBED_METADATA_MIME_TYPE,
                encoding=None,
                encoding_confidence=None,
                language=None,
                language_confidence=None,
                kind=MediaKind.VIDEO,
                category="embed_metadata",
                relevance_score=0.5,
                mime_conflict=False,
            ),
        )

        self._logger.info(
            "fetch_completed_embed_metadata",
            url=context.url,
            final_url=final_url,
            status_code=status_code,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            result_kind=result.kind,
            body_read_mode="embed_metadata",
            observed_bytes=0,
            metadata_status="embed_metadata",
            embed_host=self._context_text(
                context=context,
                key="embed_host",
            ),
            source_page_url=self._context_text(
                context=context,
                key="source_page_url",
            ),
        )

        return result

    def build_head_only_metadata(
        self,
        *,
        context: FetchRequestContext,
        head_preflight_result: HeadPreflightResult,
    ) -> FetchResult:
        """Build a metadata-only result from HEAD without issuing a GET."""

        if (
            head_preflight_result.action
            is not HeadPreflightAction.METADATA_ONLY
        ):
            raise ValueError(
                "head-only metadata requires metadata_only action"
            )

        final_url = head_preflight_result.final_url or context.url

        status_code = (
            int(head_preflight_result.status_code)
            if head_preflight_result.status_code is not None
            else 200
        )

        headers = self._head_only_response_headers(
            head_preflight_result=head_preflight_result,
        )

        kind = MediaKind.parse(
            head_preflight_result.record_kind or context.requested_kind
        )

        normalized_mime = self._normalized_head_mime(
            content_type=head_preflight_result.content_type,
            kind=kind,
        )

        result = self._build_metadata_result(
            context=context,
            final_url=final_url,
            status_code=status_code,
            headers=headers,
            body=self._head_only_metadata_body(
                context=context,
                final_url=final_url,
                status_code=status_code,
                headers=headers,
                head_preflight_result=head_preflight_result,
            ),
            fetch_mode="head_only_oversized",
            source_content_length=(head_preflight_result.content_length),
            classified=ClassifiedContent(
                raw_content_type_header=(head_preflight_result.content_type),
                normalized_mime_type=normalized_mime,
                encoding=None,
                encoding_confidence=None,
                language=None,
                language_confidence=None,
                kind=kind,
                category=None,
                relevance_score=0.0,
                mime_conflict=False,
            ),
        )

        self._logger.info(
            "fetch_completed_head_only_metadata",
            url=context.url,
            final_url=final_url,
            status_code=status_code,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            result_kind=result.kind,
            body_read_mode="head_only_oversized",
            observed_bytes=0,
            source_content_length=(head_preflight_result.content_length),
            metadata_status="head_only_oversized",
        )

        return result

    def _build_metadata_result(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
        status_code: int,
        headers: Mapping[str, str],
        body: bytes,
        fetch_mode: str,
        source_content_length: int | None,
        classified: ClassifiedContent,
    ) -> FetchResult:
        """Persist a synthetic payload and build its immutable fetch result."""

        read_result = self._payload_materializer.write_synthetic_payload(
            url=final_url,
            status_code=status_code,
            body=body,
            fetch_mode=fetch_mode,
            source_content_length=source_content_length,
            observed_bytes=0,
        )

        return FetchResult.build(
            requested_url=context.url,
            final_url=final_url,
            status_code=status_code,
            response_headers=headers,
            payload=read_result.payload,
            body_sha256=read_result.sha256,
            classified=classified,
            fetched_at=self._now_utc().isoformat(),
        )

    def _embed_metadata_body(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
        status_code: int,
        headers: Mapping[str, str],
    ) -> bytes:
        """Build the persisted JSON body for embedded-player metadata."""

        embed_url = (
            self._context_text(
                context=context,
                key="embed_url",
            )
            or final_url
        )

        return self._json_bytes(
            {
                "url": context.url,
                "final_url": final_url,
                "status_code": status_code,
                "kind": context.requested_kind,
                "requested_kind": context.requested_kind,
                "metadata_status": "embed_metadata",
                "fetch_mode": "embed_metadata",
                "asset_fetch_mode": "embed_metadata",
                "observed_bytes": 0,
                "source_content_length": None,
                "embed_url": embed_url,
                "embed_host": self._context_text(
                    context=context,
                    key="embed_host",
                ),
                "source_page_url": self._context_text(
                    context=context,
                    key="source_page_url",
                ),
                "parent_title": self._context_text(
                    context=context,
                    key="parent_title",
                ),
                "parent_text_preview": self._context_text(
                    context=context,
                    key="parent_text_preview",
                ),
                "discovery_reason": self._context_text(
                    context=context,
                    key="discovery_reason",
                ),
                "task_context": self._json_safe_context(
                    context=context,
                ),
                "headers": dict(headers),
            }
        )

    @staticmethod
    def _head_only_response_headers(
        *,
        head_preflight_result: HeadPreflightResult,
    ) -> dict[str, str]:
        """Build persisted response headers from HEAD evidence."""

        headers = dict(head_preflight_result.headers or {})

        if head_preflight_result.content_type:
            headers.setdefault(
                "Content-Type",
                head_preflight_result.content_type,
            )

        if head_preflight_result.content_length is not None:
            headers.setdefault(
                "Content-Length",
                str(head_preflight_result.content_length),
            )

        return headers

    def _head_only_metadata_body(
        self,
        *,
        context: FetchRequestContext,
        final_url: str,
        status_code: int,
        headers: Mapping[str, str],
        head_preflight_result: HeadPreflightResult,
    ) -> bytes:
        """Build the persisted JSON body for a HEAD-only media result."""

        return self._json_bytes(
            {
                "url": context.url,
                "final_url": final_url,
                "status_code": status_code,
                "kind": context.requested_kind,
                "content_type": (head_preflight_result.content_type),
                "content_length": (head_preflight_result.content_length),
                "metadata_status": ("head_only_oversized"),
                "fetch_mode": ("head_only_oversized"),
                "observed_bytes": 0,
                "source_content_length": (
                    head_preflight_result.content_length
                ),
                "head_action_reason": (head_preflight_result.action_reason),
                "headers": dict(headers),
            }
        )

    @staticmethod
    def _normalized_head_mime(
        *,
        content_type: str | None,
        kind: MediaKind,
    ) -> str | None:
        """Resolve a canonical MIME value for a HEAD-only result."""

        normalized = normalize_mime_type(content_type)

        if normalized is not None:
            return normalized

        if kind is MediaKind.VIDEO:
            return "video/mp4"

        if kind is MediaKind.AUDIO:
            return "audio/mpeg"

        return None

    def _json_safe_context(
        self,
        *,
        context: FetchRequestContext,
    ) -> dict[str, object]:
        """Return only supported JSON-safe crawl task context values."""

        safe_context: dict[str, object] = {}

        for key, value in context.task_context.items():
            if value is None:
                continue

            if isinstance(value, bool):
                safe_context[key] = value
                continue

            if isinstance(value, int):
                safe_context[key] = value
                continue

            if isinstance(value, float):
                if math.isfinite(value):
                    safe_context[key] = value
                else:
                    self._logger.debug(
                        "metadata_task_context_value_ignored",
                        key=key,
                        value_type="non_finite_float",
                    )

                continue

            if isinstance(value, str):
                safe_context[key] = value
                continue

            self._logger.debug(
                "metadata_task_context_value_ignored",
                key=key,
                value_type=type(value).__name__,
            )

        return safe_context

    @staticmethod
    def _context_text(
        *,
        context: FetchRequestContext,
        key: str,
    ) -> str | None:
        """Return normalized text from a string-valued task context field."""

        value = context.task_context.get(key)

        if not isinstance(value, str):
            return None

        text = " ".join(value.split())

        return text or None

    @staticmethod
    def _json_bytes(
        payload: Mapping[str, object],
    ) -> bytes:
        """Serialize one metadata payload as deterministic valid JSON."""

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
