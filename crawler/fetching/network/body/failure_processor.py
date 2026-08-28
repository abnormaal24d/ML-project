"""Failure finalization and error translation for streamed response bodies."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import ClientPayloadError

from crawler.fetching.errors.exceptions import (
    ResponseBodyLimitExceeded,
    ResponseBodyReadCancelled,
    ResponseBodyTimeout,
)

if TYPE_CHECKING:
    from crawler.fetching.network.body.stream_writer import (
        ResponseBodyReadContext,
        ResponseBodyReadState,
    )
    from crawler.fetching.results.materializer import (
        FetchedPayloadMaterializer,
    )
    from logger.project_logger import ProjectLogger


class ResponseBodyFailureProcessor:
    """Finalize partial payloads, log failures and raise domain errors."""

    def __init__(
        self,
        *,
        payload_materializer: FetchedPayloadMaterializer,
        logger: ProjectLogger,
        monotonic_seconds: Callable[[], float],
    ) -> None:
        self._payload_materializer = payload_materializer
        self._logger = logger
        self._monotonic_seconds = monotonic_seconds

    def handle_timeout(
        self,
        *,
        exc: ResponseBodyTimeout,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
    ) -> None:
        partial_path = self._finalize_partial(
            context=context,
            state=state,
            reason=getattr(exc, "timeout_stage", "body_read_timeout"),
        )
        exc.partial_path = partial_path
        self._logger.warning(
            "response_body_stream_duration_timeout",
            **self._failure_context(
                context=context,
                state=state,
                partial_path=partial_path,
            ),
            remaining_bytes=self._remaining_bytes(
                context=context,
                state=state,
            ),
            timeout_stage=getattr(exc, "timeout_stage", "body_read"),
            max_idle_seconds=context.max_idle_seconds,
            max_stream_seconds=context.max_stream_seconds,
        )
        raise exc

    def handle_exception(
        self,
        *,
        exc: Exception | asyncio.CancelledError,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
    ) -> None:
        """Finalize partial state and re-raise a mapped body-read failure."""

        if isinstance(exc, ResponseBodyLimitExceeded):
            raise exc
        if isinstance(exc, ResponseBodyTimeout):
            self._logger.warning(
                "response_body_stream_duration_timeout",
                **self._failure_context(
                    context=context,
                    state=state,
                    partial_path=exc.partial_path,
                ),
                remaining_bytes=self._remaining_bytes(
                    context=context,
                    state=state,
                ),
                timeout_stage=getattr(exc, "timeout_stage", "body_read"),
                max_idle_seconds=context.max_idle_seconds,
                max_stream_seconds=context.max_stream_seconds,
            )
            raise exc
        if isinstance(exc, asyncio.CancelledError):
            self._raise_cancelled(context=context, state=state)
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            self._raise_async_timeout(context=context, state=state)
        if isinstance(exc, RuntimeError) and self._is_connection_closed_error(
            exc
        ):
            partial_path = self._finalize_partial(
                context=context,
                state=state,
                reason="connection_closed",
            )
            self._logger.warning(
                "response_body_connection_closed",
                **self._failure_context(
                    context=context,
                    state=state,
                    partial_path=partial_path,
                ),
                expected_bytes=int(context.content_length or 0),
                remaining_bytes=self._remaining_bytes(
                    context=context,
                    state=state,
                ),
            )
            raise exc
        if isinstance(exc, ClientPayloadError):
            partial_path = self._finalize_partial(
                context=context,
                state=state,
                reason="decompression_error",
            )
            self._logger.warning(
                "response_body_decompression_error",
                **self._failure_context(
                    context=context,
                    state=state,
                    partial_path=partial_path,
                ),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise exc
        if isinstance(exc, OSError):
            partial_path = self._finalize_partial(
                context=context,
                state=state,
                reason="read_failed",
            )
            self._logger.exception(
                "response_body_read_failed",
                **self._failure_context(
                    context=context,
                    state=state,
                    partial_path=partial_path,
                ),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise exc
        raise exc

    def _raise_cancelled(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
    ) -> None:
        partial_path = self._finalize_partial(
            context=context,
            state=state,
            reason="cancelled",
        )
        self._logger.warning(
            "response_body_read_cancelled",
            **self._failure_context(
                context=context,
                state=state,
                partial_path=partial_path,
            ),
        )
        raise ResponseBodyReadCancelled(
            url=context.response_url,
            observed_bytes=state.total,
            chunk_count=state.read_chunk_count,
            partial_path=partial_path,
        ) from None

    def _raise_async_timeout(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
    ) -> None:
        partial_path = self._finalize_partial(
            context=context,
            state=state,
            reason="timeout",
        )
        remaining_bytes = self._remaining_bytes(context=context, state=state)
        self._logger.warning(
            "response_body_stream_timeout",
            **self._failure_context(
                context=context,
                state=state,
                partial_path=partial_path,
            ),
            remaining_bytes=remaining_bytes,
            max_idle_seconds=context.max_idle_seconds,
            max_stream_seconds=context.max_stream_seconds,
        )
        raise ResponseBodyTimeout(
            url=context.response_url,
            observed_bytes=state.total,
            remaining_bytes=remaining_bytes,
            chunk_count=state.read_chunk_count,
            partial_path=partial_path,
            timeout_stage="async_timeout",
        ) from None

    def _finalize_partial(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        reason: str,
    ) -> Path | None:
        return self._payload_materializer.finalize_incomplete_payload(
            path=state.temp_path,
            reason=reason,
            url=context.response_url,
            status_code=context.status_code,
            content_length=context.source_content_length,
            observed_bytes=state.total,
            chunk_count=state.read_chunk_count,
            max_bytes=context.max_bytes,
            etag=context.etag,
            last_modified=context.last_modified,
        )

    def _failure_context(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        partial_path: Path | None,
    ) -> dict[str, object]:
        return {
            "url": context.response_url,
            "status_code": context.status_code,
            "response_header_content_length": context.content_length,
            "observed_bytes": state.total,
            "read_chunk_count": state.read_chunk_count,
            "stored_chunk_count": state.stored_chunk_count,
            "duration_seconds": round(
                self._monotonic_seconds() - context.started_at,
                4,
            ),
            "max_bytes": context.max_bytes,
            "partial_path": str(partial_path) if partial_path else None,
        }

    @staticmethod
    def _is_connection_closed_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "connection closed" in message
            or "connection reset" in message
            or "cannot write to closing transport" in message
            or "server disconnected" in message
            or "payload is not completed" in message
        )

    @staticmethod
    def _remaining_bytes(
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
    ) -> int:
        try:
            return max(
                0,
                int(context.source_content_length or 0)
                - int(state.total or 0),
            )
        except (TypeError, ValueError):
            return 0
