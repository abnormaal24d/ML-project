"""Read one aiohttp response body into a persisted payload."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)
from crawler.fetching.errors.exceptions import (
    ResponseBodyLimitExceeded,
    ResponseBodyTimeout,
)
from crawler.fetching.network.body.stream_writer import (
    ResponseBodyReadContext,
    ResponseBodyReadState,
)
from crawler.fetching.response.validator import (
    read_content_type_header,
)
from crawler.fetching.results.materializer import ResponseBodyReadResult

if TYPE_CHECKING:
    from aiohttp import ClientResponse

    from config.collection.fetching import ResponseBodyReaderSettings
    from config.collection.http_rules import TimeoutRulesSettings
    from crawler.fetching.network.body.failure_processor import (
        ResponseBodyFailureProcessor,
    )
    from crawler.fetching.network.body.partial_store import (
        PartialPayloadStorage,
    )
    from crawler.fetching.network.body.stream_writer import PayloadStreamWriter
    from crawler.fetching.results.materializer import (
        FetchedPayloadMaterializer,
    )
    from logger.project_logger import ProjectLogger


class AiohttpResponseBodyReader:
    """Own context creation, streaming, recovery, and materialization."""

    def __init__(
        self,
        *,
        settings: ResponseBodyReaderSettings,
        document_content_types: frozenset[str],
        timeout_rules: TimeoutRulesSettings,
        temporary_directory: Path,
        partial_payload_storage: PartialPayloadStorage,
        failure_processor: ResponseBodyFailureProcessor,
        stream_writer: PayloadStreamWriter,
        payload_materializer: FetchedPayloadMaterializer,
        monotonic_seconds: Callable[[], float],
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._document_content_types = document_content_types
        self._timeout_rules = timeout_rules
        self._temporary_directory = temporary_directory
        self._partial_payload_storage = partial_payload_storage
        self._failure_processor = failure_processor
        self._stream_writer = stream_writer
        self._payload_materializer = payload_materializer
        self._monotonic_seconds = monotonic_seconds
        self._logger = logger

    async def read(
        self,
        response: ClientResponse,
        *,
        max_bytes: int,
        allow_partial: bool = False,
        fetch_mode: str = "full",
        resume_partial_path: Path | None = None,
        resume_owner_token: str | None = None,
    ) -> ResponseBodyReadResult:
        """Read, limit, persist, and describe one response body."""

        context = self._build_context(
            response=response,
            max_bytes=max_bytes,
        )
        self._logger.debug(
            "response_body_read_started",
            url=context.response_url,
            status_code=context.status_code,
            response_header_content_length=context.content_length,
            chunk_size=context.chunk_size,
            max_bytes=context.max_bytes,
            first_byte_timeout_seconds=context.first_byte_timeout_seconds,
            read_chunk_timeout_seconds=context.read_chunk_timeout_seconds,
            max_idle_seconds=context.max_idle_seconds,
            max_stalled_reads=context.max_stalled_reads,
            max_stream_seconds=context.max_stream_seconds,
        )
        self._raise_if_header_limit_exceeded(
            context=context,
            allow_partial=allow_partial,
        )
        state = await asyncio.to_thread(
            self._create_state,
            resume_partial_path=resume_partial_path,
            resume_owner_token=resume_owner_token,
            max_bytes=max_bytes,
        )

        result = await self._stream_and_materialize(
            response=response,
            context=context,
            state=state,
            allow_partial=allow_partial,
            fetch_mode=fetch_mode,
        )
        if resume_partial_path is not None and resume_owner_token is not None:
            await asyncio.to_thread(
                self._partial_payload_storage.complete_resume,
                path=resume_partial_path,
                owner_token=resume_owner_token,
            )
        return result

    async def _stream_and_materialize(
        self,
        *,
        response: ClientResponse,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        allow_partial: bool,
        fetch_mode: str,
    ) -> ResponseBodyReadResult:
        try:
            await self._stream_writer.stream_response_to_temp_file(
                response=response,
                context=context,
                state=state,
                allow_partial=allow_partial,
            )
        except ResponseBodyLimitExceeded:
            self._partial_payload_storage.delete(path=state.temp_path)
            raise
        except ResponseBodyTimeout as exc:
            self._failure_processor.handle_timeout(
                exc=exc,
                context=context,
                state=state,
            )
        except asyncio.CancelledError as exc:
            self._failure_processor.handle_exception(
                exc=exc,
                context=context,
                state=state,
            )
        except Exception as exc:  # exception-rules: boundary-wrap-and-raise
            self._failure_processor.handle_exception(
                exc=exc,
                context=context,
                state=state,
            )

        if (
            context.source_content_length is not None
            and context.source_content_length > state.total
            and allow_partial
        ):
            state.truncated = True

        return self._payload_materializer.build_read_result(
            context=context,
            state=state,
            fetch_mode=fetch_mode,
        )

    def _build_context(
        self,
        *,
        response: ClientResponse,
        max_bytes: int,
    ) -> ResponseBodyReadContext:
        normalized_max_bytes = int(max_bytes)
        if normalized_max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        content_length = response.content_length
        if content_length is not None and int(content_length) < 0:
            raise ValueError("response Content-Length must not be negative")
        content_encoding = (
            str(response.headers.get("Content-Encoding", "")).strip().lower()
        )
        content_type = (
            normalize_mime_type(read_content_type_header(response.headers))
            or ""
        )
        read_chunk_timeout_seconds = max(
            0.001,
            float(self._timeout_rules.read_chunk_timeout_seconds),
        )
        first_byte_timeout_seconds = max(
            0.001,
            float(self._timeout_rules.first_byte_timeout_seconds),
        )
        content_stream_timeout_seconds = (
            self._timeout_rules.body_stream_timeout_for_content_type(
                content_type=content_type,
                document_content_types=self._document_content_types,
            )
        )
        max_stream_seconds = (
            float(self._timeout_rules.max_stream_seconds)
            if self._timeout_rules.max_stream_seconds is not None
            else float(content_stream_timeout_seconds)
        )

        return ResponseBodyReadContext(
            response_url=str(response.url),
            content_length=content_length,
            source_content_length=self._resolve_source_content_length(
                response=response,
                content_length=content_length,
                content_encoding=content_encoding,
            ),
            content_encoding=content_encoding,
            etag=self._optional_header(response.headers.get("ETag")),
            last_modified=self._optional_header(
                response.headers.get("Last-Modified")
            ),
            status_code=int(response.status),
            content_type=content_type,
            chunk_size=self._resolve_chunk_size(
                response=response,
                content_type=content_type,
                max_bytes=normalized_max_bytes,
            ),
            max_bytes=normalized_max_bytes,
            first_byte_timeout_seconds=first_byte_timeout_seconds,
            read_chunk_timeout_seconds=read_chunk_timeout_seconds,
            max_idle_seconds=max(
                read_chunk_timeout_seconds,
                float(self._timeout_rules.max_idle_seconds),
            ),
            max_stalled_reads=max(1, int(self._settings.max_stalled_reads)),
            max_stream_seconds=max(0.001, max_stream_seconds),
            max_decompression_ratio=float(
                self._settings.max_decompression_ratio
            ),
            started_at=self._monotonic_seconds(),
        )

    def _create_state(
        self,
        *,
        resume_partial_path: Path | None,
        resume_owner_token: str | None,
        max_bytes: int,
    ) -> ResponseBodyReadState:
        digest = hashlib.sha256()
        sniff_buffer = bytearray()
        sniff_byte_count = max(1, int(self._settings.sniff_byte_count))

        if resume_partial_path is not None:
            if resume_owner_token is None:
                raise ValueError("resume_owner_token is required")
            fd, digest, sniff_buffer, existing_size = (
                self._partial_payload_storage.open_for_resume(
                    path=resume_partial_path,
                    owner_token=resume_owner_token,
                    max_bytes=max_bytes,
                    sniff_byte_count=sniff_byte_count,
                )
            )
            return ResponseBodyReadState(
                fd=fd,
                temp_path=Path(resume_partial_path),
                digest=digest,
                sniff_buffer=sniff_buffer,
                sniff_byte_count=sniff_byte_count,
                total=existing_size,
            )

        fd, temp_path = self._partial_payload_storage.create_temp_file(
            directory=self._temporary_directory,
        )
        return ResponseBodyReadState(
            fd=fd,
            temp_path=temp_path,
            digest=digest,
            sniff_buffer=sniff_buffer,
            sniff_byte_count=sniff_byte_count,
        )

    def _resolve_chunk_size(
        self,
        *,
        response: ClientResponse,
        content_type: str,
        max_bytes: int,
    ) -> int:
        default_chunk_size = max(1, int(self._settings.chunk_size))
        binary_chunk_size = max(
            default_chunk_size,
            int(self._settings.binary_chunk_size),
        )
        large_binary_chunk_size = max(
            binary_chunk_size,
            int(self._settings.large_binary_chunk_size),
        )
        large_body_threshold = max(
            binary_chunk_size,
            int(self._settings.large_body_threshold_bytes),
        )
        content_length = int(getattr(response, "content_length", None) or 0)

        if not self._is_binary_content_type(content_type=content_type):
            return default_chunk_size
        if (
            content_length >= large_body_threshold
            or int(max_bytes) >= large_body_threshold
        ):
            return large_binary_chunk_size
        return binary_chunk_size

    def _is_binary_content_type(self, *, content_type: str) -> bool:
        return (
            content_type.startswith(
                tuple(self._settings.binary_content_type_prefixes)
            )
            or content_type in self._document_content_types
        )

    @staticmethod
    def _resolve_source_content_length(
        *,
        response: ClientResponse,
        content_length: int | None,
        content_encoding: str,
    ) -> int | None:
        if content_encoding not in {"", "identity"}:
            return None
        content_range = str(response.headers.get("Content-Range", "")).strip()
        if "/" not in content_range:
            return content_length

        total_part = content_range.rsplit("/", 1)[-1].strip()
        if not total_part or total_part == "*":
            return content_length
        try:
            total = int(total_part)
            return total if total >= 0 else None
        except ValueError:
            return content_length

    @staticmethod
    def _optional_header(value: object) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned[:1024] if cleaned else None

    def _raise_if_header_limit_exceeded(
        self,
        *,
        context: ResponseBodyReadContext,
        allow_partial: bool,
    ) -> None:
        if (
            context.content_length is None
            or context.content_length <= context.max_bytes
            or allow_partial
        ):
            return

        self._logger.warning(
            "response_body_limit_exceeded_from_headers",
            url=context.response_url,
            status_code=context.status_code,
            response_header_content_length=context.content_length,
            max_bytes=context.max_bytes,
        )
        raise ResponseBodyLimitExceeded(
            url=context.response_url,
            max_bytes=context.max_bytes,
            observed_bytes=int(context.content_length),
            chunk_count=0,
        )
