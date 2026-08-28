"""Stream response body bytes to temporary payload files."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from crawler.fetching.errors.exceptions import (
    ResponseBodyDecompressionLimitExceeded,
    ResponseBodyLimitExceeded,
    ResponseBodyTimeout,
)
from logger.project_logger import ProjectLogger


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ResponseBodyReadContext:
    """Immutable rules and response metadata for one streamed body."""

    response_url: str
    content_length: int | None
    source_content_length: int | None
    content_encoding: str
    etag: str | None
    last_modified: str | None
    status_code: int
    content_type: str
    chunk_size: int
    max_bytes: int
    first_byte_timeout_seconds: float
    read_chunk_timeout_seconds: float
    max_idle_seconds: float
    max_stalled_reads: int
    max_stream_seconds: float
    max_decompression_ratio: float
    started_at: float


@dataclass(slots=True)
class ResponseBodyReadState:
    """Mutable counters and temporary-file state for one streamed body."""

    fd: int
    temp_path: Path
    digest: _Digest
    sniff_buffer: bytearray
    sniff_byte_count: int
    total: int = 0
    read_chunk_count: int = 0
    stored_chunk_count: int = 0
    truncated: bool = False


class PayloadStreamWriter:
    """Read response chunks, write payload bytes, and apply truncation."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        monotonic_seconds: Callable[[], float],
        max_in_flight_bytes: int,
        bytes_per_second: int,
    ) -> None:
        if max_in_flight_bytes <= 0:
            raise ValueError("max_in_flight_bytes must be positive")
        if bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        self._logger = logger
        self._monotonic_seconds = monotonic_seconds
        self._byte_capacity = max_in_flight_bytes
        self._available_bytes = max_in_flight_bytes
        self._bytes_per_second = bytes_per_second
        self._capacity_changed = asyncio.Condition()
        self._pacing_lock = asyncio.Lock()
        self._next_transfer_at = 0.0

    async def stream_response_to_temp_file(
        self,
        *,
        response: Any,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        allow_partial: bool,
    ) -> None:
        with os.fdopen(state.fd, "wb") as temp_file:
            stalled_reads = 0
            last_progress_at = context.started_at

            while True:
                # Completion check MUST happen before timeout enforcement.
                # A timeout with remaining_bytes=0 after receiving the
                # expected total is a false timeout (server may not have sent
                # final chunk marker yet).
                expected_total = context.source_content_length
                if expected_total is not None:
                    try:
                        if state.total >= int(expected_total):
                            break
                    except (TypeError, ValueError):
                        pass

                self._raise_if_stream_timed_out(
                    context=context,
                    state=state,
                    last_progress_at=last_progress_at,
                )
                try:
                    timeout_seconds = self._chunk_timeout_seconds(
                        context=context,
                        state=state,
                    )
                    await self._acquire_bytes(context.chunk_size)
                    try:
                        chunk = await asyncio.wait_for(
                            response.content.read(context.chunk_size),
                            timeout=timeout_seconds,
                        )
                        await self._throttle_bytes(len(chunk))
                        if chunk:
                            should_continue = await asyncio.to_thread(
                                self._store_chunk,
                                context=context,
                                state=state,
                                chunk=chunk,
                                allow_partial=allow_partial,
                                write_chunk=temp_file.write,
                            )
                    finally:
                        await self._release_bytes(context.chunk_size)
                except (asyncio.TimeoutError, TimeoutError):
                    stalled_reads += 1
                    self._handle_stalled_read(
                        context=context,
                        state=state,
                        stalled_reads=stalled_reads,
                        timeout_seconds=timeout_seconds,
                    )
                    continue

                stalled_reads = 0
                if not chunk:
                    break
                self._raise_if_decompression_ratio_exceeded(
                    response=response,
                    context=context,
                    state=state,
                )
                last_progress_at = self._monotonic_seconds()
                if not should_continue:
                    break

            await asyncio.to_thread(self._flush_and_sync, temp_file)

    async def _acquire_bytes(self, byte_count: int) -> None:
        self._validate_reservation(byte_count)
        async with self._capacity_changed:
            await self._capacity_changed.wait_for(
                lambda: self._available_bytes >= byte_count
            )
            self._available_bytes -= byte_count

    async def _release_bytes(self, byte_count: int) -> None:
        self._validate_reservation(byte_count)
        async with self._capacity_changed:
            next_available = self._available_bytes + byte_count
            if next_available > self._byte_capacity:
                raise RuntimeError("download byte reservation released twice")
            self._available_bytes = next_available
            self._capacity_changed.notify_all()

    async def _throttle_bytes(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte_count must not be negative")
        if byte_count == 0:
            return
        async with self._pacing_lock:
            now = self._monotonic_seconds()
            scheduled_at = max(now, self._next_transfer_at)
            self._next_transfer_at = scheduled_at + (
                byte_count / self._bytes_per_second
            )
        delay = scheduled_at - now
        if delay > 0:
            await asyncio.sleep(delay)

    def _validate_reservation(self, byte_count: int) -> None:
        if byte_count <= 0 or byte_count > self._byte_capacity:
            raise ValueError(
                "download byte reservation must be positive and no larger "
                "than max_in_flight_bytes"
            )

    def _store_chunk(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        chunk: bytes,
        allow_partial: bool,
        write_chunk: Callable[[bytes], int],
    ) -> bool:
        state.read_chunk_count += 1
        chunk_len = len(chunk)
        next_total = state.total + chunk_len

        if state.read_chunk_count == 1:
            self._logger.debug(
                "response_body_first_chunk_received",
                url=context.response_url,
                status_code=context.status_code,
                first_chunk_bytes=chunk_len,
                response_header_content_length=context.content_length,
            )

        if next_total > context.max_bytes and allow_partial:
            allowed_bytes = max(0, context.max_bytes - state.total)
            if allowed_bytes > 0:
                self._write_body_bytes(
                    state=state,
                    body=chunk[:allowed_bytes],
                    write_chunk=write_chunk,
                )
            state.truncated = True
            self._logger.debug(
                "response_body_read_truncated",
                url=context.response_url,
                status_code=context.status_code,
                response_header_content_length=context.content_length,
                source_content_length=context.source_content_length,
                observed_bytes=next_total,
                byte_size=state.total,
                max_bytes=context.max_bytes,
                read_chunk_count=state.read_chunk_count,
                stored_chunk_count=state.stored_chunk_count,
                stored_body_chunk_count=state.stored_chunk_count,
            )
            return False

        if next_total > context.max_bytes:
            self._logger.warning(
                "response_body_limit_exceeded",
                url=context.response_url,
                status_code=context.status_code,
                response_header_content_length=context.content_length,
                chunk_bytes=chunk_len,
                read_chunk_count=state.read_chunk_count,
                stored_chunk_count=state.stored_chunk_count,
                observed_bytes=next_total,
                max_bytes=context.max_bytes,
            )
            raise ResponseBodyLimitExceeded(
                url=context.response_url,
                max_bytes=context.max_bytes,
                observed_bytes=next_total,
                chunk_count=state.read_chunk_count,
            )

        self._write_body_bytes(
            state=state,
            body=chunk,
            write_chunk=write_chunk,
        )
        return True

    @staticmethod
    def _write_body_bytes(
        *,
        state: ResponseBodyReadState,
        body: bytes,
        write_chunk: Callable[[bytes], int],
    ) -> None:
        if len(state.sniff_buffer) < state.sniff_byte_count:
            remaining = state.sniff_byte_count - len(state.sniff_buffer)
            state.sniff_buffer.extend(body[:remaining])

        write_chunk(body)
        state.digest.update(body)
        state.total += len(body)
        state.stored_chunk_count += 1

    @staticmethod
    def _flush_and_sync(temp_file: Any) -> None:
        temp_file.flush()
        os.fsync(temp_file.fileno())

    @staticmethod
    def _raise_if_decompression_ratio_exceeded(
        *,
        response: Any,
        context: Any,
        state: ResponseBodyReadState,
    ) -> None:
        if context.content_encoding in {"", "identity"}:
            return
        compressed_bytes = _compressed_byte_count(response.content)
        if not isinstance(compressed_bytes, int) or compressed_bytes <= 0:
            raise ResponseBodyDecompressionLimitExceeded(
                url=context.response_url,
                observed_bytes=state.total,
                compressed_bytes=1,
                max_ratio=context.max_decompression_ratio,
                chunk_count=state.read_chunk_count,
            )
        if state.total > compressed_bytes * context.max_decompression_ratio:
            raise ResponseBodyDecompressionLimitExceeded(
                url=context.response_url,
                observed_bytes=state.total,
                compressed_bytes=compressed_bytes,
                max_ratio=context.max_decompression_ratio,
                chunk_count=state.read_chunk_count,
            )

    def _raise_if_stream_timed_out(
        self,
        *,
        context: Any,
        state: Any,
        last_progress_at: float,
    ) -> None:
        now = self._monotonic_seconds()
        if now - context.started_at >= context.max_stream_seconds:
            self._raise_response_body_timeout(
                context=context,
                state=state,
                timeout_stage="max_stream_seconds",
            )
        if (
            state.total > 0
            and now - last_progress_at >= context.max_idle_seconds
        ):
            self._raise_response_body_timeout(
                context=context,
                state=state,
                timeout_stage="max_idle_seconds",
            )

    @staticmethod
    def _chunk_timeout_seconds(*, context: Any, state: Any) -> float:
        if state.read_chunk_count == 0 and state.total == 0:
            return float(context.first_byte_timeout_seconds)
        return float(context.read_chunk_timeout_seconds)

    def _handle_stalled_read(
        self,
        *,
        context: Any,
        state: Any,
        stalled_reads: int,
        timeout_seconds: float,
    ) -> None:
        timeout_stage = (
            "first_byte_timeout"
            if state.read_chunk_count == 0 and state.total == 0
            else "read_chunk_timeout"
        )
        self._logger.warning(
            "response_body_chunk_read_stalled",
            url=context.response_url,
            status_code=context.status_code,
            response_header_content_length=context.content_length,
            observed_bytes=state.total,
            read_chunk_count=state.read_chunk_count,
            stored_chunk_count=state.stored_chunk_count,
            chunk_count=state.read_chunk_count,
            stalled_reads=stalled_reads,
            max_stalled_reads=context.max_stalled_reads,
            timeout_stage=timeout_stage,
            timeout_seconds=timeout_seconds,
            first_byte_timeout_seconds=context.first_byte_timeout_seconds,
            read_chunk_timeout_seconds=context.read_chunk_timeout_seconds,
            elapsed_seconds=round(
                self._monotonic_seconds() - context.started_at, 4
            ),
            max_idle_seconds=context.max_idle_seconds,
            max_stream_seconds=context.max_stream_seconds,
        )
        if (
            timeout_stage == "first_byte_timeout"
            or stalled_reads >= context.max_stalled_reads
        ):
            self._raise_response_body_timeout(
                context=context,
                state=state,
                timeout_stage=timeout_stage,
            )

    def _raise_response_body_timeout(
        self,
        *,
        context: Any,
        state: Any,
        timeout_stage: str,
    ) -> None:
        # Use source_content_length (resolved from Content-Range for 206
        # partials) when available; fall back to header content_length.
        expected = context.source_content_length
        remaining = 0
        try:
            if expected is not None:
                remaining = max(0, int(expected) - int(state.total or 0))
        except (TypeError, ValueError):
            remaining = 0

        raise ResponseBodyTimeout(
            url=context.response_url,
            observed_bytes=state.total,
            remaining_bytes=remaining,
            chunk_count=state.read_chunk_count,
            partial_path=state.temp_path,
            timeout_stage=timeout_stage,
        ) from None


def _compressed_byte_count(content: object) -> int | None:
    """Return transport bytes through the supported response-content contract.

    The adapter is intentionally instance based.  It accepts aiohttp's public
    runtime value when available and a project/test adapter value, while the
    caller remains fail-closed when neither supplies a positive integer.
    """

    for attribute in ("total_raw_bytes", "total_compressed_bytes"):
        value = getattr(content, attribute, None)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            return value
    return None
