"""Materialize persisted fetched payloads from read state or metadata."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crawler.fetching.results.payload import FetchedPayload

if TYPE_CHECKING:
    from crawler.fetching.network.body.partial_store import (
        PartialPayloadStorage,
    )
    from crawler.fetching.network.body.stream_writer import (
        ResponseBodyReadContext,
        ResponseBodyReadState,
    )
    from logger.project_logger import ProjectLogger


@dataclass(frozen=True, slots=True)
class ResponseBodyReadResult:
    """Persisted payload and integrity facts produced by materialization."""

    payload: FetchedPayload
    sha256: str
    byte_size: int
    chunk_count: int
    truncated: bool = False
    source_content_length: int | None = None

    @property
    def has_body_bytes(self) -> bool:
        return self.byte_size > 0

    @property
    def is_shorter_than_declared_source(self) -> bool:
        if self.source_content_length is None:
            return False
        return self.byte_size < int(self.source_content_length)


class FetchedPayloadMaterializer:
    """Materialize completed, synthetic, and partial fetched payloads."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        partial_payload_storage: PartialPayloadStorage,
        temporary_directory: Path,
        sniff_byte_count: int,
        monotonic_seconds: Callable[[], float],
    ) -> None:
        self._logger = logger
        self._partial_payload_storage = partial_payload_storage
        self._monotonic_seconds = monotonic_seconds
        self._temporary_directory = temporary_directory
        self._sniff_byte_count = max(1, int(sniff_byte_count))

    def build_read_result(
        self,
        *,
        context: ResponseBodyReadContext,
        state: ResponseBodyReadState,
        fetch_mode: str,
    ) -> ResponseBodyReadResult:
        resolved_fetch_mode = self._resolve_fetch_mode(
            read_mode=fetch_mode,
            truncated=state.truncated,
        )
        payload_complete = self._is_complete_payload(
            fetch_mode=resolved_fetch_mode,
            truncated=state.truncated,
            observed_bytes=state.total,
            source_content_length=context.source_content_length,
        )

        if state.stored_chunk_count == 0:
            self._logger.debug(
                "response_body_empty",
                url=context.response_url,
                status_code=context.status_code,
                response_header_content_length=context.content_length,
                duration_seconds=round(
                    self._monotonic_seconds() - context.started_at, 4
                ),
            )

        sha256 = state.digest.hexdigest()
        duration_seconds = round(
            self._monotonic_seconds() - context.started_at, 4
        )
        payload = FetchedPayload(
            temp_path=state.temp_path,
            byte_size=state.total,
            sha256_hex=sha256,
            sniff_bytes=bytes(state.sniff_buffer),
            chunk_count=state.stored_chunk_count,
            truncated=state.truncated,
            source_content_length=context.source_content_length,
            fetch_mode=resolved_fetch_mode,
            is_complete_payload=payload_complete,
            observed_bytes=state.total,
            duration_seconds=duration_seconds,
        )

        self._logger.debug(
            "response_body_read_completed",
            url=context.response_url,
            status_code=context.status_code,
            response_header_content_length=context.content_length,
            byte_size=state.total,
            read_chunk_count=state.read_chunk_count,
            stored_chunk_count=state.stored_chunk_count,
            chunk_count=state.stored_chunk_count,
            truncated=state.truncated,
            source_content_length=context.source_content_length,
            fetch_mode=resolved_fetch_mode,
            is_complete_payload=payload_complete,
            sha256_prefix=sha256[:12],
            payload_path=str(state.temp_path),
            duration_seconds=duration_seconds,
        )

        return ResponseBodyReadResult(
            payload=payload,
            sha256=sha256,
            byte_size=state.total,
            chunk_count=state.stored_chunk_count,
            truncated=state.truncated,
            source_content_length=context.source_content_length,
        )

    def write_synthetic_payload(
        self,
        *,
        url: str,
        status_code: int,
        body: bytes,
        fetch_mode: str,
        source_content_length: int | None,
        observed_bytes: int = 0,
    ) -> ResponseBodyReadResult:
        """
        Persist a small synthetic metadata payload without reading a body.
        """

        normalized_body = bytes(body)
        digest = hashlib.sha256(normalized_body)
        sha256 = digest.hexdigest()
        fd, temp_path = self._partial_payload_storage.create_temp_file(
            directory=self._temporary_directory,
        )
        try:
            with os.fdopen(fd, "wb") as temp_file:
                temp_file.write(normalized_body)
        except OSError:
            temp_path.unlink(missing_ok=True)
            raise

        chunk_count = 1 if normalized_body else 0
        payload = FetchedPayload(
            temp_path=temp_path,
            byte_size=len(normalized_body),
            sha256_hex=sha256,
            sniff_bytes=normalized_body[: self._sniff_byte_count],
            chunk_count=chunk_count,
            truncated=False,
            source_content_length=source_content_length,
            fetch_mode=fetch_mode,
            is_complete_payload=False,
            observed_bytes=int(observed_bytes),
            duration_seconds=0.0,
        )

        self._logger.debug(
            "synthetic_metadata_payload_written",
            url=url,
            status_code=status_code,
            byte_size=len(normalized_body),
            source_content_length=source_content_length,
            fetch_mode=fetch_mode,
            observed_bytes=observed_bytes,
            sha256_prefix=sha256[:12],
            payload_path=str(temp_path),
        )

        return ResponseBodyReadResult(
            payload=payload,
            sha256=sha256,
            byte_size=len(normalized_body),
            chunk_count=chunk_count,
            truncated=False,
            source_content_length=source_content_length,
        )

    def finalize_incomplete_payload(
        self,
        *,
        path: Path,
        reason: str,
        url: str,
        status_code: int,
        content_length: int | None,
        observed_bytes: int,
        chunk_count: int,
        max_bytes: int,
        etag: str | None,
        last_modified: str | None,
    ) -> Path | None:
        """
        Delete or preserve a partial payload using explicit reader settings.
        """

        preserved = self._partial_payload_storage.finalize_incomplete_payload(
            path=path,
            reason=reason,
            url=url,
            status_code=status_code,
            content_length=content_length,
            observed_bytes=observed_bytes,
            chunk_count=chunk_count,
            max_bytes=max_bytes,
            etag=etag,
            last_modified=last_modified,
        )
        return Path(preserved) if preserved is not None else None

    @staticmethod
    def _resolve_fetch_mode(*, read_mode: str, truncated: bool) -> str:
        normalized = str(read_mode).strip().lower()
        if normalized in {"metadata_only", "metadata_probe"}:
            return "metadata_only"
        if normalized == "fetch_partial" or truncated:
            return "partial"
        return "full"

    @staticmethod
    def _is_complete_payload(
        *,
        fetch_mode: str,
        truncated: bool,
        observed_bytes: int,
        source_content_length: int | None,
    ) -> bool:
        if fetch_mode != "full" or truncated:
            return False
        if source_content_length is None:
            return True
        return observed_bytes >= int(source_content_length)
