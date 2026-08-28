"""Fetch-domain response body reading and failure translation."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import ClientResponse

from crawler.classification.media_kind import MediaKind
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    ResponseBodyLimitExceeded,
    ResponseBodyReadCancelled,
    ResponseBodyTimeout,
    RetryableFetchError,
)
from crawler.fetching.errors.translator import (
    connection_closed_error,
    ignored_range_probe_error,
    response_body_timeout_error,
    response_body_timeout_exception,
    truncated_range_probe_error,
)

_CONTENT_RANGE_PATTERN = re.compile(
    r"^bytes (?P<start>0|[1-9][0-9]*)-(?P<end>0|[1-9][0-9]*)/"
    r"(?P<total>[1-9][0-9]*)$"
)
_REQUEST_RANGE_PATTERN = re.compile(
    r"^bytes=(?P<start>0|[1-9][0-9]*)-(?P<end>0|[1-9][0-9]*)?$"
)


class _ContentRangeError(ValueError):
    """Raised when a partial response violates its byte-range schema."""


if TYPE_CHECKING:
    from crawler.fetching.network.body.partial_store import (
        PartialPayloadStorage,
    )
    from crawler.fetching.network.body.reader import AiohttpResponseBodyReader
    from crawler.fetching.request.body_plan import BodyReadPlan
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.fetching.results.materializer import ResponseBodyReadResult
    from logger.project_logger import ProjectLogger


class FetchResponseBodyReader:
    """Read a response body and map transport failures to fetch errors."""

    def __init__(
        self,
        *,
        response_body_reader: AiohttpResponseBodyReader,
        partial_payload_storage: PartialPayloadStorage,
        logger: ProjectLogger,
    ) -> None:
        self._response_body_reader = response_body_reader
        self._partial_payload_storage = partial_payload_storage
        self._logger = logger

    async def read(
        self,
        *,
        context: FetchRequestContext,
        response: ClientResponse,
        final_url: str,
        read_plan: BodyReadPlan,
    ) -> ResponseBodyReadResult:
        """Read, validate and persist one fetch response body."""

        requested_url = context.url
        try:
            return await self._read_validated_body(
                context=context,
                response=response,
                final_url=final_url,
                read_plan=read_plan,
            )
        except ResponseBodyReadCancelled as exc:
            self._logger.warning(
                "fetch_response_body_read_cancelled",
                url=requested_url,
                final_url=final_url,
                requested_kind=context.requested_kind,
                acceptance_mode=context.acceptance_mode,
                observed_bytes=exc.observed_bytes,
                chunk_count=exc.chunk_count,
                partial_path=(
                    str(exc.partial_path)
                    if exc.partial_path is not None
                    else None
                ),
            )
            raise
        except asyncio.CancelledError:
            self._logger.warning(
                "fetch_response_body_read_cancelled",
                url=requested_url,
                final_url=final_url,
                requested_kind=context.requested_kind,
                acceptance_mode=context.acceptance_mode,
            )
            raise
        except ResponseBodyTimeout as exc:
            requested_kind = context.requested_kind
            is_media = requested_kind in {MediaKind.AUDIO, MediaKind.VIDEO}
            self._logger.warning(
                "fetch_response_body_timeout",
                url=requested_url,
                final_url=final_url,
                requested_kind=requested_kind,
                acceptance_mode=context.acceptance_mode,
                observed_bytes=exc.observed_bytes,
                remaining_bytes=exc.remaining_bytes,
                chunk_count=exc.chunk_count,
                partial_path=(
                    str(exc.partial_path)
                    if exc.partial_path is not None
                    else None
                ),
                timeout_stage=exc.timeout_stage,
                body_read_mode=read_plan.mode,
                media_timeout_retryable=is_media,
            )
            raise response_body_timeout_error(exc, is_media=is_media) from exc
        except TimeoutError as exc:
            raise response_body_timeout_exception(exc) from exc
        except RuntimeError as exc:
            if not self._is_connection_closed_error(exc):
                raise
            self._logger.warning(
                "fetch_response_body_connection_closed_retry",
                url=requested_url,
                final_url=final_url,
                requested_kind=context.requested_kind,
                acceptance_mode=context.acceptance_mode,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise connection_closed_error() from exc
        except ResponseBodyLimitExceeded as exc:
            self._logger.warning(
                "fetch_skipped",
                url=requested_url,
                final_url=final_url,
                reason="response_body_limit_exceeded",
                observed_bytes=exc.observed_bytes,
                max_bytes=exc.max_bytes,
                requested_kind=context.requested_kind,
                acceptance_mode=context.acceptance_mode,
                body_read_mode=read_plan.mode,
            )
            raise IgnoredFetchError(
                reason="response_body_limit_exceeded",
                observed_bytes=exc.observed_bytes,
            ) from exc

    async def _read_validated_body(
        self,
        *,
        context: FetchRequestContext,
        response: ClientResponse,
        final_url: str,
        read_plan: BodyReadPlan,
    ) -> ResponseBodyReadResult:
        self._validate_range_response(
            response=response,
            read_plan=read_plan,
        )
        status_code = int(getattr(response, "status", 0) or 0)
        if (
            read_plan.expects_range_response
            and status_code != 206
            and self._content_length_exceeds_probe(
                response=response,
                max_bytes=read_plan.max_bytes,
            )
        ):
            raise ignored_range_probe_error()

        read_result = await self._response_body_reader.read(
            response,
            max_bytes=read_plan.max_bytes,
            allow_partial=read_plan.allow_partial,
            fetch_mode=read_plan.mode,
            resume_partial_path=read_plan.resume_partial_path,
            resume_owner_token=read_plan.resume_owner_token,
        )
        if (
            read_plan.expects_range_response
            and status_code != 206
            and read_result.truncated
        ):
            self._delete_temp_payload(
                path=read_result.payload.temp_path,
                reason="truncated_range_probe",
                url=context.url,
                final_url=final_url,
            )
            raise truncated_range_probe_error()

        if read_result.byte_size <= 0:
            self._delete_temp_payload(
                path=read_result.payload.temp_path,
                reason="empty_response_body",
                url=context.url,
                final_url=final_url,
            )
            raise IgnoredFetchError(
                reason="empty_response_body",
                observed_bytes=0,
            )
        return read_result

    def _validate_range_response(
        self,
        *,
        response: ClientResponse,
        read_plan: BodyReadPlan,
    ) -> None:
        status_code = int(getattr(response, "status", 0) or 0)
        if read_plan.resume_partial_path is not None and status_code != 206:
            self._discard_partial_resume(read_plan)
            raise RetryableFetchError(
                "server ignored resumed byte range",
                retry_class="fetch_retryable",
                retry_error_kind="resume_range_ignored",
                status_code=status_code,
            )
        if status_code != 206:
            return

        try:
            self._validate_resume_validator(
                response=response,
                read_plan=read_plan,
            )
            request_start = self._expected_range_start(read_plan.headers)
            if request_start is None:
                raise _ContentRangeError("unexpected 206 response")
            self._validate_content_range_header(
                headers=response.headers,
                expected_start=request_start,
                response_content_length=getattr(
                    response,
                    "content_length",
                    None,
                ),
            )
        except _ContentRangeError as exc:
            if read_plan.resume_partial_path is not None:
                self._discard_partial_resume(read_plan)
            raise RetryableFetchError(
                str(exc),
                retry_class="fetch_retryable",
                retry_error_kind="invalid_content_range",
                status_code=status_code,
            ) from exc

    @staticmethod
    def _expected_range_start(headers: dict[str, str]) -> int | None:
        raw = next(
            (
                value
                for key, value in headers.items()
                if key.strip().lower() == "range"
            ),
            None,
        )
        if raw is None:
            return None
        match = _REQUEST_RANGE_PATTERN.fullmatch(str(raw).strip())
        if match is None:
            raise _ContentRangeError(f"invalid Range request header: {raw!r}")
        start = int(match.group("start"))
        end_text = match.group("end")
        if end_text is not None and int(end_text) < start:
            raise _ContentRangeError("Range request end precedes start")
        return start

    @staticmethod
    def _validate_content_range_header(
        *,
        headers: object,
        expected_start: int,
        response_content_length: int | None,
    ) -> None:
        items = getattr(headers, "items", None)
        if items is None:
            raise _ContentRangeError("response headers are unavailable")
        raw = next(
            (
                value
                for key, value in items()
                if str(key).strip().lower() == "content-range"
            ),
            None,
        )
        if raw is None:
            raise _ContentRangeError("206 response lacks Content-Range")
        match = _CONTENT_RANGE_PATTERN.fullmatch(str(raw).strip())
        if match is None:
            raise _ContentRangeError(f"invalid Content-Range: {raw!r}")

        start = int(match.group("start"))
        end = int(match.group("end"))
        total = int(match.group("total"))
        if start != expected_start:
            raise _ContentRangeError(
                "Content-Range start does not match requested offset"
            )
        if end < start:
            raise _ContentRangeError("Content-Range end precedes start")
        if end >= total:
            raise _ContentRangeError("Content-Range end exceeds total size")
        if (
            response_content_length is not None
            and response_content_length != end - start + 1
        ):
            raise _ContentRangeError(
                "Content-Length does not match Content-Range segment length"
            )

    def _discard_partial_resume(self, read_plan: BodyReadPlan) -> None:
        if (
            read_plan.resume_partial_path is None
            or read_plan.resume_owner_token is None
        ):
            raise ValueError("resume cleanup requires owned partial state")
        self._partial_payload_storage.discard_partial(
            path=read_plan.resume_partial_path,
            owner_token=read_plan.resume_owner_token,
        )

    @staticmethod
    def _validate_resume_validator(
        *,
        response: ClientResponse,
        read_plan: BodyReadPlan,
    ) -> None:
        if read_plan.resume_partial_path is None:
            return
        if read_plan.resume_etag is not None:
            actual = str(response.headers.get("ETag", "")).strip()
            if actual != read_plan.resume_etag:
                raise _ContentRangeError("resumed response ETag changed")
            return
        if read_plan.resume_last_modified is not None:
            actual = str(response.headers.get("Last-Modified", "")).strip()
            if actual != read_plan.resume_last_modified:
                raise _ContentRangeError(
                    "resumed response Last-Modified changed"
                )
            return
        raise _ContentRangeError("resumed response has no bound validator")

    def _delete_temp_payload(
        self,
        *,
        path: Path,
        reason: str,
        url: str,
        final_url: str,
    ) -> None:
        try:
            self._partial_payload_storage.delete(path=path)
        except OSError as exc:
            self._logger.warning(
                "fetch_temp_payload_delete_failed",
                url=url,
                final_url=final_url,
                reason=reason,
                payload_path=str(path),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

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
    def _content_length_exceeds_probe(
        *,
        response: ClientResponse,
        max_bytes: int,
    ) -> bool:
        content_length = getattr(response, "content_length", None)
        if content_length is None:
            return False
        try:
            return int(content_length) > int(max_bytes)
        except (TypeError, ValueError):
            return False
