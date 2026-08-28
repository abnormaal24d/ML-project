"""Fetch-layer exception types."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from crawler.exceptions.crawler_error import (
    CrawlerError,
    CrawlerTimeoutError,
    RetryableCrawlerError,
)

if TYPE_CHECKING:
    from pathlib import Path


class FetchError(CrawlerError):
    """Base class for fetch-layer failures."""


class RulesDeniedFetchError(FetchError):
    """Non-retryable rules or server denial (e.g. 403)."""

    pass


class RetryableFetchError(FetchError, RetryableCrawlerError):
    """Fetch-layer failure that carries scheduler retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        retry_class: str = "fetch_retryable",
        retry_error_kind: str | None = None,
        status_code: int | None = None,
        observed_bytes: int | None = None,
        partial_path: Path | None = None,
        retry_after_seconds: float | None = None,
        unconditional_retry_performed: bool = False,
        retry_budget_seconds_remaining: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_class = retry_class
        self.retry_error_kind = retry_error_kind or retry_class
        self.status_code = status_code
        self.observed_bytes = observed_bytes
        self.partial_path = partial_path
        self.retry_after_seconds = retry_after_seconds
        self.unconditional_retry_performed = unconditional_retry_performed
        self.retry_budget_seconds_remaining = retry_budget_seconds_remaining


class FetchTimeoutError(RetryableFetchError, CrawlerTimeoutError):
    """Retryable timeout raised while fetching a response."""


class IgnoredFetchError(FetchError):
    """Raised when a fetch attempt should be skipped without retrying."""

    def __init__(
        self,
        *,
        reason: str,
        observed_bytes: int = 0,
        metrics_recorded: bool = False,
        status_code: int | None = None,
        final_url: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.observed_bytes = max(0, int(observed_bytes))
        self.metrics_recorded = bool(metrics_recorded)
        self.status_code = status_code
        self.final_url = final_url


class ResponseBodyLimitExceeded(FetchError, ValueError):
    """Raised when a response body exceeds the configured byte budget."""

    def __init__(
        self,
        *,
        url: str,
        max_bytes: int,
        observed_bytes: int,
        chunk_count: int,
    ) -> None:
        super().__init__("response body exceeded configured max bytes")
        self.url = url
        self.max_bytes = max_bytes
        self.observed_bytes = observed_bytes
        self.chunk_count = chunk_count


class ResponseBodyDecompressionLimitExceeded(ResponseBodyLimitExceeded):
    """Raised when decoded bytes exceed the compressed-byte ratio budget."""

    def __init__(
        self,
        *,
        url: str,
        observed_bytes: int,
        compressed_bytes: int,
        max_ratio: float,
        chunk_count: int,
    ) -> None:
        super().__init__(
            url=url,
            max_bytes=max(1, int(compressed_bytes * max_ratio)),
            observed_bytes=observed_bytes,
            chunk_count=chunk_count,
        )
        self.compressed_bytes = compressed_bytes
        self.max_ratio = max_ratio


class ResponseBodyReadCancelled(asyncio.CancelledError):
    """Cancellation with response-body progress fields for retry handling."""

    def __init__(
        self,
        *,
        url: str,
        observed_bytes: int,
        chunk_count: int,
        partial_path: Path | None,
    ) -> None:
        super().__init__("response body read cancelled")
        self.url = url
        self.observed_bytes = observed_bytes
        self.chunk_count = chunk_count
        self.partial_path = partial_path


class ResponseBodyTimeout(FetchTimeoutError):
    """Retryable timeout raised while reading a response body."""

    def __init__(
        self,
        *,
        url: str,
        observed_bytes: int,
        chunk_count: int,
        partial_path: Path | None,
        remaining_bytes: int | None = None,
        timeout_stage: str = "body_read",
    ) -> None:
        message = (
            f"response body read timed out during {timeout_stage}; "
            f"url={url}; "
            f"observed_bytes={max(0, int(observed_bytes))}; "
            f"remaining_bytes={remaining_bytes}; "
            f"chunk_count={max(0, int(chunk_count))}"
        )

        super().__init__(
            message,
            retry_class="body_timeout",
            retry_error_kind=timeout_stage,
            observed_bytes=max(0, int(observed_bytes)),
            partial_path=partial_path,
        )
        self.url = url
        self.observed_bytes = max(0, int(observed_bytes))
        self.chunk_count = max(0, int(chunk_count))
        self.partial_path = partial_path
        self.remaining_bytes = remaining_bytes
        self.timeout_stage = timeout_stage
