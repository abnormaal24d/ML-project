"""Translate transport and body-read failures into fetch-domain errors."""

from __future__ import annotations

from crawler.fetching.errors.exceptions import (
    ResponseBodyTimeout,
    RetryableFetchError,
)


def _exception_message(
    exc: BaseException,
    *,
    fallback: str,
) -> str:
    message = str(exc).strip()
    return message if message else fallback


def transport_timeout_error(exc: BaseException) -> RetryableFetchError:
    """Wrap a transport timeout exception as a retryable fetch error."""

    message = _exception_message(
        exc,
        fallback=f"{type(exc).__name__} during fetch transport",
    )
    return RetryableFetchError(
        f"request timed out: {message}",
        retry_class="transport_timeout",
        retry_error_kind="transport_timeout",
    )


def transport_client_error(exc: BaseException) -> RetryableFetchError:
    """Wrap a transport client exception as a retryable fetch error."""

    return RetryableFetchError(
        _exception_message(
            exc,
            fallback=f"{type(exc).__name__} during fetch transport",
        ),
        retry_class="transport_error",
        retry_error_kind=type(exc).__name__,
    )


def truncated_range_probe_error() -> RetryableFetchError:
    """Build a retryable error for truncated metadata range probes."""

    return RetryableFetchError(
        "metadata range probe returned a truncated non-range body"
    )


def ignored_range_probe_error() -> RetryableFetchError:
    """Build a retryable error for ignored metadata range probes."""

    return RetryableFetchError(
        "metadata range probe was ignored by the server"
    )


def response_body_timeout_error(
    exc: ResponseBodyTimeout,
    *,
    is_media: bool,
) -> RetryableFetchError:
    """Wrap a ResponseBodyTimeout as a retryable fetch error."""

    prefix = (
        "media response body read timed out"
        if is_media
        else "response body read timed out"
    )
    return RetryableFetchError(
        (
            f"{prefix}: {exc}; "
            f"observed_bytes={exc.observed_bytes}; "
            f"remaining_bytes={exc.remaining_bytes}; "
            f"chunk_count={exc.chunk_count}; "
            f"timeout_stage={exc.timeout_stage}"
        ),
        retry_class="body_timeout",
        retry_error_kind=exc.timeout_stage,
        observed_bytes=exc.observed_bytes,
        partial_path=exc.partial_path,
    )


def response_body_timeout_exception(exc: BaseException) -> RetryableFetchError:
    """Wrap a generic body-read timeout as a retryable fetch error."""

    message = _exception_message(
        exc,
        fallback=f"{type(exc).__name__} while reading response body",
    )
    return RetryableFetchError(
        f"response body read timed out: {message}",
        retry_class="body_timeout",
        retry_error_kind="body_timeout",
    )


def connection_closed_error() -> RetryableFetchError:
    """Build a retryable error for prematurely closed response bodies."""

    return RetryableFetchError(
        "response body connection closed",
        retry_class="transport_error",
        retry_error_kind="connection_closed",
    )
