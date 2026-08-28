"""Pure processor failure policy and outcome translation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.runtime.concurrency import TransientLockRaceError

if TYPE_CHECKING:
    from collections.abc import Mapping


_TRANSIENT_LOCK_RACE_WAIT_SECONDS = 1.0


class RetryableOutcomeFields(TypedDict):
    """Retry diagnostics derived from a retryable fetch error."""

    observed_bytes: int | None
    partial_path: str | None
    unconditional_retry_performed: bool
    retry_budget_seconds_remaining: float | None


class ProcessorFailureHandler:
    """Translate processor and fetch failures into ``ProcessorOutcome`` values.

    The handler is intentionally side-effect free. Logging, error persistence,
    and storage-failure handling belong to the boundary that owns execution.
    """

    def __init__(
        self,
        *,
        default_retry_wait_seconds: float,
    ) -> None:
        self._default_retry_wait_seconds = float(default_retry_wait_seconds)

    @property
    def transient_lock_race_wait_seconds(self) -> float:
        return _TRANSIENT_LOCK_RACE_WAIT_SECONDS

    def ignored(
        self,
        *,
        exc: IgnoredFetchError,
        stage: str,
    ) -> ProcessorOutcome:
        """Translate an ignored fetch into a stable dropped outcome."""
        reason = self.ignored_outcome_reason(exc)
        metadata: dict[str, object] = {
            "ignored_fetch_reason": exc.reason,
        }

        if exc.status_code is not None:
            metadata["status_code"] = int(exc.status_code)

        if exc.final_url:
            metadata["final_url"] = exc.final_url

        return ProcessorOutcome.dropped(
            stage=stage,
            reason=reason,
            error_type=type(exc).__name__,
            error=str(exc),
            metadata=metadata,
        )

    def retryable(
        self,
        *,
        exc: RetryableFetchError,
        stage: str,
    ) -> ProcessorOutcome:
        """Translate a retryable fetch failure into scheduler semantics."""
        retry_fields = self.retryable_outcome_fields(exc)
        wait_seconds = self._resolved_retry_wait_seconds(
            exc.retry_after_seconds
        )
        return ProcessorOutcome.deferred(
            stage=stage,
            reason="retryable_fetch_error",
            retry_after_seconds=wait_seconds,
            retry_class=str(exc.retry_class).strip() or "fetch_retryable",
            retry_error_kind=(
                str(exc.retry_error_kind).strip()
                or str(exc.retry_class).strip()
                or "fetch_retryable"
            ),
            counts_toward_task_retry_budget=True,
            terminal_eligible=True,
            error_type=type(exc).__name__,
            error=str(exc),
            metadata={
                "observed_bytes": retry_fields["observed_bytes"],
                "partial_path": retry_fields["partial_path"],
                "unconditional_retry_performed": (
                    retry_fields["unconditional_retry_performed"]
                ),
                "retry_budget_seconds_remaining": (
                    retry_fields["retry_budget_seconds_remaining"]
                ),
            },
        )

    def transient_lock_race(
        self,
        *,
        exc: TransientLockRaceError,
        stage: str,
    ) -> ProcessorOutcome:
        """Translate a transient lock race into a bounded deferral."""
        return ProcessorOutcome.deferred(
            stage=stage,
            reason="transient_lock_race",
            retry_after_seconds=self.transient_lock_race_wait_seconds,
            retry_class="transient_lock_race",
            counts_toward_task_retry_budget=True,
            terminal_eligible=True,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def handler_exception(
        self,
        *,
        exc: Exception,
    ) -> ProcessorOutcome:
        """Translate an uncaught handler exception into a dropped outcome."""
        return ProcessorOutcome.dropped(
            stage="handler",
            reason="handler_exception",
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def processor_exception(
        self,
        *,
        stage: str,
        exc: Exception | None = None,
        reason: str = "processor_exception",
        error_type: str | None = None,
        error: str | None = None,
    ) -> ProcessorOutcome:
        """Translate a processor-stage failure into a dropped outcome."""
        resolved_error_type = error_type
        resolved_error = error
        if exc is not None:
            resolved_error_type = type(exc).__name__
            resolved_error = str(exc)

        return ProcessorOutcome.dropped(
            stage=stage,
            reason=reason,
            error_type=resolved_error_type or "",
            error=resolved_error or "",
        )

    def quality_rejected(
        self,
        *,
        reject_reason: str,
        quality_fields: Mapping[str, object],
    ) -> ProcessorOutcome:
        """Return a quality rejection carrying validated extension metadata."""
        return ProcessorOutcome.dropped(
            stage="quality",
            reason=reject_reason,
            detail=reject_reason,
            metadata=quality_fields,
        )

    @staticmethod
    def ignored_outcome_reason(exc: IgnoredFetchError) -> str:
        """Map an ignored fetch error to a stable processor reason."""
        message = str(exc.reason or exc).strip().lower()
        if not message:
            return "ignored_fetch"
        if "response_body_limit_exceeded" in message:
            return "response_body_limit_exceeded"
        if message == "not_modified":
            return "not_modified"
        if message == "server_denied_403":
            return "http_403_ignored"
        if message.startswith("non_success_status_"):
            status = message.removeprefix("non_success_status_")
            if status == "403":
                return "http_403_ignored"
            if status == "404":
                return "http_404_ignored"
            return "http_status_ignored"
        if "image_metadata_unreadable" in message:
            return "image_metadata_unreadable"
        return "ignored_fetch"

    @staticmethod
    def retryable_outcome_fields(
        exc: RetryableFetchError,
    ) -> RetryableOutcomeFields:
        """Build canonical retry diagnostics from a retryable fetch error."""
        return {
            "observed_bytes": exc.observed_bytes,
            "partial_path": (
                str(exc.partial_path) if exc.partial_path is not None else None
            ),
            "unconditional_retry_performed": (
                exc.unconditional_retry_performed
            ),
            "retry_budget_seconds_remaining": (
                exc.retry_budget_seconds_remaining
            ),
        }

    def _resolved_retry_wait_seconds(
        self,
        retry_after_seconds: object,
    ) -> float:
        """Return the canonical retry delay for every handler path."""
        if isinstance(retry_after_seconds, (int, float)) and not isinstance(
            retry_after_seconds,
            bool,
        ):
            return float(retry_after_seconds)
        return self._default_retry_wait_seconds
