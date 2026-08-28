"""Record fetch attempt metrics and host-profile feedback."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    ResponseBodyReadCancelled,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.http_rules import HttpStatusRulesSettings
    from crawler.fetching.feedback.transport_recorder import (
        TransportFeedbackRecorder,
    )
    from crawler.fetching.request.context import (
        FetchRequestContext,
    )
    from crawler.governance.host_suppression import HostSuppressionStore


class FetchAttemptOutcomeRecorder:
    """Record fetch multimodal for metrics and host suppression."""

    def __init__(
        self,
        *,
        feedback_recorder: TransportFeedbackRecorder,
        host_suppression_store: HostSuppressionStore,
        status_rules: HttpStatusRulesSettings,
        logger: ProjectLogger,
    ) -> None:
        self._feedback_recorder = feedback_recorder
        self._host_suppression_store = host_suppression_store
        self._status_rules = status_rules
        self._logger = logger

    async def record_ignored_fetch(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
        started_at: float,
        exc: IgnoredFetchError,
        final_host: str | None = None,
    ) -> None:
        """Record metrics for a fetch skipped with IgnoredFetchError."""

        effective_host = final_host or context.host

        if status_code >= 400:
            await self.record_non_success_feedback(
                context=context,
                status_code=status_code,
                started_at=started_at,
                observed_bytes=exc.observed_bytes,
                final_host=final_host,
            )

        if not exc.metrics_recorded:
            try:
                await self._feedback_recorder.record_skipped(
                    host=effective_host,
                    reason=exc.reason,
                    latency_seconds=asyncio.get_running_loop().time()
                    - started_at,
                    bytes_downloaded=exc.observed_bytes,
                )
                exc.metrics_recorded = True
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as record_exc:
                self._logger.warning(
                    "fetch_skipped_feedback_failed",
                    url=context.url,
                    host=effective_host,
                    reason=exc.reason,
                    error_type=type(record_exc).__name__,
                    error=str(record_exc),
                )

    async def record_cancelled_fetch(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
        final_url: str,
        started_at: float,
        bytes_downloaded: int,
        exc: asyncio.CancelledError,
    ) -> None:
        """Log and record metrics for a cancelled fetch attempt."""

        observed_bytes = bytes_downloaded
        partial_path = None

        if isinstance(exc, ResponseBodyReadCancelled):
            observed_bytes = exc.observed_bytes
            partial_path = (
                str(exc.partial_path) if exc.partial_path is not None else None
            )

        self._logger.warning(
            "fetch_cancelled",
            url=context.url,
            final_url=final_url,
            host=context.host,
            status_code=status_code if status_code > 0 else None,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            bytes_downloaded=observed_bytes,
            partial_path=partial_path,
            latency_seconds=round(
                asyncio.get_running_loop().time() - started_at,
                4,
            ),
        )

        try:
            await asyncio.shield(
                self._feedback_recorder.record_skipped(
                    host=context.host,
                    reason="cancelled",
                    latency_seconds=asyncio.get_running_loop().time()
                    - started_at,
                    bytes_downloaded=observed_bytes,
                ),
            )
        except (OSError, RuntimeError):
            self._logger.debug(
                "fetch_cancelled_metrics_record_failed",
                url=context.url,
                host=context.host,
            )

    async def record_attempt_feedback(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
        final_url: str,
        started_at: float,
        bytes_downloaded: int,
        quality_score: float | None,
        final_host: str | None = None,
    ) -> None:
        """Record latency, bytes, and quality for a completed attempt."""

        effective_host = final_host or context.host
        try:
            await self._feedback_recorder.record(
                host=effective_host,
                status_code=status_code,
                latency_seconds=asyncio.get_running_loop().time() - started_at,
                bytes_downloaded=bytes_downloaded,
                quality_score=quality_score,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._logger.warning(
                "fetch_feedback_record_failed",
                url=context.url,
                final_url=final_url,
                host=context.host,
                status_code=status_code,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def record_non_success_feedback(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
        started_at: float,
        observed_bytes: int,
        final_host: str | None = None,
    ) -> None:
        """Record host-suppression and metrics feedback for non-success HTTP statuses."""

        self._record_suppression_feedback_for_non_success(
            context=context,
            status_code=status_code,
        )
        await self._record_non_success_metrics(
            context=context,
            status_code=status_code,
            started_at=started_at,
            observed_bytes=observed_bytes,
            final_host=final_host,
        )
        self._logger.debug(
            "fetch_non_success_status_recorded",
            url=context.url,
            host=context.host,
            status_code=status_code,
            requested_kind=context.requested_kind,
            acceptance_mode=context.acceptance_mode,
            skip_reason=f"non_success_status_{status_code}",
        )

    def _record_suppression_feedback_for_non_success(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
    ) -> None:
        if status_code == 403 or status_code in self._status_rules.retryable:
            self._host_suppression_store.record_response_status(
                host=context.host,
                status_code=status_code,
            )

    async def _record_non_success_metrics(
        self,
        *,
        context: FetchRequestContext,
        status_code: int,
        started_at: float,
        observed_bytes: int,
        final_host: str | None = None,
    ) -> None:
        effective_host = final_host or context.host
        try:
            await self._feedback_recorder.record(
                host=effective_host,
                status_code=status_code,
                latency_seconds=asyncio.get_running_loop().time() - started_at,
                bytes_downloaded=observed_bytes,
                quality_score=None,
                count_toward_metrics=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._logger.warning(
                "fetch_feedback_record_failed",
                url=context.url,
                host=effective_host,
                status_code=status_code,
                error_type=type(exc).__name__,
                error=str(exc),
            )
