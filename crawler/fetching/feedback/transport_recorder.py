"""Fetch feedback recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.fetching import FetcherSettings
    from crawler.governance.rate_limit.rate_limiter import RateLimiter
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics


class TransportFeedbackRecorder:
    """Record adaptive rate-control feedback and fetch metrics."""

    def __init__(
        self,
        *,
        settings: FetcherSettings,
        rate_limiter: RateLimiter,
        metrics: CollectionMetrics | None,
        logger: ProjectLogger,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._metrics = metrics
        self._record_rate_limiter_feedback = (
            settings.record_rate_limiter_feedback
        )
        self._record_fetch_metrics = settings.record_fetch_metrics
        self._logger = logger

    async def record(
        self,
        *,
        host: str,
        status_code: int,
        latency_seconds: float,
        bytes_downloaded: int,
        quality_score: float | None,
        count_toward_rate_feedback: bool = True,
        count_toward_metrics: bool = True,
    ) -> None:
        """Record completed fetch outcome feedback."""
        rate_limiter_recorded = False
        if count_toward_rate_feedback:
            rate_limiter_recorded = await self._record_rate_limiter_result(
                host=host,
                status_code=status_code,
                latency_seconds=latency_seconds,
            )
        metrics_recorded = False
        if count_toward_metrics:
            metrics_recorded = self._record_fetch_metric(
                host=host,
                status_code=status_code,
                latency_seconds=latency_seconds,
                bytes_downloaded=bytes_downloaded,
                quality_score=quality_score,
            )

        self._logger.debug(
            "fetch_feedback_recorded",
            host=host,
            status_code=status_code,
            latency_seconds=round(latency_seconds, 3),
            bytes_downloaded=bytes_downloaded,
            quality_score=(
                None if quality_score is None else round(quality_score, 3)
            ),
            rate_limiter_recorded=rate_limiter_recorded,
            metrics_recorded=metrics_recorded,
            count_toward_rate_feedback=count_toward_rate_feedback,
            count_toward_metrics=count_toward_metrics,
        )

    async def record_skipped(
        self,
        *,
        host: str,
        reason: str,
        latency_seconds: float,
        bytes_downloaded: int,
    ) -> None:
        """Record an acceptance-controlled fetch skip."""
        metrics_recorded = self._record_skipped_metric(
            host=host,
            reason=reason,
            latency_seconds=latency_seconds,
            bytes_downloaded=bytes_downloaded,
        )

        self._logger.debug(
            "fetch_skip_feedback_recorded",
            host=host,
            reason=reason,
            latency_seconds=round(latency_seconds, 3),
            bytes_downloaded=bytes_downloaded,
            metrics_recorded=metrics_recorded,
        )

    async def _record_rate_limiter_result(
        self,
        *,
        host: str,
        status_code: int,
        latency_seconds: float,
    ) -> bool:
        if not self._record_rate_limiter_feedback:
            return False

        await self._rate_limiter.report_result(
            host=host,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )
        return True

    def _record_fetch_metric(
        self,
        *,
        host: str,
        status_code: int,
        latency_seconds: float,
        bytes_downloaded: int,
        quality_score: float | None,
    ) -> bool:
        if not self._record_fetch_metrics:
            return False

        metrics = self._metrics
        if metrics is None:
            return False

        if not metrics.enabled:
            return False

        metrics.record_fetch(
            host=host,
            status_code=status_code,
            latency_seconds=latency_seconds,
            bytes_downloaded=bytes_downloaded,
            quality_score=quality_score,
        )
        return True

    def _record_skipped_metric(
        self,
        *,
        host: str,
        reason: str,
        latency_seconds: float,
        bytes_downloaded: int,
    ) -> bool:
        if not self._record_fetch_metrics:
            return False

        metrics = self._metrics
        if metrics is None:
            return False

        if not metrics.enabled:
            return False

        metrics.record_fetch_skipped(
            host=host,
            reason=reason,
            latency_seconds=latency_seconds,
            bytes_downloaded=bytes_downloaded,
        )
        return True
