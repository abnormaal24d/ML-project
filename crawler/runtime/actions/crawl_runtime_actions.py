"""Crawler pause, stop, and periodic runtime actions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypedDict

from config.collection.training_input_gate import CrawlOutputGateSettings
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.worker.pool.worker_pool import WorkerPool
from crawler.worker.worker_scaler import WorkerScaler
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from crawler.runtime.state.crawl_state_writer import CrawlStateWriter


class CrawlOutputReadinessReport(TypedDict):
    """Typed metrics emitted by the crawler output-readiness gate."""

    ready: bool
    unmet_requirements: tuple[str, ...]
    object_records_total: int
    requests_total: int
    successful_requests_total: int
    quality_score: float
    modality_counts: dict[str, int]


class CrawlOutputSnapshot(Protocol):
    """Dataset counters required by crawl-output readiness evaluation."""

    @property
    def object_records_total(self) -> int: ...

    @property
    def modality_counts(self) -> tuple[tuple[str, int], ...]: ...


class CrawlRuntimeActions:
    """
    Coordinate crawler pause, stop decisions, and periodic runtime actions.
    """

    def __init__(
        self,
        *,
        shutdown_poll_interval_seconds: float,
        crawl_output_gate: CrawlOutputGateSettings,
        worker_scaler: WorkerScaler,
        worker_pool: WorkerPool,
        control_directory: CrawlerControlDirectory,
        logger: ProjectLogger,
        min_workers: int,
        now: float,
        progress_interval: float,
        checkpoint_interval: float | None,
        metrics_interval: float | None,
        emit_progress: Callable[[], None],
        emit_metrics: Callable[[], None],
        state_writer: CrawlStateWriter,
        dataset_snapshot: Callable[[], CrawlOutputSnapshot] | None,
        metrics: CollectionMetrics | None,
    ) -> None:
        self._shutdown_poll_interval_seconds = shutdown_poll_interval_seconds
        self._crawl_output_gate = crawl_output_gate
        self._worker_scaler = worker_scaler
        self._worker_pool = worker_pool
        self._control_directory = control_directory
        self._logger = logger
        self._min_workers = min_workers

        self._emit_progress = emit_progress
        self._emit_metrics = emit_metrics
        self._state_writer = state_writer

        self._dataset_snapshot = dataset_snapshot
        self._metrics = metrics

        self._pause_active = False

        self._progress_interval = progress_interval
        self._checkpoint_interval = checkpoint_interval
        self._metrics_interval = metrics_interval

        self._next_progress_at = now + progress_interval
        self._next_checkpoint_at = (
            None if checkpoint_interval is None else now + checkpoint_interval
        )
        self._next_metrics_at = (
            None if metrics_interval is None else now + metrics_interval
        )

    async def handle_pause_if_requested(self) -> bool:
        """Pause crawler workers while a runtime pause flag is active."""

        if not self._control_directory.should_pause():
            return False

        if not self._pause_active:
            self._pause_active = True
            await self._worker_scaler.stop()
            await self._worker_pool.scale_to(0)
            self._logger.warning("crawler_paused")

        await asyncio.sleep(self._shutdown_poll_interval_seconds)
        return True

    async def resume_if_needed(self) -> None:
        """Resume crawler workers after a previous pause."""

        if not self._pause_active:
            return

        self._pause_active = False

        await self._worker_pool.scale_to(self._min_workers)
        self._worker_scaler.start()

        self._logger.info("crawler_resumed")

    def should_stop(self) -> bool:
        """Return whether the crawler loop should stop."""

        return self._control_directory.consume_stop()

    def control_poll_interval_seconds(self) -> float:
        """Return the bounded interval for checking operator controls."""

        return self._shutdown_poll_interval_seconds

    def crawl_output_readiness_report(self) -> CrawlOutputReadinessReport:
        """Return crawler output readiness and unmet thresholds."""

        readiness = self._crawl_output_gate

        if not readiness.enabled:
            return self._ready_report(
                ready=True,
                unmet_requirements=(),
                object_records_total=0,
                requests_total=0,
                successful_requests_total=0,
                quality_score=0.0,
                modality_counts={},
            )

        if self._dataset_snapshot is None:
            return self._unavailable_readiness_report()

        if self._metrics is None or not self._metrics.enabled:
            return self._unavailable_readiness_report()

        dataset_snapshot = self._dataset_snapshot()
        metrics_snapshot = self._metrics.snapshot(host_limit=0)
        modality_counts = dict(dataset_snapshot.modality_counts)

        unmet = self._collect_unmet_readiness_requirements(
            object_records_total=dataset_snapshot.object_records_total,
            successful_requests_total=metrics_snapshot.successes_total,
            quality_score=metrics_snapshot.quality_score,
            modality_counts=modality_counts,
        )

        return self._ready_report(
            ready=not unmet,
            unmet_requirements=tuple(unmet),
            object_records_total=dataset_snapshot.object_records_total,
            requests_total=metrics_snapshot.requests_total,
            successful_requests_total=metrics_snapshot.successes_total,
            quality_score=round(metrics_snapshot.quality_score, 4),
            modality_counts=modality_counts,
        )

    def wakeup_deadline(self) -> float:
        """Return the nearest scheduled runtime action timestamp."""

        wakeups = [self._next_progress_at]

        if self._next_checkpoint_at is not None:
            wakeups.append(self._next_checkpoint_at)

        if self._next_metrics_at is not None:
            wakeups.append(self._next_metrics_at)

        return min(wakeups)

    async def run_due_actions(self, *, current_time: float) -> None:
        """Run progress, checkpoint and metrics actions when due."""

        if current_time >= self._next_progress_at:
            self._emit_progress()
            self._next_progress_at = current_time + self._progress_interval

        checkpoint_interval = self._checkpoint_interval
        if (
            self._checkpoint_due(current_time)
            and checkpoint_interval is not None
        ):
            await self._state_writer.write_checkpoint(final=False)
            self._next_checkpoint_at = current_time + checkpoint_interval

        metrics_interval = self._metrics_interval
        if self._metrics_due(current_time) and metrics_interval is not None:
            self._emit_metrics()
            self._next_metrics_at = current_time + metrics_interval

    def _unavailable_readiness_report(self) -> CrawlOutputReadinessReport:
        return self._ready_report(
            ready=False,
            unmet_requirements=(
                "crawl_output_readiness_dependencies_unavailable",
            ),
            object_records_total=0,
            requests_total=0,
            successful_requests_total=0,
            quality_score=0.0,
            modality_counts={},
        )

    def _collect_unmet_readiness_requirements(
        self,
        *,
        object_records_total: int,
        successful_requests_total: int,
        quality_score: float,
        modality_counts: dict[str, int],
    ) -> list[str]:
        readiness = self._crawl_output_gate
        unmet: list[str] = []

        if object_records_total < readiness.min_raw_objects_total:
            unmet.append(
                f"object_records_total<{readiness.min_raw_objects_total}"
            )

        if successful_requests_total < readiness.min_successful_requests_total:
            unmet.append(
                "successful_requests_total<"
                f"{readiness.min_successful_requests_total}"
            )

        if quality_score < readiness.min_quality_score:
            unmet.append(f"quality_score<{readiness.min_quality_score:.2f}")

        required_modalities = {
            "page": readiness.minimum_records.page,
            "document": readiness.minimum_records.document,
            "image": readiness.minimum_records.image,
            "audio": readiness.minimum_records.audio,
            "video": readiness.minimum_records.video,
        }

        for modality, minimum in required_modalities.items():
            if modality_counts.get(modality, 0) < minimum:
                unmet.append(f"{modality}<{minimum}")

        return unmet

    def _checkpoint_due(self, current_time: float) -> bool:
        return (
            self._next_checkpoint_at is not None
            and self._checkpoint_interval is not None
            and current_time >= self._next_checkpoint_at
        )

    def _metrics_due(self, current_time: float) -> bool:
        return (
            self._next_metrics_at is not None
            and self._metrics_interval is not None
            and current_time >= self._next_metrics_at
        )

    @staticmethod
    def _ready_report(
        *,
        ready: bool,
        unmet_requirements: tuple[str, ...],
        object_records_total: int,
        requests_total: int,
        successful_requests_total: int,
        quality_score: float,
        modality_counts: dict[str, int],
    ) -> CrawlOutputReadinessReport:
        return {
            "ready": ready,
            "unmet_requirements": unmet_requirements,
            "object_records_total": object_records_total,
            "requests_total": requests_total,
            "successful_requests_total": successful_requests_total,
            "quality_score": quality_score,
            "modality_counts": modality_counts,
        }
