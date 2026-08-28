"""Crawl run result and summary utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot


class CrawlStopTrigger(str, Enum):
    """Why the crawl run stopped."""

    FRONTIER_DRAINED = "frontier_drained"
    OUTPUT_READY = "output_ready"
    STOP_REQUESTED = "stop_requested"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    DELAYED_BACKLOG_DEFERRED = "delayed_backlog_deferred"
    NO_ACCEPTED_SEEDS = "no_accepted_seeds"
    FAILED = "failed"


class CrawlTerminalOutcome(str, Enum):
    """Dataset-level outcome derived from crawl result and readiness gate."""

    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CrawlRunResult:
    """Summary returned by one completed crawler run.

    Public schema used by application bootstrap and callers.
    """

    stop_trigger: CrawlStopTrigger
    terminal_outcome: CrawlTerminalOutcome
    completed_tasks: int
    worker_failures: int
    task_failures_total: int
    non_fatal_timeouts_total: int
    retry_exhausted_total: int
    average_processing_seconds: float
    requests_total: int = 0
    successful_requests_total: int = 0
    object_records_total: int = 0
    eligible_records_total: int = 0
    root_seeds_total: int = 0
    root_seeds_succeeded: int = 0
    root_seeds_transient_failed: int = 0
    root_seeds_governance_blocked: int = 0
    required_dependency_failures: int = 0
    unmet_requirements: tuple[str, ...] = ()
    quality_score: float = 0.0
    modality_counts: dict[str, int] | None = None
    output_ready: bool = False


def build_run_result(
    *,
    worker_snapshot: WorkerPoolSnapshot,
    stop_trigger: CrawlStopTrigger,
    terminal_outcome: CrawlTerminalOutcome,
    requests_total: int = 0,
    successful_requests_total: int = 0,
    object_records_total: int = 0,
    eligible_records_total: int = 0,
    root_seeds_total: int = 0,
    root_seeds_succeeded: int = 0,
    root_seeds_transient_failed: int = 0,
    root_seeds_governance_blocked: int = 0,
    required_dependency_failures: int = 0,
    unmet_requirements: tuple[str, ...] = (),
    quality_score: float = 0.0,
    modality_counts: dict[str, int] | None = None,
    output_ready: bool = False,
) -> CrawlRunResult:
    """Build the public CrawlRunResult summary from pool counters."""

    return CrawlRunResult(
        stop_trigger=stop_trigger,
        terminal_outcome=terminal_outcome,
        completed_tasks=worker_snapshot.completed_task_count,
        worker_failures=worker_snapshot.failure_count,
        task_failures_total=worker_snapshot.failure_count,
        non_fatal_timeouts_total=worker_snapshot.non_fatal_timeout_count,
        retry_exhausted_total=worker_snapshot.retry_exhausted_count,
        average_processing_seconds=round(
            worker_snapshot.average_processing_seconds, 3
        ),
        requests_total=requests_total,
        successful_requests_total=successful_requests_total,
        object_records_total=object_records_total,
        eligible_records_total=eligible_records_total,
        root_seeds_total=root_seeds_total,
        root_seeds_succeeded=root_seeds_succeeded,
        root_seeds_transient_failed=root_seeds_transient_failed,
        root_seeds_governance_blocked=root_seeds_governance_blocked,
        required_dependency_failures=required_dependency_failures,
        unmet_requirements=unmet_requirements,
        quality_score=quality_score,
        modality_counts=modality_counts,
        output_ready=output_ready,
    )


def active_worker_task_summaries(
    worker_snapshot: WorkerPoolSnapshot,
) -> tuple[dict[str, object], ...]:
    """Return JSON-ready active task data from one pool snapshot."""

    return tuple(
        {
            "worker_id": task.worker_id,
            "task_id": task.task_id,
            "url": task.url,
            "kind": task.kind,
            "busy_seconds": task.busy_seconds,
            "retiring": task.retiring,
        }
        for task in worker_snapshot.active_tasks
    )
