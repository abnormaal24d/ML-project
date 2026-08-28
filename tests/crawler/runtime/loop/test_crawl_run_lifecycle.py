"""Terminal-outcome and result contracts for one crawl lifecycle."""

from __future__ import annotations

import pytest

from crawler.runtime.loop.crawl_run_summary import (
    CrawlStopTrigger,
    CrawlTerminalOutcome,
    build_run_result,
)
from crawler.runtime.loop.crawl_run_supervisor import (
    _classify_terminal_outcome,
)
from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot


def _snapshot() -> WorkerPoolSnapshot:
    return WorkerPoolSnapshot(
        size=0,
        effective_worker_count=0,
        retiring_worker_count=0,
        busy_worker_count=0,
        idle_worker_count=0,
        completed_task_count=7,
        failure_count=0,
        non_fatal_timeout_count=1,
        retry_exhausted_count=0,
        average_processing_seconds=0.4,
        longest_busy_seconds=0.0,
        active_tasks=(),
    )


def test_frontier_drained_is_success() -> None:
    outcome = _classify_terminal_outcome(
        stop_trigger=CrawlStopTrigger.FRONTIER_DRAINED,
    )

    assert outcome is CrawlTerminalOutcome.SUCCESS


@pytest.mark.parametrize(
    "stop_trigger",
    (
        CrawlStopTrigger.STOP_REQUESTED,
        CrawlStopTrigger.INTERRUPTED,
        CrawlStopTrigger.DELAYED_BACKLOG_DEFERRED,
        CrawlStopTrigger.NO_ACCEPTED_SEEDS,
    ),
)
def test_non_drain_stops_stay_incomplete(
    stop_trigger: CrawlStopTrigger,
) -> None:
    outcome = _classify_terminal_outcome(
        stop_trigger=stop_trigger,
    )

    assert outcome is CrawlTerminalOutcome.INCOMPLETE


def test_build_run_result_carries_output_ready() -> None:
    result = build_run_result(
        worker_snapshot=_snapshot(),
        stop_trigger=CrawlStopTrigger.FRONTIER_DRAINED,
        terminal_outcome=CrawlTerminalOutcome.SUCCESS,
        unmet_requirements=("audio<5", "video<5"),
        output_ready=False,
    )

    assert result.terminal_outcome is CrawlTerminalOutcome.SUCCESS
    assert result.output_ready is False
    assert result.unmet_requirements == ("audio<5", "video<5")


def test_build_run_result_defaults_output_ready_to_false() -> None:
    result = build_run_result(
        worker_snapshot=_snapshot(),
        stop_trigger=CrawlStopTrigger.FRONTIER_DRAINED,
        terminal_outcome=CrawlTerminalOutcome.SUCCESS,
    )

    assert result.output_ready is False
