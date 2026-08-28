"""Terminal-outcome regression tests for the crawl supervisor."""

from __future__ import annotations

from crawler.runtime.loop.crawl_run_summary import (
    CrawlStopTrigger,
    CrawlTerminalOutcome,
)
from crawler.runtime.loop.crawl_run_supervisor import (
    _classify_terminal_outcome,
)


def test_output_ready_is_a_successful_terminal_trigger() -> None:
    assert (
        _classify_terminal_outcome(
            stop_trigger=CrawlStopTrigger.OUTPUT_READY,
        )
        is CrawlTerminalOutcome.SUCCESS
    )
