"""Contracts for the Settings-free crawl phase runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from config.collection.processors import PageProcessorSettings
from crawler.runtime.loop.crawl_run_summary import CrawlTerminalOutcome
from datachecker.manifests.crawl_state_manifest_writer import (
    CrawlStateManifestWriter,
)
from datachecker.workflow_decision import (
    WorkflowAction,
    WorkflowDecisionReason,
    WorkflowExecutionPlan,
)
from orchestration.workflow.crawl.phase_runner import CrawlPhaseRunner
from orchestration.workflow.phase import PhaseStatus

_BASE_PAGE = PageProcessorSettings()
_FOCUSED_PAGE = PageProcessorSettings(max_non_page_media_per_page=48)


class _RecordingExecuteCrawl:
    def __init__(self, dataset_outcome: CrawlTerminalOutcome) -> None:
        self.calls: list[dict[str, object]] = []
        self._dataset_outcome = dataset_outcome

    async def __call__(
        self,
        *,
        crawl_attempt_id: str,
        crawl_state_manifest_writer: CrawlStateManifestWriter,
        page_settings: PageProcessorSettings,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "crawl_attempt_id": crawl_attempt_id,
                "crawl_state_manifest_writer": (crawl_state_manifest_writer),
                "page_settings": page_settings,
            }
        )
        return SimpleNamespace(dataset_outcome=self._dataset_outcome)


def _runner(
    tmp_path: Path,
    *,
    execute_crawl: _RecordingExecuteCrawl,
) -> CrawlPhaseRunner:
    async def run_blocking(func: object, /, *args: object, **kwargs: object):
        kwargs.pop("timeout_seconds", None)
        return func(**kwargs)

    return CrawlPhaseRunner(
        logger=SimpleNamespace(info=lambda *a, **k: None),
        run_blocking=run_blocking,
        io_timeout_seconds=1.0,
        source_registry_hash="source-hash",
        crawl_settings_hash="settings-hash",
        missing_by_media_kind=lambda coverage_gaps: dict(coverage_gaps),
        resolve_focus_kinds=lambda *, missing_by_kind: (
            ("document",)
            if any(missing > 0 for missing in missing_by_kind.values())
            else ()
        ),
        resolve_focused_page_settings=lambda *, focus_kinds: (
            _FOCUSED_PAGE if focus_kinds else _BASE_PAGE
        ),
        crawl_state_manifest_writer=cast(
            CrawlStateManifestWriter,
            SimpleNamespace(
                write_crawl_state_started=lambda **k: SimpleNamespace(
                    attempt_id="attempt-1"
                ),
                write_crawl_state_failed=lambda **k: None,
                write_crawl_state_cancelled=lambda **k: None,
                write_crawl_state_incomplete=lambda **k: None,
                read_current_state=lambda: SimpleNamespace(
                    attempt_id="attempt-1"
                ),
                resolve_latest_crawl_attempt=lambda: SimpleNamespace(
                    raw_run_directory=None, run_summary_path=None
                ),
            ),
        ),
        commit_crawl=lambda **k: SimpleNamespace(),
        execute_crawl=execute_crawl,
    )


def _plan(tmp_path: Path) -> WorkflowExecutionPlan:
    return WorkflowExecutionPlan(
        action=WorkflowAction.CRAWL,
        reason=WorkflowDecisionReason.COVERAGE_TARGETS_NOT_MET,
        coverage_gaps={"modality:document": 5},
    )


@pytest.mark.asyncio
async def test_focus_flows_into_execute_crawl_as_page_override(
    tmp_path: Path,
) -> None:
    execute_crawl = _RecordingExecuteCrawl(
        dataset_outcome=CrawlTerminalOutcome.SUCCESS
    )
    runner = _runner(tmp_path, execute_crawl=execute_crawl)

    outcome = await runner.run(_plan(tmp_path))

    assert outcome.status is PhaseStatus.SUCCEEDED
    assert len(execute_crawl.calls) == 1
    call = execute_crawl.calls[0]
    assert call["crawl_attempt_id"] == "attempt-1"
    assert call["page_settings"] is _FOCUSED_PAGE


@pytest.mark.asyncio
async def test_no_focus_passes_base_page_policy(tmp_path: Path) -> None:
    execute_crawl = _RecordingExecuteCrawl(
        dataset_outcome=CrawlTerminalOutcome.SUCCESS
    )
    runner = _runner(tmp_path, execute_crawl=execute_crawl)

    plan = WorkflowExecutionPlan(
        action=WorkflowAction.CRAWL,
        reason=WorkflowDecisionReason.COVERAGE_TARGETS_NOT_MET,
        coverage_gaps={},
    )
    await runner.run(plan)

    assert execute_crawl.calls[0]["page_settings"] is _BASE_PAGE


@pytest.mark.asyncio
async def test_failure_outcome_is_reported(tmp_path: Path) -> None:
    execute_crawl = _RecordingExecuteCrawl(
        dataset_outcome=CrawlTerminalOutcome.INCOMPLETE
    )
    runner = _runner(tmp_path, execute_crawl=execute_crawl)

    outcome = await runner.run(_plan(tmp_path))

    assert outcome.status is PhaseStatus.INCOMPLETE
