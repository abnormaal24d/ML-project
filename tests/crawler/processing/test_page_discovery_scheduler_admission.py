from __future__ import annotations

from typing import Any

import pytest

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.processing.page_discovery_reporting import PageDiscoveryReporting
from crawler.processing.page_discovery_scheduler_admission import (
    PageDiscoverySchedulerAdmission,
    SchedulerAdmissionContractError,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
)


class _FailingBudgetScheduler:
    def __init__(self) -> None:
        self.enqueue_calls = 0

    async def discovery_drain_budget(self, **_: object) -> int:
        raise RuntimeError("scheduler snapshot unavailable")

    async def enqueue_many(self, *_: object, **__: object) -> tuple[Any, ...]:
        self.enqueue_calls += 1
        return ()


class _RecordingLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, dict[str, object]]] = []
        self.errors: list[tuple[str, dict[str, object]]] = []

    def exception(self, event: str, **fields: object) -> None:
        self.exceptions.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))

    def info(self, _event: str, **_fields: object) -> None:
        return None


class _IdGenerator:
    def __init__(self) -> None:
        self._sequence = 0

    def generate(self) -> str:
        self._sequence += 1
        return f"task-{self._sequence}"


class _MismatchedDecisionScheduler:
    def __init__(self, *, returned_count: int) -> None:
        self.returned_count = returned_count
        self.enqueue_calls = 0

    async def discovery_drain_budget(self, **kwargs: object) -> int:
        return int(kwargs["configured_cap"])

    async def enqueue_many(
        self,
        tasks: tuple[CrawlTask, ...],
        **_: object,
    ) -> tuple[ScheduleDecision, ...]:
        self.enqueue_calls += 1
        return tuple(
            ScheduleDecision.reject(
                reason=ScheduleDecisionReason.URL_FILTERED,
                normalized_url=tasks[min(index, len(tasks) - 1)].url,
            )
            for index in range(self.returned_count)
        )


class _EchoDecisionScheduler:
    def __init__(self) -> None:
        self.submitted: tuple[CrawlTask, ...] = ()
        self.enqueue_kwargs: list[dict[str, object]] = []

    async def discovery_drain_budget(self, **kwargs: object) -> int:
        return int(kwargs["configured_cap"])

    async def enqueue_many(
        self,
        tasks: tuple[CrawlTask, ...],
        **kwargs: object,
    ) -> tuple[ScheduleDecision, ...]:
        self.submitted = tasks
        self.enqueue_kwargs.append(dict(kwargs))
        return tuple(
            ScheduleDecision.reject(
                reason=ScheduleDecisionReason.URL_FILTERED,
                normalized_url=task.url,
            )
            for task in tasks
        )


@pytest.mark.asyncio
async def test_budget_failure_fails_closed_and_records_distinct_reason() -> (
    None
):
    scheduler = _FailingBudgetScheduler()
    logger = _RecordingLogger()
    service = PageDiscoverySchedulerAdmission(
        scheduler=scheduler,
        logger=logger,  # type: ignore[arg-type]
        id_generator=_IdGenerator(),
    )
    tasks = (
        CrawlTask(
            url="https://example.test/image.jpg",
            source_name="test",
            kind=MediaKind.IMAGE,
        ),
        CrawlTask(
            url="https://example.test/audio.mp3",
            source_name="test",
            kind=MediaKind.AUDIO,
        ),
    )

    decisions, submitted = await service.admit_selected_tasks(
        parent_url="https://example.test/page",
        selected_tasks=tasks,
    )

    assert submitted == ()
    assert scheduler.enqueue_calls == 0
    assert [decision.accepted for decision in decisions] == [False, False]
    assert [decision.reason for decision in decisions] == [
        ScheduleDecisionReason.SCHEDULER_UNAVAILABLE,
        ScheduleDecisionReason.SCHEDULER_UNAVAILABLE,
    ]
    assert logger.exceptions == [
        (
            "discovery_drain_budget_failed",
            {
                "url": "https://example.test/page",
                "selected": 2,
                "error_type": "RuntimeError",
            },
        )
    ]

    reporting = PageDiscoveryReporting(
        logger=logger,  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )
    counts = reporting.count_admission_decisions(
        tasks=tasks,
        decisions=decisions,
    )

    assert counts.scheduler_rejected == 2
    assert counts.rejected_by_reason == {"scheduler_unavailable": 2}
    assert counts.metrics.as_payload()["truncated_by_reason"] == {
        "audio:scheduler:scheduler_unavailable": 1,
        "image:scheduler:scheduler_unavailable": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned_count", "expected_fragment"),
    (
        (1, "submitted=2, returned=1"),
        (3, "submitted=2, returned=3"),
    ),
)
async def test_scheduler_decision_count_mismatch_raises_contract_error(
    returned_count: int,
    expected_fragment: str,
) -> None:
    scheduler = _MismatchedDecisionScheduler(
        returned_count=returned_count,
    )
    logger = _RecordingLogger()
    service = PageDiscoverySchedulerAdmission(
        scheduler=scheduler,
        logger=logger,  # type: ignore[arg-type]
        id_generator=_IdGenerator(),
    )
    tasks = (
        CrawlTask(
            url="https://example.test/a.jpg",
            source_name="test",
            task_id="task-a",
            kind=MediaKind.IMAGE,
        ),
        CrawlTask(
            url="https://example.test/b.jpg",
            source_name="test",
            task_id="task-b",
            kind=MediaKind.IMAGE,
        ),
    )

    with pytest.raises(
        SchedulerAdmissionContractError,
        match=expected_fragment,
    ):
        await service.admit_selected_tasks(
            parent_url="https://example.test/page",
            selected_tasks=tasks,
        )

    assert scheduler.enqueue_calls == 1
    assert logger.errors == [
        (
            "discovery_scheduler_contract_violation",
            {
                "url": "https://example.test/page",
                "selected": 2,
                "submitted": 2,
                "returned_decisions": returned_count,
                "error_type": "SchedulerAdmissionContractError",
                "error": (
                    "scheduler returned an invalid decision count: "
                    f"submitted=2, returned={returned_count}"
                ),
            },
        )
    ]


@pytest.mark.asyncio
async def test_admission_assigns_stable_ids_before_scheduler_submission() -> (
    None
):
    scheduler = _EchoDecisionScheduler()
    logger = _RecordingLogger()
    service = PageDiscoverySchedulerAdmission(
        scheduler=scheduler,
        logger=logger,  # type: ignore[arg-type]
        id_generator=_IdGenerator(),
    )
    tasks = (
        CrawlTask(
            url="https://example.test/a.jpg",
            source_name="test",
            kind=MediaKind.IMAGE,
        ),
        CrawlTask(
            url="https://example.test/b.mp3",
            source_name="test",
            kind=MediaKind.AUDIO,
        ),
    )

    decisions, submitted = await service.admit_selected_tasks(
        parent_url="https://example.test/page",
        selected_tasks=tasks,
    )

    task_ids = [task.task_id for task in submitted]
    assert scheduler.submitted == submitted
    assert scheduler.enqueue_kwargs == [{}]
    assert task_ids == ["task-1", "task-2"]
    assert [decision.reason for decision in decisions] == [
        ScheduleDecisionReason.URL_FILTERED,
        ScheduleDecisionReason.URL_FILTERED,
    ]
    assert logger.errors == []


def test_alignment_uses_stable_task_id_not_object_identity() -> None:
    selected = (
        CrawlTask(
            url="https://example.test/a.jpg",
            source_name="test",
            task_id="task-a",
            kind=MediaKind.IMAGE,
        ),
        CrawlTask(
            url="https://example.test/b.jpg",
            source_name="test",
            task_id="task-b",
            kind=MediaKind.IMAGE,
        ),
    )
    submitted_clone = selected[0].clone(priority=10)
    scheduler_decision = ScheduleDecision.reject(
        reason=ScheduleDecisionReason.URL_FILTERED,
        normalized_url=submitted_clone.url,
    )

    aligned = PageDiscoverySchedulerAdmission.align_admission_decisions(
        selected_tasks=selected,
        submitted_tasks=(submitted_clone,),
        scheduler_decisions=(scheduler_decision,),
    )

    assert aligned[0] is scheduler_decision
    assert aligned[1].reason == ScheduleDecisionReason.SCHEDULER_BACKPRESSURE


@pytest.mark.asyncio
async def test_duplicate_task_ids_raise_before_scheduler_submission() -> None:
    scheduler = _EchoDecisionScheduler()
    logger = _RecordingLogger()
    service = PageDiscoverySchedulerAdmission(
        scheduler=scheduler,
        logger=logger,  # type: ignore[arg-type]
        id_generator=_IdGenerator(),
    )
    tasks = (
        CrawlTask(
            url="https://example.test/a.jpg",
            source_name="test",
            task_id="duplicate",
            kind=MediaKind.IMAGE,
        ),
        CrawlTask(
            url="https://example.test/b.jpg",
            source_name="test",
            task_id="duplicate",
            kind=MediaKind.IMAGE,
        ),
    )

    with pytest.raises(
        SchedulerAdmissionContractError,
        match="selected_tasks contains duplicate task ID 'duplicate'",
    ):
        await service.admit_selected_tasks(
            parent_url="https://example.test/page",
            selected_tasks=tasks,
        )

    assert scheduler.submitted == ()
    assert logger.errors[0][0] == "discovery_scheduler_contract_violation"
    assert logger.errors[0][1]["submitted"] == 0


def test_alignment_rejects_accepted_decision_for_different_task_id() -> None:
    submitted = CrawlTask(
        url="https://example.test/a.jpg",
        source_name="test",
        task_id="task-a",
        kind=MediaKind.IMAGE,
    )
    different_task = CrawlTask(
        url="https://example.test/b.jpg",
        source_name="test",
        task_id="task-b",
        kind=MediaKind.IMAGE,
    )
    mismatched_decision = ScheduleDecision.accept(
        normalized_url=different_task.url,
        task=different_task,
    )

    with pytest.raises(
        SchedulerAdmissionContractError,
        match=(
            "accepted scheduler decision task ID mismatch: "
            "submitted='task-a', returned='task-b'"
        ),
    ):
        PageDiscoverySchedulerAdmission.align_admission_decisions(
            selected_tasks=(submitted,),
            submitted_tasks=(submitted,),
            scheduler_decisions=(mismatched_decision,),
        )
