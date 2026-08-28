from __future__ import annotations

import asyncio
import inspect

from config.collection.discovery import SchedulingSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.completion.scheduler_retry_budget import (
    SchedulerRetryBudget,
    TaskRetryDecision,
    is_body_timeout_retry,
    is_transport_timeout_retry,
)


class _QuietLogger:
    def debug(self, message: object, **fields: object) -> None:
        del message, fields

    def warning(self, message: object, **fields: object) -> None:
        del message, fields


def _task(
    kind: MediaKind = MediaKind.PAGE,
    task_id: str = "task-1",
) -> CrawlTask:
    return CrawlTask(
        url="https://example.test/input",
        source_name="test",
        task_id=task_id,
        kind=kind,
    )


def _budget(
    **settings: object,
) -> SchedulerRetryBudget:
    return SchedulerRetryBudget(
        settings=SchedulingSettings(**settings),
        logger=_QuietLogger(),
        is_drained=lambda: False,
    )


# --- budget accounting -----------------------------------------------------


def test_non_retry_outcomes_never_consume_budget() -> None:
    budget = _budget()
    for outcome in (
        "success",
        "dropped",
        "failure",
        "cancelled",
        "interrupted",
    ):
        decision = budget.evaluate(task=_task(), outcome=outcome, fields=None)
        assert decision == TaskRetryDecision(terminal=False)
    assert budget.export_state() == {}


def test_deferred_without_budget_reason_never_consumes() -> None:
    budget = _budget()
    decision = budget.evaluate(
        task=_task(),
        outcome="deferred",
        fields={"reason": "host_not_ready", "retry_class": "host_pacing"},
    )
    assert decision == TaskRetryDecision(terminal=False)
    assert budget.export_state() == {}


def test_deferred_unknown_reason_without_flag_never_consumes() -> None:
    budget = _budget()
    decision = budget.evaluate(
        task=_task(),
        outcome="deferred",
        fields={"reason": "mystery"},
    )
    assert decision == TaskRetryDecision(terminal=False)
    assert budget.export_state() == {}


def test_deferred_reason_retryable_fetch_error_consumes() -> None:
    budget = _budget()
    decision = budget.evaluate(
        task=_task(),
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert decision == TaskRetryDecision(terminal=False)
    payload = budget.export_state()["id:task-1"]
    assert payload["http_request_attempts"] == 1
    assert payload["task_processing_attempts"] == 1
    assert payload["retryable_deferrals"] == 1
    assert payload["last_outcome"] == "deferred"
    assert payload["last_reason"] == "retryable_fetch_error"


def test_deferred_counts_flag_consumes_even_with_unknown_reason() -> None:
    budget = _budget()
    decision = budget.evaluate(
        task=_task(task_id="flagged"),
        outcome="deferred",
        fields={"counts_toward_task_retry_budget": True},
    )
    assert decision == TaskRetryDecision(terminal=False)
    payload = budget.export_state()["id:flagged"]
    assert payload["task_processing_attempts"] == 1
    assert payload["http_request_attempts"] == 0


def test_deferred_retry_class_consumes() -> None:
    budget = _budget(max_timeouts=10)
    fetch_implied = {
        "fetch_retryable",
        "body_timeout",
        "transport_timeout",
        "processor_timeout",
    }
    for retry_class in (
        *fetch_implied,
        "transient_lock_race",
    ):
        decision = budget.evaluate(
            task=_task(task_id=f"rc-{retry_class}"),
            outcome="deferred",
            fields={"retry_class": retry_class},
        )
        assert decision == TaskRetryDecision(terminal=False)
        payload = budget.export_state()[f"id:rc-{retry_class}"]
        assert payload["task_processing_attempts"] == 1
        assert payload["http_request_attempts"] == (
            1 if retry_class in fetch_implied else 0
        )


def test_timeout_outcome_always_consumes() -> None:
    budget = _budget(max_timeouts=10)
    decision = budget.evaluate(
        task=_task(task_id="to"),
        outcome="timeout",
        fields=None,
    )
    assert decision == TaskRetryDecision(terminal=False)
    payload = budget.export_state()["id:to"]
    assert payload["http_request_attempts"] == 1
    assert payload["timeouts"] == 1
    assert payload["processing_timeout_count"] == 1


def test_timeout_counter_never_counts_once_non_timeout() -> None:
    budget = _budget()
    budget.evaluate(
        task=_task(task_id="d"),
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    budget.evaluate(
        task=_task(task_id="d"),
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    payload = budget.export_state()["id:d"]
    assert payload["timeouts"] == 0
    assert payload["retryable_deferrals"] == 2


def test_outcome_and_retry_class_both_set_specific_counters() -> None:
    budget = _budget()
    budget.evaluate(
        task=_task(task_id="mix"),
        outcome="deferred",
        fields={"retry_class": "body_timeout"},
    )
    payload = budget.export_state()["id:mix"]
    assert payload["body_timeout_count"] == 1
    assert payload["processing_timeout_count"] == 0

    budget.evaluate(
        task=_task(task_id="mix"),
        outcome="deferred",
        fields={"retry_class": "processor_timeout"},
    )
    payload = budget.export_state()["id:mix"]
    assert payload["processing_timeout_count"] == 1

    budget.evaluate(
        task=_task(task_id="mix"),
        outcome="deferred",
        fields={
            "reason": "retryable_fetch_error",
            "retry_class": "fatal_parse",
        },
    )
    payload = budget.export_state()["id:mix"]
    assert payload["fatal_parse_count"] == 1


# --- retry budget exhaustion -----------------------------------------------


def test_total_attempt_exhaustion_is_terminal() -> None:
    budget = _budget(max_deferrals=10, max_timeouts=10, max_total_attempts=4)
    task = _task(task_id="ex")
    oldest_decision = None
    for _ in range(4):
        oldest_decision = budget.evaluate(
            task=task,
            outcome="deferred",
            fields={"reason": "retryable_fetch_error"},
        )
    assert oldest_decision == TaskRetryDecision(
        terminal=True,
        reason="max_total_attempts_exceeded",
    )


def test_deferral_exhaustion_is_terminal_before_attempt_wall() -> None:
    budget = _budget()
    task = _task(task_id="de")
    decisions = [
        budget.evaluate(
            task=task,
            outcome="deferred",
            fields={"reason": "retryable_fetch_error"},
        )
        for _ in range(3)
    ]
    assert decisions[-1] == TaskRetryDecision(
        terminal=True,
        reason="max_deferrals_exceeded",
    )
    assert decisions[0].terminal is False
    assert decisions[1].terminal is False


def test_timeout_exhaustion_is_terminal() -> None:
    budget = _budget(max_timeouts=1)
    decision = budget.evaluate(
        task=_task(task_id="td"),
        outcome="timeout",
        fields=None,
    )
    assert decision == TaskRetryDecision(
        terminal=True,
        reason="max_timeouts_exceeded",
    )


def test_per_kind_attempts_override() -> None:
    budget = _budget(
        max_total_attempts_by_kind={"feed": 2},
        max_deferrals=10,
        max_timeouts=10,
    )
    feed = _task(kind=MediaKind.FEED, task_id="fk")
    first = budget.evaluate(
        task=feed,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    second = budget.evaluate(
        task=feed,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert first.terminal is False
    assert second == TaskRetryDecision(
        terminal=True,
        reason="max_total_attempts_exceeded",
    )


# --- drain-mode dead lettering ---------------------------------------------


def test_drain_abandons_terminal_eligible_deferred_feed() -> None:
    budget = SchedulerRetryBudget(
        settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=10,
            max_timeouts=10,
        ),
        logger=_QuietLogger(),
        is_drained=lambda: True,
    )
    decision = budget.evaluate(
        task=_task(kind=MediaKind.FEED, task_id="fe"),
        outcome="deferred",
        fields={
            "counts_toward_task_retry_budget": True,
            "terminal_eligible": True,
        },
    )
    assert decision == TaskRetryDecision(
        terminal=True,
        reason="drain_mode_retry_task_abandoned",
    )


def test_drain_abandons_timeout_feed_outcome() -> None:
    budget = SchedulerRetryBudget(
        settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=10,
            max_timeouts=10,
        ),
        logger=_QuietLogger(),
        is_drained=lambda: True,
    )
    decision = budget.evaluate(
        task=_task(kind=MediaKind.FEED, task_id="ft"),
        outcome="timeout",
        fields=None,
    )
    assert decision == TaskRetryDecision(
        terminal=True,
        reason="drain_mode_retry_task_abandoned",
    )


def test_drain_requires_terminal_eligibility_for_deferrals() -> None:
    budget = SchedulerRetryBudget(
        settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=10,
            max_timeouts=10,
        ),
        logger=_QuietLogger(),
        is_drained=lambda: True,
    )
    decision = budget.evaluate(
        task=_task(kind=MediaKind.FEED, task_id="fe"),
        outcome="deferred",
        fields={"counts_toward_task_retry_budget": True},
    )
    assert decision == TaskRetryDecision(
        terminal=False,
        reason=None,
    )


# --- timeout budget accounting ---------------------------------------------


def test_configured_timeout_wait_seconds() -> None:
    budget = _budget(timeout_retry_wait_seconds=7.5)
    assert budget.timeout_retry_wait_seconds() == 7.5


def test_body_timeout_kind_detection() -> None:
    assert (
        is_body_timeout_retry(
            retry_class=None,
            retry_error_kind="read_chunk_timeout",
            error_type=None,
        )
        is True
    )
    assert (
        is_body_timeout_retry(
            retry_class="body_timeout",
            retry_error_kind=None,
            error_type=None,
        )
        is True
    )
    assert (
        is_body_timeout_retry(
            retry_class=None,
            retry_error_kind=None,
            error_type="transport_timeout",
        )
        is False
    )


def test_transport_timeout_kind_detection() -> None:
    assert (
        is_transport_timeout_retry(
            retry_class="fetch_timeout",
            retry_error_kind=None,
            error_type=None,
        )
        is True
    )
    assert (
        is_transport_timeout_retry(
            retry_class=None,
            retry_error_kind=None,
            error_type="timeouterror",
        )
        is True
    )
    assert (
        is_transport_timeout_retry(
            retry_class="body_timeout",
            retry_error_kind=None,
            error_type=None,
        )
        is False
    )


# --- state identity, forget, checkpoint round-trip ------------------------


def test_state_is_keyed_by_task_id_and_shared() -> None:
    budget = _budget(max_deferrals=10, max_total_attempts=10)
    task = _task(task_id="shared")
    budget.evaluate(
        task=task,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    budget.evaluate(
        task=_task(task_id="shared"),
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert budget.export_state()["id:shared"]["http_request_attempts"] == 2
    budget.evaluate(
        task=_task(task_id="other"),
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert budget.export_state()["id:other"]["http_request_attempts"] == 1


def test_forget_drops_state() -> None:
    budget = _budget()
    task = _task(task_id="gone")
    budget.evaluate(
        task=task,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert budget.export_state()["id:gone"]["http_request_attempts"] == 1
    budget.forget(task=task)
    assert budget.export_state() == {}


def test_export_restore_round_trip() -> None:
    budget = _budget()
    task = _task(task_id="cp")
    budget.evaluate(
        task=task,
        outcome="deferred",
        fields={
            "reason": "retryable_fetch_error",
            "error_type": "RetryableFetchError",
        },
    )
    payload = budget.export_state()
    restored = _budget()
    restored.restore_state(payload)
    assert restored.export_state() == payload


def test_restore_rejects_malformed_kinds() -> None:
    restored = _budget()
    try:
        restored.restore_state({"x": "string"})
    except ValueError as exc:
        assert "invalid retry budget checkpoint payload" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_restore_rejects_non_integer_counts() -> None:
    restored = _budget()
    try:
        restored.restore_state({"id:x": {"http_request_attempts": ["nope"]}})
    except ValueError as exc:
        assert "non-integer http_request_attempts" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_restore_rejects_unknown_checkpoint_fields() -> None:
    restored = _budget()
    try:
        restored.restore_state(
            {
                "id:unknown": {
                    "unknown_counter": 3,
                }
            }
        )
    except ValueError as exc:
        assert "unknown fields" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_retry_cycles_count_requeues_and_stop_at_terminal() -> None:
    budget = _budget(max_deferrals=10, max_total_attempts=3)
    task = _task(task_id="cycles")
    budget.evaluate(
        task=task,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    budget.evaluate(
        task=task,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    decision = budget.evaluate(
        task=task,
        outcome="deferred",
        fields={"reason": "retryable_fetch_error"},
    )
    assert decision == TaskRetryDecision(
        terminal=True,
        reason="max_total_attempts_exceeded",
    )
    payload = budget.export_state()["id:cycles"]
    assert payload["retry_cycles"] == 2
    assert payload["http_request_attempts"] == 3


# --- concurrency semantics ----------------------------------------------------


def test_evaluate_is_synchronous() -> None:
    method = SchedulerRetryBudget.evaluate
    assert inspect.iscoroutinefunction(method) is False


def test_concurrent_evaluations_are_serialized_at_await_boundary() -> None:
    budget = _budget(max_deferrals=100, max_total_attempts=100)
    task = _task(task_id="busy")

    async def call() -> None:
        budget.evaluate(
            task=task,
            outcome="deferred",
            fields={"reason": "retryable_fetch_error"},
        )

    async def spawn_all() -> None:
        await asyncio.gather(*(call() for _ in range(8)))

    asyncio.run(spawn_all())
    assert budget.export_state()["id:busy"]["http_request_attempts"] == 8


# --- forbidden endpoint suppression ------------------------------------------


def test_forbidden_endpoint_blocks_endpoint_but_not_other_paths() -> None:
    from crawler.scheduling.admission.admission_prerequisite_checker import (
        AdmissionPrerequisiteChecker,
    )
    from crawler.scheduling.admission.schedule_decision import (
        ScheduleDecisionReason,
    )
    from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry

    feedback = RunUrlFeedback(normalize_url=lambda url: url)
    checker = AdmissionPrerequisiteChecker(
        settings=SchedulingSettings(),
        url_filter=None,
        blacklist_repository=None,
        seen_urls=SeenUrlRegistry(max_seen=100),
        run_url_feedback=feedback,
    )
    task = _task().clone(
        url="https://repository.library.noaa.gov/gsearch?q=water&page=2"
    )

    first = checker.evaluate(
        task=task,
        closed=False,
    )
    assert first is None

    remembered = feedback.remember_forbidden_endpoint(
        url="https://repository.library.noaa.gov/gsearch?q=ocean&page=7"
    )
    assert remembered is True
    assert feedback.forbidden_endpoint_count == 1
    assert (
        feedback.is_forbidden_endpoint(
            url="https://repository.library.noaa.gov/gsearch?q=wind&page=3"
        )
        is True
    )
    assert (
        feedback.is_forbidden_endpoint(
            url="https://repository.library.noaa.gov/noaa/71203"
        )
        is False
    )

    second = checker.evaluate(
        task=task,
        closed=False,
    )
    assert second is not None
    assert second.accepted is False
    assert second.reason == ScheduleDecisionReason.FORBIDDEN_ENDPOINT_THIS_RUN

    allowed = checker.evaluate(
        task=task.clone(url="https://repository.library.noaa.gov/noaa/71203"),
        closed=False,
    )
    assert allowed is None
