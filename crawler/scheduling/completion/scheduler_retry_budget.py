"""Track per-task retry budgets for deferred and timeout multimodal."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.scheduling.scheduling_value_parser import (
    coerce_bool,
    coerce_lower_str,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import SchedulingSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask


@dataclass(frozen=True, slots=True)
class TaskRetryDecision:
    """Decision returned before requeueing a deferred/timeout task."""

    terminal: bool
    reason: str | None = None


@dataclass(slots=True)
class TaskRetryState:
    """In-memory retry state for one scheduled task identity."""

    http_request_attempts: int = 0
    task_processing_attempts: int = 0
    retry_cycles: int = 0
    retryable_deferrals: int = 0
    timeouts: int = 0
    transport_timeout_count: int = 0
    body_timeout_count: int = 0
    processing_timeout_count: int = 0
    fatal_parse_count: int = 0
    last_outcome: str | None = None
    last_reason: str | None = None
    last_retry_class: str | None = None
    last_error_type: str | None = None
    last_error: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the strict checkpoint payload for this retry state."""

        return {
            "http_request_attempts": self.http_request_attempts,
            "task_processing_attempts": self.task_processing_attempts,
            "retry_cycles": self.retry_cycles,
            "retryable_deferrals": self.retryable_deferrals,
            "timeouts": self.timeouts,
            "transport_timeout_count": self.transport_timeout_count,
            "body_timeout_count": self.body_timeout_count,
            "processing_timeout_count": self.processing_timeout_count,
            "fatal_parse_count": self.fatal_parse_count,
            "last_outcome": self.last_outcome,
            "last_reason": self.last_reason,
            "last_retry_class": self.last_retry_class,
            "last_error_type": self.last_error_type,
            "last_error": self.last_error,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> TaskRetryState:
        """Restore retry state from a strict checkpoint payload."""

        if payload.keys() - {field.name for field in fields(cls)}:
            raise ValueError(
                "retry budget checkpoint payload contains unknown fields"
            )

        return cls(
            http_request_attempts=_int_from_payload(
                payload.get("http_request_attempts"),
                field_name="http_request_attempts",
            ),
            task_processing_attempts=_int_from_payload(
                payload.get("task_processing_attempts"),
                field_name="task_processing_attempts",
            ),
            retry_cycles=_int_from_payload(
                payload.get("retry_cycles"),
                field_name="retry_cycles",
            ),
            retryable_deferrals=_int_from_payload(
                payload.get("retryable_deferrals"),
                field_name="retryable_deferrals",
            ),
            timeouts=_int_from_payload(
                payload.get("timeouts"),
                field_name="timeouts",
            ),
            transport_timeout_count=_int_from_payload(
                payload.get("transport_timeout_count"),
                field_name="transport_timeout_count",
            ),
            body_timeout_count=_int_from_payload(
                payload.get("body_timeout_count"),
                field_name="body_timeout_count",
            ),
            processing_timeout_count=_int_from_payload(
                payload.get("processing_timeout_count"),
                field_name="processing_timeout_count",
            ),
            fatal_parse_count=_int_from_payload(
                payload.get("fatal_parse_count"),
                field_name="fatal_parse_count",
            ),
            last_outcome=_optional_str(payload.get("last_outcome")),
            last_reason=_optional_str(payload.get("last_reason")),
            last_retry_class=_optional_str(payload.get("last_retry_class")),
            last_error_type=_optional_str(payload.get("last_error_type")),
            last_error=_optional_str(payload.get("last_error")),
        )


def _int_from_payload(value: object, *, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"retry budget checkpoint payload contains non-integer "
            f"{field_name}"
        )
    if value < 0:
        raise ValueError(
            f"retry budget checkpoint payload contains negative {field_name}"
        )
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "retry budget checkpoint payload contains non-string metadata"
        )
    return value


def is_body_timeout_retry(
    *,
    retry_class: str | None,
    retry_error_kind: str | None,
    error_type: str | None,
) -> bool:
    values = {retry_class, retry_error_kind, error_type}
    return bool(
        values
        & {
            "body_timeout",
            "responsebodytimeout",
            "first_byte_timeout",
            "read_chunk_timeout",
            "max_idle_seconds",
            "max_stream_seconds",
        }
    )


def is_transport_timeout_retry(
    *,
    retry_class: str | None,
    retry_error_kind: str | None,
    error_type: str | None,
) -> bool:
    values = {retry_class, retry_error_kind, error_type}
    return bool(
        values
        & {
            "transport_timeout",
            "fetch_timeout",
            "timeouterror",
            "async_timeout",
        }
    )


class SchedulerRetryBudget:
    """Evaluate whether deferred or timeout tasks may be retried."""

    _RETRY_BUDGET_REASONS = frozenset(
        {
            "retryable_fetch_error",
            "transient_lock_race",
            "processor_timeout",
            "fetch_timeout",
            "body_timeout",
            "transport_timeout",
            "handler_timeout",
        }
    )

    _FETCH_IMPLIED_REASONS = frozenset(
        {
            "retryable_fetch_error",
            "processor_timeout",
            "fetch_timeout",
            "body_timeout",
            "transport_timeout",
            "handler_timeout",
            "fatal_parse",
        }
    )

    _FETCH_IMPLIED_RETRY_CLASSES = frozenset(
        {
            "fetch_retryable",
            "processor_timeout",
            "fetch_timeout",
            "body_timeout",
            "transport_timeout",
            "handler_timeout",
            "fatal_parse",
        }
    )

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        logger: ProjectLogger,
        is_drained: Callable[[], bool],
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._is_drained = is_drained
        self._state_by_task_key: dict[str, TaskRetryState] = {}

    def evaluate(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> TaskRetryDecision:
        """Consume retry budget and return whether the task should terminate."""

        if not self._counts_toward_retry_budget(
            task=task,
            outcome=outcome,
            fields=fields,
        ):
            return TaskRetryDecision(terminal=False)

        key = self._state_key(task=task)
        state = self._state_by_task_key.setdefault(key, TaskRetryState())

        reason = coerce_lower_str(fields.get("reason") if fields else None)
        retry_class = coerce_lower_str(
            fields.get("retry_class") if fields else None
        )
        error_type = coerce_lower_str(
            fields.get("error_type") if fields else None
        )
        error = coerce_lower_str(fields.get("error") if fields else None)

        state.task_processing_attempts += 1
        state.last_outcome = outcome
        state.last_reason = reason
        state.last_retry_class = retry_class
        state.last_error_type = error_type
        state.last_error = error
        retry_error_kind = coerce_lower_str(
            fields.get("retry_error_kind") if fields else None
        )

        if outcome == "deferred":
            state.retryable_deferrals += 1
        elif outcome == "timeout":
            state.timeouts += 1
            state.processing_timeout_count += 1

        if outcome == "timeout" or self._is_fetch_implied_retry(
            reason=reason,
            retry_class=retry_class,
        ):
            state.http_request_attempts += 1

        if is_body_timeout_retry(
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            error_type=error_type,
        ):
            state.body_timeout_count += 1
        elif is_transport_timeout_retry(
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            error_type=error_type,
        ):
            state.transport_timeout_count += 1
        elif retry_class == "processor_timeout":
            state.processing_timeout_count += 1
        elif retry_class == "fatal_parse":
            state.fatal_parse_count += 1

        if self._should_terminal_on_drain(
            task=task,
            outcome=outcome,
            fields=fields,
        ):
            return TaskRetryDecision(
                terminal=True,
                reason="drain_mode_retry_task_abandoned",
            )

        max_attempts = self._task_limit(
            task=task,
            by_kind=self._settings.max_total_attempts_by_kind,
            feed_default=self._settings.feed_max_total_attempts,
            default=self._settings.max_total_attempts,
        )
        max_deferrals = self._task_limit(
            task=task,
            by_kind=self._settings.max_deferrals_by_kind,
            feed_default=self._settings.feed_max_deferrals,
            default=self._settings.max_deferrals,
        )
        max_timeouts = self._task_limit(
            task=task,
            by_kind=self._settings.max_timeouts_by_kind,
            feed_default=self._settings.feed_max_timeouts,
            default=self._settings.max_timeouts,
        )

        if state.http_request_attempts >= max_attempts >= 0:
            return TaskRetryDecision(
                terminal=True,
                reason="max_total_attempts_exceeded",
            )

        if state.retryable_deferrals >= max_deferrals >= 0:
            return TaskRetryDecision(
                terminal=True,
                reason="max_deferrals_exceeded",
            )

        if state.processing_timeout_count >= max_timeouts >= 0:
            return TaskRetryDecision(
                terminal=True,
                reason="max_timeouts_exceeded",
            )

        state.retry_cycles += 1

        self._logger.debug(
            "scheduler_task_retry_budget_consumed",
            task_id=task.task_id,
            url=task.url,
            kind=task.kind,
            outcome=outcome,
            reason=reason,
            retry_class=retry_class,
            retry_error_kind=retry_error_kind,
            http_request_attempts=state.http_request_attempts,
            task_processing_attempts=state.task_processing_attempts,
            retry_cycles=state.retry_cycles,
            retryable_deferrals=state.retryable_deferrals,
            timeouts=state.timeouts,
            transport_timeout_count=state.transport_timeout_count,
            body_timeout_count=state.body_timeout_count,
            processing_timeout_count=state.processing_timeout_count,
            fatal_parse_count=state.fatal_parse_count,
            max_attempts=max_attempts,
            max_deferrals=max_deferrals,
            max_timeouts=max_timeouts,
        )
        return TaskRetryDecision(terminal=False)

    def forget(self, *, task: CrawlTask) -> None:
        """Drop retry state after a task completes or is abandoned."""

        self._state_by_task_key.pop(self._state_key(task=task), None)

    def export_state(self) -> dict[str, object]:
        """Return checkpoint payload for all tracked retry state."""

        return {
            key: state.to_payload()
            for key, state in self._state_by_task_key.items()
        }

    def restore_state(self, payload: dict[str, object]) -> None:
        """Replace tracked retry state from a strict checkpoint payload."""

        self.apply_restore_state(self.parse_restore_state(payload))

    @staticmethod
    def parse_restore_state(
        payload: dict[str, object],
    ) -> dict[str, TaskRetryState]:
        """Validate the complete retry payload without mutating live state."""

        restored: dict[str, TaskRetryState] = {}
        for key, value in payload.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, dict)
            ):
                raise ValueError("invalid retry budget checkpoint payload")
            restored[key] = TaskRetryState.from_payload(value)

        return restored

    def apply_restore_state(
        self,
        state: dict[str, TaskRetryState],
    ) -> None:
        """Commit retry state that has already passed full validation."""

        self._state_by_task_key = dict(state)

    def timeout_retry_wait_seconds(self) -> float:
        """Return configured delay before requeueing timeout retries."""

        return self._settings.timeout_retry_wait_seconds

    def _counts_toward_retry_budget(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> bool:
        if outcome == "timeout":
            return True

        if outcome != "deferred":
            return False

        if coerce_bool(
            fields.get("counts_toward_task_retry_budget") if fields else None,
            default=False,
        ):
            return True

        reason = coerce_lower_str(fields.get("reason") if fields else None)
        retry_class = coerce_lower_str(
            fields.get("retry_class") if fields else None
        )

        if reason in self._RETRY_BUDGET_REASONS:
            return True

        if retry_class in {
            "fetch_retryable",
            "body_timeout",
            "transport_timeout",
            "fetch_timeout",
            "transient_lock_race",
            "processor_timeout",
        }:
            return True

        if reason == "host_not_ready":
            return False

        if task.kind is MediaKind.FEED and reason in {"retryable_fetch_error"}:
            return True

        return False

    def _is_fetch_implied_retry(
        self,
        *,
        reason: str | None,
        retry_class: str | None,
    ) -> bool:
        """Whether a deferred outcome implies an HTTP fetch actually occurred."""

        return reason in self._FETCH_IMPLIED_REASONS or (
            retry_class in self._FETCH_IMPLIED_RETRY_CLASSES
        )

    def _should_terminal_on_drain(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> bool:
        if not self._settings.dead_letter_on_drain:
            return False

        if task.kind is not MediaKind.FEED:
            return False

        if outcome not in {"deferred", "timeout"}:
            return False

        if not self._terminal_eligible(fields=fields, outcome=outcome):
            return False

        return self._is_drained()

    def _terminal_eligible(
        self,
        *,
        fields: dict[str, object] | None,
        outcome: str,
    ) -> bool:
        if outcome == "timeout":
            return True

        if coerce_bool(
            fields.get("terminal_eligible") if fields else None,
            default=False,
        ):
            return True

        reason = coerce_lower_str(fields.get("reason") if fields else None)
        retry_class = coerce_lower_str(
            fields.get("retry_class") if fields else None
        )

        return reason in self._RETRY_BUDGET_REASONS or retry_class in {
            "fetch_retryable",
            "body_timeout",
            "transport_timeout",
            "fetch_timeout",
            "transient_lock_race",
        }

    def _task_limit(
        self,
        *,
        task: CrawlTask,
        by_kind: Mapping[str, int],
        feed_default: int,
        default: int,
    ) -> int:
        if task.kind.value in by_kind:
            return max(0, int(by_kind[task.kind.value]))

        if task.kind is MediaKind.FEED:
            return feed_default

        return default

    def _state_key(self, *, task: CrawlTask) -> str:
        if task.task_id:
            return f"id:{task.task_id}"
        return f"url:{task.kind}:{task.url}"
