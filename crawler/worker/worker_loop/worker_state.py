"""Worker lifecycle state model."""

from __future__ import annotations


class WorkerState:
    """Mutable lifecycle and timing state for one worker."""

    __slots__ = (
        "worker_id",
        "busy",
        "processing",
        "retire_when_idle",
        "fatal_failure_recorded",
        "current_started_at",
        "current_processing_started_at",
        "accumulated_processing_seconds",
        "current_task_id",
        "current_url",
        "current_kind",
        "last_outcome",
        "last_duration_seconds",
        "last_processing_duration_seconds",
        "processed_count",
    )

    def __init__(self, worker_id: int) -> None:
        self.worker_id: int = worker_id
        self.busy: bool = False
        self.processing: bool = False
        self.retire_when_idle: bool = False
        self.fatal_failure_recorded: bool = False
        self.current_started_at: float | None = None
        self.current_processing_started_at: float | None = None
        self.accumulated_processing_seconds: float = 0.0
        self.current_task_id: str | None = None
        self.current_url: str | None = None
        self.current_kind: str | None = None
        self.last_outcome: str | None = None
        self.last_duration_seconds: float | None = None
        self.last_processing_duration_seconds: float | None = None
        self.processed_count: int = 0

    def start_task(
        self,
        *,
        started_at: float,
        task_id: str,
        url: str,
        kind: str,
    ) -> None:
        self.busy = True
        self.processing = True
        self.current_started_at = started_at
        self.current_processing_started_at = started_at
        self.accumulated_processing_seconds = 0.0
        self.current_task_id = task_id
        self.current_url = url
        self.current_kind = kind

    def finish_task(
        self,
        *,
        finished_at: float,
    ) -> tuple[float, float]:
        total_seconds = 0.0
        if self.current_started_at is not None:
            total_seconds = finished_at - self.current_started_at
            self.last_duration_seconds = total_seconds

        processing_seconds = self.accumulated_processing_seconds
        if self.processing and self.current_processing_started_at is not None:
            processing_seconds += (
                finished_at - self.current_processing_started_at
            )

        self.last_processing_duration_seconds = processing_seconds
        self.busy = False
        self.processing = False
        self.current_started_at = None
        self.current_processing_started_at = None
        self.accumulated_processing_seconds = 0.0
        self.current_task_id = None
        self.current_url = None
        self.current_kind = None

        return (
            max(0.0, total_seconds),
            max(0.0, processing_seconds),
        )

    def record_outcome(self, outcome: str) -> None:
        self.last_outcome = outcome
        self.processed_count += 1

    def pause_processing(self, *, paused_at: float) -> bool:
        if not self.busy or not self.processing:
            return False

        if self.current_processing_started_at is not None:
            self.accumulated_processing_seconds += max(
                0.0,
                paused_at - self.current_processing_started_at,
            )

        self.processing = False
        self.current_processing_started_at = None
        return True

    def resume_processing(self, *, resumed_at: float) -> bool:
        if not self.busy or self.processing:
            return False

        self.processing = True
        self.current_processing_started_at = resumed_at
        return True

    def active_processing_seconds(self, *, now: float) -> float:
        active_seconds = self.accumulated_processing_seconds
        if self.processing and self.current_processing_started_at is not None:
            active_seconds += max(
                0.0,
                now - self.current_processing_started_at,
            )
        return max(0.0, active_seconds)
