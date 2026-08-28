"""Worker activity context helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from crawler.worker.worker_loop.worker_state import WorkerState

_CURRENT_WORKER_ACTIVITY: ContextVar[WorkerActivityController | None] = (
    ContextVar(
        "current_worker_activity",
        default=None,
    )
)


@dataclass(slots=True)
class WorkerActivityController:
    """Coordinate processing and waiting phases for one worker task."""

    state: WorkerState
    clock: Callable[[], float]
    _wait_depth: int = 0

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Bind this controller to the current async execution context."""

        token = _CURRENT_WORKER_ACTIVITY.set(self)
        try:
            yield
        finally:
            _reset_worker_activity(token)

    @contextmanager
    def waiting(self) -> Iterator[None]:
        """Temporarily exclude waiting time from active-processing metrics."""

        self._enter_waiting()
        try:
            yield
        finally:
            self._leave_waiting()

    def _enter_waiting(self) -> None:
        if self._wait_depth == 0:
            paused_at = self.clock()
            self.state.pause_processing(paused_at=paused_at)
        self._wait_depth += 1

    def _leave_waiting(self) -> None:
        if self._wait_depth == 0:
            return

        self._wait_depth -= 1
        if self._wait_depth != 0:
            return

        resumed_at = self.clock()
        self.state.resume_processing(resumed_at=resumed_at)


@contextmanager
def bind_worker_activity(
    controller: WorkerActivityController,
) -> Iterator[None]:
    """Bind a worker-activity controller for the current async context."""

    with controller.bind():
        yield


@contextmanager
def waiting_phase() -> Iterator[None]:
    """Mark the current worker as waiting when a controller is bound."""

    controller = _CURRENT_WORKER_ACTIVITY.get()
    if controller is None:
        yield
        return

    with controller.waiting():
        yield


def _reset_worker_activity(
    token: Token[WorkerActivityController | None],
) -> None:
    try:
        _CURRENT_WORKER_ACTIVITY.reset(token)
    except ValueError:
        _CURRENT_WORKER_ACTIVITY.set(None)
