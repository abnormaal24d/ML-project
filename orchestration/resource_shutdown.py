"""Deterministic shutdown sequencing for runtime-owned resources."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

from orchestration.errors import ShutdownError, ShutdownStepError


def _coerce_resource_shutdown_timeout_seconds(
    timeout_seconds: float,
) -> float:
    """Return a finite, positive shutdown deadline budget."""

    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "resource shutdown timeout must be a finite positive number"
        ) from exc

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            "resource shutdown timeout must be a finite positive number"
        )
    return timeout


def _consume_abandoned_task_result(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned task result once it eventually finishes."""

    if task.cancelled():
        return

    try:
        task.result()
    except BaseException:
        # The task was deliberately detached after the resource shutdown
        # deadline. Retrieving its outcome prevents an unhandled-task warning.
        return


class ResourceShutdownManager:
    """Execute runtime shutdown steps in a deterministic order."""

    def __init__(
        self,
        *,
        resource_shutdown_timeout_seconds: float,
    ) -> None:
        self._steps: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        self._step_names: set[str] = set()
        self._closed = False
        self._resource_shutdown_timeout_seconds = (
            _coerce_resource_shutdown_timeout_seconds(
                resource_shutdown_timeout_seconds
            )
        )

    def add_step(
        self,
        *,
        name: str,
        close: Callable[[], Awaitable[None]],
    ) -> None:
        """Register a uniquely named shutdown step."""

        step_name = name.strip()
        if not step_name:
            raise ValueError("shutdown step name must not be empty")
        if self._closed:
            raise RuntimeError("shutdown manager is already closed")
        if step_name in self._step_names:
            raise ValueError(f"duplicate shutdown step: {step_name}")
        if not callable(close):
            raise TypeError(f"{step_name} close handler is not callable")

        self._steps.append((step_name, close))
        self._step_names.add(step_name)

    async def aclose(self) -> None:
        """Execute all registered shutdown steps within one deadline budget."""

        if self._closed:
            return
        self._closed = True

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._resource_shutdown_timeout_seconds
        errors: list[ShutdownStepError] = []
        steps = tuple(reversed(self._steps))

        for index, (step_name, close) in enumerate(steps):
            remaining = deadline - loop.time()
            if remaining <= 0:
                for skipped_step_name, _ in steps[index:]:
                    errors.append(
                        ShutdownStepError(
                            step_name=skipped_step_name,
                            cause=TimeoutError(
                                "resource shutdown deadline expired before "
                                "the close handler started"
                            ),
                        )
                    )
                break

            try:
                close_task = asyncio.ensure_future(close())
            except Exception as exc:
                errors.append(
                    ShutdownStepError(step_name=step_name, cause=exc)
                )
                continue

            _done, pending = await asyncio.wait(
                (close_task,),
                timeout=remaining,
            )
            if pending:
                close_task.cancel()
                close_task.add_done_callback(_consume_abandoned_task_result)
                errors.append(
                    ShutdownStepError(
                        step_name=step_name,
                        cause=TimeoutError(
                            "close handler exceeded the remaining resource "
                            "shutdown deadline"
                        ),
                    )
                )
                continue

            try:
                close_task.result()
            except (asyncio.CancelledError, Exception) as exc:
                errors.append(
                    ShutdownStepError(step_name=step_name, cause=exc)
                )

        self._steps.clear()
        self._step_names.clear()
        if errors:
            raise ShutdownError(errors=tuple(errors))
