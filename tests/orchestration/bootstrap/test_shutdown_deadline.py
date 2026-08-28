from __future__ import annotations

import asyncio
import signal

import pytest

from orchestration.bootstrap.shutdown import install_signal_handlers
from orchestration.errors import ShutdownError
from orchestration.resource_shutdown import ResourceShutdownManager


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_shutdown_manager_uses_one_deadline_for_all_close_handlers() -> None:
    async def scenario() -> tuple[float, ShutdownError]:
        manager = ResourceShutdownManager(
            resource_shutdown_timeout_seconds=0.03
        )
        blocked = asyncio.Event()

        async def never_finishes() -> None:
            try:
                await blocked.wait()
            except asyncio.CancelledError:
                return

        manager.add_step(name="blocked", close=never_finishes)

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        with pytest.raises(ShutdownError) as raised:
            await manager.aclose()
        return loop.time() - started_at, raised.value

    elapsed, error = asyncio.run(scenario())

    assert elapsed < 0.15
    assert [step.step_name for step in error.errors] == ["blocked"]
    assert isinstance(error.errors[0].original_error, TimeoutError)


def test_shutdown_manager_continues_after_a_failed_callback() -> None:
    async def scenario() -> ShutdownError:
        manager = ResourceShutdownManager(
            resource_shutdown_timeout_seconds=1.0
        )
        calls: list[str] = []

        async def close_after_failure() -> None:
            calls.append("after_failure")

        async def fail() -> None:
            calls.append("fail")
            raise RuntimeError("close failed")

        manager.add_step(name="after_failure", close=close_after_failure)
        manager.add_step(name="fail", close=fail)

        with pytest.raises(ShutdownError) as raised:
            await manager.aclose()

        assert calls == ["fail", "after_failure"]
        return raised.value

    error = asyncio.run(scenario())

    assert [step.step_name for step in error.errors] == ["fail"]
    assert isinstance(error.errors[0].original_error, RuntimeError)


def test_signal_callback_cancels_workflow_task_but_not_siblings() -> None:
    async def scenario() -> None:
        logger = _Logger()
        loop = asyncio.get_running_loop()
        sibling_cancelled = False

        async def sibling() -> None:
            nonlocal sibling_cancelled
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled = True
                raise

        workflow_task = asyncio.create_task(
            asyncio.Event().wait(),
            name="workflow",
        )
        sibling_task = asyncio.create_task(
            sibling(),
            name="sibling",
        )
        await asyncio.sleep(0)

        shutdown_requested = False

        def request_shutdown(sig: object) -> None:
            nonlocal shutdown_requested
            if shutdown_requested:
                return
            shutdown_requested = True
            workflow_task.cancel()

        install_signal_handlers(
            loop=loop,
            logger=logger,
            shutdown_callback_factory=lambda sig: (
                lambda: request_shutdown(sig)
            ),
        )

        # Simulate signal by invoking callback directly
        request_shutdown(signal.SIGTERM)
        await asyncio.sleep(0)

        assert workflow_task.cancelling()
        assert not sibling_task.cancelled()
        assert not sibling_cancelled

        sibling_task.cancel()
        try:
            await sibling_task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_signal_callback_is_idempotent() -> None:
    async def scenario() -> tuple[int, _Logger]:
        logger = _Logger()
        loop = asyncio.get_running_loop()
        cancel_count = 0

        async def workflow() -> None:
            nonlocal cancel_count
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_count += 1
                raise

        workflow_task = asyncio.create_task(
            workflow(),
            name="workflow",
        )
        await asyncio.sleep(0)

        shutdown_requested = False

        def request_shutdown(sig: object) -> None:
            nonlocal shutdown_requested
            if shutdown_requested:
                return
            shutdown_requested = True
            workflow_task.cancel()

        install_signal_handlers(
            loop=loop,
            logger=logger,
            shutdown_callback_factory=lambda sig: (
                lambda: request_shutdown(sig)
            ),
        )

        request_shutdown(signal.SIGTERM)
        request_shutdown(signal.SIGTERM)
        await asyncio.sleep(0)

        return cancel_count, logger

    cancel_count, _logger = asyncio.run(scenario())

    assert cancel_count == 1
