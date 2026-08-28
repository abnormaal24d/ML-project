"""Concurrency contracts for transactional dataset writes."""

from __future__ import annotations

import asyncio
import threading

import pytest

from config.settings.datasets import RawDatasetWriterSettings
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter


def _build_writer() -> DatasetWriter:
    writer = DatasetWriter.__new__(DatasetWriter)
    writer._settings = RawDatasetWriterSettings(
        raw_persist_offload_to_thread=True,
    )
    writer._async_write_lock = asyncio.Lock()
    return writer


@pytest.mark.asyncio
async def test_offload_defers_cancellation_until_write_finishes() -> None:
    writer = _build_writer()
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def write_operation() -> str:
        started.set()
        release.wait(timeout=5)
        completed.set()
        return "written"

    task = asyncio.create_task(writer._offload(write_operation))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    assert not completed.is_set()

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed.is_set()


@pytest.mark.asyncio
async def test_offload_serializes_mutating_operations() -> None:
    writer = _build_writer()
    events: list[str] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        events.append("first-start")
        first_started.set()
        release_first.wait(timeout=5)
        events.append("first-end")

    def second() -> None:
        events.append("second-start")
        events.append("second-end")

    first_task = asyncio.create_task(writer._offload(first))
    await asyncio.to_thread(first_started.wait, 5)
    second_task = asyncio.create_task(writer._offload(second))
    await asyncio.sleep(0)

    assert events == ["first-start"]

    release_first.set()

    await first_task
    await second_task

    assert events == [
        "first-start",
        "first-end",
        "second-start",
        "second-end",
    ]


@pytest.mark.asyncio
async def test_offload_propagates_write_failure() -> None:
    writer = _build_writer()

    def failing_write() -> None:
        raise OSError("disk write failed")

    with pytest.raises(OSError, match="disk write failed"):
        await writer._offload(failing_write)
