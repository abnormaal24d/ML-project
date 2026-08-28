"""Crawler runtime lock-ownership helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


class TransientLockRaceError(RuntimeError):
    """Raised before a condition notification without lock ownership."""


def condition_notify_all(condition: asyncio.Condition) -> None:
    """Notify waiters only while the condition lock is held.

    ``asyncio.Condition.notify_all`` raises a generic, implementation-specific
    ``RuntimeError`` when called without the lock.  Perform the stable public
    precondition check first so callers receive a project-owned typed error
    without relying on interpreter error-message text.
    """

    if not condition.locked():
        raise TransientLockRaceError(
            "condition notification requires the condition lock"
        )
    condition.notify_all()


@asynccontextmanager
async def owned_lock(lock: asyncio.Lock) -> AsyncIterator[None]:
    """Acquire and release one lock without reclassifying body exceptions."""

    async with lock:
        yield
