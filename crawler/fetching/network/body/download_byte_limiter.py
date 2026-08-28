"""Bound aggregate in-flight response bytes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class DownloadByteLimiter:
    """Coordinate a shared byte-capacity budget across async downloads."""

    def __init__(
        self,
        *,
        max_in_flight_bytes: int,
        bytes_per_second: int,
        monotonic_seconds: Callable[[], float],
    ) -> None:
        if isinstance(max_in_flight_bytes, bool) or max_in_flight_bytes <= 0:
            raise ValueError("max_in_flight_bytes must be positive")
        if isinstance(bytes_per_second, bool) or bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        self._capacity = int(max_in_flight_bytes)
        self._bytes_per_second = int(bytes_per_second)
        self._monotonic_seconds = monotonic_seconds
        self._reserved = 0
        self._condition = asyncio.Condition()

    async def acquire(self, byte_count: int) -> None:
        count = self._validate_count(byte_count)
        if count > self._capacity:
            raise ValueError("reservation exceeds max_in_flight_bytes")
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._reserved + count <= self._capacity
            )
            self._reserved += count

    async def release(self, byte_count: int) -> None:
        count = self._validate_count(byte_count)
        async with self._condition:
            if count > self._reserved:
                raise RuntimeError("download byte reservation released twice")
            self._reserved -= count
            self._condition.notify_all()

    @staticmethod
    def _validate_count(byte_count: int) -> int:
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise TypeError("byte_count must be an integer")
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        return byte_count


__all__ = ["DownloadByteLimiter"]
