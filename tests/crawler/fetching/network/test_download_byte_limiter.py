from __future__ import annotations

import asyncio

import pytest

from crawler.fetching.network.body.download_byte_limiter import (
    DownloadByteLimiter,
)


@pytest.mark.asyncio
async def test_byte_capacity_is_shared_and_released() -> None:
    limiter = DownloadByteLimiter(
        max_in_flight_bytes=8,
        bytes_per_second=1_000,
        monotonic_seconds=lambda: 0.0,
    )
    await limiter.acquire(8)
    blocked = asyncio.create_task(limiter.acquire(1))
    await asyncio.sleep(0)
    assert not blocked.done()

    await limiter.release(8)
    await asyncio.wait_for(blocked, timeout=1.0)
    await limiter.release(1)


@pytest.mark.asyncio
async def test_invalid_or_duplicate_reservation_fails() -> None:
    limiter = DownloadByteLimiter(
        max_in_flight_bytes=8,
        bytes_per_second=1_000,
        monotonic_seconds=lambda: 0.0,
    )
    with pytest.raises(ValueError):
        await limiter.acquire(9)
    with pytest.raises(RuntimeError, match="released twice"):
        await limiter.release(1)
