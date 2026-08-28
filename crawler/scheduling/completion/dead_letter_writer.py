"""Dead-letter persistence contract for terminal scheduler tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


type DeadLetterStatus = Literal[
    "retry_exhausted",
    "failed",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class DeadLetterRecord:
    """One terminal scheduler disposition to persist for later recovery."""

    task: CrawlTask
    status: DeadLetterStatus
    original_outcome: str
    detail: str
    fields: dict[str, object]


class DeadLetterWriter(Protocol):
    """Persist terminal scheduler dispositions outside scheduler locks."""

    async def append(self, record: DeadLetterRecord) -> None: ...


__all__ = [
    "DeadLetterRecord",
    "DeadLetterStatus",
    "DeadLetterWriter",
]
