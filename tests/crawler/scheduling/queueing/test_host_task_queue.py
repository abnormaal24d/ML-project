"""Regression tests: lower numeric priority score pops first.

The scheduler convention is: lower numeric priority score means higher
scheduling priority. The heap must pop the minimum priority first, and
``sequence`` must preserve FIFO ordering for equal priority scores.
"""

from __future__ import annotations

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue


class _IdentityNormalizer:
    def normalize(self, host: object) -> object:
        return host


def _task(url: str) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="test",
        kind=MediaKind.PAGE,
        source_type="seed",
    )


def _queue(host_normalizer: object | None = None) -> HostTaskQueue:
    return HostTaskQueue(
        host_normalizer=host_normalizer or _IdentityNormalizer(),
    )


def _pop_urls(queue: HostTaskQueue, count: int) -> list[str]:
    return [queue.pop().url for _ in range(count)]


def test_lower_priority_score_pops_first() -> None:
    queue = _queue()
    queue.push(host="h", priority=10, sequence=0, task=_task("low"))
    queue.push(host="h", priority=-5, sequence=1, task=_task("high"))
    queue.push(host="h", priority=3, sequence=2, task=_task("medium"))

    assert _pop_urls(queue, 3) == ["high", "medium", "low"]


def test_equal_priority_keeps_fifo_sequence_order() -> None:
    queue = _queue()
    queue.push(host="h", priority=3, sequence=10, task=_task("a"))
    queue.push(host="h", priority=3, sequence=11, task=_task("b"))
    queue.push(host="h", priority=3, sequence=12, task=_task("c"))

    assert _pop_urls(queue, 3) == ["a", "b", "c"]


def test_pop_item_returns_priority_and_sequence_metadata() -> None:
    queue = _queue()
    task = _task("only")
    queue.push(host="h", priority=-7, sequence=42, task=task)

    host, priority, sequence, popped = queue.pop_item()

    assert host == "h"
    assert priority == -7
    assert sequence == 42
    assert popped is task


def test_snapshot_items_uses_true_pop_order() -> None:
    queue = _queue()
    queue.push(host="h", priority=10, sequence=0, task=_task("low"))
    queue.push(host="h", priority=-5, sequence=1, task=_task("high"))
    queue.push(host="h", priority=3, sequence=2, task=_task("medium"))

    snapshot = queue.snapshot_items()

    assert [item[3].url for item in snapshot] == ["high", "medium", "low"]


def test_snapshot_and_restore_preserves_priority_order() -> None:
    queue = _queue()
    queue.push(host="h", priority=10, sequence=0, task=_task("low"))
    queue.push(host="h", priority=-5, sequence=1, task=_task("high"))
    queue.push(host="h", priority=3, sequence=2, task=_task("medium"))

    restored = _queue()
    restored.restore_state(queue.snapshot_state())

    assert _pop_urls(restored, 3) == ["high", "medium", "low"]


def test_restore_orders_without_priority_migration() -> None:
    queue = _queue()
    queue.push(host="h", priority=-7, sequence=0, task=_task("seed"))
    queue.push(host="h", priority=16, sequence=1, task=_task("discovered"))

    restored = _queue()
    restored.restore_state(queue.snapshot_state())

    assert _pop_urls(restored, 2) == ["seed", "discovered"]


def test_real_host_normalizer_respects_priority_semantics() -> None:
    queue = _queue(host_normalizer=HostNormalizer())
    queue.push(host="example.com", priority=10, sequence=0, task=_task("low"))
    queue.push(host="example.com", priority=-5, sequence=1, task=_task("high"))

    assert _pop_urls(queue, 2) == ["high", "low"]
