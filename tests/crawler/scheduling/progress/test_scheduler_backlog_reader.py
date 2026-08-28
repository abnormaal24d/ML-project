"""Regression tests: pending measures only the ready queue.

``max_pending_per_host`` is a ready-queue cap (docs/architecture/
scheduler_retry_semantics.md). Delayed (pacing) tasks and in-flight tasks
must never consume pending frontier capacity.
"""

from __future__ import annotations

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.scheduling.progress.active_task_registry import ActiveTaskRegistry
from crawler.scheduling.progress.scheduler_backlog_reader import (
    SchedulerBacklogReader,
)
from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue


def _task(
    *,
    url: str,
    kind: MediaKind = MediaKind.PAGE,
    source_type: str = "discovered_link",
) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="test",
        kind=kind,
        source_type=source_type,
    )


def _build_reader(
    *,
    ready_tasks: list[CrawlTask] | None = None,
    delayed_tasks: list[CrawlTask] | None = None,
    inflight_count_by_host: dict[str, int] | None = None,
) -> SchedulerBacklogReader:
    host_normalizer = HostNormalizer()
    host_queue = HostTaskQueue(host_normalizer=host_normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=host_normalizer)
    active_registry = ActiveTaskRegistry(host_normalizer=host_normalizer)
    for index, task in enumerate(ready_tasks or ()):
        host_queue.push(
            host="example.com",
            priority=1,
            sequence=index,
            task=task,
        )
    for index, task in enumerate(delayed_tasks or ()):
        delayed_queue.push(
            host="example.com",
            priority=1,
            sequence=index,
            task=task,
            wait_seconds=60.0,
        )
    if inflight_count_by_host:
        active_registry.inflight_count_by_host.update(inflight_count_by_host)
    return SchedulerBacklogReader(
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        max_feeds_per_host=2,
    )


def test_host_pending_counts_only_ready_queue_not_delayed_or_inflight() -> (
    None
):
    reader = _build_reader(
        ready_tasks=[
            _task(url=f"https://example.com/page-{i}") for i in range(11)
        ],
        delayed_tasks=[
            _task(url=f"https://example.com/delayed-{i}") for i in range(50)
        ],
        inflight_count_by_host={"example.com": 3},
    )

    assert reader.host_pending(host="example.com") == 11


def test_host_pending_for_unknown_host_is_zero() -> None:
    reader = _build_reader(
        ready_tasks=[_task(url="https://example.com/page")],
    )

    assert reader.host_pending(host="other.example") == 0


def test_host_pending_for_none_has_no_host_bucket() -> None:
    host_normalizer = HostNormalizer()
    host_queue = HostTaskQueue(host_normalizer=host_normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=host_normalizer)
    for index in range(3):
        host_queue.push(
            host="a.example",
            priority=1,
            sequence=index,
            task=_task(url=f"https://a.example/{index}"),
        )
    host_queue.push(
        host="b.example",
        priority=1,
        sequence=3,
        task=_task(url="https://b.example/0"),
    )
    reader = SchedulerBacklogReader(
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        max_feeds_per_host=2,
    )

    assert reader.host_pending(host="a.example") == 3
    assert reader.host_pending(host="b.example") == 1
    assert reader.host_pending(host=None) == 0


def test_kind_host_pending_counts_only_ready_tasks_of_that_kind() -> None:
    reader = _build_reader(
        ready_tasks=(
            [_task(url=f"https://example.com/page-{i}") for i in range(5)]
            + [
                _task(url=f"https://example.com/img-{i}", kind=MediaKind.IMAGE)
                for i in range(2)
            ]
        ),
        delayed_tasks=[
            _task(url=f"https://example.com/d-{i}", kind=MediaKind.PAGE)
            for i in range(50)
        ],
    )

    assert (
        reader.kind_host_pending(host="example.com", kind=MediaKind.PAGE) == 5
    )
    assert (
        reader.kind_host_pending(host="example.com", kind=MediaKind.IMAGE) == 2
    )
    assert (
        reader.kind_host_pending(host="example.com", kind=MediaKind.VIDEO) == 0
    )


def test_kind_host_pending_if_needed_uses_ready_only_and_skips_seeds() -> None:
    reader = _build_reader(
        ready_tasks=[_task(url="https://example.com/page")],
    )
    seed_task = _task(
        url="https://example.com/seed",
        source_type="seed",
    )
    non_seed_task = _task(url="https://example.com/discovered")

    assert (
        reader.kind_host_pending_if_needed(
            task=non_seed_task,
            host="example.com",
        )
        == 1
    )
    assert (
        reader.kind_host_pending_if_needed(
            task=seed_task,
            host="example.com",
        )
        == 0
    )


def test_pending_maps_still_expose_delayed_and_combined_for_export() -> None:
    reader = _build_reader(
        ready_tasks=[_task(url="https://example.com/ready")],
        delayed_tasks=[_task(url="https://example.com/delayed")],
    )

    ready, delayed, combined = reader.pending_maps()

    assert ready == {"example.com": 1}
    assert delayed == {"example.com": 1}
    assert combined == {"example.com": 2}
