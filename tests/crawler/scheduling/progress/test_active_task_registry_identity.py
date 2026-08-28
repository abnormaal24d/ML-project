"""Active-task tracking must use the scheduler's canonical task identity."""

from __future__ import annotations

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.scheduling.progress.active_task_registry import ActiveTaskRegistry


def _task(*, kind: MediaKind, source_type: str) -> CrawlTask:
    return CrawlTask(
        url="https://example.test/shared",
        source_name="source",
        task_id=f"{kind.value}-{source_type}",
        kind=kind,
        source_type=source_type,
    )


def test_active_dispatching_and_dead_letter_records_keep_same_url_tasks_distinct() -> (
    None
):
    registry = ActiveTaskRegistry(host_normalizer=HostNormalizer())
    page = _task(kind=MediaKind.PAGE, source_type="seed")
    image = _task(kind=MediaKind.IMAGE, source_type="asset")
    video = _task(kind=MediaKind.VIDEO, source_type="asset")

    registry.add(host="example.test", priority=0, sequence=1, task=page)
    registry.add(host="example.test", priority=0, sequence=2, task=image)
    registry.mark_dispatching(
        host="example.test",
        priority=0,
        sequence=3,
        task=video,
    )

    assert registry.count == 2
    assert registry.dispatching_count == 1
    assert registry.remove(task=image).task is image  # type: ignore[union-attr]
    assert registry.count == 1
    assert registry.activate_dispatching(task=video).task is video  # type: ignore[union-attr]
    assert registry.count == 2

    page_record = registry.remove(task=page)
    assert page_record is not None
    registry.stage_dead_letter(record=page_record)
    assert registry.remove_dead_letter_pending(task=page) is page_record
    assert registry.remove(task=video).task is video  # type: ignore[union-attr]
