"""Identity creation is explicit and cloning remains deterministic."""

from __future__ import annotations

import pytest

from crawler.crawl_tasks.crawl_task import CrawlTask, CrawlTaskBuildRequest


class _IdGenerator:
    def __init__(self, value: str = "generated-task-identifier") -> None:
        self._value = value

    def generate(self) -> str:
        return self._value


def test_clone_never_creates_identity() -> None:
    task = CrawlTask(
        url="https://example.test/source",
        source_name="example",
    )

    cloned = task.clone(url="https://example.test/clone")

    assert task.task_id is None
    assert cloned.task_id is None


def test_ensure_id_uses_required_generator() -> None:
    task = CrawlTask(
        url="https://example.test/source",
        source_name="example",
    )

    identified = task.ensure_id(id_generator=_IdGenerator())

    assert identified.task_id == "generated-task-i"
    assert task.task_id is None


def test_build_requires_explicit_identity_dependency() -> None:
    request = CrawlTaskBuildRequest(
        url="https://example.test/source",
        source_name="example",
    )

    with pytest.raises(TypeError, match="id_generator"):
        CrawlTask.build(request=request)  # type: ignore[call-arg]

    built = CrawlTask.build(request=request, id_generator=_IdGenerator())
    assert built.task_id == "generated-task-i"
